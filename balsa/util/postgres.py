# Copyright 2022 The Balsa Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Postgres connector: issues commands and parses results."""
import pprint
import re
import subprocess

import pandas as pd

from balsa.util import plans_lib
import pg_executor


def GetServerVersion():
    with pg_executor.Cursor() as cursor:
        cursor.execute('show server_version;')
        row = cursor.fetchone()
        return row[0]


def GetServerConfigs():
    """Returns all live configs as [(param, value, help)]."""
    with pg_executor.Cursor() as cursor:
        cursor.execute('show all;')
        return cursor.fetchall()


def GetServerConfigsAsDf():
    """Returns all live configs as [(param, value, help)]."""
    data = GetServerConfigs()
    return pd.DataFrame(data, columns=['param', 'value', 'help']).drop('help',
                                                                       axis=1)


def _SetGeneticOptimizer(flag, cursor):
    # NOTE: DISCARD would erase settings specified via SET commands.  Make sure
    # no DISCARD ALL is called unexpectedly.
    assert cursor is not None
    assert flag in ['on', 'off', 'default'], flag
    cursor.execute('set geqo = {};'.format(flag))
    assert cursor.statusmessage == 'SET'


def DropBufferCache():
    # WARNING: no effect if PG is running on another machine
    subprocess.check_output(['free', '&&', 'sync'])
    subprocess.check_output(
        ['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'])
    subprocess.check_output(['free'])

    with pg_executor.Cursor() as cursor:
        cursor.execute('DISCARD ALL;')


def ExplainAnalyzeSql(sql,
                      comment=None,
                      verbose=False,
                      geqo_off=False,
                      timeout_ms=None,
                      is_test=False,
                      remote=False):
    """Runs EXPLAIN ANALYZE.

    Returns:
      If remote:
        A pg_executor.Result.
      Else:
        A ray.ObjectRef of the above.
    """
    return _run_explain('explain (verbose, analyze, format json)',
                        sql,
                        comment,
                        verbose,
                        geqo_off,
                        timeout_ms,
                        is_test=is_test,
                        remote=remote)


def SqlToPlanNode(sql,
                  comment=None,
                  verbose=False,
                  keep_scans_joins_only=False,
                  cursor=None):
    """Issues EXPLAIN(format json) on a SQL string; parse into our AST node."""
    # Use of 'verbose' would alias-qualify all column names in pushed-down
    # filters, which are beneficial for us (e.g., this ensures that
    # Node.to_sql() returns a non-ambiguous SQL string).
    # Ref: https://www.postgresql.org/docs/11/sql-explain.html
    geqo_off = comment is not None and len(comment) > 0
    result = _run_explain('explain(verbose, format json)',
                          sql,
                          comment,
                          verbose,
                          geqo_off=geqo_off,
                          cursor=cursor).result
    json_dict = result[0][0][0]
    node = ParsePostgresPlanJson(json_dict)
    if not keep_scans_joins_only:
        return node, json_dict
    return plans_lib.FilterScansOrJoins(node), json_dict


def ExecuteSql(sql, hint=None, check_hint=False, verbose=False):
    geqo_off = hint is not None and len(hint) > 0
    result = _run_explain('explain (verbose, analyze, format json)',
                          sql,
                          hint,
                          verbose,
                          geqo_off=geqo_off).result
    json_dict = result[0][0][0]
    node = ParsePostgresPlanJson(json_dict)
    node = plans_lib.FilterScansOrJoins(node)
    return json_dict, node


def ContainsPhysicalHints(hint_str):
    HINTS = [
        'SeqScan',
        'IndexScan',
        'IndexOnlyScan',
        'NestLoop',
        'HashJoin',
        'MergeJoin',
    ]
    for hint in HINTS:
        if hint in hint_str:
            return True
    return False


def GetCostFromPg(sql, hint, verbose=False, check_hint_used=False):
    with pg_executor.Cursor() as cursor:
        # GEQO must be disabled for hinting larger joins to work.
        _SetGeneticOptimizer('off', cursor)
        node0 = SqlToPlanNode(sql, comment=hint, verbose=verbose,
                              cursor=cursor)[0]
        # This copies top-level node's cost (e.g., Aggregate) to the new top level
        # node (a Join).
        node = plans_lib.FilterScansOrJoins(node0)
        _SetGeneticOptimizer('default', cursor)  # Restores.

    if check_hint_used:
        expected = hint
        actual = node.hint_str(with_physical_hints=ContainsPhysicalHints(hint))
        assert expected == actual, 'Expected={}\nActual={}, actual node:\n{}\nSQL=\n{}'.format(
            expected, actual, node, sql)

    return node.cost


def GetLatencyFromPg(sql, hint, verbose=False, check_hint_used=False):
    with pg_executor.Cursor() as cursor:
        # GEQO must be disabled for hinting larger joins to work.
        # Why 'verbose': makes ParsePostgresPlanJson() able to access required
        # fields, e.g., 'Output' and 'Alias'.  Also see SqlToPlanNode() comment.
        geqo_off = hint is not None and len(hint) > 0
        result = _run_explain('explain(verbose, format json, analyze)',
                              sql,
                              hint,
                              verbose=True,
                              geqo_off=geqo_off,
                              cursor=cursor).result

    json_dict = result[0][0][0]
    latency = float(json_dict['Execution Time'])
    node0 = ParsePostgresPlanJson(json_dict)
    node = plans_lib.FilterScansOrJoins(node0)

    if check_hint_used:
        expected = hint
        actual = node.hint_str(with_physical_hints=ContainsPhysicalHints(hint))
        assert expected == actual, 'Expected={}\nActual={}, actual node:\n{}\nSQL=\n{}'.format(
            expected, actual, node, sql)

    return latency


def GetCardinalityEstimateFromPg(sql, verbose=False):
    _, json_dict = SqlToPlanNode(sql, verbose=verbose)
    return json_dict['Plan']['Plan Rows']


def _run_explain(explain_str,
                 sql,
                 comment,
                 verbose,
                 geqo_off=False,
                 timeout_ms=None,
                 cursor=None,
                 is_test=False,
                 remote=False):
    """
    Run the given SQL statement with appropriate EXPLAIN commands.

    timeout_ms is for both setting the timeout for PG execution and for the PG
    cluster manager, which will release the server after timeout expires.
    """
    # if is_test:
    #     assert remote, "testing queries must run on remote Postgres servers"
    if cursor is None and not remote:
        with pg_executor.Cursor() as cursor:
            return _run_explain(explain_str, sql, comment, verbose, geqo_off,
                                timeout_ms, cursor, remote)

    end_of_comment_idx = sql.find('*/')
    if end_of_comment_idx == -1:
        existing_comment = None
    else:
        split_idx = end_of_comment_idx + len('*/\n')
        existing_comment = sql[:split_idx]
        sql = sql[split_idx:]

    # Fuse hint comments.
    if comment:
        assert comment.startswith('/*+') and comment.endswith('*/'), (
            'Don\'t know what to do with these', sql, existing_comment, comment)
        if existing_comment is None:
            fused_comment = comment
        else:
            comment_body = comment[len('/*+ '):-len(' */')].rstrip()
            existing_comment_body_and_tail = existing_comment[len('/*+'):]
            fused_comment = '/*+\n' + comment_body + '\n' + existing_comment_body_and_tail
    else:
        fused_comment = existing_comment

    if fused_comment:
        s = fused_comment + '\n' + str(explain_str).rstrip() + '\n' + sql
    else:
        s = str(explain_str).rstrip() + '\n' + sql

    if remote:
        assert cursor is None
        return pg_executor.ExecuteRemote(s, verbose, geqo_off, timeout_ms)
    else:
        return pg_executor.Execute(s, verbose, geqo_off, timeout_ms, cursor)


def _FilterExprsByAlias(exprs, table_alias):
    # Look for <table_alias>.<stuff>.
    pattern = re.compile('.*\(?\\b{}\\b\..*\)?'.format(table_alias))
    return list(filter(pattern.match, exprs))


def ParsePostgresPlanJson(json_dict):
    """Takes JSON dict, parses into a Node."""
    curr = json_dict['Plan']

    def _parse_pg(json_dict, select_exprs=None, indent=0):
        op = json_dict['Node Type']
        cost = json_dict['Total Cost']
        if op == 'Aggregate':
            op = json_dict['Partial Mode'] + op
            if select_exprs is None:
                # Record the SELECT <select_exprs> at the topmost Aggregate.
                # E.g., ['min(mi.info)', 'min(miidx.info)', 'min(t.title)'].
                select_exprs = json_dict['Output']

        # Record relevant info.
        curr_node = plans_lib.Node(op)
        curr_node.cost = cost
        # Only available if 'analyze' is set (actual execution).
        curr_node.actual_time_ms = json_dict.get('Actual Total Time')
        if 'Relation Name' in json_dict:
            curr_node.table_name = json_dict['Relation Name']
            curr_node.table_alias = json_dict['Alias']

        # Unary predicate on a table.
        if 'Filter' in json_dict:
            assert 'Scan' in op, json_dict
            assert 'Relation Name' in json_dict, json_dict
            curr_node.info['filter'] = json_dict['Filter']

        if 'Scan' in op and select_exprs:
            # Record select exprs that belong to this leaf.
            # Assume: SELECT <exprs> are all expressed in terms of aliases.
            filtered = _FilterExprsByAlias(select_exprs, json_dict['Alias'])
            if filtered:
                curr_node.info['select_exprs'] = filtered

        # Recurse.
        if 'Plans' in json_dict:
            for n in json_dict['Plans']:
                curr_node.children.append(
                    _parse_pg(n, select_exprs=select_exprs, indent=indent + 2))

        # Special case.
        if op == 'Bitmap Heap Scan':
            for c in curr_node.children:
                if c.node_type == 'Bitmap Index Scan':
                    # 'Bitmap Index Scan' doesn't have the field 'Relation Name'.
                    c.table_name = curr_node.table_name
                    c.table_alias = curr_node.table_alias

        return curr_node

    return _parse_pg(curr)


def EstimateFilterRows(nodes):
    """For each node, issues an EXPLAIN to estimates #rows of unary preds.

    Writes result back into node.info['all_filters_est_rows'], as { relation
    id: num rows }.
    """
    if isinstance(nodes, plans_lib.Node):
        nodes = [nodes]
    cache = {}
    with pg_executor.Cursor() as cursor:
        for node in nodes:
            for table_id, pred in node.info['all_filters'].items():
                key = (table_id, pred)
                if key not in cache:
                    sql = 'EXPLAIN(format json) SELECT * FROM {} WHERE {};'.format(
                        table_id, pred)
                    cursor.execute(sql)
                    json_dict = cursor.fetchall()[0][0][0]
                    num_rows = json_dict['Plan']['Plan Rows']
                    cache[key] = num_rows
    print('{} unique filters'.format(len(cache)))
    pprint.pprint(cache)
    for node in nodes:
        d = {}
        for table_id, pred in node.info['all_filters'].items():
            d[table_id] = cache[(table_id, pred)]
        node.info['all_filters_est_rows'] = d


def GetAllTableNumRows(rel_names):
    """Ask PG how many number of rows each rel in rel_names has.

    Returns:
      A dict, {rel name: # rows}.
    """

    CACHE = {
        'aka_name': 901343,
        'aka_title': 361472,
        'cast_info': 36244344,
        'char_name': 3140339,
        'comp_cast_type': 4,
        'company_name': 234997,
        'company_type': 4,
        'complete_cast': 135086,
        'info_type': 113,
        'keyword': 134170,
        'kind_type': 7,
        'link_type': 18,
        'movie_companies': 2609129,
        'movie_info': 14835720,
        'movie_info_idx': 1380035,
        'movie_keyword': 4523930,
        'movie_link': 29997,
        'name': 4167491,
        'person_info': 2963664,
        'role_type': 12,
        'title': 2528312,
    }

    d = {}
    with pg_executor.Cursor() as cursor:
        for rel_name in rel_names:
            if rel_name in CACHE:
                # Kind of slow to ask PG for this.  For some reason it doesn't
                # immediately return from catalog but instead seems to do scans.
                d[rel_name] = CACHE[rel_name]
                continue

            sql = 'SELECT count(*) FROM {};'.format(rel_name)
            print('Issue:', sql)
            cursor.execute(sql)
            num_rows = cursor.fetchall()[0][0]
            print(num_rows)
            d[rel_name] = num_rows
    return d
