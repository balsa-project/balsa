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
"""Plan search: dynamic programming, beam search, etc."""
import collections
import copy

import numpy as np

from balsa import costing
from balsa import hyperparams
import balsa.util.plans_lib as plans_lib

# Nest Loop lhs/rhs whitelist. Empirically determined from Postgres plans.  A
# more general solution is to delve into PG source code.
_NL_WHITE_LIST = set([
    ('Nested Loop', 'Index Scan'),
    ('Nested Loop', 'Seq Scan'),  # NOTE: SeqScan needs to be "small".
    ('Nested Loop', 'Index Only Scan'),
    ('Hash Join', 'Index Scan'),
    ('Hash Join', 'Index Only Scan'),
    ('Merge Join', 'Index Scan'),
    ('Seq Scan', 'Index Scan'),
    ('Seq Scan', 'Nested Loop'),  # NOTE: SeqScan needs to be "small".
    ('Seq Scan', 'Index Only Scan'),
    ('Index Scan', 'Index Scan'),
    ('Index Scan', 'Seq Scan'),  # NOTE: SeqScan needs to be "small".
])


def IsJoinCombinationOk(join_type,
                        left,
                        right,
                        avoid_eq_filters=False,
                        engine='postgres',
                        use_plan_restrictions=True):
    """Checks whether hinting a join would be accepted by Postgres.

    Due to Postgres' internal implementations, pg_hint_plan would pass through
    all hints but Postgres may still silently reject and rewrite them.  Here we
    guard against this using empirical checks (which may be more conservative
    than the exact criteria that PG uses).

    Args:
      join_type: str, the op of the planned join.
      left: Node, left child of the planned join with its scan type assigned.
      right: Node, rihgt child of the planned join with its scan type assigned.
      avoid_eq_filters: bool, whether to avoid for certain equality filter scans
          required for Ext-JOB planning to work.
    Returns:
      bool, whether the planned join is going to be respected.
    """
    if not use_plan_restrictions:
        return True

    if join_type == 'Nested Loop':
        return IsNestLoopOk(left, right)

    # Avoid problematic info_type + filter predicate combination for Ext-JOB
    # hints.
    if avoid_eq_filters:
        if right.table_name == 'info_type' and _IsFilterScan(right):
            return False
        if left.table_name == 'info_type' and _IsFilterScan(left):
            return False

    if join_type == 'Hash Join':
        return IsHashJoinOk(left, right)

    return True


def _IsFilterScan(n):
    return n.HasEqualityFilters() and n.IsScan()


# @profile
def _IsSmallScan(n):
    # Special case: check this leaf is "small".  Here we use a hack and treat
    # the small dim tables *_type as small.  A general solution is to delve
    # into PG code and figure out how many rows/how much cost are deemed as
    # small (likely need to compare with work_mem, etc).
    return n.table_name.endswith('_type')


# @profile
def IsNestLoopOk(left, right):
    """Nested Loop hint is only respected by PG in some scenarios."""
    l_op = left.node_type
    r_op = right.node_type
    if (l_op, r_op) not in _NL_WHITE_LIST:
        return False
    # Special cases: check the Seq Scan side is "small".
    if (l_op, r_op) == ('Seq Scan', 'Nested Loop'):
        return _IsSmallScan(left)
    if (l_op, r_op) in [
        ('Index Scan', 'Seq Scan'),
        ('Nested Loop', 'Seq Scan'),
    ]:
        return _IsSmallScan(right)
    # All other cases OK.
    return True


# @profile
def IsHashJoinOk(left, right):
    """Hash Join hint is only respected by PG in some scenarios."""
    l_op = left.node_type
    r_op = right.node_type

    if (l_op, r_op) == ('Index Scan', 'Hash Join'):
        # Allows iff (1) LHS is small and (2) there are select exprs.  This
        # rule is empirically determined.
        l_exprs = left.GetSelectExprs()
        r_exprs = right.GetSelectExprs()
        return _IsSmallScan(left) and (len(l_exprs) or len(r_exprs))
    # All other cases OK.
    return True


def EnumerateScanOps(node, scan_ops):
    if not node.IsScan():
        yield node
    else:
        for scan_op in scan_ops:
            if scan_op == 'Index Only Scan':
                continue
            yield node.ToScanOp(scan_op)


# @profile
def EnumerateJoinWithOps(left,
                         right,
                         join_ops,
                         scan_ops,
                         avoid_eq_filters=False,
                         engine='postgres',
                         use_plan_restrictions=True):
    """Yields all valid JoinOp(ScanOp(left), ScanOp(right))."""
    for join_op in join_ops:
        for l in EnumerateScanOps(left, scan_ops):
            for r in EnumerateScanOps(right, scan_ops):
                if not IsJoinCombinationOk(join_op, l, r, avoid_eq_filters,
                                           engine, use_plan_restrictions):
                    continue
                join = plans_lib.Node(join_op)
                join.children = [l, r]
                yield join


class DynamicProgramming(object):
    """Bottom-up dynamic programming plan search."""

    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        p.Define('cost_model', costing.NullCost.Params(),
                 'Params of the cost model to use.')
        p.Define('search_space', 'bushy',
                 'Options: bushy, dbmsx, bushy_norestrict.')

        # Physical planning.
        p.Define('plan_physical_ops', False, 'Do we plan physical joins/scans?')

        # On enumeration hook.
        p.Define(
            'collect_data_include_suboptimal', True, 'Call on enumeration'
            ' hooks on suboptimal plans for each k-relation?')
        return p

    def __init__(self, params):
        self.params = params.Copy()
        p = self.params
        self.cost_model = p.cost_model.cls(p.cost_model)
        self.on_enumerated_hooks = []

        assert p.search_space in ('bushy', 'dbmsx',
                                  'bushy_norestrict'), 'Not implemented.'

        self.join_ops = ['Join']
        self.scan_ops = ['Scan']
        self.use_plan_restrictions = (p.search_space != 'bushy_norestrict')

    def SetPhysicalOps(self, join_ops, scan_ops):
        """Must be called once if p.plan_physical_ops is true."""
        p = self.params
        assert p.plan_physical_ops
        self.join_ops = copy.deepcopy(join_ops)
        self.scan_ops = copy.deepcopy(scan_ops)

    def PushOnEnumeratedHook(self, func):
        """Executes func(Node, cost) on each enumerated and costed subplan.

        This can be useful for, e.g., collecting value function training data.

        The subplan does not have to be an optimal one.
        """
        self.on_enumerated_hooks.append(func)

    def PopOnEnumeratedHook(self):
        self.on_enumerated_hooks.pop()

    def Run(self, query_node, query_str):
        """Executes DP planning for a given query node/string.

        Returns:
           A tuple of:
             best_node: balsa.Node;
             dp_tables: dict of size N (number of table in query), where
               dp_table[i] is a dict mapping a sorted string of a relation set
               (e.g., 'mi,t'), to (cost, the best plan that joins this set).
        """
        p = self.params
        join_graph, all_join_conds = query_node.GetOrParseSql()
        assert len(join_graph.edges) == len(all_join_conds)
        # Base tables to join.
        query_leaves = query_node.CopyLeaves()
        dp_tables = collections.defaultdict(dict)  # level -> dp_table
        # Fill in level 1.
        for leaf_node in query_leaves:
            dp_tables[1][leaf_node.table_alias] = (0, leaf_node)

        fns = {
            'bushy': self._dp_bushy_search_space,
            'dbmsx': self._dp_dbmsx_search_space,
            'bushy_norestrict': self._dp_bushy_search_space,
        }
        fn = fns[p.search_space]
        return fn(query_node, join_graph, all_join_conds, query_leaves,
                  dp_tables)

    def _dp_bushy_search_space(self, original_node, join_graph, all_join_conds,
                               query_leaves, dp_tables):
        p = self.params
        num_rels = len(query_leaves)

        for level in range(2, num_rels + 1):
            dp_table = dp_tables[level]
            for level_i in range(1, level):
                level_j = level - level_i
                dp_table_i = dp_tables[level_i]
                dp_table_j = dp_tables[level_j]

                for l_ids, l_tup in dp_table_i.items():
                    for r_ids, r_tup in dp_table_j.items():
                        l = l_tup[1]
                        r = r_tup[1]
                        if not plans_lib.ExistsJoinEdgeInGraph(
                                l, r, join_graph):
                            # No join clause linking two sides.  Skip.
                            continue
                        l_ids_splits = l_ids.split(',')
                        r_ids_splits = r_ids.split(',')
                        if len(np.intersect1d(l_ids_splits, r_ids_splits)) > 0:
                            # A relation exists in both sides.  Skip.
                            continue
                        join_ids = ','.join(sorted(l_ids_splits + r_ids_splits))

                        # Otherwise, form a new join.
                        for join in EnumerateJoinWithOps(
                                l,
                                r,
                                self.join_ops,
                                self.scan_ops,
                                use_plan_restrictions=self.use_plan_restrictions
                        ):
                            join_conds = join.KeepRelevantJoins(all_join_conds)

                            cost = self.cost_model(join, join_conds)

                            if p.collect_data_include_suboptimal:
                                # Call registered hooks on the costed subplan.
                                for hook in self.on_enumerated_hooks:
                                    hook(join, cost)

                            # Record if better cost.
                            if join_ids not in dp_table or dp_table[join_ids][
                                    0] > cost:
                                dp_table[join_ids] = (cost, join)

        if not p.collect_data_include_suboptimal:
            # Only collect data on an optimal plan for each k-relation.
            for level in range(2, num_rels + 1):
                dp_table = dp_tables[level]
                for ids, tup in dp_table.items():
                    cost, plan = tup[0], tup[1]
                    for hook in self.on_enumerated_hooks:
                        hook(plan, cost)

        return list(dp_tables[num_rels].values())[0][1], dp_tables

    def _dp_dbmsx_search_space(self, original_node, join_graph, all_join_conds,
                               query_leaves, dp_tables):
        """For Dbmsx."""
        raise NotImplementedError


if __name__ == '__main__':
    p = DynamicProgramming.Params()
    print(p)
    dp = p.cls(p)
    print(dp)

    node = plans_lib.Node('Scan', table_name='title').with_alias('t')
    node.info['sql_str'] = 'SELECT * FROM title t;'
    print(dp.Run(node, node.info['sql_str']))
