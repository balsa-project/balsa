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

import re

import networkx as nx


def _CanonicalizeJoinCond(join_cond):
    """join_cond: 4-tuple"""
    t1, c1, t2, c2 = join_cond
    if t1 < t2:
        return join_cond
    return t2, c2, t1, c1


def _DedupJoinConds(join_conds):
    """join_conds: list of 4-tuple (t1, c1, t2, c2)."""
    canonical_join_conds = [_CanonicalizeJoinCond(jc) for jc in join_conds]
    return sorted(set(canonical_join_conds))


def _GetJoinConds(sql):
    """Returns a list of join conditions in the form of (t1, c1, t2, c2)."""
    join_cond_pat = re.compile(
        r"""
        (\w+)  # 1st table
        \.     # the dot "."
        (\w+)  # 1st table column
        \s*    # optional whitespace
        =      # the equal sign "="
        \s*    # optional whitespace
        (\w+)  # 2nd table
        \.     # the dot "."
        (\w+)  # 2nd table column
        """, re.VERBOSE)
    join_conds = join_cond_pat.findall(sql)
    return _DedupJoinConds(join_conds)


def _GetGraph(join_conds):
    g = nx.Graph()
    for t1, c1, t2, c2 in join_conds:
        g.add_edge(t1, t2, join_keys={t1: c1, t2: c2})
    return g


def _FormatJoinCond(tup):
    t1, c1, t2, c2 = tup
    return f"{t1}.{c1} = {t2}.{c2}"


def ParseSql(sql, filepath=None, query_name=None):
    """Parses a SQL string into (nx.Graph, a list of join condition strings).

    Both use aliases to refer to tables.
    """
    join_conds = _GetJoinConds(sql)
    graph = _GetGraph(join_conds)
    join_conds = [_FormatJoinCond(c) for c in join_conds]
    return graph, join_conds
