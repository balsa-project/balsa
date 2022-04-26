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
"""Execution plan definitions and processing."""
import copy
import collections
import functools
import re

import networkx as nx
import numpy as np

from balsa.util import simple_sql_parser


class Node(object):
    """Basic AST node class.

    Example usage:
       n = Node('Nested Loop')
       n.cost = 23968.1
       n.info = {'explain_json': json_dict, 'sql_str': sql_str, ...}
       n.children = [...]
    """

    def __init__(self, node_type, table_name=None, cost=None):
        self.node_type = node_type
        self.cost = cost  # Total cost.
        self.actual_time_ms = None
        self.info = {}  # Extended info such as a JSON dict.
        self.children = []

        # Set for leaves (i.e., scan nodes).
        self.table_name = table_name
        self.table_alias = None

        # Internal cached fields.
        self._card = None  # Used in MinCardCost.
        self._leaf_scan_op_copies = {}

    def with_alias(self, alias):
        self.table_alias = alias
        return self

    def get_table_id(self, with_alias=True, alias_only=False):
        """Table id for disambiguation."""
        if with_alias and self.table_alias:
            if alias_only:
                return self.table_alias
            return self.table_name + ' AS ' + self.table_alias
        assert self.table_name is not None
        return self.table_name

    @functools.lru_cache(maxsize=128)
    def to_str(self, with_cost=True, indent=0):
        s = '' if indent == 0 else ' ' * indent
        if self.table_name is None:
            if with_cost:
                s += '{} cost={}\n'.format(self.node_type, self.cost)
            else:
                s += '{}\n'.format(self.node_type)
        else:
            if with_cost:
                s += '{} [{}] cost={}\n'.format(self.node_type,
                                                self.get_table_id(), self.cost)
            else:
                s += '{} [{}]\n'.format(
                    self.node_type,
                    self.get_table_id(),
                )
        for c in self.children:
            s += c.to_str(with_cost=with_cost, indent=indent + 2)
        return s

    def GetOrParseSql(self):
        """Parses the join graph of this node into (nx.Graph, join conds).

        If self.info['sql_str'] exists, parses this SQL string.  Otherwise
        parses the result of self.to_sql(self.info['all_join_conds'])---this is
        usually used for manually constructued sub-plans.

        Internally, try to read from a cached pickle file if it exists.
        """
        graph = self.info.get('parsed_join_graph')
        join_conds = self.info.get('parsed_join_conds')
        if graph is None or join_conds is None:
            sql_str = self.info.get('sql_str') or self.to_sql(
                self.info['overall_join_conds'], with_filters=False)
            graph, join_conds = simple_sql_parser.ParseSql(
                sql_str, self.info.get('path'), self.info.get('query_name'))
            assert graph is not None, sql_str

            self.info['parsed_join_graph'] = graph
            self.info['parsed_join_conds'] = join_conds
        return graph, join_conds

    def GetOrParseJoinGraph(self):
        """Returns this Node's join graph as a networkx.Graph."""
        graph = self.info.get('parsed_join_graph')
        if graph is None:
            # Restricting an overall graph onto this Node's leaf IDs.
            overall_graph = self.info.get('overall_join_graph')
            if overall_graph is not None:
                subgraph_view = overall_graph.subgraph(
                    self.leaf_ids(alias_only=True))
                assert len(subgraph_view) > 0
                graph = subgraph_view
                self.info['parsed_join_graph'] = subgraph_view
                return graph
            return self.GetOrParseSql()[0]
        return graph

    def GetSelectExprs(self):
        """Returns a list of SELECT exprs associated with this Node.

        These expressions are the ultimate query outputs.  During parsing into
        balsa.Node objects, we push down that original list to the corresponding
        leaves.
        """
        select_exprs = []

        def _Fn(l):
            exprs = l.info.get('select_exprs')
            if exprs:
                select_exprs.extend(exprs)

        MapLeaves(self, _Fn)
        return select_exprs

    def GetFilters(self):
        """Returns a list of filter conditions associated with this Node.

        The filters are parsed by Postgres and for each table, include all
        pushed-down predicates associated with that table.
        """
        filters = []
        MapLeaves(self, lambda l: filters.append(l.info.get('filter')))
        filters = list(filter(lambda s: s is not None, filters))
        return filters

    def GetEqualityFilters(self):
        """Returns the list of equality filter predicates.

        These expressions are of the form rel.attr = VALUE.
        """
        eq_filters = self.info.get('eq_filters')
        if eq_filters is None:
            filters = self.GetFilters()
            pattern = re.compile('[a-z][\da-z_]*\.[\da-z_]*\ = [\da-z_]+')
            equality_conds = []
            for clause in filters:
                equality_conds.extend(pattern.findall(clause))
            eq_filters = list(set(equality_conds))
            self.info['eq_filters'] = eq_filters
        return eq_filters

    def GetFilteredAttributes(self):
        """Returns the list of filtered attributes ([<alias>.<attr>])."""
        attrs = self.info.get('filtered_attributes')
        if attrs is None:
            attrs = []
            filters = self.GetFilters()
            # Look for <alias>.<attr>.
            pattern = re.compile('[a-z][\da-z_]*\.[\da-z_]+')
            for clause in filters:
                attrs.extend(pattern.findall(clause))
            attrs = list(set(attrs))
            self.info['filtered_attributes'] = attrs
        return attrs

    def KeepRelevantJoins(self, all_join_conds):
        """Returns join conditions relevant to this Node."""
        aliases = self.leaf_ids(alias_only=True)

        def _KeepRelevantJoins(s):
            splits = s.split('=')
            l, r = splits[0].strip(), splits[1].strip()
            l_alias = l.split('.')[0]
            r_alias = r.split('.')[0]
            return l_alias in aliases and r_alias in aliases

        joins = list(filter(_KeepRelevantJoins, all_join_conds))
        return joins

    @functools.lru_cache(maxsize=8)
    def leaf_ids(self, with_alias=True, return_depths=False, alias_only=False):
        ids = []
        if not return_depths:
            MapLeaves(
                self,
                lambda l: ids.append(l.get_table_id(with_alias, alias_only)))
            return ids

        depths = []

        def _Helper(leaf, depth):
            ids.append(leaf.get_table_id(with_alias, alias_only))
            depths.append(depth)

        MapLeavesWithDepth(self, _Helper)
        return ids, depths

    def Copy(self):
        """Returns a deep copy of self."""
        return copy.deepcopy(self)

    def CopyLeaves(self):
        """Returns a list of deep copies of the leaf nodes."""
        leaves = []
        MapLeaves(self, lambda leaf: leaves.append(copy.deepcopy(leaf)))
        return leaves

    def GetLeaves(self):
        """Returns a list of references to the leaf nodes."""
        leaves = []
        MapLeaves(self, lambda leaf: leaves.append(leaf))
        return leaves

    def IsJoin(self):
        return 'Join' in self.node_type or self.node_type == 'Nested Loop'

    def IsScan(self):
        return 'Scan' in self.node_type

    def HasEqualityFilters(self):
        return len(self.GetEqualityFilters()) > 0

    def ToScanOp(self, scan_op):
        """Retrieves a deep copy of self with scan_op assigned."""
        assert not self.children, 'This node must be a leaf.'
        copied = self._leaf_scan_op_copies.get(scan_op)
        if copied is None:
            copied = copy.deepcopy(self)
            copied.node_type = scan_op
            self._leaf_scan_op_copies[scan_op] = copied
        return copied

    def to_sql(self,
               all_join_conds,
               with_filters=True,
               with_select_exprs=False):
        # Join and filter predicates.
        joins = self.KeepRelevantJoins(all_join_conds)
        if with_filters:
            filters = self.GetFilters()
        else:
            filters = []
        # FROM.
        from_str = self.leaf_ids()
        from_str = ', '.join(from_str)
        # SELECT.
        if with_select_exprs:
            select_exprs = self.GetSelectExprs()
        else:
            select_exprs = []
        select_str = '*' if len(select_exprs) == 0 else ','.join(select_exprs)

        if len(filters) > 0 and len(joins) > 0:
            sql = 'SELECT {} FROM {} WHERE {} AND {};'.format(
                select_str, from_str, ' AND '.join(joins),
                ' AND '.join(filters))
        elif len(joins) > 0:
            sql = 'SELECT {} FROM {} WHERE {};'.format(select_str, from_str,
                                                       ' AND '.join(joins))
        elif len(filters) > 0:
            sql = 'SELECT {} FROM {} WHERE {};'.format(select_str, from_str,
                                                       ' AND '.join(filters))
        else:
            sql = 'SELECT {} FROM {};'.format(select_str, from_str)
        return sql

    @functools.lru_cache(maxsize=2)
    def hint_str(self, with_physical_hints=False):
        """Produces a plan hint such that query_plan (Node) is respected."""
        scans = []
        joins = []

        def helper(t):
            node_type = t.node_type.replace(' ', '')
            # PG uses the former & the extension expects the latter.
            node_type = node_type.replace('NestedLoop', 'NestLoop')
            if t.IsScan():
                scans.append(node_type + '(' + t.table_alias + ')')
                return [t.table_alias], t.table_alias
            rels = []  # Flattened
            leading = []  # Hierarchical
            for child in t.children:
                a, b = helper(child)
                rels.extend(a)
                leading.append(b)
            joins.append(node_type + '(' + ' '.join(rels) + ')')
            return rels, leading

        _, leading_hierarchy = helper(self)

        leading = 'Leading(' + str(leading_hierarchy) \
                    .replace('\'', '') \
                    .replace('[', '(') \
                    .replace(']', ')') \
                    .replace(',', '') + ')'
        if with_physical_hints:
            # Reverse the join hints to print largest/outermost joins first.
            atoms = joins[::-1] + scans + [leading]
        else:
            atoms = [leading]  # Join order hint only.
        query_hint = '\n '.join(atoms)
        return '/*+ ' + query_hint + ' */'

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return str(self) < str(other)


class WorkloadInfo(object):
    """Stores sets of possible relations/aliases/join types, etc.

    From a list of all Nodes, parse
    - all relation names
    - all join types
    - all scan types.
    These can also be specified manually for a workload.

    Attributes:
      rel_names, rel_ids, scan_types, join_types, all_ops: ndarray of sorted
        strings.
    """

    def __init__(self, nodes):
        rel_names = set()
        rel_ids = set()
        scan_types = set()
        join_types = set()
        all_ops = set()

        all_attributes = set()

        all_filters = collections.defaultdict(set)

        def _fill(root, node):
            all_ops.add(node.node_type)

            if node.table_name is not None:
                rel_names.add(node.table_name)
                rel_ids.add(node.get_table_id())

            if node.info and 'filter' in node.info:
                table_id = node.get_table_id()
                all_filters[table_id].add(node.info['filter'])

            if node.info and 'sql_str' in node.info:
                # We want "all" attributes but as an optimization, we keep the
                # attributes that are known to be filter-able.
                attrs = node.GetFilteredAttributes()
                all_attributes.update(attrs)

            if 'Scan' in node.node_type:
                scan_types.add(node.node_type)
            elif node.IsJoin():
                join_types.add(node.node_type)

            for c in node.children:
                _fill(root, c)

        for node in nodes:
            _fill(node, node)

        self.rel_names = np.asarray(sorted(list(rel_names)))
        self.rel_ids = np.asarray(sorted(list(rel_ids)))
        self.scan_types = np.asarray(sorted(list(scan_types)))
        self.join_types = np.asarray(sorted(list(join_types)))
        self.all_ops = np.asarray(sorted(list(all_ops)))
        self.all_attributes = np.asarray(sorted(list(all_attributes)))

    def SetPhysicalOps(self, join_ops, scan_ops):
        old_scans = self.scan_types
        old_joins = self.join_types
        if scan_ops is not None:
            self.scan_types = np.asarray(sorted(list(scan_ops)))
        if join_ops is not None:
            self.join_types = np.asarray(sorted(list(join_ops)))
        new_all_ops = [
            op for op in self.all_ops
            if op not in old_scans and op not in old_joins
        ]
        new_all_ops = new_all_ops + list(self.scan_types) + list(
            self.join_types)
        if len(self.all_ops) != len(new_all_ops):
            print('Search space (old=query nodes; new=agent action space):')
            print('old:', old_scans, old_joins, self.all_ops)
            print('new:', self.scan_types, self.join_types, self.all_ops)
        self.all_ops = np.asarray(sorted(list(set(new_all_ops))))

    def WithJoinGraph(self, join_graph):
        """Transforms { table -> neighbors } into internal representation."""
        self.join_edge_set = set()
        for t1, neighbors in join_graph.items():
            for t2 in neighbors:
                self.join_edge_set.add((t1, t2))
                self.join_edge_set.add((t2, t1))

    def Copy(self):
        return copy.deepcopy(self)

    def HasPhysicalOps(self):
        if not np.array_equal(self.scan_types, ['Scan']):
            return True
        if not np.array_equal(self.join_types, ['Join']):
            return True
        return False

    def __repr__(self):
        fmt = 'rel_names: {}\nrel_ids: {}\nscan_types: {}\n' \
        'join_types: {}\nall_ops: {}\nall_attributes: {}'
        return fmt.format(self.rel_names, self.rel_ids, self.scan_types,
                          self.join_types, self.all_ops, self.all_attributes)


def ExistsJoinEdgeInGraph(node1, node2, join_graph):
    """Checks if two nodes are connected via an edge in the join graph."""
    assert isinstance(join_graph, nx.Graph), join_graph
    leaves1 = node1.leaf_ids(alias_only=True)
    leaves2 = node2.leaf_ids(alias_only=True)
    edges = join_graph.edges
    for name1 in leaves1:
        for name2 in leaves2:
            if (name1, name2) in edges:
                return True
    return False


def MapNode(node, func):
    """Applies func over each subnode of 'node'."""
    func(node)
    for c in node.children:
        MapNode(c, func)


def MapNodeWithDepth(node, func, depth=0):
    """Applies func: (node, depth) -> U over each subnode of 'node'.

    The current node has a depth of 'depth' (defaults to 0). Any node in
    node.children has a depth of depth+1, etc.
    """
    func(node, depth)
    for c in node.children:
        MapNodeWithDepth(c, func, depth + 1)


def MapLeaves(node, func):
    """Applies func: node -> U over each leaf of 'node'."""

    def f(n):
        if len(n.children) == 0:
            func(n)

    MapNode(node, f)


def MapLeavesWithDepth(node, func, depth=0):
    """Applies func: (node, depth) -> U over each leaf of 'node'.

    The current node has a depth of 'depth' (defaults to 0). Any node in
    node.children has a depth of depth+1, etc.
    """
    assert node is not None

    def f(n, d):
        if len(n.children) == 0:
            func(n, d)

    MapNodeWithDepth(node, f, depth)


def RewriteAsGenericJoinsScans(nodes):

    def f(node):
        op = node.node_type
        if 'Scan' in op:
            node.node_type = 'Scan'
        elif 'Join' in op or 'Nested Loop' == op:
            node.node_type = 'Join'

    if isinstance(nodes, Node):
        nodes = [nodes]

    for node in nodes:
        MapNode(node, f)


def GatherUnaryFiltersInfo(nodes, with_alias=True, alias_only=False):
    """For each node, gather leaf filters into root.

    For node in nodes:
      fills in node.info['all_filters'] which is { relation name : pushed-down
      filter for that relation }.
    """

    if isinstance(nodes, Node):
        nodes = [nodes]

    for node in nodes:
        d = {}

        def f(leaf):
            if 'filter' in leaf.info:
                # It's possible to encounter a scan without any filters, e.g.,
                #
                #     ->  Index Scan using char_name_pkey on char_name chn
                #              Index Cond: (id = ci.person_role_id)
                #
                # whose corresponding pred is just 'chn.id = ci.person_role_id'.
                table_id = leaf.get_table_id(with_alias, alias_only)
                assert table_id not in d, (leaf.info, table_id, d)
                d[table_id] = leaf.info['filter']

        MapLeaves(node, f)
        node.info['all_filters'] = d


def FilterScansOrJoins(nodes):
    """Filters the trees: keeps only the scan and join nodes.

    Input nodes are copied and are not modified in-place.

    Examples of removed nodes (all unary): Aggregate, Gather, Hash, Materialize.
    """
    singleton_input = False
    if isinstance(nodes, Node):
        singleton_input = True
        nodes = [nodes]

    def _filter(node):
        if not node.IsScan() and not node.IsJoin():
            assert len(node.children) == 1, node
            return _filter(node.children[0])
        node.children = [_filter(c) for c in node.children]
        return node

    filtered = []
    for n in nodes:
        new_node = _filter(n.Copy())
        # Save top-level node's info and cost (which might be latency value
        # from actual execution), since the top-level node may get filtered
        # away.
        new_node.info = n.info
        new_node.cost = n.cost
        new_node.actual_time_ms = n.actual_time_ms
        filtered.append(new_node)
    if singleton_input:
        return filtered[0]
    return filtered


def GetAllSubtrees(nodes):
    """For node in nodes: yield_all_subtrees(node)."""
    trees = []

    def _fn(node, trees):
        trees.append(node)
        for c in node.children:
            _fn(c, trees)

    if isinstance(nodes, Node):
        nodes = [nodes]

    for node in nodes:
        _fn(node, trees)
    return trees


def GetAllSubtreesNoLeaves(nodes):
    """For node in nodes: yield_all_subtrees(node)."""
    trees = []

    def _fn(node, trees):
        if len(node.children):
            trees.append(node)
            for c in node.children:
                _fn(c, trees)

    if isinstance(nodes, Node):
        nodes = [nodes]

    for node in nodes:
        _fn(node, trees)
    return trees


class Featurizer(object):

    def __call__(self, node):
        """Node -> np.ndarray."""
        raise NotImplementedError

    def FeaturizeLeaf(self, node):
        """Featurizes a leaf Node."""
        raise NotImplementedError

    def Merge(self, node, left_vec, right_vec):
        """Featurizes a Node by merging the feature vectors of LHS/RHS."""
        raise NotImplementedError

    def Fit(self, nodes):
        """Computes normalization statistics; no-op for stateless."""
        return

    def PerturbQueryFeatures(self, query_feat, distribution):
        """Randomly perturbs a query feature vec returned by __call__()."""
        return query_feat

    def WithWorkloadInfo(self, workload_info):
        self.workload_info = workload_info
        return self


class TreeNodeFeaturizer(Featurizer):
    """Featurizes a single Node.

    Feature vector:
       [ one-hot for operator ] [ multi-hot for all relations under this node ]

    Width: |all_ops| + |rel_ids|.
    """

    def __init__(self, workload_info):
        self.workload_info = workload_info
        self.ops = workload_info.all_ops
        self.one_ops = np.eye(self.ops.shape[0], dtype=np.float32)
        self.rel_ids = workload_info.rel_ids
        assert not workload_info.HasPhysicalOps(), 'Physical ops found; use a '\
            'featurizer that supports them (PhysicalTreeNodeFeaturizer).'

    def __call__(self, node):
        num_ops = len(self.ops)
        vec = np.zeros(num_ops + len(self.rel_ids), dtype=np.float32)
        # Node type.
        vec[:num_ops] = self.one_ops[np.where(self.ops == node.node_type)[0][0]]
        # Joined tables: [table: 1].
        joined = node.leaf_ids()
        for rel_id in joined:
            idx = np.where(self.rel_ids == rel_id)[0][0]
            vec[idx + num_ops] = 1.0
        assert vec[num_ops:].sum() == len(joined)
        return vec

    def FeaturizeLeaf(self, node):
        assert node.IsScan()
        vec = np.zeros(len(self.ops) + len(self.rel_ids), dtype=np.float32)
        rel_id = node.get_table_id()
        rel_idx = np.where(self.rel_ids == rel_id)[0][0]
        vec[len(self.ops) + rel_idx] = 1.0
        return vec

    def Merge(self, node, left_vec, right_vec):
        assert node.IsJoin()
        len_join_enc = len(self.ops)
        # The relations under 'node' and their scan types.  Merging <=> summing.
        vec = left_vec + right_vec
        # Make sure the first part is correct.
        vec[:len_join_enc] = self.one_ops[np.where(
            self.ops == node.node_type)[0][0]]
        return vec


class PhysicalTreeNodeFeaturizer(TreeNodeFeaturizer):
    """Featurizes a single Node with support for physical operators.

    Feature vector:
       [ one-hot for join operator ] concat
       [ multi-hot for all relations under this node ]

    Width: |join_ops| + |rel_ids| * |scan_ops|.
    """

    def __init__(self, workload_info):
        self.workload_info = workload_info
        self.join_ops = workload_info.join_types
        self.scan_ops = workload_info.scan_types
        self.rel_ids = workload_info.rel_ids
        self.join_one_hot = np.eye(len(self.join_ops), dtype=np.float32)

    def __call__(self, node):
        # Join op of this node.
        if node.IsJoin():
            join_encoding = self.join_one_hot[np.where(
                self.join_ops == node.node_type)[0][0]]
        else:
            join_encoding = np.zeros(len(self.join_ops), dtype=np.float32)
        # For each table: [ one-hot of scan ops ].  Concat across tables.
        scan_encoding = np.zeros(len(self.scan_ops) * len(self.rel_ids),
                                 dtype=np.float32)
        joined = node.CopyLeaves()
        for rel_node in joined:
            rel_id = rel_node.get_table_id()
            rel_idx = np.where(self.rel_ids == rel_id)[0][0]
            scan_operator_idx = np.where(
                self.scan_ops == rel_node.node_type)[0][0]
            idx = rel_idx * len(self.scan_ops) + scan_operator_idx
            scan_encoding[idx] = 1.0
        # Concatenate to create final node encoding.
        vec = np.concatenate((join_encoding, scan_encoding))
        return vec

    def FeaturizeLeaf(self, node):
        assert node.IsScan()
        vec = np.zeros(len(self.join_ops) +
                       len(self.scan_ops) * len(self.rel_ids),
                       dtype=np.float32)
        rel_id = node.get_table_id()
        rel_idx = np.where(self.rel_ids == rel_id)[0][0]
        scan_operator_idx = np.where(self.scan_ops == node.node_type)[0][0]
        idx = rel_idx * len(self.scan_ops) + scan_operator_idx
        vec[len(self.join_ops) + idx] = 1.0
        return vec

    def Merge(self, node, left_vec, right_vec):
        assert node.IsJoin()
        len_join_enc = len(self.join_ops)
        # The relations under 'node' and their scan types.  Merging <=> summing.
        vec = left_vec + right_vec
        # Make sure the first part is correct.
        vec[:len_join_enc] = self.join_one_hot[np.where(
            self.join_ops == node.node_type)[0][0]]
        return vec


class PreOrderSequenceFeaturizer(Featurizer):

    def __init__(self, workload_info):
        self.workload_info = workload_info

        # Union of all relation names and operator types.
        self.vocab = np.concatenate(
            (workload_info.all_ops, workload_info.rel_names))

        print('PreOrderSequenceFeaturizer vocab', self.vocab)

    def pad(self):
        return len(self.vocab)

    def _pre_order(self, parent, n, vecs):
        """Each node yields up to 2 tokens: <op type> <rel name (if scan)>."""
        vecs.append(np.where(self.vocab == n.node_type)[0][0])
        if len(n.children) == 0:
            name = n.table_name
            vecs.append(np.where(self.vocab == name)[0][0])
        else:
            for c in n.children:
                self._pre_order(n, c, vecs)

    def __call__(self, node):
        vecs = []
        self._pre_order(None, node, vecs)
        return np.asarray(vecs).astype(np.int64, copy=False)


class ParentPositionFeaturizer(Featurizer):
    """Node -> parent ID, where IDs are assigned DFS-order."""

    def __init__(self, workload_info):
        self.pad_idx = len(workload_info.rel_names)
        self.pad_idx = 50  # FIXME

    def pad(self):
        return self.pad_idx

    def _walk(self, parent, n, vecs, parent_id, curr_id):
        vecs.append(parent_id)
        if len(n.children) == 0:
            # [Scan, TableName] corresponds to the same parent.
            vecs.append(parent_id)
        for c in n.children:
            self._walk(n, c, vecs, parent_id=curr_id, curr_id=curr_id + 1)

    def __call__(self, node):
        vecs = []
        self._walk(None, node, vecs, parent_id=-1, curr_id=0)
        vecs = np.asarray(vecs).astype(np.int64, copy=False)
        vecs += 1  # Shift by one since we have -1 for root.
        return np.arange(len(vecs))  # FIXME


class QueryFeaturizer(Featurizer):
    """Concat [join graph] [table T: log(est_rows(T) + 1)/log(max_rows)]."""

    def __init__(self, workload_info):
        self.workload_info = workload_info
        self.MAX_ROW_COUNT = max(workload_info.table_num_rows.values())
        self.table_id_to_name = lambda table_id: table_id.split(' ')[0]
        self.table_id_to_alias = lambda table_id: table_id.split(' ')[-1]
        print(
            'QueryFeaturizer, {} rel_ids'.format(len(
                self.workload_info.rel_ids)), self.workload_info.rel_ids)

    def __call__(self, node):
        vec = np.zeros_like(self.workload_info.rel_ids, dtype=np.float32)

        # Joined tables.
        # [table: row count of each table]
        joined = node.leaf_ids()
        for rel_id in joined:
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx] = self.workload_info.table_num_rows[self.table_id_to_name(
                rel_id)]

        # Filtered tables.
        for rel_id, est_rows in node.info['all_filters_est_rows'].items():
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            assert vec[idx] > 0, (node, node.info['all_filters_est_rows'])
            total_rows = self.workload_info.table_num_rows[
                self.table_id_to_name(rel_id)]

            # NOTE: without ANALYZE, for some reason this predicate is
            # estimated to have 703 rows, whereas the table only has 4 rows:
            #   (kind IS NOT NULL) AND ((kind)::text <> 'production
            #   companies'::text)
            # With ANALYZE run, this assert passes.
            assert est_rows >= 0 and est_rows <= total_rows, (node.info,
                                                              est_rows,
                                                              total_rows)
            sel = est_rows / total_rows
            # TODO: an additional signal is to provide selectivity.
            vec[idx] = est_rows

        vec = np.log(vec + 1.0) / np.log(self.MAX_ROW_COUNT)

        # Query join graph features: adjacency matrix.
        query_join_graph = node.GetOrParseJoinGraph()
        all_aliases = [
            self.table_id_to_alias(table)
            for table in self.workload_info.rel_ids
        ]
        adj_matrix = nx.to_numpy_array(query_join_graph, nodelist=all_aliases)
        # Sufficient to grab the upper-triangular portion (since graph is
        # undirected).  k=1 means don't grab the diagnoal (all 1s).
        triu = adj_matrix[np.triu_indices(len(all_aliases),
                                          k=1)].astype(np.float32)

        features = np.concatenate((triu, vec), axis=None)
        return features
