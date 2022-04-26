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

import collections
import copy
import gc
import glob
import os
import pickle
import pprint
import time

import numpy as np

from balsa.models import treeconv
from balsa.util import graphs, plans_lib, postgres


def TreeConvFeaturize(plan_featurizer, subplans):
    """Returns (featurized plans, tree conv indexes) tensors."""
    assert len(subplans) > 0
    # This class currently requires batch-featurizing, due to internal
    # padding.  This is different from our other per-node Featurizers.
    print('Calling make_and_featurize_trees()...')
    t1 = time.time()
    trees, indexes = treeconv.make_and_featurize_trees(subplans,
                                                       plan_featurizer)
    print('took {:.1f}s'.format(time.time() - t1))
    return trees, indexes


class Experience(object):

    def __init__(
        self,
        data,
        tree_conv=False,
        keep_scans_joins_only=True,
        plan_featurizer_cls=plans_lib.PreOrderSequenceFeaturizer,
        query_featurizer_cls=plans_lib.QueryFeaturizer,
        workload_info=None,
    ):
        self.tree_conv = tree_conv
        if keep_scans_joins_only:
            print('plans_lib.FilterScansOrJoins()')
            self.nodes = plans_lib.FilterScansOrJoins(data)
        else:
            self.nodes = data
        self.initial_size = len(self.nodes)

        self.plan_featurizer_cls = plan_featurizer_cls
        self.query_featurizer_cls = query_featurizer_cls
        self.query_featurizer = None
        self.workload_info = workload_info

        #### Affects query featurization.
        print('plans_lib.GatherUnaryFiltersInfo()')
        # Requires:
        #   leaf.info['filter'] for leaf under node.
        # Writes:
        #   node.info['all_filters'], which is used by EstimateFilterRows()
        #   below.
        plans_lib.GatherUnaryFiltersInfo(self.nodes)

        print('postgres.EstimateFilterRows()')
        # Get histogram-estimated # rows of filters from PG.
        # Requires:
        #   node.info['all_filters']
        # Writes:
        #   node.info['all_filters_est_rows'], which is used by, e.g.,
        #   plans_lib.QueryFeaturizer.
        #
        # TODO: check that first N nodes don't change.
        postgres.EstimateFilterRows(self.nodes)

    def Save(self, path):
        """Saves all Nodes in the current replay buffer to a file."""
        if os.path.exists(path):
            old_path = path
            path = '{}-{}'.format(old_path, time.time())
            print('Path {} exists, appending current time: {}'.format(
                old_path, path))
            assert not os.path.exists(path), path
        to_save = (self.initial_size, self.nodes)
        with open(path, 'wb') as f:
            pickle.dump(to_save, f)
        print('Saved Experience to:', path)

    def Load(self, path_glob, keep_last_fraction=1):
        """Loads multiple serialized Experience buffers into a single one.

        The 'initial_size' Nodes from self would be kept, while those from the
        loaded buffers would be dropped.  Internally, checked that all buffers
        and self have the same 'initial_size' field.
        """
        paths = glob.glob(os.path.expanduser(path_glob))
        if not paths:
            raise ValueError('No replay buffer files found')
        assert 0 <= keep_last_fraction <= 1, keep_last_fraction
        # query name -> set(plan string)
        total_unique_plans_table = collections.defaultdict(set)
        total_num_unique_plans = 0
        initial_nodes_len = len(self.nodes)
        for path in paths:
            t1 = time.time()
            print('Loading replay buffer', path)
            # np.load() is much faster than pickle.load; disabling gc provides
            # further speedups.
            gc.disable()
            loaded = np.load(path, allow_pickle=True)
            gc.enable()
            print('  ...took {:.1f} seconds'.format(time.time() - t1))
            initial_size, nodes = loaded
            # Sanity checks & some invariant checks.  A more stringent check
            # would be to check that:
            #   buffer 1: qname_0 qname_1 ...
            #   ...
            #   buffer N: qname_0 qname_1, ...
            # I.e., query names all correspond.
            assert type(initial_size) is int and type(nodes) is list, path
            assert initial_size == self.initial_size, (path, initial_size,
                                                       self.initial_size)
            assert len(nodes) >= initial_size and len(
                nodes) % initial_size == 0, (len(nodes), path)
            nodes_executed = nodes[initial_size:]
            if keep_last_fraction < 1:
                assert len(nodes_executed) % initial_size == 0
                num_iters = len(nodes_executed) // initial_size
                keep_num_iters = int(num_iters * keep_last_fraction)
                print('  orig len {} keeping the last fraction {} ({} iters)'.
                      format(len(nodes_executed), keep_last_fraction,
                             keep_num_iters))
                nodes_executed = nodes_executed[-(keep_num_iters *
                                                  initial_size):]
            self.nodes.extend(nodes_executed)
            # Analysis.
            num_unique_plans, unique_plans_table = Experience.CountUniquePlans(
                self.initial_size, nodes_executed)
            total_num_unique_plans_prev = total_num_unique_plans
            total_num_unique_plans = Experience.MergeUniquePlansInto(
                unique_plans_table, total_unique_plans_table)
            print('  num_unique_plans from loaded buffer {}; actually '\
                  'new unique plans contributed (after merging) {}'.
                  format(num_unique_plans,
                         total_num_unique_plans - total_num_unique_plans_prev))
        print('Loaded {} nodes from {} buffers; glob={}, paths:\n{}'.format(
            len(self.nodes) - initial_nodes_len, len(paths), path_glob,
            '\n'.join(paths)))
        print('Total unique plans (num_query_execs):', total_num_unique_plans)

    @classmethod
    def CountUniquePlans(cls, num_templates, nodes):
        assert len(nodes) % num_templates == 0, (len(nodes), num_templates)
        unique_plans = collections.defaultdict(set)
        for i in range(num_templates):
            query_name = nodes[i].info['query_name']
            hint_set = unique_plans[query_name]
            for j, node in enumerate(nodes[i::num_templates]):
                assert node.info['query_name'] == query_name, (
                    node.info['query_name'], query_name)
                hint = node.hint_str(with_physical_hints=True)
                hint_set.add(hint)
        num_unique_plans = sum([len(s) for s in unique_plans.values()])
        return num_unique_plans, unique_plans

    @classmethod
    def MergeUniquePlansInto(cls, from_table, into_table):
        for query_name, unique_plans in from_table.items():
            into = into_table[query_name]
            into_table[query_name] = into.union(unique_plans)
        return sum([len(s) for s in into_table.values()])

    def prepare(self, rewrite_generic=False, verbose=False):
        #### Affects plan featurization.
        if rewrite_generic:
            print('Rewriting all joins -> Join, all scans -> Scan')
            plans_lib.RewriteAsGenericJoinsScans(self.nodes)
        ############################

        if self.workload_info is None:
            print('Creating WorkloadInfo')
            self.workload_info = plans_lib.WorkloadInfo(self.nodes)
            self.workload_info.WithJoinGraph(
                graphs.JOIN_ORDER_BENCHMARK_JOIN_GRAPH)

        rel_names, scan_types, join_types, all_ops = (
            self.workload_info.rel_names, self.workload_info.scan_types,
            self.workload_info.join_types, self.workload_info.all_ops)
        print('{} rels: {}'.format(len(rel_names), rel_names))
        print('{} rel_ids: {}'.format(len(self.workload_info.rel_ids),
                                      self.workload_info.rel_ids))
        print('{} scans: {}'.format(len(scan_types), scan_types))
        print('{} joins: {}'.format(len(join_types), join_types))
        print('{} all ops: {}'.format(len(all_ops), all_ops))

        # count(*) from T for all T in self.workload_info.rel_names.
        self.workload_info.table_num_rows = postgres.GetAllTableNumRows(
            self.workload_info.rel_names)

        if self.tree_conv:
            assert issubclass(self.plan_featurizer_cls,
                              plans_lib.TreeNodeFeaturizer)
            self.featurizer = self.plan_featurizer_cls(self.workload_info)
            self.pos_featurizer = None
        else:
            self.featurizer = self.plan_featurizer_cls(self.workload_info)
            self.pos_featurizer = plans_lib.ParentPositionFeaturizer(
                self.workload_info)
        if self.query_featurizer is None:
            # Instantiate only once as it may have global normalization stats.
            if isinstance(self.query_featurizer_cls, plans_lib.Featurizer):
                # A stateful, already-instantiated featurizer.
                self.query_featurizer = self.query_featurizer_cls
            else:
                self.query_featurizer = self.query_featurizer_cls(
                    self.workload_info)
        # Update its WorkloadInfo (does the wi really change?).
        self.query_featurizer.WithWorkloadInfo(self.workload_info)

        if isinstance(self.featurizer, plans_lib.PreOrderSequenceFeaturizer):
            # Assert featurizers output same size.
            ns = plans_lib.GetAllSubtrees([self.nodes[0]])
            f = self.featurizer(ns[0])
            pf = self.pos_featurizer(ns[0])
            assert len(f) == len(pf), (len(f), len(pf))

    def GetFirstIndexForTemplate(self,
                                 template_index,
                                 skip_first_n,
                                 use_last_n_iters=-1):
        """Allows featurizing just last n iters' data."""
        # The layout of self.nodes looks like:
        #
        #   (expert nodes) template 0, ..., template initial_size-1
        #         (iter 0) template 0, ..., template initial_size-1
        #         (iter 1) template 0, ..., template initial_size-1
        #              ...
        #         (iter k) template 0, ..., template initial_size-1
        #
        # 'template_index' refers to which template we are looking for.
        #
        # 'skip_first_n' allows skipping the first initial_size expert nodes.
        # It can only be 0 or initial_size.
        #
        # 'use_last_n_iters': if -1, consider all iterations; otherwise,
        # consider the last few iters; in both cases 'skip_first_n' is applied
        # first.
        #
        # Example: if template_index = 1, skip_first_n = 94, use_last_n_iters =
        # 1 and there are 2 agent-iters now, we should return 1 + 94 + 94.  If
        # use_last_n_iters is -1 instead, we return 1+94.
        assert skip_first_n in [0, self.initial_size], (skip_first_n,
                                                        self.initial_size)
        num_iters = (len(self.nodes) - skip_first_n) // self.initial_size
        i_start = template_index + skip_first_n
        # FIXME: this seems buggy; how is it correct for use_last_n_iters=-1?
        if num_iters <= use_last_n_iters:
            return i_start
        return i_start + (num_iters - use_last_n_iters) * self.initial_size

    def ComputeBestLatencies(self, template_index, skip_first_n,
                             with_physical_hints, skip_training_on_timeouts):
        """Computes best latencies per subplan for a particular query template.

        Uses all data for a template to build a dictionary:
            hint->(best latency, best subplan).
        This should be called once for each query tempalte index, since
        template can be seen as an episode/a goal.

        Returns:
          (The said dict, num_subtrees for this template, a num_iter-sized list
          of all per-iter subtrees).
        """
        i_start = template_index + skip_first_n
        subplan_to_best = {}  # Hint str -> (cost, subplan).
        num_subtrees = 0
        all_subtrees = []
        for j, node in enumerate(self.nodes[i_start::self.initial_size]):
            if skip_training_on_timeouts and getattr(node, 'is_timeout', False):
                all_subtrees.append([])
                continue
            subtrees = plans_lib.GetAllSubtreesNoLeaves(node)
            num_subtrees += len(subtrees)
            all_subtrees.append(subtrees)
            for t in subtrees:
                t_key = t.hint_str(with_physical_hints)
                curr_cost, _ = subplan_to_best.get(t_key, (1e30, None))
                if node.cost < curr_cost:
                    subplan_to_best[t_key] = (node.cost, t)
        return subplan_to_best, num_subtrees, all_subtrees

    def _featurize_dedup(self,
                         rewrite_generic=False,
                         verbose=False,
                         skip_first_n=0,
                         physical_execution_hindsight=False,
                         on_policy=False,
                         use_last_n_iters=-1,
                         use_new_data_only=False,
                         skip_training_on_timeouts=False):
        if use_last_n_iters > 0 and use_new_data_only:
            print('Both use_last_n_iters > 0 and use_new_data_only are set: '\
                  'letting the latter take precedence.')
        if on_policy:
            assert use_last_n_iters == -1, 'Cannot have both on_policy and'\
                ' use_last_n_iters={} set.'.format(use_last_n_iters)
            use_last_n_iters = 1

        self.prepare(rewrite_generic, verbose)
        # Training data.
        all_query_vecs = []
        all_feat_vecs = []  # plan features
        all_pos_vecs = []  # plan positions/indexes
        all_costs = []
        all_subtrees = []
        # Logging.
        # Num subtrees in the whole buffer (after skip_first_n).
        num_total_subtrees = 0
        # Num new subplans from the latest iter.
        num_new_datapoints = 0

        # Loop through each template.
        for i in range(self.initial_size):
            # Global label correction.
            subplan_to_best, num_subtrees, subtrees = self.ComputeBestLatencies(
                i,
                skip_first_n,
                with_physical_hints=not rewrite_generic,
                skip_training_on_timeouts=skip_training_on_timeouts)
            num_total_subtrees += num_subtrees
            # [(best cost, best subplan)]
            to_featurize = []

            if use_last_n_iters > 0:
                # Ignore the dedup and use all subplans from the last n iters.
                for iter_k_subplans in subtrees[-use_last_n_iters:]:
                    for subplan in iter_k_subplans:
                        key = subplan.hint_str(
                            with_physical_hints=not rewrite_generic)
                        best_cost, _ = subplan_to_best[key]
                        to_featurize.append((best_cost, subplan))

                # Log how many points from the last iter have new labels.
                last_iter_template_cost = \
                    self.nodes[-self.initial_size + i].cost
                last_iter_subplans = iter_k_subplans
                new_data_points = []
                if last_iter_subplans:
                    # last_iter_subplans can be empty (if timeouts and skip
                    # training on timeouts).
                    for tup in to_featurize[-len(last_iter_subplans):]:
                        best_cost = tup[0]
                        if best_cost == last_iter_template_cost:
                            num_new_datapoints += 1
                            new_data_points.append(tup)

                if use_new_data_only:
                    to_featurize = new_data_points
            else:
                # Dedup: keep unique subplans in all previous iters.
                to_featurize = subplan_to_best.values()

            # Actually featurize and fill the tensor buffers.
            # (1) All sub-plans share the same episodic goal -- the root query.
            query_feat = self.query_featurizer(self.nodes[i + skip_first_n])
            all_query_vecs.extend([query_feat] * len(to_featurize))
            # (2) Costs/plan feats.
            if not self.tree_conv:
                for best_cost, best_subplan in to_featurize:
                    all_costs.append(best_cost)
                    all_feat_vecs.append(self.featurizer(best_subplan))
                    all_pos_vecs.append(self.pos_featurizer(best_subplan))
            else:
                for best_cost, best_subplan in to_featurize:
                    all_costs.append(best_cost)
                    all_subtrees.append(best_subplan)

        # Tree conv requires batch-featurization.
        if self.tree_conv and all_subtrees:
            assert len(all_feat_vecs) == 0 and len(all_pos_vecs) == 0
            all_feat_vecs, all_pos_vecs = TreeConvFeaturize(
                self.featurizer, all_subtrees)

        # Logging.
        print(
            'num_total_subtrees={} num_featurized_subtrees={} '\
            'num_new_datapoints={}'
            .format(num_total_subtrees, len(all_query_vecs),
                    num_new_datapoints))
        print('head')
        for i in range(3):
            print('  query={:.3f} feat={} cost={}'.format(
                all_query_vecs[i].sum(), all_feat_vecs[i].sum(), all_costs[i]))
        print('tail')
        for i in range(3):
            j = -1 - i
            print('  query={:.3f} feat={} cost={}'.format(
                all_query_vecs[j].sum(), all_feat_vecs[j].sum(), all_costs[j]))
        return (all_query_vecs, all_feat_vecs, all_pos_vecs, all_costs,
                num_new_datapoints)

    def _featurize_hindsight_relabeling(self,
                                        rewrite_generic=False,
                                        verbose=False,
                                        skip_first_n=0,
                                        physical_execution_hindsight=False,
                                        use_last_n_iters=-1):
        assert self.tree_conv
        self.prepare(rewrite_generic, verbose)

        # Training data.
        all_query_vecs = []
        all_feat_vecs = []  # plan features
        all_pos_vecs = []  # plan positions/indexes
        all_costs = []
        all_subtrees = []
        num_total_subtrees = 0

        def TopDownCollect(node,
                           hindsight_goal_costs,
                           accum,
                           info_to_attach,
                           is_top_level=True):
            assert node.IsJoin() or node.IsScan(), node
            if node.IsScan():
                return
            if (node.actual_time_ms is not None or
                (skip_first_n == 0 and len(self.nodes) == self.initial_size and
                 is_top_level)):
                # The 'actual_time_ms' field would be None only for
                #
                # (1) the non-top-level operators in a timeout plan.  The
                #   top-level operator would be manually assigned a timeout
                #   label to the field 'actual_time_ms'.
                # (2) 'initial_size' many nodes (expert experience).  In this
                #   case we don't actually use these labels, so grab the
                #   top-level 'cost' field which is the expert latency.
                #
                # If not None, yield a hindsight goal/label.
                goal = node
                goal.info = copy.deepcopy(info_to_attach)  # TODO: copy needed?
                # 'goal_cost': latency for finishing computing the 'node' tree.
                if skip_first_n == 0 and len(self.nodes) == self.initial_size:
                    assert node.actual_time_ms is None, node
                    goal_cost = node.cost
                else:
                    goal_cost = node.actual_time_ms
                hindsight_goal_costs.append((goal, goal_cost))
            # Treating 'node' as a subplan, yield a data point per each
            # higher-level goal.  Note that 'node' can be a goal itself.
            for goal, goal_cost in hindsight_goal_costs:
                accum.append(
                    SubplanGoalCost(subplan=node, goal=goal, cost=goal_cost))
            for c in node.children:
                TopDownCollect(c,
                               hindsight_goal_costs,
                               accum,
                               info_to_attach,
                               is_top_level=False)

        for i in range(self.initial_size):
            # Loop through each template.
            i_start = self.GetFirstIndexForTemplate(i, skip_first_n,
                                                    use_last_n_iters)
            accum = []
            # Required by the query featurizer.
            info_to_attach = ['all_filters_est_rows']
            info_to_attach = {k: self.nodes[i].info[k] for k in info_to_attach}
            for j, node in enumerate(self.nodes[i_start::self.initial_size]):
                # Recurse down 'node' and collect all SubplanGoalCost with
                # hindsight relabeling.
                TopDownCollect(
                    node.Copy(),  # TODO: need this copy?
                    hindsight_goal_costs=[],
                    accum=accum,
                    info_to_attach=info_to_attach)
            num_total_subtrees += len(accum)
            # Dedup 'accum' by keeping the best latency per (subplan,goal).
            # (subplan, goal) -> best latency.
            best = collections.defaultdict(lambda: np.inf)
            ret = {}
            for point in accum:
                # NOTE: when this function turns the 'goal' part into a string,
                # some information is not preserved (e.g., the string doesn't
                # record filter info).  However, since we assume 'points' all
                # come from the same query, this simplification is OK for
                # uniquifying.
                key = point.ToSubplanGoalHint(
                    with_physical_hints=not rewrite_generic)
                if point.cost < best[key]:
                    best[key] = point.cost
                    ret[key] = point
            # Featurize ret.values(), each of which is a SubplanGoalCost.
            for point in ret.values():
                all_query_vecs.append(self.query_featurizer(point.goal))
                all_costs.append(point.cost)
                all_subtrees.append(point.subplan)
            if verbose:
                print(
                    '{} subplan_to_best,'.format(
                        self.nodes[i].info['query_name']), len(ret), 'entries')
                pprint.pprint(list(ret.values()))

        assert len(all_feat_vecs) == 0 and len(all_pos_vecs) == 0
        all_feat_vecs, all_pos_vecs = TreeConvFeaturize(self.featurizer,
                                                        all_subtrees)

        # Logging.
        print('num_total_subtrees={} num_unique_subtrees={}'.format(
            num_total_subtrees, len(all_query_vecs)))
        print('head')
        for i in range(3):
            print('  query={} feat={} cost={}'.format(all_query_vecs[i].sum(),
                                                      all_feat_vecs[i].sum(),
                                                      all_costs[i]))
        print('tail')
        for i in range(3):
            j = -1 - i
            print('  query={} feat={} cost={}'.format(all_query_vecs[j].sum(),
                                                      all_feat_vecs[j].sum(),
                                                      all_costs[j]))
        return all_query_vecs, all_feat_vecs, all_pos_vecs, all_costs

    def featurize(self,
                  rewrite_generic=False,
                  verbose=False,
                  skip_first_n=0,
                  deduplicate=False,
                  physical_execution_hindsight=False,
                  on_policy=False,
                  use_last_n_iters=-1,
                  use_new_data_only=False,
                  skip_training_on_timeouts=False):
        if physical_execution_hindsight:
            assert deduplicate
            return self._featurize_hindsight_relabeling(
                rewrite_generic,
                verbose,
                skip_first_n,
                use_last_n_iters=use_last_n_iters)
        if deduplicate:
            return self._featurize_dedup(
                rewrite_generic,
                verbose,
                skip_first_n,
                on_policy=on_policy,
                use_last_n_iters=use_last_n_iters,
                use_new_data_only=use_new_data_only,
                skip_training_on_timeouts=skip_training_on_timeouts)

        if use_last_n_iters == -1:
            # No dedup; use all previous iters.
            # This can still be supported by _featurize_dedup() if we pass
            # use_last_n_iters=<num_iters>.
            assert skip_first_n in [0, self.initial_size], (skip_first_n,
                                                            self.initial_size)
            num_iters = (len(self.nodes) - skip_first_n) // self.initial_size
            return self._featurize_dedup(
                rewrite_generic,
                verbose,
                skip_first_n,
                on_policy=on_policy,
                use_last_n_iters=num_iters,  # Note the change.
                use_new_data_only=use_new_data_only,
                skip_training_on_timeouts=skip_training_on_timeouts)

        # TODO: is there any important conf using the code path below?
        # Consider removing.

        assert use_last_n_iters == 1, 'Not implemented yet.'

        assert not on_policy, 'Not implemented for this case yet.'

        self.prepare(rewrite_generic, verbose)

        # Training data.
        all_query_vecs = []
        all_feat_vecs = []  # plan features
        all_pos_vecs = []  # plan positions/indexes
        all_costs = []

        # For each query, extract training data.
        def FeaturizeAllSubtrees(inp_node, all_subtrees, subplan_to_best_cost):
            """Accumulate subtree info."""
            subtrees = plans_lib.GetAllSubtreesNoLeaves(inp_node)
            for t in subtrees:
                t_key = t.hint_str(with_physical_hints=not rewrite_generic)
                curr_cost = subplan_to_best_cost.get(t_key, 1e30)
                subplan_to_best_cost[t_key] = min(curr_cost, inp_node.cost)
                if not self.tree_conv:
                    all_feat_vecs.append(self.featurizer(t))
                    all_pos_vecs.append(self.pos_featurizer(t))
                    assert len(all_feat_vecs[-1]) == len(all_pos_vecs[-1])

            all_subtrees.extend(subtrees)

        subtrees = []

        # This is assumed by the index/looping logic below.
        assert use_last_n_iters == -1, 'Not implemented yet.'

        for i in range(self.initial_size):
            query_all_subtrees = []
            # Hint str -> cost
            # TODO: rewrite this to use self.ComputeBestLatencies().
            subplan_to_best_cost = {}
            i_start = i + skip_first_n
            for j, node in enumerate(self.nodes[i_start::self.initial_size]):
                prev = len(query_all_subtrees)
                FeaturizeAllSubtrees(node, query_all_subtrees,
                                     subplan_to_best_cost)
                curr = len(query_all_subtrees)
                num_subtrees = curr - prev

                # All sub-plans share the same episodic goal -- the root query.
                query_feat = self.query_featurizer(node)
                all_query_vecs.extend([query_feat] * num_subtrees)

            for subtree in query_all_subtrees:
                t_key = subtree.hint_str(
                    with_physical_hints=not rewrite_generic)
                all_costs.append(subplan_to_best_cost[t_key])

            subtrees.extend(query_all_subtrees)

            if verbose:
                print(
                    '{} subplan_to_best_cost,'.format(
                        self.nodes[i].info['query_name']),
                    len(subplan_to_best_cost), 'entries')
                pprint.pprint(subplan_to_best_cost)

        if self.tree_conv:
            assert len(all_feat_vecs) == 0 and len(all_pos_vecs) == 0
            all_feat_vecs, all_pos_vecs = TreeConvFeaturize(
                self.featurizer, subtrees)

        return all_query_vecs, all_feat_vecs, all_pos_vecs, all_costs

    def add(self, node):
        self.nodes.append(node)

    def DropAgentExperience(self):
        old_len = len(self.nodes)
        self.nodes = self.nodes[:self.initial_size]
        new_len = len(self.nodes)
        print('Dropped agent experience (prev len {}, new len {})'.format(
            old_len, new_len))


class SimpleReplayBuffer(Experience):
    """A simple replay buffer.

    It featurizes each element in 'self.nodes' independently, without
    performing subtree extraction, deduplication, take-the-minimum-cost, or
    assuming relationships among query nodes.

    Usage:

       nodes = [ <list of subplans from a single query, dedup'd> ]
       buffer = SimpleReplayBuffer(nodes)

       nodes = [ <N dedup'd subplans from Q1>; <M from Q2> ]
       buffer = SimpleReplayBuffer(nodes)

       # Simply featurizes each element independently.
       data = buffer.featurize()
    """

    def featurize(self, rewrite_generic=False, verbose=False):
        self.featurize_with_subplans(self.nodes, rewrite_generic, verbose)

    def featurize_with_subplans(self,
                                subplans,
                                rewrite_generic=False,
                                verbose=False):
        t1 = time.time()
        assert len(subplans) == len(self.nodes), (len(subplans),
                                                  len(self.nodes))
        self.prepare(rewrite_generic, verbose)
        all_query_vecs = [None] * len(self.nodes)
        all_feat_vecs = [None] * len(self.nodes)
        all_pa_pos_vecs = [None] * len(self.nodes)
        all_costs = [None] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            all_query_vecs[i] = self.query_featurizer(node)
            all_costs[i] = node.cost

        print('Spent {:.1f}s'.format(time.time() - t1))
        if isinstance(self.featurizer, plans_lib.TreeNodeFeaturizer):
            all_feat_vecs, all_pa_pos_vecs = TreeConvFeaturize(
                self.featurizer, subplans)
        else:
            for i, node in enumerate(self.nodes):
                all_feat_vecs[i] = self.featurizer(subplans[i])

        # Debug print: check if query vectors are different/same.
        for i in range(min(len(self.nodes), 10)):
            print('query={} plan={} cost={}'.format(
                (all_query_vecs[i] *
                 np.arange(1, 1 + len(all_query_vecs[i]))).sum(),
                all_feat_vecs[i], all_costs[i]))

        return all_query_vecs, all_feat_vecs, all_pa_pos_vecs, all_costs


class SubplanGoalCost(
        collections.namedtuple(
            'SubplanGoalCost',
            ['subplan', 'goal', 'cost'],
        )):
    """A collected training data point; wrapper around (subplan, goal, cost).

    Attributes:

      subplan: a balsa.Node.
      goal: a balsa.Node.
      cost: the cost of 'goal'.  Specifically: start from subplan, eventually
        reaching 'goal' (joining all leaf nodes & with all filters taken into
        account), what's the cost of the terminal plan?
    """

    def ToSubplanGoalHint(self, with_physical_hints=False):
        """subplan's hint_str()--optionally with physical ops--and the goal."""
        return 'subplan=\'{}\', goal=\'{}\''.format(
            self.subplan.hint_str(with_physical_hints),
            ','.join(sorted(self.goal.leaf_ids(alias_only=True))))

    def __repr__(self):
        """Basic string representation for quick inspection."""
        return 'SubplanGoalCost(subplan=\'{}\', goal=\'{}\', cost={})'.format(
            self.subplan.hint_str(),
            ','.join(sorted(self.goal.leaf_ids(alias_only=True))), self.cost)
