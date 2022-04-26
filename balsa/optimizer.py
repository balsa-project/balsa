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
import time

import numpy as np
import torch

from balsa import search
from balsa.models import treeconv
from balsa.util import dataset as ds
from balsa.util import plans_lib

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class PlannerConfig(
        collections.namedtuple(
            'PlannerConfig',
            [
                'search_space', 'enable_nestloop', 'enable_hashjoin',
                'enable_mergejoin'
            ],
        )):
    """Experimental: a simple tuple recording what ops can be planned."""

    @classmethod
    def Get(cls, name):
        return {
            'NestLoopHashJoin': PlannerConfig.NestLoopHashJoin(),
            'LeftDeepNestLoop': PlannerConfig.LeftDeepNestLoop(),
            'LeftDeepNestLoopHashJoin':
                PlannerConfig.LeftDeepNestLoopHashJoin(),
            'LeftDeep': PlannerConfig.LeftDeep(),
            'Dbmsx': PlannerConfig.Dbmsx(),
        }[name]

    @classmethod
    def Default(cls):
        return cls(search_space='bushy',
                   enable_nestloop=True,
                   enable_hashjoin=True,
                   enable_mergejoin=True)

    @classmethod
    def NestLoopHashJoin(cls):
        return cls(search_space='bushy',
                   enable_nestloop=True,
                   enable_hashjoin=True,
                   enable_mergejoin=False)

    @classmethod
    def LeftDeepNestLoop(cls):
        return cls(search_space='leftdeep',
                   enable_nestloop=True,
                   enable_hashjoin=False,
                   enable_mergejoin=False)

    @classmethod
    def LeftDeepNestLoopHashJoin(cls):
        return cls(search_space='leftdeep',
                   enable_nestloop=True,
                   enable_hashjoin=True,
                   enable_mergejoin=False)

    @classmethod
    def LeftDeep(cls):
        return cls(search_space='leftdeep',
                   enable_nestloop=True,
                   enable_hashjoin=True,
                   enable_mergejoin=True)

    @classmethod
    def Dbmsx(cls):
        return cls(search_space='dbmsx',
                   enable_nestloop=True,
                   enable_hashjoin=True,
                   enable_mergejoin=True)

    def KeepEnabledJoinOps(self, join_ops):
        ops = []
        for op in join_ops:
            if op == 'Nested Loop' and self.enable_nestloop:
                ops.append(op)
            elif op == 'Hash Join' and self.enable_hashjoin:
                ops.append(op)
            elif op == 'Merge Join' and self.enable_mergejoin:
                ops.append(op)
        assert len(ops) > 0, (self, join_ops)
        return ops


class Optimizer(object):
    """Creates query execution plans using learned model."""

    def __init__(
        self,
        workload_info,
        plan_featurizer,
        parent_pos_featurizer,
        query_featurizer,
        inverse_label_transform_fn,
        model,
        tree_conv=False,
        beam_size=10,
        search_until_n_complete_plans=1,
        plan_physical=False,
        use_label_cache=True,
        use_plan_restrictions=True,
    ):
        self.workload_info = workload_info
        self.plan_featurizer = plan_featurizer
        self.parent_pos_featurizer = parent_pos_featurizer
        self.query_featurizer = query_featurizer
        self.inverse_label_transform_fn = inverse_label_transform_fn
        self.use_label_cache = use_label_cache
        self.use_plan_restrictions = use_plan_restrictions

        # Plan search params
        if not plan_physical:
            jts = workload_info.join_types
            assert np.array_equal(jts, ['Join']), jts
            sts = workload_info.scan_types
            assert np.array_equal(sts, ['Scan']), sts
        self.plan_physical = plan_physical
        self.beam_size = beam_size
        self.search_until_n_complete_plans = search_until_n_complete_plans
        self.tree_conv = tree_conv
        self.SetModel(model)

        # Debugging.
        self.total_joins = 0
        self.total_random_triggers = 0
        self.num_queries_with_random = 0

    def SetModel(self, model):
        self.value_network = model.to(DEVICE)
        # Set the model in eval mode.  Only affects modules (e.g., Dropout)
        # that have different behaviors in train vs. in test.  Do it once here
        # so that Optimizer.plan() below doesn't have to do it on every query.
        self.value_network.eval()
        # Reset the cache due to model being changed.
        self.label_cache = {}

    # @profile
    def infer(self, query_node, plan_nodes, set_model_eval=False):
        """Forward pass.

        Args:
            query_node: a plans_lib.Node object. Represents the query context.
            plan_nodes: a list of plans_lib.Node objects. The set of plans to
              score.

        Returns:
            costs, a float. Higher costs indicate more expensive plans.
        """
        labels = [None] * len(plan_nodes)
        plans, idx = [], []
        if self.use_label_cache:
            # Gather cached labels
            lookup_keys = [(query_node.info['query_name'],
                            plan.to_str(with_cost=False))
                           for plan in plan_nodes]
            for i, lookup_key in enumerate(lookup_keys):
                label = self.label_cache.get(lookup_key)
                if label is not None:
                    labels[i] = label
                else:
                    plans.append(plan_nodes[i])
                    idx.append(i)
            # No plans to score.
            if len(plans) == 0:
                return labels
        else:
            plans = plan_nodes

        # Perform inference on new plans.
        if set_model_eval:
            # Expensive.  Caller should try to call only once.
            self.value_network.eval()
        with torch.no_grad():
            query_enc = self.query_featurizer(query_node)
            all_query_vecs = [query_enc] * len(plans)
            all_plans = []
            all_indexes = []
            if self.tree_conv:
                all_plans, all_indexes = treeconv.make_and_featurize_trees(
                    plans, self.plan_featurizer)
            else:
                for plan_node in plans:
                    all_plans.append(self.plan_featurizer(plan_node))

                if self.parent_pos_featurizer is not None:
                    for plan_node in plans:
                        all_indexes.append(
                            self.parent_pos_featurizer(plan_node))

            if self.tree_conv or hasattr(self.plan_featurizer, 'pad'):
                query_feat = torch.from_numpy(np.asarray(all_query_vecs)).to(
                    DEVICE, non_blocking=True)
                plan_feat = torch.from_numpy(np.asarray(all_plans)).to(
                    DEVICE, non_blocking=True)
                pos_feat = torch.from_numpy(np.asarray(all_indexes)).to(
                    DEVICE, non_blocking=True)
                cost = self.value_network(query_feat, plan_feat,
                                          pos_feat).cpu().numpy()
            else:
                all_costs = [1] * len(all_plans)
                batch = ds.PlansDataset(
                    all_query_vecs,
                    all_plans,
                    all_indexes,
                    all_costs,
                    transform_cost=False,
                    return_indexes=False,
                )
                loader = torch.utils.data.DataLoader(batch,
                                                     batch_size=len(all_plans),
                                                     shuffle=False)
                processed_batch = list(loader)[0]
                query_feat, plan_feat = processed_batch[0].to(
                    DEVICE), processed_batch[1].to(DEVICE)
                cost = self.value_network(query_feat, plan_feat).cpu().numpy()

            cost = self.inverse_label_transform_fn(cost)
            plan_labels = cost.reshape(-1,).tolist()

            if self.use_label_cache:
                # Update the cache with the labels.
                for i in range(len(plan_labels)):
                    labels[idx[i]] = plan_labels[i]
                    self.label_cache[lookup_keys[idx[i]]] = plan_labels[i]
            else:
                labels = plan_labels
            return labels

    def plan(self, query_node, search_method, **kwargs):
        if search_method == 'beam_bk':
            return self._beam_search_bk(query_node,
                                        beam_size=self.beam_size,
                                        **kwargs)
        raise ValueError(f'Unsupported search_method: {search_method}')

    def _get_possible_plans(self,
                            query_node,
                            state,
                            join_graph,
                            bushy=False,
                            planner_config=None,
                            avoid_eq_filters=False):
        """Expands a state.  Returns a list of successor states."""
        if not bushy:
            if planner_config.search_space == 'leftdeep':
                func = self._get_possible_plans_left_deep
            else:
                func = self._get_possible_plans_dbmsx
        else:
            func = self._get_possible_plans_bushy
        return func(query_node,
                    state,
                    join_graph,
                    planner_config=planner_config,
                    avoid_eq_filters=avoid_eq_filters)

    # @profile
    def _get_possible_plans_bushy(self,
                                  query_node,
                                  state,
                                  join_graph,
                                  planner_config=None,
                                  avoid_eq_filters=False):
        possible_joins = []
        num_rels = len(state)
        for i in range(num_rels):
            for j in range(num_rels):
                if i == j:
                    continue
                l = state[i]
                r = state[j]
                # Hinting a join between non-neighbors may fail (PG may
                # disregard the hint).
                if not plans_lib.ExistsJoinEdgeInGraph(l, r, join_graph):
                    continue
                for plan in self._enumerate_plan_operators(
                        l,
                        r,
                        planner_config=planner_config,
                        avoid_eq_filters=avoid_eq_filters):
                    possible_joins.append((plan, i, j))
        return possible_joins

    def _get_possible_plans_dbmsx(self,
                                  query_node,
                                  state,
                                  join_graph,
                                  planner_config=None,
                                  avoid_eq_filters=False):
        raise NotImplementedError

    def _get_possible_plans_left_deep(self,
                                      query_node,
                                      state,
                                      join_graph,
                                      planner_config=None,
                                      avoid_eq_filters=False):
        possible_joins = []
        num_rels = len(state)
        join_index = None
        for i, s in enumerate(state):
            if s.IsJoin():
                assert join_index is None, 'two joins found'
                join_index = i
        if join_index is None:
            # Base state: all unspecified scans.
            scored = set()
            for i in range(num_rels):
                for j in range(num_rels):
                    if i == j:
                        continue
                    if (i, j) in scored:
                        continue
                    scored.add((i, j))
                    l = state[i]
                    r = state[j]
                    # Hinting a join between non-neighbors may fail (PG may
                    # disregard the hint).
                    if not plans_lib.ExistsJoinEdgeInGraph(l, r, join_graph):
                        continue
                    for plan in self._enumerate_plan_operators(
                            l,
                            r,
                            planner_config=planner_config,
                            avoid_eq_filters=avoid_eq_filters):
                        possible_joins.append((plan, i, j))
        else:
            i, l = join_index, state[join_index]
            for j in range(0, len(state)):
                if j == i:
                    continue
                r = state[j]
                # Hinting a join between non-neighbors may fail (PG may
                # disregard the hint).
                if not plans_lib.ExistsJoinEdgeInGraph(l, r, join_graph):
                    continue
                for plan in self._enumerate_plan_operators(
                        l,
                        r,
                        planner_config=planner_config,
                        avoid_eq_filters=avoid_eq_filters):
                    possible_joins.append((plan, i, j))
        return possible_joins

    def _enumerate_plan_operators(self,
                                  left,
                                  right,
                                  planner_config=None,
                                  avoid_eq_filters=False):
        join_ops = self.workload_info.join_types
        scan_ops = self.workload_info.scan_types
        if planner_config:
            join_ops = planner_config.KeepEnabledJoinOps(join_ops)
        # Hack.
        if planner_config and planner_config.search_space == 'dbmsx':
            engine = 'dbmsx'
        else:
            engine = 'postgres'
        return search.EnumerateJoinWithOps(
            left,
            right,
            join_ops=join_ops,
            scan_ops=scan_ops,
            avoid_eq_filters=avoid_eq_filters,
            engine=engine,
            use_plan_restrictions=self.use_plan_restrictions)

    # @profile
    def _make_new_states(self, state, costs, possible_joins):
        num_rels = len(state)
        valid_costs = [None] * len(possible_joins)
        valid_new_states = [None] * len(possible_joins)
        for i in range(len(possible_joins)):
            join, left_idx, right_idx = possible_joins[i]
            join.cost = costs[i]
            new_state = state[:]  # Shallow copy.
            new_state[left_idx] = join
            del new_state[right_idx]
            new_state_cost = -1e30
            for rel in new_state:
                if rel.IsJoin():
                    # Goodness(state) = max V_theta(subplan), for all subplan
                    # in state.
                    new_state_cost = max(new_state_cost, rel.cost)
            valid_costs[i] = new_state_cost
            valid_new_states[i] = new_state
        return valid_costs, valid_new_states

    # @profile
    def _beam_search_bk(self,
                        query_node,
                        beam_size=10,
                        bushy=False,
                        return_all_found=False,
                        planner_config=None,
                        verbose=False,
                        avoid_eq_filters=False,
                        epsilon_greedy=0):
        """Produce a plan via beam search.

        Args:
          query_node: a Node, a parsed version of the query to optimize.  In
            principle we should take the raw SQL string, but this is a
            convenient proxy.
          beam_size: size of the fixed set of most promising Nodes to be
            explored.
        """
        if planner_config:
            if bushy:
                assert planner_config.search_space == 'bushy', planner_config
            else:
                assert planner_config.search_space != 'bushy', planner_config
        planning_start_t = time.time()
        # Join graph.
        join_graph, _ = query_node.GetOrParseSql()
        # Base tables to join.
        query_leaves = query_node.GetLeaves()
        # A "state" is a list of Nodes, each representing a partial plan. If a
        # state has only one element, then it is a complete plan.
        init_state = query_leaves
        # A fringe is a priority queue of (cost of a state, a state).
        fringe = [(0, init_state)]

        # Bookkeeping of open (unexpanded) and closed (expanded) states.
        # Reference: page 2 of
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.435.447&rep=rep1&type=pdf
        # TODO: can factor out these logic into a Fringe / a FringeState class.
        # TODO: Unify 'states_open' and 'fringe'.
        states_open = {}  # StateHash(state) -> cost.
        states_expanded = {}  # StateHash(state) -> cost.

        def StateHash(state):
            """Orderless hashing."""
            return hash(
                frozenset([
                    # to_str() is faster than hint_str(); this can be further
                    # optimized by using shorter strings.
                    subplan.to_str(with_cost=False) for subplan in state
                ]))

        def MarkInOpen(state_cost, state, state_hash):
            states_open[state_hash] = state_cost

        def RemoveFromOpen(state_cost, state):
            h = StateHash(state)
            prev_cost = states_open.pop(h)
            assert prev_cost == state_cost, (prev_cost, state_cost, state,
                                             states_open)

        def MoveFromOpenToExpanded(state_cost, state):
            h = StateHash(state)
            prev_cost = states_open.pop(h)
            assert prev_cost == state_cost, (prev_cost, state_cost, state,
                                             states_open)
            states_expanded[h] = state_cost

        def GetFromOpenOrExpanded(state):
            h = StateHash(state)
            ret = states_open.get(h)
            if ret is not None:
                return ret, h
            return states_expanded.get(h), h

        MarkInOpen(0, init_state, StateHash(init_state))

        is_eps_greedy_triggered = False

        terminal_states = []
        while len(terminal_states) < self.search_until_n_complete_plans and \
              fringe:
            state_cost, state = fringe.pop(0)
            MoveFromOpenToExpanded(state_cost, state)
            if len(state) == 1:
                # A terminal.
                terminal_states.append((state_cost, state))
                continue

            possible_plans = self._get_possible_plans(
                query_node,
                state,
                join_graph,
                bushy=bushy,
                planner_config=planner_config,
                avoid_eq_filters=avoid_eq_filters)
            costs = self.infer(query_node,
                               [join for join, _, _ in possible_plans])
            valid_costs, valid_new_states = self._make_new_states(
                state, costs, possible_plans)

            for i, (valid_cost,
                    new_state) in enumerate(zip(valid_costs, valid_new_states)):
                # Add to open if it is not in open or expanded.
                ret, state_hash = GetFromOpenOrExpanded(new_state)
                if ret is None:
                    fringe.append((valid_cost, new_state))
                    MarkInOpen(valid_cost, new_state, state_hash)
                else:
                    prev_cost = ret
                    assert valid_cost == prev_cost, (valid_cost, prev_cost,
                                                     new_state, states_open,
                                                     states_expanded)
            r = np.random.rand()
            if r < epsilon_greedy:
                # Randomly pick one state in the fringe and discard the rest.
                # Note that 'fringe' at this step can have larger than
                # 'beam_size' elements.
                rand_idx = np.random.randint(len(fringe))
                new_fringe = [fringe[rand_idx]]
                # Remove the discarded states from 'open' so that they may be
                # able to be explored down the line.
                for i, fringe_elem in enumerate(fringe):
                    if i == rand_idx:
                        continue
                    _state_cost, _state = fringe_elem
                    RemoveFromOpen(_state_cost, _state)
                # Swap.
                fringe = new_fringe

                # Debugging.
                self.total_random_triggers += 1
                is_eps_greedy_triggered = True

            fringe = sorted(fringe, key=lambda x: x[0])
            fringe = fringe[:beam_size]

        planning_time = (time.time() - planning_start_t) * 1e3
        print('Planning took {:.1f}ms'.format(planning_time))

        # Print terminal_states.
        if verbose:
            print('terminal_states:')
        all_found = []
        min_cost = np.min([c for c, s in terminal_states])
        min_cost_idx = np.argmin([c for c, s in terminal_states])
        for i, (cost, state) in enumerate(terminal_states):
            all_found.append((cost, state[0]))
            if verbose:
                if cost == min_cost:
                    print('  {:.1f} {}  <-- cheapest'.format(
                        cost,
                        str([s.hint_str(self.plan_physical) for s in state])))
                else:
                    print('  {:.1f} {}'.format(
                        cost,
                        str([s.hint_str(self.plan_physical) for s in state])))
        ret = [
            planning_time, terminal_states[min_cost_idx][1][0],
            terminal_states[min_cost_idx][0]
        ]
        if return_all_found:
            ret.append(all_found)

        self.total_joins += len(query_leaves) - 1
        self.num_queries_with_random += int(is_eps_greedy_triggered)

        return ret

    def SampleRandomPlan(self, query_node, bushy=True):
        """Samples a random, valid plan."""
        planning_start_t = time.time()
        join_graph, _ = query_node.GetOrParseSql()
        query_leaves = query_node.CopyLeaves()
        num_rels = len(query_leaves)
        num_random_plans = 100
        num_random_plans = 1000

        def _SampleOne(state):
            while len(state) > 1:
                possible_plans = self._get_possible_plans(query_node,
                                                          state,
                                                          join_graph,
                                                          bushy=bushy)
                _, valid_new_states = self._make_new_states(
                    state, [0.0] * len(possible_plans), possible_plans)
                rand_idx = np.random.randint(len(valid_new_states))
                state = valid_new_states[rand_idx]
            predicted = self.infer(query_node, [state[0]])
            return predicted, state

        best_predicted = [np.inf]
        best_state = None
        for _ in range(num_random_plans):
            state = query_leaves
            assert len(state) == num_rels, len(state)
            predicted, state = _SampleOne(state)
            if predicted[0] < best_predicted[0]:
                best_predicted = predicted
                best_state = state

        planning_time = (time.time() - planning_start_t) * 1e3
        predicted = best_predicted
        state = best_state
        print('Found best random plan out of {}:'.format(num_random_plans))
        print('  {:.1f} {}'.format(
            predicted[0], str([s.hint_str(self.plan_physical) for s in state])))
        print('Planning took {:.1f}ms'.format(planning_time))
        return predicted[0], state[0]
