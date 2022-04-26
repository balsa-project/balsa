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
"""Balsa simulation agent."""
import collections
import copy
import hashlib
import os
import pickle
import time

from absl import app
from absl import logging
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn.functional as F

import balsa
from balsa import costing
from balsa import envs
from balsa import experience
from balsa import hyperparams
from balsa import models
from balsa import optimizer as balsa_opt
from balsa import search
from balsa.util import dataset as ds
from balsa.util import plans_lib
from balsa.util import postgres
import train_utils


class SimModel(pl.LightningModule):

    def __init__(self,
                 use_tree_conv,
                 query_feat_dims,
                 plan_feat_dims,
                 mlp_hiddens,
                 tree_conv_version=None,
                 loss_type=None,
                 torch_invert_cost=None,
                 query_featurizer=None,
                 perturb_query_features=False):
        super().__init__()
        assert loss_type in [None, 'mean_qerror'], loss_type
        self.save_hyperparameters()
        self.use_tree_conv = use_tree_conv
        if use_tree_conv:
            self.tree_conv = models.treeconv.TreeConvolution(
                feature_size=query_feat_dims,
                plan_size=plan_feat_dims,
                label_size=1,
                version=tree_conv_version)
        else:
            self.mlp = balsa.models.MakeMlp(input_size=query_feat_dims +
                                            plan_feat_dims,
                                            num_outputs=1,
                                            hiddens=mlp_hiddens,
                                            activation='relu')
        self.loss_type = loss_type
        self.torch_invert_cost = torch_invert_cost
        self.query_featurizer = query_featurizer
        self.perturb_query_features = perturb_query_features

    def forward(self, query_feat, plan_feat, indexes=None):
        if self.use_tree_conv:
            return self.tree_conv(query_feat, plan_feat, indexes)
        return self.mlp(torch.cat([query_feat, plan_feat], -1))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._ComputeLoss(batch)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        val_loss = self._ComputeLoss(batch)
        result = pl.EvalResult(checkpoint_on=val_loss, early_stop_on=val_loss)
        result.log('val_loss', val_loss, prog_bar=True)
        return result

    def _ComputeLoss(self, batch):
        query_feat, plan_feat, *rest = batch
        target = rest[-1]
        if self.training and self.perturb_query_features is not None:
            # No-op for non-enabled featurizers.
            query_feat = self.query_featurizer.PerturbQueryFeatures(
                query_feat, distribution=self.perturb_query_features)
        if self.use_tree_conv:
            assert len(rest) == 2
            output = self.forward(query_feat, plan_feat, rest[0])
        else:
            assert len(rest) == 1
            output = self.forward(query_feat, plan_feat)
        if self.loss_type == 'mean_qerror':
            output_inverted = self.torch_invert_cost(output.reshape(-1,))
            target_inverted = self.torch_invert_cost(target.reshape(-1,))
            return train_utils.QErrorLoss(output_inverted, target_inverted)
        return F.mse_loss(output.reshape(-1,), target.reshape(-1,))

    def on_after_backward(self):
        if self.global_step % 50 == 0:
            norm_dict = self.grad_norm(norm_type=2)
            total_norm = norm_dict['grad_2.0_norm_total']
            self.logger.log_metrics({'total_grad_norm': total_norm},
                                    step=self.global_step)


class SimQueryFeaturizer(plans_lib.Featurizer):
    """Implements the query featurizer.

        Query node -> [ multi-hot of what tables are present ]
                    * [ each-table's selectivities ]
    """

    def __init__(self, workload_info):
        self.workload_info = workload_info

    def __call__(self, node):
        vec = np.zeros(self.dims, dtype=np.float32)

        # Joined tables: [table: 1].
        joined = node.leaf_ids()
        for rel_id in joined:
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx] = 1.0

        # Filtered tables.
        table_id_to_name = lambda table_id: table_id.split(' ')[0]  # Hack.

        for rel_id, est_rows in node.info['all_filters_est_rows'].items():
            if rel_id not in joined:
                # Due to the way we copy Nodes and populate this info field,
                # leaf_ids() might be a subset of info['all_filters_est_rows'].
                continue

            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            total_rows = self.workload_info.table_num_rows[table_id_to_name(
                rel_id)]

            # NOTE: without ANALYZE, for some reason this predicate is
            # estimated to have 703 rows, whereas the table only has 4 rows:
            #   (kind IS NOT NULL) AND ((kind)::text <> 'production
            #   companies'::text)
            # With ANALYZE run, this assert passes.
            assert est_rows >= 0 and est_rows <= total_rows, (node.info,
                                                              est_rows,
                                                              total_rows)
            vec[idx] = est_rows / total_rows
        return vec

    def PerturbQueryFeatures(self, query_feat, distribution):
        """Randomly perturbs a query feature vec returned by __call__()."""
        selectivities = query_feat
        # Table: for each chance of each joined table being perturbed:
        #     % of original query features kept
        #     mean # tables scaled
        #
        #   0.5: ~3% original; mean # tables scaled 3.6
        #   0.3: ~10.5% original; mean # tables scaled 2.1
        #   0.25: ~13.9-16.6% original; mean # tables scaled 1.8-1.9
        #   0.2: ~23.6% original; mean # tables scaled 1.5
        #
        # % kept original:
        #   ((multipliers > 1).sum(1) == 0).sum().float() / len(multipliers)
        # Mean # tables scaled:
        #   (multipliers > 1).sum(1).float().mean()
        #
        # "Default": chance = 0.25, unif = [0.5, 2].
        chance, unif = distribution

        should_scale = torch.rand(selectivities.shape,
                                  device=selectivities.device) < chance
        # The non-zero entries are joined tables.
        should_scale *= (selectivities > 0)
        # Sample multipliers ~ Unif[l, r].
        multipliers = torch.rand(
            selectivities.shape,
            device=selectivities.device) * (unif[1] - unif[0]) + unif[0]
        multipliers *= should_scale
        # Now, the 0 entries mean "should not scale", which needs to be
        # translated into using a multiplier of 1.
        multipliers[multipliers == 0] = 1
        # Perturb.
        new_selectivities = torch.clamp(selectivities * multipliers, max=1)
        return new_selectivities

    @property
    def dims(self):
        return len(self.workload_info.rel_ids)


class SimQueryFeaturizerV2(SimQueryFeaturizer):
    """Concat SimQueryFeaturizer's output with indicators of filtered columns.

    Query feature vec
    = [each table: selectivity (0 if non-joined)]
      concat [bools of filtered cols].
    """

    def __call__(self, node):
        parent_vec = super().__call__(node)
        num_tables = len(self.workload_info.rel_ids)
        filtered_attrs = node.GetFilteredAttributes()
        for attr in filtered_attrs:
            idx = np.where(self.workload_info.all_attributes == attr)[0][0]
            parent_vec[num_tables + idx] = 1.0
        return parent_vec

    @property
    def dims(self):
        return len(self.workload_info.rel_ids) + len(
            self.workload_info.all_attributes)


class SimQueryFeaturizerV3(SimQueryFeaturizer):
    """[table->bool] concat [filtered col->selectivity]."""

    def __call__(self, node):
        vec = np.zeros(self.dims, dtype=np.float32)
        # Joined tables: [table: 1].
        joined = node.leaf_ids()
        for rel_id in joined:
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx] = 1.0
        num_tables = len(self.workload_info.rel_ids)

        # Filtered cols.
        rel_id_to_est_rows = node.info['all_filters_est_rows']
        leaves = node.GetLeaves()
        for leaf in leaves:
            leaf_filters = leaf.GetFilters()
            if not leaf_filters:
                continue
            # PG's parser groups all pushed-down filters by table.
            assert len(leaf_filters) == 1, leaf_filters
            leaf_filter = leaf_filters[0]

            # Get the overall selectivity of this expr.
            table_id = leaf.get_table_id()
            expr_est_rows = rel_id_to_est_rows[table_id]
            table_name = leaf.get_table_id(with_alias=False)
            total_rows = self.workload_info.table_num_rows[table_name]
            assert expr_est_rows >= 0 and expr_est_rows <= total_rows, (
                node.info, expr_est_rows, total_rows)
            table_expr_selectivity = expr_est_rows / total_rows

            # Assign this selectivity to all filtered columns in this expr.
            # Note that the expr may contain multiple cols & OR, in which case
            # we make a simplification to assign the same sel. to all cols.
            filtered_attrs = leaf.GetFilteredAttributes()
            for attr in filtered_attrs:
                idx = np.where(self.workload_info.all_attributes == attr)[0][0]
                vec[num_tables + idx] = table_expr_selectivity
        return vec

    @property
    def dims(self):
        return len(self.workload_info.rel_ids) + len(
            self.workload_info.all_attributes)


class SimQueryFeaturizerV4(plans_lib.Featurizer):
    """Raw estimated rows per table -> log(1+x) -> min_max scaling."""

    def __init__(self, workload_info):
        self.workload_info = workload_info
        self._min = None
        self._max = None
        self._range = None
        self._min_torch = None
        self._max_torch = None
        self._range_torch = None

    def __call__(self, node):
        vec = self._FeaturizePreScaling(node)
        return (vec - self._min) / self._range

    def PerturbQueryFeatures(self, query_feat, distribution):
        """Randomly perturbs a query feature vec returned by __call__()."""
        _min = self._min_torch.to(query_feat.device)
        _max = self._max_torch.to(query_feat.device)
        _range = self._range_torch.to(query_feat.device)
        pre_scaling = query_feat * _range + _min
        est_rows = torch.exp(pre_scaling) - 1.0
        # Chance of each joined table being perturbed.
        #   0.5: ~3% original; mean # tables scaled 3.6
        #   0.25: ~16.6% original; mean # tables scaled 1.8
        #   0.3: ~10.5% original; mean # tables scaled 2.1
        #
        # % kept original:
        #   ((multipliers > 1).sum(1) == 0).sum().float() / len(multipliers)
        # Mean # tables scaled:
        #   (multipliers > 1).sum(1).float().mean()
        #
        # "Default": chance = 0.25, unif = [0.5, 2].
        chance, unif = distribution
        should_scale = torch.rand(est_rows.shape,
                                  device=est_rows.device) < chance
        # The non-zero entries are joined tables.
        should_scale *= (est_rows > 0)
        # Sample multipliers ~ Unif[l, r].
        multipliers = torch.rand(est_rows.shape, device=est_rows.device) * (
            unif[1] - unif[0]) + unif[0]
        multipliers *= should_scale
        # Now, the 0 entries mean "should not scale", which needs to be
        # translated into using a multiplier of 1.
        multipliers[multipliers == 0] = 1
        # Perturb.
        new_est_rows = est_rows * multipliers
        # Re-perform transforms.
        logged = torch.log(1.0 + new_est_rows)
        logged_clamped = torch.min(logged, _max)
        new_query_feat_transformed = (logged_clamped - _min) / _range
        return new_query_feat_transformed

    def _FeaturizePreScaling(self, node):
        vec = np.zeros(self.dims, dtype=np.float32)
        table_id_to_name = lambda table_id: table_id.split(' ')[0]  # Hack.
        joined = node.leaf_ids()
        # Joined tables: [table: rows of table].
        for rel_id in joined:
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            total_rows = self.workload_info.table_num_rows[table_id_to_name(
                rel_id)]
            vec[idx] = total_rows
        # Filtered tables: [table: estimated rows of table].
        for rel_id, est_rows in node.info['all_filters_est_rows'].items():
            if rel_id not in joined:
                # Due to the way we copy Nodes and populate this info field,
                # leaf_ids() might be a subset of info['all_filters_est_rows'].
                continue
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            total_rows = self.workload_info.table_num_rows[table_id_to_name(
                rel_id)]
            assert est_rows >= 0 and est_rows <= total_rows, (node.info,
                                                              est_rows,
                                                              total_rows)
            vec[idx] = est_rows
        # log1p.
        return np.log(1.0 + vec)

    def Fit(self, nodes):
        assert self._min is None and self._max is None, (self._min, self._max)
        pre_scaling = np.asarray(
            [self._FeaturizePreScaling(node) for node in nodes])
        self._min = np.min(pre_scaling, 0)
        self._max = np.max(pre_scaling, 0)
        self._range = self._max - self._min
        # For PerturbQueryFeatures().
        self._min_torch = torch.from_numpy(self._min)
        self._max_torch = torch.from_numpy(self._max)
        self._range_torch = torch.from_numpy(self._range)
        logging.info('log(1+est_rows): min {}\nmax {}'.format(
            self._min, self._max))

    @property
    def dims(self):
        return len(self.workload_info.rel_ids)


class SimPlanFeaturizer(plans_lib.Featurizer):
    """Implements the plan featurizer.

        plan node -> [ multi-hot of tables on LHS ] [ same for RHS ]
    """

    def __init__(self, workload_info):
        self.workload_info = workload_info

    def __call__(self, node):
        vec = np.zeros(self.dims, dtype=np.float32)

        # Tables on LHS.
        for rel_id in node.children[0].leaf_ids():
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx] = 1.0

        # Tables on RHS.
        for rel_id in node.children[1].leaf_ids():
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx + len(self.workload_info.rel_ids)] = 1.0

        return vec

    @property
    def dims(self):
        return len(self.workload_info.rel_ids) * 2


class Sim(object):
    """Balsa simulation."""

    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        # Train.
        p.Define('epochs', 100, 'Maximum training epochs.  '\
                 'Early-stopping may kick in.')
        p.Define('gradient_clip_val', 0, 'Clip the gradient norm computed over'\
                 ' all model parameters together. 0 means no clipping.')
        p.Define('bs', 2048, 'Batch size.')
        # Validation.
        p.Define('validate_fraction', 0.1,
                 'Sample this fraction of the dataset as the validation set.  '\
                 '0 to disable validation.')
        # Search, train-time.
        p.Define('search', search.DynamicProgramming.Params(),
                 'Params of the enumeration routine to use for training data.')
        # Search space.
        p.Define('plan_physical', False,
                 'Learn and plan physical scans/joins, or just join orders?')
        # Infer, test-time.
        p.Define('infer_search_method', 'beam_bk', 'Options: beam_bk.')
        p.Define('infer_beam_size', 10, 'Beam size.')
        p.Define('infer_search_until_n_complete_plans', 1,
                 'Search until how many complete plans?')
        # Workload.
        p.Define('workload', envs.JoinOrderBenchmark.Params(),
                 'Params of the Workload, i.e., a set of queries.')
        # Data collection.
        p.Define('skip_data_collection_geq_num_rels', None,
                 'If specified, do not collect data for queries with at '\
                 'least this many number of relations.')
        p.Define(
            'generic_ops_only_for_min_card_cost', False,
            'If using MinCardCost, whether to enumerate generic ops only.')
        p.Define('sim_data_collection_intermediate_goals', True,
                 'For each query, also collect sim data with intermediate '\
                 'query goals?')
        # Featurizations.
        p.Define('plan_featurizer_cls', SimPlanFeaturizer,
                 'Featurizer to use for plans.')
        p.Define('query_featurizer_cls', SimQueryFeaturizer,
                 'Featurizer to use for queries.')
        p.Define('label_transforms', ['log1p', 'standardize'],
                 'Transforms for labels.')
        p.Define('perturb_query_features', None, 'See experiments.')
        # Eval.
        p.Define('eval_output_path', 'eval-cost.csv',
                 'Path to write evaluation output into.')
        p.Define('eval_latency_output_path', 'eval-latency.csv',
                 'Path to write evaluation latency output into.')
        # Model/loss.
        p.Define('tree_conv_version', None, 'Options: None, V2.')
        p.Define('loss_type', None, 'Options: None (MSE), mean_qerror.')
        return p

    @classmethod
    def HashOfSimData(cls, p):
        """Gets the hash that should determine the simulation data."""
        # Use (a few attributes inside Params, Postgres configs) as hash key.
        # Using PG configs is necessary because things like PG version / PG
        # optimizer settings affect collected costs.
        # NOTE: in theory, other stateful effects such as whether ANALYZE has
        # been called on a PG database also affects the collected costs.
        _RELEVANT_HPARAMS = [
            'search',
            'workload',
            'skip_data_collection_geq_num_rels',
            'generic_ops_only_for_min_card_cost',
            'plan_physical',
        ]
        param_vals = [p.Get(hparam) for hparam in _RELEVANT_HPARAMS]
        param_vals = [
            v.ToText() if isinstance(v, hyperparams.Params) else str(v)
            for v in param_vals
        ]
        spec = '\n'.join(param_vals)
        if p.search.cost_model.cls is costing.PostgresCost:
            # Only PostgresCost would depend on PG configs.
            pg_configs = map(str, postgres.GetServerConfigs())
            spec += '\n'.join(pg_configs)
        hash_sim = hashlib.sha1(spec.encode()).hexdigest()[:8]
        return hash_sim

    @classmethod
    def HashOfFeaturizedData(cls, p):
        """Gets the hash that should determine the final featurized tensors."""
        # Hash(HashOfSimData(), featurization specs).
        # NOTE: featurized data involves asking Postgres for cardinality
        # estimates of filters.  So in theory, here the hash calculation should
        # depend on postgres.GetServerConfigs().  Most relevant are the PG
        # version & whether ANALYZE has been run (this is not tracked by any PG
        # config).  Here let's make an assumption that all PG versions with
        # ANALYZE ran produce the same estimates, which is reasonable because
        # they are just histograms.
        hash_sim = cls.HashOfSimData(p)
        _FEATURIZATION_HPARAMS = [
            'plan_featurizer_cls',
            'query_featurizer_cls',
        ]
        param_vals = [str(p.Get(hparam)) for hparam in _FEATURIZATION_HPARAMS]
        spec = str(hash_sim) + '\n'.join(param_vals)
        hash_feat = hashlib.sha1(spec.encode()).hexdigest()[:8]
        return hash_feat

    def __init__(self, params):
        self.params = params.Copy()
        p = self.params
        # Plumb through same flags.
        p.search.plan_physical_ops = p.plan_physical
        p.search.cost_model.cost_physical_ops = p.plan_physical
        logging.info(p)

        # Instantiate search.
        self.search = p.search.cls(p.search)

        # Instantiate workload.
        self.workload = p.workload.cls(p.workload)
        wi = self.workload.workload_info
        generic_join = np.array(['Join'])
        generic_scan = np.array(['Scan'])
        if not p.plan_physical:
            # These are used in optimizer.py (for planning).
            wi.join_types = generic_join
            wi.scan_types = generic_scan
        else:
            self.search.SetPhysicalOps(join_ops=wi.join_types,
                                       scan_ops=wi.scan_types)
        if self.IsPlanPhysicalButUseGenericOps():
            self.search.SetPhysicalOps(join_ops=generic_join,
                                       scan_ops=generic_scan)

        # A list of SubplanGoalCost.
        self.simulation_data = []

        self.planner = None
        self.query_featurizer = None

        self.all_nodes = self.workload.Queries(split='all')
        self.train_nodes = self.workload.Queries(split='train')
        self.test_nodes = self.workload.Queries(split='test')
        logging.info('{} train queries: {}'.format(
            len(self.train_nodes),
            [node.info['query_name'] for node in self.train_nodes]))
        logging.info('{} test queries: {}'.format(
            len(self.test_nodes),
            [node.info['query_name'] for node in self.test_nodes]))

        plans_lib.RewriteAsGenericJoinsScans(self.all_nodes)

        # This call ensures that node.info['all_filters_est_rows'] is written,
        # which is used by the query featurizer.
        experience.SimpleReplayBuffer(self.all_nodes)

    def IsPlanPhysicalButUseGenericOps(self):
        p = self.params
        # This is a logical-only cost model.  Let's only enumerate generic ops.
        return (p.plan_physical and p.generic_ops_only_for_min_card_cost and
                isinstance(self.search.cost_model, costing.MinCardCost))

    def _MakeOnEnumeratedHook(self, accum, info_to_attach, num_rels):
        """Records all possible training points from a single trajectory."""
        p = self.params

        def Hook(plan, cost):
            if (not p.sim_data_collection_intermediate_goals and
                    len(plan.GetLeaves()) < num_rels):
                # Ablation: don't collect data on any plans/costs that have
                # fewer than 'num_rels' (the original query) tables.
                return
            query_node = plan.Copy()
            # NOTE: must make a copy as info can get new fields.
            query_node.info = dict(info_to_attach)
            query_node.cost = cost

            def _Helper(node):
                if node.IsJoin():
                    accum.append(
                        experience.SubplanGoalCost(
                            subplan=node,
                            goal=query_node,
                            cost=cost,
                        ))

            plans_lib.MapNode(query_node, _Helper)

        return Hook

    def _DedupDataPoints(self, points):
        """Deduplicates 'points' (assumed to be from the same query).

        For each unique (goal,subplan), keep the single datapoint with the best
        cost.  We need to check for smaller costs due to our data collection.

        Example:

            Enumerated plan: ((mc cn) t), say cost 100.
            Among all data points yielded from this plan, we will have:

                goal = {mc, cn, t}
                subplan = (mc cn)
                cost = 100

            However, the search procedure may enumerate another plan for the
            same goal, say (t (mc cn)) with cost 200.  Among all data points
            yielded from this plan, we will have:

                goal = {mc, cn, t}
                subplan = (mc cn)
                cost = 200

        So, we really want to keep the first, i.e., record only the cheapest
        for each unique (goal,subplan).
        """
        p = self.params
        best_cost = collections.defaultdict(lambda: np.inf)
        ret = {}
        for point in points:
            # NOTE: when this function turns the 'goal' part into a string,
            # some information is not preserved (e.g., the string doesn't
            # record filter info).  However, since we assume 'points' all come
            # from the same query, this simplification is OK for uniquifying.
            key = point.ToSubplanGoalHint(with_physical_hints=p.plan_physical)
            if point.cost < best_cost[key]:
                best_cost[key] = point.cost
                ret[key] = point
        logging.info('{} points before uniquifying, {} after'.format(
            len(points), len(ret)))
        return ret.values()

    def _SimulationDataPath(self):
        p = self.params
        hash_key = Sim.HashOfSimData(p)
        return 'data/sim-data-{}.pkl'.format(hash_key)

    def _LoadSimulationData(self):
        path = self._SimulationDataPath()
        try:
            with open(path, 'rb') as f:
                self.simulation_data = pickle.load(f)
        except Exception as e:
            return False
        logging.info('Loaded simulation data (len {}) from: {}'.format(
            len(self.simulation_data), path))
        logging.info('Training data (first 50, total {}):'.format(
            len(self.simulation_data)))
        logging.info('\n'.join(map(str, self.simulation_data[:50])))
        return True

    def _SaveSimulationData(self):
        path = self._SimulationDataPath()
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.simulation_data, f)
            logging.info('Saved simulation data (len {}) to: {}'.format(
                len(self.simulation_data), path))
        except Exception as e:
            logging.warning('Failed saving sim data:\n{}'.format(e))

    def _FeaturizedDataPath(self):
        p = self.params
        hash_key = Sim.HashOfFeaturizedData(p)
        return 'data/sim-featurized-{}.pkl'.format(hash_key)

    def _LoadFeaturizedData(self):
        path = self._FeaturizedDataPath()
        try:
            with open(path, 'rb') as f:
                data = torch.load(f)
        except Exception as e:
            return False, None
        logging.info('Loaded featurized data (len {}) from: {}'.format(
            len(data[0]), path))
        return True, data

    def _SaveFeaturizedData(self, data):
        path = self._FeaturizedDataPath()
        try:
            with open(path, 'wb') as f:
                torch.save(data, f)
            logging.info('Saved featurized data (len {}) to: {}'.format(
                len(data[0]), path))
        except Exception as e:
            logging.warning('Failed saving featurized data:\n{}'.format(e))

    def CollectSimulationData(self, try_load=True):
        p = self.params
        if try_load:
            done = self._LoadSimulationData()
            if done:
                return

        start = time.time()
        num_collected = 0
        for query_node in self.train_nodes:
            # TODO: can parallelize this loop.  Take care of the hooks.
            num_rels = len(query_node.leaf_ids())
            logging.info('query={} num_rels={}\n{}'.format(
                query_node.info['query_name'], num_rels,
                query_node.info['sql_str']))
            if p.skip_data_collection_geq_num_rels is not None:
                if num_rels >= p.skip_data_collection_geq_num_rels:
                    continue
            num_collected += 1

            # Accumulate data points from this query.
            accum = []
            info_to_attach = {
                'overall_join_graph': query_node.info['parsed_join_graph'],
                'overall_join_conds': query_node.info['parsed_join_conds'],
                'path': query_node.info['path'],
            }
            self.search.PushOnEnumeratedHook(
                self._MakeOnEnumeratedHook(accum, info_to_attach, num_rels))

            # Enumerate plans.
            self.search.Run(query_node, query_node.info['sql_str'])

            self.search.PopOnEnumeratedHook()

            # Dedup accumulated data points.
            accum = self._DedupDataPoints(accum)

            self.simulation_data.extend(accum)

        simulation_time = time.time() - start

        logging.info('Collection done, stats:')
        logging.info('  num_queries={} num_collected_queries={} num_points={}'\
                     ' latency_s={:.1f}'.format(
            len(self.train_nodes), num_collected, len(self.simulation_data),
            simulation_time))

        if try_load:
            self._SaveSimulationData()

        return simulation_time, len(self.simulation_data)

    def _MakeModel(self, query_feat_dims, plan_feat_dims):
        p = self.params
        use_tree_conv = issubclass(p.plan_featurizer_cls,
                                   plans_lib.TreeNodeFeaturizer)
        logging.info('SIM query_feat_dims={} plan_feat_dims={}'.format(
            query_feat_dims, plan_feat_dims))
        logging.info('SIM query_feat={} plan_feat={}'.format(
            p.query_featurizer_cls, p.plan_featurizer_cls))
        return SimModel(
            use_tree_conv=use_tree_conv,
            query_feat_dims=query_feat_dims,
            plan_feat_dims=plan_feat_dims,
            # [128, 64] => 0.1 MB
            # [256] * 4 => 0.8 MB
            # [512] * 4 => 3.1 MB
            mlp_hiddens=[512] * 3,  # 2.1 MB
            tree_conv_version=p.tree_conv_version,
            loss_type=p.loss_type,
            torch_invert_cost=self.train_dataset.dataset.TorchInvertCost,
            query_featurizer=self.query_featurizer,
            perturb_query_features=p.perturb_query_features)

    def _MakeDatasetAndLoader(self, data):
        p = self.params
        all_query_vecs = data[0]
        all_feat_vecs = data[1]
        all_costs = data[3]
        # 'use_positions' is True iff we want to use a TreeConv to process the
        # subplans.  If using a non-tree-aware featurization, it becomes
        # unused.
        use_positions = data[2][0] is not None
        all_pa_pos_vecs = data[2]
        dataset = ds.PlansDataset(
            all_query_vecs,
            all_feat_vecs,
            all_pa_pos_vecs,
            all_costs,
            transform_cost=p.label_transforms,
            cross_entropy=False,
            return_indexes=use_positions,
            tree_conv=use_positions,
        )
        assert 0 <= p.validate_fraction <= 1, p.validate_fraction
        num_train = int(len(dataset) * (1 - p.validate_fraction))
        num_validation = len(dataset) - num_train
        assert num_train > 0 and num_validation >= 0, len(dataset)
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [num_train, num_validation])
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=p.bs,
            shuffle=True,
            pin_memory=True,
        )
        if num_validation > 0:
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1024)
        else:
            val_loader = None
        logging.info('num_train={} num_validation={}'.format(
            num_train, num_validation))
        return train_ds, train_loader, val_ds, val_loader

    def _LogPostgresConfigs(self, wandb_logger):
        """Logs live Postgres server configs to a file and uploads to W&B."""
        wandb_run = wandb_logger.experiment
        df = postgres.GetServerConfigsAsDf()
        path = os.path.join(wandb_run.dir, 'postgres-conf.txt')
        df.to_csv(path, index=False, header=True)

    def _FeaturizeTrainingData(self, try_load=True):
        """Pre-processes/featurizes simulation data into tensors."""
        p = self.params
        wi = self.workload.workload_info.Copy()
        if not p.plan_physical:
            # This original WorkloadInfo has all physical ops, but during
            # learning the training data would only have all sort of scans plus
            # only logical Join nodes.  So we rewrite the vocab here.
            wi.all_ops = np.asarray(['Join', 'Scan'])
        elif self.IsPlanPhysicalButUseGenericOps():
            wi.all_ops = np.sort(np.concatenate((wi.all_ops, ['Join', 'Scan'])))
            wi.join_types = np.sort(np.concatenate((wi.join_types, ['Join'])))
            wi.scan_types = np.sort(np.concatenate((wi.scan_types, ['Scan'])))
        wi.table_num_rows = postgres.GetAllTableNumRows(wi.rel_names)
        self.training_workload_info = wi

        # Instantiate query featurizer once with train nodes, since it may need
        # to calculate normalization statistics.
        self.query_featurizer = p.query_featurizer_cls(wi)
        self.query_featurizer.Fit(self.workload.Queries(split='train'))

        if try_load:
            done, data = self._LoadFeaturizedData()
            if done:
                return data

        if not self.simulation_data:
            self.CollectSimulationData(try_load)

        logging.info('Creating SimpleReplayBuffer')
        # The constructor of SRB realy only needs goal/query Nodes for
        # instantiating workload info metadata and featurizers (e.g., grab all
        # table names).
        goals = [p.goal for p in self.simulation_data]
        exp = experience.SimpleReplayBuffer(
            goals,
            workload_info=wi,  # Pass this in to significantly speed up ctor.
            plan_featurizer_cls=p.plan_featurizer_cls,
            query_featurizer_cls=self.query_featurizer,
            # Saves expensive filtering; simulation data is already {Join,
            # <Scan types>} only.
            keep_scans_joins_only=False,
        )
        logging.info('featurize_with_subplans()')
        data = exp.featurize_with_subplans(
            subplans=[p.subplan for p in self.simulation_data],
            rewrite_generic=not p.plan_physical)

        if try_load:
            self._SaveFeaturizedData(data)
        return data

    def _MakeTrainer(self, loggers=None):
        p = self.params
        is_inner_params = True
        if loggers is None:
            is_inner_params = False
            loggers = [
                pl_loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                             version=None,
                                             name='lightning_logs'),
                pl_loggers.WandbLogger(save_dir=os.getcwd()),
            ]
        p_dict = balsa.utils.SanitizeToText(dict(p))
        if is_inner_params:
            # If 'loggers' is passed, some outer experiment hparams has
            # specified a 'cls' field, so let's not let our cls: <class
            # 'sim.Sim'> overwrite that.
            p_dict.pop('cls', None)
        for logger in loggers:
            logger.log_hyperparams(p_dict)
        assert isinstance(loggers[-1], pl_loggers.WandbLogger), loggers[-1]
        self._LogPostgresConfigs(wandb_logger=loggers[-1])
        return pl.Trainer(
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=p.epochs,
            # Add logging metrics per this many batches.
            row_log_interval=10,
            # Do validation per this many train epochs.
            check_val_every_n_epoch=1,
            # Patience = # of validations with no improvements before stopping.
            early_stop_callback=pl.callbacks.EarlyStopping(patience=5,
                                                           mode='min',
                                                           verbose=True),
            weights_summary='full',
            logger=loggers,
            gradient_clip_val=p.gradient_clip_val,
            num_sanity_val_steps=2 if p.validate_fraction > 0 else 0,
        )

    def _GetPlanner(self):
        p = self.params
        if self.planner is None:
            wi = self.training_workload_info
            wi_for_ops_to_enum = wi
            if self.IsPlanPhysicalButUseGenericOps():
                wi_for_ops_to_enum = copy.deepcopy(wi)
                # We want to make sure the Optimizer enumerates just physical
                # ops (no generic ops), but still being able to use
                # query_featurizer/plan_featurizer that knows about both
                # physical+generic ops.
                wi_for_ops_to_enum.all_ops = np.asarray([
                    op for op in wi_for_ops_to_enum.all_ops
                    if op not in ['Join', 'Scan']
                ])
                wi_for_ops_to_enum.join_types = np.asarray([
                    op for op in wi_for_ops_to_enum.join_types if op != 'Join'
                ])
                wi_for_ops_to_enum.scan_types = np.asarray([
                    op for op in wi_for_ops_to_enum.scan_types if op != 'Scan'
                ])
            self.planner = balsa_opt.Optimizer(
                wi_for_ops_to_enum,
                p.plan_featurizer_cls(wi),
                None,  # parent_pos_featurizer
                self.query_featurizer.WithWorkloadInfo(wi),
                self.train_dataset.dataset.InvertCost,
                self.model,
                tree_conv=self.model.use_tree_conv,
                beam_size=p.infer_beam_size,
                search_until_n_complete_plans=p.
                infer_search_until_n_complete_plans,
                plan_physical=p.plan_physical)
        else:
            # Otherwise, 'self.model' might have been updated since the planner
            # is created.  Update it.
            self.planner.SetModel(self.model)
        return self.planner

    def Train(self, train_data=None, load_from_checkpoint=None, loggers=None):
        p = self.params
        # Pre-process and featurize data.
        data = train_data
        if data is None:
            data = self._FeaturizeTrainingData()

        # Make the DataLoader.
        logging.info('_MakeDatasetAndLoader()')
        self.train_dataset, self.train_loader, _, self.val_loader = \
            self._MakeDatasetAndLoader(data)
        batch = next(iter(self.train_loader))
        logging.info(
            'Example batch (query,plan,indexes,cost):\n{}'.format(batch))

        # Initialize model.
        _, query_feat_dims = batch[0].shape
        if issubclass(p.plan_featurizer_cls, plans_lib.TreeNodeFeaturizer):
            # make_and_featurize_trees() tranposes the latter 2 dims.
            unused_bs, plan_feat_dims, unused_max_tree_nodes = batch[1].shape
            logging.info(
                'unused_bs, plan_feat_dims, unused_max_tree_nodes {}'.format(
                    (unused_bs, plan_feat_dims, unused_max_tree_nodes)))
        else:
            unused_bs, plan_feat_dims = batch[1].shape
        self.model = self._MakeModel(query_feat_dims=query_feat_dims,
                                     plan_feat_dims=plan_feat_dims)
        balsa.models.ReportModel(self.model)

        # Train or load.
        self.trainer = self._MakeTrainer(loggers=loggers)
        if load_from_checkpoint:
            self.model = SimModel.load_from_checkpoint(load_from_checkpoint)
            logging.info(
                'Loaded pretrained checkpoint: {}'.format(load_from_checkpoint))
        else:
            self.trainer.fit(self.model, self.train_loader, self.val_loader)
        return data

    def FreeData(self):
        self.simulation_data = None
        self.train_loader = None
        self.val_loader = None
        self.train_dataset.dataset.FreeData()

    def Infer(self, node, planner_config=None):
        """Runs query planning on 'node'.

        Returns:
          plan: the found plan.
          cost: the learned, inverse-transformed cost of the found plan.
        """
        p = self.params
        planner = self._GetPlanner()
        bushy = True
        if planner_config is not None:
            bushy = planner_config.search_space == 'bushy'
        time, plan, cost = planner.plan(node,
                                        search_method=p.infer_search_method,
                                        bushy=bushy,
                                        planner_config=planner_config)
        return plan, cost

    def Predict(self, query_node, nodes):
        """Runs forward pass on 'nodes' to predict their costs."""
        return self._GetPlanner().infer(query_node, nodes)

    def _LoadBestCheckpointForEval(self):
        """Loads the checkpoint with the best validation loss."""
        train_utils.LoadBestCheckpointForEval(self.model, self.trainer)

    def EvaluateCost(self, planner_config=None, split='all'):
        """Reports cost sub-optimalities w.r.t. Postgres."""
        p = self.params
        metrics = []
        qnames = []
        num_rels = []

        self._LoadBestCheckpointForEval()

        if not isinstance(self.search.cost_model, costing.PostgresCost):
            return

        for query_node in self.workload.Queries(split=split):
            qnames.append(query_node.info['query_name'])
            num_rels.append(len(query_node.leaf_ids()))

            # Call FilterScanOrJoins so that things like Aggregate/Hash are
            # removed and don't add parentheses in the hint string.
            pg_plan_str = plans_lib.FilterScansOrJoins(query_node).hint_str()
            logging.info('query={} num_rels={}'.format(qnames[-1],
                                                       num_rels[-1]))
            logging.info('postgres_plan={} postgres_cost={}'.format(
                pg_plan_str, query_node.cost))

            found_plan, predicted_cost = self.Infer(
                query_node, planner_config=planner_config)

            if isinstance(self.search.cost_model, costing.PostgresCost):
                # Score via PG.
                # Due to quirkiness in PG (e.g., different FROM orderings produce
                # different costs, even though the join orders are exactly the
                # same), we use the original SQL query here with 'found_plan''s
                # hint_str().  Doing this makes sure suboptimality is at best 1.0x
                # for non-GEQO plans.
                actual_cost = self.search.cost_model.ScoreWithSql(
                    found_plan, query_node.info['sql_str'])
                found_plan_str = found_plan.hint_str()

                suboptimality = actual_cost / query_node.cost

                # Sanity checks.
                try:
                    if suboptimality > 1:
                        assert pg_plan_str != found_plan_str, (pg_plan_str,
                                                               found_plan_str)
                    elif suboptimality < 0.99:
                        # Check that we can only do better than PG plan under the
                        # cost model if GEQO is enabled.  Otherwise PG uses
                        # exhaustive DP so should be optimal.
                        #
                        # We use 0.99 because e.g., q13a, even the query plans are
                        # the same w/ and w/o hinting, a top Gather node can have
                        # slightly different costs, on the order of ~0.2).
                        GEQO_THRESHOLD = 12
                        assert num_rels[-1] >= GEQO_THRESHOLD, num_rels[-1]
                except Exception as e:
                    print(e)
            else:
                actual_cost = predicted_cost
                suboptimality = 1

            # Logging.
            metrics.append(suboptimality)
            logging.info('  predicted_cost={:.1f}'.format(predicted_cost))
            logging.info('  actual_cost={:.1f}'.format(actual_cost))
            logging.info('  suboptimality={:.1f}'.format(suboptimality))

        df = pd.DataFrame({
            'query': qnames,
            'num_rel': num_rels,
            'suboptimality': metrics
        })
        df.to_csv(p.eval_output_path)
        self.trainer.logger.log_metrics({
            'subopt_mean': np.mean(metrics),
            'subopt_max': np.asarray(metrics).max()
        })
        logging.info('suboptimalities:\n{}'.format(
            df['suboptimality'].describe()))

    def EvaluateLatency(self, planner_config=None, split='all'):
        p = self.params
        metrics = []
        qnames = []
        num_rels = []

        for query_node in self.workload.Queries(split=split):
            qnames.append(query_node.info['query_name'])
            num_rels.append(len(query_node.leaf_ids()))

            # Call FilterScanOrJoins so that things like Aggregate/Hash are
            # removed and don't add parentheses in the hint string.
            pg_plan_str = plans_lib.FilterScansOrJoins(query_node).hint_str()
            logging.info('query={} num_rels={}'.format(qnames[-1],
                                                       num_rels[-1]))
            logging.info('postgres_plan={} postgres_cost={}'.format(
                pg_plan_str, query_node.cost))

            found_plan, predicted_cost = self.Infer(
                query_node, planner_config=planner_config)
            actual_cost = postgres.GetLatencyFromPg(
                query_node.info['sql_str'],
                found_plan.hint_str(p.plan_physical))

            # Logging.
            metrics.append(actual_cost)
            logging.info('  predicted_cost={:.1f}'.format(predicted_cost))
            logging.info('  actual_latency_ms={:.1f}'.format(actual_cost))

        df = pd.DataFrame({
            'query': qnames,
            'num_rel': num_rels,
            'latency': metrics,
        })
        df.to_csv(p.eval_latency_output_path)
        self.trainer.logger.log_metrics({
            'latency_sum_s': np.sum(metrics) / 1e3,
        })
        logging.info('Latencies:\n{}'.format(df['latency'].describe()))


def MakeTestQuery():
    a_join_b = plans_lib.Node('Join')
    a_join_b.children = [
        plans_lib.Node('Scan', table_name='A').with_alias('a'),
        plans_lib.Node('Scan', table_name='B').with_alias('b'),
    ]
    query_node = plans_lib.Node('Join')
    query_node.children = [
        a_join_b,
        plans_lib.Node('Scan', table_name='C').with_alias('c')
    ]
    query_node.info['query_name'] = 'test'
    query_node.info[
        'sql_str'] = 'SELECT * FROM a, b, c WHERE a.id = b.id AND a.id = c.id;'

    # Simple check: Copy() deep-copies the children nodes.
    copied = query_node.Copy()
    assert id(copied) != id(query_node)
    for i in range(2):
        assert id(copied.children[i]) != id(query_node.children[i])

    return query_node


def Main(argv):
    """For testing. run.py automatically instantiates a simulation agent."""
    del argv  # Unused.

    p = Sim.Params()

    p.workload.query_glob = '*.sql'
    p.workload.query_glob = None
    p.generic_ops_only_for_min_card_cost = True

    p.skip_data_collection_geq_num_rels = 12
    p.search.cost_model = costing.PostgresCost.Params()
    p.search.cost_model = costing.MinCardCost.Params()

    p.plan_featurizer_cls = plans_lib.TreeNodeFeaturizer

    # Infer.
    p.infer_beam_size = 20
    p.infer_search_until_n_complete_plans = 10

    p.plan_physical = True
    if p.plan_physical:
        # Use a plan featurizer that can process physical ops.
        p.plan_featurizer_cls = plans_lib.PhysicalTreeNodeFeaturizer

    # Pre-training via simulation data.
    sim = Sim(p)
    sim.CollectSimulationData()
    # Use None to retrain; pass a ckpt to reload.
    sim_ckpt = None
    train_data = None
    for i in range(5):
        train_data = sim.Train(train_data, load_from_checkpoint=sim_ckpt)
        sim.params.eval_output_path = 'eval-cost-{}.csv'.format(i)
        sim.params.eval_latency_output_path = 'eval-latency-{}.csv'.format(i)
        sim.EvaluateCost()
        sim.EvaluateLatency()


if __name__ == '__main__':
    app.run(Main)
