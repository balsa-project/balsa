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
"""Balsa.

Usage:

    # Experiment configs are declared in experiments.py.
    # Look up the name and pass --run <name>.
    python -u run.py --run <name> 2>&1 | tee run.log

Use Main() to modify hparams for debugging.
"""
import collections
import copy
import logging
import os
import pickle
import pprint
import signal
import time

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import psycopg2
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import ray
import ray.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import wandb

import balsa
from balsa import costing
from balsa import envs
from balsa import execution
from balsa import plan_analysis
from balsa.experience import Experience
from balsa.models.transformer import ReportModel
from balsa.models.transformer import Transformer
from balsa.models.transformer import TransformerV2
from balsa.models.treeconv import TreeConvolution
import balsa.optimizer as optim
from balsa.util import dataset as ds
from balsa.util import plans_lib
from balsa.util import postgres

import sim as sim_lib
import pg_executor
from pg_executor import dbmsx_executor
import train_utils
import experiments  # noqa # pylint: disable=unused-import

FLAGS = flags.FLAGS
flags.DEFINE_string('run', 'Balsa_JOBRandSplit', 'Experiment config to run.')
flags.DEFINE_boolean('local', False,
                     'Whether to use local engine for query execution.')


def GetDevice():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def Save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path


def SaveText(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        f.write(text)
        f.write('\n')
    os.replace(path + '.tmp', path)
    return path


def MakeModel(p, exp, dataset):
    dev = GetDevice()
    num_label_bins = int(
        dataset.costs.max().item()) + 2  # +1 for 0, +1 for ceil(max cost).
    query_feat_size = len(exp.query_featurizer(exp.nodes[0]))
    batch = exp.featurizer(exp.nodes[0])
    assert batch.ndim == 1
    plan_feat_size = batch.shape[0]

    if p.tree_conv:
        labels = num_label_bins if p.cross_entropy else 1
        return TreeConvolution(feature_size=query_feat_size,
                               plan_size=plan_feat_size,
                               label_size=labels,
                               version=p.tree_conv_version).to(dev)
    else:
        plan_vocab_size = exp.featurizer.pad() + 1  # +1 for PAD.
        parent_pos_vocab_size = exp.pos_featurizer.pad() + 1
        d_model = 256
        d_ff = 1024
        num_layers = 4
        num_heads = 4
        clazz = TransformerV2 if p.v2 else Transformer
        return clazz(
            plan_vocab_size,
            parent_pos_vocab_size,
            d_model,
            num_heads,
            d_ff,
            num_layers,
            d_query_feat=query_feat_size,
            plan_pad_idx=exp.featurizer.pad(),
            parent_pos_pad_idx=exp.pos_featurizer.pad(),
            use_pos_embs=p.pos_embs,
            dropout=p.dropout,
            cross_entropy=p.cross_entropy,
            max_label_bins=num_label_bins,
        ).to(dev)


@ray.remote
def ExecuteSql(query_name,
               sql_str,
               hint_str,
               hinted_plan,
               query_node,
               predicted_latency,
               curr_timeout_ms=None,
               found_plans=None,
               predicted_costs=None,
               silent=False,
               is_test=False,
               use_local_execution=True,
               plan_physical=True,
               repeat=1,
               engine='postgres'):
    """Executes a query.

    Returns:
      If use_local_execution:
        A (pg_executor, dbmsx_executor).Result.
      Else:
        A ray.ObjectRef of the above.
    """
    # Unused args.
    del query_name, hinted_plan, query_node, predicted_latency, found_plans,\
        predicted_costs, silent, is_test, plan_physical

    assert engine in ('postgres', 'dbmsx'), engine
    if engine == 'postgres':
        return postgres.ExplainAnalyzeSql(sql_str,
                                          comment=hint_str,
                                          verbose=False,
                                          geqo_off=True,
                                          timeout_ms=curr_timeout_ms,
                                          remote=not use_local_execution)
    else:
        return DbmsxExecuteSql(sql_str,
                               comment=hint_str,
                               timeout_ms=curr_timeout_ms,
                               remote=not use_local_execution,
                               repeat=repeat)


def AddCommentToSql(sql_str, comment, engine):
    """Adds a comment (hint string) to a SQL string."""
    fns = {
        'postgres': PostgresAddCommentToSql,
        'dbmsx': DbmsxAddCommentToSql,
    }
    return fns[engine](sql_str, comment)


def PostgresAddCommentToSql(sql_str, comment=None):
    """Postgres: <comment> <SELECT ...>."""
    return comment + '\n' + sql_str


def DbmsxAddCommentToSql(sql_str, comment=None):
    raise NotImplementedError


def DbmsxExecuteSql(sql_str,
                    comment=None,
                    timeout_ms=None,
                    remote=True,
                    repeat=1):
    raise NotImplementedError


def DbmsxNodeToHintStr(node, with_physical_hints=False):
    """Converts a plans_lib.Node plan into Dbmsx-compatible hint string."""
    raise NotImplementedError


def HintStr(node, with_physical_hints, engine):
    if engine == 'postgres':
        return node.hint_str(with_physical_hints=with_physical_hints)
    assert engine == 'dbmsx', engine
    return DbmsxNodeToHintStr(node, with_physical_hints=with_physical_hints)


def ParseExecutionResult(result_tup,
                         query_name,
                         sql_str,
                         hint_str,
                         hinted_plan,
                         query_node,
                         predicted_latency,
                         curr_timeout_ms=None,
                         found_plans=None,
                         predicted_costs=None,
                         silent=False,
                         is_test=False,
                         use_local_execution=True,
                         plan_physical=True,
                         repeat=None,
                         engine='postgres'):
    del repeat  # Unused.
    messages = []
    result = result_tup.result
    has_timeout = result_tup.has_timeout
    server_ip = result_tup.server_ip
    if has_timeout:
        assert not result, result
    if engine == 'dbmsx':
        real_cost = -1 if has_timeout else result_tup.latency
    else:
        if has_timeout:
            real_cost = -1
        else:
            json_dict = result[0][0][0]
            real_cost = json_dict['Execution Time']
    if hint_str is not None:
        # Check that the hint has been respected.  No need to check if running
        # baseline.
        do_hint_check = True
        if engine == 'dbmsx':
            raise NotImplementedError
        else:
            if not has_timeout:
                executed_node = postgres.ParsePostgresPlanJson(json_dict)
            else:
                # Timeout has occurred & 'result' is empty.  Fallback to
                # checking against local Postgres.
                print('Timeout occurred; checking the hint against local PG.')
                executed_node, _ = postgres.SqlToPlanNode(sql_str,
                                                          comment=hint_str,
                                                          verbose=False)
            executed_node = plans_lib.FilterScansOrJoins(executed_node)
            executed_hint_str = executed_node.hint_str(
                with_physical_hints=plan_physical)
        if do_hint_check and hint_str != executed_hint_str:
            print('initial\n', hint_str)
            print('after\n', executed_hint_str)
            msg = 'Hint not respected for {}; server_ip={}'.format(
                query_name, server_ip)
            try:
                assert False, msg
            except Exception as e:
                print(e, flush=True)
                import ipdb
                ipdb.set_trace()

    if not silent:
        messages.append('{}Running {}: hinted plan\n{}'.format(
            '[Test set] ' if is_test else '', query_name, hinted_plan))
        messages.append('filters')
        messages.append(pprint.pformat(query_node.info['all_filters']))
        messages.append('')
        messages.append('q{},{:.1f},{}'.format(query_node.info['query_name'],
                                               real_cost, hint_str))
        messages.append(
            '{} Execution time: {:.1f} (predicted {:.1f}) curr_timeout_ms={}'.
            format(query_name, real_cost, predicted_latency, curr_timeout_ms))

    if hint_str is None or silent:
        # Running baseline: don't print debug messages below.
        return result_tup, real_cost, server_ip, '\n'.join(messages)

    messages.append('Expert plan: latency, predicted, hint')
    expert_hint_str = query_node.hint_str()
    expert_hint_str_physical = query_node.hint_str(with_physical_hints=True)
    messages.append('  {:.1f} (predicted {:.1f})  {}'.format(
        query_node.cost, query_node.info['curr_predicted_latency'],
        expert_hint_str))
    if found_plans:
        if predicted_costs is None:
            predicted_costs = [None] * len(found_plans)
        messages.append('SIM-predicted costs, predicted latency, plan: ')
        min_p_latency = np.min([p_latency for p_latency, _ in found_plans])
        for p_cost, found in zip(predicted_costs, found_plans):
            p_latency, found_plan = found
            found_hint_str = found_plan.hint_str()
            found_hint_str_physical = HintStr(found_plan,
                                              with_physical_hints=True,
                                              engine=engine)
            extras = [
                'cheapest' if p_latency == min_p_latency else '',
                '[expert plan]'
                if found_hint_str_physical == expert_hint_str_physical else '',
                '[picked]' if found_hint_str_physical == hint_str else ''
            ]
            extras = ' '.join(filter(lambda s: s, extras)).strip()
            if extras:
                extras = '<-- {}'.format(extras)
            if p_cost:
                messages.append('  {:.1f}  {:.1f}  {}  {}'.format(
                    p_cost, p_latency, found_hint_str, extras))
            else:
                messages.append('          {:.1f}  {}  {}'.format(
                    p_latency, found_hint_str, extras))
    messages.append('-' * 80)
    return result_tup, real_cost, server_ip, '\n'.join(messages)


def _GetQueryFeaturizerClass(p):
    return {
        True: sim_lib.SimQueryFeaturizer,
        False: plans_lib.QueryFeaturizer,
        'SimQueryFeaturizerV2': sim_lib.SimQueryFeaturizerV2,
        'SimQueryFeaturizerV3': sim_lib.SimQueryFeaturizerV3,
        'SimQueryFeaturizerV4': sim_lib.SimQueryFeaturizerV4,
    }[p.sim_query_featurizer]


def TrainSim(p, loggers=None):
    sim_p = sim_lib.Sim.Params()
    # Copy over relevant params.
    sim_p.workload.query_dir = p.query_dir
    sim_p.workload.query_glob = p.query_glob
    sim_p.workload.test_query_glob = p.test_query_glob
    sim_p.workload.search_space_join_ops = p.search_space_join_ops
    sim_p.workload.search_space_scan_ops = p.search_space_scan_ops
    sim_p.skip_data_collection_geq_num_rels = 12
    if p.cost_model == 'mincardcost':
        sim_p.search.cost_model = costing.MinCardCost.Params()
    else:
        sim_p.search.cost_model = costing.PostgresCost.Params()
    sim_p.query_featurizer_cls = _GetQueryFeaturizerClass(p)
    sim_p.plan_featurizer_cls = plans_lib.TreeNodeFeaturizer
    sim_p.infer_search_method = p.search_method
    sim_p.infer_beam_size = p.beam
    sim_p.infer_search_until_n_complete_plans = p.search_until_n_complete_plans
    if p.plan_physical:
        sim_p.plan_physical = True
        # Use a physical-aware plan featurizer.
        sim_p.plan_featurizer_cls = plans_lib.PhysicalTreeNodeFeaturizer
    sim_p.generic_ops_only_for_min_card_cost = \
        p.generic_ops_only_for_min_card_cost
    sim_p.label_transforms = p.label_transforms
    sim_p.tree_conv_version = p.tree_conv_version
    sim_p.loss_type = p.loss_type
    sim_p.gradient_clip_val = p.gradient_clip_val
    sim_p.bs = p.bs
    sim_p.epochs = p.epochs
    sim_p.perturb_query_features = p.perturb_query_features
    sim_p.validate_fraction = p.validate_fraction

    # Instantiate.
    sim = sim_lib.Sim(sim_p)
    if p.sim_checkpoint is None:
        sim.CollectSimulationData()
    sim.Train(load_from_checkpoint=p.sim_checkpoint, loggers=loggers)
    sim.model.freeze()
    sim.EvaluateCost()
    sim.FreeData()
    return sim


def InitializeModel(p,
                    model,
                    sim,
                    soft_assign_tau=0.0,
                    soft_assign_use_ema=False,
                    ema_source_tm1=None):
    """Initializes model weights.

    Given model_(t-1), sim, ..., ema_source_tm1, initializes model_t as follows.

    If soft_assign_use_ema is False:

        model := soft_assign_tau*model + (1-soft_assign_tau)*sim.

        In particular:
        - soft_assign_tau = 0 means always reinitializes 'model' with 'sim'.
        - soft_assign_tau = 1 means don't reinitialize 'model'; keep training it
            across value iterations.

        A value of 0.1 seems to perform well.

    Otherwise, use an exponential moving average of "source networks":

        source_t = soft_assign_tau * source_(t-1)
                     + (1-soft_assign_tau) model_(t-1)
        model_t := source_t

        In particular:
        - soft_assign_tau = 0 means don't reinitialize 'model'; keep training it
            across value iterations.
        - soft_assign_tau = 1 means always reinitializes 'model' with 'sim'.

        A value of 0.05 seems to perform well.

    For both schemes, before training 'model' for the very first time it is
    always initialized with the simulation model 'sim'.

    Args:
      p: params.
      model: current iteration's value model.
      sim: the trained-in-sim model.
      soft_assign_tau: if positive, soft initializes 'model' using the formula
        described above.
      soft_assign_use_ema: whether to use an exponential moving average of
        "source networks".
      ema_source_tm1: the EMA of source networks at iteration t-1.
    """

    def Rename(state_dict):
        new_state_dict = collections.OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('tree_conv.'):
                new_key = key.replace('tree_conv.', '')
            new_state_dict[new_key] = value
        return new_state_dict

    sim_weights = sim.model.state_dict()
    sim_weights_renamed = copy.deepcopy(Rename(sim_weights))
    model_weights = model.state_dict()
    assert model_weights.keys() == sim_weights_renamed.keys()

    tau = soft_assign_tau
    if tau:
        if not soft_assign_use_ema:
            print('Assigning real model := {}*SIM + {}*previous real model'.
                  format(1 - tau, tau))
            for key, param in model_weights.items():
                param.requires_grad = False
                param = param * tau + sim_weights_renamed[key] * (1.0 - tau)
                param.requires_grad = True
        else:
            # Use an exponential moving average of source networks.
            if ema_source_tm1 is None:
                ema_source_tm1 = sim_weights_renamed
            assert isinstance(ema_source_tm1,
                              collections.OrderedDict), ema_source_tm1
            assert ema_source_tm1.keys() == model_weights.keys()
            # Calculates source_t for current iteration t:
            #    source_t = tau * source_(t-1) + (1-tau) model_(t-1)
            with torch.no_grad():
                ema_source_t = copy.deepcopy(ema_source_tm1)
                for key, param in model_weights.items():
                    ema_source_t[key] = tau * ema_source_tm1[key] + (
                        1.0 - tau) * param
            # Assign model_t := source_t.
            model.load_state_dict(ema_source_t)
            print('Initialized from EMA source network: tau={}'.format(tau))
            # Return source_t for next iter's use.
            return ema_source_t
    else:
        model.load_state_dict(sim_weights_renamed)
        print('Initialized from SIM weights.')

    if p.finetune_out_mlp_only:
        for name, param in model.named_parameters():
            if 'out_mlp' not in name:
                param.detach_()
                param.requires_grad = False
                print('Freezing', name)

    if p.param_noise:
        for layer in model.out_mlp:
            if isinstance(layer, nn.Linear):
                print('Adding N(0, {}) to out_mlp\'s {}.'.format(
                    p.param_noise, layer))

                def _Add(w):
                    w.requires_grad = False
                    w.add_(
                        torch.normal(mean=0.0,
                                     std=p.param_noise,
                                     size=w.shape,
                                     device=w.device))
                    w.requires_grad = True

                _Add(layer.weight)


class BalsaModel(pl.LightningModule):
    """Wraps an nn.Module into a pl.LightningModule."""

    def __init__(self,
                 params,
                 model,
                 loss_type=None,
                 torch_invert_cost=None,
                 query_featurizer=None,
                 perturb_query_features=None,
                 l2_lambda=0,
                 learning_rate=None,
                 optimizer_state_dict=None,
                 reduce_lr_within_val_iter=False):
        super().__init__()
        self.logging_prefix = ''
        self.params = params.Copy()
        self.model = model
        assert loss_type in [None, 'mean_qerror'], loss_type
        self.loss_type = loss_type
        self.torch_invert_cost = torch_invert_cost
        self.query_featurizer = query_featurizer
        self.perturb_query_features = perturb_query_features
        self.l2_lambda = l2_lambda
        self.optimizer_state_dict = optimizer_state_dict
        # Assume constant LR within each value iter.  Reasonable for on-pol but
        # probably need tuning for off-pol.
        self.learning_rate = learning_rate
        # Optionally, reduce within each trainer.fit() call (i.e., an iter).
        self.reduce_lr_within_val_iter = reduce_lr_within_val_iter

    def SetLoggingPrefix(self, prefix):
        """Useful for prepending value iteration numbers."""
        self.logging_prefix = prefix

    def forward(self, query_feat, plan_feat, indexes):
        return self.model(query_feat, plan_feat, indexes)

    def configure_optimizers(self):
        p = self.params
        if p.adamw:
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=p.adamw)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
            )
        if self.optimizer_state_dict is not None:
            # Checks the params are the same.
            # 'params': [139581476513104, ...]
            curr = optimizer.state_dict()['param_groups'][0]['params']
            prev = self.optimizer_state_dict['param_groups'][0]['params']
            assert curr == prev, (curr, prev)
            print('Loading last iter\'s optimizer state.')
            # Prev optimizer state's LR may be stale.
            optimizer.load_state_dict(self.optimizer_state_dict)
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            assert optimizer.state_dict(
            )['param_groups'][0]['lr'] == self.learning_rate
            print('LR', self.learning_rate)
        if not self.reduce_lr_within_val_iter:
            return optimizer
        print('returning optimizer + scheduler')
        scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        'min',
                                                        patience=5,
                                                        verbose=True),
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_early_stop_on',  # Bug: cannot use 'val_loss'.
        }
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        self.latest_per_iter_lr = self.trainer.optimizers[0].state_dict(
        )['param_groups'][0]['lr']

    def training_step(self, batch, batch_idx):
        loss, l2_loss = self._ComputeLoss(batch)
        result = pl.TrainResult(minimize=loss)
        # Log both a per-iter metric and an overall metric for comparison.
        result.log('{}loss'.format(self.logging_prefix), loss, prog_bar=False)
        result.log('train_loss', loss, prog_bar=True)
        if self.l2_lambda > 0:
            result.log('l2_loss', l2_loss, prog_bar=False)
        return result

    def validation_step(self, batch, batch_idx):
        val_loss, l2_loss = self._ComputeLoss(batch)
        result = pl.EvalResult(checkpoint_on=val_loss, early_stop_on=val_loss)
        result.log('{}val_loss'.format(self.logging_prefix),
                   val_loss,
                   prog_bar=False)
        result.log('val_loss', val_loss, prog_bar=True)
        if self.l2_lambda > 0:
            result.log('val_l2_loss', l2_loss, prog_bar=False)
        return result

    def _ComputeLoss(self, batch):
        p = self.params
        dev = GetDevice()
        query_feat = batch.query_feats
        if self.training and self.perturb_query_features is not None:
            # No-op for non-enabled featurizers.
            query_feat = self.query_featurizer.PerturbQueryFeatures(
                query_feat, distribution=self.perturb_query_features)
        query_feat, plan_feat, indexes, target = (query_feat.to(dev),
                                                  batch.plans.to(dev),
                                                  batch.indexes.to(dev),
                                                  batch.costs.to(dev))
        output = self.forward(query_feat, plan_feat, indexes)
        if p.cross_entropy:
            log_probs = output.log_softmax(-1)
            target_dist = torch.zeros_like(log_probs)
            # Scalar 46.25 represented as: 0.75 * 46 + 0.25 * 47.
            ceil = torch.ceil(target)
            w_ceil = ceil - target
            floor = torch.floor(target)
            w_floor = 1 - w_ceil
            target_dist.scatter_(1,
                                 ceil.long().unsqueeze(1), w_ceil.unsqueeze(1))
            target_dist.scatter_(1,
                                 floor.long().unsqueeze(1),
                                 w_floor.unsqueeze(1))
            loss = (-target_dist * log_probs).sum(-1).mean()
        else:
            if self.loss_type == 'mean_qerror':
                output_inverted = self.torch_invert_cost(output.reshape(-1,))
                target_inverted = self.torch_invert_cost(target.reshape(-1,))
                loss = train_utils.QErrorLoss(output_inverted, target_inverted)
            else:
                loss = F.mse_loss(output.reshape(-1,), target.reshape(-1,))
        if self.l2_lambda > 0:
            l2_loss = torch.tensor(0., device=loss.device, requires_grad=True)
            for param in self.parameters():
                l2_loss = l2_loss + torch.norm(param).pow(2)
            l2_loss = self.l2_lambda * 0.5 * l2_loss
            loss += l2_loss
            return loss, l2_loss
        return loss, None

    def on_after_backward(self):
        if self.global_step % 10 == 0:
            norm_dict = self.grad_norm(norm_type=2)
            total_grad_norm = norm_dict['grad_2.0_norm_total']
            total_norm = torch.stack([
                torch.norm(param) for param in self.parameters()
            ]).sum().detach()
            self.logger.log_metrics(
                {
                    'total_grad_norm': total_grad_norm,
                    'total_norm': total_norm,
                },
                step=self.global_step)


class BalsaAgent(object):
    """The Balsa agent."""

    def __init__(self, params):
        self.params = params.Copy()
        p = self.params
        print('BalsaAgent params:\n{}'.format(p))

        self.sim = None
        self.ema_source_net = None
        self.timeout_controller = execution.PerQueryTimeoutController(
            timeout_slack=p.timeout_slack,
            no_op=not p.use_timeout,
            relax_timeout_factor=p.relax_timeout_factor,
            relax_timeout_on_n_timeout_iters=p.relax_timeout_on_n_timeout_iters,
            initial_timeout_ms=p.initial_timeout_ms)
        self.query_execution_cache = execution.QueryExecutionCache()
        self.best_plans = execution.QueryExecutionCache()
        self.trainer = None
        self.loggers = None

        # Labels.
        self.label_mean = None
        self.label_std = None
        self.label_running_stats = envs.RunningStats()

        # EMA/SWA.
        #   average name -> dict
        self.moving_average_model_state_dict = collections.defaultdict(dict)
        #   average name -> counter
        self.moving_average_counter_dict = collections.defaultdict(int)

        # LR schedule.
        self.lr_schedule = train_utils.GetLrSchedule(p)

        # Optimizer state.
        self.prev_optimizer_state_dict = None
        # Ray.
        if p.use_local_execution:
            ray.init(resources={'pg': 1})
        else:
            # Cluster access: make sure the cluster has been launched.
            import uuid
            ray.init(address='auto',
                     namespace=f'{uuid.uuid4().hex[:4]}',
                     logging_level=logging.ERROR)
        try:
            print('Connected to ray!  Resources:', ray.available_resources())
        except RuntimeError as e:
            if 'dictionary changed size during iteration' not in str(e):
                raise e
            print('Connected to ray but ray.available_resources() failed, '
                  'likely indicating issues with the cluster.\nTry running '
                  '1 run only and see if tasks go through or get stuck.'
                  '  Exception:\n   {}'.format(e))
        # Workload.
        self.workload = self._MakeWorkload()
        self.all_nodes = self.workload.Queries(split='all')
        self.train_nodes = self.workload.Queries(split='train')
        self.test_nodes = self.workload.Queries(split='test')
        print(len(self.train_nodes), 'train queries:',
              [node.info['query_name'] for node in self.train_nodes])
        print(len(self.test_nodes), 'test queries:',
              [node.info['query_name'] for node in self.test_nodes])
        if p.test_query_glob is None:
            print('Consider all queries as training nodes.')
        # Rewrite ops if physical plan is not used.
        if not p.plan_physical:
            plans_lib.RewriteAsGenericJoinsScans(self.all_nodes)
        # If the target engine has a dialect != Postgres, overwrite
        # node.info['sql_str'] with the dialected SQL.
        if p.engine_dialect_query_dir is not None:
            self.workload.UseDialectSql(p)

        # Unused.
        assert p.use_adaptive_lr is None
        self.adaptive_lr_schedule = None
        if p.linear_decay_to_zero:
            self.adaptive_lr_schedule = (
                train_utils.AdaptiveMetricPiecewiseDecayToZero(
                    [(0, p.lr)],
                    metric_max_value=0,  # Does not matter.
                    total_steps=p.val_iters))

        # Logging.
        self._InitLogging()
        self.timer = train_utils.Timer()
        # Experience (replay) buffer.
        self.exp, self.exp_val = self._MakeExperienceBuffer()
        self._latest_replay_buffer_path = None

        # Cleanup handlers.  Ensures that the Ray cluster state remains healthy
        # even if this driver program is killed.
        signal.signal(signal.SIGTERM, self.Cleanup)
        signal.signal(signal.SIGINT, self.Cleanup)

    def Cleanup(self, signum, frame):
        """Calls ray.shutdown() on cleanup."""
        print('Received signal {}; calling ray.shutdown().'.format(
            signal.Signals(signum).name))
        ray.shutdown()

    def _MakeWorkload(self):
        p = self.params
        if os.path.isfile(p.init_experience):
            # Load the expert optimizer experience.
            with open(p.init_experience, 'rb') as f:
                workload = pickle.load(f)
            # Filter queries based on the current query_glob.
            workload.FilterQueries(p.query_dir, p.query_glob, p.test_query_glob)
        else:
            wp = envs.JoinOrderBenchmark.Params()
            wp.query_dir = p.query_dir
            wp.query_glob = p.query_glob
            wp.test_query_glob = None
            workload = wp.cls(wp)
            # Requires baseline to run in this scenario.
            p.run_baseline = True
        return workload

    def _InitLogging(self):
        p = self.params
        self.loggers = [
            pl_loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                         version=None,
                                         name='tensorboard_logs'),
            pl_loggers.WandbLogger(save_dir=os.getcwd(), project='balsa'),
        ]
        self.summary_writer = SummaryWriter()
        self.wandb_logger = self.loggers[-1]
        p_dict = balsa.utils.SanitizeToText(dict(p))
        for logger in self.loggers:
            logger.log_hyperparams(p_dict)
        with open(os.path.join(self.wandb_logger.experiment.dir, 'params.txt'),
                  'w') as f:
            # Files saved to wandb's rundir are auto-uploaded.
            f.write(p.ToText())
        if not p.run_baseline:
            self.LogExpertExperience(self.train_nodes, self.test_nodes)

    def _MakeExperienceBuffer(self):
        p = self.params
        if not p.run_baseline and p.sim:
            wi = self.GetOrTrainSim().training_workload_info
        else:
            # E.g., if sim is disabled, we just use the overall workload info
            # (thus, this covers both train & test queries).
            wi = self.workload.workload_info
        if p.tree_conv:
            plan_feat_cls = plans_lib.TreeNodeFeaturizer
            if p.plan_physical:
                # Physical-aware plan featurizer.
                plan_feat_cls = plans_lib.PhysicalTreeNodeFeaturizer
        else:
            plan_feat_cls = plans_lib.PreOrderSequenceFeaturizer
        query_featurizer_cls = _GetQueryFeaturizerClass(p)
        if self.sim is not None:
            # Use the already instantiated query featurizer, which may contain
            # computed normalization stats.
            query_featurizer_cls = self.GetOrTrainSim().query_featurizer
        exp = Experience(self.train_nodes,
                         p.tree_conv,
                         workload_info=wi,
                         query_featurizer_cls=query_featurizer_cls,
                         plan_featurizer_cls=plan_feat_cls)
        if p.prev_replay_buffers_glob is not None:
            exp.Load(p.prev_replay_buffers_glob,
                     p.prev_replay_keep_last_fraction)
            pa = plan_analysis.PlanAnalysis.Build(exp.nodes[exp.initial_size:])
            pa.Print()

        if p.prev_replay_buffers_glob_val is not None:
            print('Building validation experience buffer...')
            exp_val = Experience(self.train_nodes,
                                 p.tree_conv,
                                 workload_info=wi,
                                 query_featurizer_cls=query_featurizer_cls,
                                 plan_featurizer_cls=plan_feat_cls)
            exp_val.Load(p.prev_replay_buffers_glob_val)
            pa = plan_analysis.PlanAnalysis.Build(
                exp_val.nodes[exp_val.initial_size:])
            pa.Print()
        else:
            exp_val = None

        return exp, exp_val

    def _MakeDatasetAndLoader(self, log=True):
        p = self.params
        do_replay_training = (p.prev_replay_buffers_glob is not None and
                              p.agent_checkpoint is None)
        if do_replay_training or (p.skip_training_on_expert and
                                  self.curr_value_iter > 0):
            # The first 'n' nodes are expert experience.  Optionally, skip
            # training on those.  At iter 0, we don't skip (impl convenience)
            # but we don't train on those data.
            skip_first_n = len(self.train_nodes)
        else:
            # FIXME: ideally, let's make sure expert nodes are not added to the
            # replay buffer all together.  This was just to make sure iter=0
            # code doesn't break (e.g., that we calculate a label mean/std).
            skip_first_n = 0
        # Use only the latest round of executions?
        on_policy = p.on_policy
        if do_replay_training and self.curr_value_iter == 0:
            # Reloading replay buffers: let's train on all data.
            on_policy = False
        # TODO: avoid repeatedly featurizing already-featurized nodes.
        tup = self.exp.featurize(
            rewrite_generic=not p.plan_physical,
            verbose=False,
            skip_first_n=skip_first_n,
            deduplicate=p.dedup_training_data,
            physical_execution_hindsight=p.physical_execution_hindsight,
            on_policy=on_policy,
            use_last_n_iters=p.use_last_n_iters,
            use_new_data_only=p.use_new_data_only,
            skip_training_on_timeouts=p.skip_training_on_timeouts)
        # [np.ndarray], torch.Tensor, torch.Tensor, [float].
        all_query_vecs, all_feat_vecs, all_pos_vecs, all_costs = tup[:4]
        num_new_datapoints = None
        if len(tup) == 5:
            num_new_datapoints = tup[-1]

        if p.label_transform_running_stats and skip_first_n > 0:
            # Use running stats to stabilize.
            assert p.label_transforms in [
                ['log1p', 'standardize'],
                ['standardize'],
                ['sqrt', 'standardize'],
            ], p.label_transforms
            assert not p.physical_execution_hindsight
            labels = np.asarray([
                executed_node.cost
                for executed_node in self.exp.nodes[-skip_first_n:]
            ])
            if p.label_transforms[0] == 'log1p':
                labels = np.log(1 + labels)
            elif p.label_transforms[0] == 'sqrt':
                labels = np.sqrt(1 + labels)
            for label in labels:
                self.label_running_stats.Record(label)
            # PlansDataset would use these as-is, when supplied.
            self.label_mean = self.label_running_stats.Mean()
            self.label_std = self.label_running_stats.Std(epsilon_guard=False)

        dataset = ds.PlansDataset(all_query_vecs,
                                  all_feat_vecs,
                                  all_pos_vecs,
                                  all_costs,
                                  tree_conv=p.tree_conv,
                                  transform_cost=p.label_transforms,
                                  label_mean=self.label_mean,
                                  label_std=self.label_std,
                                  cross_entropy=p.cross_entropy)
        if do_replay_training and self.curr_value_iter == 0:
            self.label_mean = dataset.mean
            self.label_std = dataset.std
            print("Set label mean/std to offline set!")

        if (not p.update_label_stats_every_iter and self.label_mean is None and
                len(self.exp.nodes) > len(self.query_nodes)):
            # Update the stats once, as soon as some experience is collected.
            self.label_mean = dataset.mean
            self.label_std = dataset.std

        if self.exp_val is None:
            assert 0 <= p.validate_fraction <= 1, p.validate_fraction
            num_train = int(len(dataset) * (1 - p.validate_fraction))
            num_validation = len(dataset) - num_train
            assert num_train > 0 and num_validation >= 0, len(dataset)
            print('num_train={} num_validation={}'.format(
                num_train, num_validation))
            train_ds, val_ds = torch.utils.data.random_split(
                dataset, [num_train, num_validation])
            train_labels = np.asarray(all_costs)[train_ds.indices]
        else:
            tup = self.exp_val.featurize(
                rewrite_generic=not p.plan_physical,
                verbose=False,
                skip_first_n=skip_first_n,
                deduplicate=p.dedup_training_data,
                physical_execution_hindsight=p.physical_execution_hindsight,
                on_policy=False,
                use_last_n_iters=-1,
                use_new_data_only=False,
                skip_training_on_timeouts=p.skip_training_on_timeouts)
            (all_query_vecs_val, all_feat_vecs_val, all_pos_vecs_val,
             all_costs_val) = tup[:4]
            dataset_val = ds.PlansDataset(all_query_vecs_val,
                                          all_feat_vecs_val,
                                          all_pos_vecs_val,
                                          all_costs_val,
                                          tree_conv=p.tree_conv,
                                          transform_cost=p.label_transforms,
                                          label_mean=self.label_mean,
                                          label_std=self.label_std,
                                          cross_entropy=p.cross_entropy)
            train_ds, val_ds = dataset, dataset_val
            train_labels = all_costs
        if p.tree_conv:
            collate_fn = ds.InputBatch
        else:
            collate_fn = lambda xs: ds.InputBatch(
                xs,
                plan_pad_idx=self.exp.featurizer.pad(),
                parent_pos_pad_idx=self.exp.pos_featurizer.pad())

        train_loader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=p.bs,
                                                   shuffle=True,
                                                   collate_fn=collate_fn,
                                                   pin_memory=True)
        if p.validate_fraction > 0:
            val_loader = torch.utils.data.DataLoader(val_ds,
                                                     batch_size=p.bs,
                                                     collate_fn=collate_fn)
        else:
            val_loader = None
        if log:
            self._LogDatasetStats(train_labels, num_new_datapoints)

        return train_ds, train_loader, val_ds, val_loader

    def _LogDatasetStats(self, train_labels, num_new_datapoints):
        # Track # of training trees that are not timeouts.
        num_normal_trees = (np.asarray(train_labels) !=
                            self.timeout_label()).sum()
        data = [
            ('train/iter-{}-num-trees'.format(self.curr_value_iter),
             len(train_labels), self.curr_value_iter),
            ('train/num-trees', len(train_labels), self.curr_value_iter),
            ('train/iter-{}-num-normal-trees'.format(self.curr_value_iter),
             num_normal_trees, self.curr_value_iter),
            ('train/num-normal-trees', num_normal_trees, self.curr_value_iter),
            ('curr_value_iter', self.curr_value_iter, self.curr_value_iter),
        ]
        if num_new_datapoints is not None:
            data.append(('train/num-new-datapoints', num_new_datapoints,
                         self.curr_value_iter))
        self.LogScalars(data)

    def _MakeModel(self, dataset, train_from_scratch=False):
        p = self.params
        if not hasattr(self, 'model') or p.skip_sim_init_iter_1p:
            # Init afresh if either the model has not been constructed, or if
            # 'p.skip_sim_init_iter_1p', which explicitly says we want a fresh
            # model on iters >= 1.
            print('MakeModel afresh')
            model = MakeModel(p, self.exp, dataset)
        else:
            # Some training was performed before.  Weights would be
            # re-initialized by InitializeModel() below.
            model = self.model
        print('InitializeModel curr_value_iter={}'.format(self.curr_value_iter))
        if p.sim:
            should_skip = p.skip_sim_init_iter_1p and hasattr(self, 'model')
            if not should_skip:
                soft_assign_tau = 0.0
                if p.param_tau and hasattr(self, 'model'):
                    # Allows soft assign only if some training has been done.
                    soft_assign_tau = p.param_tau
                if train_from_scratch:
                    print('Training from scratch; forcing tau := 0.')
                    soft_assign_tau = 0.0
                self.ema_source_net = InitializeModel(
                    p,
                    model,
                    self.GetOrTrainSim(),
                    soft_assign_tau=soft_assign_tau,
                    soft_assign_use_ema=p.use_ema_source,
                    ema_source_tm1=self.ema_source_net)
        elif p.param_tau == 0.0:
            print('Reset model to randomized weights!')
            model.reset_weights()

        # Wrap it to get pytorch_lightning niceness.
        model = BalsaModel(
            p,
            model,
            loss_type=p.loss_type,
            torch_invert_cost=dataset.TorchInvertCost,
            query_featurizer=self.exp.query_featurizer,
            perturb_query_features=p.perturb_query_features,
            l2_lambda=p.l2_lambda,
            learning_rate=self.lr_schedule.Get()
            if self.adaptive_lr_schedule is None else
            self.adaptive_lr_schedule.Get(),
            optimizer_state_dict=self.prev_optimizer_state_dict,
            reduce_lr_within_val_iter=p.reduce_lr_within_val_iter)
        print('iter', self.curr_value_iter, 'lr', model.learning_rate)
        if p.agent_checkpoint is not None and self.curr_value_iter == 0:
            ckpt = torch.load(p.agent_checkpoint,
                              map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt['state_dict'])
            self.model = model.model
            print('Loaded value network checkpoint at iter',
                  self.curr_value_iter)
        if self.curr_value_iter == 0:
            ReportModel(model)
        return model

    def _MakeTrainer(self, train_loader):
        p = self.params
        max_steps = None
        # Control the number of SGD updates taken on each transition.  Use the
        # first 2 checks so that it's easier to reason about.
        if (p.use_last_n_iters > 0 and p.epochs == 1 and
                p.per_transition_sgd_steps > 0):
            # The replay window has n iters' data.  Each iter ages out the
            # oldest iter.  If we take 1 step per transition in the replay
            # window, then each transition would get updated n times.
            desired_update_fraction = float(
                p.per_transition_sgd_steps) / p.use_last_n_iters
            # 1 epoch = 1 pass over the replay window.
            # Total num steps per epoch * desired_update_fraction.
            max_steps = int(np.ceil(
                len(train_loader) * desired_update_fraction))
            print('per_transition_sgd_steps={} max_batches={} '
                  'num_batches_per_epoch={}'.format(p.per_transition_sgd_steps,
                                                    max_steps,
                                                    len(train_loader)))
        return pl.Trainer(
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=p.epochs,
            max_steps=max_steps,
            # Add logging metrics per this many batches.
            row_log_interval=1,
            # Do validation per this many train epochs.
            check_val_every_n_epoch=p.validate_every_n_epochs,
            # Patience = # of validations with no improvements before stopping.
            early_stop_callback=pl.callbacks.EarlyStopping(
                patience=p.validate_early_stop_patience,
                mode='min',
                verbose=True),
            weights_summary=None,
            logger=self.loggers,
            gradient_clip_val=p.gradient_clip_val,
            num_sanity_val_steps=2 if p.validate_fraction > 0 else 0,
        )

    def _LoadBestCheckpointForEval(self, model, trainer):
        """Loads the checkpoint with the best validation loss."""
        train_utils.LoadBestCheckpointForEval(model, trainer)

    def timeout_label(self):
        return 4096 * 1000

    def LogScalars(self, metrics):
        if not isinstance(metrics, list):
            assert len(metrics) == 3, 'Expected (tag, val, global_step)'
            metrics = [metrics]
        for tag, val, global_step in metrics:
            self.summary_writer.add_scalar(tag, val, global_step=global_step)
        d = dict([(tag, val) for tag, val, _ in metrics])
        assert len(set([gs for _, _, gs in metrics])) == 1, metrics
        self.wandb_logger.log_metrics(d)

    def LogExpertExperience(self, expert_train_nodes, expert_test_nodes):
        p = self.params
        total_s = 0
        data_to_log = []
        num_joins = []
        for node in expert_train_nodes:
            # Real latency in ms was assigned to node.cost as impl convenience.
            data_to_log.append(
                ('latency_expert/q{}'.format(node.info['query_name']),
                 node.cost / 1e3, 0))
            total_s += node.cost / 1e3
            num_joins.append(len(node.leaf_ids()) - 1)
        data_to_log.append(('latency_expert/workload', total_s, 0))
        print('latency_expert/workload (seconds): {:.2f} ({} queries)'.format(
            total_s, len(expert_train_nodes)))

        if p.test_query_glob is not None:
            total_s_test = 0
            for node in expert_test_nodes:
                data_to_log.append(
                    ('latency_expert_test/q{}'.format(node.info['query_name']),
                     node.cost / 1e3, 0))
                total_s_test += node.cost / 1e3
                num_joins.append(len(node.leaf_ids()) - 1)
            data_to_log.append(
                ('latency_expert_test/workload', total_s_test, 0))
            print('latency_expert_test/workload (seconds): {:.2f} ({} queries)'.
                  format(total_s_test, len(expert_test_nodes)))
        data_to_log.append(('curr_value_iter', 0, 0))
        self.LogScalars(data_to_log)
        print('Number of joins [{}, {}], avg {:.1f}'.format(
            np.min(num_joins), np.max(num_joins), np.mean(num_joins)))

    def GetOrTrainSim(self):
        p = self.params
        if self.sim is None:
            self.sim = TrainSim(p, self.loggers)
        return self.sim

    def RunBaseline(self):
        p = self.params
        print('Dropping buffer cache.')
        postgres.DropBufferCache()
        print('Running queries as-is (baseline PG performance)...')

        def Args(node):
            return {
                'query_name': node.info['query_name'],
                'sql_str': node.info['sql_str'],
                'hint_str': None,
                'hinted_plan': None,
                'query_node': node,
                'predicted_latency': 0,
                'silent': True,
                'use_local_execution': p.use_local_execution,
                'engine': p.engine,
            }

        tasks = []
        for node in self.all_nodes:
            # Run the query.
            tasks.append(
                ExecuteSql.options(resources={
                    f'node:{ray.util.get_node_ip_address()}': 1,
                }).remote(**Args(node)))
        if not p.use_local_execution:
            refs = ray.get(tasks)
        else:
            refs = tasks
        for i, node in enumerate(self.all_nodes):
            result_tup = ray.get(refs[i])
            assert isinstance(
                result_tup,
                (pg_executor.Result, dbmsx_executor.Result)), result_tup
            result, real_cost, _, message = ParseExecutionResult(
                result_tup, **Args(node))
            # Save real cost (execution latency) to actual.
            node.cost = real_cost
            print('---------------------------------------')
            if p.engine == 'postgres':
                node.info['explain_json'] = result[0][0][0]
                # 'node' is a PG plan; doesn't make sense to print if executed
                # on a different engine.
                print(node)
            print(message)
            print('q{},{:.1f} (baseline)'.format(node.info['query_name'],
                                                 real_cost))
            print('Execution time: {}'.format(real_cost))
        # NOTE: if engine != pg, we're still saving PG plans but with target
        # engine's latencies.  This mainly affects debug strings.
        Save(self.workload, './data/initial_policy_data.pkl')
        self.LogExpertExperience(self.train_nodes, self.test_nodes)

    def Train(self, train_from_scratch=False):
        p = self.params
        self.timer.Start('train')
        train_ds, train_loader, _, val_loader = self._MakeDatasetAndLoader(
            log=not train_from_scratch)
        # Fields accessed: 'costs' (for p.cross_entropy; unused);
        # 'TorchInvertCost', 'InvertCost'.  We don't access the actual data.
        # Thus, it doesn't matter if we use a Dataset referring to the entire
        # data or just the train data.  (Subset.dataset returns the entire
        # original data is where the subset is sampled.)
        #
        # The else branch is for when self.exp_val is not None
        # (p.prev_replay_buffers_glob_val).
        plans_dataset = train_ds.dataset if isinstance(
            train_ds, torch.utils.data.Subset) else train_ds
        model = self._MakeModel(plans_dataset, train_from_scratch)
        if train_from_scratch:
            model.SetLoggingPrefix('train_from_scratch/iter-{}-'.format(
                self.curr_value_iter))
        else:
            model.SetLoggingPrefix('train/iter-{}-'.format(
                self.curr_value_iter))
        trainer = self._MakeTrainer(train_loader)
        if train_from_scratch:
            trainer.fit(model, train_loader, val_loader)
        elif not (self.curr_value_iter == 0 and p.skip_training_on_expert and
                  (p.prev_replay_buffers_glob is None or
                   p.agent_checkpoint is not None)):
            # This condition only affects the first ever call to Train().
            # Iteration 0 doesn't have a timeout limit, so during the second
            # call to Train() we would always have self.curr_value_iter == 1.
            trainer.fit(model, train_loader, val_loader)
            self.model = model.model
            # Optimizer state dict now available.
            self.prev_optimizer_state_dict = None
            if p.inherit_optimizer_state:
                self.prev_optimizer_state_dict = trainer.optimizers[
                    0].state_dict()
        # Load best ckpt.
        self._LoadBestCheckpointForEval(model, trainer)
        self.timer.Stop('train')
        return model, plans_dataset

    def _SampleInternalNode(self, node):
        num_leaves = len(node.leaf_ids())
        num_internal = num_leaves - 1
        assert num_internal > 0, node

        def _Sample(subnode, remaining_internal):
            if len(subnode.children) == 0:
                return None, remaining_internal
            if np.random.rand() < 1. / remaining_internal:
                # Pick this internal node.
                return subnode, None
            # Left branch.
            sampled, rem = _Sample(subnode.children[0], remaining_internal - 1)
            if sampled is not None:
                return sampled, None
            # Right branch.
            return _Sample(subnode.children[1], rem)

        sampled_node, _ = _Sample(node, num_internal)
        return sampled_node

    def SelectPlan(self, found_plans, predicted_latency, found_plan, planner,
                   query_node):
        """Exploration + action selection."""
        p = self.params
        # Sanity check that at most one exploration strategy is specified.
        num_explore_schemes = (p.epsilon_greedy + p.explore_soft_v +
                               p.explore_visit_counts +
                               p.explore_visit_counts_sort +
                               p.explore_visit_counts_latency_sort)
        assert num_explore_schemes <= 1
        if p.epsilon_greedy:
            assert p.epsilon_greedy_random_transform + \
                p.epsilon_greedy_random_plan <= 1
        if p.epsilon_greedy > 0:
            r = np.random.rand()
            if r < p.epsilon_greedy:
                # Epsilon-greedy policy.
                if p.epsilon_greedy_random_transform:
                    # Randomly transform the best found plan.
                    print('Before: {}'.format(found_plan.hint_str()))
                    sampled_node = self._SampleInternalNode(found_plan)
                    cs = sampled_node.children
                    sampled_node.children = [cs[1], cs[0]]
                    print('After: {}'.format(found_plan.hint_str()))
                elif p.epsilon_greedy_random_plan and self.curr_value_iter > 0:
                    # Randomly pick a plan.
                    predicted_latency, found_plan = planner.SampleRandomPlan(
                        query_node)
                else:
                    # Randomly pick a plan from all found plans.
                    rand_idx = np.random.randint(len(found_plans))
                    predicted_latency, found_plan = found_plans[rand_idx]
        elif p.explore_soft_v:
            # Sample proportional to exp (-V_theta(s)).
            with torch.no_grad():
                v_values = torch.tensor([-v for v, _ in found_plans],
                                        dtype=torch.float32)
                v_values -= v_values.max()
                softmax = torch.softmax(v_values, dim=0)
                rand_idx = torch.multinomial(softmax, num_samples=1).item()
            predicted_latency, found_plan = found_plans[rand_idx]
        elif (p.explore_visit_counts or p.explore_visit_counts_sort or
              p.explore_visit_counts_latency_sort):
            visit_counts = np.zeros(len(found_plans), dtype=np.float32)
            query_name = query_node.info['query_name']
            for i, (_, plan) in enumerate(found_plans):
                hint_str = HintStr(plan,
                                   with_physical_hints=p.plan_physical,
                                   engine=p.engine)
                visit_counts[i] = self.query_execution_cache.GetVisitCount(
                    key=(query_name, hint_str))
            visit_sum = visit_counts.sum()
            if visit_sum > 0:
                # If none are visited, skip this step.
                if p.explore_visit_counts:
                    # Sample proportional to
                    #    visit_sum / (1 + num_visits(plan_i))
                    # Disregarding predicted V() is sort of saying they are
                    # probably all similar, let's just use visit counts.
                    scores = visit_sum * 1.0 / (1.0 + visit_counts)
                    with torch.no_grad():
                        scores = torch.from_numpy(scores)
                        rand_idx = torch.multinomial(scores,
                                                     num_samples=1).item()
                    predicted_latency, found_plan = found_plans[rand_idx]
                    print('counts', visit_counts, 'sampled_idx', rand_idx,
                          'sampled_cnt', visit_counts[rand_idx])
                else:
                    # Sort by (visit count, predicted latency).  Execute the
                    # smallest.
                    assert p.explore_visit_counts_sort or \
                           p.explore_visit_counts_latency_sort
                    # Cast to int so the debug messages look nicer.
                    visit_counts = visit_counts.astype(np.int32, copy=False)

                    # If all plans have been visited, sort by predicted latency.
                    if (p.explore_visit_counts_latency_sort and
                            all([x > 0 for x in visit_counts])):
                        found_plans_sorted, visit_counts_sorted = zip(
                            *sorted(zip(found_plans, visit_counts),
                                    key=lambda tup: tup[0][0]))
                    else:
                        found_plans_sorted, visit_counts_sorted = zip(
                            *sorted(zip(found_plans, visit_counts),
                                    key=lambda tup: (tup[1], tup[0][0])))
                    predicted_latency, found_plan = found_plans_sorted[0]
                    print(
                        'selected cnt,latency=({}, {});'.format(
                            visit_counts_sorted[0], predicted_latency),
                        'sorted:',
                        list(
                            zip(visit_counts_sorted,
                                map(lambda tup: tup[0], found_plans_sorted))))

        return predicted_latency, found_plan

    def PlanAndExecute(self, model, planner, is_test=False, max_retries=3):
        p = self.params
        model.eval()
        to_execute = []
        tasks = []
        if p.sim:
            sim = self.GetOrTrainSim()
        positions_of_min_predicted = []
        nodes = self.test_nodes if is_test else self.train_nodes

        # Plan the workload.
        kwargs = []
        task_lambdas = []
        exec_results = []
        if not is_test:
            self.timeout_controller.OnIterStart()
        planner_config = None
        if p.planner_config is not None:
            planner_config = optim.PlannerConfig.Get(p.planner_config)
        epsilon_greedy_within_beam_search = 0
        if not is_test and p.epsilon_greedy_within_beam_search > 0:
            epsilon_greedy_within_beam_search = \
                p.epsilon_greedy_within_beam_search

        self.timer.Start('plan_test_set' if is_test else 'plan')
        for i, node in enumerate(nodes):
            print('---------------------------------------')
            tup = planner.plan(
                node,
                p.search_method,
                bushy=p.bushy,
                return_all_found=True,
                verbose=False,
                planner_config=planner_config,
                epsilon_greedy=epsilon_greedy_within_beam_search,
                # prevents Ext-JOB test query hints from failing.
                avoid_eq_filters=is_test and p.avoid_eq_filters,
            )
            planning_time, found_plan, predicted_latency, found_plans = tup
            predicted_latency, found_plan = self.SelectPlan(
                found_plans, predicted_latency, found_plan, planner, node)
            print('{}q{}, predicted time: {:.1f}'.format(
                '[Test set] ' if is_test else '', node.info['query_name'],
                predicted_latency))
            # Calculate monitoring info.
            predicted_costs = None
            if p.sim:
                predicted_costs = sim.Predict(node,
                                              [tup[1] for tup in found_plans])
            # Model-predicted latency of the expert plan.  Allows us to track
            # what exactly the model thinks of the expert plan.
            node.info['curr_predicted_latency'] = planner.infer(node, [node])[0]
            self.LogScalars([('predicted_latency_expert_plans/q{}'.format(
                node.info['query_name']),
                              node.info['curr_predicted_latency'] / 1e3,
                              self.curr_value_iter)])

            hint_str = HintStr(found_plan,
                               with_physical_hints=p.plan_physical,
                               engine=p.engine)
            hinted_plan = found_plan

            # Launch tasks.
            if is_test:
                curr_timeout = None

                # Roughly 18 mins.  Good enough to cover disk filled error.
                curr_timeout = 1100000
            else:
                curr_timeout = self.timeout_controller.GetTimeout(node)
            print('q{},(predicted {:.1f}),{}'.format(node.info['query_name'],
                                                     predicted_latency,
                                                     hint_str))
            to_execute.append((node.info['sql_str'], hint_str, planning_time,
                               found_plan, predicted_latency, curr_timeout))
            if p.use_cache:
                exec_result = self.query_execution_cache.Get(
                    key=(node.info['query_name'], hint_str))
            else:
                exec_result = None
            exec_results.append(exec_result)
            kwarg = {
                'query_name': node.info['query_name'],
                'sql_str': node.info['sql_str'],
                'hint_str': hint_str,
                'hinted_plan': hinted_plan,
                'query_node': node,
                'predicted_latency': predicted_latency,
                'curr_timeout_ms': curr_timeout,
                'found_plans': found_plans,
                'predicted_costs': predicted_costs,
                'is_test': is_test,
                'use_local_execution': p.use_local_execution,
                'plan_physical': p.plan_physical,
                'engine': p.engine,
            }

            kwargs.append(kwarg)
            if exec_result is None:
                # Lambdas are late-binding; use a default argument value to
                # ensure that when invoked later, the correct kwarg is used.
                fn = lambda task_index=i: ExecuteSql.options(resources={
                    f'node:{ray.util.get_node_ip_address()}': 1,
                }).remote(**kwargs[task_index])
            else:
                # Cache hit.  See comment above for why the default arg val.
                fn = lambda task_index=i: ray.put(exec_results[task_index])
            task_lambdas.append(fn)
            tasks.append(fn())

            # Logging: which terminal plan is the predicted cheapest?
            min_p_latency = 1e30
            min_pos = 0
            for pos, (p_latency, found_plan) in enumerate(found_plans):
                if p_latency < min_p_latency:
                    min_p_latency = p_latency
                    min_pos = pos
            positions_of_min_predicted.append(min_pos)

        self.timer.Stop('plan_test_set' if is_test else 'plan')
        self.timer.Start('wait_for_executions_test_set'
                         if is_test else 'wait_for_executions')
        self.wandb_logger.log_metrics({
            'train/position-of-min-predicted-cost-plan':
                wandb.Histogram(positions_of_min_predicted),
        })
        # Wait for all execution of the planned queries.
        print('{}Waiting on Ray tasks...value_iter={}'.format(
            '[Test set] ' if is_test else '', self.curr_value_iter))
        try:
            refs = ray.get(tasks)
        except Exception as e:
            print('ray.get(tasks) received exception:', e)
            time.sleep(10)
            print('Canceling Ray tasks.')
            for task in tasks:
                ray.cancel(task)
            if max_retries > 0:
                print('Retrying PlanAndExecute() (max_retries={}).'.format(
                    max_retries))
                return self.PlanAndExecute(model,
                                           planner,
                                           is_test,
                                           max_retries=max_retries - 1)
            else:
                print('Retries exhausted; raising the exception.')
                raise e
        execution_results = []
        for i, task in enumerate(refs):
            result_tup = None
            is_cached_plan = True
            if isinstance(task, ray.ObjectRef):
                # New plan: remote PG execution.
                try:
                    result_tup = ray.get(task)
                    is_cached_plan = False
                except ray.exceptions.RayTaskError as e:
                    # This can happen when the server gets crashed by other
                    # drivers, this driver's connection would break down.  OK
                    # to wait for the server to recover and retry.
                    #
                    # Alternatively, it can also be this 'task' wrote too large
                    # temp files, causing a DiskFull error.  We will treat it
                    # as a timeout if this happens twice on different servers.
                    #
                    # NOTE: if dbmsx is enabled, handle it here too.
                    is_disk_full = 'psycopg2.errors.DiskFull' in str(e)
                    num_secs = 8 + (np.random.rand() * 5)
                    print('Exception received:\n{}\nSleeping for {} secs'
                          ' before retrying.'.format(e, num_secs))
                    time.sleep(num_secs)
                    # Here, 'task' is likely unresponsive so calling
                    # ray.get(task) would hang.  Let's resubmit the task.
                    print('Resubmitting.')
                    new_task = task_lambdas[i]()
                    print('Calling ray.get() on the new task.')
                    new_ref = ray.get(new_task)
                    try:
                        result_tup = ray.get(new_ref)
                    except psycopg2.errors.DiskFull:
                        # Catch double DiskFull errors and treat as a timeout.
                        # TODO: what if a test query triggered this?
                        assert is_disk_full, 'DiskFull should happen twice.'
                        print('DiskFull happens twice; treating as a timeout.'
                              '  *NOTE* The agent will train on the timeout '
                              'label regardless of whether use_timeout is set.')
                        result_tup = pg_executor.Result(result=[],
                                                        has_timeout=True,
                                                        server_ip=None)
                    # except dbmsx.DatabaseError as e:
                    #     # Handle DBMS-X errors.
                    #     raise NotImplementedError
                    is_cached_plan = False
                    print('Retry succeeded.')
            elif isinstance(task, (pg_executor.Result, dbmsx_executor.Result)):
                # New plan: local PG execution.
                result_tup = task
                is_cached_plan = False
            else:
                # This happens on a query execution cache hit.
                assert isinstance(task, tuple), task
                assert len(task) == 2, task
                # The cache records either a positive latency or -1 to indicate
                # timeouts.  See FeedbackExecution().
                # FIXME: is there a logic error here?  Cases: (1) cached
                # execution didn't timeout, but its latency exceeded current
                # timeout; (2) if relaxation enabled, the converse may happen:
                # cached execution timed out before, but now timeout is bigger
                # so it could've finished exec.
                cached_result_tup = task[0][0]
                result_tup = cached_result_tup
            assert isinstance(
                result_tup,
                (pg_executor.Result, dbmsx_executor.Result)), result_tup
            result_tups = ParseExecutionResult(result_tup, **kwargs[i])
            assert len(result_tups) == 4
            print(result_tups[-1])  # Messages.
            execution_results.append(result_tups[:-1])
            # Increment counts for training.
            if not is_test:
                if is_cached_plan:
                    self.curr_iter_skipped_queries += 1
                    if self.adaptive_lr_schedule is not None:
                        self.adaptive_lr_schedule.SetOrTakeMax(
                            self.curr_iter_skipped_queries)
                else:
                    self.num_query_execs += 1
        self.timer.Stop('wait_for_executions_test_set'
                        if is_test else 'wait_for_executions')
        return to_execute, execution_results

    def FeedbackExecution(self, to_execute, execution_results):
        p = self.params
        results = []
        iter_total_latency = 0
        iter_max_latency = 0
        has_timeouts = False
        num_timeouts = 0
        # Errors the current policy incurs on (agent plans for train queries,
        # expert plans for train queries).
        agent_plans_diffs = []
        expert_plans_diffs = []
        for node, result_tup, to_execute_tup in zip(self.train_nodes,
                                                    execution_results,
                                                    to_execute):
            result, real_cost, server_ip = result_tup
            _, hint_str, planning_time, actual, predicted_latency, \
                curr_timeout = to_execute_tup
            # Record execution result, potentially with real_cost = -1
            # indicating a timeout.  The cache would only record a lower
            # latency value so once it gets a -1 label for a plan, it'd not be
            # updated again.  If a future iteration this plan is still
            # selected, it'd get the same -1 label from the cache, ensuring
            # that has_timeouts below would be set to True correctly.
            self.query_execution_cache.Put(key=(node.info['query_name'],
                                                hint_str),
                                           value=result_tup,
                                           latency=real_cost)
            self.timeout_controller.RecordQueryExecution(node, real_cost)

            # Process timeout.
            # FIXME: even when use_timeout=False, pg_executor may treat a rare
            # InternalError_ or OperationalError as a timeout event.  These are
            # rare but could incorrectly get a timeout label below.  We should
            # fix this by marking a Node as a timeout & allowing Experience to
            # skip featurizing those marked nodes.
            if real_cost < 0:
                has_timeouts = True
                num_timeouts += 1
                self.num_total_timeouts += 1
                if p.special_timeout_label:
                    real_cost = self.timeout_label()
                    print('Timeout detected! Assigning a special label',
                          real_cost, '(server_ip={})'.format(server_ip))
                else:
                    real_cost = curr_timeout * 2
                    print('Timeout detected! Assigning 2*timeout as label',
                          real_cost, '(server_ip={})'.format(server_ip))
                # At this point, 'actual' is a Node produced from the agent
                # consisting of just scan/join nodes.  It has gone through hint
                # checks in ParseExecutionResult() -- i.e., it should be the
                # same as the EXPLAIN result from a local PG with an
                # agent-produced hint string.
                #
                # We manually fill in this field for hindsight labeling (if
                # enabled) to work.  Intermediate goals are not collected since
                # we don't know what those "sub-latencies" are.
                actual.actual_time_ms = real_cost
                # Mark a special timeout field.
                actual.is_timeout = True
            else:
                agent_plans_diffs.append((real_cost - predicted_latency) / 1e3)
            expert_plans_diffs.append(
                (node.cost - node.info['curr_predicted_latency']) / 1e3)

            assert real_cost > 0, real_cost
            actual.cost = real_cost
            actual.info = copy.deepcopy(node.info)
            actual.info.pop('explain_json')

            # Put into experience/replay buffer.
            self.exp.add(actual)
            # Update the best plan cache.
            self.best_plans.Put(key=node.info['query_name'],
                                value=actual,
                                latency=real_cost)

            # Logging.
            results.append(result)
            iter_total_latency += real_cost
            iter_max_latency = max(iter_max_latency, real_cost)
            self.LogScalars([
                ('latency/q{}'.format(node.info['query_name']), real_cost / 1e3,
                 self.curr_value_iter),
                # Max per-query latency in this iter.  This bounds
                # the time required for query execution if we were
                # to parallelize everything.
                ('curr_iter_max_ms', iter_max_latency, self.curr_value_iter),
            ])

        # Logging.
        self.LogScalars([
            # Prediction errors.
            ('latency/mean_l1_agent_secs', np.mean(np.abs(agent_plans_diffs)),
             self.curr_value_iter),
            ('latency/mean_pred-tgt_agent_secs', -np.mean(agent_plans_diffs),
             self.curr_value_iter),
            ('latency/mean_l1_expert_secs', np.mean(np.abs(expert_plans_diffs)),
             self.curr_value_iter),
            ('latency/mean_pred-tgt_expert_secs', -np.mean(expert_plans_diffs),
             self.curr_value_iter),
            # Timeout metrics.
            ('curr_timeout',
             curr_timeout / 1e3 if curr_timeout is not None else 0,
             self.curr_value_iter),
            ('num_total_timeouts', self.num_total_timeouts,
             self.curr_value_iter),
            ('num_timeouts', num_timeouts, self.curr_value_iter),
            # X-axis.
            ('curr_value_iter', self.curr_value_iter, self.curr_value_iter),
        ])

        return iter_total_latency, has_timeouts

    def _SaveReplayBuffer(self, iter_total_latency):
        p = self.params
        # "<class 'experiments.ConfigName'>" -> "ConfigName".
        experiment = str(p.cls).split('.')[-1][:-2]
        path = 'data/replay-{}-{}execs-{}nodes-{}s-{}iters-{}.pkl'.format(
            experiment, self.num_query_execs, len(self.exp.nodes),
            int(iter_total_latency / 1e3), self.curr_value_iter,
            self.wandb_logger.experiment.id)
        self.exp.Save(path)
        # Remove previous.
        if self._latest_replay_buffer_path is not None:
            os.remove(self._latest_replay_buffer_path)
        self._latest_replay_buffer_path = path

    def LogTestExperience(self,
                          to_execute_test,
                          execution_results,
                          tag='latency_test'):
        assert len(self.test_nodes) == len(execution_results)
        iter_total_latency = 0
        rows = []
        data = []
        has_timeouts = False
        # Errors the current policy incurs on (agent plans for test queries,
        # expert plans for test queries).
        agent_plans_diffs = []
        expert_plans_diffs = []
        for node, to_execute, result_tup in zip(self.test_nodes,
                                                to_execute_test,
                                                execution_results):
            _, real_cost, _ = result_tup
            if real_cost < 0:
                has_timeouts = True
                break
            iter_total_latency += real_cost
            rows.append((node.info['query_name'], real_cost / 1e3,
                         self.curr_value_iter))
            data.append(('{}/q{}'.format(tag, node.info['query_name']),
                         real_cost / 1e3, self.curr_value_iter))
            # Tracks prediction errors.
            agent_plans_diffs.append((real_cost - to_execute[-2]) / 1e3)
            expert_plans_diffs.append(
                (node.cost - node.info['curr_predicted_latency']) / 1e3)
        if has_timeouts:
            # "Timeouts" for test set queries are rare events such as
            # out-of-disk errors due to a lot of intermediate results being
            # written out.
            print(
                '[Test set {}] timeout events detected during eval'.format(tag))
            return
        # Log a table of latencies, sorted by descending latency.
        rows = list(sorted(rows, key=lambda r: r[1], reverse=True))
        table = wandb.Table(columns=['query_name', tag, 'curr_value_iter'],
                            rows=rows)
        self.wandb_logger.experiment.log({'{}_table'.format(tag): table})

        data.extend([
            (tag + '/workload', iter_total_latency / 1e3, self.curr_value_iter),
            (tag + '/mean_l1_agent_secs', np.mean(np.abs(agent_plans_diffs)),
             self.curr_value_iter),
            (tag + '/mean_pred-tgt_agent_secs', -np.mean(agent_plans_diffs),
             self.curr_value_iter),
            (tag + '/mean_l1_expert_secs', np.mean(np.abs(expert_plans_diffs)),
             self.curr_value_iter),
            (tag + '/mean_pred-tgt_expert_secs', -np.mean(expert_plans_diffs),
             self.curr_value_iter),
            ('num_query_execs', self.num_query_execs, self.curr_value_iter),
            ('curr_value_iter', self.curr_value_iter, self.curr_value_iter),
        ])
        if tag == 'latency_test':
            self.overall_best_test_latency = min(self.overall_best_test_latency,
                                                 iter_total_latency / 1e3)
            val_to_log = self.overall_best_test_latency
        elif tag == 'latency_test_swa':
            self.overall_best_test_swa_latency = min(
                self.overall_best_test_swa_latency, iter_total_latency / 1e3)
            val_to_log = self.overall_best_test_swa_latency
        else:
            assert tag == 'latency_test_ema', tag
            self.overall_best_test_ema_latency = min(
                self.overall_best_test_ema_latency, iter_total_latency / 1e3)
            val_to_log = self.overall_best_test_ema_latency
        data.append((tag + '/workload_best', val_to_log, self.curr_value_iter))
        self.LogScalars(data)

    def _MakePlanner(self, model, dataset):
        p = self.params
        if self.sim is not None and self.sim.IsPlanPhysicalButUseGenericOps():
            # With generic Scan/Join removed.
            wi = self.sim._GetPlanner().workload_info
        else:
            wi = self.exp.workload_info
        return optim.Optimizer(
            wi,
            self.exp.featurizer,
            self.exp.pos_featurizer,
            self.exp.query_featurizer,
            # NOTE: unit seems wrong if initialized from SIM.
            dataset.InvertCost,
            model,
            p.tree_conv,
            p.beam,
            search_until_n_complete_plans=p.search_until_n_complete_plans,
            plan_physical=p.plan_physical,
            use_plan_restrictions=p.real_use_plan_restrictions)

    def RunOneIter(self):
        p = self.params
        self.curr_iter_skipped_queries = 0
        # Train the model.
        model, dataset = self.Train()
        # Replay buffer reset (if enabled).
        if self.curr_value_iter == p.replay_buffer_reset_at_iter:
            self.exp.DropAgentExperience()

        planner = self._MakePlanner(model, dataset)
        # Use the model to plan the workload.  Execute the plans and get
        # latencies.
        to_execute, execution_results = self.PlanAndExecute(model,
                                                            planner,
                                                            is_test=False)
        # Add exeuction results to the experience buffer.
        iter_total_latency, has_timeouts = self.FeedbackExecution(
            to_execute, execution_results)
        # Logging.
        if not has_timeouts:
            self.overall_best_train_latency = min(
                self.overall_best_train_latency, iter_total_latency / 1e3)
            to_log = [
                ('latency/workload', iter_total_latency / 1e3,
                 self.curr_value_iter),
                ('latency/workload_best', self.overall_best_train_latency,
                 self.curr_value_iter),
                ('num_query_execs', self.num_query_execs, self.curr_value_iter),
                ('num_queries_with_eps_random', planner.num_queries_with_random,
                 self.curr_value_iter),
                ('curr_iter_skipped_queries', self.curr_iter_skipped_queries,
                 self.curr_value_iter),
                ('curr_value_iter', self.curr_value_iter, self.curr_value_iter),
                ('lr', model.learning_rate, self.curr_value_iter),
            ]
            if p.reduce_lr_within_val_iter:
                to_log.append(('iter_final_lr', model.latest_per_iter_lr,
                               self.curr_value_iter))
            self.LogScalars(to_log)
        self.SaveBestPlans()
        if (self.curr_value_iter + 1) % 5 == 0:
            self.SaveAgent(model, iter_total_latency)
        # Run and log test queries.
        self.EvaluateTestSet(model, planner)

        if p.track_model_moving_averages:
            # Update model averages.
            # 1. EMA.  Aka Polyak averaging.
            if self.curr_value_iter >= 0:
                self.UpdateMovingAverage(model,
                                         moving_average='ema',
                                         ema_decay=p.ema_decay)
                if (self.curr_value_iter + 1) % 5 == 0:
                    # Use EMA to evaluate test set too.
                    self.SwapMovingAverage(model, moving_average='ema')
                    # Clear the planner's label cache.
                    planner.SetModel(model)
                    self.EvaluateTestSet(model, planner, tag='latency_test_ema')
                    self.SwapMovingAverage(model, moving_average='ema')

            # 2. SWA: Stochastic weight averaging.
            if self.curr_value_iter >= 75:
                self.UpdateMovingAverage(model, moving_average='swa')
                if (self.curr_value_iter + 1) % 5 == 0:
                    self.SwapMovingAverage(model, moving_average='swa')
                    # Clear the planner's label cache.
                    planner.SetModel(model)
                    self.EvaluateTestSet(model, planner, tag='latency_test_swa')
                    self.SwapMovingAverage(model, moving_average='swa')

        return has_timeouts

    def SaveBestPlans(self):
        """Saves the best plans found so far.

        Write to best_plans/, under the run directory managed by wandb:

        - <query_name>.sql: best plan for each query as a commented .sql file
        - all.sql: all commented sql texts concatenated together
        - latencies.txt: a CSV containing "query name -> latency", including an
            "all" entry for the total latency
        - plans.pkl: a map {query name -> plans_lib.Node, the best plan}

        The all.sql file (or the individual query sql files) are ready to be
        piped to a SQL shell for execution.
        """
        p = self.params
        best_plans_dir = os.path.join(self.wandb_logger.experiment.dir,
                                      'best_plans/')
        qnames = []
        latencies = []
        sqls = []
        total_ms = 0
        all_nodes = {}
        # For calling wandb.save() to continuously upload on update.
        w = self.wandb_logger.experiment
        wandb_dir = w.dir
        # Save all plans.
        for query_name, value_tup in sorted(self.best_plans._cache.items()):
            best_node, best_latency = value_tup
            all_nodes[query_name] = best_node
            hint = HintStr(best_node,
                           with_physical_hints=p.plan_physical,
                           engine=p.engine)
            sql = AddCommentToSql(best_node.info['sql_str'], hint, p.engine)
            path = SaveText(
                sql, os.path.join(best_plans_dir, '{}.sql'.format(query_name)))
            w.save(path, base_path=wandb_dir)
            sqls.append(sql)
            qnames.append(query_name)
            latencies.append(best_latency)
            total_ms += best_latency
        qnames.append('all')
        latencies.append(total_ms)
        # all.sql.
        path = SaveText('\n'.join(sqls), os.path.join(best_plans_dir,
                                                      'all.sql'))
        w.save(path, base_path=wandb_dir)
        # latencies.txt.
        pd.DataFrame({
            'query': qnames,
            'latency_ms': latencies
        }).to_csv(os.path.join(best_plans_dir, 'latencies.txt'),
                  header=True,
                  index=False)
        w.save(os.path.join(best_plans_dir, 'latencies.txt'),
               base_path=wandb_dir)
        # plans.pkl.
        path = Save(all_nodes, os.path.join(best_plans_dir, 'plans.pkl'))
        w.save(path, base_path=wandb_dir)

    def SaveAgent(self, model, iter_total_latency):
        """Saves the complete execution state of the agent."""
        # TODO: not complete state, currently missing:
        #  - query exec cache
        #  - moving averages
        #  - a bunch of fields (see Run())
        # TODO: support reloading & resume.

        # Model weights.  Saved under <wandb dir>/checkpoint.pt.
        #
        # Model weights can be reloaded with:
        #   model = TheModelClass(*args, **kwargs)
        #   model.load_state_dict(torch.load(PATH))
        ckpt_path = os.path.join(self.wandb_logger.experiment.dir,
                                 'checkpoint.pt')
        torch.save(model.state_dict(), ckpt_path)
        SaveText(
            'value_iter,{}'.format(self.curr_value_iter),
            os.path.join(self.wandb_logger.experiment.dir,
                         'checkpoint-metadata.txt'))
        print('Saved iter={} checkpoint to: {}'.format(self.curr_value_iter,
                                                       ckpt_path))

        # Replay buffer.  Saved under data/.
        self._SaveReplayBuffer(iter_total_latency)

    def UpdateMovingAverage(self, model, moving_average, ema_decay=None):
        """Updates the current EMA/SWA using 'model'."""
        assert moving_average in ['swa', 'ema'], moving_average
        moving_average_state_dict = self.moving_average_model_state_dict[
            moving_average]
        num_averages = self.moving_average_counter_dict[moving_average]
        self.moving_average_counter_dict[moving_average] = num_averages + 1
        if moving_average == 'swa':
            # Stochastic weight average: a simple average of model iterates.
            new_val_weight = 1.0 / (num_averages + 1)
            for key, param in model.state_dict().items():
                # Zero-init.
                if key not in moving_average_state_dict:
                    moving_average_state_dict[key] = torch.zeros_like(
                        param.data)
                avg_buffer = moving_average_state_dict[key]
                # variable += new_val_weight * (value - variable)
                diff = (param.data - avg_buffer) * new_val_weight
                avg_buffer.add_(diff)
        else:
            # Exponential moving average of model iterates.
            # We don't correct for bias because we init the first EMA with w(0).
            new_val_weight = 1.0 - ema_decay
            for key, param in model.state_dict().items():
                # Init from the first model iterate, w(0).
                if key not in moving_average_state_dict:
                    moving_average_state_dict[key] = param.data.clone().detach()
                avg_buffer = moving_average_state_dict[key]
                # variable += new_val_weight * (value - variable)
                diff = (param.data - avg_buffer) * new_val_weight
                avg_buffer.add_(diff)

    def SwapMovingAverage(self, model, moving_average):
        """Swaps the current EMA/SWA with 'model' in-place."""
        assert moving_average in ['swa', 'ema'], moving_average
        moving_average_state_dict = self.moving_average_model_state_dict[
            moving_average]
        assert moving_average_state_dict.keys() == model.state_dict().keys()
        for key, param in model.state_dict().items():
            avg_buffer = moving_average_state_dict[key]
            tmp = torch.empty_like(param.data)
            tmp.copy_(param.data)
            param.data.copy_(avg_buffer)
            avg_buffer.copy_(tmp)

    def EvaluateTestSet(self, model, planner, tag='latency_test'):
        # TODO: exclude running time for evaluating test set.
        p = self.params
        num_iters_done = self.curr_value_iter + 1
        if p.test_query_glob is None or \
           num_iters_done < p.test_after_n_iters or \
           num_iters_done % p.test_every_n_iters != 0:
            return
        if p.test_using_retrained_model:
            print(
                '[Test set] training a new model just for test set reporting.')
            # Retrain a new 'model' and build a 'planner'.
            model, dataset = self.Train(train_from_scratch=True)
            planner = self._MakePlanner(model, dataset)
        to_execute_test, execution_results_test = self.PlanAndExecute(
            model, planner, is_test=True)
        self.LogTestExperience(to_execute_test, execution_results_test, tag=tag)

    def LogTimings(self):
        """Logs timing statistics."""
        p = self.params
        stages = ['train', 'plan', 'wait_for_executions']
        num_iters_done = self.curr_value_iter + 1
        if p.test_query_glob is not None and \
           num_iters_done >= p.test_after_n_iters and \
           num_iters_done % p.test_every_n_iters == 0:
            stages += ['plan_test_set', 'wait_for_executions_test_set']
        timings = [self.timer.GetLatestTiming(s) for s in stages]
        iter_total_s = sum(timings)
        cumulative_timings = [self.timer.GetTotalTiming(s) for s in stages]
        total_s = sum(cumulative_timings)
        data_to_log = []
        for stage, timing, cumulative_timing in zip(stages, timings,
                                                    cumulative_timings):
            data_to_log.extend([
                # Time, this iter.
                ('timing/{}'.format(stage), timing, self.curr_value_iter),
                # %Time of this iter's total.
                ('timing_pct/{}'.format(stage), timing / iter_total_s,
                 self.curr_value_iter),
                # Total time since beginning.
                ('timing_cumulative/{}'.format(stage), cumulative_timing,
                 self.curr_value_iter),
                # Total %time of all iters so far.
                ('timing_cumulative_pct/{}'.format(stage),
                 cumulative_timing / total_s, self.curr_value_iter),
            ])
        # X-axis.
        data_to_log.append(
            ('curr_value_iter', self.curr_value_iter, self.curr_value_iter))
        self.LogScalars(data_to_log)

    def Run(self):
        p = self.params
        if p.run_baseline:
            return self.RunBaseline()
        else:
            self.curr_value_iter = 0
            self.num_query_execs = 0
            self.num_total_timeouts = 0
            self.overall_best_train_latency = np.inf
            self.overall_best_test_latency = np.inf
            self.overall_best_test_swa_latency = np.inf
            self.overall_best_test_ema_latency = np.inf
            # For reporting cleaner hint strings for expert plans, remove their
            # unary ops (e.g., Aggregates).  These calls return copies, so
            # self.{all,train,test}_nodes no longer share any references.
            self.train_nodes = plans_lib.FilterScansOrJoins(self.train_nodes)
            self.test_nodes = plans_lib.FilterScansOrJoins(self.test_nodes)

        while self.curr_value_iter < p.val_iters:
            has_timeouts = self.RunOneIter()
            self.LogTimings()

            if (p.early_stop_on_skip_fraction is not None and
                    self.curr_iter_skipped_queries >=
                    p.early_stop_on_skip_fraction * len(self.train_nodes)):
                break

            if p.drop_cache and p.use_local_execution:
                print('Dropping buffer cache.')
                postgres.DropBufferCache()

            if p.increment_iter_despite_timeouts:
                # Always increment the iteration counter.  This makes it fairer
                # to compare runs with & without the timeout mechanism (or even
                # between timeout runs).
                self.curr_value_iter += 1
                self.lr_schedule.Step()
                if self.adaptive_lr_schedule is not None:
                    self.adaptive_lr_schedule.Step()
            else:
                if has_timeouts:
                    # Don't count this value iter.
                    # NOTE: it is possible for runs with use_timeout=False to
                    # have timeout events.  This can happen due to pg_executor
                    # encountering an out-of-memory / internal error and
                    # treating an execution as a timeout.
                    pass
                else:
                    self.curr_value_iter += 1
                    self.lr_schedule.Step()
                    if self.adaptive_lr_schedule is not None:
                        self.adaptive_lr_schedule.Step()


def Main(argv):
    del argv  # Unused.
    name = FLAGS.run
    print('Looking up params by name:', name)
    p = balsa.params_registry.Get(name)

    p.use_local_execution = FLAGS.local
    # Override params here for quick debugging.
    # p.sim_checkpoint = None
    # p.epochs = 1
    # p.val_iters = 0
    # p.query_glob = ['7*.sql']
    # p.test_query_glob = ['7c.sql']
    # p.search_until_n_complete_plans = 1

    agent = BalsaAgent(p)
    agent.Run()


if __name__ == '__main__':
    app.run(Main)
