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
"""Balsa experiment configs.

See README.md for the main configurations to run.
"""
import balsa
from balsa import hyperparams

# 19 most slow-running queries sorted by Postgres latency.
SLOW_TEST_QUERIES = [
    '16b.sql', '17a.sql', '17e.sql', '17f.sql', '17b.sql', '19d.sql', '17d.sql',
    '17c.sql', '10c.sql', '26c.sql', '25c.sql', '6d.sql', '6f.sql', '8c.sql',
    '18c.sql', '9d.sql', '30a.sql', '19c.sql', '20a.sql'
]

# A random split using seed 52.  Test latency is chosen to be close to the
# bootstrapped mean.
RAND_52_TEST_QUERIES = [
    '8a.sql', '16a.sql', '2a.sql', '30c.sql', '17e.sql', '20a.sql', '26b.sql',
    '12b.sql', '15b.sql', '15d.sql', '10b.sql', '15a.sql', '4c.sql', '4b.sql',
    '22b.sql', '17c.sql', '24b.sql', '10a.sql', '22c.sql'
]

LR_SCHEDULES = {
    'C': {
        'lr_piecewise': [
            (0, 0.001),
            (50, 0.0005),
            (100, 0.00025),
            (150, 0.000125),
            (200, 0.0001),
        ]
    },
    # Delay C's decay by 10 iters.
    'C10': {
        'lr_piecewise': [
            (0, 0.001),
            (50 + 10, 0.0005),
            (100 + 10, 0.00025),
            (150 + 10, 0.000125),
            (200 + 10, 0.0001),
        ]
    },
}


class BalsaParams(object):
    """Params for run.BalsaAgent."""

    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        p.Define('db', 'imdbload', 'Name of the Postgres database.')
        p.Define('query_dir', 'queries/join-order-benchmark',
                 'Directory of the .sql queries.')
        p.Define(
            'query_glob', '*.sql',
            'If supplied, glob for this pattern. Otherwise, use all queries. Example: 29*.sql.'
        )
        p.Define(
            'test_query_glob', None,
            'Similar usage as query_glob. If None, treat all queries as training nodes.'
        )
        p.Define('engine', 'postgres',
                 'The execution engine.  Options: postgres.')
        p.Define('engine_dialect_query_dir', None,
                 'Directory of the .sql queries in target engine\'s dialect.')
        p.Define('run_baseline', False,
                 'If true, just load the queries and run them.')
        p.Define(
            'drop_cache', True,
            'If true, drop the buffer cache at the end of each value iteration.'
        )
        p.Define(
            'plan_physical', True,
            'If true, plans physical scan/join operators.  '\
            'Otherwise, just join ordering.'
        )
        p.Define('cost_model', 'postgrescost',
                 'A choice of postgrescost, mincardcost.')
        p.Define('bushy', True, 'Plans bushy query execution plans.')
        p.Define('search_space_join_ops',
                 ['Hash Join', 'Merge Join', 'Nested Loop'],
                 'Action space: join operators to learn and use.')
        p.Define('search_space_scan_ops',
                 ['Index Scan', 'Index Only Scan', 'Seq Scan'],
                 'Action space: scan operators to learn and use.')

        # LR.
        p.Define('lr', 1e-3, 'Learning rate.')
        p.Define('lr_decay_rate', None, 'If supplied, use ExponentialDecay.')
        p.Define('lr_decay_iters', None, 'If supplied, use ExponentialDecay.')
        p.Define('lr_piecewise', None, 'If supplied, use Piecewise.  Example:'\
                 '[(0, 1e-3), (200, 1e-4)].')
        p.Define('use_adaptive_lr', None, 'Experimental.')
        p.Define('use_adaptive_lr_decay_to_zero', None, 'Experimental.')
        p.Define('final_decay_rate', None, 'Experimental.')
        p.Define('linear_decay_to_zero', False,
                 'Linearly decay from lr to 0 in val_iters.')
        p.Define('reduce_lr_within_val_iter', False,
                 'Reduce LR within each val iter?')

        # Training.
        p.Define('inherit_optimizer_state', False, 'Experimental.  For Adam.')
        p.Define('epochs', 100, 'Num epochs to train.')
        p.Define('bs', 1024, 'Batch size.')
        p.Define('val_iters', 500, '# of value iterations.')
        p.Define('increment_iter_despite_timeouts', False,
                 'Increment the iteration counter even if timeouts occurred?')
        p.Define('loss_type', None, 'Options: None (MSE), mean_qerror.')
        p.Define('cross_entropy', False, 'Use cross entropy loss formulation?')
        p.Define('l2_lambda', 0, 'L2 regularization lambda.')
        p.Define('adamw', None,
                 'If not None, the weight_decay param for AdamW.')
        p.Define('label_transforms', ['log1p', 'standardize'],
                 'Transforms for labels.')
        p.Define('label_transform_running_stats', False,
                 'Use running mean and std to standardize labels?'\
                 '  May affect on-policy.')
        p.Define('update_label_stats_every_iter', True,
                 'Update mean/std stats of labels every value iteration?  This'\
                 'means the scaling of the prediction targers will shift.')
        p.Define('gradient_clip_val', 0, 'Clip the gradient norm computed over'\
                 ' all model parameters together. 0 means no clipping.')
        p.Define('early_stop_on_skip_fraction', None,
                 'If seen plans for x% of train queries produced, early stop.')
        # Validation.
        p.Define('validate_fraction', 0.1,
                 'Sample this fraction of the dataset as the validation set.  '\
                 '0 to disable validation.')
        p.Define('validate_every_n_epochs', 5,
                 'Run validation every this many training epochs.')
        p.Define(
            'validate_early_stop_patience', 3,
            'Number of validations with no improvements before early stopping.'\
            '  Thus, the maximum # of wasted train epochs = '\
            'this * validate_every_n_epochs).'
        )
        # Testing.
        p.Define('test_every_n_iters', 1,
                 'Run test set every this many value iterations.')
        p.Define('test_after_n_iters', 0,
                 'Start running test set after this many value iterations.')
        p.Define('test_using_retrained_model', False,
                 'Whether to retrain a model from scratch just for testing.')
        p.Define('track_model_moving_averages', False,
                 'Track EMA/SWA of the agent?')
        p.Define('ema_decay', 0.95, 'Use an EMA model to evaluate on test.')

        # Pre-training.
        p.Define('sim', True, 'Initialize from a pre-trained SIM model?')
        p.Define('finetune_out_mlp_only', False, 'Freeze all but out_mlp?')
        p.Define(
            'sim_checkpoint', None,
            'Path to a pretrained SIM checkpoint.  Load it instead '
            'of retraining.')
        p.Define(
            'param_noise', 0.0,
            'If non-zero, add Normal(0, std=param_noise) to Linear weights '\
            'of the pre-trained net.')
        p.Define(
            'param_tau', 1.0,
            'If non-zero, real_model_t = tau * real_model_tm1 + (1-tau) * SIM.')
        p.Define(
            'use_ema_source', False,
            'Use an exponential moving average of source networks?  If so, tau'\
            ' is used as model_t := source_t :='\
            ' tau * source_(t-1) + (1-tau) * model_(t-1).'
        )
        p.Define(
            'skip_sim_init_iter_1p', False,
            'Starting from the 2nd iteration, skip initializing from '\
            'simulation model?'
        )
        p.Define(
            'generic_ops_only_for_min_card_cost', False,
            'This affects sim model training and only if MinCardCost is used. '\
            'See sim.py for documentation.')
        p.Define(
            'sim_data_collection_intermediate_goals', True,
            'This affects sim model training.  See sim.py for documentation.')

        # Training data / replay buffer.
        p.Define(
            'init_experience', 'data/initial_policy_data.pkl',
            'Initial data set of query plans to learn from. By default, this'\
            ' is the expert optimizer experience collected when baseline'\
            ' performance is evaluated.'
        )
        p.Define('skip_training_on_expert', True,
                 'Whether to skip training on expert plan-latency pairs.')
        p.Define(
            'dedup_training_data', True,
            'Whether to deduplicate training data by keeping the best cost per'\
            ' subplan per template.'
        )
        p.Define('on_policy', False,
                 'Whether to train on only data from the latest iteration.')
        p.Define(
            'use_last_n_iters', -1,
            'Train on data from this many latest iterations.  If on_policy,'\
            ' this flag is ignored and treated as 1 (latest iter).  -1 means'\
            ' train on all previous iters.')
        p.Define('skip_training_on_timeouts', False,
                 'Skip training on executions that were timeout events?')
        p.Define(
            'use_new_data_only', False,
            'Experimental; has effects if on_policy or use_last_n_iters > 0.'\
            '  Currently only implemented in the dedup_training_data branch.')
        p.Define(
            'per_transition_sgd_steps', -1, '-1 to disable.  Takes effect only'\
            ' for when p.use_last_n_iters>0 and p.epochs=1.  This controls the'\
            ' average number of SGD updates taken on each transition.')
        p.Define('physical_execution_hindsight', False,
                 'Apply hindsight labeling to physical execution data?')
        p.Define(
            'replay_buffer_reset_at_iter', None,
            'If specified, clear all agent replay data at this iteration.')
        # Offline replay.
        p.Define(
            'prev_replay_buffers_glob', None,
            'If specified, load previous replay buffers and merge them as training purpose.'
        )
        p.Define(
            'prev_replay_buffers_glob_val', None,
            'If specified, load previous replay buffers and merge them as validation purpose.'
        )
        p.Define(
            'agent_checkpoint', None,
            'Path to a pretrained agent checkpoint.  Load it instead '
            'of retraining.')
        p.Define('prev_replay_keep_last_fraction', 1,
                 'Keep the last fraction of the previous replay buffers.')
        # Modeling: tree convolution (suggested).
        p.Define('tree_conv', True,
                 'If true, use tree convolutional neural net.')
        p.Define('tree_conv_version', None, 'Options: None.')
        p.Define('sim_query_featurizer', True,
                 'If true, use SimQueryFeaturizer to produce query features.')
        # Featurization.
        p.Define('perturb_query_features', None,
                 'If not None, randomly perturb query features on each forward'\
                 ' pass, and this flag specifies '\
                 '(perturb_prob_per_table, [scale_min, scale_max]).  '\
                 'A multiplicative scale is drawn from '\
                 'Unif[scale_min, scale_max].  Only performed when training '\
                 'and using a query featurizer with perturbation implemented.')

        # Modeling: Transformer (deprecated).  Enabled when tree_conv is False.
        p.Define('v2', True, 'If true, use TransformerV2.')
        p.Define('pos_embs', True, 'Use positional embeddings?')
        p.Define('dropout', 0.0, 'Dropout prob for transformer stack.')

        # Inference.
        p.Define('check_hint', True, 'Check hints are respected?')
        p.Define('beam', 20, 'Beam size.')
        p.Define(
            'search_method', 'beam_bk',
            'Algorithm used to search for execution plans with cost model.')
        p.Define(
            'search_until_n_complete_plans', 10,
            'Keep doing plan search for each query until this many complete'\
            ' plans have been found.  Returns the predicted cheapest one out'\
            ' of them.  Recommended: 10.')
        p.Define('planner_config', None, 'See optimizer.py#PlannerConfig.')
        p.Define(
            'avoid_eq_filters', False,
            'Avoid certain equality filters during planning (required for Ext-JOB).'
        )

        p.Define('sim_use_plan_restrictions', True, 'Experimental.')
        p.Define('real_use_plan_restrictions', True, 'Experimental.')

        # Exploration during inference.
        p.Define(
            'epsilon_greedy', 0,
            'Epsilon-greedy policy: with epsilon probability, execute a'\
            ' randomly picked plan out of all complete plans found, rather'\
            ' than the predicted-cheapest one out of them.')
        p.Define('epsilon_greedy_random_transform', False,
                 'Apply eps-greedy to randomly transform the best found plan?')
        p.Define('epsilon_greedy_random_plan', False,
                 'Apply eps-greedy to randomly pick a plan?')
        p.Define('epsilon_greedy_within_beam_search', False,
                 'Apply eps-greedy to within beam search?')
        p.Define('explore_soft_v', False,
                 'Sample an action from the soft V-distribution?')
        p.Define('explore_visit_counts', False, 'Explores using a visit count?')
        p.Define('explore_visit_counts_sort', False,
                 'Explores by executing the plan with the smallest '\
                 '(visit count, predicted latency) out of k-best plans?')
        p.Define('explore_visit_counts_latency_sort', False,
                 'Explores using explore_visit_counts_sort if there exists '\
                 'a plan that has a 0 visit count. Else sorts by predicted latency.')

        # Safe execution.
        p.Define('use_timeout', True, 'Use a timeout safeguard?')
        p.Define('initial_timeout_ms', None, 'Timeout for iter 0 if not None.')
        p.Define('special_timeout_label', True,
                 'Use a constant timeout label (4096 sec)?')
        p.Define('timeout_slack', 2,
                 'A multiplier: timeout := timeout_slack * max_query_latency.')
        p.Define('relax_timeout_factor', None,
                 'If not None, a positive factor to multiply with the current'\
                 ' timeout when relaxation conditions are met.')
        p.Define('relax_timeout_on_n_timeout_iters', None,
                 'If there are this many timeout iterations up to now, relax'\
                 ' the current timeout by relax_timeout_factor.')

        # Execution.
        p.Define('use_local_execution', False,
                 'For query executions, connect to local engine or the remote'\
                 ' cluster?  Non-execution EXPLAINs are always issued to'\
                 ' local.')
        p.Define('use_cache', True, 'Skip executing seen plans?')
        return p


########################## Baselines ##########################


@balsa.params_registry.Register
class Baseline(BalsaParams):

    def Params(self):
        p = super().Params()
        p.run_baseline = True
        return p


@balsa.params_registry.Register
class BaselineExtJOB(Baseline):

    def Params(self):
        p = super().Params()
        p.query_glob = ['*.sql']
        p.query_dir = 'queries/join-order-benchmark-extended'
        p.test_query_glob = ['e*.sql']
        return p


########################## Main Balsa agents ##########################


@balsa.params_registry.Register
class MinCardCost(BalsaParams):

    def Params(self):
        p = super().Params()
        p.cost_model = 'mincardcost'
        p.sim_checkpoint = None
        # Exploration schemes.
        p.explore_visit_counts = True
        return p


@balsa.params_registry.Register
class MinCardCostSortCnts(MinCardCost):

    def Params(self):
        return super().Params().Set(
            explore_visit_counts=False,
            explore_visit_counts_sort=True,
        )


@balsa.params_registry.Register
class MinCardCostOnPol(MinCardCostSortCnts):

    def Params(self):
        p = super().Params()

        from_p = BalsaParams().Params()
        from_p.cost_model = 'mincardcost'
        from_p.query_glob = ['*.sql']
        from_p.test_query_glob = 'TODO: Subclasses should fill this.'
        from_p.sim_checkpoint = None
        # Exploration schemes.
        from_p.explore_visit_counts = False
        from_p.explore_visit_counts_sort = True

        p = hyperparams.CopyFieldsTo(from_p, p)
        return p.Set(on_policy=True)


@balsa.params_registry.Register
class Rand52MinCardCostOnPol(MinCardCostOnPol):

    def Params(self):
        p = super().Params()
        p.test_query_glob = RAND_52_TEST_QUERIES
        p.sim_checkpoint = 'checkpoints/sim-MinCardCost-rand52split-680secs.ckpt'
        return p


@balsa.params_registry.Register
class Rand52MinCardCostOnPolLrC(Rand52MinCardCostOnPol):

    def Params(self):
        return super().Params().Set(**LR_SCHEDULES['C'])


@balsa.params_registry.Register  # keep
class Balsa_JOBRandSplit(Rand52MinCardCostOnPolLrC):

    def Params(self):
        p = super().Params()
        p.increment_iter_despite_timeouts = True
        p = p.Set(**LR_SCHEDULES['C10'])
        return p


@balsa.params_registry.Register
class Balsa_JOBRandSplitReplay(Balsa_JOBRandSplit):  # keep

    def Params(self):
        p = super().Params()

        p.validate_fraction = 0.1
        p.epochs = 100
        p.lr_piecewise = [(0, 5 * 1e-4), (1, 1e-4)]

        p.val_iters = 100
        p.validate_every_n_epochs = 1
        p.validate_early_stop_patience = 5

        # Change path to point to the desired buffers:
        p.prev_replay_buffers_glob = './data/replay-Balsa_JOBRandSplit-*'
        # Choose one buffer as a hold-out validation set if desired:
        p.prev_replay_buffers_glob_val = None
        p.skip_training_on_timeouts = False
        return p


@balsa.params_registry.Register
class SlowMinCardCost(MinCardCostOnPol):

    def Params(self):
        p = super().Params()
        p.use_timeout = True
        p.test_query_glob = SLOW_TEST_QUERIES
        p.sim_checkpoint = 'checkpoints/sim-MinCardCost-slowsplit-610secs.ckpt'
        # ExponentialDecay(2e-3, 0.1, 100).
        p = p.Set(lr=2e-3, lr_decay_rate=0.1, lr_decay_iters=100)
        return p


@balsa.params_registry.Register
class SlowMinCardCostNoDecay(SlowMinCardCost):

    def Params(self):
        # No decay.
        return super().Params().Set(lr=1e-3,
                                    lr_decay_rate=None,
                                    lr_decay_iters=None)


@balsa.params_registry.Register
class SlowMinCardCostLrC(SlowMinCardCostNoDecay):

    def Params(self):
        return super().Params().Set(**LR_SCHEDULES['C'])


@balsa.params_registry.Register  # keep
class Balsa_JOBSlowSplit(SlowMinCardCostLrC):

    def Params(self):
        p = super().Params()
        p.increment_iter_despite_timeouts = True
        p = p.Set(**LR_SCHEDULES['C10'])
        return p


@balsa.params_registry.Register
class Balsa_JOBSlowSplitReplay(Balsa_JOBSlowSplit):  # keep

    def Params(self):
        p = super().Params()

        p.validate_fraction = 0.1
        p.epochs = 100
        p.lr_piecewise = [(0, 5 * 1e-4), (1, 1e-4)]

        p.val_iters = 100
        p.validate_every_n_epochs = 1
        p.validate_early_stop_patience = 5

        # Change path to point to the desired buffers:
        p.prev_replay_buffers_glob = './data/replay-Balsa_JOBSlowSplit-*'
        # Choose one buffer as a hold-out validation set if desired:
        p.prev_replay_buffers_glob_val = None
        p.skip_training_on_timeouts = False
        return p


########################## Generalizing to Ext-JOB ##########################


@balsa.params_registry.Register
class ExtJOBMinCardCostOnPol(Rand52MinCardCostOnPol):

    def Params(self):
        p = super().Params()
        p.query_dir = 'queries/join-order-benchmark-extended'
        p.query_glob = ['*.sql']
        p.test_query_glob = ['e*.sql']

        p.sim_checkpoint = ('checkpoints/' +
                            'sim-MinCardCost-train113JOB-784s-13epochs.ckpt')

        # Save some time.
        p.test_after_n_iters = 80
        p.test_every_n_iters = 1

        # This required so plan hints do not fail for certain queries
        # in Ext-JOB that contain equality filters.
        p.avoid_eq_filters = True
        return p


@balsa.params_registry.Register  # keep
class Balsa_TrainJOB_TestExtJOB(ExtJOBMinCardCostOnPol):

    def Params(self):
        return super().Params().Set(inherit_optimizer_state=True)


@balsa.params_registry.Register  # keep
class Balsa1x_TrainJOB_TestExtJOB(Balsa_TrainJOB_TestExtJOB):

    def Params(self):
        p = super().Params()
        p.validate_every_n_epochs = 1
        p.validate_early_stop_patience = 5
        p.lr_piecewise = [(0, 1e-3), (1, 1e-4)]
        # Change path to point to the desired buffers:
        p.prev_replay_buffers_glob = './replays/EXTJOB/train/replay-Balsa_TrainJOB_TestExtJOB-<fill_me_in>.pkl'
        p.test_after_n_iters = 0
        p.test_every_n_iters = 1
        # Disable timeout for further exploration
        p.use_timeout = False
        return p


@balsa.params_registry.Register  # keep
class Balsa8x_TrainJOB_TestExtJOB(Balsa1x_TrainJOB_TestExtJOB):

    def Params(self):
        p = super().Params()
        # Change paths to point to the desired buffers:
        p.prev_replay_buffers_glob = './replays/EXTJOB/train/*pkl'
        # Choose one buffer as a hold-out validation set if desired:
        p.prev_replay_buffers_glob_val = './replays/EXTJOB/val/*pkl'
        return p


########################## Neo-impl experiments ##########################


@balsa.params_registry.Register
class NeoImplRand52(BalsaParams):

    def Params(self):
        p = super().Params()
        p.query_glob = ['*.sql']
        p.test_query_glob = RAND_52_TEST_QUERIES

        p.test_every_n_iters = 1

        # Algorithmic choices below.

        # No simulator.
        p.sim = False
        p.sim_checkpoint = None

        # Off-policy, retrain always.
        p.on_policy = False
        p.param_tau = 1

        # Use demonstrations from expert.
        p.skip_training_on_expert = False

        # No timeouts.
        p.use_timeout = False
        # NOTE: this flag is necessary because there could be some 'treating as
        # timeouts' feedback even when use_timeout is set to False.  In those
        # cases, skip training on those labels.
        p.skip_training_on_timeouts = True

        return p


@balsa.params_registry.Register
class NeoImplRand52Reset(NeoImplRand52):

    def Params(self):
        return super().Params().Set(param_tau=0.0)


@balsa.params_registry.Register  # keep
class NeoImpl_JOBRandSplit(NeoImplRand52Reset):

    def Params(self):
        return super().Params().Set(dedup_training_data=False)


########################## Ablation: sim ##########################


@balsa.params_registry.Register  # keep
class JOBRandSplit_NoSim(Balsa_JOBRandSplit):

    def Params(self):
        # TODO: there's a minor bug in this code path:
        #   iter 0: rand-init weights W0
        #           plan & execute
        #
        #   iter 1: rand-init weights W1
        #           train W1 on iter 0's data
        #           plan & execute
        #
        #   iter 2: use last iter's updated weights (self.model)
        #           train, plan, execute
        #
        # The minor bug is that iter 1 should train W0 rather than
        # re-initializes the weights.  The code is currently not set up to do
        # that.
        #
        # This is very minor and we should spend time on other things.
        p = super().Params()
        # No simulator.
        p.sim = False
        p.sim_checkpoint = None
        return p


@balsa.params_registry.Register  # keep
class JOBRandSplit_PostgresSim(Balsa_JOBRandSplit):

    def Params(self):
        p = super().Params()
        # Use PostgresCost as the simulator.
        p.cost_model = 'postgrescost'
        p.sim_checkpoint = 'checkpoints/sim-PostgresCost-rand52split-26epochs.ckpt'
        return p


########################## Ablation: timeouts ##########################


@balsa.params_registry.Register  # keep
class JOBRandSplit_NoTimeout(Balsa_JOBRandSplit):

    def Params(self):
        p = super().Params()
        # No timeouts.
        p.use_timeout = False
        # NOTE: this flag is necessary because there could be some 'treating as
        # timeouts' feedback even when use_timeout is set to False.  In those
        # cases, skip training on those labels.
        p.skip_training_on_timeouts = True
        return p


########################## Ablation: Training scheme ##########################


@balsa.params_registry.Register  # keep
class JOBRandSplit_RetrainScheme(Balsa_JOBRandSplit):

    def Params(self):
        p = super().Params()
        # Off pol, tau=0, no inherit, no LR decay.
        p.on_policy = False
        p.param_tau = 0
        p.lr_piecewise = None
        return p


######################### Ablation: exploration #########################


@balsa.params_registry.Register  # keep
class JOBRandSplit_EpsGreedy(Balsa_JOBRandSplit):

    def Params(self):
        p = super().Params()
        p.explore_visit_counts_sort = False
        p.epsilon_greedy_within_beam_search = 0.0025
        return p


@balsa.params_registry.Register  # keep
class JOBRandSplit_NoExplore(Balsa_JOBRandSplit):

    def Params(self):
        p = super().Params()
        p.explore_visit_counts_sort = False
        return p
