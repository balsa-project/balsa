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
"""Utility functions for training."""

import collections
import time

import numpy as np
import torch


def LoadBestCheckpointForEval(model, trainer):
    """Loads the checkpoint with the best validation loss."""
    best_path = trainer.checkpoint_callback.best_model_path
    if len(best_path) > 0:
        print('Loading best checkpoint: {} (current_epoch={})'.format(
            best_path, trainer.current_epoch))
        old_sum = sum(
            [weight.sum().item() for _, weight in model.named_parameters()])
        ckpt = torch.load(best_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['state_dict'])
        new_sum = sum(
            [weight.sum().item() for _, weight in model.named_parameters()])
        best_epoch = best_path.split('epoch=')[-1].split('.')[0]
        if '_v' in best_epoch:
            # E.g., '99_v0'
            best_epoch = best_epoch.split('_v')[0]
        best_epoch = int(best_epoch)
    else:
        print('No best checkpoint found (run validaiton yet?); '\
              'model left unchanged.')


def QErrorLoss(outputs, targets):
    return torch.where(outputs > targets, outputs / targets,
                       targets / outputs).mean()


def GetLrSchedule(p):
    if p.lr_decay_rate is not None:
        assert p.lr_decay_iters is not None
        return ExponentialDecay(init_value=p.lr,
                                decay_rate=p.lr_decay_rate,
                                decay_steps=p.lr_decay_iters)
    if p.lr_piecewise is not None:
        return Piecewise(p.lr_piecewise, final_decay_rate=p.final_decay_rate)
    return Constant(p.lr)


class Schedule(object):
    """A schedule of a variable (e.g., learning rate; epsilon)."""

    def Get(self):
        """Returns the value at the current step."""
        raise NotImplementedError

    def GetStep(self, step):
        """Returns the value at a specific step."""
        raise NotImplementedError

    def Step(self):
        """Increments the "step" count."""
        raise NotImplementedError


class Constant(Schedule):
    """A constant value for every step."""

    def __init__(self, value):
        self.value = value

    def Get(self):
        return self.value

    def GetStep(self, step):
        return self.value

    def Step(self):
        pass


class ExponentialDecay(Schedule):
    """Value at time t = init * (decay_rate ** (t / decay_steps)).

    I.e., over each decay_steps, decay the starting value by decay_rate.
    """

    def __init__(self, init_value, decay_rate, decay_steps):
        self.init_value = init_value
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.curr_step = 0

    def Get(self):
        return self.GetStep(self.curr_step)

    def GetStep(self, step):
        return self.init_value * self.decay_rate**(step / self.decay_steps)

    def Step(self):
        self.curr_step += 1


class Piecewise(Schedule):
    """Piecewise schedule given a [(start_step, value)] list.

    Example:
        # Step [0, 200): value 1e-3.
        # Step [200, ...): value 1e-4.
        >>> s = Piecewise([(0, 1e-3), (200, 1e-4)])
        >>> s.GetStep(0)
        0.001
        >>> s.GetStep(199)
        0.001
        >>> s.GetStep(200)
        0.0001
        >>> s.GetStep(2000)
        0.0001
    """

    def __init__(self, schedule_values, final_decay_rate=None):
        start_steps = [t[0] for t in schedule_values]
        assert start_steps == sorted(start_steps)
        assert len(start_steps) >= 1, schedule_values
        self.schedule_values = schedule_values
        self.curr_step = 0
        # For final stage: if not None, exponentially decay according to this
        # rate.
        self.final_decay_rate = final_decay_rate
        self.last_stage_start_t = None

    def Get(self):
        return self.GetStep(self.curr_step)

    def GetStep(self, step):
        if len(self.schedule_values) == 1:
            # Treat as constant LR.
            return self.schedule_values[0][1]
        for (l_step, l), (r_step, r) in zip(self.schedule_values[:-1],
                                            self.schedule_values[1:]):
            if l_step <= step < r_step:
                return l
        # Last stage.
        if self.last_stage_start_t is None:
            self.last_stage_start_t = self.curr_step
        # Optionally, exponential decay.
        if self.final_decay_rate:
            t = self.curr_step - self.last_stage_start_t
            lr_start = self.schedule_values[-1][1]
            return lr_start * (self.final_decay_rate**t)
        return r

    def Step(self):
        self.curr_step += 1


class AdaptiveMetricPiecewise(Piecewise):

    def __init__(self, schedule_values, metric_max_value):
        super().__init__(schedule_values)
        starts = [t[0] for t in schedule_values]
        assert starts == sorted(starts)
        assert len(starts) >= 1, schedule_values
        self.schedule_values = [
            (t[0] * metric_max_value, t[1]) for t in schedule_values
        ]
        self.metric_max_value = metric_max_value
        self.metric_value = 0

    def Get(self):
        return self.GetStep(self.metric_value)

    def SetOrTakeMax(self, value):
        if self.metric_value is None or value > self.metric_value:
            self.metric_value = value


class AdaptiveMetricPiecewiseDecayToZero(AdaptiveMetricPiecewise):

    def __init__(self,
                 schedule_values,
                 metric_max_value,
                 total_steps,
                 final_decay_rate=None):
        super().__init__(schedule_values, metric_max_value)

        self.last_stage_start_t = None
        self.total_steps = total_steps
        # For final stage: if not None, linearly decay; otherwise exponentially
        # decay according to this rate.
        self.final_decay_rate = final_decay_rate

    def Get(self):
        if self.metric_value < self.schedule_values[-1][0]:
            return self.GetStep(self.metric_value)
        # Last stage: decay to near zero based on 'step'.
        if self.last_stage_start_t is None:
            self.last_stage_start_t = self.curr_step

        t = self.curr_step - self.last_stage_start_t
        T = self.total_steps - self.last_stage_start_t - 1
        lr_start = self.schedule_values[-1][1]

        if self.final_decay_rate is not None:
            return lr_start * (self.final_decay_rate**t)

        return lr_start * (1 - t / T)


class Timer(object):
    """Simple stage-based timer."""

    def __init__(self):
        self.curr_stage = None
        self.curr_stage_start = None
        # stage -> a list of timings.
        self.stage_timing_dict = collections.defaultdict(list)

    def Start(self, stage):
        assert self.curr_stage is None, 'Forgot to call Stop()?'
        self.curr_stage_start = time.time()
        self.curr_stage = stage

    def Stop(self, stage):
        assert self.curr_stage == stage, 'curr_stage={} != {}'.format(
            self.curr_stage, stage)
        self.stage_timing_dict[stage].append(time.time() -
                                             self.curr_stage_start)
        self.curr_stage = None

    def GetLatestTiming(self, stage):
        assert self.stage_timing_dict[stage], 'No timing info yet: ' + stage
        return self.stage_timing_dict[stage][-1]

    def GetTotalTiming(self, stage):
        assert self.stage_timing_dict[stage], 'No timing info yet: ' + stage
        return sum(self.stage_timing_dict[stage])
