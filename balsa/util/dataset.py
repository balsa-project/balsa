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

import numpy as np
import torch


# From: https://github.com/pytorch/fairseq/blob/master/fairseq/data/data_utils.py#L29
def collate_tokens(values,
                   pad_idx,
                   eos_idx=None,
                   left_pad=False,
                   move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


class PlansDataset(torch.utils.data.Dataset):
    """A dataset of execution plans and associated costs."""

    def __init__(self,
                 query_feats,
                 plans,
                 indexes,
                 costs,
                 tree_conv=False,
                 transform_cost=True,
                 label_mean=None,
                 label_std=None,
                 cross_entropy=False,
                 return_indexes=True):
        """Dataset of plans/parent positions/costs.

        Args:
          query_feats: a list of np.ndarray (float).
          plans: a list of np.ndarray (int64).
          indexes: a list of np.ndarray (int64).
          costs: a list of floats.
          transform_cost (optional): if True, log and standardize.
        """
        assert len(plans) == len(costs) and len(query_feats) == len(plans)

        query_feats = [torch.from_numpy(xs) for xs in query_feats]
        if not tree_conv:
            plans = [torch.from_numpy(xs) for xs in plans]
            if return_indexes:
                assert len(plans) == len(indexes)
                indexes = [torch.from_numpy(xs) for xs in indexes]

        self.query_feats = query_feats
        self.plans = plans
        self.indexes = indexes

        if not isinstance(transform_cost, list):
            transform_cost = [transform_cost]
        self.transform_cost = transform_cost
        self.cross_entropy = cross_entropy
        self.return_indexes = return_indexes

        self.label_mean = label_mean
        self.label_std = label_std

        if cross_entropy:
            # Classification.

            # We don't care too much about a diff of a few millis, so scale the
            # raw millis values by this factor.
            #
            # 0.1: roughly, distinguish every 10ms.
            # 0.01: roughly, distinguish every 100ms.
            # 0.001: roughly, distinguish every second.
            self.MS_SCALE_FACTOR = 1  # millis * factor.
            self.MS_SCALE_FACTOR = 1e-2  # millis * factor.
            self.MS_SCALE_FACTOR = 1e-1  # millis * factor.

            costs = np.asarray(costs) * self.MS_SCALE_FACTOR
            # Use invertible transform on scalar x:
            #     x -> sqrt(x + 1) - 1 + transform_eps * x
            # where transform_eps is a param.
            self.TRANSFORM_EPS = 1e-3

            costs = np.sqrt(costs + 1.) - 1 + self.TRANSFORM_EPS * costs
            self.costs = torch.as_tensor(costs).to(torch.float32)
            print('transformed costs, min', costs.min(), 'max', costs.max())
        else:
            # Regression.
            for t in transform_cost:
                fn = self._transform_fn(t)
                costs = fn(costs)
            self.costs = torch.as_tensor(costs).to(torch.float)

    def _transform_fn(self, transform_name):

        def log1p(xs):
            return np.log(np.asarray(xs) + 1.0)

        def standardize(xs):
            self._EPS = 1e-6
            if self.label_mean is None:
                self.mean = np.mean(xs)
                self.std = np.std(xs)
            else:
                self.mean = self.label_mean
                self.std = self.label_std
            print('costs stats mean {} std {}'.format(self.mean, self.std))
            return (xs - self.mean) / (self.std + self._EPS)

        def min_max(xs):
            self.label_min = np.min(xs)
            self.label_max = np.max(xs)
            self.label_range = self.label_max - self.label_min
            print('costs stats min {} max {}'.format(self.label_min,
                                                     self.label_max))
            return (xs - self.label_min) / self.label_range

        transforms = {
            'log1p': log1p,
            True: standardize,
            'standardize': standardize,
            False: (lambda xs: xs),
            'min_max': min_max,
            'sqrt': lambda xs: (np.sqrt(1 + np.asarray(xs))),
        }
        return transforms[transform_name]

    def _inverse_transform_fn(self, transform_name, use_torch=False):

        def log1p_inverse(xs):
            return np.exp(xs) - 1.0

        def log1p_inverse_torch(xs):
            return torch.exp(xs) - 1.0

        def standardize_inverse(xs):
            return xs * (self.std + self._EPS) + self.mean

        def min_max_inverse(xs):
            return xs * self.label_range + self.label_min

        transforms = {
            'log1p': log1p_inverse,
            True: standardize_inverse,
            'standardize': standardize_inverse,
            False: (lambda xs: xs),
            'min_max': min_max_inverse,
            'sqrt': lambda xs: (xs**2 - 1),
        }
        if use_torch:
            transforms['log1p'] = log1p_inverse_torch
        return transforms[transform_name]

    def InvertCost(self, cost):
        """Convert model outputs back to latency space."""
        if self.cross_entropy:
            with torch.no_grad():
                softmax = torch.softmax(torch.from_numpy(cost), -1)
                expected = (torch.arange(softmax.shape[-1]) * softmax).sum(-1)

                # Inverse transform.
                # predicted_latency = ((expected + 1)**2 - 1).numpy()
                # return predicted_latency

                e = expected.numpy()
                assert self.TRANSFORM_EPS == 1e-3
                x = 1e3 * (e - np.sqrt(1e3 * e + 251001) + 501)
                return x / self.MS_SCALE_FACTOR
        else:
            for t in reversed(self.transform_cost):
                fn = self._inverse_transform_fn(t)
                cost = fn(cost)
            return cost

    def TorchInvertCost(self, cost):
        """Convert model outputs back to latency space."""
        assert not self.cross_entropy, 'Not implemented'
        for t in reversed(self.transform_cost):
            fn = self._inverse_transform_fn(t, use_torch=True)
            cost = fn(cost)
        return cost

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, idx):
        if self.return_indexes:
            return self.query_feats[idx], self.plans[idx], self.indexes[
                idx], self.costs[idx]
        return self.query_feats[idx], self.plans[idx], self.costs[idx]

    def FreeData(self):
        self.query_feats = self.plans = self.indexes = self.costs = None


class InputBatch(object):
    """Produce (plans, index, costs) mini-batches, inserting PAD tokens.

    Usage:
        loader = DataLoader(
            PlansDataset(...),
            batch_size=32,
            shuffle=True,
            collate_fn=lambda xs: InputBatch(...))
    """

    def __init__(self, data, plan_pad_idx=None, parent_pos_pad_idx=None):
        data = list(zip(*data))
        self.query_feats = torch.stack(data[0], 0)
        if plan_pad_idx is not None and parent_pos_pad_idx is not None:
            # Transformer.
            self.plans = collate_tokens(data[1], pad_idx=plan_pad_idx)
            self.indexes = collate_tokens(data[2], pad_idx=parent_pos_pad_idx)
        else:
            self.plans = torch.stack(data[1], 0)
            self.indexes = torch.stack(data[2], 0)
        self.costs = torch.stack(data[3], 0)

    def __len__(self):
        return len(self.plans)
