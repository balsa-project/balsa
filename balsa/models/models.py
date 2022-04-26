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
from torch import nn

_ACTIVATIONS = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
}


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


def MakeMlp(input_size, num_outputs, hiddens, activation):
    layers = []
    prev_layer_size = input_size
    for size in hiddens:
        layers.append(nn.Linear(prev_layer_size, size))
        layers.append(_ACTIVATIONS[activation]())
        prev_layer_size = size
    # Output layer.
    layers.append(nn.Linear(prev_layer_size, num_outputs))
    return nn.Sequential(*layers)
