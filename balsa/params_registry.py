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
"""Registry.

Example: register an experiment config.

    @balsa.params_registry.Register
    class MyConfig(object):
        def Params(self):
            # Returns a balsa.hyperparams.Params that fully specifies this
            # experiment.

Example: look up registered experiments by name.

    > p1 = balsa.params_registry.Get('PgCostJOB')
    > p2 = balsa.params_registry.Get('PgCostJOBLight')
    > print(p1.TextDiff(p2))
"""


class _RegistryHelper(object):
    """Helper class."""
    # Global dictionary mapping subclass name to registered params.
    _PARAMS = {}

    @classmethod
    def Register(cls, real_cls):
        k = real_cls.__name__
        assert k not in cls._PARAMS, '{} already registered!'.format(k)
        cls._PARAMS[k] = real_cls
        return real_cls


Register = _RegistryHelper.Register


def Get(name):
    if name not in _RegistryHelper._PARAMS:
        raise LookupError('{} not found in registered params: {}'.format(
            name, list(sorted(_RegistryHelper._PARAMS.keys()))))
    p = _RegistryHelper._PARAMS[name]().Params()
    return p


def GetAll():
    return dict(_RegistryHelper._PARAMS)
