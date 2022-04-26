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

from balsa.util import postgres


class CardEst(object):
    """Base class for cardinality estimators."""

    def __call__(self, node, join_conds):
        raise NotImplementedError()


class PostgresCardEst(CardEst):

    def __init__(self):
        self._cache = {}

    def _HashKey(self, node):
        """Computes a hash key based on the logical contents of 'node'.

        Specifically, hash on the sorted sets of table IDs and their filters.

        NOTE: Postgres can produce slightly different cardinality estimates
        when all being equal but just the FROM list ordering tables
        differently.  Here, we ignore this slight difference.
        """
        sorted_filters = '\n'.join(sorted(node.GetFilters()))
        sorted_leaves = '\n'.join(sorted(node.leaf_ids()))
        return sorted_leaves + sorted_filters

    def __call__(self, node, join_conds):
        key = self._HashKey(node)
        card = self._cache.get(key)
        if card is None:
            sql_str = node.to_sql(join_conds)
            card = postgres.GetCardinalityEstimateFromPg(sql=sql_str)
            self._cache[key] = card
        return card
