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
"""Cost models."""

from balsa import card_est
from balsa import hyperparams
from balsa.util import postgres


class CostModel(object):
    """Base class for a cost model."""

    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        p.Define('cost_physical_ops', False,
                 'Costs physical ops or just join orders?')
        return p

    def __init__(self, params):
        self.params = params.Copy()

    def __call__(self, node, join_conds):
        """Costs a balsa.Node with asscoiated join clauses.

        Filter information is stored in the leaf node objects and should be
        taken into account.
        """
        raise NotImplementedError('Abstract method')

    def ScoreWithSql(self, node, sql):
        """Scores a balsa.Node by using its hint_str with sql."""
        raise NotImplementedError('Abstract method')


class NullCost(CostModel):
    """Sets the cost of any plan to 0."""

    def __call__(self, node, join_conds):
        return 0

    def ScoreWithSql(self, node, sql):
        return 0


class PostgresCost(CostModel):
    """The Postgres cost model."""

    def __call__(self, node, join_conds):
        # NOTE: Postgres could fail a HashJoin hint with "SELECT * ..." but
        # accept the hint with "SELECT min(...) ...".
        #
        # For cost model learning (sim), when collecting data there is indeed a
        # difference between:
        #
        #    <query features> <subplan> <cost of the query: SELECT * ...>
        #  vs.
        #    <query features> <subplan> <cost of the query: SELECT <exprs> ...>
        #
        # In practice it should not affect the estimated costs too much.
        sql_str = node.to_sql(join_conds, with_select_exprs=True)
        return self.ScoreWithSql(node, sql_str)

    def ScoreWithSql(self, node, sql):
        p = self.params
        cost = postgres.GetCostFromPg(
            sql=sql,
            hint=node.hint_str(with_physical_hints=p.cost_physical_ops),
            check_hint_used=True,
        )
        return cost


class MinCardCost(CostModel):
    """A minimizing-cardinality cost model.

    C_out: counts intermediate number of tuples.

    This cost model ignores physical scan/join methods and is suitable for
    local join order planning.

    C(T) = |T|                    T is a base table
    C(T) = |filter(T)|            T is a base table, with filters
    C(T) = C(T1) + C(T2) + |T|    T is a join of T1 and T2

    References:
      * https://arxiv.org/pdf/2005.03328.pdf
      * Neumann et al.; 2 papers (cited above); one of them:
        * https://dl.acm.org/doi/pdf/10.1145/1559845.1559889
    """

    def __init__(self, params):
        super().__init__(params)
        self.card_est = card_est.PostgresCardEst()

    def __call__(self, node, join_conds):
        return self.Score(node, join_conds)

    def GetModelCardinality(self, node, join_conds):
        joins = node.KeepRelevantJoins(join_conds)
        if len(joins) == 0 and len(node.GetFilters()) == 0:
            return self.GetBaseRelCardinality(node)
        return self.card_est(node, joins)

    def GetBaseRelCardinality(self, node):
        assert node.table_name is not None, node
        return postgres.GetAllTableNumRows([node.table_name])[node.table_name]

    def Score(self, node, join_conds):
        if node._card:
            return node._card

        card = self.GetModelCardinality(node, join_conds)
        if node.IsScan():
            node._card = card
        else:
            assert node.IsJoin(), node
            c_t1 = self.Score(node.children[0], join_conds)
            c_t2 = self.Score(node.children[1], join_conds)
            node._card = card + c_t1 + c_t2

        return node._card
