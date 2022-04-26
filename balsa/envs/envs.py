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
"""Workload definitions."""
import glob
import os

import numpy as np

import balsa
from balsa import hyperparams
from balsa.util import plans_lib
from balsa.util import postgres

_EPSILON = 1e-6


def ParseSqlToNode(path):
    base = os.path.basename(path)
    query_name = os.path.splitext(base)[0]
    with open(path, 'r') as f:
        sql_string = f.read()
    node, json_dict = postgres.SqlToPlanNode(sql_string)
    node.info['path'] = path
    node.info['sql_str'] = sql_string
    node.info['query_name'] = query_name
    node.info['explain_json'] = json_dict
    node.GetOrParseSql()
    return node


class Workload(object):

    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        p.Define('query_dir', None, 'Directory to workload queries.')
        p.Define(
            'query_glob', '*.sql',
            'If supplied, glob for this pattern.  Otherwise, use all queries.'\
            '  Example: 29*.sql.'
        )
        p.Define(
            'loop_through_queries', False,
            'Loop through a random permutation of queries? '
            'Desirable for evaluation.')
        p.Define(
            'test_query_glob', None,
            'Similar usage as query_glob. If None, treating all queries'\
            ' as training nodes.'
        )
        p.Define('search_space_join_ops',
                 ['Hash Join', 'Merge Join', 'Nested Loop'],
                 'Join operators to learn.')
        p.Define('search_space_scan_ops',
                 ['Index Scan', 'Index Only Scan', 'Seq Scan'],
                 'Scan operators to learn.')
        return p

    def __init__(self, params):
        self.params = params.Copy()
        p = self.params
        # Subclasses should populate these fields.
        self.query_nodes = None
        self.workload_info = None
        self.train_nodes = None
        self.test_nodes = None

        if p.loop_through_queries:
            self.queries_permuted = False
            self.queries_ptr = 0

    def _ensure_queries_permuted(self, rng):
        """Permutes queries once."""
        if not self.queries_permuted:
            self.query_nodes = rng.permutation(self.query_nodes)
            self.queries_permuted = True

    def _get_sql_set(self, query_dir, query_glob):
        if query_glob is None:
            return set()
        else:
            globs = query_glob
            if type(query_glob) is str:
                globs = [query_glob]
            sql_files = np.concatenate([
                glob.glob('{}/{}'.format(query_dir, pattern))
                for pattern in globs
            ]).ravel()
        sql_files = set(sql_files)
        return sql_files

    def Queries(self, split='all'):
        """Returns all queries as balsa.Node objects."""
        assert split in ['all', 'train', 'test'], split
        if split == 'all':
            return self.query_nodes
        elif split == 'train':
            return self.train_nodes
        elif split == 'test':
            return self.test_nodes

    def WithQueries(self, query_nodes):
        """Replaces this Workload's queries with 'query_nodes'."""
        self.query_nodes = query_nodes
        self.workload_info = plans_lib.WorkloadInfo(query_nodes)

    def FilterQueries(self, query_dir, query_glob, test_query_glob):
        all_sql_set_new = self._get_sql_set(query_dir, query_glob)
        test_sql_set_new = self._get_sql_set(query_dir, test_query_glob)
        assert test_sql_set_new.issubset(all_sql_set_new), (test_sql_set_new,
                                                            all_sql_set_new)

        all_sql_set = set([n.info['path'] for n in self.query_nodes])
        assert all_sql_set_new.issubset(all_sql_set), (
            'Missing nodes in init_experience; '
            'To fix: remove data/initial_policy_data.pkl, or see README.')

        query_nodes_new = [
            n for n in self.query_nodes if n.info['path'] in all_sql_set_new
        ]
        train_nodes_new = [
            n for n in query_nodes_new
            if test_query_glob is None or n.info['path'] not in test_sql_set_new
        ]
        test_nodes_new = [
            n for n in query_nodes_new if n.info['path'] in test_sql_set_new
        ]
        assert len(train_nodes_new) > 0

        self.query_nodes = query_nodes_new
        self.train_nodes = train_nodes_new
        self.test_nodes = test_nodes_new

    def UseDialectSql(self, p):
        dialect_sql_dir = p.engine_dialect_query_dir
        for node in self.query_nodes:
            assert 'sql_str' in node.info and 'query_name' in node.info
            path = os.path.join(dialect_sql_dir,
                                node.info['query_name'] + '.sql')
            assert os.path.isfile(path), '{} does not exist'.format(path)
            with open(path, 'r') as f:
                dialect_sql_string = f.read()
            node.info['sql_str'] = dialect_sql_string


class JoinOrderBenchmark(Workload):

    @classmethod
    def Params(cls):
        p = super().Params()
        # Needs to be an absolute path for rllib.
        module_dir = os.path.abspath(os.path.dirname(balsa.__file__) + '/../')
        p.query_dir = os.path.join(module_dir, 'queries/join-order-benchmark')
        return p

    def __init__(self, params):
        super().__init__(params)
        p = params
        self.query_nodes, self.train_nodes, self.test_nodes = \
            self._LoadQueries()
        self.workload_info = plans_lib.WorkloadInfo(self.query_nodes)
        self.workload_info.SetPhysicalOps(p.search_space_join_ops,
                                          p.search_space_scan_ops)

    def _LoadQueries(self):
        """Loads all queries into balsa.Node objects."""
        p = self.params
        all_sql_set = self._get_sql_set(p.query_dir, p.query_glob)
        test_sql_set = self._get_sql_set(p.query_dir, p.test_query_glob)
        assert test_sql_set.issubset(all_sql_set)
        # sorted by query id for easy debugging
        all_sql_list = sorted(all_sql_set)
        all_nodes = [ParseSqlToNode(sqlfile) for sqlfile in all_sql_list]

        train_nodes = [
            n for n in all_nodes
            if p.test_query_glob is None or n.info['path'] not in test_sql_set
        ]
        test_nodes = [n for n in all_nodes if n.info['path'] in test_sql_set]
        assert len(train_nodes) > 0

        return all_nodes, train_nodes, test_nodes


class RunningStats(object):
    """Computes running mean and standard deviation.

    Usage:
        rs = RunningStats()
        for i in range(10):
            rs.Record(np.random.randn())
        print(rs.Mean(), rs.Std())
    """

    def __init__(self, n=0., m=None, s=None):
        self.n = n
        self.m = m
        self.s = s

    def Record(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = 0.
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)

    def Mean(self):
        return self.m if self.n else 0.0

    def Variance(self):
        return self.s / (self.n) if self.n else 0.0

    def Std(self, epsilon_guard=True):
        eps = 1e-6
        std = np.sqrt(self.Variance())
        if epsilon_guard:
            return np.maximum(eps, std)
        return std
