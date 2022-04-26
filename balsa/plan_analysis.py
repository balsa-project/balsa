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
"""PlanAnalysis: analyzes a list of Nodes."""
import collections

import pandas as pd

from balsa.util import plans_lib


def HumanFormat(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


class Stats(object):
    """Plan statistics."""

    def __init__(self):
        self.join_counts = collections.defaultdict(int)
        self.scan_counts = collections.defaultdict(int)
        self.shape_counts = collections.defaultdict(int)
        self.nested_loop_children_counts = collections.defaultdict(int)
        self.num_plans = 0

    def Update(self, nodes):
        for node in nodes:
            join_ops, scan_ops, nl_children = self.GetOps(node)
            for op in join_ops:
                self.join_counts[op] += 1
            for op in scan_ops:
                self.scan_counts[op] += 1
            for nl_children_ops in nl_children:
                self.nested_loop_children_counts[nl_children_ops] += 1

            shape = self.GetShape(node)
            self.shape_counts[shape] += 1
        self.num_plans += len(nodes)
        return self

    def GetShape(self, node):

        def IsLeftDeep(n):
            if n.IsScan():
                return True
            if n.children[1].IsJoin():
                return False
            return IsLeftDeep(n.children[0])

        def IsRightDeep(n):
            if n.IsScan():
                return True
            if n.children[0].IsJoin():
                return False
            return IsRightDeep(n.children[1])

        if IsLeftDeep(node):
            shape = 'left_deep'
        elif IsRightDeep(node):
            shape = 'right_deep'
        else:
            shape = 'bushy'
        return shape

    def GetOps(self, node):
        join_ops, scan_ops, nl_children = [], [], []

        def Fn(n):
            if n.IsJoin():
                join_ops.append(n.node_type)
                if n.node_type == 'Nested Loop':
                    nl_children.append(
                        (n.children[0].node_type, n.children[1].node_type))
            elif n.IsScan():
                scan_ops.append(n.node_type)

        plans_lib.MapNode(node, Fn)
        return join_ops, scan_ops, nl_children

    def Print(self):
        print('Total num plans:', self.num_plans)

        def DoPrint(cnts):
            df = pd.DataFrame(cnts,
                              index=['count']).T.sort_values('count',
                                                             ascending=False)
            df['%'] = df['count'] / df['count'].sum() * 100.0
            df['%'] = df['%'].apply(lambda t: '{:.0f}%'.format(t))
            df['count'] = df['count'].apply(HumanFormat)
            print(df)
            print()

        print('Join ops')
        DoPrint(self.join_counts)
        print('Scan ops')
        DoPrint(self.scan_counts)
        print('Shapes')
        DoPrint(self.shape_counts)

        print('NL children')
        DoPrint(self.nested_loop_children_counts)


class PlanAnalysis(object):
    """Plan analysis of a list of Nodes."""

    def __init__(self):
        self.total_stats = Stats()

    @classmethod
    def Build(cls, nodes):
        analysis = PlanAnalysis()
        return analysis.Update(nodes)

    def Update(self, nodes):
        self.total_stats.Update(nodes)
        return self

    def Print(self):
        print('===== Plan Analysis =====')
        self.total_stats.Print()
