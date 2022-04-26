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
"""PlanAnalysis: analyzes a list of Nodes.

Example usage:

    python plan_analysis.py --replay <pkl path>
"""
import collections
import gc

from absl import app
from absl import flags
from absl import logging
import joblib
import pandas as pd

from balsa import plan_analysis

FLAGS = flags.FLAGS

flags.DEFINE_string('replay', None, 'Path to a replay pickle file.')


def Main(argv):
    del argv  # Unused.
    if FLAGS.replay is None:
        return

    logging.info('Loading {}'.format(FLAGS.replay))
    gc.disable()
    initial_size, nodes = joblib.load(FLAGS.replay)
    gc.enable()
    logging.info(' ...done, initial_size={}, len={}'.format(
        initial_size, len(nodes)))

    # Analyze expert.
    logging.info('Expert:')
    pa = plan_analysis.PlanAnalysis.Build(nodes[:initial_size])
    pa.Print()

    nodes = nodes[initial_size:]
    logging.info('Dropped first initial_size nodes.')

    pa = plan_analysis.PlanAnalysis.Build(nodes)
    pa.Print()

    pa.iter_stats = collections.defaultdict(plan_analysis.Stats)
    assert len(nodes) % initial_size == 0
    num_iters = len(nodes) // initial_size
    for i in range(num_iters):
        start = i * initial_size
        end = start + initial_size
        iter_stats = plan_analysis.Stats()
        iter_stats.Update(nodes[start:end])
        pa.iter_stats[i] = iter_stats

    def ToDf(iter_idx, stats):
        return pd.DataFrame({
            'iter_double_counting_timeouts': [iter_idx],
            **stats.join_counts,
            **stats.scan_counts,
            **stats.shape_counts
        })

    dfs = [ToDf(i, stats) for i, stats in pa.iter_stats.items()]
    dfs = []
    myd = dict(pa.iter_stats)
    for i, stats in pa.iter_stats.items():
        dfs.append(ToDf(i, stats))
    df = pd.concat(dfs)
    csv_path = FLAGS.replay + '.csv'
    logging.info('Per-iter stats:\n{}'.format(str(df.head())))
    logging.info('Writing results to: {}'.format(csv_path))
    df.to_csv(csv_path, index=False, header=True)


if __name__ == '__main__':
    app.run(Main)
