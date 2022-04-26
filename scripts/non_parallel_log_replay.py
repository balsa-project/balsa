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
"""Replay a run log to generate non-parallel timing results.

Usage:

  >> bash scripts/run_replay.sh 2>&1 | tee replay.log
"""
import os
from datetime import datetime
from pathlib import Path

from absl import app
from absl import flags

import numpy as np
import pandas as pd
from tqdm import tqdm

from balsa.execution import QueryExecutionCache

FLAGS = flags.FLAGS
flags.DEFINE_string('input', 'input.log', 'Input log file from W&B.')
flags.DEFINE_string('output_dir', None, 'Output dir.')
flags.DEFINE_bool('include_test_wait_time', True,
                  'Whether to include time to execute test queries in x-axis.')
flags.DEFINE_bool('increment_iter_despite_timeouts', True,
                  'Increment value iteration despite timeouts.')


def find(lst, substr):
    idx = 0
    for i, v in enumerate(lst):
        if substr in v:
            idx = i
    return idx


def postprocess_repeated_iters(workload):
    """Only used for increment_iter_despite_timeouts=False.
    If an iteration is repeated, it sets latency/workload
    and num_skipped_queries across rows to be the value of the last row
    (the successful iteration).
    """
    n = len(workload['value_iter'])
    for i in reversed(range(n)):
        val_iter = workload['value_iter'][i]
        if val_iter > workload['value_iter'][0] and workload['value_iter'][
                i - 1] == val_iter:
            workload['latency/workload'][i -
                                         1] = workload['latency/workload'][i]
            workload['num_skipped_queries'][
                i - 1] = workload['num_skipped_queries'][i]


def collapse_timeout_iters(workload, is_test):
    """Collapses timeout iters when increment_iter_despite_timeouts=False.

    This is done by setting the y-value (latency/workload, num_skipped_queries)
    to the same value as the successful iteration and then summing across
    repeated rows for other metrics (train_time, wall_time, etc.) so we can
    take this information into account when accumulating time between
    successful iterations.

    This method is destructive to workload.
    """
    postprocess_repeated_iters(workload)

    groupby_col_ops = {
        'planning_time': sum,
        'wall_exec_time': sum,
        # Using 'min' is ok as rows with the same value_iter have the same
        # values for these metrics, after postprocess_repeated_iters().
        'latency/workload': min,
        'num_skipped_queries': min,
    }
    if not is_test:
        groupby_col_ops.update({'training_time': sum})

    df = pd.DataFrame.from_dict(workload)
    df = df.groupby('value_iter', as_index=False).aggregate(groupby_col_ops)

    return pd.DataFrame.to_dict(df, orient='list')


def main(argv):
    del argv  # Unused.

    in_path = os.path.expanduser(FLAGS.input)
    print(f'Reading logfile at {in_path}')

    increment_iter_despite_timeouts = FLAGS.increment_iter_despite_timeouts

    with open(in_path) as f:
        log = f.readlines()

    log = [l for l in log if 'pid=' not in l \
                              or 'Planning' in l \
                              or 'curr_timeout_ms=' in l \
                              or '] CUDA_VISIBLE_DEVICES:' in l \
                               or ('predicted' in l and '),/*+' in l)]

    training_workload = {
        'value_iter': [],
        'training_time': [],
        'planning_time': [],
        'wall_exec_time': [],
        'latency/workload': [],
        'num_skipped_queries': [],
    }

    testing_workload = {
        'value_iter': [],
        'planning_time': [],
        'wall_exec_time': [],
        'latency/workload': [],
        'num_skipped_queries': [],
    }

    query_cache = QueryExecutionCache()
    curr_value_iter = 0
    curr_timeout_ms = None

    iter_has_timeout = False
    iter_train_planning_time = 0
    iter_train_exec_time = 0
    iter_train_exec_wall_time = 0
    iter_train_skipped_queries = 0
    iter_train_skipped_due_timeout = []

    iter_test_planning_time = 0
    iter_test_exec_time = 0
    iter_test_exec_wall_time = 0
    iter_test_skipped_queries = 0

    total_planning_events = 0
    total_exec_events = 0

    train_queries = []
    test_queries = []
    for i, l in enumerate(log):
        if 'train queries' in l:
            line = l[l.index('[') + 1:]
            line = line[:line.index(']')]
            train_queries = [
                'q' + q.replace("'", "").strip() for q in line.split(',')
            ]
        elif 'test queries' in l:
            line = l[l.index('[') + 1:]
            line = line[:line.index(']')]
            test_queries = [
                'q' + q.replace("'", "").strip() for q in line.split(',')
            ]
            break

    print('{} train queries:'.format(len(train_queries)), train_queries)
    print('{} test queries:'.format(len(test_queries)), test_queries)

    count = 0
    seen_lines = set()
    for i, l in tqdm(enumerate(log), disable=True, total=len(log)):
        if 'pid=' in l:
            if 'Planning' in l:
                plan_str_idx = l.index('Planning')
                l = l[plan_str_idx:]
                ms_str_idx = l.index('ms')
                l = l[:ms_str_idx]
                log[i] = l
            elif 'predicted' in l and '/*+' in l:
                pass

        if '] CUDA_VISIBLE_DEVICES:' in l:  # training time
            cursor = i
            start = cursor

            is_training = 'Epoch 0' in ''.join(log[i:i + 100])

            if 'Loaded pretrained checkpoint' in log[start + 1]:
                # Skip SIM loading (not a training iter)
                continue

            while '] Saving latest checkpoint..' not in log[cursor]:
                cursor += 1

            assert (start, -1) not in seen_lines
            seen_lines.add((start, -1))

            if not is_training:
                if len(training_workload['training_time']) == 0:
                    training_workload['training_time'].append(0)
                continue

            start_training = log[start]
            end_training = log[cursor]

            start_training = start_training.split(' ')
            day_idx = find(start_training, 'I0')
            ts_idx = find(start_training, '.py') - 2
            start_day = start_training[day_idx]
            start_day = int(start_day[start_day.index('I0') + 2:])
            start_time = start_training[ts_idx]

            end_training = end_training.split(' ')
            day_idx = find(end_training, 'I0')
            ts_idx = find(end_training, '.py') - 2
            end_day = end_training[day_idx]
            end_day = int(end_day[end_day.index('I0') + 2:])
            end_time = end_training[ts_idx]

            dates = ['20.12.2016', '21.12.2016']
            day_delta = end_day - start_day
            start_ms = datetime.strptime(
                f'{dates[0]} {start_time}',
                '%d.%m.%Y %H:%M:%S.%f').timestamp() * 1000
            end_ms = datetime.strptime(
                f'{dates[day_delta]} {end_time}',
                '%d.%m.%Y %H:%M:%S.%f').timestamp() * 1000
            delta_ms = end_ms - start_ms

            # Log training time
            training_workload['training_time'].append(delta_ms)

        elif 'Planning' in l:  # planning time and query caching
            cursor = i
            line = log[cursor].strip()
            planning_time = float(line.split(' ')[-1][:-2])
            cursor += 1

            # Get query plan
            while '/*' not in log[cursor]:
                cursor += 1

            start = cursor
            while '---------------------------------------' not in log[cursor] and \
                'Waiting on Ray tasks' not in log[cursor]:
                cursor += 1

            hint_str = log[start:cursor]

            assert (start, cursor) not in seen_lines
            seen_lines.add((start, cursor))

            query_name = log[start].split('predicted')[0]
            query_name = 'q' + query_name.split('q')[-1].split(',')[0]

            if 'qtpch' in log[start]:
                query_name = 'qtpch-' + query_name

            hint_str[0] = hint_str[0][hint_str[0].index('/*'):]
            hint_str = ''.join(hint_str)

            wellformed_qname = (query_name[0] == 'q' and 3 <= len(query_name) <= 5) or \
                               ('tpch' in query_name and 10 <= len(query_name) <= 12)
            assert wellformed_qname, query_name

            total_planning_events += 1

            # Log planning time
            if query_name in train_queries:
                iter_train_planning_time += planning_time

            if query_name == train_queries[-1].strip():
                training_workload['planning_time'].append(
                    iter_train_planning_time)

                count += 1

                iter_train_planning_time = 0

            if query_name in test_queries:
                iter_test_planning_time += planning_time

            if query_name == test_queries[-1].strip():
                testing_workload['value_iter'].append(curr_value_iter - 1)
                testing_workload['planning_time'].append(
                    iter_test_planning_time)

                iter_test_planning_time = 0

        elif 'curr_timeout_ms=' in l:
            timeout = l.split('=')[-1]

            line = log[i].strip()

            # Find relevant execution info
            time = line.split(' ')[-4]
            cursor = i
            while 'Execution' not in log[cursor]:
                cursor -= 1
            query_name = log[cursor].split(' ')[0]

            time = float(time)

            if 'q' + query_name in train_queries:
                if 'None' in timeout:
                    curr_timeout_ms = None
                else:
                    curr_timeout_ms = float(timeout)

            # Get executed query plan
            original_cursor = cursor
            cursor -= 1
            while '/*' not in log[cursor]:
                cursor -= 1

            start = cursor
            hint_str = log[start:original_cursor]

            query_name = hint_str[0].split(',')[0]
            hint_str[0] = hint_str[0][hint_str[0].index('/*'):]
            hint_str = ''.join(hint_str)

            execution_time = time
            wall_time = time
            cached = query_cache.Get(key=(query_name, hint_str)) is not None

            total_exec_events += 1

            if time == -1:  # There was a timeout
                iter_has_timeout = True
                execution_time = curr_timeout_ms
                wall_time = curr_timeout_ms
            else:  # Get execution result
                if cached:
                    wall_time -= execution_time
                else:
                    query_cache.Put(key=(query_name, hint_str),
                                    value=None,
                                    latency=0)

            # Log execution time
            if query_name in train_queries:
                iter_train_exec_time += execution_time
                iter_train_exec_wall_time += wall_time
                if cached:
                    iter_train_skipped_queries += 1

            if query_name == train_queries[-1].strip():
                training_workload['wall_exec_time'].append(
                    iter_train_exec_wall_time)
                training_workload['latency/workload'].append(
                    iter_train_exec_time)
                training_workload['num_skipped_queries'].append(
                    iter_train_skipped_queries)
                training_workload['value_iter'].append(curr_value_iter)

                if iter_has_timeout:
                    # In wandb logging path, we don't report a timeout iter's
                    # metrics.
                    # print('  Excluding train metrics @ iter={} due to timeout'.
                    #       format(curr_value_iter))
                    iter_train_skipped_due_timeout.append(curr_value_iter)

                if iter_has_timeout and not increment_iter_despite_timeouts:
                    pass
                else:
                    curr_value_iter += 1

                iter_has_timeout = False
                iter_train_exec_time = 0
                iter_train_exec_wall_time = 0
                iter_train_skipped_queries = 0

            if query_name in test_queries:
                iter_test_exec_time += execution_time
                iter_test_exec_wall_time += wall_time
                if cached:
                    iter_test_skipped_queries += 1

            if query_name == test_queries[-1].strip():
                testing_workload['wall_exec_time'].append(
                    iter_test_exec_wall_time)
                testing_workload['latency/workload'].append(iter_test_exec_time)
                testing_workload['num_skipped_queries'].append(
                    iter_test_skipped_queries)

                iter_test_exec_time = 0
                iter_test_exec_wall_time = 0
                iter_test_skipped_queries = 0

    assert all([
        len(v) == len(training_workload['value_iter'])
        for v in training_workload.values()
    ]), ('Inconsistent iteration lengths.',
         [(k, len(v)) for k, v in training_workload.items()])

    train_value_iters = len(training_workload['value_iter'])
    test_value_iters = len(testing_workload['value_iter'])

    assert total_planning_events == train_value_iters * len(
        train_queries) + test_value_iters * len(test_queries)

    assert total_planning_events == total_exec_events

    if not increment_iter_despite_timeouts:
        training_workload = collapse_timeout_iters(training_workload,
                                                   is_test=False)
        # NOTE: testing_workload's collapsed metrics don't quite correspond to
        # w&b logged metrics.  See soft-galaxy-3797.  However for the purpose
        # of counting wallclock this should be fine.
        testing_workload = collapse_timeout_iters(testing_workload,
                                                  is_test=True)

    # Pad test_df_dict accordingly.  Purpose: some cls had test eval start
    # after X iters and eval may run every Y iters.
    missing_value_iters = set(training_workload['value_iter']) - set(
        testing_workload['value_iter'])
    for k, v in testing_workload.items():
        if k == 'value_iter':
            testing_workload[k] = training_workload[k]
        else:
            new_col = []
            j = 0
            for i in training_workload['value_iter']:
                if i in missing_value_iters:
                    new_col.append(0)
                else:
                    new_col.append(v[j])
                    j += 1
            testing_workload[k] = new_col

    df = pd.DataFrame.from_dict(training_workload)

    # TODO: Add training query execution time to test curve x-axis?
    test_df = pd.DataFrame.from_dict(testing_workload)
    test_df['iter_time'] = test_df['wall_exec_time'] + test_df['planning_time']

    train_iter_time = df['wall_exec_time'] + df['planning_time'] + df[
        'training_time']
    if FLAGS.include_test_wait_time:
        train_iter_time += test_df['iter_time']

    df['iter_time'] = train_iter_time
    df['relative_timestamp'] = np.cumsum(df['iter_time'])

    if increment_iter_despite_timeouts:
        df = df.drop(iter_train_skipped_due_timeout)

    if FLAGS.output_dir is None:
        output_dir = os.path.dirname(FLAGS.input)
    else:
        output_dir = FLAGS.output_dir

    # TODO: Dump test_df CSV as well
    os.makedirs(output_dir, exist_ok=True)
    train_out_path = os.path.join(
        output_dir, '{}.train_replay.csv'.format(os.path.basename(FLAGS.input)))
    df.to_csv(train_out_path)
    print(f'Saved to {train_out_path}')


if __name__ == '__main__':
    app.run(main)
