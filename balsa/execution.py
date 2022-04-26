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
"""Execution helpers."""

import numpy as np


class PerQueryTimeoutController(object):
    """Bounds the total duration of an iteration over the workload."""

    def __init__(self,
                 timeout_slack=2,
                 no_op=False,
                 relax_timeout_factor=None,
                 relax_timeout_on_n_timeout_iters=None,
                 initial_timeout_ms=None):
        self.timeout_slack = timeout_slack
        self.no_op = no_op
        self.relax_timeout_factor = relax_timeout_factor
        self.relax_timeout_on_n_timeout_iters = relax_timeout_on_n_timeout_iters
        # Fields to maintain.
        self.iter_timeout_ms = initial_timeout_ms
        self.curr_iter_ms = None
        self.curr_iter_max_ms = None
        self.curr_iter_has_timeouts = False
        self.num_consecutive_timeout_iters = 0
        self.iter_executed = False

    def GetTimeout(self, query_node):
        if self.no_op:
            return None
        if self.iter_timeout_ms is None:
            return None
        # If all queries in the previous iteration time out, iter_timeout_ms
        # would be -1e30; guard against this by returning 0 (no timeout).
        return max(self.iter_timeout_ms, 0)

    def RecordQueryExecution(self, query_node, latency_ms):
        del query_node
        if self.no_op:
            return
        self.iter_executed = True
        if latency_ms < 0:
            # This query timed out.
            self.curr_iter_has_timeouts = True
        else:
            # This query finished within timeout.
            self.curr_iter_ms += latency_ms
            self.curr_iter_max_ms = max(self.curr_iter_max_ms, latency_ms)

    def OnIterStart(self):
        if self.no_op:
            return
        # NOTE: Suppose we call OnIterStart() then due to errors do not call
        # any RecordQueryExecution().  Due to retries we call OnIterStart()
        # again.  At this point, curr_iter_max_ms = -1e30 and iter_timeout_ms =
        # None and iter_executed = False, and we should not update the timeout.
        if self.curr_iter_max_ms is not None and self.iter_executed:
            # Update timeout.
            if self.iter_timeout_ms is None:
                self.iter_timeout_ms = \
                    self.curr_iter_max_ms * self.timeout_slack
            elif not self.curr_iter_has_timeouts:
                self.iter_timeout_ms = min(
                    self.iter_timeout_ms,
                    self.curr_iter_max_ms * self.timeout_slack)

            # How long have we been consecutively timing out up to now?
            if self.curr_iter_has_timeouts:
                self.num_consecutive_timeout_iters += 1
            else:
                self.num_consecutive_timeout_iters = 0

            # Optionally, relax the timeout.
            if (self.relax_timeout_factor is not None and
                    self.num_consecutive_timeout_iters >=
                    self.relax_timeout_on_n_timeout_iters):
                self.iter_timeout_ms *= self.relax_timeout_factor

        self.curr_iter_ms = 0
        self.curr_iter_max_ms = -1e30
        self.curr_iter_has_timeouts = False
        self.iter_executed = False


class QueryExecutionCache(object):
    """A simple cache mapping key -> (best value, best latency).

    To record (best result, best latency) per (query name, plan):

        # Maps key to (best value, best latency).
        Put(key=(query_name, hint_str), value=result_tup, latency=latency)

    To record (best Node, best latency) per query name:

        Put(key=query_name, value=node, latency=latency)
    """

    def __init__(self):
        self._cache = {}
        self._counts = {}

    def size(self):
        return len(self._cache)

    def Put(self, key, value, latency):
        """Put.

        Updates key -> (value, latency), iff
        (1) no existing value is found or
        (2) latency < the existing latency.

        Args:
          key: the key.  E.g., (query_name, hint_str) which identifies a unique
            query plan.
          value: the value.  E.g., a ResultTup or a Node.
          latency: the latency.
        """
        prior_result = self._cache.get(key)
        if prior_result is None:
            prior_latency = np.inf
        else:
            prior_latency = prior_result[1]
        if latency < prior_latency:
            self._cache[key] = (value, latency)
        # Update visit counts.
        cnt = self.GetVisitCount(key)
        self._counts[key] = cnt + 1

    def Get(self, key):
        return self._cache.get(key)

    def GetVisitCount(self, key):
        return self._counts.get(key, 0)
