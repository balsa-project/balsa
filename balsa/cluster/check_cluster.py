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
"""Checks the Ray cluster is ready."""
import os
import subprocess
import time
import yaml

from absl import app
from absl import flags
from absl import logging
import ray

FLAGS = flags.FLAGS
# TODO: support scaling up the cluster directly from here.
# flags.DEFINE_integer("n_nodes", 4, "number of nodes to launch", short_name="n")
flags.DEFINE_string(
    "cluster_config",
    "cluster.yml",
    "Path to the cluster config file relative to the script location",
)


def run(cmd, **kwargs):
    logging.info("$ " + cmd)
    return subprocess.run(cmd, shell=True, **kwargs)


def check_resources_allocated_impl(required_resources, ray_resources):
    for key, val in required_resources.items():
        ray_val = ray_resources.get(key, 0)
        if ray_val < val:
            return False
    return True


def check_resources_allocated(num_nodes, num_tries=20, wait_time=60):
    ray.init(address="auto")
    required_resources = {"pg": num_nodes}
    while num_tries > 0:
        ray_resources = ray.cluster_resources()
        ready = check_resources_allocated_impl(required_resources,
                                               ray_resources)
        if ready:
            logging.info(f"Ray cluster is ready: {ray_resources}")
            return
        logging.info(
            f"Ray cluster is not ready yet, sleeping for {wait_time} secs: required={required_resources}, actual={ray_resources}"
        )
        logging.info(
            "It might take 10+ minutes for the cluster to be ready after launching it."
        )
        time.sleep(wait_time)
        num_tries -= 1
    raise RuntimeError(
        "Ray cluster is not ready. Type 'ray status' to see the cluster status."
    )


def main(argv):
    del argv  # Unused.

    cluster_config_path = os.path.expanduser(FLAGS.cluster_config)
    with open(cluster_config_path, 'r') as f:
        config = yaml.safe_load(f)

    check_resources_allocated(config['initial_workers'])


if __name__ == "__main__":
    app.run(main)
