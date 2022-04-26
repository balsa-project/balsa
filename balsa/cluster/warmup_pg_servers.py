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
"""Runs all queries on all servers using Async IO.

When a VM is first launched, the storage is cold and Postgres is slow. This
script will run the benchmark queries repeatedly until the execution time
becomes normal (TARGET_LATENCY_SECS).
"""
import asyncio
import ipaddress
import json
import logging
import os
import subprocess
import sys
import re

import aiopg

CLUSTER_CONFIG_DEFAULT_PATH = '~/balsa/balsa/cluster/cluster.yml'
WORKLOAD = "job"
USE_LOCAL_POSTGRES = False
TEST_QUERIES_ONLY = False
IP_ADDR_REGEX = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'

if WORKLOAD == "job":
    DSN_FMT = "postgres://psycopg:psycopg@{server}/imdbload"
    QUERIES_DIR = "~/balsa/queries/join-order-benchmark"
    QUERY_TIMEOUT = 3600
    TARGET_LATENCY_SECS = 200
else:
    raise ValueError(WORKLOAD)

logging.basicConfig(
    format="%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
    level=logging.INFO,
)


def load_queries(queries_dir=QUERIES_DIR):
    queries_dir = os.path.expanduser(queries_dir)
    if TEST_QUERIES_ONLY:
        raise NotImplementedError
    else:
        files = os.scandir(queries_dir)
        queries = [(f.name, open(f.path).read())
                   for f in files
                   if f.is_file() and f.name.endswith(".sql")]
    return sorted(queries)


async def execute_query(cur, query):
    """
    Execute the query and return its execution time.
    """
    query = "EXPLAIN (ANALYZE true, FORMAT json) " + query
    await cur.execute(query, timeout=QUERY_TIMEOUT)
    result = await cur.fetchall()
    json_dict = result[0][0][0]
    latency = float(json_dict['Execution Time']) / 1000
    return latency, json.dumps(json_dict, indent=2)


async def execute_queries(server, cur, queries):
    """
    Execute the benchmark queries, returning total execution time.
    """
    await cur.execute("DISCARD ALL")
    logging.debug(f"Server {server} executing DISCARD ALL")
    total_latency = 0
    count = 0
    n_queries = len(queries)
    plans = []
    for qname, query in queries:
        latency, plan = await execute_query(cur, query)
        count += 1
        logging.debug(
            f"Server {server} executing {qname} latency: {latency:.2f}s ({count}/{n_queries})"
        )
        total_latency += latency
        plans.append(plan)
    return total_latency, "\n\n".join(plans)


async def benchmark_server(server, queries, runs=5):
    logging.info(f"Benchmarking {server}...")
    dsn = DSN_FMT.format(server=server)
    try:
        async with aiopg.connect(dsn) as conn:
            async with conn.cursor() as cur:
                logging.info(
                    f"Server {server} executing all queries... This may take a while."
                )
                latencies = []
                while runs > 0:
                    latency, all_plans = await execute_queries(
                        server, cur, queries)
                    logging.info(f"{server} total latency: {latency:.2f}s")
                    latencies.append(latency)
                    if latency < TARGET_LATENCY_SECS:
                        return server, True, latencies, all_plans
                    runs -= 1
                return server, False, latencies, all_plans
    except Exception as e:
        logging.info(f"Error connecting to {server}: {e}")
        logging.info(
            "Please check if the machine allows all inbound traffic over TCP.")
        return server, False, [], ""


def diff_all(servers):
    assert len(servers) > 0, servers
    ref_server = servers[0]
    for cur_server in servers[1:]:
        diff_output = subprocess.run([
            "diff", "--ignore-all-space", f"{ref_server}-plans.txt",
            f"{cur_server}-plans.txt"
        ],
                                     capture_output=True).stdout.decode("ascii")
        diffs = diff_output.split("---\n")
        for diff in diffs:
            if "Time" in diff:
                continue
            print(diff)
            print("-----------")


def print_ips(servers):
    for ip in servers:
        ip = str(ip)
        print("pg-{} ansible_host={}".format(ip.replace(".", "-"), ip))
    exit(0)


async def main(cluster_config_file):
    logging.info("Hello, world!")
    if USE_LOCAL_POSTGRES:
        servers = ["localhost"]
    else:
        ray_nodes_out = subprocess.run(
            "ray get-worker-ips ~/ray_bootstrap_config.yaml",
            shell=True,
            capture_output=True)
        if ray_nodes_out.returncode != 0:
            print(ray_nodes_out.stderr.decode())
            exit(ray_nodes_out.returncode)
        ray_nodes_out = ray_nodes_out.stdout
        # we only need to warmup worker nodes
        nodes_ip = re.findall(IP_ADDR_REGEX, ray_nodes_out.decode().strip())
        servers = nodes_ip
        servers = [ipaddress.ip_address(ip) for ip in servers]

    queries = load_queries()
    results = await asyncio.gather(
        *[benchmark_server(s, queries) for s in servers])
    for server, ready, latencies, all_plans in results:
        ready = "is ready" if ready else "is not ready"
        logging.info(f"Server {server} {ready}, latencies={latencies}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise ValueError('Too many arguments')
    elif len(sys.argv) == 2:
        cluster_config_file = sys.argv[1]
    else:
        cluster_config_file = CLUSTER_CONFIG_DEFAULT_PATH
    cluster_config_file = os.path.expanduser(cluster_config_file)
    asyncio.run(main(cluster_config_file))
