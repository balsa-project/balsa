#!/bin/bash
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
set -ex

conda create -n balsa python=3.7 -y
eval "$(conda shell.bash hook)"
conda activate balsa
pip install -r requirements.txt
pip install -e .
pip install -e pg_executor
pip install boto3

cd ~/
wget https://ftp.postgresql.org/pub/source/v12.5/postgresql-12.5.tar.gz
tar xzvf postgresql-12.5.tar.gz
cd postgresql-12.5
./configure --prefix=/data/postgresql-12.5 --without-readline
sudo make -j
sudo make install

echo 'export PATH=/data/postgresql-12.5/bin:$PATH' >> ~/.bashrc
export PATH=/data/postgresql-12.5/bin:$PATH

cd ~/
git clone https://github.com/ossc-db/pg_hint_plan.git -b REL12_1_3_7 || true
cd pg_hint_plan
# Modify Makefile: change line
#   PG_CONFIG = pg_config
# to
#   PG_CONFIG = /data/postgresql-12.5/bin/pg_config
sed -i 's/PG_CONFIG = pg_config/PG_CONFIG = \/data\/postgresql-12.5\/bin\/pg_config/g' Makefile
# vim Makefile
make
sudo make install

cd ~/
mkdir -p datasets/job && pushd datasets/job
wget -c http://homepages.cwi.nl/~boncz/job/imdb.tgz && tar -xvzf imdb.tgz && popd
# Prepend headers to CSV files
conda activate balsa && python3 ~/balsa/scripts/prepend_imdb_headers.py

# Enable re-running this script.
# Avoid errors like:
#     initdb: error: directory "/home/ubuntu/imdb" exists but is not empty
#     If you want to create a new database system, either remove or empty
#     the directory "/home/ubuntu/imdb" or run initdb
#     with an argument other than "/home/ubuntu/imdb".
set +e

# Create and start the DB
pg_ctl -D ~/imdb initdb

# Copy custom PostgreSQL configuration.
cp ~/balsa/conf/balsa-postgresql.conf ~/imdb/postgresql.conf

# Start the server
pg_ctl -D ~/imdb start -l logfile

# Load data + run analyze (can take several minutes)
cd ~/balsa
bash load-postgres/load_job_postgres.sh ~/datasets/job

echo 'host    all             all             0.0.0.0/0            trust' >> ~/imdb/pg_hba.conf
