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
"""Prepends column names to headerless IMDB CSVs."""
import os

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('csv_dir', 'datasets/job', 'Directory to IMDB CSVs.')


def PrependLine(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


def Main(argv):
    del argv  # Unused.

    # Column names for http://homepages.cwi.nl/~boncz/job/imdb.tgz.
    _COLUMNS = {
        'aka_name': [
            'id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf',
            'name_pcode_nf', 'surname_pcode', 'md5sum'
        ],
        'aka_title': [
            'id', 'movie_id', 'title', 'imdb_index', 'kind_id',
            'production_year', 'phonetic_code', 'episode_of_id', 'season_nr',
            'episode_nr', 'note', 'md5sum'
        ],
        'cast_info': [
            'id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
            'role_id'
        ],
        'char_name': [
            'id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf',
            'surname_pcode', 'md5sum'
        ],
        'company_name': [
            'id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf',
            'name_pcode_sf', 'md5sum'
        ],
        'company_type': ['id', 'kind'],
        'comp_cast_type': ['id', 'kind'],
        'complete_cast': ['id', 'movie_id', 'subject_id', 'status_id'],
        'info_type': ['id', 'info'],
        'keyword': ['id', 'keyword', 'phonetic_code'],
        'kind_type': ['id', 'kind'],
        'link_type': ['id', 'link'],
        'movie_companies': [
            'id', 'movie_id', 'company_id', 'company_type_id', 'note'
        ],
        'movie_info': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
        'movie_info_idx': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
        'movie_keyword': ['id', 'movie_id', 'keyword_id'],
        'movie_link': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
        'name': [
            'id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf',
            'name_pcode_nf', 'surname_pcode', 'md5sum'
        ],
        'person_info': ['id', 'person_id', 'info_type_id', 'info', 'note'],
        'role_type': ['id', 'role'],
        'title': [
            'id', 'title', 'imdb_index', 'kind_id', 'production_year',
            'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr',
            'episode_nr', 'series_years', 'md5sum'
        ],
    }

    for table_name, columns in _COLUMNS.items():
        filename = os.path.join(FLAGS.csv_dir, '{}.csv'.format(table_name))
        line = ','.join(columns)
        logging.info('Prepending header to {}: {}'.format(filename, line))
        PrependLine(filename, line)


if __name__ == '__main__':
    app.run(Main)
