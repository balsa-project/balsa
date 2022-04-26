-- Fig. 2 of "Query Optimization Through the Looking Glass, and What We Found
-- Running the Join Order Benchmark".
CREATE TABLE name (
  id integer NOT NULL PRIMARY KEY,
  name text NOT NULL,
  imdb_index character varying(12),
  imdb_id integer,
  gender character varying(1),
  name_pcode_cf character varying(5),
  name_pcode_nf character varying(5),
  surname_pcode character varying(5),
  md5sum character varying(32)
);

CREATE TABLE char_name (
  id integer NOT NULL PRIMARY KEY,
  name text NOT NULL,
  imdb_index character varying(12),
  imdb_id integer,
  name_pcode_nf character varying(5),
  surname_pcode character varying(5),
  md5sum character varying(32)
);

CREATE TABLE comp_cast_type (
  id integer NOT NULL PRIMARY KEY,
  kind character varying(32) NOT NULL
);

CREATE TABLE company_name (
  id integer NOT NULL PRIMARY KEY,
  name text NOT NULL,
  country_code character varying(255),
  imdb_id integer,
  name_pcode_nf character varying(5),
  name_pcode_sf character varying(5),
  md5sum character varying(32)
);

CREATE TABLE company_type (
  id integer NOT NULL PRIMARY KEY,
  kind character varying(32) NOT NULL
);

CREATE TABLE info_type (
  id integer NOT NULL PRIMARY KEY,
  info character varying(32) NOT NULL
);

CREATE TABLE keyword (
  id integer NOT NULL PRIMARY KEY,
  keyword text NOT NULL,
  phonetic_code character varying(5)
);

CREATE TABLE kind_type (
  id integer NOT NULL PRIMARY KEY,
  kind character varying(15) NOT NULL
);

CREATE TABLE link_type (
  id integer NOT NULL PRIMARY KEY,
  link character varying(32) NOT NULL
);

CREATE TABLE role_type (
  id integer NOT NULL PRIMARY KEY,
  role character varying(32) NOT NULL
);

CREATE TABLE title (
  id integer NOT NULL PRIMARY KEY,
  title text NOT NULL,
  imdb_index character varying(12),
  kind_id integer NOT NULL REFERENCES kind_type(id),
  production_year integer,
  imdb_id integer,
  phonetic_code character varying(5),
  episode_of_id integer,
  season_nr integer,
  episode_nr integer,
  series_years character varying(49),
  md5sum character varying(32)
);

CREATE TABLE aka_name (
    id integer NOT NULL PRIMARY KEY REFERENCES name(id),
    person_id integer NOT NULL,
    name text NOT NULL,
    imdb_index character varying(12),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE aka_title (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL, -- REFERENCES title(id),
    title text NOT NULL,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note text,
    md5sum character varying(32)
);

CREATE TABLE cast_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL REFERENCES aka_name(id),
    movie_id integer NOT NULL REFERENCES title(id),
    person_role_id integer REFERENCES char_name(id),
    note text,
    nr_order integer,
    role_id integer NOT NULL REFERENCES role_type(id)
);

CREATE TABLE complete_cast (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer REFERENCES title(id),
    subject_id integer NOT NULL REFERENCES comp_cast_type(id),
    status_id integer NOT NULL REFERENCES comp_cast_type(id)
);

CREATE TABLE movie_companies (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL REFERENCES title(id),
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note text
);

CREATE TABLE movie_info (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL REFERENCES title(id),
    info_type_id integer NOT NULL REFERENCES info_type(id),
    info text NOT NULL,
    note text
);

CREATE TABLE movie_info_idx (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL REFERENCES title(id),
    info_type_id integer NOT NULL REFERENCES info_type(id),
    info text NOT NULL,
    note text
);

CREATE TABLE movie_keyword (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL REFERENCES title(id),
    keyword_id integer NOT NULL REFERENCES keyword(id)
);

CREATE TABLE movie_link (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL REFERENCES title(id),
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL REFERENCES link_type(id)
);

CREATE TABLE person_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL REFERENCES name(id),
    info_type_id integer NOT NULL REFERENCES info_type(id),
    info text NOT NULL,
    note text
);
