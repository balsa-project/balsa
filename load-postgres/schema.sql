CREATE TABLE aka_name (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    name text NOT NULL collate "C",
    imdb_index character varying(12) collate "C",
    name_pcode_cf character varying(5) collate "C",
    name_pcode_nf character varying(5) collate "C",
    surname_pcode character varying(5) collate "C",
    md5sum character varying(32) collate "C"
);

CREATE TABLE aka_title (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    title text NOT NULL collate "C",
    imdb_index character varying(12) collate "C",
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code character varying(5) collate "C",
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note text collate "C",
    md5sum character varying(32) collate "C"
);

CREATE TABLE cast_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    movie_id integer NOT NULL,
    person_role_id integer,
    note text collate "C",
    nr_order integer,
    role_id integer NOT NULL
);

CREATE TABLE char_name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL collate "C",
    imdb_index character varying(12) collate "C",
    imdb_id integer,
    name_pcode_nf character varying(5) collate "C",
    surname_pcode character varying(5) collate "C",
    md5sum character varying(32) collate "C"
);

CREATE TABLE comp_cast_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL collate "C"
);

CREATE TABLE company_name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL collate "C",
    country_code character varying(255) collate "C",
    imdb_id integer,
    name_pcode_nf character varying(5) collate "C",
    name_pcode_sf character varying(5) collate "C",
    md5sum character varying(32) collate "C"
);

CREATE TABLE company_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL collate "C"
);

CREATE TABLE complete_cast (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    subject_id integer NOT NULL,
    status_id integer NOT NULL
);

CREATE TABLE info_type (
    id integer NOT NULL PRIMARY KEY,
    info character varying(32) NOT NULL collate "C"
);

CREATE TABLE keyword (
    id integer NOT NULL PRIMARY KEY,
    keyword text NOT NULL collate "C",
    phonetic_code character varying(5) collate "C"
);

CREATE TABLE kind_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(15) NOT NULL collate "C"
);

CREATE TABLE link_type (
    id integer NOT NULL PRIMARY KEY,
    link character varying(32) NOT NULL collate "C"
);

CREATE TABLE movie_companies (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note text collate "C"
);

CREATE TABLE movie_info (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL collate "C",
    note text collate "C"
);

CREATE TABLE movie_info_idx (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL collate "C",
    note text collate "C"
);

CREATE TABLE movie_keyword (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    keyword_id integer NOT NULL
);

CREATE TABLE movie_link (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL
);

CREATE TABLE name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL collate "C",
    imdb_index character varying(12) collate "C",
    imdb_id integer,
    gender character varying(1) collate "C",
    name_pcode_cf character varying(5) collate "C",
    name_pcode_nf character varying(5) collate "C",
    surname_pcode character varying(5) collate "C",
    md5sum character varying(32) collate "C"
);

CREATE TABLE person_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL collate "C",
    note text collate "C"
);

CREATE TABLE role_type (
    id integer NOT NULL PRIMARY KEY,
    role character varying(32) NOT NULL collate "C"
);

CREATE TABLE title (
    id integer NOT NULL PRIMARY KEY,
    title text NOT NULL collate "C",
    imdb_index character varying(12) collate "C",
    kind_id integer NOT NULL,
    production_year integer,
    imdb_id integer,
    phonetic_code character varying(5) collate "C",
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years character varying(49) collate "C",
    md5sum character varying(32) collate "C"
);
