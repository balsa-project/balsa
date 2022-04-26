ALTER TABLE title ADD FOREIGN KEY (kind_id) REFERENCES kind_type;
ALTER TABLE aka_name ADD FOREIGN KEY (id) REFERENCES name;
-- psql:add_fks.sql:3: ERROR:  insert or update on table "cast_info" violates foreign key constraint "cast_info_person_id_fkey"
--   DETAIL:  Key (person_id)=(901344) is not present in table "aka_name".
-- ALTER TABLE cast_info ADD FOREIGN KEY (person_id) REFERENCES aka_name;
ALTER TABLE cast_info ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE cast_info ADD FOREIGN KEY (person_role_id) REFERENCES char_name;
ALTER TABLE cast_info ADD FOREIGN KEY (role_id) REFERENCES role_type;
ALTER TABLE complete_cast ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE complete_cast ADD FOREIGN KEY (subject_id) REFERENCES comp_cast_type;
ALTER TABLE complete_cast ADD FOREIGN KEY (status_id) REFERENCES comp_cast_type;
ALTER TABLE movie_companies ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE movie_info ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE movie_info ADD FOREIGN KEY (info_type_id) REFERENCES info_type;
ALTER TABLE movie_info_idx ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE movie_info_idx ADD FOREIGN KEY (info_type_id) REFERENCES info_type;
ALTER TABLE movie_keyword ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE movie_keyword ADD FOREIGN KEY (keyword_id) REFERENCES keyword;
ALTER TABLE movie_link ADD FOREIGN KEY (movie_id) REFERENCES title;
ALTER TABLE movie_link ADD FOREIGN KEY (link_type_id) REFERENCES link_type;
ALTER TABLE person_info ADD FOREIGN KEY (person_id) REFERENCES name;
ALTER TABLE person_info ADD FOREIGN KEY (info_type_id) REFERENCES info_type;
