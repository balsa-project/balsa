SELECT
    MIN(mi_idx.info) AS rating,
    MIN(t.title) AS movie_title
FROM
    movie_info_idx mi_idx,
    movie_keyword mk,
    title t
WHERE
    mi_idx.info > '2.0'
    AND t.production_year > 2005
    AND t.id = mi_idx.movie_id
    AND t.id = mk.movie_id
    AND mk.movie_id = mi_idx.movie_id;
