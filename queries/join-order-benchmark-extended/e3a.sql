SELECT
    MIN(mi_idx.info) AS rating,
    MIN(t.title) AS mainstream_movie
FROM
    movie_companies mc,
    movie_info mi,
    movie_info_idx mi_idx,
    title t
WHERE
    mi.info IN ('Drama', 'Horror', 'Western', 'Family')
    AND mi_idx.info > '7.0'
    AND t.production_year BETWEEN 2000 AND 2010
    AND t.id = mi.movie_id
    AND t.id = mi_idx.movie_id
    AND t.id = mc.movie_id;
