SELECT
    MIN(mi.info) AS release_date,
    MIN(t.title) AS modern_internet_movie
FROM
    aka_title at,
    movie_companies mc,
    movie_info mi,
    movie_keyword mk,
    title t
WHERE
    mi.note LIKE '%internet%'
    AND mi.info IS NOT NULL
    AND (mi.info LIKE 'USA:% 199%'
        OR mi.info LIKE 'USA:% 200%')
    AND t.production_year > 2000
    AND t.id = at.movie_id
    AND t.id = mi.movie_id
    AND t.id = mk.movie_id
    AND t.id = mc.movie_id;
