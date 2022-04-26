SELECT
    count(*)
FROM
    title AS t,
    movie_keyword AS mk,
    keyword AS k,
    info_type AS it,
    movie_info AS mi
WHERE
    it.id = 3
    AND it.id = mi.info_type_id
    AND mi.movie_id = t.id
    AND mk.keyword_id = k.id
    AND mk.movie_id = t.id
    AND lower(k.keyword) LIKE '%fight%'
    AND lower(mi.info) LIKE '%action%';
