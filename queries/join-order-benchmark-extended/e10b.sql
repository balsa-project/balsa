SELECT
    min(n.name),
    min(t.title)
FROM
    info_type AS it1,
    info_type AS it2,
    movie_info_idx AS mi_idx,
    title AS t,
    cast_info AS ci,
    name AS n,
    person_info AS pi
WHERE
    lower(it1.info) LIKE 'rating'
    AND it1.id = mi_idx.info_type_id
    AND t.id = mi_idx.movie_id
    AND t.id = ci.movie_id
    AND ci.person_id = n.id
    AND n.id = pi.person_id
    AND pi.info_type_id = it2.id
    AND lower(it2.info) LIKE '%birth%'
    AND lower(pi.info) LIKE '%usa%'
    AND (lower(mi_idx.info) LIKE '0%'
        OR lower(mi_idx.info) LIKE '1%'
        OR lower(mi_idx.info) LIKE '2%');
