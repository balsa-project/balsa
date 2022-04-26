SELECT
    n.name
FROM
    title AS t,
    name AS n,
    cast_info AS ci,
    movie_info AS mi,
    info_type AS it1,
    info_type AS it2,
    person_info AS pi
WHERE
    t.id = ci.movie_id
    AND t.id = mi.movie_id
    AND ci.person_id = n.id
    AND it1.id = 3
    AND it1.id = mi.info_type_id
    AND (lower(mi.info) LIKE '%romance%'
        OR lower(mi.info) LIKE '%action%')
    AND lower(it2.info) LIKE '%birth%'
    AND pi.person_id = n.id
    AND pi.info_type_id = it2.id
    AND lower(pi.info) LIKE '%usa%'
GROUP BY
    n.name
ORDER BY
    count(*) DESC;
