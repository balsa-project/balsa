SELECT
    min(t.title),
    min(pi.info)
FROM
    person_info AS pi,
    info_type AS it1,
    info_type AS it2,
    name AS n,
    cast_info AS ci,
    title AS t,
    movie_info AS mi
WHERE
    t.id = mi.movie_id
    AND it2.id = 3
    AND mi.info_type_id = it2.id
    AND lower(mi.info) LIKE '%sci%'
    AND t.id = ci.movie_id
    AND ci.person_id = n.id
    AND n.id = pi.person_id
    AND lower(it1.info) LIKE 'birth date'
    AND pi.info_type_id = it1.id
    AND lower(pi.info) LIKE '%199%';
