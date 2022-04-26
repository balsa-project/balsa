SELECT
    count(*)
FROM
    info_type AS it1,
    info_type AS it2,
    title AS t,
    movie_info AS mi,
    cast_info AS ci,
    name AS n,
    person_info AS pi
WHERE
    t.id = ci.movie_id
    AND ci.person_id = n.id
    AND n.id = pi.person_id
    AND lower(it2.info) LIKE '%birth%'
    AND pi.info_type_id = it2.id
    AND (lower(pi.info) LIKE '%uk%'
        OR lower(pi.info) LIKE '%spain%'
        OR lower(pi.info) LIKE '%germany%'
        OR lower(pi.info) LIKE '%italy%'
        OR lower(pi.info) LIKE '%croatia%'
        OR lower(pi.info) LIKE '%algeria%'
        OR lower(pi.info) LIKE '%south%'
        OR lower(pi.info) LIKE '%austria%'
        OR lower(pi.info) LIKE '%hungary%')
    AND lower(it1.info) LIKE '%count%'
    AND mi.info_type_id = it1.id
    AND t.id = mi.movie_id
    AND lower(mi.info) LIKE '%france%';
