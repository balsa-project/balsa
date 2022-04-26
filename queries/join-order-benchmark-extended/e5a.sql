SELECT
    MIN(mc.note) AS production_note,
    MIN(t.title) AS movie_title,
    MIN(t.production_year) AS movie_year
FROM
    company_type ct,
    movie_companies mc,
    title t
WHERE
    ct.kind = 'production companies'
    AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
    AND t.production_year > 2008
    AND ct.id = mc.company_type_id
    AND t.id = mc.movie_id;
