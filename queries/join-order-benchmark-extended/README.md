# Extended JOB Queries
These queries were designed to test systems trained on the regular JOB queries. They use the same relations, but have very different semantics.

### Changelog
- Formatted queries with [`pg_format`](https://github.com/darold/pgFormatter)
- Changed all instances of `ilike` to `lower(attr) like`
- Changed all instances of alias `mii` and `miidx` to `mi_idx`
- Add `count(*)` in the `ORDER BY` clause of `e12a.sql` and `e12b.sql` to the select as an attribute, and then order by attribute instead of count(*) directly
- Changed all instances of alias `cn` to `chn`
- In `e2a` and `e2b`, there is a cartesian product created by joining `role_type AS rt`. Based on the join graph provided in Leis et al., `role_type` can only be joined with `cast_info`. We add the join condition `ci.role_id = rt.id` to both `e2a` and `e2b`.

Source: https://github.com/RyanMarcus/imdb_pg_dataset/tree/master/job_extended
