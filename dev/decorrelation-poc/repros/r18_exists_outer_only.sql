-- #26695 EXISTS over a groupless aggregate with an outer-only HAVING
CREATE TABLE t1(a INT) AS VALUES (1), (2), (-1);
CREATE TABLE t2(a INT) AS VALUES (1);
SELECT a FROM t1 WHERE EXISTS (SELECT count(*) FROM t2 WHERE t2.a = t1.a HAVING t1.a > 0) ORDER BY a;
