-- #26692 two equal subplans in a subquery
CREATE TABLE t(a INT, b INT) AS VALUES (1,1),(2,2);
CREATE TABLE t2(a INT, b INT) AS VALUES (1,1),(3,30);
SELECT t.a FROM t, t2 AS s WHERE EXISTS (SELECT 1 FROM (SELECT * FROM t2 AS s WHERE s.a = t.a) x CROSS JOIN t2 AS s) ORDER BY 1;
