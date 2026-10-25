-- correlated filter inside a UNION ALL branch of an EXISTS subquery (no open issue found)
CREATE TABLE t1(a INT) AS VALUES (1), (2), (3);
CREATE TABLE t2(a INT) AS VALUES (1);
CREATE TABLE t3(a INT) AS VALUES (3);
SELECT a FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.a = t1.a UNION ALL SELECT 1 FROM t3 WHERE t3.a = t1.a) ORDER BY a;
