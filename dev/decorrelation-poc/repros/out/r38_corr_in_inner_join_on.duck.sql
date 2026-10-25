-- outer reference inside an inner join ON clause of the subquery
CREATE TABLE t1(a INT); INSERT INTO t1 VALUES (1), (2);
CREATE TABLE t2(a INT); INSERT INTO t2 VALUES (1), (2);
CREATE TABLE t3(a INT); INSERT INTO t3 VALUES (1);
SELECT a FROM t1 WHERE EXISTS (SELECT 1 FROM t2 JOIN t3 ON t2.a = t3.a AND t3.a = t1.a) ORDER BY a;
