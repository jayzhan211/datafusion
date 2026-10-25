-- subquery inside CASE in a projection, correlated EXISTS and correlated scalar
CREATE TABLE t1(a INT); INSERT INTO t1 VALUES (1), (2);
CREATE TABLE t2(a INT, y INT); INSERT INTO t2 VALUES (1, 5);
SELECT a, CASE WHEN EXISTS (SELECT 1 FROM t2 WHERE t2.a = t1.a) THEN 'y' ELSE 'n' END AS e FROM t1 ORDER BY a;
SELECT a, CASE WHEN a > 1 THEN (SELECT max(y) FROM t2 WHERE t2.a = t1.a) ELSE -1 END AS m FROM t1 ORDER BY a;
