-- #26704 subquery inside an aggregate FILTER / argument
CREATE TABLE t1 (a INT, b VARCHAR, x INT); INSERT INTO t1 VALUES (1, 'p', 10), (2, 'p', 20), (3, 'q', 30), (NULL, 'q', 40), (2, 'q', 50);
CREATE TABLE t2 (a INT, y INT); INSERT INTO t2 VALUES (2, 7), (2, 9), (3, 5), (NULL, 1);
SELECT count(*) FILTER (WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.a = t1.a)) AS n FROM t1;
SELECT t1.b, sum((SELECT max(t2.y) FROM t2 WHERE t2.a = t1.a)) AS s FROM t1 GROUP BY t1.b ORDER BY t1.b;
