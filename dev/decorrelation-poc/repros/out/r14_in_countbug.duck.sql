-- #24960 (IN half): IN over a correlated groupless count
CREATE TABLE t1(a INT); INSERT INTO t1 VALUES (1),(2);
CREATE TABLE t2(a INT, b INT); INSERT INTO t2 VALUES (1,10),(1,NULL);
SELECT a FROM t1 WHERE 0 IN (SELECT count(*) FROM t2 WHERE t2.a = t1.a) ORDER BY a;
