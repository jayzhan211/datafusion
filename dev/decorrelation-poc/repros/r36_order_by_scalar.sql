-- correlated scalar subquery in ORDER BY
CREATE TABLE t1(a INT) AS VALUES (1), (2);
CREATE TABLE t2(a INT, y INT) AS VALUES (1, 5), (2, 3);
SELECT a FROM t1 ORDER BY (SELECT max(y) FROM t2 WHERE t2.a = t1.a);
