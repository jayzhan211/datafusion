-- correlated scalar subquery in SELECT whose correlation is a range predicate, inside a projection with CASE
CREATE TABLE t1(a INT) AS VALUES (1), (2);
CREATE TABLE t2(a INT, y INT) AS VALUES (1, 5), (2, 7);
SELECT a, (SELECT y FROM t2 WHERE t2.a = t1.a) AS y FROM t1 ORDER BY a;
