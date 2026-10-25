-- #14554 correlated disjunction under a groupless aggregate
CREATE TABLE t1 (id INT, name VARCHAR, v INT); INSERT INTO t1 VALUES (1,'a',0),(2,'b',5),(3,'z',1);
CREATE TABLE t2 (id INT, name VARCHAR); INSERT INTO t2 VALUES (1,'b'),(2,'x'),(3,'x');
SELECT t1.id FROM t1 WHERE (SELECT count(*) FROM t2 WHERE t1.id = t2.id OR t1.name = t2.name) > t1.v ORDER BY 1;
