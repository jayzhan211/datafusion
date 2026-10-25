-- #4656 row-valued IN
CREATE TABLE t1(id INT, name VARCHAR); INSERT INTO t1 VALUES (1,'aa'), (2,'bb');
CREATE TABLE t2(id INT, name VARCHAR); INSERT INTO t2 VALUES (1,'aa'), (2,'aa');
SELECT id, name FROM t2 WHERE (id, name) IN (SELECT id, name FROM t1 WHERE name = 'aa') ORDER BY 1;
