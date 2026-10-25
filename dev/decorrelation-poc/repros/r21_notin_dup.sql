-- #25480 NOT IN whose correlation repeats the IN equality
CREATE TABLE t1(id INT, z INT) AS VALUES (1,10), (2,20), (NULL,30), (4,40);
CREATE TABLE t2(id INT, z INT) AS VALUES (1,5), (NULL,50), (5,10);
SELECT id FROM t1 WHERE id NOT IN (SELECT t2.id FROM t2 WHERE t2.id = t1.id) ORDER BY id;
