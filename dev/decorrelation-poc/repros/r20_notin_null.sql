-- #25347 correlated NOT IN with a NULL in the subquery
CREATE TABLE t_out(id INT, g INT) AS VALUES (1, 1), (2, 2);
CREATE TABLE t_in(id INT, g INT)  AS VALUES (1, 1), (NULL, 2);
SELECT id FROM t_out WHERE 3 NOT IN (SELECT id FROM t_in WHERE t_in.g = t_out.g) ORDER BY id;
