-- #25789 correlated filter below the nullable side of an outer join
CREATE TABLE o(k INT) AS VALUES (1), (5);
CREATE TABLE a(id INT) AS VALUES (1), (2);
CREATE TABLE b(id INT, y INT) AS VALUES (1, 1), (2, 2);
SELECT o.k, EXISTS (SELECT 1 FROM a LEFT JOIN (SELECT * FROM b WHERE b.y = o.k) AS b ON a.id = b.id WHERE b.y IS NULL) AS e FROM o ORDER BY k;
