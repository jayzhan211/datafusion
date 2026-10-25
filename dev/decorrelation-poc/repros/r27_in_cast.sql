-- #26033 correlated IN whose output column is cast
CREATE TABLE o(k INT, b BIGINT) AS VALUES (1, 1), (2, 2), (3, 30);
CREATE TABLE i(y INT) AS VALUES (1), (2), (3);
SELECT o.k FROM o WHERE o.b IN (SELECT i.y FROM i WHERE i.y = o.k) ORDER BY o.k;
