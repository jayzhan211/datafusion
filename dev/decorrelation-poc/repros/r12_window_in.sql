-- #26032 window function above a correlated filter that repeats the IN predicate
CREATE TABLE o(k INT) AS VALUES (1), (2), (5);
CREATE TABLE i(id INT, y INT) AS VALUES (1, 1), (2, 2);
SELECT o.k FROM o WHERE o.k IN (SELECT i.y FROM i WHERE i.y = o.k QUALIFY row_number() OVER (ORDER BY i.id) = 1) ORDER BY o.k;
