-- #21201 LATERAL referencing the outer row outside WHERE
CREATE TABLE o(k INT) AS VALUES (1), (2);
CREATE TABLE i(k INT, v INT) AS VALUES (1, 10), (2, 20);
SELECT o.k, s.w FROM o, LATERAL (SELECT i.v + o.k AS w FROM i WHERE i.k = o.k) s ORDER BY 1;
