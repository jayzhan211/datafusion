-- #25837 LATERAL with a derived-table alias that differs from the table name
CREATE TABLE o(k INT) AS VALUES (1), (2);
CREATE TABLE l(id INT, v INT) AS VALUES (1, 10), (2, 20);
SELECT o.k, s.v FROM o, LATERAL (SELECT l2.v FROM (SELECT * FROM l WHERE l.id = o.k) AS l2) AS s ORDER BY o.k;
