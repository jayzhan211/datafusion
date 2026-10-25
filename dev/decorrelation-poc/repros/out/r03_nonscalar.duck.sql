-- #16134 scalar subquery returning 2 rows, uncorrelated (default config) and correlated
CREATE TABLE s (a VARCHAR); INSERT INTO s VALUES ('a'),('b');
CREATE TABLE d (k INT, v INT); INSERT INTO d VALUES (1,10),(1,11);
SELECT s.a, (SELECT x FROM (VALUES (1), (2)) t(x)) sub FROM s ORDER BY 1,2;
SELECT k, (SELECT max(v) FROM d d2 WHERE d2.k = d.k GROUP BY d2.k, d2.v) FROM d ORDER BY 1,2;
