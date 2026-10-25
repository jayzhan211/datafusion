CREATE TABLE o(k INT) AS VALUES (1), (2);
CREATE TABLE i(k INT, v INT) AS VALUES (1, 10), (1, 20), (2, 5);
set datafusion.explain.format = 'indent';
EXPLAIN SELECT o.k, s.c FROM o, LATERAL (SELECT count(*) AS c FROM i WHERE i.k = o.k HAVING count(*) > 1) s ORDER BY 1;
SELECT o.k, s.c FROM o, LATERAL (SELECT count(*) AS c FROM i WHERE i.k = o.k HAVING count(*) = 0) s ORDER BY 1;
