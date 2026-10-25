-- #26503 count(*) HAVING on an outer column
SELECT x.a, x.d, (SELECT count(*) FROM (VALUES (1)) AS y(c) WHERE y.c = x.d HAVING count(*) >= x.a) AS v FROM (VALUES (1, 1), (2, 1), (1, 9)) AS x(a, d) ORDER BY x.d, x.a;
