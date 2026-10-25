-- #26505 HAVING that is true for empty input
SELECT x.a, (SELECT sum(y.b) FROM (VALUES (1, 10)) AS y(a, b) WHERE y.a = x.a HAVING sum(y.b) IS NULL) AS v FROM (VALUES (1), (2)) AS x(a) ORDER BY x.a;
