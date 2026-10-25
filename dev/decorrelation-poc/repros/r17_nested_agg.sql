-- #26508 aggregate above a correlated groupless aggregate
SELECT x.a, (SELECT count(*) FROM (SELECT max(y.b) FROM (VALUES (1, 10)) AS y(a, b) WHERE y.a = x.a)) AS v1 FROM (VALUES (1), (2)) AS x(a) ORDER BY x.a;
