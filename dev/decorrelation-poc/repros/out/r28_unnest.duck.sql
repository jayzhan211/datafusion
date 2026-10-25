-- #11773 correlated UNNEST in a subquery
SELECT id FROM (SELECT 1 id, [1, 2] arr UNION ALL SELECT 2 id, [3] arr) r WHERE EXISTS (SELECT 1 FROM UNNEST(arr) AS u(x) WHERE u.x > 2) ORDER BY id;
