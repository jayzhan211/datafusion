-- outer reference in the subquery's SELECT list (scalar), not only in WHERE
CREATE TABLE s(amount INT, cur VARCHAR) AS VALUES (2,'USD'), (3,'EUR');
CREATE TABLE e(amount INT, cur VARCHAR) AS VALUES (10,'USD'), (20,'EUR');
SELECT (SELECT max(e.amount) * s.amount FROM e WHERE e.cur = s.cur) AS x FROM s ORDER BY 1;
SELECT (SELECT max(e.amount + s.amount) FROM e WHERE e.cur = s.cur) AS x FROM s ORDER BY 1;
