-- #16134 with the left-join path for uncorrelated scalar subqueries
set datafusion.optimizer.enable_physical_uncorrelated_scalar_subquery = false;
CREATE TABLE s (a VARCHAR); INSERT INTO s VALUES ('a'),('b');
SELECT s.a, (SELECT x FROM (VALUES (1), (2)) t(x)) sub FROM s ORDER BY 1,2;
