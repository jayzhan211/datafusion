-- range-correlated aggregate (dbolympics shape, #14554 area)
CREATE TABLE o (k INT); INSERT INTO o VALUES (0),(2),(5);
CREATE TABLE i (y INT); INSERT INTO i VALUES (1),(2),(3);
SELECT o.k, (SELECT count(*) FROM i WHERE i.y > o.k) AS c FROM o ORDER BY 1;
