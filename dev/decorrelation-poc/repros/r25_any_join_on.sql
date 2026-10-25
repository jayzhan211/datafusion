-- #26573 = ANY in JOIN ON over an aliased table
CREATE TABLE t (a INT) AS VALUES (1), (2);
CREATE TABLE s (b INT) AS VALUES (1);
SELECT * FROM t AS x JOIN t AS y ON x.a = ANY (SELECT b FROM s) ORDER BY 1,2;
