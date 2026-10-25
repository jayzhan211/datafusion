-- q064 shape: correlated scalar subquery in JOIN ON
CREATE TABLE p(id INT); INSERT INTO p VALUES (1), (2);
CREATE TABLE q(pid INT, v INT); INSERT INTO q VALUES (1, 5), (2, 7);
CREATE TABLE r(pid INT, v INT); INSERT INTO r VALUES (1, 5), (2, 1);
SELECT p.id FROM p JOIN r ON r.pid = p.id AND r.v = (SELECT max(q.v) FROM q WHERE q.pid = p.id) ORDER BY 1;
