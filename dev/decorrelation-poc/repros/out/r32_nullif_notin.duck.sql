-- #25024 nullable expression over a NOT NULL column in NOT IN
CREATE TABLE outer_t(id VARCHAR NOT NULL); INSERT INTO outer_t VALUES ('z');
CREATE TABLE inner_t(id VARCHAR NOT NULL); INSERT INTO inner_t VALUES ('a');
SELECT id FROM outer_t WHERE (NULLIF(id, 'z') NOT IN (SELECT id FROM inner_t)) IS NULL;
