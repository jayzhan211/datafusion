-- #16137 correlated scalar subquery without aggregation
CREATE TABLE src (id INT, parent_id INT, name VARCHAR);
INSERT INTO src VALUES (1, NULL, 'Europe'), (2, 1, 'Warsaw'), (3, 1, 'Paris');
SELECT id, name, (SELECT p.name FROM src p WHERE p.id = s.parent_id) AS parent_name FROM src s ORDER BY id;
