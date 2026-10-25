-- #25808 LATERAL + DISTINCT + non-equality correlation
CREATE TABLE po(k INT) AS VALUES (0), (5);
CREATE TABLE pb(x INT, y INT) AS VALUES (1, 1), (1, 2), (2, 3);
SELECT po.k, s.x FROM po, LATERAL (SELECT DISTINCT pb.x FROM pb WHERE pb.y > po.k) AS s ORDER BY po.k, s.x;
