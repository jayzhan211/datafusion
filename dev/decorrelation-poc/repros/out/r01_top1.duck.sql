-- #26710 correlated scalar subquery with ORDER BY ... LIMIT 1
CREATE TABLE invoices (id INT, order_id INT, total INT);
INSERT INTO invoices VALUES (1,1,10),(2,1,20),(3,2,30),(4,2,40);
CREATE TABLE orders (id INT);
INSERT INTO orders VALUES (1),(2),(3);
SELECT o.id, (SELECT i.total FROM invoices i WHERE i.order_id = o.id ORDER BY i.id DESC LIMIT 1) AS last_total FROM orders o ORDER BY o.id;
