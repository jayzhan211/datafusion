-- #15558 depth-2 correlation
CREATE TABLE employees (employee_id INT, employee_name VARCHAR, dept_id INT, salary INT);
INSERT INTO employees VALUES (1,'a',1,10),(2,'b',1,20),(3,'c',1,30),(4,'d',2,40);
SELECT e1.employee_name FROM employees e1 WHERE e1.salary >= (SELECT AVG(e2.salary) FROM employees e2 WHERE e2.dept_id = e1.dept_id AND e2.salary >= (SELECT AVG(e3.salary) FROM employees e3 WHERE e3.dept_id = e1.dept_id)) ORDER BY 1;
