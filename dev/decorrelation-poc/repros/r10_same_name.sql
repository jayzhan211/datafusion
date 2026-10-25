-- #26234 same-named correlated columns of two inner relations
CREATE TABLE test(k INT) AS VALUES (1), (2);
SELECT outer_t.k FROM test outer_t WHERE EXISTS (SELECT 1 FROM test one, test two WHERE one.k = outer_t.k AND two.k = outer_t.k + 1) ORDER BY 1;
