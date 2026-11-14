-- One Parquet file read as both join inputs, as in the reproducer of
-- https://github.com/apache/datafusion/issues/24768.
--
-- `k` is unique and in shuffled order: 2654435761 is odd, so multiplying by
-- it is a bijection modulo 2^32. `payload` is 34-41 bytes.
COPY (
  SELECT
    (value * 2654435761) % 4294967296 AS k,
    concat('payload-', value, '-', repeat('x', 24)) AS payload
  FROM range(0, ${HJ_SPILL_ROWS:-20000000})
)
TO 'sql_benchmarks/hj_spill/scratch/t.parquet'
STORED AS PARQUET;

CREATE EXTERNAL TABLE t_build
STORED AS PARQUET
LOCATION 'sql_benchmarks/hj_spill/scratch/t.parquet';

CREATE EXTERNAL TABLE t_probe
STORED AS PARQUET
LOCATION 'sql_benchmarks/hj_spill/scratch/t.parquet';
