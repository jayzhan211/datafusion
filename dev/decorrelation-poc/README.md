# Decorrelation prototype: repros and fuzzer

These scripts produced the numbers in
[#5492](https://github.com/apache/datafusion/issues/5492#issuecomment-6395684651).
Each one runs `datafusion-cli` binaries and compares their rows with a reference
engine. Nothing here is meant to be merged as is.

You need Python 3, `duckdb` (1.1.3 was used) and `sqlite3` on `PATH`, or set
`DUCKDB` and `SQLITE` to their paths. Build `datafusion-cli` from this branch
and from `main`, and pass both binaries to compare them.

| Script | What it runs |
|---|---|
| `harness.py <datafusion-cli> [repro ...]` | The 44 repros in `repros/`, mostly one per open issue (named in each file's first line). Every query runs after the file's setup and is compared with DuckDB. `repros/out/*.duck.sql` holds the DuckDB version of a repro whose syntax differs. |
| `fuzz.py <n> <seed> <out_dir> <datafusion-cli> ...` | `n` random correlated queries. Writes `<out_dir>/<binary>.tsv` with one line per query: id, status (`PASS`, `WRONG`, `ERROR` or `SKIP`), reference engine, query. The #5492 numbers used 3,000 queries for each of the seeds 7, 11, 23, 37 and 41. |
| `shape_catalog.py <datafusion-cli> ...` | The 11 shapes of the catalog linked in #5492, plus one control query. |

Run every script with `python3 -I`.

The fuzzer uses SQLite as the reference, because SQLite runs a correlated
subquery once per outer row. It falls back to DuckDB when SQLite can't run the
query (most `LATERAL` queries). The reference column reads `sqlite!=duckdb`
when the two engines disagree. DuckDB 1.1.3 itself returns wrong rows for a few
shapes: `NOT EXISTS` with a `<>` correlation on NULLs, correlated grouping sets,
and `LATERAL` with a nested `EXISTS` on NULL outer values.
