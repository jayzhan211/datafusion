"""Run the shape catalog from apache/datafusion#5492 on datafusion-cli binaries and DuckDB.

Usage: python3 -I shape_catalog.py <datafusion-cli> [<datafusion-cli> ...]

Prints one line per shape: DuckDB's row count (or error), then RIGHT, WRONG or ERR
for each binary. Two shapes (an aggregate over outer columns only) are not valid
SQL, so DuckDB rejects them.
"""

import csv
import io
import os
import subprocess
import sys
import tempfile

DUCKDB = os.environ.get("DUCKDB", "duckdb")
SETUP = """CREATE TABLE t1(t1_id INT, t1_name TEXT, t1_int INT) AS VALUES (11, 'a', 1),(22, 'b', 2),(33, 'c', 3),(44, 'd', 4);
CREATE TABLE t2(t2_id INT, t2_name TEXT, t2_int INT) AS VALUES (11, 'z', 3),(22, 'y', 1),(44, 'x', 3),(55, 'w', 3);
CREATE TABLE t3(t3_id INT PRIMARY KEY, t3_name TEXT, t3_int INT) AS VALUES (11, 'e', 3),(22, 'f', 1),(44, 'g', 3),(55, 'h', 3);
"""
DUCK_SETUP = """CREATE TABLE t1(t1_id INT, t1_name TEXT, t1_int INT); INSERT INTO t1 VALUES (11, 'a', 1),(22, 'b', 2),(33, 'c', 3),(44, 'd', 4);
CREATE TABLE t2(t2_id INT, t2_name TEXT, t2_int INT); INSERT INTO t2 VALUES (11, 'z', 3),(22, 'y', 1),(44, 'x', 3),(55, 'w', 3);
CREATE TABLE t3(t3_id INT PRIMARY KEY, t3_name TEXT, t3_int INT); INSERT INTO t3 VALUES (11, 'e', 3),(22, 'f', 1),(44, 'g', 3),(55, 'h', 3);
"""
SHAPES = [
 ("depth-2 skip-level", "SELECT t1.t1_id FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.t2_id = t1.t1_id AND EXISTS (SELECT 1 FROM t3 WHERE t3.t3_id > t1.t1_id))"),
 ("negative control", "SELECT t1.t1_id FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.t2_id = t1.t1_id AND EXISTS (SELECT 1 FROM t3 WHERE t3.t3_id = t2.t2_id))"),
 ("JOIN ON EXISTS both sides", "SELECT t1.t1_id FROM t1 JOIN t2 ON EXISTS (SELECT 1 FROM t3 WHERE t3.t3_id = t1.t1_id AND t3.t3_id = t2.t2_id)"),
 ("EXISTS inside scalar", "SELECT t1.t1_id, (SELECT COUNT(*) FROM t2 WHERE EXISTS (SELECT 1 FROM t3 WHERE t3.t3_id = t2.t2_id AND t3.t3_id = t1.t1_id)) AS c FROM t1"),
 ("outer ref in agg arg", "SELECT t1.t1_int, (SELECT SUM(t2.t2_int + t1.t1_int) FROM t2) FROM t1"),
 ("outer ref above agg", "SELECT t1.t1_int, (SELECT MIN(t2.t2_int) + t1.t1_int FROM t2) AS c FROM t1"),
 ("no FROM, outer-only agg", "SELECT t1.t1_int, (SELECT SUM(t1.t1_int)) AS c FROM t1"),
 ("HAVING outer-only agg", "SELECT t1.t1_id FROM t1 WHERE EXISTS (SELECT t2.t2_id FROM t2 GROUP BY t2.t2_id HAVING t2.t2_id > MIN(t1.t1_int))"),
 ("bare outer col + LIMIT", "SELECT t1.t1_int, (SELECT t1.t1_int FROM t2 LIMIT 1) AS c FROM t1"),
 ("> ALL with outer arith", "SELECT t1.t1_id FROM t1 WHERE t1.t1_id > ALL(SELECT (t2.t2_id + t1.t1_id - 1) / 2 FROM t2 WHERE t2.t2_id IS NOT NULL)"),
 ("inequality under COUNT", "SELECT t1.t1_int, (SELECT COUNT(*) FROM t2 WHERE t2.t2_int > t1.t1_int) AS c FROM t1"),
 ("outer col as GROUPING SETS key", "SELECT t1.t1_int, (SELECT SUM(t2.t2_int) FROM t2 GROUP BY GROUPING SETS((t1.t1_int))) AS c FROM t1"),
]
def run(cmd, sql, stdin):
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as f:
        f.write(sql); n = f.name
    try:
        p = subprocess.run(cmd if stdin else cmd + ["-f", n], stdin=open(n) if stdin else None, capture_output=True, text=True, timeout=120)
    finally:
        os.unlink(n)
    return p.stdout, p.stderr
def rows(out):
    d = list(csv.reader(io.StringIO(out)))
    return sorted(tuple(r) for r in d[1:]) if d else []
def norm(rs):
    return sorted(tuple("NULL" if v in ("", "NULL") else v for v in r) for r in rs)
bins = sys.argv[1:]
for name, q in SHAPES:
    o, e = run([DUCKDB, "-csv"], DUCK_SETUP + q + ";\n", True)
    ref = None if e.strip() else norm(rows(o))
    line = [name, "duck:" + ("ERR " + e.strip().splitlines()[0][:90] if ref is None else f"{len(ref)} rows")]
    for b in bins:
        o, e = run([b, "-q", "--format", "csv"], SETUP + q + ";\n", False)
        el = [l for l in e.strip().splitlines() if l.strip()]
        if el:
            msg = el[-1][:110]
            line.append(f"{os.path.basename(b)}:ERR {msg}")
            continue
        got = norm(rows(o))
        st = "RIGHT" if ref is not None and got == ref else ("WRONG" if ref is not None else "ran")
        line.append(f"{os.path.basename(b)}:{st} {got if st!='RIGHT' else ''}")
    print(" | ".join(line))
