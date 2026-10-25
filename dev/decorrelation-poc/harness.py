"""Compare every query of the decorrelation repros with DuckDB.

Usage: python3 -I harness.py <datafusion-cli> [repro ...]

Each repro file holds setup statements (CREATE, INSERT, SET) and queries. Every
query runs on its own after the setup, in DataFusion and in DuckDB (from
out/<name>.duck.sql when that exists), and the rows are compared ignoring
column names. Prints one line per query and a summary.
"""

import csv
import io
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPROS = os.path.join(HERE, "repros")
DUCKDB = os.environ.get("DUCKDB", "duckdb")


def statements(path):
    text = "\n".join(
        line for line in open(path).read().splitlines() if not line.strip().startswith("--")
    )
    return [s.strip() for s in text.split(";") if s.strip()]


def is_query(stmt):
    return re.match(r"(?is)^\s*(select|with|\()", stmt) is not None


def run(cmd, sql):
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as f:
        f.write(sql)
        name = f.name
    try:
        if cmd[0] == DUCKDB:
            p = subprocess.run(cmd, stdin=open(name), capture_output=True, text=True, timeout=120)
        else:
            p = subprocess.run(cmd + ["-f", name], capture_output=True, text=True, timeout=120)
        return p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT"
    finally:
        os.unlink(name)


def norm(v):
    if v == "" or v == "NULL":
        return None
    if v in ("true", "false"):
        return v == "true"
    try:
        return round(float(v), 6)
    except ValueError:
        return v


def rows_of(stdout, stderr, skip_lines):
    """The rows of the query result, or None if the query failed."""
    if stderr.strip():
        return None
    rows = list(csv.reader(io.StringIO(stdout)))[skip_lines:]
    if not rows:
        return []
    width = len(rows[0])
    # A row of one NULL column prints as an empty line.
    return [tuple(norm(v) for v in (row or [""] * width)) for row in rows[1:]]


def compare(dfcli, name):
    path = os.path.join(REPROS, name + ".sql")
    duck_path = os.path.join(REPROS, "out", name + ".duck.sql")
    df_stmts = statements(path)
    duck_stmts = statements(duck_path) if os.path.exists(duck_path) else df_stmts
    df_setup = [s for s in df_stmts if not is_query(s)]
    duck_setup = [s for s in duck_stmts if not is_query(s) and not s.lower().startswith("set datafusion")]
    df_queries = [s for s in df_stmts if is_query(s)]
    duck_queries = [s for s in duck_stmts if is_query(s)]
    n_insert = sum(1 for s in df_setup if s.lower().startswith("insert"))
    results = []
    for k, q in enumerate(df_queries):
        dq = duck_queries[k] if k < len(duck_queries) else q
        df_out, df_err = run([dfcli, "-q", "--format", "csv"], ";\n".join(df_setup + [q]) + ";\n")
        duck_out, duck_err = run([DUCKDB, "-csv"], ";\n".join(duck_setup + [dq]) + ";\n")
        df_rows = rows_of(df_out, df_err, 2 * n_insert)
        duck_rows = rows_of(duck_out, duck_err, 0)
        if duck_rows is None:
            status = "DUCK_ERROR"
        elif df_rows is None:
            status = "ERROR"
        elif sorted(df_rows, key=repr) == sorted(duck_rows, key=repr):
            status = "PASS"
        else:
            status = "WRONG"
        detail = ""
        if status == "ERROR":
            err = [l for l in df_err.strip().splitlines() if l.strip()]
            detail = (err[0] if len(err) == 1 else " | ".join(err[:1] + err[-1:]))[:200] if err else ""
        elif status == "WRONG":
            detail = f"df={df_rows[:4]} duck={duck_rows[:4]}"
        elif status == "DUCK_ERROR":
            detail = duck_err.strip().splitlines()[0][:120] if duck_err.strip() else ""
        results.append((f"{name}#{k + 1}", status, detail))
    return results


def main():
    dfcli = sys.argv[1]
    names = sys.argv[2:] or sorted(
        f[:-4] for f in os.listdir(REPROS) if re.match(r"r\d+.*\.sql$", f)
    )
    counts = {}
    for name in names:
        for qid, status, detail in compare(dfcli, name):
            counts[status] = counts.get(status, 0) + 1
            print(f"{status:10} {qid:32} {detail}")
    print("summary:", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
