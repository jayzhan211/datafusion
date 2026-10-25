"""Randomized differential test of correlated subqueries against SQLite and DuckDB.

Usage: python3 -I fuzz.py <n_queries> <seed> <out_dir> <datafusion-cli> [<datafusion-cli> ...]

Generates queries that combine subquery kinds (scalar, top-1, EXISTS, IN,
NOT IN, LATERAL), positions (SELECT, WHERE, aggregate argument, ORDER BY) and
operators inside the subquery (filters with equality and range correlation,
GROUP BY, HAVING, DISTINCT, LIMIT, window, UNION ALL, outer joins, nesting).
Runs each query in SQLite, DuckDB and every given DataFusion binary, and writes
<out_dir>/<binary name>.tsv with one line per query: id, status, reference,
query. The reference is SQLite, which runs a correlated subquery once per outer
row, or DuckDB for what SQLite lacks (LATERAL); queries where the two disagree
are marked in the reference column. Status is PASS, WRONG (rows differ from the
reference), ERROR (DataFusion fails) or SKIP (no reference result).
"""

import concurrent.futures
import csv
import io
import os
import random
import subprocess
import sys
import tempfile

DUCKDB = os.environ.get("DUCKDB", "duckdb")
SQLITE = os.environ.get("SQLITE", "sqlite3")

SETUP = """
CREATE TABLE t1(a INT, b INT, c INT);
INSERT INTO t1 VALUES (1,1,1),(1,2,NULL),(2,2,3),(2,NULL,1),(3,1,2),(NULL,2,2),(4,4,4),(1,1,1);
CREATE TABLE t2(a INT, b INT, c INT);
INSERT INTO t2 VALUES (1,1,5),(1,3,NULL),(2,2,1),(2,2,2),(NULL,1,3),(3,NULL,1),(5,5,5),(1,NULL,2);
CREATE TABLE t3(a INT, b INT, c INT);
INSERT INTO t3 VALUES (1,2,1),(2,1,NULL),(2,2,2),(NULL,NULL,1),(3,3,3),(1,1,2);
"""
N_INSERTS = 3


class Gen:
    def __init__(self, rng):
        self.r = rng
        self.lite_rewrites = []

    def pick(self, xs):
        return self.r.choice(xs)

    def col(self, t):
        return f"{t}.{self.pick('abc')}"

    def corr(self, inner, outer):
        """A correlation conjunct between inner and outer."""
        i, o = self.col(inner), self.col(outer)
        return self.pick([
            f"{i} = {o}", f"{i} = {o}", f"{i} = {o}",
            f"{i} < {o}", f"{i} >= {o}", f"{i} <> {o}",
            f"{i} = {o} + 1", f"{i} IS NOT DISTINCT FROM {o}",
            f"({i} = {o} OR {i} IS NULL)",
        ])

    def local(self, t):
        c = self.col(t)
        return self.pick([f"{c} > 1", f"{c} IS NULL", f"{c} IS NOT NULL", f"{c} <> 2", f"{c} <= 3"])

    def where(self, inner, outer):
        conj = [self.corr(inner, outer)]
        if self.r.random() < 0.4:
            conj.append(self.corr(inner, outer))
        if self.r.random() < 0.4:
            conj.append(self.local(inner))
        return " AND ".join(conj)

    def source(self, inner, outer):
        """FROM clause of the subquery, with the correlation somewhere in it."""
        choice = self.r.random()
        if choice < 0.6:
            return inner, self.where(inner, outer)
        if choice < 0.75:
            # correlated filter below the nullable side of an outer join
            jt = self.pick(["LEFT JOIN", "JOIN", "RIGHT JOIN"])
            return (f"{inner} {jt} (SELECT * FROM t3 WHERE {self.where('t3', outer)}) t3 "
                    f"ON t3.a = {inner}.a", self.local(inner))
        if choice < 0.85:
            # correlated filter inside a UNION ALL
            return (f"(SELECT a, b, c FROM {inner} WHERE {self.where(inner, outer)} "
                    f"UNION ALL SELECT a, b, c FROM t3 WHERE {self.where('t3', outer)}) {inner}", "TRUE")
        if choice < 0.93:
            # a window over the correlated input
            return (f"(SELECT a, b, row_number() OVER (PARTITION BY a ORDER BY b NULLS LAST, c NULLS LAST) AS c "
                    f"FROM {inner} WHERE {self.where(inner, outer)}) {inner}", "TRUE")
        # nested: a second-level subquery that reads both levels
        return (inner, f"{self.where(inner, outer)} AND EXISTS (SELECT 1 FROM t3 WHERE "
                       f"t3.a = {inner}.b AND {self.corr('t3', outer)})")

    def agg(self, t):
        c = self.col(t)
        return self.pick(["count(*)", f"count({c})", f"sum({c})", f"min({c})", f"max({c})"])

    def scalar(self, outer):
        """A scalar subquery reading `outer`."""
        frm, where = self.source("t2", outer)
        kind = self.r.random()
        if kind < 0.55:
            having = ""
            if self.r.random() < 0.25:
                having = " HAVING " + self.pick(["count(*) > 1", "count(*) = 0", "sum(t2.b) IS NULL", "max(t2.a) > 1"])
            expr = self.agg("t2")
            if self.r.random() < 0.2:
                expr = f"{expr} + 1"
            return f"(SELECT {expr} FROM {frm} WHERE {where}{having})"
        if kind < 0.8:
            # top-1 with a total order
            off = " OFFSET 1" if self.r.random() < 0.15 else ""
            return (f"(SELECT t2.{self.pick('abc')} FROM {frm} WHERE {where} "
                    f"ORDER BY t2.a DESC NULLS LAST, t2.b NULLS LAST, t2.c NULLS LAST LIMIT 1{off})")
        # grouped by the correlated column: at most one row
        return (f"(SELECT {self.agg('t2')} FROM t2 WHERE t2.a = {outer}.a "
                f"AND {self.local('t2')} GROUP BY t2.a)")

    def predicate_subquery(self, outer):
        frm, where = self.source("t2", outer)
        sel = self.pick(["1", "t2.a", "DISTINCT t2.b", "t2.c"])
        tail = self.pick(["", "", "", " LIMIT 1", " GROUP BY t2.b", " ORDER BY 1",
                          " GROUP BY ROLLUP(t2.b)", " GROUP BY GROUPING SETS ((t2.b), ())"])
        if "GROUP BY" in tail:
            sel = "t2.b"
        kind = self.pick(["EXISTS", "NOT EXISTS", "IN", "NOT IN", "IN"])
        # A LIMIT picks a row only up to the order, which must then be total.
        if tail == " LIMIT 1" and kind in ("IN", "NOT IN"):
            tail = " ORDER BY 1 NULLS LAST LIMIT 1" if sel.startswith("DISTINCT") else (
                " ORDER BY t2.a NULLS LAST, t2.b NULLS LAST, t2.c NULLS LAST LIMIT 1")
        # SQLite has no grouping sets; ((b), ()) is the groups of b plus one
        # grand-total row.
        lite_tail = tail
        if "ROLLUP" in tail or "GROUPING SETS" in tail:
            lite_tail = " GROUP BY t2.b UNION ALL SELECT NULL"
        if kind in ("IN", "NOT IN"):
            if sel in ("1",):
                sel = "t2.a"
            c = self.col(outer)
            q = f"{c} {kind} (SELECT {sel} FROM {frm} WHERE {where}"
            return self.lite(q + tail + ")", q + lite_tail + ")")
        q = f"{kind} (SELECT {sel} FROM {frm} WHERE {where}"
        return self.lite(q + tail + ")", q + lite_tail + ")")

    def lite(self, q, lite_q):
        """Records the SQLite spelling of the fragment `q`."""
        if q != lite_q:
            self.lite_rewrites.append((q, lite_q))
        return q

    def query(self):
        o = "t1"
        shape = self.r.random()
        if shape < 0.3:
            return f"SELECT t1.a, t1.b, {self.scalar(o)} AS v FROM t1"
        if shape < 0.5:
            return f"SELECT t1.a, t1.b FROM t1 WHERE {self.predicate_subquery(o)}"
        if shape < 0.6:
            return f"SELECT t1.a, t1.c, {self.predicate_subquery(o)} AS m FROM t1"
        if shape < 0.68:
            op = self.pick(["=", "<", ">=", "<>"])
            return f"SELECT t1.a, t1.b FROM t1 WHERE t1.{self.pick('abc')} {op} {self.scalar(o)}"
        if shape < 0.74:
            return (f"SELECT t1.a, t1.b FROM t1 WHERE t1.c > 2 OR {self.predicate_subquery(o)}")
        if shape < 0.82:
            frm, where = self.source("t2", o)
            body = self.pick([
                f"SELECT {self.agg('t2')} AS x FROM {frm} WHERE {where}",
                f"SELECT t2.a AS x FROM {frm} WHERE {where}",
                f"SELECT t2.b AS x FROM {frm} WHERE {where} ORDER BY t2.a NULLS LAST, t2.b NULLS LAST, t2.c NULLS LAST LIMIT 2",
                f"SELECT DISTINCT t2.b AS x FROM {frm} WHERE {where}",
            ])
            jt = self.pick(["CROSS JOIN LATERAL", "LEFT JOIN LATERAL"])
            on = " ON TRUE" if jt.startswith("LEFT") else ""
            return f"SELECT t1.a, t1.b, s.x FROM t1 {jt} ({body}) s{on}"
        if shape < 0.86:
            return f"SELECT t1.b, sum({self.scalar(o)}) AS s FROM t1 GROUP BY t1.b"
        if shape < 0.9:
            frm, where = self.source("t2", o)
            q = (f"SELECT t1.a, t1.b, s.x, s.n FROM t1 CROSS JOIN LATERAL ("
                 f"SELECT t2.b AS x, count(*) AS n FROM {frm} WHERE {where} "
                 f"GROUP BY {self.pick(['ROLLUP(t2.b)', 'GROUPING SETS ((t2.b), ())', 'CUBE(t2.b)'])}) s")
            if frm == "t2":
                # Without LATERAL: the groups of the matching rows per outer
                # row, plus one grand-total row for every outer row.
                lite_q = (f"SELECT t1.a, t1.b, t2.b AS x, count(*) AS n FROM t1 JOIN t2 ON {where} "
                          f"GROUP BY t1.rowid, t1.a, t1.b, t2.b "
                          f"UNION ALL SELECT t1.a, t1.b, NULL, (SELECT count(*) FROM t2 WHERE {where}) FROM t1")
                self.lite(q, lite_q)
            return q
        if shape < 0.95:
            return (f"SELECT t1.b, count(*) FILTER (WHERE {self.predicate_subquery(o)}) AS n "
                    f"FROM t1 GROUP BY t1.b")
        return f"SELECT t1.a, t1.b, t1.c FROM t1 ORDER BY {self.scalar(o)} NULLS LAST, t1.a, t1.b, t1.c"


def run(cmd, sql):
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as f:
        f.write(sql)
        name = f.name
    try:
        if cmd[0] in (DUCKDB, SQLITE):
            p = subprocess.run(cmd, stdin=open(name), capture_output=True, text=True, timeout=60)
        else:
            p = subprocess.run(cmd + ["-f", name], capture_output=True, text=True, timeout=60)
        return p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT"
    finally:
        os.unlink(name)


def norm(v):
    if v == "" or v == "NULL":
        return None
    if v in ("true", "false"):
        return 1.0 if v == "true" else 0.0
    try:
        return round(float(v), 6)
    except ValueError:
        return v


def rows_of(stdout, stderr, skip):
    if stderr.strip():
        return None
    rows = list(csv.reader(io.StringIO(stdout)))[skip:]
    if not rows:
        return []
    width = len(rows[0])
    return sorted((tuple(norm(v) for v in (row or [""] * width)) for row in rows[1:]), key=repr)


def check(qs, binaries):
    q, lite_q = qs
    duck = rows_of(*run([DUCKDB, "-csv"], SETUP + q + ";\n"), 0)
    lite = rows_of(*run([SQLITE, "-csv", "-header", ":memory:"], SETUP + lite_q + ";\n"), 0)
    if lite is not None:
        ref, ref_name = lite, "sqlite" if duck is None or duck == lite else "sqlite!=duckdb"
    else:
        ref, ref_name = duck, "duckdb"
    out = []
    for b in binaries:
        df = rows_of(*run([b, "-q", "--format", "csv"], SETUP + q + ";\n"), 2 * N_INSERTS)
        if ref is None:
            out.append(("SKIP", ref_name))
        elif df is None:
            out.append(("ERROR", ref_name))
        elif df == ref:
            out.append(("PASS", ref_name))
        else:
            out.append(("WRONG", ref_name))
    return out


def main():
    n, seed, out_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    binaries = sys.argv[4:]
    os.makedirs(out_dir, exist_ok=True)
    g = Gen(random.Random(seed))
    queries = []
    for _ in range(n):
        g.lite_rewrites = []
        q = g.query()
        lite_q = q
        for frag, lite_frag in g.lite_rewrites:
            lite_q = lite_q.replace(frag, lite_frag)
        queries.append((q, lite_q))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda q: check(q, binaries), queries))
    for bi, b in enumerate(binaries):
        counts = {}
        with open(os.path.join(out_dir, os.path.basename(b) + ".tsv"), "w") as f:
            for k, ((q, _), r) in enumerate(zip(queries, results)):
                status, ref = r[bi]
                counts[status] = counts.get(status, 0) + 1
                f.write(f"{k}\t{status}\t{ref}\t{q}\n")
        print(os.path.basename(b), " ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
