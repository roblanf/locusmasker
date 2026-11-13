"""Empirical validation of locusmasker by random locus swapping (the planted-swap benchmark).

For each benchmark dataset: convert to FASTA+partitions once, run a baseline, then in each
replicate plant swap events (pick two random taxa, exchange their sequence in one locus) with
partners drawn log-uniformly across the divergence spectrum, rerun the tool, and score:
  - detection per planted cell (both sides of each swap), with the pair divergence m(a,b): flag (fails the
    too-divergent test), amb (fails it with fewer comparators than requested), sim (masked as the identified
    recipient of a too-similar pair), miss, untested (no comparators at the locus)
  - false positives: cells masked in the run but neither planted nor masked in the baseline, split into
    adjacent (a swapped taxon is among the cell's comparators or too-similar partners at that locus) and clean
A run is read from its masked.csv (the masked sequences) and report.npz (every tested cell); the genome-wide
divergence matrix, which a run no longer writes, comes from lo.divergence_matrix once per dataset.

Layout under --bench (default validation/bench):
  <dataset>/base.fasta, base.parts.txt, meta.json   the converted input, shared by every tool version
  <dataset>/v<version>/baseline/                     the tool's outputs on the unmodified data (files over 50 MB other
                                                     than masked.csv and report.npz are dropped)
  <dataset>/v<version>/baseline.json                 natural masks, tested loci, tested cells, the tool that made them
  <dataset>/v<version>/m.npz                         names and m (%, NaN where undefined), as the tool computes them
  results/v<version>/<dataset>_rep<r>.json           one per job
Each tool version gets its own baselines and results, so an earlier benchmark is never overwritten. Within a version a
baseline or result made by a different locusmasker.py (sha256) is redone: a stale baseline would count the code
change as false positives.

Usage: python swap_bench.py pilot|full [--bench DIR] [--datasets A,B] [--reps N] [--workers 16]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # one thread per worker, in this process
    os.environ[_v] = "1"                                                    # and in the tool runs it starts
import argparse, csv, hashlib, io, json, re, shutil, subprocess, sys, time, zipfile, zlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{REPO}/test_data"
HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = f"{REPO}/locusmasker.py"
HEAVY = {"Wu_2018_dna", "Wu_2018_aa", "Shen_2018", "Jarvis_2015", "Irisarri_2017", "Misof_2014"}
MIN_LOCI = 10                                   # the tool refuses an alignment with fewer loci
DATATYPE = {"dna": "dna", "aa": "aa", "protein": "aa"}   # meta.json's datatype (as the NEXUS file declares it) -> --datatype
sys.path.insert(0, REPO)
import locusmasker as lo
TAG = f"v{lo.__version__}"


def tool_id():
    """the tool as it is on disk now: version and the start of its sha256. A job is comparable with its baseline only
    when both were made by the same file"""
    src = open(TOOL, "rb").read()
    ver = re.search(rb'^__version__ = "([^"]+)"', src, re.M)
    return f"{ver.group(1).decode() if ver else '?'}+{hashlib.sha256(src).hexdigest()[:12]}"


def compress_ranges(sites):
    """0-based sorted site array -> RAxML range string (1-based, with \\step runs)"""
    runs, i = [], 0
    while i < len(sites):
        j = i + 1
        step = sites[j] - sites[i] if j < len(sites) else 1
        while j + 1 < len(sites) and sites[j + 1] - sites[j] == step: j += 1
        if j == i + 1 and step != 1:                       # lone pair with odd step: emit singly
            runs.append(f"{sites[i] + 1}"); i += 1; continue
        a, b = sites[i], sites[j if j < len(sites) else i]
        runs.append(f"{a + 1}" if a == b else f"{a + 1}-{b + 1}" + (f"\\{step}" if step != 1 else ""))
        i = j + 1
    return ", ".join(runs)


def prep(bench, ds):
    out = f"{bench}/{ds}"; os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/meta.json"):
        meta, st = json.load(open(f"{out}/meta.json")), "cached"
    else:
        try:
            names, seqs, loci, fmt, dtype, _ = lo.read_input(f"{DATA}/{ds}/alignment.nex", None)
        except SystemExit as e:
            return ds, f"unusable: {e}"
        meta, st = {"ntax": len(names), "nchar": len(seqs[0]), "nloci": len(loci), "dtype": dtype}, "ok"
        if len(loci) >= MIN_LOCI and dtype in DATATYPE:                 # an unusable dataset keeps only its meta.json
            with open(f"{out}/base.fasta", "w") as fh:
                for t, s in zip(names, seqs): fh.write(f">{t}\n{s if isinstance(s, str) else bytes(s).decode()}\n")   # the readers return bytes
            with open(f"{out}/base.parts.txt", "w") as fh:
                for name, sites in loci: fh.write(f"DNA, {name} = {compress_ranges(sites)}\n")
        json.dump(meta, open(f"{out}/meta.json", "w"))
    if meta["nloci"] < MIN_LOCI: return ds, f"unusable: {meta['nloci']} loci (the tool needs {MIN_LOCI})"
    if meta["dtype"] not in DATATYPE: return ds, f"unusable: the alignment declares no datatype ({meta['dtype']})"
    return ds, st


def run_tool(aln, parts, out, jobdir, dtype, lean=False):
    """one single-threaded run with default settings; returns the tool's identity and its log (stderr). lean: no
    sequences or site codes in the report (a swap job reads only the cell columns; the decisions are unchanged)"""
    tid = tool_id()
    cmd = [sys.executable, TOOL, "--aln", aln, "--partitions", parts, "--out", out, "--datatype", DATATYPE[dtype], "--threads", "1"]
    if lean: cmd += ["--report-codes", "none", "--report-sequences", "none"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10800, cwd=jobdir)
    if r.returncode != 0: raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "failed")
    return tid, r.stderr


class Run:
    """a finished run as the benchmark reads it: masked.csv (one row per masked sequence) and report.npz (taxa x loci
    cell columns; a cell is tested when it has comparators at its locus)"""
    def __init__(self, out):
        self.flags = {(r["locus"], r["taxon"]) for r in csv.DictReader(open(f"{out}/masked.csv", newline=""))}
        self.z = zipfile.ZipFile(f"{out}/report.npz"); self.members = set(self.z.namelist())
        self.taxa = self.json("taxa.json")["names"]; loci = self.json("loci.json")
        self.loci, self.status = loci["names"], loci["status"]
        self.ti = {t: i for i, t in enumerate(self.taxa)}; self.li = {l: i for i, l in enumerate(self.loci)}
        for k in ("n_comp", "n_fail", "n_similar", "side", "excess_m", "partner", "comparators"): setattr(self, k, self.npy(f"cells/{k}.npy"))
        self.rows = None                                    # sparse numeric columns: (taxon, locus) -> row, for cells with a fail
        if "cells/sparse_taxon.npy" in self.members:
            self.rows = {(int(t), int(l)): j for j, (t, l) in enumerate(zip(self.npy("cells/sparse_taxon.npy").tolist(), self.npy("cells/sparse_locus.npy").tolist()))}
        self.n_cells = int((self.n_comp > 0).sum()); self._sim = {}

    def npy(self, k): return np.load(io.BytesIO(self.z.read(k)))
    def json(self, k): return json.loads(self.z.read(k))

    def num(self, col, t, l):
        if self.rows is None: return col[t, l]
        j = self.rows.get((t, l)); return None if j is None else col[j]

    def cell(self, locus, taxon):
        """a tested cell's verdict (det), comparators failed (nf), significant too-similar partners (ns), its role in the
        reported pair and that partner (par), and its excess divergence in units of m (exm); None when not tested. Side
        bits: 1|16 too divergent, 4 with fewer comparators than requested, 2|32 identified recipient, 128 donor"""
        t, l = self.ti.get(taxon), self.li.get(locus)
        if t is None or l is None or not self.n_comp[t, l]: return None
        s, ns = int(self.side[t, l]), int(self.n_similar[t, l])
        det = ("amb" if s & 4 else "flag") if s & 17 else ("sim" if (locus, taxon) in self.flags else "miss")
        p, x = self.num(self.partner, t, l), self.num(self.excess_m, t, l)
        return dict(det=det, nf=int(self.n_fail[t, l]), ns=ns, role="" if not ns else "recipient" if s & 34 else "donor" if s & 128 else "undetermined",
                    par=self.taxa[int(p)] if p is not None and p >= 0 else None, exm=round(float(x), 3) if x is not None and np.isfinite(x) else None)

    def near(self, locus, taxon):
        """the taxa a cell was held against at its locus: its comparators and its significant too-similar partners"""
        t, l = self.ti[taxon], self.li[locus]
        c = self.num(self.comparators, t, l)
        out = {self.taxa[x] for x in (c.tolist() if c is not None else []) if x >= 0}
        if l not in self._sim:
            k = f"similar/{l}.json"; self._sim[l] = self.json(k) if k in self.members else []
        for e in self._sim[l]:                                   # 0.12: one entry per too-similar test (taxon, partner)
            if e.get("fail") and t in (e["taxon"], e.get("partner", -1)): out.add(self.taxa[e["partner"] if e["taxon"] == t else e["taxon"]])
        return out


def baseline(bench, ds):
    d = f"{bench}/{ds}"; v = f"{d}/{TAG}"; out = f"{v}/baseline"
    if os.path.exists(f"{v}/baseline.json") and os.path.exists(f"{v}/m.npz") and json.load(open(f"{v}/baseline.json")).get("tool") == tool_id():
        return ds, "cached"
    t0 = time.time(); dtype = json.load(open(f"{d}/meta.json"))["dtype"]
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)            # an earlier build of this version: redone whole
    tid, log = run_tool(f"{d}/base.fasta", f"{d}/base.parts.txt", out, v, dtype)
    open(f"{out}/run.log", "w").write(log)
    R = Run(out)
    tested = [nm for nm, st in zip(R.loci, R.status) if st == "tested"]
    buf = io.StringIO(); real, lo.LOG.real = lo.LOG.real, buf          # the matrix's log goes to run.log, not the console
    try: names, m, _ = lo.divergence_matrix(f"{d}/base.fasta", f"{d}/base.parts.txt", DATATYPE[dtype], fit_overlap=20, n_comp=5, threads=1)
    finally: lo.LOG.real = real
    open(f"{out}/run.log", "a").write("\nthe benchmark's divergence matrix (lo.divergence_matrix):\n" + buf.getvalue())
    np.savez_compressed(f"{v}/m.npz", names=np.array(names), m=m)
    for f in os.listdir(out):                                           # keep the small files only
        if f not in ("masked.csv", "report.npz") and os.path.getsize(f"{out}/{f}") > 50e6: os.remove(f"{out}/{f}")
    json.dump({"flags": sorted(map(list, R.flags)), "tested_loci": tested, "n_cells": R.n_cells, "tool": tid,
               "runtime": time.time() - t0}, open(f"{v}/baseline.json", "w"))
    return ds, f"ok {time.time() - t0:.0f}s, {len(R.flags)} natural masks, {len(tested)} loci tested"


def swap_job(bench, ds, rep):
    d = f"{bench}/{ds}"; v = f"{d}/{TAG}"; res = f"{bench}/results/{TAG}/{ds}_rep{rep}.json"
    if os.path.exists(res) and json.load(open(res)).get("tool") == tool_id(): return ds, rep, "cached"
    t0 = time.time()
    meta = json.load(open(f"{d}/meta.json")); base = json.load(open(f"{v}/baseline.json"))
    names, seqs = lo.read_fasta(open(f"{d}/base.fasta").read())
    n, L = len(names), len(seqs[0])
    loci = lo.parse_raxml_partitions(f"{d}/base.parts.txt", L)
    M = np.load(f"{v}/m.npz"); m = M["m"]
    bidx = {t: i for i, t in enumerate(M["names"].tolist())}
    alphabet = lo.DNA if meta["dtype"] == "dna" else lo.PROTEIN
    lut = np.full(256, -1, dtype=np.int8)
    for i, c in enumerate(alphabet): lut[ord(c)] = i; lut[ord(c.lower())] = i
    arr = np.frombuffer(b"".join(q.encode() if isinstance(q, str) else bytes(q) for q in seqs), dtype=np.uint8).reshape(n, L).copy()
    pres = lut[arr] >= 0
    tested = [(nm, s) for nm, s in loci if nm in set(base["tested_loci"])]
    rng = np.random.default_rng(zlib.crc32(f"{ds}_rep{rep}".encode()))     # reproducible across processes
    used, events, percap = set(), [], {}
    target = min(24, max(2, n // 6))
    for _ in range(40 * target):
        if len(events) >= target: break
        nm, sites = tested[rng.integers(len(tested))]
        npl = pres[:, sites].sum(1)
        cand = [i for i in range(n) if npl[i] >= 20 and i not in used]
        cap = max(1, int(0.02 * (npl >= 20).sum()))
        if len(cand) < 2 or percap.get(nm, 0) >= cap: continue
        a = int(rng.choice(cand))
        parts = [(m[bidx[names[a]], bidx[names[b]]], b) for b in cand if b != a]
        parts = sorted((p for p in parts if np.isfinite(p[0])))
        if not parts: continue
        j = min(int(len(parts) ** rng.random()) - 1, len(parts) - 1)
        m_ab, b = parts[max(j, 0)]
        if (pres[a, sites] & pres[b, sites]).sum() < 20: continue
        tmp = arr[a, sites].copy(); arr[a, sites] = arr[b, sites]; arr[b, sites] = tmp
        used.update((a, b)); percap[nm] = percap.get(nm, 0) + 1
        events.append({"locus": nm, "a": names[a], "b": names[b], "m_ab": float(m_ab),
                       "lsites": len(sites)})
    jobdir = f"{v}/rep{rep}"; shutil.rmtree(jobdir, ignore_errors=True); os.makedirs(jobdir)
    mod = f"{jobdir}/mod.fasta"
    with open(mod, "w") as fh:
        for i, t in enumerate(names): fh.write(f">{t}\n{arr[i].tobytes().decode()}\n")
    out = f"{jobdir}/scan"
    err, tid, R = None, None, None
    try:
        tid, _ = run_tool(mod, f"{d}/base.parts.txt", out, jobdir, meta["dtype"], lean=True)
        R = Run(out)
    except Exception as e:
        err = str(e)
    planted = set()
    if not err:
        flags = R.flags
        for ev in events:
            for side in ("a", "b"):
                key = (ev["locus"], ev[side]); planted.add(key)
                c = R.cell(*key)
                if c is None:
                    ev[f"det_{side}"] = "untested"; continue
                for k_, v_ in c.items(): ev[f"{k_}_{side}"] = v_
        basefl = set(map(tuple, base["flags"]))
        fp = sorted(flags - planted - basefl)
        lost = sorted(basefl - flags)
        # collateral vs clean FPs: was a swapped taxon among the FP cell's comparators or too-similar partners there?
        evloc = {}
        for ev in events: evloc.setdefault(ev["locus"], set()).update((ev["a"], ev["b"]))
        adj = {k: bool(R.near(*k) & evloc.get(k[0], set())) for k in fp}
        fp_adj = [list(k) for k in fp if adj[k]]
        fp_clean = [list(k) for k in fp if not adj[k]]
    else:
        fp, lost, fp_adj, fp_clean = [], [], [], []
    os.makedirs(os.path.dirname(res), exist_ok=True)
    json.dump({"dataset": ds, "rep": rep, "error": err, "tool": tid, "events": events,
               "fp_adjacent": fp_adj, "fp_clean": fp_clean, "lost": lost, "n_cells": R.n_cells if R else 0,
               "runtime": time.time() - t0}, open(res, "w"))
    shutil.rmtree(jobdir, ignore_errors=True)
    note = "" if err or tid == base["tool"] else f" (the tool changed since the baseline: {base['tool']} -> {tid})"
    return ds, rep, err or f"ok {time.time() - t0:.0f}s, {len(events)} events, {len(fp)} FP{note}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=("pilot", "full"), default="full",
                    help="pilot: Dornburg_2012, Oaks_2011, Wainwright_2012, one replicate each; full: every NEXUS dataset in test_data")
    ap.add_argument("--bench", default=f"{HERE}/bench", help="the benchmark's directory (default: validation/bench)")
    ap.add_argument("--datasets", help="comma-separated datasets in test_data, instead of the mode's list")
    ap.add_argument("--reps", type=int, help="replicates per dataset (default: pilot 1; full 8, 3 for the heavy datasets)")
    ap.add_argument("--workers", type=int, default=16, help="jobs at once, each a single-threaded tool run (default 16)")
    a = ap.parse_args(); bench = os.path.abspath(a.bench); pilot = a.mode == "pilot"
    os.makedirs(f"{bench}/results/{TAG}", exist_ok=True)
    datasets = ([s.strip() for s in a.datasets.split(",") if s.strip()] if a.datasets else
                ["Dornburg_2012", "Oaks_2011", "Wainwright_2012"] if pilot else
                sorted(d for d in os.listdir(DATA) if os.path.isdir(f"{DATA}/{d}") and os.path.exists(f"{DATA}/{d}/alignment.nex")))   # NEXUS datasets only (test_data also holds FASTA sets and scratch)
    missing = [d for d in datasets if not os.path.exists(f"{DATA}/{d}/alignment.nex")]
    if missing: sys.exit(f"no {DATA}/<dataset>/alignment.nex for: {', '.join(missing)}")
    print(f"LocusMasker {tool_id()}: {bench}, {TAG}", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        usable = []
        for ds, st in ex.map(prep, [bench] * len(datasets), datasets):
            print(f"prep {ds}: {st}", flush=True)
            if not st.startswith("unusable"): usable.append(ds)
        futs = {ex.submit(baseline, bench, ds): ds for ds in usable}
        okbase = []
        for fu in as_completed(futs):
            try: ds, st = fu.result(); print(f"baseline {ds}: {st}", flush=True); okbase.append(ds)
            except Exception as e: print(f"baseline {futs[fu]}: FAILED {e}", flush=True)
        nrep = lambda ds: a.reps if a.reps is not None else (1 if pilot else 3) if (ds in HEAVY or pilot) else 8
        jobs = [(ds, r) for ds in sorted(okbase) for r in range(nrep(ds))]
        futs = {ex.submit(swap_job, bench, ds, r): (ds, r) for ds, r in jobs}
        for fu in as_completed(futs):
            try: ds, r, st = fu.result(); print(f"swap {ds} rep{r}: {st}", flush=True)
            except Exception as e: print(f"swap {futs[fu]}: FAILED {e}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
