#!/usr/bin/env python3
"""Profile locusmasker.py over the test_data corpus: wall time, CPU, peak memory, the tool's own stage timings and
the thread counts it chose, per dataset and per thread cap; optionally a cProfile run (single thread, function level)
and a py-spy run (threaded, every thread sampled). Every run is its own process under /usr/bin/time -v, one after the
other unless --jobs says otherwise (only sensible for single-thread runs).

    python validation/profile_corpus.py --out DIR --threads 96                      # timing runs at a cap
    python validation/profile_corpus.py --out DIR --threads 1 --skip BIG... --jobs 4 # single-thread timing runs
    python validation/profile_corpus.py --out DIR --cprofile --skip BIG... --jobs 4  # cProfile at one thread
    python validation/profile_corpus.py --out DIR --pyspy PYSPY --threads 96 --datasets BIG...
    python validation/profile_corpus.py --out DIR --report                            # markdown tables from the results
    python validation/profile_corpus.py --out AFTER --compare BASELINE                # before/after table, and whether the masks agree

Results: DIR/<dataset>/t<T>/ (run.log, result.json, masked.csv), DIR/<dataset>/cprofile/ (profile.prof, cprofile.txt),
DIR/<dataset>/pyspy_t<T>/ (profile.raw, pyspy.txt), and DIR/results.jsonl, one line per run (later lines override
earlier ones for the same dataset, kind and thread count in --report). The run's other outputs are deleted unless
--keep-outputs."""
import argparse, collections, glob, json, os, pstats, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
TOOL = os.path.join(ROOT, "locusmasker.py"); PY = sys.executable; DATA = os.path.join(ROOT, "test_data")
STAGES = ("read", "counts", "pairs0", "rates", "pairs", "comparators", "dispersion", "test", "write")


def corpus(names=None):
    """every dataset folder with an alignment, smallest file first: name, alignment, partitions, datatype, bytes"""
    out = []
    for d in sorted(glob.glob(os.path.join(DATA, "*/"))):
        name = os.path.basename(os.path.dirname(d))
        if names and name not in names: continue
        nex, fa, parts = (os.path.join(d, f) for f in ("alignment.nex", "alignment.fasta", "partitions.txt"))
        if os.path.exists(nex):
            with open(nex, "rb") as fh: head = fh.read(8192).decode("latin-1")
            m = re.search(r"datatype\s*=\s*(\w+)", head, re.I); dt = m.group(1).lower() if m else "dna"
            out.append(dict(name=name, aln=nex, parts=None, dtype="aa" if dt == "protein" else "dna", bytes=os.path.getsize(nex)))
        elif os.path.exists(fa) and os.path.exists(parts):
            dt = "dna"; ry = os.path.join(d, "README.yaml")
            if os.path.exists(ry):
                with open(ry) as fh: m = re.search(r"datatype:\s*(\w+)", fh.read())
                if m and m.group(1).lower() == "protein": dt = "aa"
            out.append(dict(name=name, aln=fa, parts=parts, dtype=dt, bytes=os.path.getsize(fa)))
    missing = set(names or []) - {x["name"] for x in out}
    if missing: sys.exit(f"no alignment found for {sorted(missing)}")
    return sorted(out, key=lambda x: x["bytes"])


EXTRA = []                                                            # more options for the tool (--extra)


def tool_args(ds, threads, out_dir):
    return [TOOL, "--aln", ds["aln"], "--datatype", ds["dtype"], "--out", out_dir, "--threads", str(threads), "--progress", "none"] + \
           (["--partitions", ds["parts"]] if ds["parts"] else []) + list(EXTRA)


TIME_RE = {"wall": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", "user": r"User time \(seconds\): ([\d.]+)",
           "sys": r"System time \(seconds\): ([\d.]+)", "rss_mb": r"Maximum resident set size \(kbytes\): (\d+)"}


def hms(s):
    v = 0.0
    for x in s.split(":"): v = v * 60 + float(x)
    return v


def parse_log(log):
    """the tool's thread choices and its last line (the message when it stopped)"""
    ch = {}
    for line in log.splitlines():
        m = re.match(r"pass one: (\d+) of (\d+) threads", line)
        if m: ch["pass1"] = int(m.group(1))
        m = re.match(r"pair stage: (\d+) chunks? of ([\d,]+) pairs, about ([\d,]+) MB each, (\d+) thread", line)
        if m: ch.setdefault("pairs", []).append(dict(chunks=int(m.group(1)), chunk=int(m.group(2).replace(",", "")), mb=int(m.group(3).replace(",", "")), threads=int(m.group(4))))
        m = re.match(r"comparators: (\d+) thread", line)
        if m: ch["comparators"] = int(m.group(1))
        m = re.match(r"test stage: (\d+) thread", line)
        if m: ch["test"] = int(m.group(1))
        m = re.match(r"threads: (\d+) asked, (\d+) CPUs", line)
        if m: ch["cpus"] = int(m.group(2))
    lines = [l for l in log.splitlines() if l.strip()]
    return ch, (lines[-1] if lines else "")


def read_manifest(run_dir):
    import zipfile
    p = os.path.join(run_dir, "report.npz")
    if not os.path.exists(p): return None
    with zipfile.ZipFile(p) as Z:
        man = json.loads(Z.read("manifest.json")); lj = json.loads(Z.read("loci.json"))
    c = man["counts"]
    return dict(taxa=c["taxa"], loci=c["loci"], sites=int(sum(lj["length"])), cells=c["cells"], masked=c["masked"], tests=man["test"]["tests"],
                threads=man["options"]["threads"], timings=man.get("timings", {}), version=man["version"])


def tidy(run_dir, keep):
    if keep or not os.path.isdir(run_dir): return
    for f in os.listdir(run_dir):
        if f not in ("masked.csv",): os.remove(os.path.join(run_dir, f))


def run_timing(ds, threads, out, timeout, keep):
    d = os.path.join(out, ds["name"], f"t{threads}"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d); run_dir = os.path.join(d, "run")
    cmd = ["/usr/bin/time", "-v", PY] + tool_args(ds, threads, run_dir); t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout); rc = r.returncode; err = r.stderr
    except subprocess.TimeoutExpired as e:
        rc = "timeout"; err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    log = re.split(r"^Command (?:exited with non-zero status|terminated by signal)|^\tCommand being timed", err, flags=re.M)[0]
    with open(os.path.join(d, "run.log"), "w") as fh: fh.write(err)
    res = dict(kind="timing", dataset=ds["name"], threads=threads, rc=rc, harness_wall=round(time.time() - t0, 2), dtype=ds["dtype"], bytes=ds["bytes"])
    for k, rx in TIME_RE.items():
        m = re.search(rx, err)
        if m: res[k] = hms(m.group(1)) if k == "wall" else (int(m.group(1)) // 1024 if k == "rss_mb" else float(m.group(1)))
    res["choices"], res["last_line"] = parse_log(log)
    man = read_manifest(run_dir)
    if man: res.update(man)
    tidy(run_dir, keep)
    with open(os.path.join(d, "result.json"), "w") as fh: json.dump(res, fh, indent=1)
    return res


def run_cprofile(ds, out, timeout, keep):
    d = os.path.join(out, ds["name"], "cprofile"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d); run_dir = os.path.join(d, "run")
    prof = os.path.join(d, "profile.prof"); txt = os.path.join(d, "cprofile.txt"); argv = tool_args(ds, 1, run_dir)
    code = f"""import cProfile, pstats, sys, io
sys.argv = {argv!r}; sys.path.insert(0, {ROOT!r})
import locusmasker as lo
pr = cProfile.Profile(); pr.enable()
try: lo.main()
except SystemExit: pass
pr.disable(); pr.dump_stats({prof!r})
with open({txt!r}, "w") as fh:
    for key in ("tottime", "cumulative"):
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats(key).print_stats(50); fh.write(f"==== sorted by {{key}}\\n" + s.getvalue() + "\\n")
"""
    t0 = time.time()
    try: r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=timeout); rc = r.returncode; err = r.stderr
    except subprocess.TimeoutExpired as e: rc = "timeout"; err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    with open(os.path.join(d, "run.log"), "w") as fh: fh.write(err)
    res = dict(kind="cprofile", dataset=ds["name"], threads=1, rc=rc, harness_wall=round(time.time() - t0, 2))
    if os.path.exists(prof):
        st = pstats.Stats(prof); total = st.total_tt; rows = []
        for (f, ln, fn), (cc, nc, tt, ct, callers) in st.stats.items():
            rows.append(dict(func=fn, file=os.path.basename(f), line=ln, ncalls=nc, tottime=round(tt, 3), cumtime=round(ct, 3)))
        rows.sort(key=lambda x: -x["tottime"]); res["total_tt"] = round(total, 2); res["top"] = rows[:25]
    tidy(run_dir, keep)
    with open(os.path.join(d, "result.json"), "w") as fh: json.dump(res, fh, indent=1)
    return res


def run_pyspy(ds, threads, out, pyspy, timeout, keep, rate=20):
    d = os.path.join(out, ds["name"], f"pyspy_t{threads}"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d); run_dir = os.path.join(d, "run")
    raw = os.path.join(d, "profile.raw"); cmd = [pyspy, "record", "--format", "raw", "--rate", str(rate), "--output", raw, "--", PY] + tool_args(ds, threads, run_dir)
    t0 = time.time()
    try: r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout); rc = r.returncode; err = r.stderr + r.stdout
    except subprocess.TimeoutExpired as e: rc = "timeout"; err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    with open(os.path.join(d, "run.log"), "w") as fh: fh.write(err)
    res = dict(kind="pyspy", dataset=ds["name"], threads=threads, rc=rc, harness_wall=round(time.time() - t0, 2), rate=rate)
    if os.path.exists(raw):
        self_t, incl = collections.Counter(), collections.Counter(); total = 0
        with open(raw) as fh:
            for line in fh:
                stack, _, cnt = line.rstrip("\n").rpartition(" ")
                if not cnt.isdigit(): continue
                cnt = int(cnt); frames = stack.split(";"); total += cnt; self_t[frames[-1]] += cnt
                for f in set(frames): incl[f] += cnt
        res["samples"] = total
        res["self"] = [dict(frame=f, pct=round(100.0 * c / max(total, 1), 1)) for f, c in self_t.most_common(30)]
        res["inclusive"] = [dict(frame=f, pct=round(100.0 * c / max(total, 1), 1)) for f, c in incl.most_common(40)]
        with open(os.path.join(d, "pyspy.txt"), "w") as fh:
            fh.write(f"{total} samples at {rate} Hz over every active thread (a sample per thread per tick, so percentages sum to more than 100 when threads run in parallel)\n\n== self time\n")
            for x in res["self"]: fh.write(f"{x['pct']:6.1f}%  {x['frame']}\n")
            fh.write("\n== inclusive\n")
            for x in res["inclusive"]: fh.write(f"{x['pct']:6.1f}%  {x['frame']}\n")
    tidy(run_dir, keep)
    with open(os.path.join(d, "result.json"), "w") as fh: json.dump(res, fh, indent=1)
    return res


def fmt_t(v): return "" if v is None else (f"{v:.1f} s" if v < 100 else f"{v / 60:.1f} min")


def report(out):
    rows = load_results(out)
    names = sorted({k[0] for k in rows}, key=lambda n: rows.get((n, "timing", 1), rows.get((n, "timing", 96), {})).get("bytes", 0))
    caps = sorted({k[2] for k in rows if k[1] == "timing"})
    print("| dataset | taxa | loci | sites | " + " | ".join(f"{t} thr: wall / CPU% / RSS" for t in caps) + " | speedup | threads chosen (pass1 / pairs / comparators / test) |")
    print("|---|---|---|---|" + "---|" * len(caps) + "---|---|")
    tot = collections.defaultdict(float)
    for n in names:
        cells = []; walls = {}
        base = next((rows[(n, "timing", t)] for t in caps if (n, "timing", t) in rows), None)
        if base is None: continue
        for t in caps:
            r = rows.get((n, "timing", t))
            if r is None: cells.append("not run"); continue
            if r["rc"] != 0: cells.append(f"stopped: {r['last_line'][:60]}" if r["rc"] != "timeout" else "TIMEOUT"); continue
            w = r.get("wall"); cpu = 100.0 * (r.get("user", 0) + r.get("sys", 0)) / max(w, 1e-9); walls[t] = w; tot[t] += w
            cells.append(f"{fmt_t(w)} / {cpu:.0f}% / {r.get('rss_mb', 0) / 1024:.1f} GB")
        sp = f"{walls[min(walls)] / walls[max(walls)]:.1f}x" if len(walls) == 2 else ""
        hi = rows.get((n, "timing", max(caps)))
        ch = hi.get("choices", {}) if hi else {}
        chosen = f"{ch.get('pass1', '-')} / {','.join(str(p['threads']) for p in ch.get('pairs', [])) or '-'} / {ch.get('comparators', '-')} / {ch.get('test', '-')}" if hi and hi["rc"] == 0 else ""
        print(f"| {n} | {base.get('taxa', '')} | {base.get('loci', '')} | {base.get('sites', '')} | " + " | ".join(cells) + f" | {sp} | {chosen} |")
    print("| **total wall** | | | | " + " | ".join(fmt_t(tot[t]) for t in caps) + " | | |")
    for t in caps:
        print(f"\n### Stage times at {t} thread{'s' if t > 1 else ''} (seconds)\n")
        print("| dataset | " + " | ".join(STAGES) + " | total |"); print("|---|" + "---|" * (len(STAGES) + 1))
        for n in names:
            r = rows.get((n, "timing", t))
            if not r or r["rc"] != 0 or not r.get("timings"): continue
            tm = r["timings"]; print(f"| {n} | " + " | ".join(f"{tm.get(s, 0):.1f}" for s in STAGES) + f" | {tm.get('total', 0):.1f} |")
    cp = [rows[k] for k in rows if k[1] == "cprofile" and rows[k].get("top")]
    if cp:
        print("\n### cProfile, single thread: the five functions with the most own time per dataset\n")
        print("| dataset | total | top functions (own time, s) |"); print("|---|---|---|")
        for r in sorted(cp, key=lambda r: -r.get("total_tt", 0)):
            print(f"| {r['dataset']} | {r['total_tt']:.1f} | " + "; ".join(f"{x['func']} ({x['file']}:{x['line']}) {x['tottime']:.1f}" for x in r["top"][:5]) + " |")
    ps = [rows[k] for k in rows if k[1] == "pyspy" and rows[k].get("self")]
    if ps:
        print("\n### py-spy, threaded runs: the frames with the most self time (percent of samples, every active thread)\n")
        for r in sorted(ps, key=lambda r: -r.get("samples", 0)):
            print(f"**{r['dataset']}, {r['threads']} threads, {r['samples']} samples**: " + "; ".join(f"{x['frame']} {x['pct']}%" for x in r["self"][:8]))


def load_results(out):
    rows = {}
    with open(os.path.join(out, "results.jsonl")) as fh:
        for line in fh:
            r = json.loads(line); rows[(r["dataset"], r["kind"], r["threads"])] = r
    return rows


def compare(after, before):
    """wall time, CPU and peak memory before and after per dataset and thread cap, and whether masked.csv is the same"""
    A, B = load_results(after), load_results(before)
    caps = sorted({k[2] for k in A if k[1] == "timing"} & {k[2] for k in B if k[1] == "timing"})
    names = sorted({k[0] for k in A if k[1] == "timing"}, key=lambda n: B.get((n, "timing", caps[0]), {}).get("bytes", 0))
    print("| dataset | taxa x loci | " + " | ".join(f"{t} thr before -> after (speedup) / CPU% / peak GB" for t in caps) + " | masks |")
    print("|---|---|" + "---|" * len(caps) + "---|")
    tot = collections.defaultdict(lambda: [0.0, 0.0]); same_all = True
    for n in names:
        cells = []; masks = []
        for t in caps:
            a, b = A.get((n, "timing", t)), B.get((n, "timing", t))
            if not a or not b or a["rc"] != 0 or b["rc"] != 0:
                cells.append("" if (a and b and a["rc"] != 0 and b["rc"] != 0) else "not run" if not a or not b else f"rc {a['rc']} / {b['rc']}"); continue
            tot[t][0] += b["wall"]; tot[t][1] += a["wall"]
            cpu = lambda r: 100.0 * (r.get("user", 0) + r.get("sys", 0)) / max(r["wall"], 1e-9)
            cells.append(f"{fmt_t(b['wall'])} -> {fmt_t(a['wall'])} ({b['wall'] / max(a['wall'], 1e-9):.1f}x) / {cpu(b):.0f}% -> {cpu(a):.0f}% / {b.get('rss_mb', 0) / 1024:.1f} -> {a.get('rss_mb', 0) / 1024:.1f}")
            fa, fb = (os.path.join(d, n, f"t{t}", "run", "masked.csv") for d in (after, before))
            if os.path.exists(fa) and os.path.exists(fb):
                same = open(fa).read() == open(fb).read(); masks.append(same); same_all &= same
        base = next((B[(n, "timing", t)] for t in caps if (n, "timing", t) in B and B[(n, "timing", t)]["rc"] == 0), None)
        if base is None: continue
        print(f"| {n} | {base.get('taxa', '')} x {base.get('loci', '')} | " + " | ".join(cells) + " | " + ("identical" if masks and all(masks) else "DIFFER" if masks else "") + " |")
    print("| **total** | | " + " | ".join(f"{fmt_t(tot[t][0])} -> {fmt_t(tot[t][1])} ({tot[t][0] / max(tot[t][1], 1e-9):.1f}x)" for t in caps) + f" | {'all identical' if same_all else 'see above'} |")
    print("\n### Stage times after (seconds, the larger datasets)\n")
    for t in caps:
        rows = [(n, A[(n, "timing", t)]) for n in names if (n, "timing", t) in A and A[(n, "timing", t)]["rc"] == 0 and A[(n, "timing", t)].get("wall", 0) >= 5]
        if not rows: continue
        print(f"**{t} thread{'s' if t > 1 else ''}**\n\n| dataset | " + " | ".join(STAGES) + " | total |"); print("|---|" + "---|" * (len(STAGES) + 1))
        for n, r in rows:
            tm = r.get("timings", {}); print(f"| {n} | " + " | ".join(f"{tm.get(s, 0):.1f}" for s in STAGES) + f" | {tm.get('total', 0):.1f} |")
        print()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, nargs="*", default=[])
    ap.add_argument("--datasets", nargs="*"); ap.add_argument("--skip", nargs="*", default=[]); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=4 * 3600); ap.add_argument("--cprofile", action="store_true"); ap.add_argument("--pyspy")
    ap.add_argument("--keep-outputs", action="store_true"); ap.add_argument("--report", action="store_true"); ap.add_argument("--compare")
    ap.add_argument("--extra", default="", help="more options for the tool, one string: --extra \"--block-taxa 50 --candidates 20\""); a = ap.parse_args()
    global EXTRA; EXTRA = a.extra.split()
    if a.report: report(a.out); return
    if a.compare: compare(a.out, a.compare); return
    os.makedirs(a.out, exist_ok=True); sets = [d for d in corpus(a.datasets) if d["name"] not in a.skip]
    jobs = []
    for ds in sets:
        for t in a.threads:
            if a.pyspy: jobs.append((f"{ds['name']} py-spy t{t}", lambda ds=ds, t=t: run_pyspy(ds, t, a.out, a.pyspy, a.timeout, a.keep_outputs)))
            else: jobs.append((f"{ds['name']} t{t}", lambda ds=ds, t=t: run_timing(ds, t, a.out, a.timeout, a.keep_outputs)))
        if a.cprofile: jobs.append((f"{ds['name']} cprofile", lambda ds=ds: run_cprofile(ds, a.out, a.timeout, a.keep_outputs)))
    def do(job):
        label, fn = job; t0 = time.time(); res = fn()
        with open(os.path.join(a.out, "results.jsonl"), "a") as fh: fh.write(json.dumps(res) + "\n")
        print(f"{label}: rc={res['rc']} {fmt_t(res.get('wall', time.time() - t0))}" + (f", {res.get('rss_mb', 0)} MB" if 'rss_mb' in res else ""), flush=True)
    if a.jobs > 1:
        with ThreadPoolExecutor(a.jobs) as ex: list(ex.map(do, jobs))
    else:
        for job in jobs: do(job)


if __name__ == "__main__": main()
