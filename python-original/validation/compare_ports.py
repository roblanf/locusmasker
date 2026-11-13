#!/usr/bin/env python3
"""Compare the outputs of locusmasker.py and of the Rust tool (the crate at the repository root, binary `locusmasker`) byte for byte.

    python validation/compare_ports.py                    every fixture dataset, the standard run
    python validation/compare_ports.py dna_small aa_small  these datasets
    python validation/compare_ports.py --dirs PY_OUT RUST_OUT   two output folders already written

What is compared: the masked alignment and masked.csv as files; tests.tsv.gz decompressed; every member of report.npz
but manifest.json; every member of the report's core (inline in report.html) and of every report_files chunk, decoded;
and the manifests without the fields that differ between any two runs (date, timings, command, versions, system);
the page around the bundle, less the bundle's compressed size. The tool's version (the Rust program has its own, the script
0.14.0) is left out wherever it is written: the manifest, report_files/index.js and the page's loading line.
The compressed containers themselves are not compared: their bytes hold timestamps and depend on the zlib build."""
import argparse, base64, glob, gzip, hashlib, io, json, os, re, shutil, subprocess, sys, tempfile, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
VOLATILE = ("date", "timings", "command", "python", "numpy", "system", "commit", "version")   # version: the Rust program has its own, the script 0.14.0
VERSION_RE = re.compile(r'"version": ?"[^"]*"')
CASES = {  # name -> (alignment, partitions, datatype, extra options)
    **{k: (f"synth/{k}/alignment.fasta", f"synth/{k}/partitions.txt", None, []) for k in ("dna_small", "aa_small", "dna_clean", "aa_clean", "dna_dense", "dna_rc", "dna_blocks")},
    "aa_random_trees": ("sim/aa_random_trees/alignment.fasta", "sim/aa_random_trees/partitions.txt", None, []),
    "cannon_aa": ("real/cannon_aa/alignment.nex", None, "aa", []), "near_dna": ("real/near_dna/alignment.nex", None, "dna", []),
    "dna_small_t1": ("synth/dna_small/alignment.fasta", "synth/dna_small/partitions.txt", None, ["--donor-threshold", "1"]),
    "dna_small_nexus": ("synth/dna_small/alignment.nex", None, "dna", []), "dna_small_quirky": ("synth/dna_small/alignment_quirky.nex", None, "dna", []),
    "dna_small_phylip": ("synth/dna_small/alignment.phy", "synth/dna_small/partitions.txt", "dna", []),
    "dna_small_interleaved": ("synth/dna_small/alignment_interleaved.phy", "synth/dna_small/partitions.txt", "dna", []),
    "dna_small_alltests": ("synth/dna_small/alignment.fasta", "synth/dna_small/partitions.txt", None, ["--all-tests"]),
    "aa_small_alltests": ("synth/aa_small/alignment.fasta", "synth/aa_small/partitions.txt", None, ["--all-tests"]),
    "dna_small_similar_only": ("synth/dna_small/alignment.fasta", "synth/dna_small/partitions.txt", None, ["--mask-similar-only"]),
    "dna_small_different_only": ("synth/dna_small/alignment.fasta", "synth/dna_small/partitions.txt", None, ["--mask-different-only"]),
    "dna_small_no_bs": ("synth/dna_small/alignment.fasta", "synth/dna_small/partitions.txt", None, ["--no-broad-sample"]),
    "dna_small_k3": ("synth/dna_small/alignment.fasta", "synth/dna_small/partitions.txt", None, ["--n-comp", "3"]),
    "dna_small_k7": ("synth/dna_small/alignment.fasta", "synth/dna_small/partitions.txt", None, ["--n-comp", "7", "--fit-overlap", "40", "--alpha", "0.01"]),
    "dna_rc_masked": ("synth/dna_rc/alignment.fasta", "synth/dna_rc/partitions.txt", None, ["--report-codes", "masked", "--report-sequences", "masked"]),
    "dna_dense_fails": ("synth/dna_dense/alignment.fasta", "synth/dna_dense/partitions.txt", None, ["--report-sequences", "fails", "--candidates", "7"]),
}
PATCHED = {  # name -> (base case, constants set in both tools: lo.NAME in Python, LOCUSMASKER_NAME for the Rust port)
    "dna_small_sparse": ("dna_small", dict(DENSE_CELLS_MAX=10)), "dna_blocks_gtdb": ("dna_blocks", dict(DENSE_CELLS_MAX=10), ["--report-codes", "masked", "--report-sequences", "masked"]),
    "aa_small_nofull": ("aa_small", dict(FULL_M_MAX_TAXA=10)), "dna_rc_nofull_sparse": ("dna_rc", dict(FULL_M_MAX_TAXA=10, DENSE_CELLS_MAX=10)),
    "dna_dense_c3_nofull": ("dna_dense", dict(FULL_M_MAX_TAXA=10), ["--candidates", "3"]), "dna_dense_c3": ("dna_dense", {}, ["--candidates", "3"]),
    "aa_random_trees_c4": ("aa_random_trees", {}, ["--candidates", "4"]), "aa_random_trees_nofull": ("aa_random_trees", dict(FULL_M_MAX_TAXA=10)),
    "dna_small_bm5": ("dna_small", dict(BEST_MATCH_ALL_MAX_TAXA=5)), "dna_small_noslab": ("dna_small", dict(SWEEP_BLOCK_MIN=8)), "cannon_aa_noslab_nofull": ("cannon_aa", dict(SWEEP_BLOCK_MIN=8, FULL_M_MAX_TAXA=10)),
    "dna_blocks_sample500": ("dna_blocks", dict(REPORT_SAMPLE_CELLS=500)), "dna_blocks_sample100": ("dna_blocks", dict(REPORT_SAMPLE_CELLS=100)), "dna_blocks_sample20000": ("dna_blocks", dict(REPORT_SAMPLE_CELLS=20000)),
    "aa_small_sample50": ("aa_small", dict(REPORT_SAMPLE_CELLS=50)), "aa_random_trees_sample3000": ("aa_random_trees", dict(REPORT_SAMPLE_CELLS=3000)), "dna_rc_sparse_sample7": ("dna_rc", dict(REPORT_SAMPLE_CELLS=7, DENSE_CELLS_MAX=10)),
    "near_dna_sparse_alltests": ("near_dna", dict(DENSE_CELLS_MAX=10), ["--all-tests"]),
}
for _k, _v in PATCHED.items(): CASES[_k] = CASES[_v[0]][:3] + (CASES[_v[0]][3] + (list(_v[2]) if len(_v) > 2 else []),)


def sha(b): return hashlib.sha256(b).hexdigest()


def zip_members(data):
    z = zipfile.ZipFile(io.BytesIO(data)); return {k: z.read(k) for k in z.namelist()}


def stable_manifest(b):
    m = json.loads(b)
    for k in VOLATILE: m.pop(k, None)
    return json.dumps(m, sort_keys=True).encode()


def collect(out):
    """name -> bytes of everything compared in an output folder"""
    got = {}
    for f in sorted(os.listdir(out)):
        p = os.path.join(out, f)
        if f == "masked.csv" or ".masked." in f: got[f] = open(p, "rb").read()
        elif f == "tests.tsv.gz": got["tests.tsv"] = gzip.open(p, "rb").read()
        elif f == "report.npz":
            for k, v in zip_members(open(p, "rb").read()).items(): got[f"npz:{k}"] = stable_manifest(v) if k == "manifest.json" else v
        elif f == "report.html":
            html = open(p, encoding="utf-8").read(); m = re.search(r'<script id="bundle"[^>]*>([^<]*)</script>', html)
            if m is None: m = max(re.finditer(r'>([A-Za-z0-9+/=]{2000,})<', html), key=lambda x: len(x.group(1)), default=None)
            if m is None: got["html:core"] = b"(no bundle found)"
            else:
                for k, v in zip_members(base64.b64decode(m.group(1))).items(): got[f"core:{k}"] = stable_manifest(v) if k == "manifest.json" else v
                page = html[:m.start(1)] + html[m.end(1):]                              # the page around the bundle, less the bundle's
                page = re.sub(r'"bundle_bytes": \d+', '"bundle_bytes": 0', page, count=1)
                got["html:page"] = VERSION_RE.sub('"version": ""', page, count=1).encode()        # and the version in the loading screen's line   # compressed size (it depends on the zlib build)
        elif f == "report_files":
            for c in sorted(os.listdir(p)):
                txt = open(os.path.join(p, c), encoding="utf-8").read(); m = re.match(r'LM_CHUNK\("(\w+)",(\d+),"([^"]*)"\);\n$', txt)
                if m is None: got[f"chunk:{c}"] = VERSION_RE.sub('"version":""', txt).encode()   # index.js: the layout, the version, the dataset
                else:
                    for k, v in zip_members(base64.b64decode(m.group(3))).items(): got[f"chunk:{c}:{k}"] = v
    return got


def describe(k, a, b):
    """a short account of how two members differ"""
    if a is None or b is None: return "only in " + ("rust" if a is None else "python")
    if k.endswith(".npy"):
        import numpy as np
        try:
            x, y = np.load(io.BytesIO(a)), np.load(io.BytesIO(b))
            if x.shape != y.shape or x.dtype != y.dtype: return f"shape/dtype {x.shape} {x.dtype} vs {y.shape} {y.dtype}"
            d = ~((x == y) | ((x != x) & (y != y))) if x.dtype.kind == "f" else (x != y)
            i = np.argwhere(d); j = tuple(i[0])
            return f"{int(d.sum())} of {x.size} elements differ; first at {j}: python {x[j]!r} rust {y[j]!r}"
        except Exception as e: return f"npy unreadable ({e})"
    ta, tb = a.decode("utf-8", "replace"), b.decode("utf-8", "replace"); n = next((i for i, (p, q) in enumerate(zip(ta, tb)) if p != q), min(len(ta), len(tb)))
    return f"lengths {len(ta)} / {len(tb)}; first difference at char {n}: python {ta[max(0, n - 40):n + 60]!r} rust {tb[max(0, n - 40):n + 60]!r}"


def compare(py_out, rs_out, verbose=True):
    A, B = collect(py_out), collect(rs_out); bad = [k for k in sorted(set(A) | set(B)) if A.get(k) != B.get(k)]
    if verbose:
        for k in bad[:12]: print(f"   {k}: {describe(k, A.get(k), B.get(k))}")
        if len(bad) > 12: print(f"   ... and {len(bad) - 12} more")
    return len(A), bad


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cases", nargs="*"); ap.add_argument("--dirs", nargs=2); ap.add_argument("--keep"); ap.add_argument("--threads", default="4")
    ap.add_argument("--rust", default=os.path.join(os.path.dirname(ROOT), "target", "release", "locusmasker")); ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--env", action="append", default=[], help="NAME=VALUE for the Rust run (the test hooks, e.g. LOCUSMASKER_DENSE_CELLS_MAX=10)")
    a = ap.parse_args()
    if a.dirs: n, bad = compare(*a.dirs); print("identical" if not bad else f"{len(bad)} of {n} differ"); sys.exit(1 if bad else 0)
    work = a.keep or tempfile.mkdtemp(prefix="lm_ports_"); os.makedirs(work, exist_ok=True); failed = 0
    for name in a.cases or list(CASES):
        aln, parts, dt, extra = CASES[name]
        if dt is None: dt = json.load(open(os.path.join(DATA, os.path.dirname(aln), "truth.json")))["datatype"]
        args = ["--aln", os.path.join(DATA, aln), "--datatype", dt, "--progress", "none", "--threads", a.threads] + (["--partitions", os.path.join(DATA, parts)] if parts else []) + extra
        outs = []
        consts = PATCHED[name][1] if name in PATCHED else {}
        patch = "import sys; sys.path.insert(0, %r); import locusmasker as lo; sys.argv[0] = lo.__file__; %s lo.main()" % (ROOT, "".join(f"lo.{k} = {v}; " for k, v in consts.items()))
        for tool, cmd in (("py", [a.python, "-c", patch] if consts else [a.python, os.path.join(ROOT, "locusmasker.py")]), ("rs", [a.rust])):
            out = os.path.join(work, name, tool); shutil.rmtree(out, ignore_errors=True)
            env = dict(os.environ, **dict(e.split("=", 1) for e in a.env), **{f"LOCUSMASKER_{k}": str(v) for k, v in consts.items()}) if tool == "rs" else None
            r = subprocess.run(cmd + args + ["--out", out], capture_output=True, text=True, env=env); outs.append((out, r))
        (po, pr), (ro, rr) = outs
        if pr.returncode != rr.returncode:
            failed += 1; print(f"{name}: exit status python {pr.returncode} rust {rr.returncode}\n   python: {pr.stderr.strip().splitlines()[-1:]}\n   rust: {rr.stderr.strip().splitlines()[-3:]}"); continue
        n, bad = compare(po, ro, verbose=False)
        print(f"{name}: " + (f"identical ({n} members)" if not bad else f"{len(bad)} of {n} members differ"))
        if bad: failed += 1; compare(po, ro)
    if not a.keep: shutil.rmtree(work, ignore_errors=True)
    print("all identical" if not failed else f"{failed} cases differ"); sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
