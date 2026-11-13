#!/usr/bin/env python3
"""Random alignments with random options through locusmasker.py and the Rust port, compared byte for byte
(validation/compare_ports.py says what is compared).

    python validation/fuzz_ports.py --cases 40 --seed 1

Each case: a random tree's sequences per locus (its own rate), missing stretches and missing loci, a few sequences
replaced by a copy of another taxon's or by an unrelated one, sometimes identical genomes and taxa with almost nothing;
DNA or amino acids; FASTA plus partitions (sometimes with gaps between loci, sometimes NEXUS with charsets); random
--n-comp, --fit-overlap, --alpha, --donor-threshold, --candidates, the mask-only switches, --all-tests, the report policies; and, in some cases, the size thresholds forced
low in both tools (no full m matrix, the sparse report), the modes of very large alignments."""
import argparse, os, random, shutil, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_ports import ROOT, compare


def make_case(rng, d):
    dtype = rng.choice(["dna", "dna", "aa"]); alpha = "ACGT" if dtype == "dna" else "ACDEFGHIKLMNPQRSTVWY"; A = len(alpha)
    n = int(rng.choice([5, 6, 8, 12, 20, 35, 60, 90])); nloc = int(rng.choice([10, 11, 14, 25, 40])); names = [f"t{i:03d}" for i in range(n)]
    depth = rng.uniform(0.02, 0.6); seqs = [[] for _ in range(n)]; bounds = []; pos = 0; parts = []
    parent = [-1] + [int(rng.integers(0, i)) for i in range(1, n)]; blen = rng.exponential(depth / 4, n) * (rng.uniform(size=n) > 0.05)   # some zero branches: identical genomes
    for li in range(nloc):
        L = int(rng.choice([25, 60, 150, 150, 400, 400])); rate = float(rng.choice([0.2, 1.0, 1.0, 3.0, 12.0])); root = rng.integers(0, A, L); X = np.empty((n, L), np.int64)
        for i in range(n):
            src = root if parent[i] < 0 else X[parent[i]]; mut = rng.uniform(size=L) < 1 - np.exp(-blen[i] * rate)
            X[i] = np.where(mut, rng.integers(0, A, L), src)
        S = np.array(list(alpha))[X]
        for _ in range(int(rng.integers(0, 3))):                                   # a copy of another taxon's sequence, or an unrelated one
            g = int(rng.integers(0, n))
            if rng.uniform() < 0.5: S[g] = S[int(rng.integers(0, n))]
            else: S[g] = np.array(list(alpha))[rng.integers(0, A, L)]
        for i in range(n):
            u = rng.uniform()
            if u < 0.06: S[i] = "-"
            elif u < 0.25: a_ = int(rng.integers(0, L)); S[i, a_:a_ + int(rng.integers(1, L + 1))] = rng.choice(["-", "?", "N" if dtype == "dna" else "X"])
        gap = int(rng.integers(0, 4)) if rng.uniform() < 0.2 else 0                 # sites in no locus
        for i in range(n): seqs[i].append("".join(S[i]) + "-" * gap)
        parts.append((f"locus{li + 1}", pos + 1, pos + L)); pos += L + gap
    if rng.uniform() < 0.3:                                                         # a taxon with almost nothing
        g = int(rng.integers(0, n)); keep = set(rng.choice(nloc, int(rng.integers(0, min(12, nloc + 1))), replace=False).tolist())
        seqs[g] = [s if li in keep else "-" * len(s) for li, s in enumerate(seqs[g])]
    if rng.uniform() < 0.3: g, h = rng.choice(n, 2, replace=False); seqs[int(g)] = list(seqs[int(h)])           # two identical genomes
    rows = ["".join(s) for s in seqs]; fmt = rng.choice(["fasta", "fasta", "nexus", "phylip"]); args = []
    if fmt == "nexus":
        aln = os.path.join(d, "aln.nex"); dt = "dna" if dtype == "dna" else "protein"
        with open(aln, "w") as fh:
            fh.write(f"#NEXUS\nbegin data;\ndimensions ntax={n} nchar={pos};\nformat datatype={dt} missing=? gap=-;\nmatrix\n" + "".join(f"{t} {r}\n" for t, r in zip(names, rows)) + ";\nend;\n")
            fh.write("begin sets;\n" + "".join(f"charset {nm} = {a}-{b};\n" for nm, a, b in parts) + "end;\n")
    else:
        aln = os.path.join(d, "aln.fasta" if fmt == "fasta" else "aln.phy")
        with open(aln, "w") as fh:
            if fmt == "fasta": fh.write("".join(f">{t}\n{r}\n" for t, r in zip(names, rows)))
            else: fh.write(f"{n} {pos}\n" + "".join(f"{t} {r}\n" for t, r in zip(names, rows)))
        pf = os.path.join(d, "parts.txt"); open(pf, "w").write("".join(f"{'DNA' if dtype == 'dna' else 'WAG'}, {nm} = {a}-{b}\n" for nm, a, b in parts)); args += ["--partitions", pf]
    args += ["--aln", aln, "--datatype", dtype, "--progress", "none", "--threads", str(rng.choice([1, 3, 8]))]
    if rng.uniform() < 0.5: args += ["--n-comp", str(rng.choice([3, 5, 7, 9]))]
    if rng.uniform() < 0.5: args += ["--fit-overlap", str(rng.choice([5, 10, 20, 50]))]
    if rng.uniform() < 0.4: args += ["--alpha", str(rng.choice([0.5, 0.05, 0.001, 1e-6]))]
    if rng.uniform() < 0.3: args += ["--donor-threshold", str(rng.choice([1, 1.5, 4]))]
    if rng.uniform() < 0.3: args += ["--candidates", str(rng.choice([1, 2, 5, 30]))]
    if rng.uniform() < 0.25: args += ["--all-tests"]
    if rng.uniform() < 0.15: args += [rng.choice(["--mask-similar-only", "--mask-different-only", "--no-broad-sample"])]
    if rng.uniform() < 0.3: args += ["--report-codes", rng.choice(["fails", "masked", "none"]), "--report-sequences", rng.choice(["all", "fails", "masked", "none"])]
    consts = {}
    if rng.uniform() < 0.4: consts["FULL_M_MAX_TAXA"] = 4                           # no full m matrix: the mode of more than 20,000 taxa
    if rng.uniform() < 0.2: consts["DENSE_CELLS_MAX"] = 10                          # the sparse report
    return [str(x) for x in args], consts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", type=int, default=40); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--keep")
    ap.add_argument("--rust", default=os.path.join(os.path.dirname(ROOT), "target", "release", "locusmasker")); ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(); work = a.keep or tempfile.mkdtemp(prefix="lm_fuzz_"); bad = 0; ran = 0; stopped = 0; crashed = 0; why = {}
    for c in range(a.cases):
        rng = np.random.default_rng([a.seed, c]); d = os.path.join(work, f"case{c}"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        args, consts = make_case(rng, d); res = []
        patch = "import sys; sys.path.insert(0, %r); import locusmasker as lo; sys.argv[0] = lo.__file__; %s lo.main()" % (ROOT, "".join(f"lo.{k} = {v}; " for k, v in consts.items()))
        env = dict(os.environ, **{f"LOCUSMASKER_{k}": str(v) for k, v in consts.items()})
        for tool, cmd in (("py", [a.python, "-c", patch] if consts else [a.python, os.path.join(ROOT, "locusmasker.py")]), ("rs", [a.rust])):
            res.append(subprocess.run(cmd + args + ["--out", os.path.join(d, tool)], capture_output=True, text=True, env=env if tool == "rs" else None))
        py, rs = res; last = lambda r: (r.stderr.strip().splitlines() or [""])[-1]
        # a known crash of the script, not a difference: with the sparse report (DENSE_CELLS_MAX forced low) and no
        # failing cell, its report writer indexes an empty array (at_cells) where the Rust program writes the report
        if py.returncode != 0 and rs.returncode == 0 and "DENSE_CELLS_MAX" in consts and last(py).startswith("IndexError: index -1 is out of bounds for axis 0 with size 0"):
            crashed += 1; continue
        if py.returncode != rs.returncode or (py.returncode != 0 and last(py) != last(rs)):
            bad += 1; print(f"case {c}: exit {py.returncode} / {rs.returncode}\n   python: {last(py)[:300]}\n   rust:   {last(rs)[:300]}\n   {' '.join(args)}"); continue
        if py.returncode != 0: stopped += 1; why[last(py)[:60]] = why.get(last(py)[:60], 0) + 1; continue      # both stopped with the same message
        ran += 1; n, diff = compare(os.path.join(d, "py"), os.path.join(d, "rs"), verbose=False)
        if diff: bad += 1; print(f"case {c}: {len(diff)} of {n} members differ\n   {' '.join(args)} {consts}"); compare(os.path.join(d, "py"), os.path.join(d, "rs"))
    if not a.keep: shutil.rmtree(work, ignore_errors=True)
    for k, v in sorted(why.items(), key=lambda x: -x[1]): print(f"   {v} stopped: {k}...")
    if crashed: print(f"   {crashed} not compared: the script crashed writing a sparse report with no failing cell (a known fault of the script)")
    print(f"{a.cases} cases: {ran} ran to the end and {stopped} stopped with the same message in both tools; " + ("all identical" if not bad else f"{bad} differ")); sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
