"""Score the planted-swap benchmark (swap_bench.py): detection against relative divergence, and false positives.

    python score_bench.py [--bench DIR] [--tag v0.11.0]

A planted cell's relative divergence is the swap pair's m over the median m of the planted taxon's five closest taxa,
both from the dataset's m.npz (the genome-wide divergence matrix of the baseline). It is detected when it fails the
too-divergent test (flag, or amb with fewer comparators than requested) or is masked as the identified recipient of a
too-similar pair (sim). A job is scored only when the same locusmasker.py made it and its baseline."""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from swap_bench import TAG

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bench", default=f"{HERE}/bench", help="the benchmark's directory (default: validation/bench)")
ap.add_argument("--tag", default=TAG, help=f"the tool version whose results to score (default: {TAG}, the current one)")
a = ap.parse_args(); B = a.bench
res = [json.load(open(f)) for f in sorted(glob.glob(f"{B}/results/{a.tag}/*.json"))]
base, nbh = {}, {}
for d in sorted({r["dataset"] for r in res}):
    try:
        base[d] = json.load(open(f"{B}/{d}/{a.tag}/baseline.json"))["tool"]
        Z = np.load(f"{B}/{d}/{a.tag}/m.npz"); M = Z["m"]
        Ms = np.sort(np.where(np.isfinite(M), M, np.inf), axis=1); nbh[d] = dict(zip(Z["names"].tolist(), np.median(Ms[:, :5], axis=1)))
    except FileNotFoundError: pass
rel, det, side, role, cells, fpc, fpa, lost, errs, stale = [], [], [], [], 0, 0, 0, 0, 0, 0
for r in res:
    if r["error"]: errs += 1; continue
    if r.get("tool") != base.get(r["dataset"]): stale += 1; continue      # the tool changed after the baseline was made
    cells += r["n_cells"]; fpc += len(r["fp_clean"]); fpa += len(r["fp_adjacent"]); lost += len(r["lost"])
    for ev in r["events"]:
        for s in ("a", "b"):
            st = ev.get(f"det_{s}", "untested")
            if st == "untested": continue
            nb = nbh.get(r["dataset"], {}).get(ev[s], np.nan)
            if not np.isfinite(nb): continue
            rel.append(ev["m_ab"] / max(nb, 0.05)); det.append(st in ("flag", "amb", "sim")); side.append(st); role.append(ev.get(f"role_{s}", ""))
rel, det, side, role = np.array(rel), np.array(det, bool), np.array(side), np.array(role)
nrun = len(res) - errs - stale
print(f"{a.tag}: {len(res)} runs ({errs} errors{f', {stale} stale: rerun swap_bench.py' if stale else ''}), {cells:,} cells tested, {len(rel)} planted cells scored")
print(f"FP clean {fpc} ({100*fpc/max(cells,1):.4f}%), adjacent {fpa}, baseline masks lost {lost} ({lost/max(nrun,1):.1f}/run)")
for lo, hi in [(0, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 10), (10, 30), (30, 1e9)]:
    s = (rel >= lo) & (rel < hi)
    if s.sum(): print(f"  rel {lo:>4}-{hi if hi < 1e9 else 'inf':<4} n={int(s.sum()):5d} detected {100*det[s].mean():5.1f}%")
s = rel >= 3
if s.sum():
    print(f"rel >= 3: {100*det[s].mean():.1f}% of {int(s.sum())}; by side among detected: {dict((str(k), int(c)) for k, c in zip(*np.unique(side[s & det], return_counts=True)))}")
    u = s & ~det & (role != "")
    print(f"rel >= 3 not detected but too similar to another taxon: {int(u.sum())} {dict((str(k), int(c)) for k, c in zip(*np.unique(role[u], return_counts=True)))}")
