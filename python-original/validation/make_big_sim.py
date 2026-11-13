#!/usr/bin/env python3
"""A large simulated alignment with planted events, for validating the block-sweep engine at size (the plan's M0 pin at
25K taxa and M3 at 200K; notes/LARGEDATA_2026-11.md). Not used by the test suite.

    python validation/make_big_sim.py --taxa 25000 --out test_data/Sim_25K --pilot            the tree and the scale choice
    python validation/make_big_sim.py --taxa 25000 --out test_data/Sim_25K --scale 0.05       the alignment and the events

AliSim (iqtree3) simulates every locus on ONE random Yule-Harding tree under LG+R12 with a per-locus rate (log-normal)
and a branch scale chosen so that the nearest-neighbour divergence resembles a real phylogenomic dataset: --pilot writes
the tree and prints, for candidate scales, the nearest-neighbour patristic distance of the tips against Crisp_2024's
nearest-neighbour m (AliSim's default branch lengths have a mean of 0.1 whatever the tip count, so a 25K-tip tree would
otherwise put every sister pair 20% apart). The events are planted with the logic of tests/data/make_fixtures.py
(contaminants, near and far copies, clone clusters of identical genomes, one donor over several recipients, thin taxa,
whole-marker absences) on disjoint taxa, and truth.json records them. Layout: the corpus's (alignment.fasta,
partitions.txt, README.yaml) plus truth.json and tree.nwk, so validation/profile_corpus.py can run it."""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
MODEL = "LG+R12"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(f"{' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


# ------------------------------------------------------------------ the tree
def make_tree(iqtree, out, n, seed):
    """a random Yule-Harding tree of n tips from AliSim, written to <out>/tree.nwk"""
    tmp = os.path.join(out, "tmp"); os.makedirs(tmp, exist_ok=True); pre = os.path.join(tmp, "tree")
    run([iqtree, "--alisim", pre, "-t", f"RANDOM{{yh,{n}}}", "-m", "LG", "--length", "10", "--seed", str(seed), "-af", "fasta"])
    shutil.copy(pre + ".treefile", os.path.join(out, "tree.nwk")); return os.path.join(out, "tree.nwk")


class Tree:
    """tips, root paths and prefix sums of a Newick tree, for distances between a few taxa and everyone (Bio.Phylo parses it)"""
    def __init__(self, path):
        from Bio import Phylo
        self.t = Phylo.read(path, "newick"); tips = self.t.get_terminals(); self.names = [c.name for c in tips]; self.pos = {nm: i for i, nm in enumerate(self.names)}
        self.paths = {c.name: self.t.get_path(c) for c in tips}
        self.pref = {nm: np.concatenate([[0.0], np.cumsum([x.branch_length or 0.0 for x in p])]) for nm, p in self.paths.items()}
        self.depth = {nm: float(self.pref[nm][-1]) for nm in self.names}
        self.ids = {nm: [id(x) for x in p] for nm, p in self.paths.items()}

    def row(self, a):
        """patristic distances from tip a to every tip (index order)"""
        pa = self.ids[a]; pref_a = self.pref[a]; first = {node: k for k, node in enumerate(pa)}; d = np.empty(len(self.names))
        for j, b in enumerate(self.names):
            k = 0; pb = self.ids[b]
            for node in pb:
                if node in first and first[node] == k: k += 1
                else: break
            d[j] = self.depth[a] + self.depth[b] - 2.0 * pref_a[k]
        return d

    def nearest_all(self):
        """every tip's nearest tip by patristic distance: for each ancestor, the shallowest tip in the sibling subtree"""
        clades = list(self.t.find_clades(order="postorder")); mind = {}
        for c in clades:
            if c.is_terminal(): mind[id(c)] = (0.0, c.name)
            else: mind[id(c)] = min((mind[id(ch)][0] + (ch.branch_length or 0.0), mind[id(ch)][1]) for ch in c.clades)
        parent = {}
        for c in clades:
            for ch in c.clades: parent[id(ch)] = c
        out = {}
        for nm in self.names:
            node = self.paths[nm][-1]; up = 0.0; best = (np.inf, None)
            while id(node) in parent:
                p = parent[id(node)]; up += node.branch_length or 0.0
                for sib in p.clades:
                    if sib is not node:
                        cand = up + (sib.branch_length or 0.0) + mind[id(sib)][0]
                        if cand < best[0]: best = (cand, mind[id(sib)][1])
                node = p
            out[nm] = best
        return out


# ------------------------------------------------------------------ the loci
def simulate_loci(iqtree, out, tree, nloc, scale, seed, jobs, len_lo=40, len_hi=60):
    rng = np.random.default_rng(seed); lens = rng.integers(len_lo, len_hi + 1, nloc); rates = np.exp(rng.normal(0.0, 0.5, nloc))
    tmp = os.path.join(out, "tmp"); os.makedirs(tmp, exist_ok=True)
    def one(li):
        pre = os.path.join(tmp, f"locus{li + 1}")
        if not os.path.exists(pre + ".fa"):
            run([iqtree, "--alisim", pre, "-t", tree, "-m", MODEL, "--length", str(int(lens[li])), "--branch-scale", f"{scale * rates[li]:.6g}", "--seed", str(seed + li + 1), "-af", "fasta"])
        return pre + ".fa"
    with ThreadPoolExecutor(jobs) as ex: files = list(ex.map(one, range(nloc)))
    return files, lens, rates


def read_fasta_bytes(path):
    names, seqs, cur = [], [], []
    for line in open(path, "rb"):
        if line.startswith(b">"):
            if cur: seqs.append(b"".join(cur)); cur = []
            names.append(line[1:].split()[0].decode())
        else: cur.append(line.strip())
    seqs.append(b"".join(cur)); return names, seqs


# ------------------------------------------------------------------ the events (tests/data/make_fixtures.py's logic, on the letter matrix; '-' is missing)
def plant(rng, names, X, bounds, tree, spec):
    n, nloc = len(names), len(bounds) - 1; free = list(range(n)); rng.shuffle(free); truth = []; GAP = ord("-"); AA = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", np.uint8)
    def take():
        if not free: raise RuntimeError("no taxon left to plant on")
        return free.pop()
    def locus(*taxa):
        return int(rng.choice([li for li in range(nloc) if all((X[t, bounds[li]:bounds[li + 1]] != GAP).sum() >= 20 for t in taxa)]))
    if spec.get("pabsent", 0.0):                                               # whole-locus absences: at most eight per taxon, so no taxon turns thin by accident
        for t in range(n):
            k = min(int(rng.binomial(nloc, spec["pabsent"])), 8)
            for li in rng.choice(nloc, k, replace=False).tolist(): X[t, bounds[li]:bounds[li + 1]] = GAP
    for _ in range(spec.get("excluded", 0)):
        t = take(); keep = sorted(rng.choice(nloc, 6, replace=False).tolist())
        for li in range(nloc):
            if li not in keep: X[t, bounds[li]:bounds[li + 1]] = GAP
        truth.append(dict(kind="excluded", taxon=names[t], loci_present=[f"locus{li + 1}" for li in keep]))
    for k in spec.get("clones", []):
        d = take(); copies = [take() for _ in range(k - 1)]
        for r in copies: X[r] = X[d]
        truth.append(dict(kind="clone", donor=names[d], copies=[names[r] for r in copies]))
    for _ in range(spec.get("multicopy", 0)):
        d = take(); rs = [take() for _ in range(3)]; li = locus(d, *rs); a, b = bounds[li], bounds[li + 1]
        for r in rs: X[r, a:b] = X[d, a:b]
        truth.append(dict(kind="multicopy", donor=names[d], recipients=[names[r] for r in rs], locus=f"locus{li + 1}"))
    for _ in range(spec.get("contam", 0)):
        t = take(); li = locus(t); a, b = bounds[li], bounds[li + 1]; seg = X[t, a:b]; pres = seg != GAP
        seg[pres] = AA[rng.integers(0, 20, int(pres.sum()))]; truth.append(dict(kind="contam", taxon=names[t], locus=f"locus{li + 1}"))
    freeset = set(free)
    for _ in range(spec.get("nearcopy", 0)):
        d = take(); freeset.discard(d); dist = tree.row(names[d]); order = [c for c in np.argsort(dist, kind="stable") if c in freeset]; cands = order[2:5]
        r = cands[int(rng.integers(len(cands)))]; free.remove(r); freeset.discard(r); li = locus(d, r); a, b = bounds[li], bounds[li + 1]
        X[r, a:b] = X[d, a:b]; truth.append(dict(kind="nearcopy", donor=names[d], recipient=names[r], locus=f"locus{li + 1}", patristic=float(dist[r])))
    for _ in range(spec.get("farcopy", 0)):
        r = take(); freeset.discard(r); dist = tree.row(names[r]); d = max(freeset, key=lambda c: dist[c]); free.remove(d); freeset.discard(d); li = locus(r, d); a, b = bounds[li], bounds[li + 1]
        X[r, a:b] = X[d, a:b]; truth.append(dict(kind="farcopy", donor=names[d], recipient=names[r], locus=f"locus{li + 1}", patristic=float(dist[d])))
    return truth


SPECS = {25000: dict(seed=1, pabsent=0.05, excluded=60, clones=[300, 100, 50, 20, 10, 5, 5, 3, 3] + [2] * 11, multicopy=10, contam=250, nearcopy=250, farcopy=125),
         200000: dict(seed=2, pabsent=0.05, excluded=500, clones=[300, 200, 100, 100, 50, 50, 20, 20, 10, 10] + [5] * 5 + [3] * 5 + [2] * 10, multicopy=30, contam=2000, nearcopy=2000, farcopy=1000)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--taxa", type=int, required=True); ap.add_argument("--loci", type=int, default=120); ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float); ap.add_argument("--pilot", action="store_true"); ap.add_argument("--jobs", type=int, default=16); ap.add_argument("--iqtree", default="iqtree3"); ap.add_argument("--seed", type=int)
    ap.add_argument("--locus-sites", type=int, nargs=2, default=[40, 60], metavar=("LO", "HI"), help="locus lengths drawn uniformly from [LO, HI] (default 40 60, the 25K pin; M3 uses 200 400: 50-site loci give the copy tests no room)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); spec = dict(SPECS[min(SPECS, key=lambda k: abs(k - a.taxa))]); seed = a.seed if a.seed is not None else spec["seed"]
    if shutil.which(a.iqtree) is None: sys.exit(f"{a.iqtree} not found")
    tree_path = os.path.join(a.out, "tree.nwk")
    if not os.path.exists(tree_path): t0 = time.time(); make_tree(a.iqtree, a.out, a.taxa, seed); print(f"tree: {tree_path} ({time.time() - t0:.0f} s)")
    tree = Tree(tree_path); assert len(tree.names) == a.taxa
    if a.pilot:
        nn = np.array([v[0] for v in tree.nearest_all().values()])
        import locusmasker as lo
        ds = os.path.join(ROOT, "test_data", "Crisp_2024"); names_, m, _ = lo.divergence_matrix(f"{ds}/alignment.fasta", f"{ds}/partitions.txt", "dna", threads=16)
        nn_m = np.nanmin(np.where(np.isfinite(m), m, np.nan), axis=1) / 100.0
        print(f"Crisp_2024 nearest-neighbour m (as a fraction): p10 {np.percentile(nn_m, 10):.4f} median {np.median(nn_m):.4f} p90 {np.percentile(nn_m, 90):.4f}")
        for sc in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            print(f"scale {sc:5.2f}: nearest-neighbour patristic p10 {np.percentile(nn * sc, 10):.4f} median {np.median(nn * sc):.4f} p90 {np.percentile(nn * sc, 90):.4f}   (tree depth median {np.median(list(tree.depth.values())) * sc:.3f})")
        return
    if a.scale is None: sys.exit("--scale is needed (run --pilot first)")
    t0 = time.time(); files, lens, rates = simulate_loci(a.iqtree, a.out, tree_path, a.loci, a.scale, seed, a.jobs, a.locus_sites[0], a.locus_sites[1]); print(f"{a.loci} loci simulated ({time.time() - t0:.0f} s)")
    names = None; cols = []
    for f in files:
        nm, sq = read_fasta_bytes(f)
        if names is None: names = nm; order = {x: i for i, x in enumerate(nm)}
        cols.append(np.array([np.frombuffer(sq[order[x]], np.uint8) for x in names]))
    X = np.concatenate(cols, axis=1); bounds = np.r_[0, np.cumsum(lens)]; del cols
    rng = np.random.default_rng(seed + 1000); truth = plant(rng, names, X, bounds, tree, spec)
    with open(f"{a.out}/alignment.fasta", "wb") as fh:
        for i, nm in enumerate(names): fh.write(b">" + nm.encode() + b"\n" + X[i].tobytes() + b"\n")
    with open(f"{a.out}/partitions.txt", "w") as fh:
        for li in range(a.loci): fh.write(f"LG, locus{li + 1} = {bounds[li] + 1}-{bounds[li + 1]}\n")
    import collections
    meta = dict(name=os.path.basename(a.out), datatype="aa", taxa=len(names), loci=a.loci, sites=int(bounds[-1]), events=truth, spec=spec, scale=a.scale, seed=seed, model=MODEL,
                locus_lengths=[int(x) for x in lens], locus_rates=[round(float(r), 4) for r in rates], event_counts=dict(collections.Counter(e["kind"] for e in truth)))
    json.dump(meta, open(f"{a.out}/truth.json", "w"), indent=1)
    open(f"{a.out}/README.yaml", "w").write(f'''study:
    reference: "simulated by validation/make_big_sim.py (AliSim, IQ-TREE 3) on {time.strftime('%Y-%m-%d')}: one random Yule-Harding tree of {len(names):,} tips, {a.loci} loci of {a.locus_sites[0]}-{a.locus_sites[1]} residues under {MODEL} with log-normal locus rates, branch scale {a.scale}, planted events in truth.json"
    year: 2026
data:
    datatype: protein
    taxa: {len(names)}
    loci: {a.loci}
    sites: {int(bounds[-1])}
    format: FASTA plus RAxML-style partitions.txt
''')
    shutil.rmtree(os.path.join(a.out, "tmp")); print(f"{a.out}: {len(names):,} taxa, {a.loci} loci, {int(bounds[-1]):,} sites; events {meta['event_counts']} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
