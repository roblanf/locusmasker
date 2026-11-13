#!/usr/bin/env python3
"""The test suite's inputs: synthetic alignments with planted truths, excerpts of real alignments, and the golden outputs.

    python tests/data/make_fixtures.py            rewrites tests/data/synth/* (and tests/data/real/* when test_data/ is present)
    python tests/data/make_fixtures.py --golden   also runs the tool on every synthetic and simulated dataset and rewrites tests/data/golden/*
    python tests/data/make_fixtures.py --check    regenerates in memory and reports whether the committed files are identical

Every dataset is deterministic (numpy's default_rng with a fixed seed), so the committed files can be regenerated and
compared. Each synthetic dataset simulates sequences down a random bifurcating tree (one random length and rate per
locus, gamma site rates), drops fragments, then plants events on disjoint taxa and records them in truth.json:
  contam      a sequence replaced by random residues at one locus         -> masked, too divergent
  nearcopy    a donor's sequence copied over a close relative's at a locus -> the recipient masked as the copy (or both
              masked, direction uncertain, when the two fit their relatives equally badly)
  farcopy     the most distant taxon's sequence copied over a taxon's      -> too divergent from its relatives and too
              similar to the donor: masked by both tests
  sistercopy  one member of a cherry copied over the other                 -> too similar; the direction is usually
              uncertain, so both are masked
  excluded    a taxon with residues at too few loci                        -> excluded from the test, left unmasked
  skipped     a locus where too few taxa have residues                     -> the locus is not tested
  ambiguous   a taxon present in a short window at one locus that only three other taxa cover, then contaminated
              there -> fails with fewer comparators than requested (the ambiguous bit)
The same dna dataset is written in every input format the tool reads (FASTA + RAxML partitions, plain NEXUS, a NEXUS
with comments, quoted names and an interleaved matrix that only Biopython parses, relaxed and interleaved PHYLIP), so
the formats can be checked against each other."""
import argparse, csv, hashlib, io, json, os, sys, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
DNA, PROTEIN = "ACGT", "ACDEFGHIKLMNPQRSTVWY"


# ------------------------------------------------------------------ simulation
def random_tree(rng, n, depth):
    """a random bifurcating tree: parent and branch length per node, the leaves, and the patristic distance matrix"""
    parent, blen, leaves, nxt = {0: None}, {0: 0.0}, [0], 1
    while len(leaves) < n:
        i = leaves.pop(int(rng.integers(len(leaves))))
        for _ in range(2):
            parent[nxt] = i; blen[nxt] = float(rng.exponential(depth / np.log2(n))); leaves.append(nxt); nxt += 1
    leaves = sorted(leaves)
    def path(i):
        p = []
        while i is not None: p.append(i); i = parent[i]
        return p
    paths = {l: path(l) for l in leaves}
    # the patristic distance of a pair is the sum of the branches from each leaf up to (not including) their last common
    # ancestor. The leaves of an internal node's two subtrees meet exactly there, so the matrix is filled one node at a
    # time; each leaf's partial sums are Python's sum() over the same prefix of its path, so the values are bit for bit
    # those of the pairwise loop this replaces (which was quadratic in Python and too slow beyond a few hundred taxa)
    pos = {l: j for j, l in enumerate(leaves)}
    partial = {l: {v: sum(blen[u] for u in paths[l][:k]) for k, v in enumerate(paths[l])} for l in leaves}
    children = {}
    for v, pv in parent.items():
        if pv is not None: children.setdefault(pv, []).append(v)
    below = {}
    for v in sorted(parent, reverse=True):                           # a child always has a larger id than its parent
        below[v] = [v] if v in pos else [x for c in children[v] for x in below[c]]
    pat = np.zeros((n, n))
    for v, ch in children.items():
        ia = np.array([pos[x] for x in below[ch[0]]]); ib = np.array([pos[x] for x in below[ch[1]]])
        pa = np.array([partial[x][v] for x in below[ch[0]]]); pb = np.array([partial[x][v] for x in below[ch[1]]])
        blk = pa[:, None] + pb[None, :]; pat[np.ix_(ia, ib)] = blk; pat[np.ix_(ib, ia)] = blk.T
    cherries = [(a_, b_) for a_, la in enumerate(leaves) for b_, lb in enumerate(leaves) if a_ < b_ and parent[la] == parent[lb]]
    return parent, blen, leaves, pat, cherries


def simulate(rng, n, nloc, A, depth, len_lo, len_hi):
    """names, residues (n x L int8, -1 = missing), locus bounds, patristic distances, cherries"""
    parent, blen, leaves, pat, cherries = random_tree(rng, n, depth)
    lens = rng.integers(len_lo, len_hi + 1, nloc); locrate = np.exp(rng.uniform(np.log(0.4), np.log(2.5), nloc))
    pieces = {l: [] for l in leaves}
    for li in range(nloc):
        Ll = int(lens[li]); srate = rng.gamma(0.6, 1.0 / 0.6, Ll)
        node_seq = {0: rng.integers(0, A, Ll).astype(np.int8)}
        for v in sorted(parent):
            if parent[v] is None: continue
            base = node_seq[parent[v]]; d = blen[v] * locrate[li] * float(np.exp(rng.normal(0, 0.3)))
            p = 1 - np.exp(-d * srate); s = base.copy(); mut = rng.random(Ll) < p
            s[mut] = (s[mut] + rng.integers(1, A, int(mut.sum()))) % A; node_seq[v] = s
        for l in leaves: pieces[l].append(node_seq[l])
    X = np.stack([np.concatenate(pieces[l]) for l in leaves])
    names = [f"t{j:02d}" for j in range(n)]
    return names, X, np.r_[0, np.cumsum(lens)], pat, cherries


def plant(rng, names, X, bounds, pat, cherries, spec, A):
    """events on disjoint taxa; returns truth records. X is modified in place. The structural events (excluded taxon,
    skipped locus, ambiguous cell) come first and reserve their taxa and loci; the copies and contaminants then use
    only free taxa and the other loci"""
    n, nloc = len(names), len(bounds) - 1
    free = list(range(n)); rng.shuffle(free); reserved_loci = set()
    def take(cond=None):
        for i, t in enumerate(free):
            if cond is None or cond(t): return free.pop(i)
        raise RuntimeError("no taxon left to plant on")
    def locus(*taxa):                                            # a free locus where every taxon named has residues enough for an event (the present datasets: every locus)
        need = spec.get("event_min_sites", 20)
        return int(rng.choice([li for li in range(nloc) if li not in reserved_loci and all((X[t, bounds[li]:bounds[li + 1]] >= 0).sum() >= need for t in taxa)]))
    truth = []
    for li in range(nloc):                                      # fragments: a contiguous stretch dropped per (taxon, locus)
        for t in range(n):
            if rng.random() < spec.get("pmiss", 0.0):
                a, b = bounds[li], bounds[li + 1]; w = int((b - a) * rng.uniform(0.2, 0.7)); st = a + int(rng.integers(0, b - a - w + 1))
                X[t, st:st + w] = -1
    if spec.get("pabsent", 0.0):                                  # whole-locus absences: at most two per taxon, so no taxon turns thin by accident
        for t in range(n):
            k = min(int(rng.binomial(nloc, spec["pabsent"])), 2)
            for li in rng.choice(nloc, k, replace=False).tolist(): X[t, bounds[li]:bounds[li + 1]] = -1
    for j in range(spec.get("excluded", 0)):
        t = take(); keep = sorted(rng.choice(nloc, 6, replace=False).tolist())
        for li in range(nloc):
            if li not in keep: X[t, bounds[li]:bounds[li + 1]] = -1
        rec = dict(kind="excluded", taxon=names[t], loci_present=[f"locus{li + 1}" for li in keep])
        if j < spec.get("thin_identical", 0):                       # a thin taxon identical on every shared site to a tested one: its loci copied from a free donor
            d = take()
            for li in keep: X[t, bounds[li]:bounds[li + 1]] = X[d, bounds[li]:bounds[li + 1]]
            rec["identical_to"] = names[d]
        truth.append(rec)
    for _ in range(spec.get("skipped", 0)):
        li = nloc - 1; reserved_loci.add(li); keep = sorted(rng.choice(free, 3, replace=False).tolist())
        for t in range(n):
            if t not in keep: X[t, bounds[li]:bounds[li + 1]] = -1
        truth.append(dict(kind="skipped", locus=f"locus{li + 1}", taxa_present=[names[t] for t in keep]))
    for k in spec.get("clones", []):                               # k copies of one free genome, whole rows, at scattered indices: identical genomes
        d = take(); copies = [take() for _ in range(k - 1)]
        for r in copies: X[r] = X[d]
        truth.append(dict(kind="clone", donor=names[d], copies=[names[r] for r in copies]))
    for _ in range(spec.get("chain", 0)):                          # A ~ B and B ~ C on their shared sites while A differs from C: B lacks the sites where C differs
        a_, b_, c_ = take(), take(), take(); sites_ = rng.choice(np.flatnonzero(X[a_] >= 0), 20, replace=False)
        X[b_] = X[a_]; X[b_, sites_] = -1; X[c_] = X[a_]; X[c_, sites_] = (X[c_, sites_] + rng.integers(1, A, len(sites_))) % A
        truth.append(dict(kind="chain", taxa=[names[a_], names[b_], names[c_]], sites=int(len(sites_))))
    for _ in range(spec.get("multicopy", 0)):                      # one donor's sequence over three recipients at one locus: equal counts, a tie for closest
        d = take(); rs = [take() for _ in range(3)]; li = locus(d, *rs); a, b = bounds[li], bounds[li + 1]
        for r in rs: X[r, a:b] = X[d, a:b]
        truth.append(dict(kind="multicopy", donor=names[d], recipients=[names[r] for r in rs], locus=f"locus{li + 1}"))
    for _ in range(spec.get("ambiguous", 0)):
        t = take(); li = nloc - 2; reserved_loci.add(li); a, b = bounds[li], bounds[li + 1]; w0 = a + 10; w1 = w0 + 28
        X[t, a:w0] = -1; X[t, w1:b] = -1
        others = [c for c in np.argsort(pat[t]) if c != t and c in free and (X[c, a:b] >= 0).sum() >= 20][:3]   # only its three closest cover the window
        for c in range(n):
            if c != t and c not in others: X[c, w0:w1] = -1
        seg = X[t, w0:w1]; seg[:] = rng.integers(0, A, len(seg))
        truth.append(dict(kind="ambiguous", taxon=names[t], locus=f"locus{li + 1}", comparators=[names[c] for c in others]))
    for _ in range(spec.get("contam", 0)):
        t = take(); li = locus(t); a, b = bounds[li], bounds[li + 1]; seg = X[t, a:b]; pres = seg >= 0
        seg[pres] = rng.integers(0, A, int(pres.sum())); truth.append(dict(kind="contam", taxon=names[t], locus=f"locus{li + 1}"))
    for _ in range(spec.get("nearcopy", 0)):
        d = take(); order = [c for c in np.argsort(pat[d]) if c in free]; cands = order[2:5]     # the 3rd to 5th closest free taxa: close, but not the sister
        if not cands: raise RuntimeError("no near recipient free")
        r = cands[int(rng.integers(len(cands)))]; free.remove(r); li = locus(d, r); a, b = bounds[li], bounds[li + 1]
        X[r, a:b] = X[d, a:b]; truth.append(dict(kind="nearcopy", donor=names[d], recipient=names[r], locus=f"locus{li + 1}", patristic=float(pat[d, r])))
    for _ in range(spec.get("farcopy", 0)):
        r = take(); d = max(free, key=lambda c: pat[r, c]); free.remove(d); li = locus(r, d); a, b = bounds[li], bounds[li + 1]
        X[r, a:b] = X[d, a:b]; truth.append(dict(kind="farcopy", donor=names[d], recipient=names[r], locus=f"locus{li + 1}", patristic=float(pat[d, r])))
    for _ in range(spec.get("sistercopy", 0)):
        pair = next(((a_, b_) for a_, b_ in cherries if a_ in free and b_ in free), None)
        if pair is None: raise RuntimeError("no cherry free")
        d, r = pair; free.remove(d); free.remove(r); li = locus(d, r); a, b = bounds[li], bounds[li + 1]
        X[r, a:b] = X[d, a:b]; truth.append(dict(kind="sistercopy", donor=names[d], recipient=names[r], locus=f"locus{li + 1}", patristic=float(pat[d, r])))
    for _ in range(spec.get("hiddencontam", 0)):                   # a contaminated stretch exactly where the near side of the
        # tree has no coverage: the close-relative test compares on the clean remainder and sees nothing, but half
        # the broad sample's far members lean too divergent - the excess arm masks it red (removed 3 Nov 2026,
        # reinstated the same day under the one-direction call)
        t = take(); near = [c for c in np.argsort(pat[t]) if c in free][:spec.get("hidden_blanked", 30)]
        for c in near: free.remove(c)
        li = locus(t, *near); reserved_loci.add(li); a, b = bounds[li], bounds[li + 1]
        w0 = a + int((b - a) * 0.10); w1 = w0 + int((b - a) * 0.55)
        for c in near: X[c, w0:w1] = -1
        seg = X[t, w0:w1]; pres = seg >= 0
        seg[pres] = rng.integers(0, A, int(pres.sum()))
        truth.append(dict(kind="hiddencontam", taxon=names[t], locus=f"locus{li + 1}", relatives=[names[c] for c in near], window=int(w1 - w0)))
    for _ in range(spec.get("cladecopy", 0)):                      # a distant donor's sequence over a whole small clade, each
        # member with its own noise: the members move together, so each one's closest relatives moved with it and the
        # closest-relative tests see nothing; the broad sample sees the young coalescence with the donor's side
        # (a far shortfall: the third test's blue). The noise keeps the members apart, so no pair is "too similar"
        pair = next(((a_, b_) for a_, b_ in cherries if a_ in free and b_ in free), None)
        if pair is None: raise RuntimeError("no cherry free for cladecopy")
        near = [c for c in np.argsort(pat[pair[0]]) if c in free and c not in pair]
        C = list(pair) + near[:3]                                    # the cherry and its three nearest free taxa: a small clade
        d = max((c for c in free if c not in C), key=lambda c: float(min(pat[c, r] for r in C)))
        for r in C: free.remove(r)
        free.remove(d)
        li = locus(d, *C); reserved_loci.add(li); a, b = bounds[li], bounds[li + 1]
        for r in C:
            seg = X[d, a:b].copy(); pres = seg >= 0
            mm = (rng.random(b - a) < 0.06) & pres
            seg[mm] = (seg[mm] + rng.integers(1, A, int(mm.sum()))) % A
            X[r, a:b] = seg
        truth.append(dict(kind="cladecopy", donor=names[d], recipients=[names[r] for r in C], locus=f"locus{li + 1}", patristic=float(min(pat[d, r] for r in C))))
    for _ in range(spec.get("cladejunk", 0)):                      # a random sequence over a whole small clade, each member with
        # its own noise: the members share the junk, so each one's closest relatives agree with it and the closest-relative
        # tests see nothing; against the broad sample the clade is far too divergent (the third test's red, by majority)
        pair = next(((a_, b_) for a_, b_ in cherries if a_ in free and b_ in free), None)
        if pair is None: raise RuntimeError("no cherry free for cladejunk")
        near = [c for c in np.argsort(pat[pair[0]]) if c in free and c not in pair]
        C = list(pair) + near[:3]
        for r in C: free.remove(r)
        li = locus(*C); reserved_loci.add(li); a, b = bounds[li], bounds[li + 1]
        junk = rng.integers(0, A, b - a)
        for r in C:
            seg = junk.copy(); pres = X[r, a:b] >= 0
            mm = (rng.random(b - a) < 0.06) & pres
            seg[mm] = (seg[mm] + rng.integers(1, A, int(mm.sum()))) % A
            seg[~pres] = -1; X[r, a:b] = seg
        truth.append(dict(kind="cladejunk", recipients=[names[r] for r in C], locus=f"locus{li + 1}"))
    for _ in range(spec.get("scramble", 0)):                       # the sequences of a set of scattered taxa at one locus permuted
        # among themselves (a plate swap): each keeps a plausible sequence, just someone else's, so the row's residuals
        # scatter both ways and half its far comparators lean high (the third test's excess side, red)
        ks = spec.get("scramble_taxa", 12)
        T = [take() for _ in range(ks)]; li = locus(*T); reserved_loci.add(li); a, b = bounds[li], bounds[li + 1]
        perm = list(T[1:]) + [T[0]]                                  # a cycle: no one keeps their own
        old_rows = {t: X[t, a:b].copy() for t in T}
        for t, src in zip(T, perm): X[t, a:b] = old_rows[src]
        truth.append(dict(kind="scramble", taxa=[names[t] for t in T], locus=f"locus{li + 1}"))
    for _ in range(spec.get("sliver", 0)):                         # a cell cut to 8-15 sites: the taxon needs fewer shared sites than fit-overlap there
        t = take(); li = locus(t); a, b = bounds[li], bounds[li + 1]; w = int(rng.integers(8, 16)); st = a + int(rng.integers(0, b - a - w))
        X[t, a:st] = -1; X[t, st + w:b] = -1
        truth.append(dict(kind="sliver", taxon=names[t], locus=f"locus{li + 1}", sites=w))
    for _ in range(spec.get("lowloci", 0)):                        # present at 11-12 loci only: many of its pairs share fewer than MIN_LOCI_FOR_M loci
        have = lambda t: [li for li in range(nloc) if (X[t, bounds[li]:bounds[li + 1]] >= 0).sum() >= 20]
        t = take(lambda t: len(have(t)) >= nloc - 1); free_loci = [li for li in have(t) if li not in reserved_loci]
        drop = rng.choice(free_loci, min(int(rng.integers(2, 4)), len(have(t)) - 10), replace=False).tolist()    # it stays at ten loci or more: tested, not thin
        for li in drop: X[t, bounds[li]:bounds[li + 1]] = -1
        truth.append(dict(kind="lowloci", taxon=names[t], loci_absent=[f"locus{li + 1}" for li in drop]))
    return truth


# ------------------------------------------------------------------ writers
def residues(X, A):
    AB = DNA if A == 4 else PROTEIN
    lut = np.frombuffer((AB + "-").encode(), dtype=np.uint8); return [bytes(lut[np.where(r >= 0, r, len(AB))]).decode() for r in X]


def write_fasta_parts(d, names, seqs, bounds, dtype):
    with open(f"{d}/alignment.fasta", "w") as fh:
        for t, s in zip(names, seqs): fh.write(f">{t}\n" + "\n".join(s[i:i + 80] for i in range(0, len(s), 80)) + "\n")
    with open(f"{d}/partitions.txt", "w") as fh:
        fh.write("# one locus per line, RAxML style\n")
        for i in range(len(bounds) - 1): fh.write(f"{'DNA' if dtype == 'dna' else 'WAG'}, locus{i + 1} = {bounds[i] + 1}-{bounds[i + 1]}\n")


def write_nexus_plain(path, names, seqs, bounds, dtype):
    with open(path, "w") as fh:
        fh.write(f"#NEXUS\nbegin data;\ndimensions ntax={len(names)} nchar={len(seqs[0])};\nformat datatype={'dna' if dtype == 'dna' else 'protein'} missing=? gap=-;\nmatrix\n")
        for t, s in zip(names, seqs): fh.write(f"{t} {s}\n")
        fh.write(";\nend;\n\nbegin sets;\n")
        for i in range(len(bounds) - 1): fh.write(f"charset locus{i + 1} = {bounds[i] + 1}-{bounds[i + 1]};\n")
        fh.write("end;\n")


def write_nexus_quirky(path, names, seqs, bounds, dtype):
    """what only Biopython parses: a comment, quoted names, an interleaved matrix, a charpartition and a taxset"""
    with open(path, "w") as fh:
        fh.write(f"#NEXUS\n[written by the LocusMasker test suite]\nbegin data;\n  dimensions ntax={len(names)} nchar={len(seqs[0])};\n"
                 f"  format datatype={'dna' if dtype == 'dna' else 'protein'} missing=? gap=- interleave=yes;\n  matrix\n")
        L = len(seqs[0]); pad = max(len(t) for t in names) + 2
        for s0 in range(0, L, 60):
            for t, s in zip(names, seqs): fh.write(f"  '{t}'".ljust(pad + 3) + s[s0:s0 + 60] + "\n")
            fh.write("\n")
        fh.write("  ;\nend;\n\nbegin sets;\n")
        for i in range(len(bounds) - 1): fh.write(f"  charset locus{i + 1} = {bounds[i] + 1}-{bounds[i + 1]};\n")
        fh.write("  charpartition loci = " + ", ".join(f"locus{i + 1}:locus{i + 1}" for i in range(len(bounds) - 1)) + ";\n")
        fh.write(f"  taxset first_two = '{names[0]}' '{names[1]}';\nend;\n")


def write_phylip(path, names, seqs, interleaved):
    with open(path, "w") as fh:
        fh.write(f"{len(names)} {len(seqs[0])}\n")
        if not interleaved:
            for t, s in zip(names, seqs): fh.write(f"{t} {s}\n")
        else:
            L = len(seqs[0]); pad = max(len(t) for t in names) + 1
            for t, s in zip(names, seqs): fh.write(t.ljust(pad) + s[:50] + "\n")
            for s0 in range(50, L, 50):
                fh.write("\n")
                for s in seqs: fh.write(s[s0:s0 + 50] + "\n")


# ------------------------------------------------------------------ the datasets
DATASETS = {
    "dna_small": dict(seed=11, n=30, nloc=14, dtype="dna", depth=0.35, lens=(140, 360), pmiss=0.12, contam=2, nearcopy=2, farcopy=1, sistercopy=1, excluded=1, skipped=1, ambiguous=1, formats="all"),
    "aa_small": dict(seed=23, n=28, nloc=14, dtype="aa", depth=0.6, lens=(120, 300), pmiss=0.10, contam=2, nearcopy=2, farcopy=1, sistercopy=1, excluded=1, skipped=1, ambiguous=1, formats="fasta"),
    "dna_clean": dict(seed=37, n=24, nloc=12, dtype="dna", depth=0.3, lens=(140, 300), pmiss=0.05, formats="fasta"),
    "aa_clean": dict(seed=41, n=20, nloc=11, dtype="aa", depth=0.5, lens=(120, 240), pmiss=0.0, formats="fasta"),
    "dna_dense": dict(seed=53, n=40, nloc=16, dtype="dna", depth=0.55, lens=(120, 260), pmiss=0.15, contam=3, nearcopy=3, farcopy=2, sistercopy=1, formats="fasta"),
    # the broad-sample test's fixture: a clade moved wholesale (its members' closest relatives moved with it, so
    # only the broad sample sees the young coalescence: blue) and a clade overwritten by junk (its members agree with
    # each other, so only the broad sample sees how far from everyone they are: red), beside ordinary contaminants
    "dna_rc": dict(seed=85, n=100, nloc=14, dtype="dna", depth=0.3, lens=(280, 420), pmiss=0.08, contam=2, cladecopy=2, cladejunk=1,
                   event_min_sites=150, formats="fasta"),
    # the multi-block dataset of the large-data work: many taxa, clusters of identical genomes, a non-transitive identity
    # chain, slivers, whole-locus absences, taxa at few loci, thin taxa identical to tested ones (its patristic matrix is
    # 2,100 x 2,100 and is not written)
    "dna_blocks": dict(seed=71, n=2100, nloc=14, dtype="dna", depth=0.45, lens=(120, 300), pmiss=0.15, pabsent=0.10, clones=[60, 4, 3, 2, 2, 3], chain=1,
                       multicopy=1, sliver=30, lowloci=20, excluded=25, thin_identical=2, skipped=1, ambiguous=1, contam=6, nearcopy=6, farcopy=3,
                       event_min_sites=120, patristic=False, formats="fasta"),      # no sister copies: at 2,100 taxa the cherries are too close for a copy to show
}


def build(name, spec):
    rng = np.random.default_rng(spec["seed"]); A = 4 if spec["dtype"] == "dna" else 20
    names, X, bounds, pat, cherries = simulate(rng, spec["n"], spec["nloc"], A, spec["depth"], *spec["lens"])
    truth = plant(rng, names, X, bounds, pat, cherries, spec, A)
    return names, X, bounds, truth, pat


def write_dataset(d, name, spec):
    names, X, bounds, truth, pat = build(name, spec); seqs = residues(X, 4 if spec["dtype"] == "dna" else 20)
    os.makedirs(d, exist_ok=True)
    write_fasta_parts(d, names, seqs, bounds, spec["dtype"])
    if spec["formats"] == "all":
        write_nexus_plain(f"{d}/alignment.nex", names, seqs, bounds, spec["dtype"])
        write_nexus_quirky(f"{d}/alignment_quirky.nex", names, seqs, bounds, spec["dtype"])
        write_phylip(f"{d}/alignment.phy", names, seqs, False); write_phylip(f"{d}/alignment_interleaved.phy", names, seqs, True)
    meta = dict(name=name, datatype=spec["dtype"], taxa=len(names), loci=len(bounds) - 1, sites=int(bounds[-1]), spec={k: v for k, v in spec.items() if k != "formats"},
                locus_bounds=[[int(a) + 1, int(b)] for a, b in zip(bounds[:-1], bounds[1:])], events=truth)
    if spec.get("patristic", True): meta["patristic"] = {names[i]: [round(float(v), 4) for v in pat[i]] for i in range(len(names))}
    json.dump(meta, open(f"{d}/truth.json", "w"), indent=1)


# ------------------------------------------------------------------ real excerpts (from test_data/, which is not in the repository)
REAL = {
    "cannon_aa": dict(src="test_data/Cannon_2016_aa/alignment.nex", taxa=None, loci=12, dtype="protein",
                      note="Cannon et al. 2016 (CC0; doi 10.5061/dryad.107232), the 78 taxa of the amino-acid alignment at its first 12 loci"),
    "near_dna": dict(src="test_data/Near_2013/alignment.nex", taxa=40, loci=None, dtype="nucleotide",
                     note="Near et al. 2013 (CC0), the first 40 taxa of the 608-taxon nucleotide alignment at all 30 loci"),
}


# ------------------------------------------------------------------ simulated by AliSim: tests/data/make_random_trees.py writes them (it needs IQ-TREE 3); listed here for the goldens
SIM = {
    "aa_random_trees": dict(dtype="aa", note="100 taxa x 100 loci of 300 residues under LG+R12, a random Yule-Harding tree per locus, nothing planted"),
}


def write_real(d, name, spec):
    import locusmasker as lo
    src = os.path.join(ROOT, spec["src"])
    if not os.path.exists(src): print(f"  {name}: {spec['src']} not present, excerpt not rewritten"); return
    names, rows, charsets, info = lo.read_nexus_fast(src)
    keep_t = list(range(len(names)))[:spec["taxa"]] if spec["taxa"] else list(range(len(names)))
    loci = list(charsets.items())[:spec["loci"]] if spec["loci"] else list(charsets.items())
    sites = np.concatenate([np.asarray(v) for _, v in loci]); bounds = np.r_[0, np.cumsum([len(v) for _, v in loci])]
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/alignment.nex", "w") as fh:
        fh.write(f"#NEXUS\nbegin data;\ndimensions ntax={len(keep_t)} nchar={len(sites)};\nformat datatype={spec['dtype']} missing={info.missing or '?'} gap={info.gap or '-'};\nmatrix\n")
        for t in keep_t: fh.write(f"{names[t]} " + bytes(np.frombuffer(rows[t], dtype=np.uint8)[sites]).decode() + "\n")
        fh.write(";\nend;\n\nbegin sets;\n")
        for (k, _), a, b in zip(loci, bounds[:-1], bounds[1:]): fh.write(f"charset {k} = {a + 1}-{b};\n")
        fh.write("end;\n")
    json.dump(dict(name=name, note=spec["note"], taxa=len(keep_t), loci=len(loci), sites=int(len(sites)), datatype="dna" if spec["dtype"] == "nucleotide" else "aa"), open(f"{d}/meta.json", "w"), indent=1)


# ------------------------------------------------------------------ golden outputs
def run_tool(aln, parts, out, dtype, extra=()):
    """the tool in this process, as the tests run it; stderr captured"""
    import locusmasker as lo
    argv = ["locusmasker.py", "--aln", aln, "--out", out, "--datatype", dtype, "--progress", "none"] + (["--partitions", parts] if parts else []) + list(extra)
    old = sys.argv, sys.stderr; sys.argv = argv; sys.stderr = io.StringIO()
    try: lo.main()
    finally: log = sys.stderr.getvalue(); sys.argv, sys.stderr = old
    return log


def array_hashes(npz):
    """sha256 of every member of report.npz but the manifest (which carries the date and the command)"""
    z = zipfile.ZipFile(npz); return {k: hashlib.sha256(z.read(k)).hexdigest() for k in sorted(z.namelist()) if k != "manifest.json"}


def write_golden(g, d, spec, extra=(), patch=None):
    """the tool's outputs on a dataset pinned under tests/data/golden; patch sets module constants for the run"""
    import tempfile, shutil, locusmasker as lo
    out = tempfile.mkdtemp(prefix="golden_"); saved = {k: getattr(lo, k) for k in (patch or {})}
    for k, v in (patch or {}).items(): setattr(lo, k, v)
    try:
        run_tool(f"{d}/alignment.fasta", f"{d}/partitions.txt", out, spec["dtype"], extra)
        os.makedirs(g, exist_ok=True); shutil.copy(f"{out}/masked.csv", f"{g}/masked.csv")
        json.dump(array_hashes(f"{out}/report.npz"), open(f"{g}/report_hashes.json", "w"), indent=1)
        man = json.load(zipfile.ZipFile(f"{out}/report.npz").open("manifest.json"))
        json.dump(dict(counts=man["counts"], test=man["test"], format_version=man["format_version"], version=man["version"]), open(f"{g}/manifest_subset.json", "w"), indent=1)
    finally:
        shutil.rmtree(out)
        for k, v in saved.items(): setattr(lo, k, v)


def write_golden_real(g, d, dtype):
    import tempfile, shutil
    out = tempfile.mkdtemp(prefix="golden_")
    try:
        run_tool(f"{d}/alignment.nex", None, out, dtype)
        os.makedirs(g, exist_ok=True); shutil.copy(f"{out}/masked.csv", f"{g}/masked.csv")
        json.dump(array_hashes(f"{out}/report.npz"), open(f"{g}/report_hashes.json", "w"), indent=1)
        man = json.load(zipfile.ZipFile(f"{out}/report.npz").open("manifest.json"))
        json.dump(dict(counts=man["counts"], test=man["test"], format_version=man["format_version"], version=man["version"]), open(f"{g}/manifest_subset.json", "w"), indent=1)
    finally: shutil.rmtree(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--golden", action="store_true"); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    if a.check:
        import tempfile, filecmp, shutil
        tmp = tempfile.mkdtemp(); same = True
        for name, spec in DATASETS.items():
            write_dataset(f"{tmp}/{name}", name, spec)
            for f in sorted(os.listdir(f"{tmp}/{name}")):
                ok = os.path.exists(f"{HERE}/synth/{name}/{f}") and filecmp.cmp(f"{tmp}/{name}/{f}", f"{HERE}/synth/{name}/{f}", shallow=False)
                if not ok: same = False; print(f"differs: synth/{name}/{f}")
        shutil.rmtree(tmp); print("committed synthetic fixtures are identical" if same else "committed synthetic fixtures DIFFER"); sys.exit(0 if same else 1)
    for name, spec in DATASETS.items():
        write_dataset(f"{HERE}/synth/{name}", name, spec); print(f"synth/{name}: {spec['n']} taxa, {spec['nloc']} loci, {spec['dtype']}")
    for name, spec in REAL.items(): write_real(f"{HERE}/real/{name}", name, spec); print(f"real/{name}")
    if a.golden:
        for name, spec in DATASETS.items(): write_golden(f"{HERE}/golden/{name}", f"{HERE}/synth/{name}", spec); print(f"golden/{name}")
        write_golden(f"{HERE}/golden/dna_small_t1", f"{HERE}/synth/dna_small", DATASETS["dna_small"], ["--donor-threshold", "1"]); print("golden/dna_small_t1")
        write_golden(f"{HERE}/golden/dna_blocks_gtdb", f"{HERE}/synth/dna_blocks", DATASETS["dna_blocks"], ["--report-codes", "masked", "--report-sequences", "masked"],
                     patch=dict(DENSE_CELLS_MAX=10)); print("golden/dna_blocks_gtdb")                        # the options a GTDB-size run uses: sparse numeric columns, masked-only codes and sequences
        for name, spec in SIM.items(): write_golden(f"{HERE}/golden/{name}", f"{HERE}/sim/{name}", spec); print(f"golden/{name}")
        for name, spec in REAL.items():
            if os.path.exists(f"{HERE}/real/{name}/alignment.nex"):
                write_golden_real(f"{HERE}/golden/{name}", f"{HERE}/real/{name}", "dna" if spec["dtype"] == "nucleotide" else "aa"); print(f"golden/{name}")


if __name__ == "__main__":
    main()
