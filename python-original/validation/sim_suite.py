"""Simulation suite for locusmasker.py (LocusMasker 0.11 outputs): random trees, planted truths, scoring.

Each config: a random bifurcating tree, 14 loci (lengths 200-3000, rates 0.3-3x, gamma site
rates, lognormal branch noise, optional fragment missingness), then planted events on disjoint
taxa: contaminants (random residues), swaps (random partner), near-copies (partner among the 5
patristically nearest; recipient recorded), a chimera (5 loci from one distant donor), and a
misalignment window (30% of a cell's sites shuffled). Runs locusmasker.py (LO_TOOL overrides the
path; LO_BASE the working directory, where each config keeps its alignment, the run's outputs in
scan/ and the run's log in scan.log) and scores masked.csv and report.npz against the truth.

One line per config, summed in TOTALS. The leading fields keep their 0.10 meaning, so parsers of
them still work and the numbers compare across versions; two had to change, marked CHANGED.
  contam, swap, chimera, misalign a/b
          a of the b planted cells masked by test 1, too divergent: masked.csv reason divergent or
          both. CHANGED: 0.10 counted side == divergent only. "both" is a cell that failed test 1
          and is also an identified copy; in 0.11 test 2 looks at every pair of taxa, so a swapped
          or chimeric cell that fails test 1 is often also named a copy of its new sequence's
          relatives, and reading "divergent" alone would score those test 1 catches as misses.
  nearcopy a/b (dir c/d)
          a of the b planted pairs are significant too-similar pairs with an identified recipient,
          either way round: the pairs 0.11 masks (report.npz similar/<locus>.json, which replaces
          similar.tsv). d = a, and c of them identify the true recipient. TOTALS prints the
          direction as "direction c/d".
  FPdiv   cells masked by test 1 (divergent or both) that were not planted; planted is every cell
          an event touched, near-copy and chimera donors included. CHANGED like the counts above.
  FPsim   significant too-similar pairs with an identified recipient in which neither taxon was
          planted at that locus: pairs, as in 0.10.
  cells   tested cells (taxon x locus with comparators), report.npz manifest counts.cells, the
          rows the old cells.tsv had: in the TOTALS header and on the huge line.
Appended, new in 0.11:
  contam_any, swap_any, chimera_any, misalign_any a/b   planted cells masked for any reason.
  nearcopy_named a/b   the true recipient is masked as a copy (reason similar or both) and
          masked.csv names the donor as its source (the masking family's partner column).
  nearcopy_reversed n  the donor is masked as a copy naming the true recipient: the wrong direction.
  swap_named a/b, swap_named_ok n   of the b planted swap cells masked for any reason, a name a
          source (role copy in the masking family's columns), and n of those name the swap partner or one of its five
          closest taxa by m (report.npz taxa/closest.npy): the lineage the cell's sequence came from.
  chimera_named a/b, chimera_named_ok n   the same for chimeric cells; the right source is the donor.
  FPsim_cells  cells masked as a copy (similar or both) where neither the cell nor its named source
          was planted at that locus.
  FPany   cells masked for any reason that were not planted.

usage: python sim_suite.py [ncpu [tag,tag,...]]          the 20 standard configs, or those named
       python sim_suite.py ncpu huge [mid|deep|shallow]  one 10,000-taxon config, run alone
"""
import csv, io, json, os, subprocess, sys, zipfile
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.environ.get("LO_TOOL", os.path.join(os.path.dirname(HERE), "locusmasker.py"))
BASE = os.environ.get("LO_BASE", f"{HERE}/simsuite")
PROTEIN = "ACDEFGHIKLMNPQRSTVWY"
KINDS = ("contam", "swap", "nearcopy", "chimera", "misalign")
ANY = ("contam", "swap", "chimera", "misalign")                  # the kinds scored as planted cells
DIV, SIM = ("divergent", "both"), ("similar", "both")           # masked.csv reasons: failed test 1; an identified copy


def sim_dataset(cfg, rng):
    n, depth, A, pmiss = cfg["n"], cfg["depth"], cfg["alpha_size"], cfg["pmiss"]
    nloc = cfg.get("nloc", 14)
    lens = rng.integers(200, 3000, nloc)
    locrate = np.exp(rng.uniform(np.log(0.3), np.log(3), nloc))
    # random bifurcating tree: split a random leaf until n, exponential branch lengths
    parent, blen = {0: None}, {0: 0.0}
    leaves, nxt = [0], 1
    while len(leaves) < n:
        i = leaves.pop(int(rng.integers(len(leaves))))
        for _ in range(2):
            parent[nxt] = i; blen[nxt] = float(rng.exponential(depth / np.log2(n)))
            leaves.append(nxt); nxt += 1
    # patristic distances between leaves (for partner choice/reporting)
    def path_to_root(i):
        p = []
        while i is not None: p.append(i); i = parent[i]
        return p
    paths = {l: path_to_root(l) for l in leaves}
    depth_of = {}
    for l in leaves:
        d = 0.0
        for v in paths[l]: d += blen[v]
        depth_of[l] = d
    def patristic(a, b):
        pa, pb = set(paths[a]), paths[b]
        lca = next(v for v in pb if v in pa)
        dl = sum(blen[v] for v in paths[a][:paths[a].index(lca)])
        dr = sum(blen[v] for v in pb[:pb.index(lca)])
        return dl + dr
    # evolve each locus down the tree
    seqs = {l: np.empty(0, np.int8) for l in leaves}
    pieces = {l: [] for l in leaves}
    for li in range(nloc):
        Ll = int(lens[li])
        srate = rng.gamma(0.5, 2.0, Ll)
        node_seq = {0: rng.integers(0, A, Ll).astype(np.int8)}
        order = sorted(parent, key=lambda v: 0 if parent[v] is None else 1)
        for v in sorted(parent):
            if parent[v] is None: continue
            base = node_seq[parent[v]]
            d = blen[v] * locrate[li] * float(np.exp(rng.normal(0, 0.4)))
            p = 1 - np.exp(-d * srate)
            s = base.copy(); mut = rng.random(Ll) < p
            s[mut] = (s[mut] + rng.integers(1, A, int(mut.sum()))) % A
            node_seq[v] = s
        for l in leaves: pieces[l].append(node_seq[l])
    seqs = {l: np.concatenate(pieces[l]) for l in leaves}
    names = [f"t{j:03d}" for j in range(len(leaves))]
    lmap = dict(zip(leaves, names))
    bounds = np.r_[0, np.cumsum(lens)]
    # missingness: contiguous fragment dropped per (leaf, locus)
    present = {}
    chars = {}
    for l in leaves:
        arr = seqs[l].copy()
        for li in range(nloc):
            if rng.random() < pmiss:
                a, b = bounds[li], bounds[li + 1]
                frac = rng.uniform(0.3, 1.0); w = int((b - a) * frac)
                st = a + int(rng.integers(0, (b - a) - w + 1)) if w < b - a else a
                arr[st:st + w] = -1
        chars[lmap[l]] = arr
    # plant events on disjoint taxa
    pool = list(names); rng.shuffle(pool); poolset = set(pool)
    events = []
    def take():
        if not pool: return None
        x = pool.pop(); poolset.discard(x); return x
    leafof = {v: k for k, v in lmap.items()}
    for _ in range(cfg["contam"]):
        t = take(); li = int(rng.integers(nloc))
        if t is None: break
        a, b = bounds[li], bounds[li + 1]
        seg = chars[t][a:b]; pres = seg >= 0
        seg[pres] = rng.integers(0, A, int(pres.sum()))
        events.append({"kind": "contam", "taxon": t, "locus": f"locus{li + 1}"})
    for _ in range(cfg["swap"]):
        t1, t2 = take(), take()
        if t2 is None: break
        li = int(rng.integers(nloc)); a, b = bounds[li], bounds[li + 1]
        tmp = chars[t1][a:b].copy(); chars[t1][a:b] = chars[t2][a:b]; chars[t2][a:b] = tmp
        events.append({"kind": "swap", "taxon": t1, "taxon2": t2, "locus": f"locus{li + 1}",
                       "pat": patristic(leafof[t1], leafof[t2])})
    for _ in range(cfg["nearcopy"]):
        t1 = take()
        if t1 is None: break
        near = sorted((patristic(leafof[t1], leafof[x]), x) for x in names if x != t1 and x in poolset)[:5]
        if not near: break
        t2 = near[int(rng.integers(len(near)))][1]; pool.remove(t2); poolset.discard(t2)
        li = int(rng.integers(nloc)); a, b = bounds[li], bounds[li + 1]
        chars[t2][a:b] = chars[t1][a:b].copy()                      # t1's copy INTO t2
        events.append({"kind": "nearcopy", "donor": t1, "recipient": t2,
                       "locus": f"locus{li + 1}", "pat": near[0][0]})
    for _ in range(cfg["chimera"]):
        t = take()
        if t is None: break
        far = max((patristic(leafof[t], leafof[x]), x) for x in names if x != t)[1]
        lis = rng.choice(nloc, 5, replace=False)
        for li in lis:
            a, b = bounds[li], bounds[li + 1]
            chars[t][a:b] = chars[far][a:b].copy()
        events.append({"kind": "chimera", "taxon": t, "donor": far,
                       "loci": [f"locus{li + 1}" for li in lis]})
    for _ in range(cfg["misalign"]):
        t = take()
        if t is None: break
        li = int(rng.integers(nloc)); a, b = bounds[li], bounds[li + 1]
        w = (b - a) * 3 // 10; st = a + int(rng.integers(0, b - a - w))
        seg = chars[t][st:st + w]
        chars[t][st:st + w] = seg[rng.permutation(len(seg))]
        events.append({"kind": "misalign", "taxon": t, "locus": f"locus{li + 1}"})
    return names, chars, bounds, events


def run_config(cfg):
    rng = np.random.default_rng(cfg["seed"] * 1000 + cfg["id"])
    names, chars, bounds, events = sim_dataset(cfg, rng)
    tag = cfg["tag"]
    d = f"{BASE}/{tag}"; os.makedirs(d, exist_ok=True)
    AB = PROTEIN if cfg["alpha_size"] == 20 else "ACGT"
    with open(f"{d}/a.fasta", "w") as fh:
        for t in names:
            s = "".join(AB[c] if c >= 0 else "-" for c in chars[t])
            fh.write(f">{t}\n{s}\n")
    with open(f"{d}/a.parts.txt", "w") as fh:
        for i in range(len(bounds) - 1):
            fh.write(f"DNA, locus{i + 1} = {bounds[i] + 1}-{bounds[i + 1]}\n")
    r = subprocess.run([sys.executable, TOOL, "--aln", f"{d}/a.fasta", "--partitions",
                        f"{d}/a.parts.txt", "--out", f"{d}/scan", "--progress", "none",
                        "--datatype", "aa" if cfg["alpha_size"] == 20 else "dna"] + (["--threads", str(cfg["threads"])] if cfg.get("threads") else []),
                       capture_output=True, text=True, timeout=cfg.get("timeout", 1800))
    with open(f"{d}/scan.log", "w") as fh: fh.write(r.stderr)        # the run's log: 0.11 writes no run.log
    if r.returncode != 0:
        return {"tag": tag, "error": r.stderr.strip().splitlines()[-1] if r.stderr else "?"}
    try:
        res = score(f"{d}/scan", events)
    except Exception as e:
        return {"tag": tag, "error": f"scoring {d}/scan: {type(e).__name__}: {e}"}
    res["tag"] = tag
    return res


def zero():
    """a score with every field at zero: a/b fields are [a, b], the rest counts"""
    res = {k: [0, 0] for k in KINDS + tuple(f"{k}_any" for k in ANY) + ("nearcopy_named", "swap_named", "chimera_named")}
    res.update(dir_ok=0, dir_called=0, fp_div=0, fp_sim=0, cells=0, nearcopy_reversed=0, swap_named_ok=0,
               chimera_named_ok=0, fp_sim_cells=0, fp_any=0)
    return res


def read_scan(scan):
    """a 0.11 run's outputs: masked.csv rows by (locus, taxon); from report.npz every significant too-similar pair
    at each locus with its identified recipient (None when no direction was identified), each taxon's five closest
    taxa by m, and the manifest"""
    with zipfile.ZipFile(f"{scan}/report.npz") as z:
        man = json.loads(z.read("manifest.json"))
        if man.get("format_version", 0) < 4:
            raise RuntimeError(f"report format {man.get('format_version')} (LocusMasker {man.get('version')}): "
                               "this suite scores the outputs of 0.11 and later (format 4)")
        names = json.loads(z.read("taxa.json"))["names"]
        loci = json.loads(z.read("loci.json"))["names"]
        pairs = {}
        for member in z.namelist():
            if member.startswith("similar/") and member.endswith(".json"):
                locus = loci[int(member[len("similar/"):-len(".json")])]
                for e in json.loads(z.read(member)):                 # 0.12: one entry per too-similar test; a failing one is a pair
                    if e.get("fail") and e.get("partner", -1) >= 0:
                        key = (locus, frozenset((names[e["taxon"]], names[e["partner"]])))
                        pairs[key] = names[e["taxon"]] if e.get("role") == "copy" else names[e["partner"]]
        cl = np.load(io.BytesIO(z.read("taxa/closest.npy")))
    closest = {names[i]: {names[j] for j in row if j >= 0} for i, row in enumerate(cl.tolist())}
    with open(f"{scan}/masked.csv", newline="") as fh:
        masked = {(r["locus"], r["taxon"]): r for r in csv.DictReader(fh)}
    return masked, pairs, closest, man


def score(scan, events):
    """one run's scores against its planted events; the module docstring defines every field"""
    masked, pairs, closest, man = read_scan(scan)
    why = {c: r["reason"] for c, r in masked.items()}
    def source(c):                                  # the taxon a masked sequence is named a copy of, else None
        r = masked.get(c)
        if r is None: return None
        return r["closest_match_partner"] if r["closest_match_role"] == "copy" else None
    res = zero(); planted = set()
    def caught(kind, c, right=None):                # one planted cell: test 1, any reason, and the source it names
        res[kind][1] += 1; res[kind][0] += why.get(c) in DIV
        res[f"{kind}_any"][1] += 1; res[f"{kind}_any"][0] += c in why
        if right is not None and c in why:
            s = source(c); res[f"{kind}_named"][1] += 1
            if s is not None: res[f"{kind}_named"][0] += 1; res[f"{kind}_named_ok"] += s in right
    for ev in events:
        k = ev["kind"]
        if k == "chimera":
            for l in ev["loci"]: caught(k, (l, ev["taxon"]), {ev["donor"]})
            planted |= {(l, t) for l in ev["loci"] for t in (ev["taxon"], ev["donor"])}
        elif k == "swap":
            l = ev["locus"]
            for t, p in ((ev["taxon"], ev["taxon2"]), (ev["taxon2"], ev["taxon"])):
                caught(k, (l, t), {p} | closest.get(p, set()))       # the cell now holds p's sequence: p's lineage is its source
            planted |= {(l, ev["taxon"]), (l, ev["taxon2"])}
        elif k == "nearcopy":
            l, dn, rc = ev["locus"], ev["donor"], ev["recipient"]
            called = pairs.get((l, frozenset((dn, rc))))                # the identified recipient of the planted pair
            res[k][1] += 1; res["nearcopy_named"][1] += 1
            if called is not None:
                res[k][0] += 1; res["dir_called"] += 1; res["dir_ok"] += called == rc
            res["nearcopy_named"][0] += why.get((l, rc)) in SIM and source((l, rc)) == dn
            res["nearcopy_reversed"] += why.get((l, dn)) in SIM and source((l, dn)) == rc
            planted |= {(l, dn), (l, rc)}
        else:
            c = (ev["locus"], ev["taxon"]); caught(k, c); planted.add(c)
    res["fp_div"] = sum(1 for c, w in why.items() if w in DIV and c not in planted)
    res["fp_sim"] = sum(1 for (l, pr), rc in pairs.items() if rc is not None and not any((l, t) in planted for t in pr))
    res["fp_sim_cells"] = sum(1 for (l, t), r in masked.items()
                              if r["reason"] in SIM and (l, t) not in planted
                              and (l, r["closest_match_partner"] or r["broad_sample_partner"]) not in planted)
    res["fp_any"] = sum(1 for c in why if c not in planted)
    res["cells"] = int(man["counts"]["cells"])
    return res


def ab(v): return f"{v[0]}/{v[1]}"


def config_line(r):
    """the fields every version has printed, in their order"""
    return (f"{r['tag']:16s} contam {ab(r['contam'])}  swap {ab(r['swap'])}  nearcopy {ab(r['nearcopy'])} (dir {r['dir_ok']}/{r['dir_called']})"
            f"  chimera {ab(r['chimera'])}  misalign {ab(r['misalign'])}  FPdiv {r['fp_div']}  FPsim {r['fp_sim']}")


def new_fields(r):
    """the fields added for 0.11, appended after the old ones"""
    return ("".join(f"  {k}_any {ab(r[f'{k}_any'])}" for k in ANY)
            + f"  nearcopy_named {ab(r['nearcopy_named'])}  nearcopy_reversed {r['nearcopy_reversed']}"
            + f"  swap_named {ab(r['swap_named'])}  swap_named_ok {r['swap_named_ok']}"
            + f"  chimera_named {ab(r['chimera_named'])}  chimera_named_ok {r['chimera_named_ok']}"
            + f"  FPsim_cells {r['fp_sim_cells']}  FPany {r['fp_any']}")


# the large datasets (10,000 taxa x 100 loci), run one at a time with all threads: mid depth like the
# 80-taxon "mid" sets, very deep (saturation between distant taxa), and very shallow (dense sampling,
# nearest neighbours a fraction of a percent apart, where parsimony would do)
_huge = dict(seed=0, n=10000, alpha_size=4, pmiss=0.15, nloc=100, contam=200, swap=200, nearcopy=100, chimera=20, misalign=100, timeout=6 * 3600)
HUGE = {"mid": dict(_huge, id=99, tag="dna_huge_s0", depth=0.2),
        "deep": dict(_huge, id=98, tag="dna_huge_deep_s0", depth=1.5),
        "shallow": dict(_huge, id=97, tag="dna_huge_shallow_s0", depth=0.01)}


def standard_configs():
    """the 20 standard configs, in the order (so with the ids, seeds and data) they have always had"""
    cfgs = []
    i = 0
    for seed in range(4):
        for depth, dn in ((0.02, "shallow"), (0.2, "mid"), (0.8, "deep")):
            cfgs.append(dict(id=i, seed=seed, tag=f"dna_{dn}_s{seed}", n=80, depth=depth,
                             alpha_size=4, pmiss=0.15, contam=3, swap=4, nearcopy=3,
                             chimera=1, misalign=2)); i += 1
    for seed in range(2):
        cfgs.append(dict(id=i, seed=seed, tag=f"prot_mid_s{seed}", n=60, depth=0.5,
                         alpha_size=20, pmiss=0.15, contam=3, swap=4, nearcopy=3,
                         chimera=1, misalign=2)); i += 1
        cfgs.append(dict(id=i, seed=seed, tag=f"dna_patchy_s{seed}", n=80, depth=0.2,
                         alpha_size=4, pmiss=0.55, contam=3, swap=4, nearcopy=3,
                         chimera=1, misalign=2)); i += 1
        cfgs.append(dict(id=i, seed=seed, tag=f"dna_null_s{seed}", n=80, depth=0.2,
                         alpha_size=4, pmiss=0.15, contam=0, swap=0, nearcopy=0,
                         chimera=0, misalign=0)); i += 1
        cfgs.append(dict(id=i, seed=seed, tag=f"dna_small_s{seed}", n=14, depth=0.1,
                         alpha_size=4, pmiss=0.1, contam=2, swap=2, nearcopy=1,
                         chimera=0, misalign=1)); i += 1
    return cfgs


def main():
    ncpu = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    os.makedirs(BASE, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[2] == "huge":                 # one large dataset, run alone with all threads
        os.environ["OMP_NUM_THREADS"] = "1"                          # the tool's own threads do the work (BLAS stays single-threaded, as in the tool)
        r = run_config(dict(HUGE[sys.argv[3] if len(sys.argv) > 3 else "mid"], threads=ncpu))
        if r.get("error"): print(f"{r['tag']}: ERROR {r['error']}"); return
        print(config_line(r) + f"  cells {r['cells']:,}" + new_fields(r))
        return
    cfgs = standard_configs()
    if len(sys.argv) > 2:                                           # only the configs named
        want = sys.argv[2].split(","); bad = [t for t in want if t not in {c["tag"] for c in cfgs}]
        if bad: sys.exit(f"unknown config {', '.join(bad)}; the configs are {', '.join(c['tag'] for c in cfgs)}")
        cfgs = [c for c in cfgs if c["tag"] in want]
    os.environ["OMP_NUM_THREADS"] = "2"
    with ProcessPoolExecutor(max_workers=max(1, min(ncpu // 2, 16, len(cfgs)))) as ex:
        results = list(ex.map(run_config, cfgs))
    tot = zero()
    for r in results:
        if r.get("error"):
            print(f"{r['tag']}: ERROR {r['error']}"); continue
        for k, v in r.items():
            if isinstance(v, list): tot[k][0] += v[0]; tot[k][1] += v[1]
            elif k not in ("tag", "error"): tot[k] += v
        print(config_line(r) + new_fields(r))
    print(f"\nTOTALS over {tot['cells']:,} cells: " + "  ".join(f"{k} {ab(tot[k])}" for k in KINDS)
          + f"  direction {tot['dir_ok']}/{tot['dir_called']}  FPdiv {tot['fp_div']}  FPsim {tot['fp_sim']}" + new_fields(tot))


if __name__ == "__main__":
    main()
