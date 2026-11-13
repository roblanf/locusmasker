#!/usr/bin/env python3
"""M0 of the large-data plan (notes/LARGEDATA_2026-11.md): measure, on the OLD engine (feea34b, the last commit before the
block sweep: it reads prepare()'s S, W, m, mj and tri_idx, which the sweep no longer builds; run it with that commit's
locusmasker.py beside it), the quantities that decide the
block-sweep engine's constants and rules, dataset by dataset, from prepare()'s structures and the pinned report of each
dataset (a profile_corpus.py run made with --keep-outputs).

    python validation/m0_measure.py --pin notes/profiling_2026-10-30/pin --out notes/m0 [--datasets A B] [--threads 96]

Per dataset it writes <out>/<dataset>.json and prints one summary line; --report prints the table of every JSON in <out>.
What is measured, and why (the plan's M0):
  closest match   per (taxon, locus), today's search (locusmasker.py 1665-1686) rerun with bookkeeping: bands of
                  candidates within 1e-9 of the minimum d that hold more than one member, bands whose members have
                  UNEQUAL d (impossible unless 2 o1 o2 >= 1e9), ties of the (vals) key broken by index, those ties with
                  members in different index blocks of the planned block size, the largest o1 o2 seen, and cells where
                  a comparator ties the minimum but a non-comparator wins (inset without the winner in the set)
  comparators     the 1-based rank in order[g] (mj ascending) of the deepest accepted comparator, and of the deepest
                  candidate inspected, per taxon over its loci: what the candidate list C must hold
  sweep-1         the rank, in the stable argsort of m0 over finite resolved non-identical candidates, of the Kn-th
                  candidate that is its class representative: what sweep 1's list must hold for the rate neighbours
  lists           taxa with fewer than C finite mj entries (never truncated) and with fewer than Kn resolved neighbours
  classes         identical-genome classes: count and the largest
  outgroups       trees in the report and how many have an outgroup
  best match      per cell with a best match: the size of the tie at the top tier and of the challenger set T, and the
                  cells where the m tie key or the challengers decided
  demand          cells whose closest match is tested (forked), cells with a failing test, masked cells: the pairs the
                  new engine must count on demand"""
import argparse, io, json, os, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import locusmasker as lo
from profile_corpus import corpus

CS = (50, 100, 200, 400, 1000)


def load_report(path):
    z = zipfile.ZipFile(path); man = json.loads(z.read("manifest.json"))
    return z, (lambda k: np.load(io.BytesIO(z.read(k)))), man


def measure(ds, pin, threads):
    t0 = time.time(); name = ds["name"]
    names, rows, loci, *_ = lo.read_input(ds["aln"], ds.get("parts"))
    z, arr, man = load_report(os.path.join(pin, name, "t96", "run", "report.npz"))
    K = man["options"]["K_eff"]; fit = man["options"]["fit_overlap"]; n, nloc = len(names), len(loci)
    P = lo.prepare(argparse.Namespace(fit_overlap=fit, threads=threads), names, rows, loci, ds["dtype"], K, time.time())
    S, W, m, mj, cls, tri_idx, diag_pos, iu = P.S, P.W, P.m, P.mj, P.cls, P.tri_idx, P.diag_pos, P.iu
    comps = arr("cells/comparators.npy"); ncomp = arr("cells/n_comp.npy"); closest = arr("cells/locus_closest.npy"); side = arr("cells/side.npy")
    best = arr("cells/best_match.npy"); nfail = arr("cells/n_fail.npy")
    if comps.ndim != 3: raise SystemExit(f"{name}: the pinned report is sparse; this script reads dense cell arrays")
    b = max(32, n // 20)                                                     # the planned block size at 96 workers
    out = dict(dataset=name, taxa=n, loci=nloc, sites=int(sum(len(s) for _, s in loci)), K=K, fit_overlap=fit, has_dup=bool(P.has_dup), block=b)

    # ---- identical classes and the identity relation (as prepare builds them)
    tot_same = lo._expand(S.sum(0, dtype=np.float64), n, iu); tot_shared = lo._expand(W.sum(0, dtype=np.float64), n, iu)
    ident = (tot_shared > 0) & (tot_same == tot_shared); np.fill_diagonal(ident, False); del tot_same, tot_shared
    sizes = np.bincount(cls); out["classes"] = dict(copies=int((cls != np.arange(n)).sum()), largest=int(sizes.max()), classes_of_two_or_more=int((sizes >= 2).sum()))

    # ---- the comparator walk's depth: rank in order[g] (stable argsort of mj) of the deepest accepted comparator
    order = np.empty((n, n), np.int32)
    for b0 in range(0, n, 256): order[b0:b0 + 256] = np.argsort(mj[b0:b0 + 256], axis=1, kind="stable")
    rank = np.empty((n, n), np.int32); rank[np.arange(n)[:, None], order] = np.arange(n, dtype=np.int32)[None, :]; del order
    nfin = np.isfinite(mj).sum(1)
    valid = comps >= 0; rk = np.where(valid, rank[np.arange(n)[:, None, None], np.maximum(comps, 0)] + 1, 0)     # (n, nloc, K), 1-based
    deepest = rk.max(2)                                                      # per cell
    inspected = np.where((ncomp == K) | (ncomp == 0), deepest, nfin[:, None]) # fewer than K: the walk ran to the end of the finite candidates
    tested = ncomp > 0
    dep_taxon = np.where(tested, deepest, 0).max(1); insp_taxon = np.where(tested, inspected, 0).max(1)
    out["comparators"] = dict(deepest_accepted_max=int(dep_taxon.max()), deepest_accepted_p999=float(np.percentile(dep_taxon[dep_taxon > 0], 99.9)) if (dep_taxon > 0).any() else 0.0,
                              deepest_inspected_max=int(insp_taxon.max()), taxa_exceeding={str(C): int((dep_taxon > C).sum()) for C in CS},
                              taxa_inspected_exceeding={str(C): int((insp_taxon > C).sum()) for C in CS}, taxa_under_C_finite={str(C): int((nfin < C).sum()) for C in CS})
    del rank, rk

    # ---- sweep 1: the rank of the Kn-th class representative among the finite, resolved, non-identical candidates by m0
    m0, m0_fl, _ = lo.pair_divergences(S, W, n, iu, threads=threads)
    Kn = min(K, n - 1); res0 = m0 > m0_fl
    dist = np.where(np.isfinite(m0) & res0 & ~ident, m0, np.inf); np.fill_diagonal(dist, np.inf)
    rep = cls == np.arange(n); ranks1 = np.zeros(n, np.int64); short = 0
    for b0 in range(0, n, 256):
        o = np.argsort(dist[b0:b0 + 256], axis=1, kind="stable"); ok = np.isfinite(np.take_along_axis(dist[b0:b0 + 256], o, 1))
        cum = np.cumsum(ok & rep[o], axis=1); has = cum[:, -1] >= Kn
        ranks1[b0:b0 + 256] = np.where(has, np.argmax(cum >= Kn, axis=1) + 1, 0); short += int((~has).sum())
    out["sweep1"] = dict(kth_representative_rank_max=int(ranks1.max()), taxa_exceeding={str(C): int((ranks1 > C).sum()) for C in CS}, taxa_with_fewer_than_Kn=short)
    del m0, m0_fl, res0, dist

    # ---- the closest-match search of today (1665-1686), rerun with bookkeeping; and the best-match tiers (2005-2013)
    mj_fin = np.isfinite(mj); HALF = lo.HALF; lset = max(lo.LSET_CHUNK, lo.LSET_ELEMENTS // max(n, 1))
    def locus_stats(li):
        Wl, Sl = W[li], S[li]; nres = Wl[diag_pos].astype(int); rows = np.flatnonzero(ncomp[:, li] > 0)
        st = dict(bands=0, unequal_d=0, vals_ties=0, cross_block_ties=0, max_o1o2=0.0, inset_not_comparator=0, mismatch=0,
                  bm_cells=0, bm_tied_max=0, bm_T_max=0, bm_tied_gt1=0, bm_T_gt0=0)
        for b0 in range(0, len(rows), lset):
            rr = rows[b0:b0 + lset]; t_ = tri_idx[rr]; ov = Wl[t_].astype(np.float64); sm = Sl[t_].astype(np.float64)
            okc = (ov >= fit) & mj_fin[rr]
            with np.errstate(invalid="ignore", divide="ignore"): d = np.where(okc, (ov - sm + HALF) / np.maximum(ov, 1.0), np.inf)
            dmin = d.min(1); has = np.isfinite(dmin); band = d <= dmin[:, None] + 1e-9; nb = band.sum(1)
            multi = has & (nb > 1); st["bands"] += int(multi.sum())
            if multi.any():
                dm = np.where(band, d, np.nan); spread = np.nanmax(dm[multi], axis=1) - np.nanmin(dm[multi], axis=1); st["unequal_d"] += int((spread > 0).sum())
                vals = np.where(band, d + 1e-9 * mj[rr] / 100.0, np.inf)
                vmin = vals.min(1); tie = band & (vals == vmin[:, None]); nt = tie.sum(1); st["vals_ties"] += int((multi & (nt > 1)).sum())
                for r_ in np.flatnonzero(multi & (nt > 1)):
                    blocks = np.flatnonzero(tie[r_]) // b
                    if len(set(blocks.tolist())) > 1: st["cross_block_ties"] += 1
                ovb = np.where(band, ov, 0.0)[multi]; top2 = -np.partition(-ovb, 1, axis=1)[:, :2]; st["max_o1o2"] = max(st["max_o1o2"], float((top2[:, 0] * top2[:, 1]).max()))
            # the winner as the tool picks it, checked against the pinned report
            win = np.where(has, band.argmax(1), -1)
            if multi.any():
                ri, cj = np.nonzero(band[multi]); sub = np.flatnonzero(multi); vals_ = d[multi][ri, cj] + 1e-9 * mj[rr[sub[ri]], cj] / 100.0
                o_ = np.lexsort((cj, vals_, ri)); firsts = o_[np.unique(ri[o_], return_index=True)[1]]; win[sub] = cj[firsts]
            st["mismatch"] += int((win != closest[rr, li]).sum())
            top = comps[rr, li]; tops = np.zeros((len(rr), n), bool); r2, c2 = np.nonzero(top >= 0); tops[r2, top[r2, c2]] = True
            inset = ((d <= dmin[:, None]) & tops).any(1); st["inset_not_comparator"] += int((has & inset & ~tops[np.arange(len(rr)), np.maximum(win, 0)]).sum())
            # best match: the tier of the broadest overlap, its tie, and the challengers (2005-2013)
            bm = best[rr, li] >= 0
            if bm.any():
                sub = np.flatnonzero(bm); wrow = ov[sub]; srow = sm[sub]; okb = wrow >= lo.BEST_MATCH_MIN_SITES; okb[np.arange(len(sub)), rr[sub]] = False
                with np.errstate(invalid="ignore", divide="ignore"): dvg = np.where(okb, (wrow - srow) / np.maximum(wrow, 1.0), np.inf)
                wmax = np.where(okb, wrow, -1.0).max(1); dtier = np.where(okb & (wrow >= wmax[:, None]), dvg, np.inf); lo_t = dtier.min(1)
                tied = (dtier == lo_t[:, None]).sum(1); T = (np.isfinite(lo_t)[:, None] & (dvg < lo_t[:, None])).sum(1)
                st["bm_cells"] += len(sub); st["bm_tied_max"] = max(st["bm_tied_max"], int(tied.max())); st["bm_T_max"] = max(st["bm_T_max"], int(T.max()))
                st["bm_tied_gt1"] += int((tied > 1).sum()); st["bm_T_gt0"] += int((T > 0).sum())
        return st
    with ThreadPoolExecutor(max(1, min(threads, 32))) as ex: stats = list(ex.map(locus_stats, range(nloc)))
    agg = {k: (max(s[k] for s in stats) if k.endswith("_max") or k == "max_o1o2" else sum(s[k] for s in stats)) for k in stats[0]}
    out["closest"] = {k: agg[k] for k in ("bands", "unequal_d", "vals_ties", "cross_block_ties", "max_o1o2", "inset_not_comparator", "mismatch")}
    out["best_match"] = {k[3:]: agg[k] for k in agg if k.startswith("bm_")}

    # ---- outgroups, demand
    trees = 0; with_og = 0
    for k in z.namelist():
        if k.startswith("trees/"):
            t = json.loads(z.read(k)); trees += len(t["cells"]); with_og += sum(1 for og in t["outgroup"] if og is not None and int(og) >= 0)   # parallel lists, one entry per masked cell
    out["trees"] = dict(trees=trees, with_outgroup=with_og)
    out["demand"] = dict(cells_tested=int(tested.sum()), forked=int((side & 64 > 0).sum()), failing=int(((nfail >= 1) | (side & 64 > 0) | (side & 2 > 0)).sum()),
                         masked=int((side & 3 > 0).sum()), best_match_cells=int((best >= 0).sum()))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summary_line(o):
    c, cm, s1, bm = o["closest"], o["comparators"], o["sweep1"], o["best_match"]
    return (f"| {o['dataset']} | {o['taxa']:,} x {o['loci']:,} | {o['classes']['copies']} / {o['classes']['largest']} | {c['bands']:,} / {c['unequal_d']} / {c['vals_ties']:,} / {c['cross_block_ties']:,} | {c['max_o1o2']:.2e} | {c['inset_not_comparator']:,} / {c['mismatch']} "
            f"| {cm['deepest_accepted_max']} / {cm['deepest_accepted_p999']:.0f} / {cm['deepest_inspected_max']} | {cm['taxa_exceeding']['200']} / {cm['taxa_inspected_exceeding']['200']} | {s1['kth_representative_rank_max']} / {s1['taxa_exceeding']['200']} / {s1['taxa_with_fewer_than_Kn']} "
            f"| {bm['tied_max']} / {bm['T_max']} / {bm['tied_gt1']:,} / {bm['T_gt0']:,} | {o['trees']['with_outgroup']} / {o['trees']['trees']} | {o['demand']['forked']:,} / {o['demand']['failing']:,} / {o['demand']['masked']:,} | {o['seconds']} |")


HEADER = ("| dataset | taxa x loci | copies / largest class | bands>1 / unequal d / vals ties / cross-block | max o1 o2 | inset, winner not a comparator / mismatches with the report "
          "| comparator depth: accepted max / p99.9 / inspected max | taxa beyond C=200: accepted / inspected | sweep-1 Kn-th representative: max rank / taxa beyond 200 / fewer than Kn "
          "| best match: tie max / T max / cells tie>1 / cells T>0 | trees with outgroup / trees | forked / failing / masked cells | s |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pin", default=os.path.join(ROOT, "notes", "profiling_2026-10-30", "pin")); ap.add_argument("--out", default=os.path.join(ROOT, "notes", "m0"))
    ap.add_argument("--datasets", nargs="*"); ap.add_argument("--threads", type=int, default=96); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.report:
        print(HEADER)
        for f in sorted(os.listdir(a.out)):
            if f.endswith(".json"): print(summary_line(json.load(open(os.path.join(a.out, f)))))
        return
    sets = {d["name"]: d for d in corpus()}
    for name in a.datasets or sorted(sets):
        if not os.path.exists(os.path.join(a.pin, name, "t96", "run", "report.npz")): print(f"{name}: no pinned report, skipped", file=sys.stderr); continue
        o = measure(sets[name], a.pin, a.threads); json.dump(o, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
        print(HEADER.splitlines()[0] if name == (a.datasets or sorted(sets))[0] else "", end=""); print(summary_line(o), flush=True)


if __name__ == "__main__":
    main()
