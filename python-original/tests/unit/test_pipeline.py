"""What a run's outputs must satisfy, read back from report.npz and masked.csv and checked against the input alignment:
the comparator choice, both tests' decisions and their side bits, the Bonferroni family, the closest match, the best
match, the site codes, the carried sequences and trees, and the masked alignment. Parametrised over every synthetic
dataset; each check is a statement of the method, not of a number the code happened to produce."""
import csv, json, math, os, re
import numpy as np
import pytest
from tests.conftest import lo, dataset, SYNTH

LN10 = math.log(10.0)


@pytest.fixture(scope="module", params=SYNTH)
def ctx(request, runs):
    """a standard run with the input decoded beside it"""
    name = request.param; ds = dataset(name); run = runs(name); R = run.report
    names, rows, loci, fmt, declared, nexus = lo.read_input(ds.aln, ds.parts)
    alphabet = lo.DNA if ds.dtype == "dna" else lo.PROTEIN; lut = np.full(256, -1, np.int8)
    for i, c in enumerate(alphabet): lut[ord(c)] = i; lut[ord(c.lower())] = i
    X = np.stack([lut[np.frombuffer(r, dtype=np.uint8)] for r in rows])
    _, m, m_hi = lo.divergence_matrix(ds.aln, ds.parts, ds.dtype)
    return dict(name=name, ds=ds, run=run, R=R, names=names, loci=loci, X=X, m=m, bar=R.man["test"]["log10_bar"], masked=run.masked)


def cells_tested(R):
    nc = R.arr("cells/n_comp.npy"); return [(int(g), int(li)) for g, li in zip(*np.nonzero(nc > 0))]


def identical(X, m):
    """taxa identical to each taxon at every shared site (the pairs with m == 0 checked against the residues), and the
    class of every taxon (connected components of that relation): the identical-genome rule of 0.13.1"""
    n = len(m); ident = {g: set() for g in range(n)}; cls = list(range(n))
    def find(a):
        while cls[a] != a: cls[a] = cls[cls[a]]; a = cls[a]
        return a
    for g, c in np.argwhere(np.triu(m == 0.0, 1)):
        both = (X[g] >= 0) & (X[c] >= 0)
        if both.any() and (X[g][both] == X[c][both]).all():
            ident[g].add(int(c)); ident[c].add(int(g)); ra, rb = find(int(g)), find(int(c)); cls[max(ra, rb)] = min(ra, rb)
    return ident, [find(g) for g in range(n)]


def shared(X, sites, i, j):
    a, b = X[i, sites], X[j, sites]; both = (a >= 0) & (b >= 0); return int(both.sum()), int(((a != b) & both).sum())


# ------------------------------------------------------------------ the shape of the report
class TestShape:
    def test_every_format_9_member_is_present(self, ctx):
        R = ctx["R"]
        for k in ("n_comp", "n_fail", "n_similar", "side", "side2", "focal_sites", "med_identity", "med_expected", "excess_m", "closest", "closest_overlap", "closest_identity", "closest_expected",
                  "best_match", "best_match_identity", "partner", "div_log10p", "sim_log10p", "locus_closest", "locus_closest_log10p", "comparators", "comp_div_log10p", "comp_sim_log10p"):
            assert R.has(f"cells/{k}.npy"), k
        assert R.has("taxa/closest.npy") and R.man["format_version"] == 9 and R.dense

    def test_cell_arrays_have_the_grid_shape(self, ctx):
        R = ctx["R"]; n, nloc, K = R.n, R.nloc, R.K
        for k in ("n_comp", "n_fail", "side", "partner", "locus_closest", "div_log10p"): assert R.arr(f"cells/{k}.npy").shape == (n, nloc), k
        for k in ("comparators", "comp_div_log10p", "comp_sim_log10p"): assert R.arr(f"cells/{k}.npy").shape == (n, nloc, K), k

    def test_tested_cells_are_the_present_cells_of_tested_taxa_at_tested_loci(self, ctx):
        R = ctx["R"]; nc = R.arr("cells/n_comp.npy"); fs = R.arr("cells/focal_sites.npy")
        excluded = {i for i, s in enumerate(R.taxa["status"]) if s != "tested"}; skipped = {i for i, s in enumerate(R.loci["status"]) if s != "tested"}
        for g in range(R.n):
            for li in range(R.nloc):
                if nc[g, li] > 0: assert g not in excluded and li not in skipped and fs[g, li] > 0
                if g in excluded or li in skipped: assert nc[g, li] == 0
        assert 0 < nc.max() <= R.K


# ------------------------------------------------------------------ the comparators
class TestComparators:
    def test_distinct_not_self_not_excluded_and_overlapping(self, ctx):
        R, X, loci = ctx["R"], ctx["X"], ctx["loci"]; excluded = {i for i, s in enumerate(R.taxa["status"]) if s != "tested"}; fs = R.arr("cells/focal_sites.npy")
        for g, li in cells_tested(R):
            comps = R.comps(g, li); assert len(comps) == R.cell("n_comp", g, li) and len(set(comps)) == len(comps) and g not in comps
            need = min(20, int(fs[g, li]))
            for c in comps:
                assert c not in excluded and shared(X, loci[li][1], g, c)[0] >= need

    def test_comparators_are_the_closest_by_m_in_rank_order(self, ctx):
        """the comparators are the K closest taxa genome-wide among those sharing enough sites, closest first; a
        closer taxon may be missing from the list only because it shares too few sites at this locus"""
        R, X, loci, m = ctx["R"], ctx["X"], ctx["loci"], ctx["m"]; fs = R.arr("cells/focal_sites.npy"); ident, cls = identical(X, m)
        for g, li in cells_tested(R)[::3]:
            comps = R.comps(g, li); mrow = m[g]; need = min(20, int(fs[g, li]))
            ms = [mrow[c] for c in comps]; assert all(np.isfinite(ms)) and ms == sorted(ms)
            assert not (ident[g] & set(comps)) and len({cls[c] for c in comps}) == len(comps), (g, li)          # never an identical genome; one comparator per class of copies
            if len(comps) == R.K:
                worst = ms[-1]; classes = {cls[c] for c in comps}
                for c in np.flatnonzero(np.isfinite(mrow) & (mrow < worst)):
                    if c not in comps and int(c) not in ident[g] and cls[c] not in classes: assert shared(X, loci[li][1], g, int(c))[0] < need, (g, li, c)

    def test_taxa_closest_are_the_five_nearest_by_m(self, ctx):
        R, m, X = ctx["R"], ctx["m"], ctx["X"]; cl = R.arr("taxa/closest.npy"); ident, _ = identical(X, m)
        for g in range(R.n):
            row = np.where(np.isfinite(m[g]), m[g], np.inf); row[list(ident[g])] = np.inf; want = [int(c) for c in np.argsort(row, kind="stable")[:5] if np.isfinite(row[c])]
            got = [int(c) for c in cl[g] if c >= 0]
            assert got == want or sorted(row[got]) == sorted(row[want]), (g, got, want)


# ------------------------------------------------------------------ test 1
class TestDivergent:
    def test_n_fail_counts_the_comparators_past_the_bar(self, ctx):
        R, bar = ctx["R"], ctx["bar"]; cd = R.arr("cells/comp_div_log10p.npy"); comp = R.arr("cells/comparators.npy")
        for g, li in cells_tested(R):
            slots = cd[g, li][comp[g, li] >= 0]; assert np.isfinite(slots).all() and (slots <= 0).all()
            assert int((slots < bar).sum()) == R.cell("n_fail", g, li)

    def test_masked_divergent_is_a_majority_of_the_comparators(self, ctx):
        R = ctx["R"]; side = R.arr("cells/side.npy"); nf = R.arr("cells/n_fail.npy"); nc = R.arr("cells/n_comp.npy")
        s2 = R.arr("cells/side2.npy") if R.has("cells/side2.npy") else np.zeros_like(side)
        for g, li in cells_tested(R):
            sd = int(side[g, li]); k, f = int(nc[g, li]), int(nf[g, li])
            full = k == R.K and f >= R.M; amb = k < R.K and f >= k // 2 + 1
            broad_red = bool(int(s2[g, li]) & 8)                                  # the broad-sample excess arm also masks divergent
            assert bool(sd & 1) == (full or amb or broad_red), (g, li, sd, k, f)
            assert bool(sd & 4) == amb
            assert not (sd & 16)                                                  # masking of divergent sequences was on

    def test_div_log10p_is_the_decisive_comparison(self, ctx):
        R = ctx["R"]; cd = R.arr("cells/comp_div_log10p.npy"); comp = R.arr("cells/comparators.npy"); nc = R.arr("cells/n_comp.npy")
        dl = R.arr("cells/div_log10p.npy"); nf = R.arr("cells/n_fail.npy"); side = R.arr("cells/side.npy")
        for g, li in cells_tested(R):
            v = dl[g, li]
            if nf[g, li] == 0 and not (side[g, li] & (2 | 32 | 128 | 64)): continue       # carried only for cells with a failing test
            if np.isnan(v): continue
            k = int(nc[g, li]); need = R.M if k == R.K else k // 2 + 1
            slots = np.sort(cd[g, li][comp[g, li] >= 0]); assert np.isclose(v, slots[need - 1], atol=1e-6), (g, li)

    def test_excess_and_medians_are_consistent(self, ctx):
        R = ctx["R"]; mi = R.arr("cells/med_identity.npy"); me = R.arr("cells/med_expected.npy"); ex = R.arr("cells/excess_m.npy"); side = R.arr("cells/side.npy"); s2 = R.arr("cells/side2.npy")
        for g, li in cells_tested(R):
            assert 0 <= mi[g, li] <= 100 and 0 <= me[g, li] <= 100 and np.isfinite(ex[g, li])
            if side[g, li] & 1 and not (side[g, li] & 4) and not (s2[g, li] & 8): assert mi[g, li] < me[g, li] and ex[g, li] > 0   # a broad-sample red mask may agree with its relatives


# ------------------------------------------------------------------ test 2
class TestSimilar:
    def pairs(self, R):
        out = {}
        for li in range(R.nloc):
            for e in R.similar(li):
                if e["fail"]: out[(li, min(e["taxon"], e["partner"]), max(e["taxon"], e["partner"]))] = e
        return out

    def test_a_masked_copy_names_a_failing_pair(self, ctx):
        R = ctx["R"]; side = R.arr("cells/side.npy"); pt = R.arr("cells/partner.npy"); pairs = self.pairs(R)
        s2 = R.arr("cells/side2.npy") if R.has("cells/side2.npy") else np.zeros_like(side)
        for g, li in cells_tested(R):
            sd = int(side[g, li])
            if (sd & (2 | 128)) and not (int(s2[g, li]) & (4 | 8)):             # a broad-sample mask, either side, names a comparator (broadsample/<locus>.json), not a pair
                p = int(pt[g, li]); assert p >= 0 and (li, min(g, p), max(g, p)) in pairs, (g, li)
            assert not (sd & 32)

    def test_roles_are_consistent(self, ctx):
        R = ctx["R"]; side = R.arr("cells/side.npy"); pt = R.arr("cells/partner.npy"); pairs = self.pairs(R); ns = R.arr("cells/n_similar.npy")
        for (li, a, b), e in pairs.items():
            if ns[a, li] > 1 or ns[b, li] > 1: continue                       # a cell in several pairs carries the role of its strongest pair (notes/KNOWN_ISSUES.md)
            sa, sb = int(side[a, li]), int(side[b, li])
            if e["uncertain"]:
                assert e["copy"] == -1 and (sa & 2) and (sb & 2), (li, a, b)
            else:
                cp = e["copy"]; src = b if cp == a else a; assert cp in (a, b)
                assert side[cp, li] & 2 and not (side[cp, li] & 8), (li, a, b)
                assert (side[src, li] & 128) or (side[src, li] & 2), (li, a, b)         # the source is a source, unless it is the copy of another pair
            if sa & 8: assert sa & 2
            if sb & 8: assert sb & 2

    def test_the_copy_fits_its_relatives_worse(self, ctx):
        """the direction: the copy is the member with the larger test-1 excess, by more than the donor threshold"""
        R = ctx["R"]; thr = R.man["options"]["donor_threshold"]; ex = R.arr("cells/excess_m.npy"); ns = R.arr("cells/n_similar.npy")
        for (li, a, b), e in self.pairs(R).items():
            if ns[a, li] > 1 or ns[b, li] > 1: continue                       # the record's copy field follows the cell's strongest pair (notes/KNOWN_ISSUES.md)
            xa, xb = abs(float(ex[a, li])), abs(float(ex[b, li]))
            if not (np.isfinite(xa) and np.isfinite(xb)): continue
            if e["uncertain"]: assert max(xa, xb) <= thr * min(xa, xb) + 1e-9
            else:
                cp = e["copy"]; assert (xa if cp == a else xb) >= (xb if cp == a else xa) and max(xa, xb) > thr * min(xa, xb) - 1e-9

    def test_n_similar_counts_the_failing_pairs_of_the_cell(self, ctx):
        R = ctx["R"]; ns = R.arr("cells/n_similar.npy"); pairs = self.pairs(R)
        want = {}
        for (li, a, b) in pairs: want[(a, li)] = want.get((a, li), 0) + 1; want[(b, li)] = want.get((b, li), 0) + 1
        for g, li in cells_tested(R): assert ns[g, li] == want.get((g, li), 0), (g, li)

    def test_sim_log10p_is_the_pairs_p_or_the_closest_test(self, ctx):
        R = ctx["R"]; sp = R.arr("cells/sim_log10p.npy"); side = R.arr("cells/side.npy"); lcp = R.arr("cells/locus_closest_log10p.npy"); pairs = self.pairs(R)
        for g, li in cells_tested(R):
            mine = [e for (l, a, b), e in pairs.items() if l == li and g in (a, b)]
            if mine: assert np.isclose(sp[g, li], min(e["log10p"] for e in mine), atol=2e-3), (g, li)
            elif side[g, li] & 64: assert np.isfinite(sp[g, li]) and np.isclose(sp[g, li], lcp[g, li], atol=1e-5)
            else: assert np.isnan(sp[g, li])

    def test_failing_pairs_are_below_the_bar_and_reported_once(self, ctx):
        R, bar = ctx["R"], ctx["bar"]
        for (li, a, b), e in self.pairs(R).items():
            assert e["log10p"] < bar + 1e-3 and e["test"] in ("relative", "outside")
        for li in range(R.nloc):
            keys = [(min(e["taxon"], e["partner"]), max(e["taxon"], e["partner"])) for e in R.similar(li) if e["fail"]]
            assert len(keys) == len(set(keys))


# ------------------------------------------------------------------ the closest match and the best match
class TestClosest:
    def test_locus_closest_is_the_fewest_mismatches_per_shared_site(self, ctx):
        """among the taxa in the test sharing at least fit-overlap sites: fewest mismatches per shared site, half a mismatch added, ties by m"""
        R, X, loci, m = ctx["R"], ctx["X"], ctx["loci"], ctx["m"]; lc = R.arr("cells/locus_closest.npy"); ident, _ = identical(X, m)
        excluded = np.array([s != "tested" for s in R.taxa["status"]]); cells = cells_tested(R); step = 2 if len(cells) < 2000 else len(cells) // 400
        for g, li in cells[::step]:
            y = int(lc[g, li]); Xs = X[:, loci[li][1]]; both = (Xs >= 0) & (Xs[g] >= 0); sh = both.sum(1); mis = ((Xs != Xs[g]) & both).sum(1)
            ok = (sh >= 20) & np.isfinite(m[g]) & ~excluded; ok[g] = False; ok[list(ident[g])] = False
            if not ok.any(): assert y == -1, (g, li); continue
            d = np.where(ok, (mis + 0.5) / np.maximum(sh, 1), np.inf); keys = sorted((d[c], m[g, c], c) for c in np.flatnonzero(ok & (d <= d.min() + 1e-9)))
            assert y == keys[0][2] or np.isclose(d[y], keys[0][0]), (g, li, y, keys[0])

    def test_closest_test_bit_means_the_closest_is_outside_the_comparators(self, ctx):
        R = ctx["R"]; side = R.arr("cells/side.npy"); lc = R.arr("cells/locus_closest.npy")
        for g, li in cells_tested(R):
            if side[g, li] & 64: assert int(lc[g, li]) not in R.comps(g, li) and int(lc[g, li]) >= 0

    def test_locus_closest_p_is_carried_and_sane(self, ctx):
        R = ctx["R"]; lc = R.arr("cells/locus_closest.npy"); lcp = R.arr("cells/locus_closest_log10p.npy"); cs = R.arr("cells/comp_sim_log10p.npy"); comp = R.arr("cells/comparators.npy")
        for g, li in cells_tested(R):
            y = int(lc[g, li])
            if y < 0: assert np.isnan(lcp[g, li]); continue
            if y in R.comps(g, li):
                k = R.comps(g, li).index(y); assert np.isclose(lcp[g, li], cs[g, li, k], equal_nan=True)
            if np.isfinite(lcp[g, li]): assert lcp[g, li] <= 0

    def test_best_match_identity_matches_the_alignment(self, ctx):
        R, X, loci = ctx["R"], ctx["X"], ctx["loci"]; bm = R.arr("cells/best_match.npy"); bi = R.arr("cells/best_match_identity.npy")
        for g, li in cells_tested(R):
            b = int(bm[g, li])
            if b < 0: assert np.isnan(bi[g, li]); continue
            sh, mis = shared(X, loci[li][1], g, b); assert sh >= lo.BEST_MATCH_MIN_SITES and np.isclose(bi[g, li], 100 * (1 - mis / sh), atol=1e-3)

    def test_best_match_is_not_beaten_on_the_same_sites(self, ctx):
        """no taxon covering as many of the focal's sites has fewer mismatches per shared site"""
        R, X, loci = ctx["R"], ctx["X"], ctx["loci"]; bm = R.arr("cells/best_match.npy"); excluded = {i for i, s in enumerate(R.taxa["status"]) if s != "tested"}
        for g, li in cells_tested(R)[::3]:
            b = int(bm[g, li])
            if b < 0: continue
            sites = loci[li][1]; shb, misb = shared(X, sites, g, b); wmax = max(shared(X, sites, g, c)[0] for c in range(R.n) if c != g and c not in excluded)
            for c in range(R.n):
                if c in (g, b) or c in excluded: continue
                sh, mis = shared(X, sites, g, c)
                if sh == wmax and shb == wmax: assert mis / sh >= misb / shb - 1e-12, (g, li, b, c)


# ------------------------------------------------------------------ the Bonferroni family and the manifest's counts
class TestFamily:
    def test_n_is_two_per_distinct_comparator_pair_plus_the_closest_tests(self, ctx):
        R = ctx["R"]; side = R.arr("cells/side.npy"); t = R.man["test"]; D = 0
        for li in range(R.nloc):
            prs = set()
            for g, _ in [(g, l) for g, l in cells_tested(R) if l == li]:
                for c in R.comps(g, li): prs.add((min(g, c), max(g, c)))
            D += len(prs)
        S = int(((side & 64) > 0).sum())
        assert t["close_relative_tests"] == D and t["closest_match_relative_tests"] == D and t["closest_match_outside_tests"] == S
        assert t["tests"] == 2 * D + S + t["broad_sample_tests"]                  # two broad-sample tests (one per side) per present sequence of a kept taxon
        pres = (ctx["R"].arr("cells/focal_sites.npy") > 0)
        assert t["broad_sample_tests"] in (0, 2 * (int(pres.sum()) - int(pres[[i for i, st in enumerate(ctx["R"].taxa["status"]) if st != "tested"]].sum())))
        assert np.isclose(t["log10_bar"], math.log10(R.man["options"]["alpha"]), atol=1e-4)          # every p is Bonferroni-adjusted (p x N), so the bar is alpha
        assert t["log_bar"] == pytest.approx(math.log(R.man["options"]["alpha"]), abs=1e-4) and t["p_values"].startswith("adjusted by Bonferroni")

    def test_counts_match_the_side_bits(self, ctx):
        R = ctx["R"]; side = R.arr("cells/side.npy"); nc = R.arr("cells/n_comp.npy"); c = R.man["counts"]
        assert c["cells"] == int((nc > 0).sum()) and c["masked"] == int(((side & 3) > 0).sum()) and c["masked_divergent"] == int((side & 1 > 0).sum())
        assert c["masked_similar"] == int((side & 2 > 0).sum()) and c["masked_both"] == int(((side & 3) == 3).sum()) and c["ambiguous"] == int((side & 4 > 0).sum())
        assert c["similar_sources"] == int((side & 128 > 0).sum()) and c["similar_uncertain"] == int((side & 8 > 0).sum()) and c["similar_tested"] == int((side & 64 > 0).sum())
        assert c["taxa"] == R.n and c["loci"] == R.nloc and c["excluded_taxa"] == sum(s != "tested" for s in R.taxa["status"]) and c["skipped_loci"] == sum(s != "tested" for s in R.loci["status"])

    def test_near_miss_counts(self, ctx):
        R = ctx["R"]; side = R.arr("cells/side.npy"); nf = R.arr("cells/n_fail.npy"); nc = R.arr("cells/n_comp.npy")
        for j in range(1, R.M):
            assert R.man["counts"]["near_miss"][str(j)] == int(((nf == j) & (nc > 0) & ((side & 1) == 0)).sum())


# ------------------------------------------------------------------ masked.csv and the masked alignment
class TestOutputs:
    def test_masked_csv_rows_are_the_masked_cells(self, ctx):
        R, masked = ctx["R"], ctx["masked"]; side = R.arr("cells/side.npy"); names = R.taxa["names"]; lnames = R.loci["names"]
        want = {(names[g], lnames[li]) for g, li in zip(*np.nonzero((side & 3) > 0))}
        assert {(r["taxon"], r["locus"]) for r in masked} == want and len(masked) == len(want)
        assert list(masked[0].keys()) == lo.MASKED_COLUMNS if masked else True

    def test_masked_csv_reasons_and_evidence(self, ctx):
        R, masked = ctx["R"], ctx["masked"]; side = R.arr("cells/side.npy"); fs = R.arr("cells/focal_sites.npy"); nf = R.arr("cells/n_fail.npy"); nc = R.arr("cells/n_comp.npy")
        for r in masked:
            g, li = R.tx(r["taxon"]), R.lx(r["locus"]); sd = int(side[g, li])
            assert r["reason"] == ("both" if (sd & 3) == 3 else "divergent" if sd & 1 else "similar")
            assert r["mask_test"] in ("close_relative", "closest_match", "broad_sample")      # the one significance: bar-held, so
            assert float(r["mask_p"]) <= 10 ** R.man["test"]["log10_bar"] + 1e-6              # every masked sequence earns at least one star (six-decimal rounding)
            assert int(r["residues_masked"]) == fs[g, li]
            if r["close_relative_fails"] == "":                                  # masked by the broad-sample test alone: its own columns carry the evidence
                assert r["broad_sample_side"] != "" and float(r["broad_sample_close_p" if r["broad_sample_side"] == "close" else "broad_sample_divergent_p"]) <= 0.05 + 1e-6; continue
            assert int(r["close_relative_fails"]) == nf[g, li] and int(r["close_relative_comparators"]) == nc[g, li]
            assert r["comparators"] == "; ".join(R.taxa["names"][c] for c in R.comps(g, li))
            if sd & 2:
                role = r["closest_match_role"]; partner = r["closest_match_partner"]
                assert (role in ("copy", "uncertain") or int(R.cell("n_similar", g, li)) > 1) and partner == R.taxa["names"][int(R.cell("partner", g, li))]   # a cell masked as the copy of one pair may be the source of a stronger one
                assert (role == "uncertain") == bool(sd & 8)
            else: assert r["closest_match_role"] in ("", "source")

    def test_masked_alignment_blanks_exactly_the_masked_stretches(self, ctx):
        R, run, X, loci, ds = ctx["R"], ctx["run"], ctx["X"], ctx["loci"], ctx["ds"]; side = R.arr("cells/side.npy")
        names, seqs = lo.read_fasta(open(run.file("alignment.masked.fasta")).read()); assert names == R.taxa["names"]
        orig = lo.read_fasta(open(ds.aln).read())[1]
        for g in range(R.n):
            s = np.frombuffer(seqs[g].encode(), dtype=np.uint8).copy(); o = np.frombuffer(orig[g].encode(), dtype=np.uint8)
            for li in range(R.nloc):
                sites = loci[li][1]
                if side[g, li] & 3: assert (s[sites] == ord("-")).all(), (g, li)
                else: assert np.array_equal(s[sites], o[sites]), (g, li)

    def test_log_names_the_counts(self, ctx):
        run, R = ctx["run"], ctx["R"]; c = R.man["counts"]
        assert f"masked {c['masked']} sequences" in run.log and f"{c['masked_divergent'] - c['masked_both']} too divergent" in run.log
        assert re.search(r"tests: [\d,]+ close-relative comparisons", run.log) and "done" in run.log


# ------------------------------------------------------------------ what the report carries for the viewer
class TestCarried:
    def test_codes_for_every_cell_with_a_failing_test(self, ctx):
        R, X, loci = ctx["R"], ctx["X"], ctx["loci"]; side = R.arr("cells/side.npy"); nf = R.arr("cells/n_fail.npy"); pt = R.arr("cells/partner.npy")
        for li in range(R.nloc):
            fail = [g for g, l in cells_tested(R) if l == li and (nf[g, l] > 0 or side[g, l] & (2 | 128))]
            if not R.has(f"codes/{li}.npy"): assert not fail, li; continue
            cells = R.arr(f"codes/{li}_cells.npy").tolist(); blk = R.arr(f"codes/{li}.npy")
            assert set(cells) >= set(fail) and blk.shape == (len(cells), len(loci[li][1])) and blk.dtype == np.uint8
            for r, g in enumerate(cells[:6]):
                row = blk[r]; sites = loci[li][1]; fx = X[g, sites]
                assert np.array_equal(row == 255, fx < 0)
                B = int(pt[g, li]) if side[g, li] & (2 | 128) else -1
                comps = [c for c in R.comps(g, li) if c != B]
                k = ((X[np.ix_(comps, sites)] != fx[None, :]) & (X[np.ix_(comps, sites)] >= 0)).sum(0) if comps else np.zeros(len(sites), int)
                plain = (row < 128) & (fx >= 0); assert np.array_equal(row[plain], k[plain])
                if B >= 0:
                    blue = (row >= 128) & (row != 255); agree = (X[B, sites] == fx) & (X[B, sites] >= 0) & (fx >= 0) & (k >= 2)
                    assert np.array_equal(blue, agree) and np.array_equal(row[blue].astype(int) - 128, np.minimum(k[blue], 125))
                else: assert not ((row >= 128) & (row != 255)).any()

    def test_sequences_are_the_input_rows_for_every_present_taxon(self, ctx):
        R, X, loci = ctx["R"], ctx["X"], ctx["loci"]; fs = R.arr("cells/focal_sites.npy")
        assert R.man["carried"]["sequences"] == "all"
        ds = ctx["ds"]; orig = lo.read_fasta(open(ds.aln).read())[1]
        excluded = {i for i, s in enumerate(R.taxa["status"]) if s != "tested"}
        for li in range(R.nloc):
            if R.loci["status"][li] != "tested": assert not R.has(f"seq/{li}.npy"); continue
            assert R.has(f"seq/{li}.npy"); taxa = R.arr(f"seq/{li}_taxa.npy").tolist(); blk = R.arr(f"seq/{li}.npy")
            assert taxa == [g for g in range(R.n) if fs[g, li] > 0 and g not in excluded]
            for r, g in enumerate(taxa): assert bytes(blk[r]).decode() == "".join(orig[g][s] for s in loci[li][1])

    def test_trees_for_every_masked_cell(self, ctx):
        """one tree pair per masked cell; its tips are every row of the alignment view - the focal, its relatives, then
        its partner, its closest match at the locus and its broad-sample representative (whichever exist, no repeats) -
        and the outgroup last"""
        R = ctx["R"]; side = R.arr("cells/side.npy"); pt = R.arr("cells/partner.npy"); lc = R.arr("cells/locus_closest.npy"); rp = R.arr("cells/bs_rep.npy")
        bsp = {}
        if R.has("cells/bs_partner.npy"):
            for l_, g_, p_ in zip(R.arr("cells/bs_locus.npy"), R.arr("cells/bs_taxon.npy"), R.arr("cells/bs_partner.npy")): bsp[(int(g_), int(l_))] = int(p_)
        for li in range(R.nloc):
            masked = [g for g, l in cells_tested(R) if l == li and side[g, l] & 3]
            if not masked: assert not R.has(f"trees/{li}.json"); continue
            T = R.js(f"trees/{li}.json"); assert T["cells"] == masked
            for k, g in enumerate(T["cells"]):
                tips = T["tips"][k]; comps = R.comps(g, li); og = T["outgroup"][k]
                assert tips[:1 + len(comps)] == [g] + comps and len(set(tips)) == len(tips)
                broad = bsp[(g, li)] if bsp.get((g, li), -1) >= 0 else int(rp[g, li])   # the broad row: the verdict pass's representative, else stage 1's
                required = {int(pt[g, li]), int(lc[g, li]), broad} - {-1}; allowed = required | ({int(rp[g, li]), bsp.get((g, li), -1)} - {-1})
                extras = tips[1 + len(comps):(og if og >= 0 else len(tips))]
                assert set(extras) <= allowed and (og == -1 or (og == len(tips) - 1 and tips[og] not in [g] + comps))
                for t in required:                                                   # every row of the alignment view is a tip
                    assert t in tips, (g, li, t)
                for s in (T["expected"][k], T["observed"][k]):
                    assert s.endswith(";") and s.count("(") == s.count(")") and all(str(t) in re.findall(r"(\d+):", s) for t in tips)
                assert T["closest"][k][0] in range(-1, len(tips)) and T["closest"][k][1] in range(-1, len(tips))

    def test_taxa_and_loci_tables(self, ctx):
        R, loci = ctx["R"], ctx["loci"]
        assert R.taxa["names"] == ctx["names"] and R.loci["names"] == [k for k, _ in loci] and R.loci["length"] == [len(s) for _, s in loci]
        for rng, (_, sites) in zip(R.loci["ranges"], loci): assert [s for a, b, st in rng for s in range(a, b + 1, st)] == sites.tolist()
        assert len(R.loci["rate"]) == R.nloc and all(v is None or v > 0 for v in R.loci["rate"]) and all(1 <= p <= 100 for p in R.loci["plateau"])
        assert R.taxa["loci_present"] == (R.arr("cells/focal_sites.npy") > 0).sum(1).tolist()
