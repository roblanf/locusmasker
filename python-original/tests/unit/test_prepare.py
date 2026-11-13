"""prepare(): the residue codes, plateaus, the exclusion of thin taxa, the on-demand pair counts, the pair divergences m
and the locus rates, each taxon's candidate list, the closest match at every cell and the classes of identical genomes,
checked against brute force on the fixture alignments; and divergence_matrix(), the entry point scripts use."""
import argparse, time
import numpy as np
import pytest
from tests.conftest import lo, dataset


def encode(rows, dtype):
    alphabet = lo.DNA if dtype == "dna" else lo.PROTEIN; lut = np.full(256, -1, np.int8)
    for i, c in enumerate(alphabet): lut[ord(c)] = i; lut[ord(c.lower())] = i
    if dtype == "dna": lut[ord("U")] = lut[ord("u")] = 3
    return np.stack([lut[np.frombuffer(r, dtype=np.uint8)] for r in rows])


def run_prepare(ds, threads=1, **opts):
    names, rows, loci, *_ = lo.read_input(ds.aln, ds.parts)
    a = argparse.Namespace(fit_overlap=20, threads=threads, candidates=opts.get("candidates"), block_taxa=opts.get("block_taxa"))
    return names, rows, loci, lo.prepare(a, names, rows, loci, ds.dtype, 5, time.time(), full=opts.get("full"))


@pytest.fixture(scope="module")
def prepared():
    """prepare() on dna_small with one thread, plus everything needed to check it"""
    ds = dataset("dna_small"); names, rows, loci, P = run_prepare(ds)
    return dict(ds=ds, names=names, rows=rows, loci=loci, P=P, X=encode(rows, "dna"))


def brute_counts(X, sites, i, j):
    Xl = X[:, sites]; both = (Xl[i] >= 0) & (Xl[j] >= 0); return int(both.sum()), int(((Xl[i] == Xl[j]) & both).sum())


def brute_mj(X, loci, P):
    """the ranking key of the old engine from the residues: m with the tie-break by pooled identity and index, inf for
    identical, excluded or unresolved pairs"""
    n = X.shape[0]; tot_same = np.zeros((n, n)); tot_shared = np.zeros((n, n))
    for _, sites in loci:
        Xl = X[:, sites]; pres = Xl >= 0
        for i in range(n):
            both = pres[i] & pres; tot_shared[i] += both.sum(1); tot_same[i] += ((Xl == Xl[i]) & both).sum(1)
    keep = P.keep; tot_shared[~keep] = 0; tot_shared[:, ~keep] = 0; tot_same[~keep] = 0; tot_same[:, ~keep] = 0
    with np.errstate(invalid="ignore", divide="ignore"): ani = np.where(tot_shared > 0, 100.0 * tot_same / tot_shared, np.nan)
    np.fill_diagonal(ani, np.nan); ident = (tot_shared > 0) & (tot_same == tot_shared); np.fill_diagonal(ident, False)
    tie = np.nan_to_num(100.0 - ani, nan=100.0) * 1e-9 + np.arange(n)[None, :] * 1e-12
    return np.where(np.isfinite(P.m_full) & ~ident, P.m_full, np.inf) + tie, ident


class TestCounts:
    def test_residue_codes(self, prepared):
        assert np.array_equal(prepared["P"].X, prepared["X"])

    def test_pair_counts_on_demand_are_exact(self, prepared):
        X, loci, P = prepared["X"], prepared["loci"], prepared["P"]; n = X.shape[0]; thin = set(P.thin.tolist())
        for li in (0, 3, len(loci) - 1):
            sites = loci[li][1]; gi = np.repeat(np.arange(n), n); ci = np.tile(np.arange(n), n)
            o, same = lo.pair_counts_at(X, sites, gi, ci, P.keep)
            for k in range(0, n * n, 7):
                i, j = int(gi[k]), int(ci[k]); w, s = brute_counts(X, sites, i, j)
                if i in thin or j in thin: assert o[k] == 0 and same[k] == 0; continue
                assert o[k] == w and same[k] == s

    def test_pair_rows_on_demand_match_the_pair_counts(self, prepared):
        X, loci, P = prepared["X"], prepared["loci"], prepared["P"]; n = X.shape[0]; sites = loci[2][1]; rows = np.array([0, 5, 17, n - 1])
        W, S = lo.pair_rows_at(X, sites, rows, P.keep, 4, slice_rows=3)
        o, same = lo.pair_counts_at(X, sites, np.repeat(rows, n), np.tile(np.arange(n), len(rows)), P.keep)
        assert np.array_equal(W.ravel(), o) and np.array_equal(S.ravel(), same)

    def test_focal_sites_counts_residues(self, prepared):
        X, loci, P = prepared["X"], prepared["loci"], prepared["P"]
        want = np.array([[(X[t, s] >= 0).sum() for _, s in loci] for t in range(X.shape[0])])
        assert np.array_equal(P.focal_sites, want)

    def test_plateaus_from_residue_frequencies(self, prepared):
        X, loci, P = prepared["X"], prepared["loci"], prepared["P"]
        for li in range(len(loci)):
            v = X[:, loci[li][1]]; v = v[v >= 0]; f = np.bincount(v, minlength=4) / len(v)
            assert np.isclose(P.plateaus[li], max(100 * (1 - (f ** 2).sum()), 1.0))


class TestExclusion:
    def test_thin_taxa_are_those_with_too_few_covered_loci(self, prepared):
        X, loci, P, ds = prepared["X"], prepared["loci"], prepared["P"], prepared["ds"]
        cov = np.array([[(X[t, s] >= 0).sum() >= 20 for _, s in loci] for t in range(X.shape[0])]).sum(1)
        assert set(P.thin.tolist()) == set(np.flatnonzero(cov < lo.MIN_LOCI_PER_TAXON).tolist())
        assert [prepared["names"][t] for t in P.thin] == [e["taxon"] for e in ds.ev("excluded")]
        assert np.array_equal(P.present, cov) and np.array_equal(P.keep, cov >= lo.MIN_LOCI_PER_TAXON)

    def test_thin_taxa_have_no_divergence_to_anyone_and_no_candidates(self, prepared):
        P = prepared["P"]
        for t in P.thin: assert np.isnan(P.m_full[t]).all() and np.isnan(P.m_full[:, t]).all() and (P.lists["cand"][t] == -1).all()
        assert not np.isin(P.thin, P.lists["cand"]).any()


class TestDivergence:
    def test_m_is_symmetric_with_nan_diagonal(self, prepared):
        m = prepared["P"].m_full; n = m.shape[0]
        assert np.isnan(np.diag(m)).all()
        off = ~np.eye(n, dtype=bool); assert np.array_equal(np.isnan(m), np.isnan(m.T)) and np.allclose(m[off], m.T[off], equal_nan=True)

    def test_m_is_a_percentage_and_finite_where_pairs_overlap(self, prepared):
        P = prepared["P"]; m = P.m_full; sub = m[np.ix_(P.keep, P.keep)]; off = ~np.eye(P.keep.sum(), dtype=bool)
        assert np.isfinite(sub[off]).all() and (sub[off] >= 0).all() and (sub[off] < 100).all()

    def test_floor_never_lowers_a_divergence(self, prepared):
        P = prepared["P"]; ok = np.isfinite(P.m_full); m_hi = np.maximum(P.m_full, P.mfl_full)
        assert np.all(m_hi[ok] >= P.m_full[ok]) and np.all(np.maximum(P.m_full - P.mfl_full, 0.0)[ok] >= 0)

    def test_identical_taxa_have_zero_divergence(self):
        ds = dataset("dna_small"); names, rows, loci, *_ = lo.read_input(ds.aln, ds.parts)
        rows = [bytearray(r) for r in rows]; rows[1][:] = rows[0]                     # taxon 1 becomes a copy of taxon 0 everywhere
        P = lo.prepare(argparse.Namespace(fit_overlap=20, threads=1), names, rows, loci, "dna", 5, time.time())
        assert P.m_full[0, 1] == 0.0 and P.mfl_full[0, 1] > 0 and P.cls[1] == 0 and P.has_dup and 1 not in P.lists["cand"][0]

    def test_m_orders_taxa_like_the_simulated_tree(self, prepared):
        """a taxon's closest relative by m is its closest by patristic distance, for most taxa"""
        P, ds, names = prepared["P"], prepared["ds"], prepared["names"]; pat = ds.truth["patristic"]; hits = 0; tot = 0
        for t, nm in enumerate(names):
            if t in P.thin: continue
            row = np.where(np.isfinite(P.m_full[t]), P.m_full[t], np.inf); nearest_m = int(np.argmin(row))
            pr = np.array(pat[nm], float); pr[t] = np.inf; pr[P.thin] = np.inf; nearest_pat = int(np.argmin(pr))
            tot += 1; hits += nearest_m == nearest_pat
        assert hits >= 0.8 * tot, (hits, tot)

    def test_rates_median_one(self, prepared):
        P, loci = prepared["P"], prepared["loci"]; ok = np.isfinite(P.rate) & (P.rate > 0); ls = np.array([len(s) for _, s in loci], float)
        assert ok.sum() >= 3 and np.isclose(lo.wmedian_stack(P.rate[ok, None], ls[ok, None])[0], 1.0)

    def test_rates_are_positive_where_defined_and_vary_by_locus(self, prepared):
        r = prepared["P"].rate; ok = np.isfinite(r)
        assert (r[ok] > 0).all() and r[ok].std() > 0

    def test_nothing_depends_on_threads_or_the_block_size(self):
        ds = dataset("dna_small"); base = run_prepare(ds)[3]
        for threads, block, cand in ((4, None, None), (1, 7, None), (3, 3, None), (2, 11, 6)):
            P = run_prepare(ds, threads=threads, block_taxa=block, candidates=cand)[3]
            for k in ("m_full", "mfl_full", "plateaus", "rate", "cls", "focal_sites"): assert np.array_equal(getattr(base, k), getattr(P, k), equal_nan=True), (k, threads, block)
            C = P.lists["cand"].shape[1]
            for k in ("cand", "key", "m", "mfl"): assert np.array_equal(base.lists[k][:, :C], P.lists[k], equal_nan=True), (k, threads, block, cand)
            for x, y in zip(base.closest, P.closest): assert np.array_equal(x, y, equal_nan=True), (threads, block)


class TestLists:
    def test_lists_are_the_prefix_of_the_ranking_order(self, prepared):
        """each taxon's candidates are the first C taxa by the old engine's key (m with the pooled-identity and index
        tie-break), identical and excluded genomes left out, with m, its floor and the key beside them"""
        X, loci, P = prepared["X"], prepared["loci"], prepared["P"]; mj, ident = brute_mj(X, loci, P); n = X.shape[0]; C = P.lists["cand"].shape[1]
        for g in range(n):
            order = np.argsort(mj[g], kind="stable"); fin = order[np.isfinite(mj[g][order])][:C]
            cand = P.lists["cand"][g]; cand = cand[cand >= 0]
            assert np.array_equal(cand, fin), g
            assert np.array_equal(P.lists["key"][g][:len(fin)], mj[g][fin]) and np.array_equal(P.lists["m"][g][:len(fin)], P.m_full[g][fin])
            assert P.lists["nfin"][g] == np.isfinite(mj[g]).sum()

    def test_a_short_list_knows_it_is_truncated(self):
        ds = dataset("dna_small"); P = run_prepare(ds, candidates=3)[3]
        assert P.lists["cand"].shape[1] == 3 and (P.lists["nfin"][P.keep] > 3).all() and (P.lists["cand"][P.keep] >= 0).all()

    def test_pair_m_reads_the_lists(self, prepared):
        P = prepared["P"]; g = np.flatnonzero(P.keep)[:4]; gi = np.repeat(g, 3); ci = P.lists["cand"][g, :3].ravel()
        m, f = lo.pair_m_lists(P, gi, ci)
        assert np.array_equal(m, P.m_full[gi, ci]) and np.array_equal(f, P.mfl_full[gi, ci])


class TestClosest:
    def test_the_closest_match_at_every_cell_is_the_old_rule(self, prepared):
        """fewest mismatches per shared site (half a mismatch added) among taxa sharing fit-overlap sites with a finite key;
        ties within 1e-9 by d + 1e-9 key / 100, then index"""
        X, loci, P = prepared["X"], prepared["loci"], prepared["P"]; mj, _ = brute_mj(X, loci, P); n = X.shape[0]
        win, dmin = lo.closest_winners(*P.closest)
        for li, (_, sites) in enumerate(loci):
            Xl = X[:, sites]
            for g in range(n):
                if not P.keep[g] or (Xl[g] >= 0).sum() == 0: continue
                best = None
                for c in range(n):
                    if c == g or not np.isfinite(mj[g, c]): continue
                    ov, same = brute_counts(X, sites, g, c)
                    if ov < 20: continue
                    best = min(best, ((ov - same + lo.HALF) / ov, c)) if best else ((ov - same + lo.HALF) / ov, c)
                if best is None: assert win[g, li] == -1; continue
                d0 = best[0]; band = []
                for c in range(n):
                    if c == g or not np.isfinite(mj[g, c]): continue
                    ov, same = brute_counts(X, sites, g, c)
                    if ov < 20: continue
                    d = (ov - same + lo.HALF) / ov
                    if d <= d0 + 1e-9: band.append((d + 1e-9 * mj[g, c] / 100.0, c))
                assert win[g, li] == min(band)[1] and dmin[g, li] == d0, (g, li)


class TestClosestKernel:
    """the compiled closest-match reduce gives the numpy rule's bits: random tables with ties, unusable pairs, rows
    without a candidate, and bands with more distinct rates than slots"""
    def kernel(self):
        if lo._pair_kernel(1) is None: pytest.skip("numba is not installed")
        return lo._closest_rows_nb

    def test_random_tables_with_ties(self):
        k = self.kernel(); rng = np.random.default_rng(7)
        for trial in range(40):
            nr, nc = rng.integers(1, 30), rng.integers(1, 60)
            W = rng.integers(0, 60, (nr, nc)).astype(np.float64); S = np.floor(W * rng.uniform(0.0, 1.0, (nr, nc)))
            eq = rng.random((nr, nc)) < 0.3; S[eq] = W[eq]                                                      # identical pairs: rate ties at zero
            mj = np.round(rng.uniform(0, 30, (nr, nc)), 2); mj[rng.random((nr, nc)) < 0.1] = np.nan; mj[rng.random((nr, nc)) < 0.2] = 0.0   # many equal keys
            cidx = rng.permutation(nc * 3)[:nc].astype(np.int64); fit = int(rng.integers(1, 25))
            a = lo._closest_rows(W, S, mj, fit, cidx); b = k(W, S, mj, fit, cidx, lo.BAND_SLOTS, lo.HALF)
            for x, y in zip(a[:4], b[:4]): assert np.array_equal(x, y, equal_nan=True), trial
            assert a[4] == b[4], trial

    def test_rows_without_candidates_and_bands_beyond_the_slots(self):
        k = self.kernel()
        W = np.full((3, 6), 4e9); S = W - np.array([[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]], float)   # rates 2.5e-10 apart: one band, six distinct rates
        mj = np.ones((3, 6)); mj[1] = np.nan; W[2, 3:] = 1.0                                                       # row 1 has no usable pair; row 2 three usable
        a = lo._closest_rows(W, S, mj, 20, np.arange(6)); b = k(W, S, mj, 20, np.arange(6), lo.BAND_SLOTS, lo.HALF)
        for x, y in zip(a[:4], b[:4]): assert np.array_equal(x, y, equal_nan=True)
        assert a[4] == b[4] == 1 and np.isinf(a[0][1]) and (a[3][1] == -1).all() and np.isfinite(a[1][0]).sum() == lo.BAND_SLOTS and np.isfinite(a[1][2]).sum() == 3


class TestClasses:
    def alignment(self, n=12, nloc=12, ll=100, seed=5):
        rng = np.random.default_rng(seed); anc = rng.integers(0, 4, nloc * ll); X = np.tile(anc, (n, 1)); mut = rng.random(X.shape) < 0.08
        X[mut] = (X[mut] + rng.integers(1, 4, int(mut.sum()))) % 4; return X, [(f"l{i}", np.arange(i * ll, (i + 1) * ll)) for i in range(nloc)]

    def rows_of(self, X): return [bytearray(bytes(np.frombuffer(b"ACGT-", dtype=np.uint8)[np.where(r >= 0, r, 4)])) for r in X]

    def test_a_non_transitive_chain_is_one_class_and_a_thin_copy_is_in_none(self):
        """A ~ B and B ~ C on their shared sites while A differs from C (B lacks the sites where C differs): the fixed point
        joins all three; a thin taxon identical to A on its few loci belongs to no class"""
        X, loci = self.alignment(); rng = np.random.default_rng(9); sites = rng.choice(X.shape[1], 20, replace=False)
        X[1] = X[0]; X[1, sites] = -1; X[2] = X[0]; X[2, sites] = (X[2, sites] + 1) % 4
        X[11] = X[0]
        for li in range(3, 12): X[11, loci[li][1]] = -1                                  # three loci: thin
        names = [f"t{i:02d}" for i in range(12)]
        P = lo.prepare(argparse.Namespace(fit_overlap=20, threads=1), names, self.rows_of(X), loci, "dna", 5, time.time())
        assert P.has_dup and P.cls[0] == P.cls[1] == P.cls[2] == 0 and 11 in P.thin and P.cls[11] == 11
        assert (P.cls[3:11] == np.arange(3, 11)).all()
        assert 1 not in P.lists["cand"][0] and 2 in P.lists["cand"][0] and 2 not in P.lists["cand"][1]   # identical pairs are never candidates; A and C are


class TestAlphabets:
    def test_dna_lowercase_and_u_are_residues_ambiguity_is_missing(self):
        rows = [bytearray(b"acgtU"), bytearray(b"ACGTT"), bytearray(b"NRY-?")]
        X = encode(rows, "dna")
        assert X[0].tolist() == [0, 1, 2, 3, 3] and X[1].tolist() == [0, 1, 2, 3, 3] and X[2].tolist() == [-1] * 5

    def test_protein_non_standard_letters_are_missing(self):
        X = encode([bytearray(b"ACDEFxBZ-*")], "aa")
        assert X[0].tolist()[:5] == [0, 1, 2, 3, 4] and X[0].tolist()[5:] == [-1] * 5

    def test_prepare_uses_the_same_encoding(self):
        ds = dataset("aa_small"); names, rows, loci, *_ = lo.read_input(ds.aln, ds.parts)
        P = lo.prepare(argparse.Namespace(fit_overlap=20, threads=1), names, rows, loci, "aa", 5, time.time())
        assert np.array_equal(P.X, encode(rows, "aa"))


class TestDivergenceMatrix:
    def test_returns_names_and_matrices(self):
        ds = dataset("dna_small"); names, m, m_hi = lo.divergence_matrix(ds.aln, ds.parts, "dna")
        assert names == lo.read_input(ds.aln, ds.parts)[0] and m.shape == (ds.ntaxa, ds.ntaxa) and np.allclose(m, m.T, equal_nan=True)
        assert np.all(m_hi[np.isfinite(m)] >= m[np.isfinite(m)]) and np.isnan(np.diag(m_hi)).all()

    def test_rejects_a_bad_datatype(self):
        ds = dataset("dna_small")
        with pytest.raises(ValueError): lo.divergence_matrix(ds.aln, ds.parts, "rna")

    def test_rejects_a_datatype_that_contradicts_the_nexus_file(self):
        ds = dataset("cannon_aa")
        with pytest.raises(ValueError): lo.divergence_matrix(ds.aln, None, "dna")

    def test_refuses_above_the_full_matrix_size(self, monkeypatch):
        monkeypatch.setattr(lo, "FULL_M_MAX_TAXA", 5); ds = dataset("dna_small")
        with pytest.raises(ValueError): lo.divergence_matrix(ds.aln, ds.parts, "dna")

    def test_same_as_prepare(self):
        ds = dataset("dna_small"); names, m, m_hi = lo.divergence_matrix(ds.aln, ds.parts, "dna", threads=2)
        P = run_prepare(ds)[3]; assert np.array_equal(m, P.m_full, equal_nan=True)


class TestOnDemandRows:
    def test_rows_of_m_on_demand_equal_the_matrix(self, prepared):
        """above FULL_M_MAX_TAXA the rows some consumers read come from the sweep on demand: the same numbers"""
        P = prepared["P"]; rows = np.array([0, 4, 9, 20]); full = P.m_full; P.m_full = None
        try: got = lo.m_rows(P, rows)
        finally: P.m_full = full
        assert np.array_equal(got, full[rows], equal_nan=True)

    @pytest.mark.parametrize("kernel", [True, False])
    def test_m_of_arbitrary_pairs_equals_the_matrix(self, prepared, kernel):
        """the pairs the best match's ties and the trees ask for, computed from the residues: the same bits, by the
        kernel and by the numpy chunks, self pairs NaN"""
        P = prepared["P"]; n = P.X.shape[0]; rng = np.random.default_rng(5)
        gi = np.r_[rng.integers(0, n, 300), 3]; ci = np.r_[rng.integers(0, n, 300), 3]
        full, ffl, args = P.m_full, P.mfl_full, P.sweep_args; P.m_full = None
        if kernel and lo._pair_kernel(1) is None: pytest.skip("numba is not installed")
        P.sweep_args = {**args, "kernel_serial": lo._pair_medians_serial if kernel else None}
        try: m, f = lo.pair_m(P, gi, ci, step=37)
        finally: P.m_full, P.sweep_args = full, args
        off = gi != ci
        assert np.array_equal(m[off], full[gi[off], ci[off]], equal_nan=True) and np.array_equal(f[off], ffl[gi[off], ci[off]], equal_nan=True)
        assert np.isnan(m[~off]).all() and np.isfinite(m[off]).any()


class TestCountsProviders:
    """the counts kept from a one-block sweep (SlabCounts) and the counts computed from the residues (ResidueCounts)
    give the same pair counts and rows"""
    def test_slab_and_residue_counts_agree(self, prepared):
        P = prepared["P"]; loci = prepared["loci"]; n = P.X.shape[0]; rng = np.random.default_rng(3)
        assert isinstance(P.counts, lo.SlabCounts) and "slab" in P.sweep_args
        res = lo.ResidueCounts(P.X, loci, P.keep, P.A_)
        for li in (0, len(loci) // 2, len(loci) - 1):
            gi = rng.integers(0, n, 40); ci = rng.integers(0, n, 40)
            a = P.counts.at(li, gi, ci); b = res.at(li, gi, ci)
            assert all(np.array_equal(x, y) for x, y in zip(a, b)) and a[0].dtype == np.int64 and b[0].dtype == np.int64
            rows = rng.choice(n, 5, replace=False); ra = P.counts.locus(li).rows(rows); rb = res.locus(li).rows(rows)
            assert all(np.array_equal(x, y) for x, y in zip(ra, rb)) and ra[0].dtype == np.float64
        assert all(len(x) == 0 for x in P.counts.at(0, [], []))

    def test_several_blocks_keep_no_slab(self):
        ds = dataset("dna_small"); P = run_prepare(ds, block_taxa=3)[3]
        assert isinstance(P.counts, lo.ResidueCounts) and "slab" not in P.sweep_args


class TestResultsTransfer:
    """a block pair's results survive the trip through an npz file, and the sweep gives the same structures whether
    its workers' results come back through files or through the pool's pipe"""
    def test_flatten_round_trip(self):
        out = dict(rows=(np.arange(3), (np.array([[1, 2]]), np.array([0.5]), np.array([[np.inf]]), np.zeros((1, 0)), np.array([2]))),
                   counters=(1, 2, 3), ident=np.zeros((0, 2), np.int64), overflow=0, full=(np.eye(2), np.ones((2, 2)) * np.nan), gidx=np.arange(2), cidx=np.arange(2, 4))
        flat = {}; lo._flatten(out, "", flat); assert all(isinstance(v, np.ndarray) for v in flat.values()) and "rows.1.3" in flat
        back = lo._unflatten(flat)
        assert set(back) == set(out) and back["counters"] == (1, 2, 3) and back["overflow"] == 0 and isinstance(back["overflow"], int)
        assert np.array_equal(back["rows"][0], out["rows"][0]) and back["rows"][1][3].shape == (1, 0) and back["ident"].shape == (0, 2) and back["ident"].dtype == np.int64
        assert np.isinf(back["rows"][1][2][0, 0]) and np.isnan(back["full"][1]).all() and np.array_equal(back["full"][0], np.eye(2))

    def test_results_dir_needs_the_filesystem(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lo, "SWEEP_RESULTS_DIR", str(tmp_path / "nowhere")); assert lo._results_dir(1000) is None
        monkeypatch.setattr(lo, "SWEEP_RESULTS_DIR", str(tmp_path)); d = lo._results_dir(1000); assert d and d.startswith(str(tmp_path)) and lo._results_dir(10 ** 18) is None

    def test_files_and_pipe_give_the_same_sweep(self, tmp_path, monkeypatch):
        ds = dataset("dna_small"); got = []
        for where in (str(tmp_path), str(tmp_path / "nowhere")):
            monkeypatch.setattr(lo, "SWEEP_RESULTS_DIR", where); got.append(run_prepare(ds, threads=4, block_taxa=5)[3])
        assert not list(tmp_path.iterdir())                                                   # every results directory removed
        a, b = got
        for k in ("cand", "key", "m", "mfl", "nfin"): assert np.array_equal(a.lists[k], b.lists[k], equal_nan=True), k
        assert all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a.closest, b.closest)) and np.array_equal(a.m_full, b.m_full, equal_nan=True)
        assert np.array_equal(a.cls, b.cls) and a.fin_pairs == b.fin_pairs and np.array_equal(a.rr, b.rr, equal_nan=True)


class TestUnresolvedPairs:
    """an identical genome is never a comparator and never votes on a rate; an unresolved pair never votes; the rates
    come from the nearest resolved, non-identical neighbours, so every taxon duplicated still gives every locus a rate"""
    def test_identical_twins_are_never_comparators_and_the_rates_come_from_the_rest(self):
        rng = np.random.default_rng(21); n, nloc, ll = 12, 12, 120
        anc = rng.integers(0, 4, nloc * ll); X = np.tile(anc, (n, 1)); mut = rng.random(X.shape) < 0.08
        X[mut] = (X[mut] + rng.integers(1, 4, int(mut.sum()))) % 4; X = np.repeat(X, 2, axis=0)                  # every taxon twice
        rows = [bytearray(bytes(np.frombuffer(b"ACGT", dtype=np.uint8)[r])) for r in X]; names = [f"t{i:02d}" for i in range(2 * n)]
        loci = [(f"l{i}", np.arange(i * ll, (i + 1) * ll)) for i in range(nloc)]
        P = lo.prepare(argparse.Namespace(fit_overlap=20, threads=1), names, rows, loci, "dna", 5, time.time())
        for g in range(2 * n):
            cand = P.lists["cand"][g]; cand = cand[cand >= 0]
            assert P.m_full[g, g ^ 1] == 0 and np.maximum(P.m_full - P.mfl_full, 0.0)[g, g ^ 1] == 0 and (g ^ 1) not in cand   # the twin: identical, never a comparator
            assert set(cand) == {h for h in range(2 * n) if h not in (g, g ^ 1)}
        assert np.isfinite(P.rate).all() and (P.rate > 0).all()

    def test_two_clades_of_identical_sequences_get_rates_from_the_other_clade(self):
        rng = np.random.default_rng(22); nloc, ll = 12, 100; anc = rng.integers(0, 4, nloc * ll); X = np.tile(anc, (20, 1))
        mut = rng.random(nloc * ll) < 0.05; X[10:, mut] = (X[10:, mut] + rng.integers(1, 4, int(mut.sum()))) % 4
        rows = [bytearray(bytes(np.frombuffer(b"ACGT", dtype=np.uint8)[r])) for r in X]; names = [f"t{i:02d}" for i in range(20)]
        loci = [(f"l{i}", np.arange(i * ll, (i + 1) * ll)) for i in range(nloc)]
        P = lo.prepare(argparse.Namespace(fit_overlap=20, threads=1), names, rows, loci, "dna", 5, time.time())
        assert np.isfinite(P.rate).all() and (P.rate > 0).all()
        m_xs = np.maximum(P.m_full - P.mfl_full, 0.0)
        assert (m_xs[:10, :10][~np.eye(10, dtype=bool)] == 0).all() and (m_xs[:10, 10:] > 0).all()
