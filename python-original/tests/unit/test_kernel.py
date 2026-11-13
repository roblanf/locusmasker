"""The pair stage's compiled kernel (numba, optional): the selection median and the per-pair medians give the bits of the
numpy path, in plain Python (so coverage sees every line) and compiled, under any thread count, slab or block size; and
the stage takes the kernel only when numba is installed, --no-numba is not given and the work is large enough."""
import sys
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from tests.conftest import lo, dataset, run_tool, run_cli, npz_hashes


def slab(rng, nloc, n, dtype=np.uint16, max_w=600, zero_frac=0.2, dmax=0.5):
    """random per-locus counts for n taxa: W shared sites (zero_frac of them zero), S = W less binomial mismatches at a
    pair-specific divergence, self pairs identical. A small max_w gives many ties."""
    iu = np.triu_indices(n); npair = len(iu[0])
    W = rng.integers(0, max_w + 1, (nloc, npair)); W[rng.random((nloc, npair)) < zero_frac] = 0
    S = W - rng.binomial(W, rng.uniform(0.0, dmax, npair)[None, :])
    self_ = iu[0] == iu[1]; S[:, self_] = W[:, self_]
    return S.astype(dtype), W.astype(dtype), iu


def rates(rng, nloc, p_off=0.2):
    ok_r = rng.random(nloc) > p_off; return np.where(ok_r, rng.uniform(0.5, 2.0, nloc), np.nan), ok_r


def both_paths(monkeypatch, S, W, iu, rr, ok_r, threads=1):
    """pair_divergences by the numpy path and by the kernel on the same counts"""
    n = int(iu[0].max()) + 1
    monkeypatch.setattr(lo, "NUMBA_PAIR_LOCI_MIN", 10 ** 12); ref = lo.pair_divergences(S, W, n, iu, rr, ok_r, threads=threads)
    monkeypatch.setattr(lo, "NUMBA_PAIR_LOCI_MIN", 0); got = lo.pair_divergences(S, W, n, iu, rr, ok_r, threads=threads)
    return ref, got


def same(a, b): return all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))


def outputs(run): return open(run.file("masked.csv")).read(), npz_hashes(run.file("report.npz"))


# ------------------------------------------------------------------ the kernel as plain Python (no numba needed)
class TestSelectionMedianInPython:
    @given(st.data())
    @settings(max_examples=200, deadline=None)
    def test_matches_wmedian1_with_ties_and_zero_weights(self, data):
        L = data.draw(st.integers(1, 120))
        v = np.array(data.draw(st.lists(st.one_of(st.floats(-1e6, 1e6), st.sampled_from([0.0, 1.0, 1.0, 2.5])), min_size=L, max_size=L)))
        w = np.array(data.draw(st.lists(st.integers(0, 6), min_size=L, max_size=L)), float)
        if w.sum() == 0: assert np.isnan(lo._wmed_select_py(v.copy(), w.copy(), L)); return
        assert lo._wmed_select_py(v.copy(), w.copy(), L) == lo.wmedian1(v, w)

    def test_long_rows_go_through_the_partition_loop(self):
        rng = np.random.default_rng(3)
        for L in (16, 17, 64, 353, 1000):
            for _ in range(20):
                v = rng.choice(np.concatenate([[0.0, 0.5, 1.0, 7.0], rng.uniform(0, 100, 10)]), L); w = rng.integers(0, 5, L).astype(float)
                if w.sum() == 0: continue
                assert lo._wmed_select_py(v.copy(), w.copy(), L) == lo.wmedian1(v, w), L

    def test_only_the_first_L_entries_count(self):
        v = np.array([5.0, 1.0, 9.0, 0.0]); w = np.array([1.0, 1.0, 1.0, 100.0])
        assert lo._wmed_select_py(v.copy(), w.copy(), 3) == lo.wmedian1(v[:3], w[:3]) == 5.0

    def test_the_weight_at_exactly_half_picks_the_lower_value(self):
        assert lo._wmed_select_py(np.array([1.0, 2.0]), np.array([1.0, 1.0]), 2) == 1.0 == lo.wmedian1(np.array([1.0, 2.0]), np.array([1.0, 1.0]))


class TestPairMediansInPython:
    def test_matches_the_numpy_pair_stage(self, monkeypatch):
        rng = np.random.default_rng(5); nloc, n = 9, 14; S, W, iu = slab(rng, nloc, n, max_w=12); rr, ok_r = rates(rng, nloc)
        monkeypatch.setattr(lo, "NUMBA_PAIR_LOCI_MIN", 10 ** 12); m, m_fl, nl = lo.pair_divergences(S, W, n, iu, rr, ok_r)
        got = lo._pair_medians_py(S, W, 0, len(iu[0]), rr, ok_r, iu[0] == iu[1], lo.MIN_LOCI_FOR_M, 100.0 * lo.HALF, 5)
        assert same(got, (m[iu], m_fl[iu], nl[iu])) and np.isfinite(got[0]).sum() > 20

    def test_a_slice_of_pairs_and_uint32_counts(self, monkeypatch):
        rng = np.random.default_rng(6); nloc, n = 7, 10; S, W, iu = slab(rng, nloc, n, dtype=np.uint32, max_w=70000)
        monkeypatch.setattr(lo, "NUMBA_PAIR_LOCI_MIN", 10 ** 12); m, m_fl, nl = lo.pair_divergences(S, W, n, iu)          # no rates: the raw divergence
        c0, c1 = 7, 40
        got = lo._pair_medians_py(S, W, c0, c1, np.ones(nloc), np.ones(nloc, bool), iu[0] == iu[1], lo.MIN_LOCI_FOR_M, 100.0 * lo.HALF, 4)
        assert same(got, (m[iu][c0:c1], m_fl[iu][c0:c1], nl[iu][c0:c1]))


# ------------------------------------------------------------------ compiled
@pytest.fixture(scope="module")
def kernel():
    k = lo._pair_kernel(4)
    if k is None: pytest.skip("numba is not installed")
    return k


class TestCompiledKernel:
    @pytest.mark.parametrize("seed,nloc,n,dtype,max_w", [(1, 12, 20, np.uint16, 600), (2, 40, 9, np.uint16, 5), (3, 3, 30, np.uint32, 90000), (4, 1, 6, np.uint16, 100), (5, 353, 12, np.uint16, 800)])
    def test_the_same_bits_as_numpy(self, kernel, monkeypatch, seed, nloc, n, dtype, max_w):
        rng = np.random.default_rng(seed); S, W, iu = slab(rng, nloc, n, dtype, max_w); rr, ok_r = rates(rng, nloc)
        ref, got = both_paths(monkeypatch, S, W, iu, rr, ok_r); assert same(ref, got)
        ref, got = both_paths(monkeypatch, S, W, iu, None, None); assert same(ref, got)

    def test_odd_rates_are_treated_alike(self, kernel, monkeypatch):
        """a zero, infinite, NaN, tiny or huge rate at a locus with ok_r set: both paths drop or keep the locus the same way"""
        rng = np.random.default_rng(7); nloc = 8; S, W, iu = slab(rng, nloc, 12)
        rr = np.array([1.0, 0.0, np.inf, np.nan, 2.0, 0.5, 1e-300, 1e300]); ok_r = np.ones(nloc, bool)
        with np.errstate(divide="ignore", invalid="ignore"): ref, got = both_paths(monkeypatch, S, W, iu, rr, ok_r)   # a real run never has a zero or NaN rate with ok_r set
        assert same(ref, got)

    def test_the_serial_kernel_is_its_own_compilation_with_the_same_bits(self, kernel):
        """the worker processes' serial kernel and the parent's parallel one are compiled (and cached) apart: sharing
        numba's cache file, whichever was compiled first was loaded for both. Same bits, odd rates included"""
        par, ser = lo._pair_medians, lo._pair_medians_serial
        assert par is not ser and par.targetoptions.get("parallel") and not ser.targetoptions.get("parallel")
        assert ser.py_func.__qualname__ != par.py_func.__qualname__
        rng = np.random.default_rng(8); nloc = 8; S, W, iu = slab(rng, nloc, 12); dg = iu[0] == iu[1]; P = len(iu[0])
        for rr, ok_r in (rates(rng, nloc), (np.array([1.0, 0.0, np.inf, np.nan, 2.0, 0.5, 1e-300, 1e300]), np.ones(nloc, bool))):
            with np.errstate(divide="ignore", invalid="ignore"):
                a = par(S, W, 0, P, rr, ok_r, dg, lo.MIN_LOCI_FOR_M, 100.0 * lo.HALF, lo.PAIR_BLOCK)
                b = ser(S, W, 0, P, rr, ok_r, dg, lo.MIN_LOCI_FOR_M, 100.0 * lo.HALF, lo.PAIR_BLOCK)
            assert same(a, b)

    def test_pairs_below_min_loci_and_the_diagonal_are_nan(self, kernel, monkeypatch):
        rng = np.random.default_rng(9); nloc, n = 6, 8; S, W, iu = slab(rng, nloc, n, zero_frac=0.6); rr, ok_r = rates(rng, nloc, 0.0)
        ref, got = both_paths(monkeypatch, S, W, iu, rr, ok_r); assert same(ref, got)
        m, m_fl, nl = got; assert np.isnan(np.diag(m)).all() and np.isnan(m[nl < lo.MIN_LOCI_FOR_M]).all() and (nl < lo.MIN_LOCI_FOR_M).any()

    @given(st.data())
    @settings(max_examples=60, deadline=None)
    def test_the_same_bits_for_any_counts(self, kernel, data):
        nloc = data.draw(st.integers(1, 15)); n = data.draw(st.integers(2, 12)); iu = np.triu_indices(n); npair = len(iu[0])
        W = np.array(data.draw(st.lists(st.integers(0, 40), min_size=nloc * npair, max_size=nloc * npair)), np.uint16).reshape(nloc, npair)
        S = np.minimum(W, np.array(data.draw(st.lists(st.integers(0, 40), min_size=nloc * npair, max_size=nloc * npair)), np.uint16).reshape(nloc, npair))
        ok_r = np.array(data.draw(st.lists(st.booleans(), min_size=nloc, max_size=nloc)))
        rr = np.where(ok_r, np.array(data.draw(st.lists(st.floats(0.1, 10), min_size=nloc, max_size=nloc))), np.nan)
        with pytest.MonkeyPatch.context() as mp: ref, got = both_paths(mp, S, W, iu, rr, ok_r)
        assert same(ref, got)

    def test_threads_slabs_and_blocks_do_not_change_the_bits(self, kernel, monkeypatch):
        rng = np.random.default_rng(8); nloc, n = 20, 60; S, W, iu = slab(rng, nloc, n, max_w=50); rr, ok_r = rates(rng, nloc)
        monkeypatch.setattr(lo, "NUMBA_PAIR_LOCI_MIN", 0); monkeypatch.setattr(lo, "PAIRS_PER_THREAD", 1)
        outs = []
        for threads, slab_, block in ((1, 10 ** 9, 256), (2, 500, 256), (3, 1, 1), (4, 97, 7)):
            monkeypatch.setattr(lo, "NUMBA_PAIR_SLAB", slab_); monkeypatch.setattr(lo, "PAIR_BLOCK", block)
            outs.append(lo.pair_divergences(S, W, n, iu, rr, ok_r, threads=threads))
        assert all(same(outs[0], o) for o in outs[1:])

    def test_the_thread_setter_caps_at_the_pool(self, kernel):
        import numba
        assert kernel[1](1) == 1 and kernel[1](10 ** 6) == numba.config.NUMBA_NUM_THREADS and kernel[1](0) == 1


# ------------------------------------------------------------------ the stage's choice
class TestDispatch:
    def test_small_work_stays_in_numpy(self, runs):
        log = runs("dna_small").log; assert "compiled kernel" not in log and "pair sweep" in log and "numpy chunks" in log

    def test_the_kernel_runs_when_the_work_is_large_enough(self, kernel, runs, tmp_out, monkeypatch):
        monkeypatch.setattr(lo, "NUMBA_PAIR_LOCI_MIN", 0)
        ds = dataset("dna_small"); r = run_tool(ds.aln, tmp_out, "dna", ds.parts); assert r.rc == 0, r.log
        assert r.log.count("compiled kernel (numba)") == 2 and outputs(r) == outputs(runs("dna_small"))

    def test_no_numba_keeps_the_numpy_stage(self, kernel, runs, tmp_out, monkeypatch):
        monkeypatch.setattr(lo, "NUMBA_PAIR_LOCI_MIN", 0)
        ds = dataset("dna_small"); r = run_tool(ds.aln, tmp_out, "dna", ds.parts, ["--no-numba"]); assert r.rc == 0, r.log
        assert "compiled kernel" not in r.log and "numpy chunks" in r.log and lo.USE_NUMBA is True and outputs(r) == outputs(runs("dna_small"))

    def test_without_numba_the_numpy_stage_runs_and_says_so(self, runs, tmp_out, monkeypatch):
        monkeypatch.setattr(lo, "NUMBA_PAIR_LOCI_MIN", 0); monkeypatch.setattr(lo, "_KERNEL", None); monkeypatch.setitem(sys.modules, "numba", None)
        ds = dataset("dna_small"); r = run_tool(ds.aln, tmp_out, "dna", ds.parts); assert r.rc == 0, r.log
        assert r.log.count("numba is not installed") == 1 and "compiled kernel" not in r.log and outputs(r) == outputs(runs("dna_small"))

    def test_the_option_is_documented(self):
        assert "--no-numba" not in run_cli(["--help"]).stdout and "--no-numba" in lo.__doc__     # not user-facing: absent from -h, documented in the module
