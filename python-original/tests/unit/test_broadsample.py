"""The broad-sample test's pieces: the numpy-only special functions against scipy, the scale null's calibration, and
the hash draws (deterministic, distinct, uniform enough, and free of numpy's generator algorithms)."""
import math
import numpy as np
import pytest
from tests.conftest import lo

scipy_special = pytest.importorskip("scipy.special", reason="the reference for the numpy-only implementations")
from scipy import stats as sstats  # noqa: E402


class TestSpecialFunctions:
    def test_gammaln_matches_scipy(self):
        x = np.concatenate([np.linspace(0.05, 3, 200), np.linspace(3, 60, 200), [0.5, 1.0, 1.5, 2.0, 24.5, 25.0]])
        np.testing.assert_allclose(lo.gammaln_np(x), scipy_special.gammaln(x), rtol=2e-12, atol=2e-12)

    def test_digamma_and_trigamma_match_scipy(self):
        x = np.concatenate([np.linspace(0.5, 30, 300), [0.5, 1.0, 2.0, 24.5]])
        np.testing.assert_allclose(lo.digamma_np(x), scipy_special.digamma(x), rtol=1e-10, atol=1e-10)   # the zero crossing near 1.4616 makes pure rtol meaningless
        np.testing.assert_allclose(lo.trigamma_np(x), scipy_special.polygamma(1, x), rtol=1e-12, atol=1e-12)

    def test_chi2_logsf_matches_scipy(self):
        rng = np.random.default_rng(1)
        nu = rng.integers(1, 60, 400).astype(float); x = rng.uniform(0, 1, 400) * nu * rng.choice([0.1, 1.0, 3.0, 10.0, 40.0], 400)
        ours = lo.chi2_logsf_np(x, nu); ref = sstats.chi2.logsf(x, nu)
        fin = np.isfinite(ref)                                         # scipy's sf underflows to 0 (logsf -inf) beyond ~1e-308; ours stays finite in log space
        np.testing.assert_allclose(ours[fin], ref[fin], rtol=1e-9, atol=1e-10)   # near p = 1 the log is ~ -1e-11: absolute error is the honest metric there
        assert np.all(ours[~fin] < -700)

    def test_chi2_logsf_far_tail_and_edges(self):
        np.testing.assert_allclose(lo.chi2_logsf_np(np.array([500.0]), np.array([3.0])), sstats.chi2.logsf(500.0, 3), rtol=1e-10)
        assert lo.chi2_logsf_np(np.array([0.0]), np.array([5.0]))[0] == 0.0
        assert lo.chi2_logsf_np(np.array([1e-12]), np.array([1.0]))[0] <= 0.0


class TestScaleNull:
    def test_calibrated_on_its_own_null(self):
        """rows drawn FROM the null (log sigma2 normal times chi-square noise): the p-values are uniform-ish and the
        smallest of 50,000 is not far beyond 1/50,000"""
        rng = np.random.default_rng(7); n = 50_000
        k = rng.integers(10, 51, n); nu = k - 1
        s2 = np.exp(rng.normal(0.0, 0.4, n))
        x = s2 * rng.chisquare(nu) / nu
        x = x / np.median(x)
        lp, tau = rng_lp = lo.bs_null(x, k)
        assert 0.25 < tau < 0.6
        p = np.exp(lp)
        assert abs(float((p < 0.1).mean()) - 0.1) < 0.02
        assert float(lp.min()) > math.log(1.0 / n) - 5.0

    def test_a_spread_row_is_extreme(self):
        rng = np.random.default_rng(3); n = 5_000
        k = np.full(n, 50); x = rng.chisquare(49, n) / 49
        x[0] = 8.0
        lp, _ = lo.bs_null(x / np.median(x), k)
        assert lp[0] < math.log(1e-12) and lp[1:].min() > math.log(1e-6)


class TestHashDraws:
    def test_distinct_in_range_and_deterministic(self):
        keys = lo._sm64(np.arange(1000, dtype=np.uint64))
        a = lo._hash_draw(keys, 500, 50); b = lo._hash_draw(keys.copy(), 500, 50)
        assert np.array_equal(a, b) and a.shape == (1000, 50)
        assert a.min() >= 0 and a.max() < 500
        assert all(len(set(r.tolist())) == 50 for r in a)

    def test_small_pool_branch(self):
        keys = lo._sm64(np.arange(40, dtype=np.uint64) + np.uint64(7))
        a = lo._hash_draw(keys, 60, 50)
        assert a.shape == (40, 50) and all(len(set(r.tolist())) == 50 for r in a) and a.max() < 60

    def test_roughly_uniform(self):
        keys = lo._sm64(np.arange(4000, dtype=np.uint64) ^ np.uint64(0xABCDEF))
        a = lo._hash_draw(keys, 500, 50)
        counts = np.bincount(a.ravel(), minlength=500)
        assert counts.min() > 0.5 * counts.mean() and counts.max() < 1.6 * counts.mean()

    def test_first_distinct_keeps_slot_order(self):
        vals = np.array([[3, 7, 3, 1, 7, 2]])
        got, ok = lo._first_distinct(vals, 4)
        assert ok[0] and got.tolist() == [[3, 7, 1, 2]]


class TestOrderStatBound:
    def test_k1_is_the_bonferroni_bound(self):
        lp = np.log(np.array([1e-4, 0.02])); nf = np.array([20, 30])
        got = lo.order_stat_bound(lp, nf, np.ones(2))
        np.testing.assert_allclose(got, np.minimum(lp + np.log(nf), 0.0))

    def test_bound_holds_by_simulation(self):
        """P(bound < log alpha under uniform p's) <= alpha, for k = 1, 2 and half"""
        rng = np.random.default_rng(5); trials = 20_000; nf = 24
        p = rng.uniform(0, 1, (trials, nf)); sp = np.sort(np.log(p), 1)
        for k in (1, 2, 12):
            b = lo.order_stat_bound(sp[:, k - 1], np.full(trials, nf), np.full(trials, k))
            assert float((b < math.log(0.05)).mean()) <= 0.05 + 0.004, k

    def test_a_shifted_half_fires_and_one_spike_does_not(self):
        nf = 24; half = np.full(nf, 0.5); half[:12] = 1e-3
        sp = np.sort(np.log(half))
        assert lo.order_stat_bound(sp[11:12], np.array([nf]), np.array([12]))[0] < math.log(1e-6)
        spike = np.full(nf, 0.5); spike[0] = 1e-30; sp2 = np.sort(np.log(spike))
        assert lo.order_stat_bound(sp2[11:12], np.array([nf]), np.array([12]))[0] == 0.0

    def test_fewer_far_comparators_than_k_gives_log1(self):
        assert lo.order_stat_bound(np.array([-30.0]), np.array([3]), np.array([5]))[0] == 0.0
