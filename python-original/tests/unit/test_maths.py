"""The numerical core: log-gamma, the exact binomial and beta-binomial tails, medians, the normal quantile and the
small helpers. scipy is the oracle; the tool itself never imports it."""
import math
import numpy as np
import pytest
from scipy import stats, special
from tests.conftest import lo


# ------------------------------------------------------------------ log-gamma and log-factorial
class TestLogGamma:
    def test_matches_math_lgamma(self):
        xs = np.concatenate([np.linspace(1e-3, 0.999, 300), np.linspace(1, 50, 500), np.logspace(np.log10(50), 6, 300)])
        got = lo.lgamma_v(xs); want = np.array([math.lgamma(float(x)) for x in xs])
        assert np.allclose(got, want, rtol=1e-12, atol=1e-10)

    def test_scalar_and_shape(self):
        assert lo.lgamma_v(5.0).shape == () and abs(float(lo.lgamma_v(5.0)) - math.log(24)) < 1e-12
        assert lo.lgamma_v(np.ones((2, 3))).shape == (2, 3)

    def test_log_fact_values(self):
        lf = lo.log_fact(20)
        assert len(lf) >= 21 and lf[0] == 0.0
        assert np.allclose(lf[:21], [math.lgamma(k + 1) for k in range(21)], rtol=1e-13)

    def test_log_fact_grows_and_keeps_values(self):
        a = lo.log_fact(10).copy(); b = lo.log_fact(5000)
        assert len(b) >= 5001 and np.array_equal(b[:len(a)], a)
        assert abs(b[5000] - math.lgamma(5001)) < 1e-6


class TestLogSumExp:
    def test_matches_scipy(self):
        rng = np.random.default_rng(1); lp = rng.normal(-5, 3, (20, 64))
        assert np.allclose(lo._logsumexp_rows(lp), special.logsumexp(lp, axis=1))

    def test_rows_with_minus_inf(self):
        lp = np.array([[-np.inf, -np.inf], [-np.inf, 0.0]])
        with np.errstate(divide="ignore"): got = lo._logsumexp_rows(lp)
        assert got[0] == -np.inf and abs(got[1]) < 1e-12


# ------------------------------------------------------------------ the exact tails
def bb_params(n, E, phi):
    """a and b of the beta-binomial with mean E and variance E + phi E^2 among n trials, from the definitions; None when
    no beta-binomial on n trials has that much variance (the tool then uses the most dispersed one there is)"""
    d = E / n; var = E + phi * E * E; V = var / (n * d * (1 - d))
    if V >= n * (1 - 1e-6) or V <= 1: return None
    s = (n - V) / (V - 1); return d * s, (1 - d) * s


def bb_tail_oracle(o, n, a, b, upper):
    """the tail summed from scipy's log pmf in log space: exact to float precision, unlike scipy's own sf/cdf far out"""
    k = np.arange(o, n + 1) if upper else np.arange(0, o + 1)
    return float(special.logsumexp(stats.betabinom.logpmf(k, n, a, b))) if len(k) else -np.inf


class TestBinomialTails:
    GRID = [(n, p, o) for n in (1, 2, 5, 20, 100, 1000, 5000) for p in (0.001, 0.05, 0.3, 0.5, 0.9) for o in sorted({0, 1, int(n * p), int(n * p) + 3, n // 2, n - 1, n})]

    def test_upper_tail_matches_scipy(self):
        n = np.array([g[0] for g in self.GRID], float); p = np.array([g[1] for g in self.GRID]); o = np.array([g[2] for g in self.GRID], float)
        got = lo.bb_tails(o, n, n * p, 0.0, True); want = stats.binom.logsf(o - 1, n.astype(int), p)
        want = np.maximum(want, lo._LOG_FLOOR)
        assert np.allclose(got, want, rtol=0, atol=1e-9), np.c_[n, p, o, got, want][~np.isclose(got, want, atol=1e-9)]

    def test_lower_tail_matches_scipy(self):
        n = np.array([g[0] for g in self.GRID], float); p = np.array([g[1] for g in self.GRID]); o = np.array([g[2] for g in self.GRID], float)
        got = lo.bb_tails(o, n, n * p, 0.0, False); want = np.maximum(stats.binom.logcdf(o, n.astype(int), p), lo._LOG_FLOOR)
        assert np.allclose(got, want, rtol=0, atol=1e-9)

    def test_far_tails_are_exact_not_one_minus(self):
        """a tail of 1e-200 must come out as 1e-200, not as 1 - P(the other side) rounded to zero"""
        got = lo.bb_tails(np.array([900.0]), np.array([1000.0]), np.array([100.0]), 0.0, True)
        want = stats.binom.logsf(899, 1000, 0.1)
        assert abs(float(got[0]) - want) < 1e-6 and want < -400 or abs(float(got[0]) - max(want, lo._LOG_FLOOR)) < 1e-6

    def test_bounds(self):
        rng = np.random.default_rng(3); n = rng.integers(1, 500, 200).astype(float); o = (rng.random(200) * (n + 1)).astype(int).astype(float); E = rng.random(200) * n
        for upper in (True, False):
            for phi in (0.0, 0.05):
                r = lo.bb_tails(o, n, E, phi, upper)
                assert np.all(r <= 0.0) and np.all(r >= lo._LOG_FLOOR) and np.all(np.isfinite(r))

    def test_zero_expectation(self):
        o = np.array([0.0, 1.0, 5.0]); n = np.array([10.0, 10.0, 10.0]); E = np.zeros(3)
        up = lo.bb_tails(o, n, E, 0.0, True); lo_ = lo.bb_tails(o, n, E, 0.0, False)
        assert up[0] == 0.0 and up[1] == lo._LOG_FLOOR and up[2] == lo._LOG_FLOOR
        assert np.all(lo_ == 0.0)

    def test_observed_beyond_the_range(self):
        assert lo.bb_tails(np.array([11.0]), np.array([10.0]), np.array([2.0]), 0.0, True)[0] == lo._LOG_FLOOR    # P(X >= 11 | n = 10) = 0
        assert lo.bb_tails(np.array([10.0]), np.array([10.0]), np.array([2.0]), 0.0, False)[0] == 0.0           # P(X <= n) = 1
        assert lo.bb_tails(np.array([0.0]), np.array([10.0]), np.array([2.0]), 0.0, True)[0] == 0.0             # P(X >= 0) = 1

    def test_empty_input(self):
        assert lo.bb_tails(np.array([]), np.array([]), np.array([]), 0.0, True).shape == (0,)

    def test_tiny_phi_is_the_binomial(self):
        n = np.array([100.0, 2000.0]); E = np.array([10.0, 50.0]); o = np.array([20.0, 80.0])
        assert np.allclose(lo.bb_tails(o, n, E, 1e-12, True), lo.bb_tails(o, n, E, 0.0, True), atol=1e-10)

    def test_a_mixed_vector_is_each_case_at_once(self):
        """mixed zero expectations, binomial-limit pairs and ordinary pairs in one call give the per-case answers"""
        n = np.array([10.0, 50.0, 200.0, 7.0]); E = np.array([0.0, 5.0, 20.0, 3.0]); o = np.array([2.0, 9.0, 35.0, 7.0])
        got = lo.bb_tails(o, n, E, 0.02, True)
        for i in range(4):
            assert np.isclose(got[i], lo.bb_tails(o[i:i + 1], n[i:i + 1], E[i:i + 1], 0.02, True)[0], atol=1e-10)


class TestBetaBinomialTails:
    CASES = [(n, E, phi, o) for n in (5, 30, 200, 1500) for E in (0.5, 3.0, 15.0) for phi in (0.01, 0.1, 0.5) for o in (0, 1, 3, 10, 40) if E < n and o <= n and bb_params(n, E, phi) is not None]

    def test_mean_and_variance_of_the_model(self):
        """the model is a beta-binomial with mean E and variance E + phi E^2: scipy's moments of the parameters implied"""
        for n, E, phi, _ in self.CASES[::7]:
            a, b = bb_params(n, E, phi); d = stats.betabinom(n, a, b)
            assert np.isclose(d.mean(), E, rtol=1e-9) and np.isclose(d.var(), E + phi * E * E, rtol=1e-9)

    def test_upper_tail_matches_scipy(self):
        for n, E, phi, o in self.CASES:
            a, b = bb_params(n, E, phi)
            got = float(lo.bb_tails(np.array([o], float), np.array([n], float), np.array([E]), phi, True)[0])
            want = max(bb_tail_oracle(o, n, a, b, True), lo._LOG_FLOOR)
            assert abs(got - want) < 1e-6 * max(1.0, abs(want)), (n, E, phi, o, got, want)

    def test_lower_tail_matches_scipy(self):
        for n, E, phi, o in self.CASES:
            a, b = bb_params(n, E, phi)
            got = float(lo.bb_tails(np.array([o], float), np.array([n], float), np.array([E]), phi, False)[0])
            want = max(bb_tail_oracle(o, n, a, b, False), lo._LOG_FLOOR)
            assert abs(got - want) < 1e-6 * max(1.0, abs(want)), (n, E, phi, o, got, want)

    def test_negligible_overdispersion_uses_the_binomial_limit(self):
        """a Beta concentration above 1e9 (a tiny phi on a very long locus) would lose nats to rounding in the gamma
        terms: the binomial is used for those pairs, and the others in the same call still get the beta-binomial"""
        o = np.array([130.0, 70.0]); n = np.array([1e6, 1e6]); E = np.array([100.0, 100.0]); phi = 1e-7
        d = E / n; rho = (d + phi * n * d) / ((1 - d) * (n - 1)); assert (1 / rho - 1 > 1e9).all()
        for upper in (True, False):
            assert np.array_equal(lo.bb_tails(o, n, E, phi, upper), lo.bb_tails(o, n, E, 0.0, upper))
            mixed = lo.bb_tails(np.array([130.0, 70.0, 40.0]), np.array([1e6, 1e6, 60.0]), np.array([100.0, 100.0, 20.0]), phi, upper)
            assert np.array_equal(mixed[:2], lo.bb_tails(o, n, E, 0.0, upper))
            assert np.array_equal(mixed[2:], lo.bb_tails(np.array([40.0]), np.array([60.0]), np.array([20.0]), phi, upper))
            assert mixed[2] != lo.bb_tails(np.array([40.0]), np.array([60.0]), np.array([20.0]), 0.0, upper)[0]   # that one is overdispersed

    def test_phi_per_pair_matches_the_scalar_calls(self):
        """one call with a dispersion per pair (the loci of a batch) is the same arithmetic as a call per locus"""
        o = np.array([3.0, 40.0, 70.0, 0.0, 12.0, 25.0]); n = np.array([50.0, 60.0, 5000.0, 30.0, 20.0, 40.0]); E = np.array([5.0, 20.0, 50.0, 0.0, 15.0, 25.0])
        phis = np.array([0.0, 0.3, 1e-7, 0.05, 0.0, 0.2])
        for upper in (True, False):
            one = lo.bb_tails(o, n, E, phis, upper)
            each = np.array([lo.bb_tails(o[i:i + 1], n[i:i + 1], E[i:i + 1], float(phis[i]), upper)[0] for i in range(len(o))])
            assert np.array_equal(one, each)
            assert np.array_equal(lo.bb_tails(o, n, E, 0.2, upper), lo.bb_tails(o, n, E, np.full(len(o), 0.2), upper))

    def test_single_site_pairs_stay_binomial(self):
        got = lo.bb_tails(np.array([1.0]), np.array([1.0]), np.array([0.3]), 0.2, True)
        assert np.isclose(got[0], math.log(0.3), atol=1e-10)

    def test_overdispersion_widens_the_tail(self):
        o = np.array([30.0]); n = np.array([100.0]); E = np.array([10.0])
        assert lo.bb_tails(o, n, E, 0.2, True)[0] > lo.bb_tails(o, n, E, 0.0, True)[0]

    def test_upper_tail_falls_with_the_observed_count(self):
        n = np.full(60, 200.0); E = np.full(60, 20.0); o = np.arange(20, 80, dtype=float)
        for phi in (0.0, 0.1):
            t = lo.bb_tails(o, n, E, phi, True); assert np.all(np.diff(t) <= 1e-12)
            t = lo.bb_tails(o, n, E, phi, False); assert np.all(np.diff(t) >= -1e-12)


class TestLogPmf:
    def test_binomial_matches_scipy(self):
        k = np.arange(0, 51, dtype=float); n = np.full(51, 50.0); E = np.full(51, 8.0)
        assert np.allclose(lo.bb_logpmf(k, n, E, 0.0), stats.binom.logpmf(k, 50, 8 / 50), atol=1e-9)

    def test_beta_binomial_matches_scipy(self):
        n, E, phi = 80, 12.0, 0.15; a, b = bb_params(n, E, phi); k = np.arange(0, 81, dtype=float)
        got = lo.bb_logpmf(k, np.full(81, float(n)), np.full(81, E), phi)
        assert np.allclose(got, stats.betabinom.logpmf(k, n, a, b), atol=1e-8)

    def test_pmf_sums_to_one(self):
        for phi in (0.0, 0.3):
            k = np.arange(0, 121, dtype=float); p = lo.bb_logpmf(k, np.full(121, 120.0), np.full(121, 30.0), phi)
            assert abs(np.exp(p).sum() - 1.0) < 1e-9

    def test_pmf_is_never_above_its_tail(self):
        """the screening use: P(X <= k) >= P(X = k)"""
        k = np.arange(0, 41, dtype=float); n = np.full(41, 40.0); E = np.full(41, 6.0)
        assert np.all(lo.bb_logpmf(k, n, E, 0.1) <= lo.bb_tails(k, n, E, 0.1, False) + 1e-9)


# ------------------------------------------------------------------ medians
def brute_wmedian(v, w):
    """the smallest value at which the cumulative weight reaches half of the total"""
    o = np.argsort(v, kind="stable"); cw = np.cumsum(w[o]); return float(v[o][np.searchsorted(cw, cw[-1] / 2.0)])


class TestMedians:
    def test_wmedian1_unit_weights_is_a_median(self):
        v = np.array([5.0, 1.0, 3.0, 2.0, 4.0]); assert lo.wmedian1(v, np.ones(5)) == 3.0

    def test_wmedian1_weights_matter(self):
        v = np.array([1.0, 2.0, 3.0]); assert lo.wmedian1(v, np.array([1.0, 1.0, 10.0])) == 3.0
        assert lo.wmedian1(v, np.array([10.0, 1.0, 1.0])) == 1.0

    def test_wmedian1_is_a_valid_weighted_median(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            k = int(rng.integers(1, 30)); v = rng.normal(size=k); w = rng.random(k) * (rng.random(k) > 0.2)
            if w.sum() == 0: continue
            med = lo.wmedian1(v, w); tot = w.sum()
            assert w[v < med].sum() <= tot / 2 + 1e-12 and w[v > med].sum() <= tot / 2 + 1e-12

    def test_wmedian1_all_zero_weights_falls_back_to_the_plain_median(self):
        assert lo.wmedian1(np.array([1.0, 5.0, 9.0]), np.zeros(3)) == 5.0

    def test_wmedian_stack_matches_brute_force(self):
        rng = np.random.default_rng(11); L, P = 12, 40
        vals = rng.normal(size=(L, P)); wts = rng.random((L, P)) * (rng.random((L, P)) > 0.3)
        vals[rng.random((L, P)) < 0.1] = np.nan
        got = lo.wmedian_stack(vals, wts)
        for p in range(P):
            ok = np.isfinite(vals[:, p]) & (wts[:, p] > 0)
            if not ok.any(): assert np.isnan(got[p])
            else: assert got[p] == brute_wmedian(vals[ok, p], wts[ok, p])

    def test_wmedian_stack_shapes_and_nan(self):
        assert lo.wmedian_stack(np.full((3, 2), np.nan), np.ones((3, 2))).tolist() == [np.nan, np.nan] or np.isnan(lo.wmedian_stack(np.full((3, 2), np.nan), np.ones((3, 2)))).all()
        assert lo.wmedian_stack(np.array([[1.0], [2.0], [3.0]]), np.array([[1.0], [0.0], [0.0]]))[0] == 1.0

    def test_pair_median_min_loci_rule(self):
        vals = np.array([[1.0, 1.0], [2.0, np.nan], [3.0, 3.0]]); W = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        v, nl = lo.pair_median(vals, W, 2, -1.0)
        assert nl.tolist() == [3, 1] and v[0] == 2.0 and v[1] == -1.0


# ------------------------------------------------------------------ the normal quantile and the binomial pmf helper
class TestNormQuantile:
    def test_matches_scipy_on_the_upper_half(self):
        for p in (0.5, 0.6, 0.9, 0.975, 0.999, 1 - 1e-6, 1 - 1e-9, 1 - 1e-12):
            assert abs(lo.norm_quantile(p) - stats.norm.ppf(p)) < 1e-8, p

    def test_lower_half_is_negative(self):
        """the inverse normal CDF of p < 0.5 is negative; the tool only ever asks for p > 0.5, but the function says it is the inverse CDF"""
        assert abs(lo.norm_quantile(0.1) - stats.norm.ppf(0.1)) < 1e-8


class TestLbinom:
    def test_integer_values_match_scipy(self):
        for o, q in ((10, 0.3), (50, 0.05), (7, 0.5)):
            for m in range(0, o + 1):
                assert abs(lo.lbinom(m, o, q) - stats.binom.logpmf(m, o, q)) < 1e-9

    def test_continuous_in_m(self):
        a, b, c = lo.lbinom(3.0, 10, 0.3), lo.lbinom(3.5, 10, 0.3), lo.lbinom(4.0, 10, 0.3)
        assert min(a, c) - 1 < b < max(a, c) + 1

    def test_q_is_clamped_away_from_zero_and_one(self):
        assert np.isfinite(lo.lbinom(0, 10, 0.0)) and np.isfinite(lo.lbinom(10, 10, 1.0))


# ------------------------------------------------------------------ small helpers
class TestHelpers:
    def test_expand_round_trips_the_upper_triangle(self):
        n = 6; iu = np.triu_indices(n); rng = np.random.default_rng(2); tri = rng.normal(size=len(iu[0]))
        M = lo._expand(tri, n, iu)
        assert np.array_equal(M, M.T) and np.array_equal(M[iu], tri)

    @pytest.mark.parametrize("sites,runs", [([0, 1, 2, 3], [[0, 3, 1]]), ([5], [[5, 5, 1]]), ([0, 2, 4, 6], [[0, 6, 2]]), ([0, 1, 2, 5], [[0, 2, 1], [5, 5, 1]]),
                                            ([0, 2], [[0, 0, 1], [2, 2, 1]]), ([0, 1, 3], [[0, 1, 1], [3, 3, 1]]), ([], [])])
    def test_ranges(self, sites, runs):
        assert lo._ranges(np.array(sites, dtype=int)) == runs

    def test_ranges_expand_back_to_the_sites(self):
        rng = np.random.default_rng(9)
        for _ in range(300):
            sites = sorted(set(rng.integers(0, 80, int(rng.integers(1, 30))).tolist()))
            back = [s for a, b, st in lo._ranges(np.array(sites)) for s in range(a, b + 1, st)]
            assert back == sites, sites

    @pytest.mark.parametrize("v,nd,out", [(1.23456, 2, "1.23"), (-0.0001, 2, "0.00"), (-1.5, 0, "-2"), (float("nan"), 2, ""), (float("inf"), 1, ""), (2, 3, "2.000"), (0.125, 2, "0.12")])
    def test_fnum(self, v, nd, out):
        assert lo.fnum(v, nd) == out

    def test_sha256(self, tmp_path):
        import hashlib
        p = tmp_path / "f"; p.write_bytes(b"abc" * 100000)
        assert lo._sha256(str(p)) == hashlib.sha256(b"abc" * 100000).hexdigest()

    def test_cpus_and_memory_are_positive(self):
        assert lo._cpus() >= 1 and lo._mem_available() > 0

    def test_git_commit_is_a_string(self):
        c = lo._git_commit(); assert isinstance(c, str) and c
