"""Property-based tests (hypothesis): round trips through the readers and writers, and the algebra the tails and
medians must obey for any input."""
import math, os
import numpy as np
import pytest
from hypothesis import given, assume, strategies as st
from tests.conftest import lo

pytestmark = pytest.mark.property

name_st = st.text(alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."), min_size=1, max_size=12)
dna_st = st.text(alphabet=st.sampled_from("ACGTacgtN-?"), min_size=1, max_size=40)


@st.composite
def alignment(draw, min_taxa=1, max_taxa=8):
    n = draw(st.integers(min_taxa, max_taxa)); L = draw(st.integers(1, 40))
    names = draw(st.lists(name_st, min_size=n, max_size=n, unique=True))
    seqs = [draw(st.text(alphabet=st.sampled_from("ACGT-?"), min_size=L, max_size=L)) for _ in range(n)]
    return names, seqs


class TestRoundTrips:
    @given(aln=alignment())
    def test_fasta(self, aln, tmp_path_factory):
        names, seqs = aln; d = tmp_path_factory.mktemp("fa")
        out = lo.write_masked("/x/a.fasta", str(d), "fasta", names, [bytearray(s.encode()) for s in seqs], None)
        assert lo.read_fasta(open(out).read()) == (names, seqs)

    @given(aln=alignment())
    def test_phylip(self, aln, tmp_path_factory):
        names, seqs = aln; d = tmp_path_factory.mktemp("phy")
        out = lo.write_masked("/x/a.phy", str(d), "phylip", names, [bytearray(s.encode()) for s in seqs], None)
        assert lo.read_phylip(open(out).read()) == (names, seqs)

    @given(aln=alignment(), data=st.data())
    def test_nexus_fast_path(self, aln, data, tmp_path_factory):
        names, seqs = aln; L = len(seqs[0]); d = tmp_path_factory.mktemp("nex")
        k = data.draw(st.integers(0, 3)); charsets = {}
        for i in range(k):
            sites = data.draw(st.lists(st.integers(0, L - 1), min_size=1, max_size=L, unique=True))
            charsets[f"c{i}"] = sorted(sites)
        info = lo.NexusInfo("dna", "?", "-", charsets, {})
        out = lo.write_masked("/x/a.nex", str(d), "nexus", names, [bytearray(s.encode()) for s in seqs], info)
        n2, r2, c2, i2 = lo.read_nexus_fast(out)
        assert n2 == names and [bytes(r).decode() for r in r2] == seqs and c2 == charsets

    @given(st.lists(st.integers(0, 200), min_size=0, max_size=60))
    def test_compact4nexus_expands_back(self, sites):
        sites = sorted(set(sites)); txt = lo._compact4nexus(sites); back = []
        for tok in txt.split():
            a, _, rest = tok.partition("-"); b, _, step = rest.partition("\\") if rest else (a, "", "")
            back.extend(range(int(a) - 1, int(b or a), int(step or 1)))
        assert back == sites

    @given(st.lists(st.integers(0, 300), min_size=0, max_size=80))
    def test_ranges_expand_back(self, sites):
        sites = sorted(set(sites)); back = [s for a, b, st_ in lo._ranges(np.array(sites, dtype=int)) for s in range(a, b + 1, st_)]
        assert back == sites

    @given(data=st.data())
    def test_partition_file(self, data, tmp_path_factory):
        """random non-overlapping partitions in every written form come back as their sites, in file order"""
        L = data.draw(st.integers(1, 120)); cuts = sorted(set(data.draw(st.lists(st.integers(1, L), max_size=6)) + [0, L]))
        lines = []; want = []
        for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            form = data.draw(st.sampled_from(["range", "list", "step"]))
            if form == "range": rhs = f"{a + 1}-{b}"; sites = list(range(a, b))
            elif form == "list": rhs = ", ".join(str(s + 1) for s in range(a, b)); sites = list(range(a, b))
            else: step = data.draw(st.integers(1, 3)); rhs = f"{a + 1}-{b}\\{step}"; sites = list(range(a, b, step))
            prefix = data.draw(st.sampled_from(["", "DNA, ", "WAG,"])); lines.append(f"{prefix}p{i} = {rhs}"); want.append((f"p{i}", sites))
        p = tmp_path_factory.mktemp("parts") / "p.txt"; p.write_text("\n".join(lines) + "\n")
        got = lo.parse_raxml_partitions(str(p), L)
        assert [(k, s.tolist()) for k, s in got] == want


def feasible(n, d, phi):
    """a beta-binomial on n trials can have the variance E + phi E^2 (above it the tool clamps the dispersion)"""
    E = n * d; V = (E + phi * E * E) / (n * d * (1 - d)); return phi == 0 or 1 < V < n * (1 - 1e-6)


class TestTails:
    @given(st.integers(1, 3000), st.floats(1e-6, 0.95), st.floats(0, 0.6), st.data())
    def test_bounds_and_monotonicity(self, n, d, phi, data):
        assume(feasible(n, d, phi))
        E = n * d; os_ = sorted(data.draw(st.lists(st.integers(0, n), min_size=2, max_size=6)))
        o = np.array(os_, float); nn = np.full(len(o), float(n)); EE = np.full(len(o), E)
        up = lo.bb_tails(o, nn, EE, phi, True); lo_ = lo.bb_tails(o, nn, EE, phi, False)
        assert np.all(up <= 0) and np.all(lo_ <= 0) and np.all(up >= lo._LOG_FLOOR) and np.all(lo_ >= lo._LOG_FLOOR)
        assert np.all(np.diff(up) <= 1e-8) and np.all(np.diff(lo_) >= -1e-8)      # 1e-8: the gamma terms of a Beta with a + b near 1e9 carry about 1e-9 of rounding

    @given(st.integers(1, 2000), st.floats(1e-4, 0.9999), st.integers(0, 2000))
    def test_binomial_tails_add_to_one(self, n, d, o):
        assume(o <= n and o >= 1)
        up = lo.bb_tails(np.array([float(o)]), np.array([float(n)]), np.array([n * d]), 0.0, True)[0]
        dn = lo.bb_tails(np.array([float(o - 1)]), np.array([float(n)]), np.array([n * d]), 0.0, False)[0]
        if up > -600 and dn > -600: assert abs(np.logaddexp(up, dn)) < 1e-8

    @given(st.integers(2, 2000), st.floats(1e-4, 0.95), st.floats(1e-4, 0.5), st.integers(0, 2000))
    def test_beta_binomial_tails_add_to_one(self, n, d, phi, o):
        assume(o <= n and o >= 1 and feasible(n, d, phi))
        up = lo.bb_tails(np.array([float(o)]), np.array([float(n)]), np.array([n * d]), phi, True)[0]
        dn = lo.bb_tails(np.array([float(o - 1)]), np.array([float(n)]), np.array([n * d]), phi, False)[0]
        if up > -600 and dn > -600: assert abs(np.logaddexp(up, dn)) < 1e-7

    @pytest.mark.xfail(strict=True, reason="PRECISION: where the variance asked for exceeds a beta-binomial's (the dispersion is clamped) the tails are off by up to about 1e-5 in log, from the gamma recurrence near its pole")
    def test_tails_stay_exact_where_the_dispersion_is_clamped(self):
        n, d, phi = 62, 0.999, 0.5; E = n * d
        up = lo.bb_tails(np.array([1.0, 42.0]), np.array([n, n], float), np.array([E, E]), phi, True)
        assert up[1] <= up[0] + 1e-9
        up1 = lo.bb_tails(np.array([1.0]), np.array([17.0]), np.array([17 * 0.9999]), 0.5, True)[0]; dn0 = lo.bb_tails(np.array([0.0]), np.array([17.0]), np.array([17 * 0.9999]), 0.5, False)[0]
        assert abs(np.logaddexp(up1, dn0)) < 1e-7

    @given(st.floats(1e-3, 1e6))
    def test_lgamma(self, x):
        assert math.isclose(float(lo.lgamma_v(x)), math.lgamma(x), rel_tol=1e-11, abs_tol=1e-9)

    @given(st.integers(1, 500), st.floats(0.01, 0.99))
    def test_pmf_sums_to_one(self, n, d):
        k = np.arange(0, n + 1, dtype=float)
        for phi in (0.0, 0.2):
            assert abs(np.exp(lo.bb_logpmf(k, np.full(n + 1, float(n)), np.full(n + 1, n * d), phi)).sum() - 1) < 1e-8


class TestMedians:
    @given(st.lists(st.floats(-100, 100), min_size=1, max_size=40), st.data())
    def test_weighted_median_is_valid(self, v, data):
        v = np.array(v); w = np.array(data.draw(st.lists(st.one_of(st.just(0.0), st.floats(1e-6, 10)), min_size=len(v), max_size=len(v))))
        assume(w.sum() > 0)
        med = lo.wmedian1(v, w); tot = w.sum()
        assert med in set(v.tolist()) and w[v < med].sum() <= tot / 2 + 1e-9 and w[v > med].sum() <= tot / 2 + 1e-9

    @given(st.integers(1, 12), st.integers(1, 12), st.data())
    def test_stack_agrees_with_the_one_dimensional_median(self, L, P, data):
        vals = np.array(data.draw(st.lists(st.floats(-50, 50), min_size=L * P, max_size=L * P))).reshape(L, P)
        wts = np.array(data.draw(st.lists(st.one_of(st.just(0.0), st.floats(1e-6, 5)), min_size=L * P, max_size=L * P))).reshape(L, P)
        got = lo.wmedian_stack(vals, wts)
        for p in range(P):
            if wts[:, p].sum() > 0: assert got[p] == lo.wmedian1(vals[:, p], wts[:, p])
            else: assert np.isnan(got[p])


class TestSmall:
    @given(st.floats(0.5001, 1 - 1e-12), st.floats(0.5001, 1 - 1e-12))
    def test_norm_quantile_is_monotonic(self, p, q):
        a, b = sorted((p, q)); assert lo.norm_quantile(a) <= lo.norm_quantile(b) + 1e-12

    @given(st.floats(allow_nan=True, allow_infinity=True), st.integers(0, 6))
    def test_fnum_never_writes_negative_zero_and_parses_back(self, v, nd):
        s = lo.fnum(v, nd)
        if not math.isfinite(v): assert s == ""
        else:
            assert s != "-" + "0." + "0" * nd and s != "-0" and abs(float(s) - v) <= 0.5 * 10 ** -nd + 1e-9 * abs(v)
