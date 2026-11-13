"""Pinned outputs of the current version on the fixture datasets. A failure here means the tool's results changed:
if the change was deliberate, re-pin with  python tests/data/make_fixtures.py --golden  and say so in the commit."""
import json, os
import pytest
from tests.conftest import lo, DATA, SYNTH, SIM, dataset, run_tool, npz_hashes

pytestmark = pytest.mark.regression
GOLDEN = os.path.join(DATA, "golden")


def check(run, gold):
    assert open(run.file("masked.csv")).read() == open(os.path.join(gold, "masked.csv")).read()
    want = json.load(open(os.path.join(gold, "report_hashes.json"))); got = npz_hashes(run.file("report.npz"))
    diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k)); assert not diff, diff
    sub = json.load(open(os.path.join(gold, "manifest_subset.json"))); man = run.report.man
    assert man["format_version"] == sub["format_version"]
    for k, v in sub["counts"].items(): assert man["counts"][k] == v, k
    for k, v in sub["test"].items(): assert man["test"][k] == v, k


@pytest.mark.parametrize("name", SYNTH + SIM)
def test_standard_run_is_pinned(runs, name):
    check(runs(name), os.path.join(GOLDEN, name))


def test_donor_threshold_one_is_pinned(tmp_out):
    ds = dataset("dna_small"); r = run_tool(ds.aln, tmp_out, "dna", ds.parts, ["--donor-threshold", "1"]); assert r.rc == 0
    check(r, os.path.join(GOLDEN, "dna_small_t1"))


def test_gtdb_like_options_are_pinned(tmp_out, monkeypatch):
    """the options a GTDB-size run uses on the multi-block dataset: sparse numeric columns (DENSE_CELLS_MAX below its cells),
    masked-only codes and sequences, best matches for failing cells only (2,100 taxa is above BEST_MATCH_ALL_MAX_TAXA)"""
    monkeypatch.setattr(lo, "DENSE_CELLS_MAX", 10)
    ds = dataset("dna_blocks"); r = run_tool(ds.aln, tmp_out, "dna", ds.parts, ["--report-codes", "masked", "--report-sequences", "masked"]); assert r.rc == 0, r.log
    check(r, os.path.join(GOLDEN, "dna_blocks_gtdb"))


@pytest.mark.real
@pytest.mark.parametrize("name", ["cannon_aa", "near_dna"])
def test_real_excerpt_is_pinned(runs, name):
    gold = os.path.join(GOLDEN, name)
    if not os.path.exists(gold): pytest.skip("no golden outputs for the real excerpts")
    check(runs(name), gold)
