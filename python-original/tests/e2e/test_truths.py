"""The method against the planted truths of the synthetic datasets: what it must find, what it must not."""
import numpy as np
import pytest
from tests.conftest import lo, dataset, SYNTH

pytestmark = pytest.mark.e2e


def sim_cols(r):
    """the closest-match family's mask columns: (partner, role)"""
    return r["closest_match_partner"], r["closest_match_role"]


def failing_pairs(R):
    out = {}
    for li in range(R.nloc):
        for e in R.similar(li):
            if e["fail"]: out[(R.loci["names"][li], frozenset((R.taxa["names"][e["taxon"]], R.taxa["names"][e["partner"]])))] = e
    return out


@pytest.mark.parametrize("name", [n for n in SYNTH if "clean" not in n and n != "dna_blocks"])
class TestPlanted:
    def test_contaminants_are_masked_as_too_divergent_by_every_comparator(self, runs, name):
        run = runs(name); ds = dataset(name)
        for e in ds.ev("contam"):
            r = run.masked_at(e["taxon"], e["locus"]); assert r is not None and r["reason"] in ("divergent", "both"), e
            assert int(r["close_relative_comparators"]) == run.report.K and int(r["close_relative_fails"]) >= run.report.M and float(r["close_relative_p"]) <= 10 ** run.report.man["test"]["log10_bar"] + 1e-6

    def test_far_copies_are_masked_by_both_tests_with_the_donor_named(self, runs, name):
        run = runs(name); ds = dataset(name)
        for e in ds.ev("farcopy"):
            r = run.masked_at(e["recipient"], e["locus"]); assert r is not None and r["reason"] in ("both", "similar"), e
            assert r["closest_match_partner"] == e["donor"] and r["closest_match_role"] == "copy" and r["closest_match_arm"] == "outside", e
            assert run.masked_at(e["donor"], e["locus"]) is None, e

    def test_near_copies_are_reported_as_too_similar_pairs(self, runs, name):
        """a copy from a close relative is a failing pair; the recipient is masked as the copy, or both are masked when
        the direction is uncertain (the two fit their relatives almost equally)"""
        run = runs(name); ds = dataset(name); pairs = failing_pairs(run.report)
        for e in ds.ev("nearcopy"):
            key = (e["locus"], frozenset((e["donor"], e["recipient"]))); assert key in pairs, e
            r = run.masked_at(e["recipient"], e["locus"]); partner, role = sim_cols(r if r is not None else {})
            assert r is not None and partner == e["donor"], e
            assert role in ("copy", "uncertain"), e

    def test_sister_copies_are_reported_as_too_similar_pairs(self, runs, name):
        run = runs(name); ds = dataset(name); pairs = failing_pairs(run.report)
        for e in ds.ev("sistercopy"):
            key = (e["locus"], frozenset((e["donor"], e["recipient"])))
            assert key in pairs, e
            masked = [t for t in (e["donor"], e["recipient"]) if run.masked_at(t, e["locus"]) is not None]
            assert masked, e

    def test_the_excluded_taxon_is_untested_unmasked_and_named(self, runs, name):
        run = runs(name); ds = dataset(name); R = run.report
        for e in ds.ev("excluded"):
            g = R.tx(e["taxon"]); assert R.taxa["status"][g].startswith("excluded") and (R.arr("cells/n_comp.npy")[g] == 0).all()
            assert not any(r["taxon"] == e["taxon"] for r in run.masked) and "excluded" in run.log and (e["taxon"] in run.log or " more (the report lists them)" in run.log)

    def test_the_skipped_locus_is_untested_and_named(self, runs, name):
        run = runs(name); ds = dataset(name); R = run.report
        for e in ds.ev("skipped"):
            li = R.lx(e["locus"]); assert R.loci["status"][li] != "tested" and (R.arr("cells/n_comp.npy")[:, li] == 0).all()
            assert e["locus"] in run.log and "not tested" in run.log

    def test_the_ambiguous_cell_fails_with_fewer_comparators(self, runs, name):
        run = runs(name); ds = dataset(name); R = run.report
        for e in ds.ev("ambiguous"):
            g, li = R.tx(e["taxon"]), R.lx(e["locus"]); sd = R.side(g, li)
            assert sd & 1 and sd & 4 and R.cell("n_comp", g, li) == 3 == len(e["comparators"]) and set(R.comps(g, li)) == {R.tx(c) for c in e["comparators"]}
            r = run.masked_at(e["taxon"], e["locus"]); assert r["reason"] in ("divergent", "both") and int(r["close_relative_comparators"]) == 3

    def test_few_masks_beyond_the_planted_ones(self, runs, name):
        run = runs(name); ds = dataset(name)
        planted = {(e.get("taxon") or e.get("recipient"), e["locus"]) for e in ds.events if "locus" in e and (e.get("taxon") or e.get("recipient"))}
        planted |= {(e["donor"], e["locus"]) for e in ds.events if "donor" in e and "locus" in e}
        planted |= {(r, e["locus"]) for e in ds.events if "recipients" in e for r in e["recipients"]}
        extra = [(r["taxon"], r["locus"], r["reason"]) for r in run.masked if (r["taxon"], r["locus"]) not in planted]
        if ds.ev("cladecopy"):                                                 # moving a whole clade orphans its sisters (their closest relatives
            event_loci = {e["locus"] for e in ds.events if e["kind"] in ("cladecopy", "cladejunk")}   # all moved), so collateral masks at the event loci are the method
            assert all(l in event_loci for _, l, _ in extra), extra            # speaking, not noise; elsewhere the dataset must stay clean
            limit = 15
        else:
            limit = 1 if ds.ntaxa <= 100 else max(1, int(0.002 * ds.ntaxa * ds.nloci))      # the multi-block dataset: a rate, not a count
        assert len(extra) <= limit, (len(extra), extra[:12])


@pytest.mark.parametrize("name", ["dna_blocks"])
class TestBlocks:
    """the multi-block dataset (2,100 taxa, clusters of identical genomes, patchy coverage): what its planted events must
    show. The structural events and the contaminants and far copies hold as on the small datasets. A copy from a close
    relative is another matter at this size: the pair is not always found (5 of 6 when the fixture was made), and when it
    is, the direction can be called the other way (the donor fits its relatives worse when the recipient is one of them:
    2 of 5), so the near copies are checked for the pair and the partner, not the direction (notes/KNOWN_ISSUES.md)"""
    def test_structural_events_and_contaminants_hold(self, runs, name):
        P = TestPlanted()
        P.test_contaminants_are_masked_as_too_divergent_by_every_comparator(runs, name); P.test_far_copies_are_masked_by_both_tests_with_the_donor_named(runs, name)
        P.test_the_excluded_taxon_is_untested_unmasked_and_named(runs, name); P.test_the_skipped_locus_is_untested_and_named(runs, name)
        P.test_the_ambiguous_cell_fails_with_fewer_comparators(runs, name); P.test_few_masks_beyond_the_planted_ones(runs, name)

    def test_near_copy_pairs_name_the_right_partner(self, runs, name):
        run = runs(name); ds = dataset(name); pairs = failing_pairs(run.report); found = 0
        for e in ds.ev("nearcopy"):
            if (e["locus"], frozenset((e["donor"], e["recipient"]))) not in pairs: continue
            found += 1; rows = [run.masked_at(t, e["locus"]) for t in (e["donor"], e["recipient"])]
            assert any(rows) and all(sim_cols(r)[0] in (e["donor"], e["recipient"]) for r in rows if r), e
        assert found >= 4, found

    def test_multicopy_recipients_are_masked_and_the_donor_is_not(self, runs, name):
        run = runs(name); ds = dataset(name)
        for e in ds.ev("multicopy"):
            for r in e["recipients"]:
                row = run.masked_at(r, e["locus"]); assert row is not None and row["reason"] in ("similar", "both"), (e, r)
                assert sim_cols(row)[1] in ("copy", "uncertain") or int(run.report.cell("n_similar", run.report.tx(r), run.report.lx(e["locus"]))) > 1, (e, r)   # a recipient is also the pair of the other recipients (KNOWN_ISSUES 4)
            assert run.masked_at(e["donor"], e["locus"]) is None, e

    def test_the_identical_genomes_are_the_planted_copies(self, runs, name):
        run = runs(name); ds = dataset(name)
        copies = sum(len(e["copies"]) for e in ds.ev("clone")) + sum(1 for e in ds.ev("excluded") if "identical_to" in e)
        assert f"{copies} genomes are identical to another at every shared site" in run.log


@pytest.mark.parametrize("name", [n for n in SYNTH if "clean" in n])
class TestClean:
    def test_nothing_is_masked(self, runs, name):
        run = runs(name); assert run.masked == [] and run.report.man["counts"]["masked"] == 0
        assert (run.report.arr("cells/side.npy") & (1 | 2 | 16 | 32)).sum() == 0

    def test_everything_present_is_tested(self, runs, name):
        R = runs(name).report; nc = R.arr("cells/n_comp.npy"); fs = R.arr("cells/focal_sites.npy")
        assert ((nc > 0) == (fs > 0)).all() and R.man["counts"]["excluded_taxa"] == 0 and R.man["counts"]["skipped_loci"] == 0
