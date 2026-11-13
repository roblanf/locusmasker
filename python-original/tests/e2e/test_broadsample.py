"""The broad-sample test end to end, on the fixture built for it (dna_rc): a clade moved wholesale is caught as too
close (blue, the donor named), a clade overwritten by junk is caught as too divergent (red, by the majority of the
sample), the test calls one direction at most, clean data raises nothing, the family enlarges the run's single bar,
and --no-broad-sample switches all of it off."""
import csv
import numpy as np
import pytest
from tests.conftest import lo, dataset, run_tool


@pytest.fixture(scope="module")
def rc_run(tmp_path_factory):
    ds = dataset("dna_rc")
    r = run_tool(ds.aln, str(tmp_path_factory.mktemp("rc")), "dna", ds.parts, ["--threads", "4"])
    assert r.rc == 0, r.log
    return ds, r


def masked_rows(r): return list(csv.DictReader(open(r.file("masked.csv"))))


def broad_masked(row):
    """masked by the broad-sample test itself: broad columns filled, no other family claims the mask"""
    return row["mask_test"] == "broad_sample" and row["close_relative_fails"] == "" and row["closest_match_role"] == ""


class TestOnThePlantedFixture:
    def test_cladecopy_recipients_are_caught_too_close(self, rc_run):
        """a distant donor's sequence over a whole clade: each member's closest relatives moved with it, so only the
        broad sample sees the young coalescence; most recipients are masked blue with the event's locus"""
        ds, r = rc_run; rows = masked_rows(r)
        events = ds.ev("cladecopy"); recips = {(e["locus"], t) for e in events for t in e["recipients"]}
        fired = [row for row in rows if row["broad_sample_side"] == "close" and row["reason"] in ("similar", "both")]   # masked too close (a divergent cell also too close is a label)
        hits = [row for row in fired if (row["locus"], row["taxon"]) in recips]
        assert len(hits) >= 4, [(_r["locus"], _r["taxon"]) for _r in fired]
        assert all(float(row["broad_sample_close_p"]) <= 0.05 + 1e-6 for row in hits)
        assert all(row["broad_sample_role"] in ("copy", "uncertain") or row["reason"] == "both" for row in hits)   # a divergent cell's too-close firing is a label: no direction
        assert all(float(row["broad_sample_partner_mismatches"]) < float(row["broad_sample_partner_expected"]) for row in hits)
        donors = {e["donor"] for e in events}
        assert any(row["broad_sample_partner"] in donors for row in hits)   # the most extreme aligned comparator names the donor

    def test_cladejunk_is_caught_too_divergent(self, rc_run):
        """a random sequence over a whole clade: the members agree with each other, so the close-relative test sees
        nothing, but a majority of the broad sample finds each of them far too divergent - masked red"""
        ds, r = rc_run; rows = masked_rows(r)
        (e,) = ds.ev("cladejunk")
        hits = [q for q in rows if q["locus"] == e["locus"] and q["taxon"] in e["recipients"] and q["broad_sample_side"] == "divergent"]
        assert len(hits) >= 3, [(q["locus"], q["taxon"], q["mask_test"]) for q in rows if q["locus"] == e["locus"]]
        for row in hits:
            assert row["reason"] == "divergent" and broad_masked(row) and row["mask_test"] == "broad_sample"
            assert float(row["broad_sample_divergent_observed"]) > float(row["broad_sample_divergent_expected"]) and float(row["broad_sample_divergent_p"]) <= 0.05 + 1e-6
            assert int(row["broad_sample_members_failing"]) * 2 > int(row["broad_sample_members"])         # a majority
            g = r.report.taxa["names"].index(row["taxon"]); li = r.report.loci["names"].index(row["locus"])
            s2 = int(r.report.arr("cells/side2.npy")[g, li])
            assert (s2 & 3) == 3 and (s2 & 8) and not (s2 & 4)            # tested, past the bar, the divergent side called

    def test_one_direction_at_most(self, rc_run):
        """the call is the stronger side: no cell carries both broad bits, and the report's call matches them"""
        ds, r = rc_run; s2 = r.report.arr("cells/side2.npy")
        assert not np.any((s2 & 4) & ((s2 & 8) > 0))
        call = r.report.arr("cells/bs_call.npy"); rl = r.report.arr("cells/bs_locus.npy"); rt = r.report.arr("cells/bs_taxon.npy")
        for j in range(len(call)):
            v = int(s2[rt[j], rl[j]])
            if v & 4: assert call[j] == 1
            if v & 8: assert call[j] == -1

    def test_plain_contaminants_still_fall_to_the_main_test(self, rc_run):
        ds, r = rc_run; rows = masked_rows(r)
        for e in ds.ev("contam"):
            (row,) = [q for q in rows if (q["locus"], q["taxon"]) == (e["locus"], e["taxon"])]
            assert row["reason"] == "divergent" and not broad_masked(row)

    def test_unplanted_masks_are_collateral_at_event_loci(self, rc_run):
        """moving a clade orphans its sisters (their closest relatives all moved), so a few unplanted cells fail the
        main tests AT THE EVENT LOCI; the broad-sample test itself adds no unplanted mask"""
        ds, r = rc_run; rows = masked_rows(r)
        planted = {(e["locus"], t) for e in ds.events for t in (e.get("recipients") or e.get("taxa") or [e.get("taxon")]) + [e.get("donor")] if t}
        event_loci = {e["locus"] for e in ds.events if e["kind"] in ("cladecopy", "cladejunk", "contam")}
        stray = [q for q in rows if (q["locus"], q["taxon"]) not in planted]
        assert all(q["locus"] in event_loci for q in stray) and len(stray) <= 12
        assert not any(broad_masked(q) for q in stray)

    def test_the_family_includes_two_tests_per_present_sequence(self, rc_run):
        ds, r = rc_run; man = r.report.man
        t = man["test"]
        assert t["broad_sample_tests"] > 0 and t["tests"] == 2 * t["close_relative_tests"] + t["closest_match_outside_tests"] + t["broad_sample_tests"]
        assert man["broad_sample"]["tested"] > 0 and man["broad_sample"]["family"] == t["broad_sample_tests"]
        assert man["broad_sample"]["close"] >= 1 and man["broad_sample"]["divergent"] >= 1
        assert man["side2_bits"]["broad_close"] == 4 and man["side2_bits"]["broad_divergent"] == 8

    def test_report_carries_the_bs_members(self, rc_run):
        ds, r = rc_run
        for k in ("cells/side2.npy", "cells/bs_locus.npy", "cells/bs_taxon.npy", "cells/bs_lo_log10p.npy",
                  "cells/bs_hi_log10p.npy", "cells/bs_partner.npy", "cells/bs_call.npy", "cells/bs_var_log10p.npy"):
            assert k in r.report.names, k
        rl = r.report.arr("cells/bs_locus.npy"); assert len(rl) >= 5
        call = r.report.arr("cells/bs_call.npy"); assert set(np.unique(call)) <= {-1, 0, 1}


class TestCleanAndOff:
    def test_clean_data_raises_nothing(self, tmp_out):
        ds = dataset("dna_clean"); r = run_tool(ds.aln, tmp_out, "dna", ds.parts, []); assert r.rc == 0, r.log
        s2 = r.report.arr("cells/side2.npy")
        assert not np.any(s2 & (4 | 8 | 64)) and (s2 & 1).any()
        assert all(row["broad_sample_side"] == "" and row["broad_sample_partner"] == "" for row in masked_rows(r))

    def test_the_switch_turns_the_test_off(self, tmp_out):
        ds = dataset("dna_rc"); r = run_tool(ds.aln, tmp_out, "dna", ds.parts, ["--no-broad-sample"]); assert r.rc == 0, r.log
        assert r.report.man["test"]["broad_sample_tests"] == 0 and r.report.man["broad_sample"] is None
        assert "cells/bs_locus.npy" not in r.report.names
        assert not np.any(r.report.arr("cells/side2.npy"))
        assert all(row["broad_sample_side"] == "" and row["broad_sample_partner"] == "" for row in masked_rows(r))
        assert "broad sample:" not in r.log
