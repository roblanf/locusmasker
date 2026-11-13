"""The taxon and locus panel: the crumb, the two views (test results, site ribbons) and the copies blocks, against the report."""
import numpy as np
import pytest
from tests.viewer.conftest import open_report, hover_text, lm_str, lm_fmt

pytestmark = pytest.mark.viewer


def most_masked_taxon(R):
    side = R.arr("cells/side.npy"); return int(np.argmax(((side & 3) > 0).sum(1)))


class TestTaxonPanel:
    def test_crumb_and_views(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html); g = most_masked_taxon(R)
        page.evaluate(f"() => LM.openUnit('taxon', {g})"); page.wait_for_selector(".panel[data-key=detail]"); page.wait_for_timeout(300)
        crumb = page.text_content(".panel[data-key=detail] .crumb"); side = R.arr("cells/side.npy"); nc = R.arr("cells/n_comp.npy")
        assert R.taxa["names"][g] in crumb and f"{int(((side[g] & 3) > 0).sum())} of {int((nc[g] > 0).sum())} sequences masked" in crumb
        opts = page.eval_on_selector_all(".panel[data-key=detail] .crumb select option", "els => els.map(e => e.value)"); assert opts == ["results", "ribbons"]

    def test_results_view_is_the_units_card_live(self, page, report_of):
        """the default view: one ribbon per test arm on the dataset-wide axes with the unit's cells, the ten strongest per
        test numbered, the mask counts in the titles, the full axis captions"""
        html, R = report_of("dna_small"); open_report(page, html); g = most_masked_taxon(R)
        page.evaluate(f"() => LM.openUnit('taxon', {g})"); page.wait_for_selector(".panel[data-key=detail] .cd-rib canvas"); page.wait_for_timeout(400)
        assert page.eval_on_selector(".panel[data-key=detail] .crumb select", "e => e.value") == "results"
        nrib = 4 if R.has("cells/bs_stat.npy") else 3
        assert len(page.query_selector_all(".panel[data-key=detail] .cd-rib")) == nrib
        painted = page.evaluate("() => { const c = document.querySelector('.panel[data-key=detail] .cd-rib canvas'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n; }")
        assert painted > 50
        assert page.query_selector_all(".panel[data-key=detail] .cd-ribrow") == []            # just the plots: no site ribbons here
        axes = page.eval_on_selector_all(".panel[data-key=detail] .cd-axis text", "els => els.map(e => e.textContent)")
        assert any(lm_str(page, "ribbons.axis_close_relative") == t for t in axes) and any(lm_str(page, "ribbons.axis_closest") == t for t in axes)   # the full axis captions, as on the card
        counts = page.eval_on_selector_all(".panel[data-key=detail] .cd-rib .cd-ribn", "els => els.map(e => +e.textContent.replace(/,/g, ''))")
        assert len(counts) == nrib and sum(counts) >= int(((side[g] & 3) > 0).sum()) if (side := R.arr("cells/side.npy")) is not None else True   # the little titles carry the mask counts
        page.hover(".panel[data-key=detail] .help-q"); page.wait_for_timeout(200); assert lm_str(page, "unit.note_results_head") in (hover_text(page) or "")

    def test_ribbons_list_the_masked_cells(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html); g = most_masked_taxon(R)
        page.evaluate(f"() => LM.openUnit('taxon', {g})"); page.wait_for_selector(".panel[data-key=detail]"); page.wait_for_timeout(200)
        page.select_option(".panel[data-key=detail] .crumb select", "ribbons"); page.wait_for_timeout(500)
        rows = page.query_selector_all(".panel[data-key=detail] .rib-row2"); side = R.arr("cells/side.npy")
        assert len(rows) == int(((side[g] & 3) > 0).sum())
        nrib = 4 if R.has("cells/bs_stat.npy") else 3
        pcols = page.eval_on_selector_all(".panel[data-key=detail] .rib-row2 .pcol", "els => els.map(e => e.textContent.trim())")
        assert len(pcols) == 4 * len(rows) and all(c == "\u2013" or c[0].isdigit() for c in pcols)                 # four p columns, one per test arm
        assert len(page.query_selector_all(".panel[data-key=detail] .rib-head2 .pcol")) == 4
        rows[0].click(); page.wait_for_selector(".panel[data-key=sequence]"); assert R.taxa["names"][g] in page.text_content(".panel[data-key=sequence] .crumb")

    def test_locus_panel_has_the_same_two_views(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html); side = R.arr("cells/side.npy"); li = int(np.argmax(((side & 3) > 0).sum(0)))
        page.evaluate(f"() => LM.openUnit('locus', {li})"); page.wait_for_selector(".panel[data-key=detail] .cd-rib canvas"); page.wait_for_timeout(300)
        assert R.loci["names"][li] in page.text_content(".panel[data-key=detail] .crumb")
        assert page.eval_on_selector_all(".panel[data-key=detail] .crumb select option", "els => els.map(e => e.value)") == ["results", "ribbons"]
        assert len(page.query_selector_all(".panel[data-key=detail] .cd-rib")) == (4 if R.has("cells/bs_stat.npy") else 3)


class TestCopies:
    def test_blocks_by_donor_recipient_and_pair(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html, "cp=donor"); page.wait_for_selector(".panel[data-key=detail]"); page.wait_for_timeout(400)
        side = R.arr("cells/side.npy"); ncop = int((((side & 2) > 0) & ((side & 8) == 0)).sum())
        assert lm_fmt(lm_str(page, "unit.copies_n"), n=ncop) in page.text_content(".panel[data-key=detail] .crumb")
        rules = page.eval_on_selector_all(".panel[data-key=detail] .rule", "els => els.map(e => e.textContent)"); assert rules and all(r.strip() for r in rules)
        counts = [int(x) for x in page.eval_on_selector_all(".panel[data-key=detail] .rule", "els => els.map(e => e.dataset.n)")]; assert counts == sorted(counts, reverse=True) and sum(counts) == ncop
        rows = page.query_selector_all(".panel[data-key=detail] .rib-row2"); assert len(rows) == ncop
        for mode in ("recipient", "pair"):
            page.select_option(".panel[data-key=detail] .crumb select", mode); page.wait_for_timeout(400)
            assert len(page.query_selector_all(".panel[data-key=detail] .rib-row2")) == ncop and page.evaluate("() => location.hash").find(f"cp={mode}") >= 0

    def test_a_single_donor_block_by_the_api(self, page, report_of):
        """the copies view holds (reachable by the API and the chord view to come; the summary's bar lists are gone)"""
        html, R = report_of("dna_small"); open_report(page, html)
        side = R.arr("cells/side.npy"); pt = R.arr("cells/partner.npy")
        donors = {}
        for g in range(R.n):
            for li in range(R.nloc):
                if side[g, li] & 2 and not side[g, li] & 8: donors[int(pt[g, li])] = donors.get(int(pt[g, li]), 0) + 1
        assert donors, "the fixture has no called copy"
        d, n = max(donors.items(), key=lambda kv: kv[1])
        page.evaluate(f"() => LM.openCopies('donor', '{d}')"); page.wait_for_selector(".panel[data-key=detail]"); page.wait_for_timeout(400)
        crumb = page.text_content(".panel[data-key=detail] .crumb"); assert R.taxa["names"][d] in crumb and lm_fmt(lm_str(page, "unit.copies_of"), k=n, n=sum(donors.values())) in crumb
        assert len(page.query_selector_all(".panel[data-key=detail] .rib-row2")) == n
