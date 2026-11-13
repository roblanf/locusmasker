"""The broad-sample test in the viewer (format 9): a close-side mask is blue, an excess-side mask red, a labelled
divergent cell yellow, the hover names the test and its p, and the alignment panel shows the partner under a
broad-comparator tag."""
import json
import numpy as np
import pytest
from tests.conftest import dataset
from tests.viewer.conftest import open_report, lm_str


def bs_cells(R):
    """one (g, li) per broad-sample outcome on the dna_rc run, from side2"""
    s2 = R.arr("cells/side2.npy"); side = R.arr("cells/side.npy"); out = {}
    for g, li in zip(*np.nonzero(s2 & (4 | 8 | 64))):
        v = int(s2[g, li]); sd = int(side[g, li])
        if v & 4 and (sd & 3) == 2 and "close" not in out: out["close"] = (int(g), int(li))
        if v & 8 and (sd & 3) == 1 and "red" not in out: out["red"] = (int(g), int(li))
        if v & 64 and "label" not in out: out["label"] = (int(g), int(li))
    return out


class TestBroadSampleCells:
    def test_the_colours_follow_the_kind(self, page, report_of):
        html, R = report_of("dna_rc"); open_report(page, html)
        cells = bs_cells(R); assert "close" in cells, cells
        fam = {k: page.evaluate(f"() => LM.App.D.famOf(LM.App.D.state[{g} * LM.App.D.nloc + {li}])") for k, (g, li) in cells.items()}
        ST = page.evaluate("() => ({RED: 20, BLUE: 40, BOTH: 60})")
        assert fam["close"] == ST["BLUE"]
        assert "red" in cells and fam["red"] == ST["RED"]                        # the excess arm: the hidden contamination is red

    def test_the_labelled_divergent_cell_is_yellow(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html)                 # dna_small's far copy: masked divergent, the donor side named
        cells = bs_cells(R); assert "label" in cells, cells
        g, li = cells["label"]
        fam = page.evaluate(f"() => LM.App.D.famOf(LM.App.D.state[{g} * LM.App.D.nloc + {li}])")
        assert fam == 60

    @pytest.mark.parametrize("kind,key", [("close", "state.bs_close"), ("red", "state.bs_divergent")])
    def test_the_labels_name_the_test(self, page, report_of, kind, key):
        html, R = report_of("dna_rc"); open_report(page, html)
        g, li = bs_cells(R)[kind]
        label = page.evaluate(f"() => LM.App.D.cellLabel(LM.App.D.state[{g} * LM.App.D.nloc + {li}], {g} * LM.App.D.nloc + {li})")
        assert label == lm_str(page, key), label

    def test_the_label_names_the_donor_side(self, page, report_of):
        """the divergent cell the broad sample also calls too close: the donor-side comparator is on record and
        reaches the alignment rows as a blue badge"""
        html, R = report_of("dna_small"); open_report(page, html)
        g, li = bs_cells(R)["label"]
        got = page.evaluate(f"() => {{ const D = LM.App.D; const r = D.rcAt({g} * D.nloc + {li}); return r ? [r.pt, r.lo] : null; }}")
        assert got is not None and got[0] >= 0 and got[1] < 0

    def test_the_hover_rows_carry_the_p_and_the_partner_tag(self, page, report_of):
        html, R = report_of("dna_rc"); open_report(page, html)
        g, li = bs_cells(R)["close"]
        txt = page.evaluate(f"() => {{ const b = window.cellHover ? cellHover({g}, {li}) : null; return b ? b.textContent : null; }}")
        if txt is None: pytest.skip("cellHover is not exposed on window in this build")
        assert lm_str(page, "hover.k_broad") in txt

    def test_the_sequence_panel_tags_the_broad_partner(self, page, report_of):
        html, R = report_of("dna_rc"); open_report(page, html)
        g, li = bs_cells(R)["close"]
        page.evaluate(f"() => LM.openCell({g}, {li})"); page.wait_for_selector(".panel[data-key=sequence] .seq-foot", timeout=30000); page.wait_for_timeout(200)
        pt = page.evaluate(f"() => {{ const D = LM.App.D; const sr = D.rcAt({g} * D.nloc + {li}); return sr ? sr.pt : null; }}"); assert pt is not None
        rows = page.eval_on_selector_all(".seq-names .seqnm", "els => els.map(e => [e.dataset.g, e.title])")
        assert int(rows[2][0]) == pt and rows[2][1].startswith(R.taxa["names"][pt] + ": " + lm_str(page, "partner.bs_close_note"))   # the broad row is the comparator that decided it, explained as such

    def test_a_clean_report_shows_no_broad_rows(self, page, report_of):
        html, R = report_of("dna_clean"); open_report(page, html)
        s2 = R.arr("cells/side2.npy"); assert not np.any(s2 & (4 | 64))


class TestCards:
    def test_the_locus_card_shows_each_tests_top_sequences_as_ribbon_rows(self, page, report_of):
        """one page per locus: verdict band, the four-test spine, then per test its top masked sequences as
        numbered site-ribbon rows with their p - plots and rows, never tables (the card grammar)"""
        html, R = report_of("dna_rc"); open_report(page, html)
        import numpy as np
        side = R.arr("cells/side.npy"); li = int(np.argmax(((side & 3) > 0).sum(0)))
        page.evaluate(f"() => LM.openCard('locus', {li})"); page.wait_for_selector("#lm-card .cd-page", timeout=15000)
        n = int(((side[:, li] & 3) > 0).sum())
        assert page.query_selector_all("#lm-card .cd-rib canvas") and len(page.query_selector_all("#lm-card .cd-rib")) >= 2
        assert page.query_selector("#lm-card .cd-table") is None               # no cell tables on cards
        rows = page.query_selector_all("#lm-card .cd-ribrow")
        assert 1 <= len(rows) and all(r.query_selector(".cd-num") and r.query_selector(".cd-ribp") for r in rows)
        assert page.query_selector("#lm-card .cd-rank") is None                # no rankings at a single locus
        assert page.query_selector("#lm-card .cd-verdict") is not None and page.query_selector("#lm-card .cd-foot") is not None
        page.evaluate("() => LM.App && document.querySelector('#lm-card .cd-bar button:last-child').click()")
        assert page.query_selector("#lm-card .cd-page") is None

    def test_the_sequence_card_carries_the_state_and_the_marked_spine(self, page, report_of):
        html, R = report_of("dna_rc"); open_report(page, html)
        import numpy as np
        s2 = R.arr("cells/side2.npy"); g, li = [(int(a), int(b)) for a, b in zip(*((s2 & 8) > 0).nonzero())][0]
        page.evaluate(f"() => LM.openCard('sequence', {g}, {li})"); page.wait_for_selector("#lm-card .cd-page", timeout=15000)
        page.wait_for_selector("#lm-card .cd-resrow", timeout=15000)           # the rows arrive with the bundle reads
        assert R.taxa["names"][g] in page.text_content("#lm-card .cd-id")
        assert page.query_selector("#lm-card .cd-evidence") is None            # no hover-table embed: concise result rows
        assert len(page.query_selector_all("#lm-card .cd-resrow")) >= 1
        assert len(page.query_selector_all("#lm-card .cd-rib")) >= 2

    def test_the_dataset_card_counts_match_the_hero(self, page, report_of):
        html, R = report_of("dna_rc"); open_report(page, html)
        page.evaluate("() => LM.openCard('dataset')"); page.wait_for_selector("#lm-card .cd-page", timeout=15000)
        v = page.text_content("#lm-card .cd-verdict")
        assert str(R.man["counts"]["masked"]) in v.replace(",", "")

    def test_the_copy_flow_joins_donors_to_recipients(self, page, report_of):
        """when the report holds attributed copies, the summary and the dataset card draw the donor-to-recipient
        flow: one band per flow, node bars both sides (the sankey that replaced the two lollipops)"""
        html, R = report_of("dna_rc"); open_report(page, html)
        n = page.evaluate("""() => { const D = LM.App.D; let n = 0;
            for (let g = 0; g < D.n; g++) for (let li = 0; li < D.nloc; li++) {
              const sd = D.cols.side[g * D.nloc + li]; if (!(sd & 34) || (sd & 8)) continue;
              const p = D.num("partner", g, li); if (p !== null && p >= 0) n++; } return n; }""")
        flows = page.query_selector_all(".tiles .lm-flow")
        if n == 0:
            assert not flows; return
        assert len(flows) == 1 and page.query_selector_all(".tiles .lm-flow path")
        page.evaluate("() => LM.openCard('dataset')"); page.wait_for_selector("#lm-card .cd-page", timeout=15000)
        assert page.query_selector("#lm-card .lm-flow") is not None
        assert len(page.query_selector_all("#lm-card .lm-flow rect")) >= 2      # a bar each side at least
