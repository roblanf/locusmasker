"""The URL hash as the page's state, in both directions, and the panel chrome."""
import pytest
from urllib.parse import unquote
from tests.viewer.conftest import open_report, cells_of_kind, lm_str

pytestmark = pytest.mark.viewer


class TestHash:
    def test_opens_a_taxon_with_a_view(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html, "t=3&dv=ribbons"); page.wait_for_selector(".panel[data-key=detail]"); page.wait_for_timeout(300)
        assert R.taxa["names"][3] in page.text_content(".panel[data-key=detail] .crumb") and page.eval_on_selector(".panel[data-key=detail] .crumb select", "e => e.value") == "ribbons"

    def test_opens_a_locus(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html, "l=2"); page.wait_for_selector(".panel[data-key=detail]"); page.wait_for_timeout(300)
        assert R.loci["names"][2] in page.text_content(".panel[data-key=detail] .crumb")

    def test_opens_a_cell(self, page, report_of):
        html, R = report_of("dna_small"); g, li = cells_of_kind(R)["divergent"]; open_report(page, html, f"c={g},{li}"); page.wait_for_selector(".panel[data-key=sequence] .seq-foot"); page.wait_for_timeout(200)
        crumb = page.text_content(".panel[data-key=sequence] .crumb"); assert R.taxa["names"][g] in crumb and R.loci["names"][li] in crumb

    def test_opens_a_view_and_an_order(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html, "view=table&o=name,length"); page.wait_for_timeout(400)
        assert page.evaluate("() => LM.App.view") == "table" and page.query_selector(".tfoot") is not None
        assert page.evaluate("() => [LM.App.order.taxaMode, LM.App.order.lociMode]") == ["name", "length"]

    def test_a_hash_from_another_dataset_is_ignored(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html, "d=0000000000&t=3"); page.wait_for_timeout(300)
        assert page.query_selector(".panel[data-key=detail]") is None

    def test_the_hash_follows_the_actions(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html)
        page.evaluate("() => LM.setView('matrix')"); page.wait_for_timeout(300); h = page.evaluate("() => location.hash")
        assert "view=matrix" in h and f"d={R.man['input']['sha256'][:10]}" in h
        page.evaluate("() => LM.openUnit('locus', 1)"); page.wait_for_timeout(400); assert "l=1" in page.evaluate("() => location.hash")
        g, li = cells_of_kind(R)["copy"]; page.evaluate(f"() => LM.openCell({g}, {li})"); page.wait_for_timeout(400); assert f"c={g},{li}" in unquote(page.evaluate("() => location.hash"))

    def test_copies_hash(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html, "cp=recipient"); page.wait_for_selector(".panel[data-key=detail]"); page.wait_for_timeout(300)
        assert page.eval_on_selector(".panel[data-key=detail] .crumb select", "e => e.value") == "recipient" and "copies" in page.text_content(".panel[data-key=detail] .crumb")


class TestPanelChrome:
    """the navigation stack: the deepest open level is the stage, every level above it a band; a band click,
    the ✕ or Esc go back up; ← → step between siblings at the open level"""
    def test_the_deepest_level_is_the_stage_and_its_ancestors_are_bands(self, page, report_of):
        html, R = report_of("dna_small"); g, li = cells_of_kind(R)["divergent"]; open_report(page, html)
        shown = lambda key: page.evaluate(f"() => getComputedStyle(document.querySelector('.panel[data-key={key}]')).display !== 'none'")
        assert shown("overview") and page.query_selector_all(".bands .band") == []
        page.evaluate(f"() => LM.openUnit('taxon', {g})"); page.wait_for_selector(".panel[data-key=detail]"); page.wait_for_timeout(150)
        assert shown("detail") and not shown("overview")
        bands = page.query_selector_all(".bands .band"); assert [b.get_attribute("data-level") for b in bands] == ["dataset"]
        assert page.query_selector(".band[data-level=dataset] canvas.mini") is not None          # the matrix thumbnail
        page.evaluate(f"() => LM.openCell({g}, {li})"); page.wait_for_selector(".panel[data-key=sequence] .seq-foot"); page.wait_for_timeout(150)
        assert shown("sequence") and not shown("detail") and not shown("overview")
        bands = page.query_selector_all(".bands .band"); assert [b.get_attribute("data-level") for b in bands] == ["dataset", "unit"]
        assert R.taxa["names"][g] in page.text_content(".band[data-level=unit]")
        page.click(".band[data-level=unit]"); page.wait_for_timeout(200)                          # back to the unit
        assert page.query_selector(".panel[data-key=sequence]") is None and shown("detail")
        page.click(".panel[data-key=detail] .tools button"); page.wait_for_timeout(200)           # the ✕: back to the dataset
        assert page.query_selector(".panel[data-key=detail]") is None and shown("overview") and page.query_selector_all(".bands .band") == []

    def test_escape_goes_up_and_the_arrows_step_between_siblings(self, page, report_of):
        html, R = report_of("dna_small"); g, li = cells_of_kind(R)["divergent"]; open_report(page, html)
        import numpy as np
        side = R.arr("cells/side.npy"); masked_loci = [int(x) for x in np.nonzero((side[g] & 3) > 0)[0]]
        page.evaluate(f"() => LM.openUnit('taxon', {g})"); page.wait_for_selector(".panel[data-key=detail]")
        page.evaluate(f"() => LM.openCell({g}, {li})"); page.wait_for_selector(".panel[data-key=sequence] .seq-foot"); page.wait_for_timeout(150)
        if len(masked_loci) > 1:                                                                   # → the next masked cell of the open taxon
            page.keyboard.press("ArrowRight"); page.wait_for_timeout(300)
            cur = page.evaluate("() => location.hash.match(/c=(\\d+)%2C(\\d+)/).slice(1).map(Number)")
            k = masked_loci.index(li); assert cur == [g, masked_loci[min(k + 1, len(masked_loci) - 1)]]
        page.keyboard.press("Escape"); page.wait_for_timeout(200)
        assert page.query_selector(".panel[data-key=sequence]") is None and page.query_selector(".panel[data-key=detail]") is not None
        page.keyboard.press("Escape"); page.wait_for_timeout(200)
        assert page.query_selector(".panel[data-key=detail]") is None

    def test_the_sequence_panel_closes_and_reopens(self, page, report_of):
        html, R = report_of("dna_small"); g, li = cells_of_kind(R)["divergent"]; open_report(page, html)
        page.evaluate(f"() => LM.openCell({g}, {li})"); page.wait_for_selector(".panel[data-key=sequence] .seq-foot")
        page.click(".panel[data-key=sequence] .tools button"); page.wait_for_timeout(200); assert page.query_selector(".panel[data-key=sequence]") is None
        page.evaluate(f"() => LM.openCell({g}, {li})"); page.wait_for_selector(".panel[data-key=sequence] .seq-foot")


class TestOlderFormatPanels:
    def test_format5_alignment_has_no_p_column_and_says_not_recorded(self, page, report_of):
        html, R = report_of("dna_small", "format5"); kinds = cells_of_kind(report_of("dna_small")[1])
        open_report(page, html); g, li = kinds["clean"]
        page.evaluate(f"() => LM.openCell({g}, {li})"); page.wait_for_selector(".panel[data-key=sequence] .seq-foot"); page.wait_for_timeout(200)
        assert page.query_selector(".seq-p") is None
        names = page.eval_on_selector_all(".seq-names .seqnm .nm", "els => els.map(e => e.textContent)"); assert names[1] == lm_str(page, "seq.name_not_recorded")
        g, li = kinds["copy"]; page.evaluate(f"() => LM.openCell({g}, {li})"); page.wait_for_timeout(400)
        names = page.eval_on_selector_all(".seq-names .seqnm .nm", "els => els.map(e => [e.textContent, e.className])"); assert "st-blue" in names[1][1]
        page.wait_for_selector(".trees-pane svg", timeout=15000)

    def test_format3_opens_the_panels(self, page, report_of):
        html, R = report_of("dna_small", "format3"); kinds = cells_of_kind(report_of("dna_small")[1]); open_report(page, html)
        page.evaluate("() => LM.openUnit('taxon', 2)"); page.wait_for_selector(".panel[data-key=detail]")
        g, li = kinds["divergent"]; page.evaluate(f"() => LM.openCell({g}, {li})"); page.wait_for_selector(".panel[data-key=sequence] .seq-foot")
        assert page.text_content(".seq-line .state").startswith(lm_str(page, "seq.v_masked"))
