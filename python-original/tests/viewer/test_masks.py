"""The masks bar: one filter (the five test arms, a taxon, a locus) and one order behind the table and the ← → paging.
Structure and counts are checked, never the words."""
import numpy as np
import pytest
from tests.viewer.conftest import open_report, cells_of_kind

pytestmark = pytest.mark.viewer


def arms(R):
    """every masked cell's test arms, from the report's bits as the viewer's maskKind reads them: close-relative when the
    divergent mask holds a majority, closest-match when the similar mask has a partner (the arm is the viewer's call),
    broad-sample close or divergent from side2 unless another mask already covered the cell"""
    side = R.arr("cells/side.npy"); nf = R.arr("cells/n_fail.npy"); nc = R.arr("cells/n_comp.npy"); ns = R.arr("cells/n_similar.npy")
    s2 = R.arr("cells/side2.npy") if R.has("cells/side2.npy") else np.zeros_like(side)
    out = {}
    for g, li in zip(*np.nonzero((side & 3) > 0)):
        a = set(); sd = int(side[g, li]); s2v = int(s2[g, li])
        if sd & 1 and (nf[g, li] >= R.M if nc[g, li] == R.K else nf[g, li] >= (nc[g, li] >> 1) + 1): a.add("close_relative")
        if sd & 2 and ns[g, li] > 0: a.add("closest_match")
        if s2v & 4 and not s2v & 32: a.add("bs_close")
        if s2v & 8 and not s2v & 32: a.add("bs_divergent")
        out[(int(g), int(li))] = a
    return out


class TestMasksBar:
    def test_the_bar_lists_every_mask_and_the_chips_narrow_it(self, page, report_of):
        html, R = report_of("dna_rc"); open_report(page, html); A = arms(R)
        n = page.evaluate("() => LM.masks.list().length"); assert n == R.man["counts"]["masked"] == len(A)
        assert page.evaluate("() => document.querySelector('.maskbar').dataset.n") == str(n)
        chips = page.eval_on_selector_all(".maskbar .chip", "els => els.map(e => e.dataset.arm)"); assert chips == ["all", "close_relative", "cm_relative", "cm_outside", "bs_close", "bs_divergent"]
        assert page.eval_on_selector_all(".maskbar .chip.on", "els => els.map(e => e.dataset.arm)") == ["all"]
        page.click(".maskbar .chip[data-arm=bs_close]"); page.wait_for_timeout(150)                          # the first click keeps that arm alone
        assert page.eval_on_selector_all(".maskbar .chip.on", "els => els.map(e => e.dataset.arm)") == ["bs_close"]
        assert page.evaluate("() => LM.masks.list().length") == sum(1 for a in A.values() if "bs_close" in a)
        page.click(".maskbar .chip[data-arm=bs_divergent]"); page.wait_for_timeout(150)                      # a second click adds an arm
        assert page.evaluate("() => LM.masks.list().length") == sum(1 for a in A.values() if a & {"bs_close", "bs_divergent"})
        page.click(".maskbar .chip[data-arm=close_relative]"); page.wait_for_timeout(150)
        got = set(map(tuple, page.evaluate("() => LM.masks.list().map(q => [q.g, q.li])")))
        assert got == {k for k, a in A.items() if a & {"bs_close", "bs_divergent", "close_relative"}}
        page.evaluate("() => LM.masks.setFamily('closest_match')"); page.wait_for_timeout(150)               # the two closest-match arms together
        assert page.evaluate("() => LM.masks.list().length") == sum(1 for a in A.values() if "closest_match" in a)
        page.click(".maskbar .chip[data-arm=all]"); page.wait_for_timeout(150)
        assert page.evaluate("() => LM.masks.list().length") == n

    def test_the_table_lists_the_bar_and_the_summary_cards_set_it(self, page, report_of):
        html, R = report_of("dna_rc"); open_report(page, html); A = arms(R)
        page.click(".cards .card >> nth=0"); page.wait_for_selector(".trow"); page.wait_for_timeout(200)     # the close-relative card: the table, narrowed
        assert page.eval_on_selector_all(".maskbar .chip.on", "els => els.map(e => e.dataset.arm)") == ["close_relative"]
        assert page.evaluate("() => document.querySelectorAll('.trow').length") == sum(1 for a in A.values() if "close_relative" in a) == page.evaluate("() => LM.masks.list().length")
        assert page.query_selector(".tfoot .chip") is None                                                   # the chips live in the bar, not the footer
        page.click(".maskbar .chip[data-arm=all]"); page.wait_for_timeout(200)
        assert page.evaluate("() => document.querySelectorAll('.trow').length") == len(A)                   # the table follows the bar

    def test_a_taxon_or_a_locus_narrows_the_list_by_name_or_from_its_panel(self, page, report_of):
        html, R = report_of("dna_rc"); open_report(page, html); A = arms(R)
        g = max({k[0] for k in A}, key=lambda t: sum(1 for k in A if k[0] == t)); name = R.taxa["names"][g]
        page.fill(".mpick[data-kind=taxon] input", name[:3]); page.wait_for_timeout(200)
        assert page.evaluate("() => document.querySelector('.mpick[data-kind=taxon] .dd').style.display") == "block"
        page.click(f".mpick[data-kind=taxon] .dd button:text-is('{name}')"); page.wait_for_timeout(200)
        assert page.evaluate("() => LM.masks.list().length") == sum(1 for k in A if k[0] == g) and page.input_value(".mpick[data-kind=taxon] input") == name
        assert "set" in page.evaluate("() => document.querySelector('.mpick[data-kind=taxon]').className")
        page.click(".mpick[data-kind=taxon] .mpick-x"); page.wait_for_timeout(200); assert page.evaluate("() => LM.masks.list().length") == len(A)
        li = max({k[1] for k in A}, key=lambda l: sum(1 for k in A if k[1] == l))
        page.evaluate(f"() => LM.openUnit('locus', {li})"); page.wait_for_selector(".panel[data-key=detail] .crumb button"); page.wait_for_timeout(200)
        page.click(".panel[data-key=detail] .crumb button >> nth=0"); page.wait_for_timeout(200)             # the panel's own button: its masks
        assert page.evaluate("() => LM.masks.list().length") == sum(1 for k in A if k[1] == li) and page.input_value(".mpick[data-kind=locus] input") == R.loci["names"][li]

    def test_the_arrows_page_the_list_in_its_order_and_the_bar_shows_the_place(self, page, report_of):
        html, R = report_of("dna_rc"); open_report(page, html); A = arms(R)
        page.evaluate("() => LM.masks.setOrder('locus')"); page.wait_for_timeout(100)
        L = page.evaluate("() => LM.masks.list().map(q => [q.g, q.li])"); assert len(L) == len(A)
        assert [tuple(x) for x in L] == sorted(A, key=lambda k: (k[1], k[0]))                                 # by locus, then taxon, in input order
        assert page.text_content(".maskbar .pos .k").startswith("–")                                         # no sequence open: not in the list
        page.keyboard.press("ArrowRight"); page.wait_for_selector(".panel[data-key=sequence] .seq-foot", timeout=15000); page.wait_for_timeout(200)
        assert page.evaluate("() => { const c = LM.App.panels.sequence && LM.masks.current(); return c; }") == 0   # → from nowhere: the first
        page.keyboard.press("ArrowRight"); page.wait_for_function("() => LM.masks.current() === 1", timeout=15000)
        page.keyboard.press("ArrowLeft"); page.wait_for_function("() => LM.masks.current() === 0", timeout=15000)
        page.keyboard.press("ArrowLeft"); page.wait_for_timeout(300); assert page.evaluate("() => LM.masks.current()") == 0   # the list has an edge
        page.click(".maskbar .pos .step >> nth=1"); page.wait_for_function("() => LM.masks.current() === 1", timeout=15000)   # the bar's own arrow
        assert "2" in page.text_content(".maskbar .pos .k")
        page.evaluate("() => LM.masks.setOrder('p')"); page.wait_for_timeout(100)
        P = page.evaluate("() => LM.masks.list().map(q => q.p)"); assert P == sorted(P)                         # most significant first

    def test_the_filter_survives_the_link(self, page, report_of):
        html, R = report_of("dna_rc"); open_report(page, html); A = arms(R); g = next(iter(A))[0]
        page.evaluate(f"() => {{ LM.masks.setTests(['bs_close', 'bs_divergent']); LM.masks.setUnit('taxon', {g}); LM.masks.setOrder('taxon'); }}"); page.wait_for_timeout(300)
        h = page.evaluate("() => location.hash"); assert "mf=" in h
        open_report(page, html, h.lstrip("#")); page.wait_for_timeout(300)
        assert page.eval_on_selector_all(".maskbar .chip.on", "els => els.map(e => e.dataset.arm)") == ["bs_close", "bs_divergent"]
        assert page.input_value(".mpick[data-kind=taxon] input") == R.taxa["names"][g] and page.evaluate("() => document.querySelector('.maskbar select').value") == "taxon"
