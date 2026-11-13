"""The overview panel: the summary's tiles and lists against the report, the matrix with its hover, click, zoom and
order controls, the ribbons and table views, and the downloads."""
import re
import numpy as np
import pytest
from tests.viewer.conftest import open_report, hover_text, lm_str, lm_fmt

pytestmark = pytest.mark.viewer


def masked_share(R):
    side = R.arr("cells/side.npy"); nc = R.arr("cells/n_comp.npy"); masked = (side & 3) > 0
    return masked.sum(1) / np.maximum((nc > 0).sum(1), 1), masked.sum(0), masked


class TestSummary:
    def test_hero_and_cards_are_the_manifest_counts(self, page, report_of):
        """the landing: the hero holds the masked total, one card per test ARM in the canonical order (the
        closest-match test splits into its relative and outside arms; their counts sum to the test's)"""
        html, R = report_of("dna_small"); open_report(page, html); c = R.man["counts"]
        share, per_locus, masked = masked_share(R)
        assert page.text_content(".topbar .sub").replace(",", "").startswith(str(c["masked"]) + " of ")   # the masked total is the top bar's, not repeated
        bs = R.man.get("broad_sample") or {}
        bsN = (bs.get("close") or 0) + (bs.get("divergent") or 0) + (bs.get("both_sides") or 0)
        cards = [int(v.replace(",", "")) for v in page.eval_on_selector_all(".cards .card .card-n", "els => els.map(e => e.textContent)")]
        n_cards = 4 if R.man.get("broad_sample") else 3
        assert len(cards) == n_cards
        assert cards[0] == c["masked_divergent"] - (bs.get("divergent") or 0)
        assert cards[1] + cards[2] == c["masked_similar"] - (bs.get("close") or 0)
        if R.man.get("broad_sample"): assert cards[3] == bsN
        stats = page.text_content(".hero-stats")
        assert f"{int((masked.sum(1) > 0).sum())} / {c['taxa']}" in stats and f"{int((per_locus > 0).sum())} / {c['loci']}" in stats

    def test_most_masked_taxa_in_order(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html); share, _, masked = masked_share(R)
        names = page.evaluate("""() => { const c = document.querySelector('.worst .worst-row:nth-child(1) .col:nth-child(1)');
            const rows = Array.from(c.querySelectorAll('.row .nm')).map(e => e.textContent);
            return rows.length ? rows : Array.from(c.querySelectorAll('.chiprow .chip')).map(e => e.textContent.trim().split(' ·')[0]); }""")
        want = [R.taxa["names"][g] for g in sorted(np.flatnonzero(masked.sum(1) > 0), key=lambda g: -share[g])][:10]
        assert len(names) == len(want) and sorted(names) == sorted(want)         # ties render as chips, magnitudes as bars
        got_share = [share[R.tx(n)] for n in names]; assert got_share == sorted(got_share, reverse=True)
        labels = page.eval_on_selector_all(".worst .worst-row:nth-child(1) .col:nth-child(1) .row .num", "els => els.map(e => e.textContent)")
        assert all(re.fullmatch(r"[\d.]+% \([\d,]+ masks?\)", l) for l in labels), labels                        # "{pct} ({n} masks)"

    def test_most_masked_loci_in_order(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html); _, per_locus, _ = masked_share(R)
        names = page.eval_on_selector_all(".worst .worst-row:nth-child(1) .col:nth-child(2) .row .nm", "els => els.map(e => e.textContent)")
        labels = page.eval_on_selector_all(".worst .worst-row:nth-child(1) .col:nth-child(2) .row .num", "els => els.map(e => e.textContent)")
        nums = [int(re.search(r"\((\d[\d,]*) masks?\)", l).group(1).replace(",", "")) for l in labels]
        tested = (R.arr("cells/n_comp.npy") > 0).sum(0); share = [per_locus[R.lx(n)] / tested[R.lx(n)] for n in names]   # ranked by the share of tested taxa
        assert share == sorted(share, reverse=True) and nums == [int(per_locus[R.lx(n)]) for n in names] and len(names) == int((per_locus > 0).sum())

    def test_the_summary_has_no_donor_bar_lists(self, page, report_of):
        """the donor/recipient/pair bar lists are gone from the summary (Rob, 3 Nov: the information returns as the
        chord view); the two most-masked lists remain"""
        html, R = report_of("dna_small"); open_report(page, html)
        heads = page.eval_on_selector_all(".worst h3", "els => els.map(e => e.textContent)")
        assert len(page.query_selector_all(".worst .worst-row:nth-child(1) .col")) == 2
        assert not any("donor" in h for h in heads)

    def test_histograms_count_the_masked_units(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html)
        svgs = page.query_selector_all(".hist-wrap svg"); assert len(svgs) == 2              # taxa, loci
        totals = page.eval_on_selector_all(".hist-wrap svg", "els => els.map(s => Array.from(s.querySelectorAll(':scope > text')).map(t => t.textContent.replace(/,/g, '')).filter(t => /^\\d+$/.test(t)).map(Number))")
        side = R.arr("cells/side.npy")
        assert sum(totals[0]) == int(((side & 3) > 0).any(1).sum())                          # taxa holding a mask: the zero bin is the subtitle's job
        assert sum(totals[1]) == int(((side & 3) > 0).any(0).sum())                          # loci holding a mask

    def test_a_list_row_opens_the_detail_panel(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html)
        sel = page.query_selector(".worst .worst-row:nth-child(1) .col:nth-child(1) .row .nm") and ".worst .worst-row:nth-child(1) .col:nth-child(1) .row" or ".worst .worst-row:nth-child(1) .col:nth-child(1) .chiprow .chip"
        name = page.text_content(sel + (" .nm" if ".row" == sel[-4:] else "")).strip().split(" ·")[0]; page.click(sel); page.wait_for_selector(".panel[data-key=detail]")
        page.wait_for_function("() => document.querySelector('.panel[data-key=detail] .crumb').textContent.length > 0")   # the unit's chunk arrives first
        assert name in page.text_content(".panel[data-key=detail] .crumb")


class TestMatrix:
    def test_matrix_draws_and_the_views_switch(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html)
        assert page.eval_on_selector_all(".panel[data-key=overview] select option", "els => els.map(e => e.value)")[:5] == ["summary", "matrix", "loci", "taxa", "table"]   # the overview's own view select (the masks bar has one too)
        page.select_option(".viewctl select, .panel[data-key=overview] select >> nth=0", "matrix"); page.wait_for_timeout(300)
        nonblank = page.evaluate("() => { const c = document.querySelector('canvas.matrix'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return [n, c.width, c.height]; }")
        assert nonblank[0] > 0.5 * nonblank[1] * nonblank[2]

    def test_hover_names_the_cell_and_click_opens_the_alignment(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html); page.evaluate("() => LM.setView('matrix')"); page.wait_for_timeout(300)
        box = page.evaluate("() => { const c = document.querySelector('canvas.matrix'); const r = c.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }")
        V = page.evaluate("() => { const V = LM.App.D && window.LM.App.panels.overview ? null : null; return null; }")
        page.mouse.move(box[0] + 20, box[1] + 10); page.wait_for_timeout(200)
        txt = hover_text(page); assert txt and " · " in txt and any(n in txt for n in R.taxa["names"]) and any(l in txt for l in R.loci["names"])
        page.mouse.click(box[0] + 20, box[1] + 10); page.wait_for_selector(".panel[data-key=sequence] .seq-head", timeout=10000)   # the stage fills once its chunk arrives
        assert page.query_selector(".seq-head") is not None

    def test_order_controls_reorder_the_rows(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html); page.evaluate("() => LM.setView('matrix')"); page.wait_for_timeout(200)
        before = page.evaluate("() => LM.App.order.taxa.slice()")
        page.evaluate("() => { const s = document.querySelectorAll('.viewctl select, .panel[data-key=overview] select'); for (const x of s) if (Array.from(x.options).some(o => o.value === 'masked')) { x.value = 'masked'; x.dispatchEvent(new Event('change')); break; } }"); page.wait_for_timeout(200)
        after = page.evaluate("() => LM.App.order.taxa.slice()"); share, _, _ = masked_share(R)
        assert after != before and [share[g] for g in after] == sorted([share[g] for g in after], reverse=True)
        page.evaluate("() => { const s = document.querySelectorAll('.viewctl select, .panel[data-key=overview] select'); for (const x of s) if (Array.from(x.options).some(o => o.value === 'name')) { x.value = 'name'; x.dispatchEvent(new Event('change')); break; } }"); page.wait_for_timeout(200)
        byname = page.evaluate("() => LM.App.order.taxa.slice()"); assert [R.taxa["names"][g] for g in byname] == sorted(R.taxa["names"])

    def test_zoom_with_ctrl_wheel(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html); page.evaluate("() => LM.setView('matrix')"); page.wait_for_timeout(200)
        box = page.evaluate("() => { const r = document.querySelector('canvas.matrix').getBoundingClientRect(); return [r.x, r.y]; }")
        before = page.evaluate("() => location.hash")
        page.evaluate("() => { const c = document.querySelector('canvas.matrix'); const r = c.getBoundingClientRect(); c.dispatchEvent(new WheelEvent('wheel', { deltaY: -300, ctrlKey: true, clientX: r.x + 50, clientY: r.y + 50, bubbles: true, cancelable: true })); }"); page.wait_for_timeout(400)
        after = page.evaluate("() => location.hash"); assert after != before and "v=" in after


class TestRibbonsAndTable:
    @pytest.mark.parametrize("view,kind", [("loci", "locus"), ("taxa", "taxon")])
    def test_ribbons_list_the_units(self, page, report_of, view, kind):
        html, R = report_of("dna_small"); open_report(page, html); page.evaluate(f"() => LM.setView('{view}')"); page.wait_for_timeout(400)
        rows = page.query_selector_all(".ribbons-view .rib-row, .rib-list .rib-row"); assert len(rows) >= 1
        canvases = page.query_selector_all(".ribbons-view canvas, .rib-list canvas"); assert len(canvases) >= 1

    def test_table_counts_sorts_and_filters(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html); page.evaluate("() => LM.setView('table')"); page.wait_for_timeout(400); c = R.man["counts"]
        assert page.text_content(".tfoot .n") == lm_fmt(lm_str(page, "table.foot"), n=c["masked"])
        heads = page.eval_on_selector_all(".thead span", "els => els.map(e => e.textContent)")
        assert heads[0] == lm_str(page, "table.h_id") and heads[1] == lm_str(page, "table.h_taxon") and lm_str(page, "table.h_bs") in heads and lm_str(page, "table.h_reason") in heads
        first = page.eval_on_selector_all(".trow", "els => els.map(e => e.children[1].textContent)")
        page.click(".thead span >> nth=1"); page.wait_for_timeout(200)
        after = page.eval_on_selector_all(".trow", "els => els.map(e => e.children[1].textContent)"); assert after == sorted(after) and len(after) == len(first)
        assert page.query_selector(".tfoot input[type=checkbox]") is None            # the table is the masked cells; no all-cells switch
        assert page.text_content(".tfoot .src") == lm_str(page, "table.source")        # a summary of masked.csv
        assert page.query_selector("button[data-act=tsv]") is None                   # and no export of its own
        # the ids are masked.csv's: the row with id k names the same cell as the file's k-th row
        page.click(".thead span >> nth=0"); page.wait_for_timeout(200)
        shown = page.eval_on_selector_all(".trow", "els => els.map(e => [e.children[0].textContent, e.children[1].textContent.trim(), e.children[2].textContent])")
        import csv as _csv, os as _os; rows = list(_csv.DictReader(open(_os.path.join(_os.path.dirname(R.path), "masked.csv"))))
        byid = {r["id"]: (r["taxon"], r["locus"]) for r in rows}
        assert len(shown) == len(rows) and all(byid[i] == (t, l) for i, t, l in shown)

    def test_a_table_row_opens_the_alignment(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html); page.evaluate("() => LM.setView('table')"); page.wait_for_timeout(300)
        name = page.text_content(".trow >> nth=0 >> span >> nth=1"); page.click(".trow >> nth=0"); page.wait_for_selector(".panel[data-key=sequence]")
        assert name.strip() in page.text_content(".panel[data-key=sequence] .crumb")

    def test_no_exports(self, page, report_of):
        """the table is a summary of masked.csv and the cards are the printable form: no TSV or PNG export of their own"""
        html, R = report_of("dna_small"); open_report(page, html)
        assert page.query_selector("button[data-act=tsv]") is None and page.query_selector("button[data-act=png]") is None
