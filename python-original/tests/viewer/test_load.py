"""Loading a report: the page comes up with the right header, refuses what it cannot read, and copes with reports that
carry less (older formats, sparse numbers, no sequences, nothing masked)."""
import base64, os
import pytest
from tests.conftest import dataset, run_tool
from tests.viewer.conftest import open_report, inline_viewer, READY, lm_str, lm_fmt

pytestmark = pytest.mark.viewer


class TestLoads:
    def test_header_shows_the_dataset_and_the_counts(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html); c = R.man["counts"]
        want = lm_fmt(lm_str(page, "loading.tab_title"), dataset=R.man["dataset"])                      # the loader retitles the tab from the catalog
        page.wait_for_function("t => document.title === t", arg=want); assert page.title() == want
        assert page.text_content(".topbar h1") == R.man["dataset"] == "dna_small"
        sub = page.text_content(".topbar .sub"); assert sub == lm_fmt(lm_str(page, "about.sub"), masked=c["masked"], cells=c["cells"], pct=f"{100 * c['masked'] / c['cells']:.1f}%", taxa=c["taxa"], loci=c["loci"])
        meta = page.text_content(".topbar .meta summary"); assert R.man["version"] in meta and R.man["date"] in meta

    def test_the_about_fold_holds_the_licences_folded(self, page, report_of):
        """the licences cost no screen while working: a fold inside the closed about fold, the full texts under it"""
        html, R = report_of("dna_small"); open_report(page, html)
        assert page.evaluate("() => document.querySelector('.topbar details.meta').open") is False
        assert page.text_content(".meta details.licences summary") == lm_str(page, "about.licences")
        assert page.evaluate("() => document.querySelector('.meta details.licences').open") is False
        assert "Copyright (c) 2026 Robert Lanfear" in page.text_content(".meta details.licences pre")

    def test_logo_and_favicons_are_in_the_page(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html)
        assert page.evaluate("() => document.querySelector('.topbar img.logo').src.startsWith('data:image/svg+xml')")
        assert page.evaluate("() => Array.from(document.querySelectorAll('link[rel=icon]')).map(l => l.sizes.value + ':' + l.href.slice(0, 14))") == ["16x16:data:image/png", "32x32:data:image/png"]

    def test_the_api_the_tests_use_is_exposed(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html)
        assert page.evaluate("() => Object.keys(window.LM).sort()") == ["App", "brush", "masks", "openCard", "openCell", "openCopies", "openUnit", "setView", "start", "step", "up"]
        assert page.evaluate("() => [LM.App.D.n, LM.App.D.nloc, LM.App.D.format, LM.App.D.K, LM.App.D.M]") == [R.n, R.nloc, 9, R.K, R.M]

    def test_the_loading_screen_is_gone(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html)
        assert page.query_selector(".loading") is None and page.query_selector("#lm-load") is None and page.query_selector(".topbar") is not None

    @pytest.mark.parametrize("name", ["aa_small", "dna_dense", "dna_clean", "aa_clean", "cannon_aa", "near_dna"])
    def test_every_fixture_report_opens(self, page, report_of, name):
        html, R = report_of(name); open_report(page, html)
        assert page.text_content(".topbar h1") == R.man["dataset"]

    def test_a_report_with_nothing_masked(self, page, report_of):
        html, R = report_of("dna_clean"); open_report(page, html)
        assert page.text_content(".topbar .sub").startswith("0 of ")                  # the masked total lives in the top bar
        cards = page.eval_on_selector_all(".cards .card .card-n", "els => els.map(e => e.textContent)")
        assert cards and all(v == "0" for v in cards) and page.query_selector_all(".worst .col .row") == []

    def test_sparse_numbers_and_no_sequences(self, page, tmp_path, monkeypatch):
        from tests import conftest as C
        ds = dataset("dna_small"); monkeypatch.setattr(C.lo, "DENSE_CELLS_MAX", 10)
        r = run_tool(ds.aln, str(tmp_path / "o"), "dna", ds.parts, ["--report-sequences", "none", "--report-codes", "none"]); R = r.report
        assert not R.dense
        open_report(page, r.file("report.html"))
        assert page.text_content(".topbar h1") == "dna_small"
        g, li = [(int(a), int(b)) for a, b in zip(*(R.arr("cells/side.npy") & 3).nonzero())][0]
        page.evaluate(f"() => LM.openCell({g}, {li})"); page.wait_for_timeout(600)
        body = page.text_content(".panel[data-key=sequence]"); assert "not carried" in body or "neither sequences nor site codes" in body


class TestRefusals:
    @pytest.mark.allow_errors
    def test_a_newer_format_is_refused_with_the_version(self, page, report_of):
        html, R = report_of("dna_small", "format9"); page.goto("file://" + html); page.wait_for_selector(".error", timeout=30000)
        txt = page.text_content(".error"); assert "10" in txt and "9.9.9" in txt            # the numbers, never the words: those are Rob's

    @pytest.mark.allow_errors
    def test_corrupted_data_holds_the_loading_screen_not_a_blank_page(self, page, report_of, tmp_path):
        """data that does not end like the tool's ZIP: the loading screen holds in its incomplete state (words are
        Rob's to change - the tests pin structure, never text)"""
        html, R = report_of("dna_small"); b64 = base64.b64encode(b"this is not a zip file" * 100).decode()
        out = inline_viewer(b64, str(tmp_path / "bad.html"))
        page.add_init_script("try { sessionStorage.lmRetry = 9 } catch (e) {}")   # hold the auto-retry off while we look
        page.goto("file://" + out); page.wait_for_selector('#lm-load[data-state="incomplete"]', timeout=30000)
        assert page.query_selector("#lm-load .ls-track") is not None

    @pytest.mark.allow_errors
    def test_a_truncated_file_holds_the_loading_screen_with_the_figures(self, page, report_of, tmp_path):
        """a page cut inside the bundle still boots (the viewer sits before the data): the loading screen holds in
        its incomplete state with the title, a partly filled bar and the X / Y figure, and retries by itself"""
        html, R = report_of("dna_small"); txt = open(html, encoding="utf-8").read()
        b = txt.index('id="lm-bundle"'); cut = str(tmp_path / "cut_bundle.html")
        open(cut, "w", encoding="utf-8").write(txt[:b + (len(txt) - b) // 2])
        page.add_init_script("try { sessionStorage.lmRetry = 9 } catch (e) {}")
        page.goto("file://" + cut); page.wait_for_selector('#lm-load[data-state="incomplete"]', timeout=30000)
        assert page.text_content("#lm-load .ls-title") == "dna_small"
        w = page.evaluate("() => parseFloat(document.querySelector('#lm-load .ls-fill').style.width)")
        assert 0 < w < 100
        num = page.text_content("#lm-load .ls-num"); assert "/" in num and ("MB" in num or "KB" in num)

    @pytest.mark.allow_errors
    def test_a_page_cut_inside_the_viewer_falls_to_the_loader(self, page, report_of, tmp_path):
        """cut before the viewer script ends, nothing can boot: the head loader shows the screen with the title and
        calls the short file incomplete (a re-copy fixes it, so it retries by itself)"""
        html, R = report_of("dna_small"); txt = open(html, encoding="utf-8").read()
        v = txt.index("LocusMasker report viewer. One classic"); cut = str(tmp_path / "cut_viewer.html")
        open(cut, "w", encoding="utf-8").write(txt[:v + 2000])
        page.add_init_script("try { sessionStorage.lmRetry = 9 } catch (e) {}")
        page.goto("file://" + cut); page.wait_for_selector('#lm-load[data-state="incomplete"]', timeout=30000)
        assert page.text_content("#lm-load .ls-title") == "dna_small"

    def test_nothing_in_the_page_reads_undefined_or_nan(self, page, report_of):
        html, R = report_of("dna_small"); open_report(page, html); page.wait_for_timeout(300)
        shown = page.evaluate("() => document.getElementById('app').innerText")                      # rendered text only, not the inlined script
        assert "undefined" not in shown and "NaN" not in shown and "null" not in shown.split()

    def test_a_page_without_data_says_so(self, page, tmp_path):
        """the viewer template without a report: a message, no file picker (the browser version is gone)"""
        out = inline_viewer("", str(tmp_path / "empty.html")); page.goto("file://" + out); page.wait_for_selector(".error", timeout=30000)
        assert page.query_selector("input[type=file]") is None


class TestOlderFormats:
    @pytest.mark.parametrize("variant", ["format5", "format3"])
    def test_older_reports_open_and_summarise(self, page, report_of, variant):
        html, R = report_of("dna_small", variant); open_report(page, html)
        assert page.evaluate("() => LM.App.D.format") == (5 if variant == "format5" else 3)
        assert page.text_content(".topbar .sub").replace(",", "").startswith(str(R.man["counts"]["masked"]) + " of ")
        for view in ("matrix", "loci", "taxa", "table", "summary"): page.evaluate(f"() => LM.setView('{view}')"); page.wait_for_timeout(150)
