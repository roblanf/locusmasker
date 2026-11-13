"""A headless Chromium for the report viewer: one browser per worker, a fresh page per test that collects script and
console errors, and helpers that build variant reports (older formats, corrupted or truncated files) from a run."""
import base64
import json, fcntl, json, os, re, zipfile
import numpy as np
import pytest
from tests.conftest import ROOT, Report, dataset, run_tool

pytestmark = pytest.mark.viewer
VIEWER = os.path.join(ROOT, "viewer")
ORANGE_INK, ORANGE, RED, BLUE, INK, FAINT = "#c45500", "#d9822b", "#b5172e", "#2f6bd8", "#0b0b0b", "#8c8a85"


@pytest.fixture(scope="session")
def browser():
    pw = pytest.importorskip("playwright.sync_api")
    p = pw.sync_playwright().start()
    try: b = p.chromium.launch()
    except Exception as e: p.stop(); pytest.skip(f"no headless chromium: {e}")
    yield b
    b.close(); p.stop()


@pytest.fixture
def page(browser, request):
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}); pg = ctx.new_page(); errs = []
    pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}")); pg.on("console", lambda m: errs.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
    pg.lm_errors = errs
    yield pg
    ctx.close()
    if not request.node.get_closest_marker("allow_errors"): assert errs == [], errs


READY = "window.LM && window.LM.App && window.LM.App.D"


def lm_str(page, path):
    """a catalog entry (viewer/strings.js), read from the page: the tests compare against the catalog, never against words"""
    return page.evaluate("p => p.split('.').reduce((o, k) => o[k], LM_STR)", path)


def lm_fmt(s, **kw):
    """the viewer's lmFmt: {name} placeholders filled"""
    return re.sub(r"\{(\w+)\}", lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0), s)


def lm_head(s):
    """the words before the first placeholder (for "contains" checks against a formatted string)"""
    return s.split("{")[0].strip()


def open_report(page, path, hash="", timeout=60000):
    page.goto("file://" + path + (("#" + hash) if hash else "")); page.wait_for_function(READY, timeout=timeout); page.wait_for_timeout(200)


def inline_viewer(b64, out):
    """report.html exactly as the tool writes it, around a given base64 bundle"""
    html = open(os.path.join(VIEWER, "viewer.html"), encoding="utf-8").read()
    html = html.replace("@@BUNDLE@@", b64)
    html = html.replace("@@META@@", json.dumps(dict(dataset="test_report", bundle_bytes=len(b64) * 3 // 4, version="test")), 1)
    html = re.sub(r'<script src="([^"]+)"></script>', lambda m: "<script>\n" + open(os.path.join(VIEWER, m.group(1)), encoding="utf-8").read() + "\n</script>", html)
    html = re.sub(r'<link rel="stylesheet" href="([^"]+)">', lambda m: "<style>\n" + open(os.path.join(VIEWER, m.group(1)), encoding="utf-8").read() + "\n</style>", html)
    def icon(m):
        src = os.path.join(VIEWER, m.group(2)); kind = "image/svg+xml" if src.endswith(".svg") else "image/png"
        return f'{m.group(1)}="data:{kind};base64,' + base64.b64encode(open(src, "rb").read()).decode("ascii") + '"'
    html = re.sub(r'(href|src)="(icons/[^"]+)"', icon, html)
    open(out, "w", encoding="utf-8").write(html); return out


def variant_npz(src, out, drop=(), manifest=None):
    """a copy of report.npz without some members and with manifest edits"""
    zi = zipfile.ZipFile(src); zo = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
    for it in zi.infolist():
        if any(it.filename.startswith(d) for d in drop): continue
        b = zi.read(it.filename)
        if it.filename == "manifest.json" and manifest: m = json.loads(b); manifest(m); b = json.dumps(m).encode()
        zo.writestr(it, b)
    zo.close(); return out


@pytest.fixture(scope="session")
def report_of(runs, shared_tmp):
    """the standard run's report.html of a dataset, and variants of it built once per session"""
    def get(name, variant=None):
        run = runs(name)
        if variant is None: return run.file("report.html"), run.report
        d = shared_tmp / "viewer" / f"{name}_{variant}"; d.mkdir(parents=True, exist_ok=True); html = str(d / "report.html"); npz = str(d / "report.npz")
        with open(str(d) + ".lock", "w") as lk:                      # one worker builds the variant; the others wait for it
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                if not os.path.exists(html):
                    src = run.file("report.npz")
                    if variant == "format5":
                        def m5(m): m["format_version"] = 5; m["carried"].pop("tests", None)
                        variant_npz(src, npz, drop=("cells/locus_closest", "cells/comp_div_log10p", "cells/comp_sim_log10p"), manifest=m5)
                    elif variant == "format3":
                        def m3(m): m["format_version"] = 3; m["options"].pop("donor_threshold", None)
                        variant_npz(src, npz, drop=("cells/locus_closest", "cells/comp_div_log10p", "cells/comp_sim_log10p", "similar/"), manifest=m3)
                    elif variant == "format9":
                        variant_npz(src, npz, manifest=lambda m: m.update(format_version=10, version="9.9.9"))
                    else: raise ValueError(variant)
                    inline_viewer(base64.b64encode(open(npz, "rb").read()).decode("ascii"), html + ".part"); os.replace(html + ".part", html)
            finally: fcntl.flock(lk, fcntl.LOCK_UN)
        return html, Report(npz)
    return get


def cells_of_kind(R):
    """one (taxon, locus) per kind of cell, from the report's side bits"""
    side = R.arr("cells/side.npy"); nc = R.arr("cells/n_comp.npy"); nf = R.arr("cells/n_fail.npy"); out = {}
    def add(k, g, li):
        if k not in out: out[k] = (int(g), int(li))
    for g in range(R.n):
        for li in range(R.nloc):
            if not nc[g, li]: continue
            sd = int(side[g, li])
            if (sd & 3) == 3: add("both", g, li)
            if sd & 8: add("uncertain", g, li)
            if (sd & 2) and not (sd & 8) and not (sd & 1): add("copy", g, li)
            if sd & 128 and not (sd & 3): add("source", g, li)
            if sd & 1 and not (sd & 2) and nf[g, li] == R.K: add("divergent", g, li)
            if sd & 4: add("ambiguous", g, li)
            if not (sd & (3 | 16 | 32 | 128)) and nf[g, li] > 0: add("near_miss", g, li)
            if not (sd & (3 | 16 | 32 | 128)) and nf[g, li] == 0 and sd & 64: add("closest_tested", g, li)
            if not (sd & (3 | 16 | 32 | 128 | 64)) and nf[g, li] == 0: add("clean", g, li)
    return out


def stars(p, bar):
    if p is None or not np.isfinite(p): return ""
    return "***" if p < bar - np.log10(50) else "**" if p < bar - np.log10(5) else "*" if p < bar else ""


def hover_text(page):
    return page.evaluate("() => { const h = document.querySelector('.hover'); return h && h.style.display !== 'none' ? h.textContent : null; }")
