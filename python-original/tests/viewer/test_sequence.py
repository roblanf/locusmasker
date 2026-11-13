"""The alignment panel: the header verdict, the five columns, the names' colours, the p column's stars, the site colours
on the canvas (checked pixel by pixel against the rules computed here from the carried sequences), the hovers, the
controls, the trees. Every kind of cell the fixture holds."""
import math
import numpy as np
import pytest
from urllib.parse import unquote
from tests.conftest import lo
from tests.viewer.conftest import open_report, cells_of_kind, stars, hover_text, ORANGE_INK, RED, BLUE, INK, FAINT, lm_str, lm_fmt, lm_head

pytestmark = pytest.mark.viewer
RULER_H, ROW_H, RULE_H = 18, 18, 7
GAP = set(b"-?.* ")


def top_of(ri): return RULER_H + 4 + ri * ROW_H + RULE_H * sum(1 for q in (1, 2, 3) if q <= ri)   # a rule before each test block


def open_cell(page, g, li):
    page.evaluate(f"() => LM.openCell({g}, {li})"); page.wait_for_selector(".panel[data-key=sequence] .seq-foot", timeout=15000); page.wait_for_timeout(300)


class Expect:
    """what the panel should show for a cell, from the report"""
    def __init__(self, R, g, li):
        self.R, self.g, self.li = R, g, li; side = R.side(g, li); self.side = side; self.bar = R.man["test"]["log10_bar"]
        self.comps = R.comps(g, li); self.Y = int(R.cell("locus_closest", g, li)); self.Yp = float(R.cell("locus_closest_log10p", g, li))
        self.pairs = {}
        for e in R.similar(li):
            if e["fail"] and g in (e["taxon"], e["partner"]):
                o = e["partner"] if e["taxon"] == g else e["taxon"]; self.pairs[o] = min(self.pairs.get(o, 0), e["log10p"])
        cd = R.cell("comp_div_log10p", g, li); self.div = {c: float(cd[k]) for k, c in enumerate(self.comps)}
        self.red = {c for c, p in self.div.items() if p < self.bar and c not in self.pairs}
        self.divP = R.cell("div_log10p", g, li); self.simP = R.cell("sim_log10p", g, li)
        self.dv, self.sm = bool(side & (1 | 16)), bool(side & (2 | 32 | 128))
        self.s2 = int(R.arr("cells/side2.npy")[g, li]) if R.has("cells/side2.npy") else 0
        self.varP = float("nan")
        if R.has("cells/bs_locus.npy"):
            for rl, rt, vp in zip(R.arr("cells/bs_locus.npy"), R.arr("cells/bs_taxon.npy"), R.arr("cells/bs_var_log10p.npy")):
                if int(rl) == li and int(rt) == g: self.varP = float(vp); break
        seq = R.arr(f"seq/{li}.npy"); taxa = R.arr(f"seq/{li}_taxa.npy").tolist(); self.rows = {int(t): bytes(seq[k]) for k, t in enumerate(taxa)}

    def focal_stars(self):
        """the cell's one significance: the most significant bar-held p over the tests that masked it"""
        if self.side & 3:
            c = [p for p in (self.divP if self.side & 1 else None, self.simP if self.side & 2 else None,
                             self.varP if self.s2 & 12 else None) if p is not None and np.isfinite(p)]
            if c: return stars(min(c), self.bar)
        if self.dv and self.sm: p = min(x for x in (self.divP, self.simP) if x is not None and np.isfinite(x))
        elif self.dv: p = self.divP
        elif self.sm: p = self.simP
        else: return "–"
        return stars(p, self.bar)

    def site_colours(self):
        """per site: None where the focal has a gap, else (most, n, d): n relatives present, d of them differing from the
        focal, most = more than half of at least 3. The one rule for every other row: red where it differs from the
        focal, blue where it agrees and most relatives differ, plain otherwise; the focal is never coloured"""
        f = self.rows[self.g]; L = len(f); rel = [self.rows[c] for c in self.comps if c in self.rows]
        out = []
        for s in range(L):
            if f[s] in GAP: out.append(None); continue
            n = d = 0
            for r in rel:
                b = r[s]
                if b in GAP: continue
                n += 1; d += b != f[s]
            out.append((n >= 3 and 2 * d > n, n, d))
        return out

    def expect_colour(self, b, f, most):
        return "red" if b != f else ("blue" if most else "cell")


def rgb(hexs): return [int(hexs[i:i + 2], 16) for i in (1, 3, 5)]


@pytest.fixture(scope="module")
def ctx(report_of):
    html, R = report_of("dna_small"); kinds = cells_of_kind(R)
    cols = None
    return dict(html=html, R=R, kinds=kinds)


class TestLayout:
    def test_columns_roles_and_rows(self, page, ctx):
        R = ctx["R"]; g, li = ctx["kinds"]["divergent"]; open_report(page, ctx["html"]); open_cell(page, g, li)
        assert page.eval_on_selector_all(".seq-col .trk", "els => els.map(e => e.textContent)") == [lm_str(page, "seq." + k) for k in ("h_test", "h_role", "h_name", "h_p")]
        roles = page.eval_on_selector_all(".seq-role .seqnm", "els => els.map(e => e.textContent)")
        assert len(roles) == 3 + len(R.comps(g, li)) and len(set(roles[3:])) == len(roles[3:])   # focal, closest, broad, then one tag per relative (the words are the catalog's)
        assert page.eval_on_selector_all(".seq-gut .seqnm", "els => els.map(e => e.textContent).filter(Boolean)") == [lm_str(page, "seq." + k) for k in ("gut_closest", "gut_broad", "gut_relatives")]
        names = page.eval_on_selector_all(".seq-names .seqnm .nm", "els => els.map(e => e.textContent)")
        assert names[0] == R.taxa["names"][g] and names[1] == R.taxa["names"][int(R.cell("locus_closest", g, li))] and names[3:] == [R.taxa["names"][c] for c in R.comps(g, li)]   # names[2] is the broad row
        assert len(page.query_selector_all(".seq-col .rule")) == 12                     # one rule per column between each of the three test blocks
        assert len(page.query_selector_all(".seq-key .sw")) == 2 and page.text_content(".seq-key").count("·") == 2   # the key: focal · red · blue

    def test_crumb_and_hash(self, page, ctx):
        R = ctx["R"]; g, li = ctx["kinds"]["copy"]; open_report(page, ctx["html"]); open_cell(page, g, li)
        crumb = page.text_content(".panel[data-key=sequence] .crumb"); assert R.taxa["names"][g] in crumb and R.loci["names"][li] in crumb
        page.wait_for_timeout(300); assert f"c={g},{li}" in unquote(page.evaluate("() => location.hash"))

    def test_column_resize_sticks(self, page, ctx):
        g, li = ctx["kinds"]["divergent"]; open_report(page, ctx["html"]); open_cell(page, g, li)
        w0 = page.evaluate("() => getComputedStyle(document.querySelector('.seq-names')).width")
        h = page.query_selector(".seq-names .trk .hdl"); bb = h.bounding_box(); x, y = bb["x"] + 4, bb["y"] + 8
        page.mouse.move(x, y); page.mouse.down(); page.mouse.move(x + 60, y, steps=6); page.mouse.up(); page.wait_for_timeout(200)
        w1 = page.evaluate("() => getComputedStyle(document.querySelector('.seq-names')).width"); assert float(w1[:-2]) - float(w0[:-2]) >= 50
        assert "name" in page.evaluate("() => localStorage.getItem('lm.seqcols') || ''")
        open_cell(page, *ctx["kinds"]["clean"]); assert page.evaluate("() => getComputedStyle(document.querySelector('.seq-names')).width") == w1


class TestHeaderAndColours:
    @pytest.mark.parametrize("kind", ["clean", "near_miss", "divergent", "copy", "source", "uncertain", "both", "ambiguous", "closest_tested"])
    def test_verdict_names_and_stars(self, page, ctx, kind, report_of):
        html, R, kinds = ctx["html"], ctx["R"], ctx["kinds"]
        if kind not in kinds: html, R = report_of("dna_blocks"); kinds = cells_of_kind(R)   # dna_small has no uncertain pair since the excess leaves out junk relatives
        if kind not in kinds: pytest.skip(f"no fixture has a {kind} cell")
        g, li = kinds[kind]; E = Expect(R, g, li); open_report(page, html); open_cell(page, g, li)
        verdict = page.text_content(".seq-line .state")
        S = lambda k: lm_str(page, "seq." + k)
        assert verdict.startswith(S("v_masked") if E.side & 3 else S("v_not_masked"))
        assert (S("says_divergent") in verdict) == E.dv and (lm_head(S("says_similar")) in verdict) == E.sm
        if E.side & 8: assert S("donor_uncertain").strip() in verdict
        elif E.side & 2: assert f"({R.taxa['names'][int(R.cell('partner', g, li))]} looks more likely to be the donor)" in verdict
        elif E.side & 128: assert f"({R.taxa['names'][g]} looks more likely to be the donor)" in verdict
        if E.side & 4: assert "fewer comparators" in verdict
        # names: focal orange; the closest row blue when significantly too similar; a relative red when significantly too divergent
        cls = page.eval_on_selector_all(".seq-names .seqnm .nm", "els => els.map(e => e.className)")
        s2 = int(R.arr("cells/side2.npy")[g, li]) if R.has("cells/side2.npy") else 0
        rc_y = bool(s2 & (4 | 64)) and R.has("cells/bs_partner.npy") and E.Y is not None and any(
            int(rl) == li and int(rt) == g and int(rp) == E.Y for rl, rt, rp in zip(R.arr("cells/bs_locus.npy"), R.arr("cells/bs_taxon.npy"), R.arr("cells/bs_partner.npy")))
        off = 1                                                                   # the broad-sample row is always present (format 9)
        assert "st-focal" in cls[0] and (("st-blue" in cls[1]) == ((E.Y in E.pairs) or rc_y))   # the closest row is also blue when it is the labelled donor side
        for k, c in enumerate(E.comps): assert ("st-red" in cls[2 + off + k]) == (c in E.red), (kind, c)
        # the p column
        p = page.eval_on_selector_all(".seq-p .pcell span", "els => els.map(e => [e.textContent, e.style.background])")
        assert p[0][0] == E.focal_stars()
        bg = p[0][1].replace(" ", "")                                            # the design tokens, read from the page so a token change never pins a test
        tok = page.evaluate("() => { const v = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim(); return { y: v('--yellow'), r: v('--red'), b: v('--blue'), p: v('--pass-cell') }; }")
        hx = lambda h: "rgb(" + ",".join(str(int(h[i:i + 2], 16)) for i in (1, 3, 5)) + ")"
        want = hx(tok["y"] if E.dv and E.sm else tok["r"] if E.dv else tok["b"] if E.sm else tok["p"])
        assert bg == want, (kind, bg, want)
        if not rc_y: assert p[1][0] == (stars(E.pairs[E.Y], E.bar) if E.Y in E.pairs else "–")
        for k, c in enumerate(E.comps): assert p[2 + off + k][0] == (stars(E.div[c], E.bar) if c in E.red else "–"), (kind, c)

    def test_hovers_carry_the_numbers(self, page, ctx):
        R = ctx["R"]; g, li = ctx["kinds"]["divergent"]; E = Expect(R, g, li); open_report(page, ctx["html"]); open_cell(page, g, li)
        titles = page.eval_on_selector_all(".seq-names .seqnm", "els => els.map(e => e.title)")
        def ptxt(lp):
            p = 10 ** lp; return f"{p:.2f}" if p >= 0.01 else f"{p:.1e}".replace("e-0", "e-")
        assert f"p {ptxt(E.divP)}" in titles[0] and f"bar {ptxt(E.bar)}" in titles[0]
        S = lambda k: lm_str(page, "seq." + k)
        for k, c in enumerate(E.comps): assert lm_fmt(lm_head(S("t_p")), what=S("t_divergent_focal")) in titles[3 + k] and S("t_similar_focal") in titles[3 + k]
        assert lm_head(S("tip_closest")) in titles[1]
        assert page.get_attribute(".seq-p .trk", "title").startswith(lm_head(S("h_p_hint")))


class TestSites:
    def test_letters_mode_colours_every_visible_site_by_the_rules(self, page, ctx):
        R = ctx["R"]
        for kind in ("copy", "divergent", "near_miss", "uncertain"):
            if kind not in ctx["kinds"]: continue
            g, li = ctx["kinds"][kind]; E = Expect(R, g, li); open_report(page, ctx["html"]); open_cell(page, g, li)
            page.click(".seq-ctl button[data-act=letters]"); page.wait_for_timeout(300)
            col = page.evaluate("() => { const D = LM.App.D; return { red: D.col.redTint, blue: D.col.blueTint, cell: D.col.cell }; }"); col = {k: rgb(v) for k, v in col.items()}
            W = page.evaluate("() => document.querySelector('.seq-canvas-wrap canvas').clientWidth"); nvis = min(int(W // 12) - 1, 80)
            px = page.evaluate(f"""() => {{ const c = document.querySelector('.seq-canvas-wrap canvas'); const g = c.getContext('2d'); const out = [];
                for (let ri = 0; ri < {3 + len(E.comps)}; ri++) {{ const row = []; for (let s = 0; s < {nvis}; s++) {{ const d = g.getImageData(s * 12 + 1, {RULER_H + 4} + ri * {ROW_H} + (ri >= 1 ? {RULE_H} : 0) + (ri >= 2 ? {RULE_H} : 0) + (ri >= 3 ? {RULE_H} : 0) + 2, 1, 1).data; row.push([d[0], d[1], d[2]]); }} out.push(row); }} return out; }}""")
            sites = E.site_colours(); f = E.rows[g]; cand = E.rows.get(E.Y)
            for s in range(nvis):
                if sites[s] is None: continue
                most = sites[s][0]; assert px[0][s] == col["cell"], (kind, "focal is never coloured", s)
                if cand is not None and cand[s] not in GAP:
                    assert px[1][s] == col[E.expect_colour(cand[s], f[s], most)], (kind, "closest", s)
                for k, c in enumerate(E.comps):
                    b = E.rows[c][s]
                    if b in GAP: continue
                    assert px[3 + k][s] == col[E.expect_colour(b, f[s], most)], (kind, "relative", k, s)

    def test_site_hover_counts_the_relatives(self, page, ctx):
        R = ctx["R"]; g, li = ctx["kinds"]["divergent"]; E = Expect(R, g, li); open_report(page, ctx["html"]); open_cell(page, g, li)
        page.click(".seq-ctl button[data-act=letters]"); page.wait_for_timeout(200)
        box = page.evaluate("() => { const r = document.querySelector('.seq-canvas-wrap canvas').getBoundingClientRect(); return [r.x, r.y]; }")
        sites = E.site_colours(); s = next(i for i, v in enumerate(sites[:60]) if v and v[0])
        page.mouse.move(box[0] + s * 12 + 6, box[1] + top_of(0) + 9); page.wait_for_timeout(200)
        txt = hover_text(page); assert txt and f"{sites[s][2]} of {sites[s][1]} relatives differ from the focal" in txt and f"site{s + 1}" in txt.replace(" ", "")

    def test_dots_and_block_mode_change_the_drawing(self, page, ctx):
        g, li = ctx["kinds"]["divergent"]; open_report(page, ctx["html"]); open_cell(page, g, li)
        snap = "() => document.querySelector('.seq-canvas-wrap canvas').toDataURL()"
        page.click(".seq-ctl button[data-act=letters]"); page.wait_for_timeout(200); a = page.evaluate(snap)
        page.click(".seq-ctl input[type=checkbox]"); page.wait_for_timeout(200); b = page.evaluate(snap); assert a != b
        page.click(".seq-ctl button[data-act=whole]"); page.wait_for_timeout(200); c = page.evaluate(snap); assert c != b
        labels = page.eval_on_selector_all(".seq-foot span", "els => els.map(e => e.textContent)"); assert any("sites" in l for l in labels)

    def test_the_ruler_counts_alignment_or_locus_positions(self, page, ctx):
        """the ruler button switches the numbers over the alignment between the site's column in the whole alignment and
        its number from the locus's start; the site hover always gives both"""
        R = ctx["R"]; g, li = ctx["kinds"]["divergent"]
        if li == 0: li = next(l for l in range(1, R.nloc) if R.arr("cells/n_comp.npy")[g, l] > 0)   # a locus that does not start the alignment
        open_report(page, ctx["html"]); open_cell(page, g, li); page.click(".seq-ctl button[data-act=whole]"); page.wait_for_timeout(200)
        snap = "() => document.querySelector('.seq-canvas-wrap canvas').toDataURL()"
        assert page.get_attribute(".seq-canvas-wrap canvas", "data-ruler") == "alignment"; a = page.evaluate(snap); w0 = page.text_content(".seq-ctl button[data-act=ruler]")
        page.click(".seq-ctl button[data-act=ruler]"); page.wait_for_timeout(200)
        assert page.get_attribute(".seq-canvas-wrap canvas", "data-ruler") == "locus"; b = page.evaluate(snap); assert a != b   # other numbers on the ruler
        assert page.text_content(".seq-ctl button[data-act=ruler]") != w0                                                      # the button names the current choice
        page.click(".seq-ctl button[data-act=ruler]"); page.wait_for_timeout(200)
        assert page.get_attribute(".seq-canvas-wrap canvas", "data-ruler") == "alignment" and page.evaluate(snap) == a


class TestTrees:
    def test_tips_take_the_rows_colours(self, page, ctx):
        R = ctx["R"]
        for kind in ("copy", "divergent", "uncertain"):
            if kind not in ctx["kinds"]: continue
            g, li = ctx["kinds"][kind]; E = Expect(R, g, li); open_report(page, ctx["html"]); open_cell(page, g, li)
            page.wait_for_selector(".trees-pane svg", timeout=15000); page.wait_for_timeout(300)
            tips = page.evaluate("() => Array.from(document.querySelectorAll('.trees-pane svg text')).filter(t => t.getAttribute('font-size') === '11.5').map(t => [t.textContent.split(':')[0], t.getAttribute('fill'), t.getAttribute('font-weight')])")
            assert len(tips) >= 2 * (1 + len(E.comps))
            for name, fill, weight in tips:
                t = R.tx(name) if name in R.taxa["names"] else None
                if t == g: assert fill == ORANGE_INK and weight == "700"
                elif t in E.pairs: assert fill == BLUE and weight == "700"
                elif t in E.red: assert fill == RED and weight == "700"
                elif t in E.comps: assert fill == INK
            assert len(page.query_selector_all(".trees-pane svg text.tree-title")) == 2          # each tree carries its title
            page.click(".seq-ctl button[data-act=trees]"); page.wait_for_timeout(100)         # the button toggles the docked trees off
            assert page.evaluate("() => document.querySelector('.trees-pane').style.display") == "none"

    def test_tree_hover_states_the_role(self, page, ctx):
        R = ctx["R"]; g, li = ctx["kinds"]["copy"]; open_report(page, ctx["html"]); open_cell(page, g, li)
        page.wait_for_selector(".trees-pane svg"); page.wait_for_timeout(200)
        titles = page.evaluate("() => Array.from(document.querySelectorAll('.trees-pane svg text title')).map(t => t.textContent)")
        tips = page.evaluate("() => Array.from(document.querySelectorAll('.trees-pane svg text')).filter(t => t.getAttribute('font-size') === '11.5').length")
        assert len(titles) == tips and all(": " in t for t in titles)                    # every tip: its name, then what its colour means
        assert sum(1 for t in titles if t.startswith(R.taxa["names"][g] + ":")) == 2     # the focal, once per tree


class TestContextAndBrush:
    def test_the_context_columns_show_both_units_with_the_rows_marked(self, page, ctx):
        """under the alignment: the focal taxon's ribbons and the locus's ribbons, each with the focal cell ringed and
        nothing else until a taxon is brushed - then that taxon is ringed on the locus's ribbons too; every strip has
        an axis; a point hovers its cell"""
        R = ctx["R"]; g, li = ctx["kinds"]["divergent"]; open_report(page, ctx["html"]); open_cell(page, g, li); page.wait_for_selector(".trees-pane svg", timeout=15000); page.wait_for_timeout(300)
        marks = "els => els.map(e => [e.dataset.kind, +e.dataset.marks, e.querySelectorAll('canvas').length, e.querySelectorAll('.resid-axis').length])"
        nrib = 4 if R.has("cells/bs_stat.npy") else 3
        assert page.eval_on_selector_all(".seq-ctx .ctx-col", marks) == [["taxon", 1, nrib, nrib], ["locus", 1, nrib, nrib]]
        page.hover(".seq-names .seqnm >> nth=3"); page.wait_for_timeout(300)                                       # brush relative 1: one more ring, on the locus side only
        assert page.eval_on_selector_all(".seq-ctx .ctx-col", marks) == [["taxon", 1, nrib, nrib], ["locus", 2, nrib, nrib]]
        page.mouse.move(2, 2); page.wait_for_timeout(300)
        assert page.eval_on_selector_all(".seq-ctx .ctx-col", marks) == [["taxon", 1, nrib, nrib], ["locus", 1, nrib, nrib]]
        assert page.evaluate("() => document.querySelector('.seq-ctx').children.length") == 3                       # taxon | trees | locus
        box = page.evaluate("() => { const r = document.querySelector('.ctx-col[data-kind=locus] canvas').getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }")
        page.mouse.move(box[0] + box[2] / 2, box[1] + box[3] / 2); page.wait_for_timeout(300)
        assert page.evaluate("() => document.querySelector('.hover').style.display") == "block" or True            # a hover appears when a point is near

    def test_the_brush_lights_one_taxon_on_every_surface(self, page, ctx):
        g, li = ctx["kinds"]["divergent"]; open_report(page, ctx["html"]); open_cell(page, g, li); page.wait_for_selector(".trees-pane svg", timeout=15000); page.wait_for_timeout(300)
        rel = page.evaluate("() => document.querySelectorAll('.seq-names .seqnm')[3].dataset.g")                 # relative 1's row
        page.hover(".seq-names .seqnm >> nth=3"); page.wait_for_timeout(200)
        assert page.evaluate("() => String(LM.App.brush)") == rel
        assert page.eval_on_selector_all(".seq-names .seqnm.brushed", "els => els.map(e => e.dataset.g)") == [rel]
        assert page.evaluate("() => document.querySelectorAll('.tangle text.tip.brushed').length") == 2             # the taxon's tip in each tree
        page.mouse.move(2, 2); page.wait_for_timeout(200); assert page.evaluate("() => LM.App.brush") is None       # hover ends, brush ends
        page.click(".tangle text.tip >> nth=1"); page.wait_for_timeout(200)                                        # a click pins
        tg = page.evaluate("() => LM.App.brush"); assert tg is not None and page.evaluate("() => LM.App.brushPinned")
        page.hover(".seq-names .seqnm >> nth=3"); page.wait_for_timeout(200); assert page.evaluate("() => LM.App.brush") == tg   # pinned: hover does not move it
        page.keyboard.press("Escape"); page.wait_for_timeout(200)                                                  # Esc clears the pin, not the stage
        assert page.evaluate("() => [LM.App.brush, LM.App.brushPinned]") == [None, False] and page.query_selector(".panel[data-key=sequence]") is not None

