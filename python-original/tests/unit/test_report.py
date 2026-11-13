"""The report files: report.npz as numpy opens it, the manifest, and report.html as one self-contained page."""
import io, base64, json, os, re, zipfile
import numpy as np
import pytest
from tests.conftest import lo, dataset, run_tool


@pytest.mark.e2e
class TestLicences:
    def test_the_page_carries_the_licences_of_what_it_embeds(self, runs):
        html = open(runs("dna_small").file("report.html"), encoding="utf-8").read()
        block = re.search(r'<script type="text/plain" id="lm-licences">(.*?)</script>', html, re.S); assert block
        t = block.group(1)
        for holder in ("Copyright (c) 2026 Robert Lanfear", "Copyright 2010-2023 Mike Bostock", "Copyright 2015, Mike Bostock", "Apache License", "SIL OPEN FONT LICENSE"):
            assert holder in t, holder
        assert "Permission is hereby granted, free of charge" in t and t.count("=" * 78) == 10
        assert "@@LICENCES@@" not in html


@pytest.mark.e2e
class TestNpz:
    def test_numpy_opens_it_like_an_npz(self, runs):
        r = runs("dna_small"); z = np.load(r.file("report.npz"))
        assert "cells/n_comp" in z.files and z["cells/n_comp"].shape == (30, 14) and z["taxa/closest"].shape[1] == 5

    def test_members_are_version_1_npy_or_json(self, runs):
        z = zipfile.ZipFile(runs("dna_small").file("report.npz"))
        for k in z.namelist():
            b = z.read(k)
            if k.endswith(".npy"): assert b[:6] == b"\x93NUMPY" and b[6] == 1
            else: assert k.endswith(".json") and json.loads(b) is not None

    def test_manifest_side_bits_and_code_map_are_the_documented_ones(self, runs):
        m = runs("dna_small").report.man
        assert m["side_bits"] == dict(masked_divergent=1, masked_similar=2, ambiguous=4, similar_uncertain=8, flagged_not_masked=16, similar_not_masked=32, similar_tested=64, similar_source=128)
        assert m["code_map"] == dict(mismatch_count_max=5, similar_base=128, tick=254, absent=255)
        assert m["carried"]["numeric_columns"] == "dense" and m["carried"]["best_match_cells"] == "all"

    def test_manifest_records_the_python_and_numpy_and_no_platform(self, runs):
        m = runs("dna_small").report.man; assert "platform" not in m and m["python"] and m["numpy"] == np.__version__


@pytest.mark.e2e
class TestHtml:
    def test_the_page_carries_the_core_and_the_folder_carries_the_chunks(self, runs):
        """the report is a page plus report_files/ (Rob, 4 Nov 2026: a folder at every size): the page holds the core inline
        (manifest, names, states, every masked or flagged cell's numbers, the ribbon sample); the folder holds one chunk per
        locus and one per block of taxa, each a one-line script handing the page a base64 zip"""
        r = runs("dna_small"); html = open(r.file("report.html"), encoding="utf-8").read(); R = r.report
        assert 'id="lm-bundle"' in html and "@@BUNDLE@@" not in html and "@@META@@" not in html
        assert not re.search(r'<script src="[^"]+"></script>', html) and not re.search(r'<link rel="stylesheet" href=', html)
        core = zipfile.ZipFile(io.BytesIO(base64.b64decode(re.search(r'id="lm-bundle">([^<]*)<', html).group(1))))
        man = json.loads(core.read("manifest.json")); lay = man["layout"]
        assert lay["kind"] == "folder" and lay["dir"] == "report_files" and lay["loci"] == R.nloc
        assert 1 <= lay["taxa_block"] <= lo.REPORT_TAXA_BLOCK and lay["taxa_blocks"] == -(-R.n // lay["taxa_block"])
        assert 1 <= lay["locus_block"] <= lo.REPORT_TAXA_BLOCK and lay["loci_blocks"] == -(-R.nloc // lay["locus_block"])
        names = set(core.namelist())
        assert {"cells/side.npy", "cells/n_comp.npy", "cells/present.npy", "core/locus.npy", "core/taxon.npy", "core/partner.npy", "core/comparators.npy", "sample/locus.npy", "sample/excess_m.npy"} <= names
        assert not any(k.startswith(("codes/", "seq/", "trees/", "similar/")) for k in names)          # per-locus members live in the chunks
        side = R.arr("cells/side.npy"); cl = np.load(io.BytesIO(core.read("core/locus.npy"))); cg = np.load(io.BytesIO(core.read("core/taxon.npy")))
        assert set(zip(cg.tolist(), cl.tolist())) == {(int(g), int(l)) for g, l in zip(*np.nonzero((side & (1 | 2 | 16 | 32)) > 0))}
        files = sorted(os.listdir(r.file("report_files")))
        assert files == sorted([f"loci_{b}.js" for b in range(lay["loci_blocks"])] + [f"taxa_{b}.js" for b in range(lay["taxa_blocks"])] + ["index.js"])
        idx = open(r.file("report_files/index.js"), encoding="utf-8").read(); assert idx.startswith("window.LM_INDEX=") and json.loads(idx[len("window.LM_INDEX="):].rstrip(";\n"))["loci"] == R.nloc
        li = 3; LB = lay["locus_block"]; b = li // LB; l0, l1 = b * LB, min(R.nloc, (b + 1) * LB)
        txt = open(r.file(f"report_files/loci_{b}.js"), encoding="utf-8").read()
        m = re.fullmatch(r'LM_CHUNK\("loci",%d,"([A-Za-z0-9+/=]+)"\);\n' % b, txt); assert m
        ch = zipfile.ZipFile(io.BytesIO(base64.b64decode(m.group(1)))); cn = set(ch.namelist())
        assert {"cells/med_identity.npy", "cells/partner.npy", "cells/comparators.npy", "cells/comp_div_log10p.npy", "cells/focal_sites.npy"} <= cn
        for k in ("med_identity", "partner", "excess_m"):                                               # a block column is the npz column's slice, (taxa, block)
            assert np.array_equal(np.load(io.BytesIO(ch.read(f"cells/{k}.npy"))), R.arr(f"cells/{k}.npy")[:, l0:l1], equal_nan=True)
        assert np.array_equal(np.load(io.BytesIO(ch.read("cells/comparators.npy"))), R.arr("cells/comparators.npy")[:, l0:l1])
        for lj in range(l0, l1):                                                                        # every locus of the block carries its own members
            if R.has(f"seq/{lj}.npy"): assert np.array_equal(np.load(io.BytesIO(ch.read(f"seq/{lj}.npy"))), R.arr(f"seq/{lj}.npy"))
            if R.has(f"trees/{lj}.json"): assert json.loads(ch.read(f"trees/{lj}.json")) == R.js(f"trees/{lj}.json")
        tb = open(r.file("report_files/taxa_0.js"), encoding="utf-8").read(); mb = re.fullmatch(r'LM_CHUNK\("taxa",0,"([A-Za-z0-9+/=]+)"\);\n', tb); assert mb
        tz = zipfile.ZipFile(io.BytesIO(base64.b64decode(mb.group(1)))); B = min(lay["taxa_block"], R.n)
        assert np.array_equal(np.load(io.BytesIO(tz.read("cells/excess_m.npy"))), R.arr("cells/excess_m.npy")[:B], equal_nan=True)
        assert np.load(io.BytesIO(tz.read("cells/comparators.npy"))).shape == (B, R.nloc, R.K)

    def test_icons_are_inlined(self, runs):
        html = open(runs("dna_small").file("report.html"), encoding="utf-8").read()
        assert 'rel="icon"' in html and html.count('href="data:image/png;base64,') >= 2 and 'src="data:image/svg+xml;base64,' in html
        assert 'icons/' not in html

    def test_viewer_and_style_are_inlined(self, runs):
        html = open(runs("dna_small").file("report.html"), encoding="utf-8").read()
        assert "LocusMasker report viewer" in html and "<style>" in html and "d3" in html and "<title>LocusMasker report</title>" in html

    def test_the_page_is_small_whatever_the_data(self, runs):
        """the page carries the core only: the states (a few bytes per cell) and the masked cells' numbers; the chunks carry the rest"""
        r = runs("dna_small"); page = os.path.getsize(r.file("report.html")); data = os.path.getsize(r.file("report.npz"))
        core = len(re.search(r'id="lm-bundle">([^<]*)<', open(r.file("report.html"), encoding="utf-8").read()).group(1)) * 3 // 4
        assert core < data and page < core * 4 / 3 + 1_300_000                 # the viewer, d3, fuse and the embedded Inter face


class TestWholeFiles:
    """the files are whole or absent; every run leaves run.log; a stale report_files folder never survives"""
    def test_a_normal_run_writes_whole_files_and_a_log(self, runs):
        r = runs("dna_small")
        assert r.exists("report.html") and r.exists("run.log") and not r.exists("taxa.tsv") and not r.exists("summary.json")
        assert not [f for f in os.listdir(r.out) if f.endswith(".part")]
        log = open(r.file("run.log"), encoding="utf-8").read(); assert "LocusMasker" in log and "outputs:" in log and "run.log" in log.split("outputs:")[1]

    def test_no_report_removes_the_folder_of_an_earlier_run(self, tmp_out):
        ds = dataset("dna_small")
        r = run_tool(ds.aln, tmp_out, "dna", ds.parts); assert r.rc == 0 and r.exists("report.html") and os.path.isdir(r.file("report_files"))
        open(r.file("report_files/locus_999.js"), "w").write("stale")                        # a leftover from another run of another size
        r = run_tool(ds.aln, tmp_out, "dna", ds.parts); assert r.rc == 0 and not os.path.exists(r.file("report_files/locus_999.js"))
        r = run_tool(ds.aln, tmp_out, "dna", ds.parts, ["--no-report"]); assert r.rc == 0           # an earlier run's report stays, named as stale
        assert r.exists("report.html") and os.path.isdir(r.file("report_files")) and "report_files" in r.log and "earlier run" in r.log


