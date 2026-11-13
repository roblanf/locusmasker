"""The letters the report carries are the alignment's. The main pass reads the rows it wants, then blanks the cells it masks
in the shared row buffers; the broad-sample block reads rows from those buffers afterwards, so a cell the main tests masked
and the broad block then named (as a partner or a representative) came back as gaps and replaced the good row (4 November
2026: Wu_2018_aa, GALEO_VAR at ENSG00000171564). These pin the three facts that stop it: every stored row equals the
input, every masked cell keeps its letters, and the broad block's rows add to the main pass's and never replace them."""
import numpy as np
import pytest
from tests.conftest import lo, dataset, run_tool, SYNTH, SIM

GAP, MISSING = ord("-"), ord("?")


def seq_rows(R, li):
    """the taxa with a stored row at the locus, and the rows"""
    return (R.arr(f"seq/{li}_taxa.npy").tolist(), R.arr(f"seq/{li}.npy")) if R.has(f"seq/{li}.npy") else ([], None)


@pytest.mark.parametrize("name", SYNTH + SIM)
def test_every_stored_row_is_the_alignments(runs, name):
    """whatever rows the report carries, each is the input alignment's row at that locus, byte for byte"""
    R = runs(name).report; ds = dataset(name)
    names, rows, loci = lo.read_input(ds.aln, ds.parts)[:3]
    assert names == R.taxa["names"]
    checked = 0
    for li in range(R.nloc):
        taxa, blk = seq_rows(R, li)
        for k, g in enumerate(taxa):
            assert np.array_equal(blk[k], rows[g, loci[li][1]]), (name, R.loci["names"][li], names[g]); checked += 1
    assert checked > 0


@pytest.mark.parametrize("name", SYNTH + SIM)
def test_masked_cells_keep_their_letters(runs, name):
    """a masked cell has a stored row, and the row holds at least the residues the test counted for it: masking blanks
    the masked alignment, never the report"""
    R = runs(name).report
    if R.man["carried"]["sequences"] == "none": pytest.skip("no sequences in this report")
    side = R.arr("cells/side.npy"); fs = R.arr("cells/focal_sites.npy"); checked = 0
    if not ((side & 3) > 0).any(): pytest.skip("nothing masked")
    for g, li in zip(*np.nonzero((side & 3) > 0)):
        taxa, blk = seq_rows(R, li); g, li = int(g), int(li)
        assert g in taxa, (name, R.taxa["names"][g], R.loci["names"][li])
        row = blk[taxa.index(g)]; letters = int(((row != GAP) & (row != MISSING)).sum())
        assert letters > 0 and letters >= int(fs[g, li]), (name, R.taxa["names"][g], R.loci["names"][li], letters, int(fs[g, li])); checked += 1
    assert checked > 0


@pytest.mark.parametrize("name", SYNTH + SIM)
def test_the_broad_blocks_rows_are_present_and_intact(runs, name):
    """every broad-sample record's focal and partner have a row with letters at the locus"""
    R = runs(name).report; fs = R.arr("cells/focal_sites.npy"); checked = 0
    for li in range(R.nloc):
        if not R.has(f"broadsample/{li}.json"): continue
        taxa, blk = seq_rows(R, li)
        for rec in R.js(f"broadsample/{li}.json"):
            for g in (int(rec["taxon"]), int(rec["partner"])):
                assert g in taxa, (name, R.taxa["names"][g], R.loci["names"][li])
                row = blk[taxa.index(g)]; assert int(((row != GAP) & (row != MISSING)).sum()) >= int(fs[g, li]) > 0, (name, R.taxa["names"][g], R.loci["names"][li]); checked += 1
    if not checked: pytest.skip("no broad-sample records")


def check_letters(R, ds, name):
    """the three facts on one report: every stored row is the alignment's, every masked cell keeps its letters, every
    broad-sample record's focal and partner have letters"""
    names, rows, loci = lo.read_input(ds.aln, ds.parts)[:3]; fs = R.arr("cells/focal_sites.npy"); side = R.arr("cells/side.npy")
    for li in range(R.nloc):
        taxa, blk = seq_rows(R, li)
        for k, g in enumerate(taxa): assert np.array_equal(blk[k], rows[g, loci[li][1]]), (name, R.loci["names"][li], names[g])
        for g in np.flatnonzero((side[:, li] & 3) > 0).tolist():
            assert g in taxa and int(((blk[taxa.index(g)] != GAP) & (blk[taxa.index(g)] != MISSING)).sum()) >= int(fs[g, li]) > 0, (name, names[g], R.loci["names"][li])
        if R.has(f"broadsample/{li}.json"):
            for rec in R.js(f"broadsample/{li}.json"):
                for g in (int(rec["taxon"]), int(rec["partner"])):
                    assert g in taxa and int(((blk[taxa.index(g)] != GAP) & (blk[taxa.index(g)] != MISSING)).sum()) >= int(fs[g, li]) > 0, (name, names[g], R.loci["names"][li])


@pytest.mark.parametrize("name", ["aa_random_trees", "dna_rc"])
def test_under_the_fails_policy_the_broad_blocks_rows_never_replace_the_main_passs(tmp_out, name):
    """the case that went wrong needs the 'fails' sequence policy (a large alignment's): the main pass carries only the
    failing cells' rows and the broad block adds its own afterwards, read from buffers the main pass has blanked by then.
    The fixtures are small, so the policy is forced; aa_random_trees has hundreds of notes on main-masked cells whose
    partner the main tests also masked, the exact pattern of Wu_2018_aa GALEO_VAR"""
    ds = dataset(name); r = run_tool(ds.aln, tmp_out, ds.dtype, ds.parts, ["--report-sequences", "fails", "--threads", "4"]); assert r.rc == 0, r.log
    R = r.report; assert R.man["carried"]["sequences"] == "fails"
    check_letters(R, ds, name)
    if name == "aa_random_trees":
        side = R.arr("cells/side.npy"); side2 = R.arr("cells/side2.npy"); pattern = 0
        for li in range(R.nloc):
            for rec in (R.js(f"broadsample/{li}.json") if R.has(f"broadsample/{li}.json") else []):
                g, p = int(rec["taxon"]), int(rec["partner"])
                if (side[g, li] & 3) and (side2[g, li] & 32) and (side[p, li] & 3): pattern += 1
        assert pattern >= 1, "the fixture no longer carries a note on a main-masked cell with a main-masked partner"
