"""Broken alignments and broken partition files. Every case must fail with the tool's own message, one that names the
problem (and the file, the partition or the site), never with a traceback; the cases that should succeed must warn."""
import gzip, os, re
import numpy as np
import pytest
from tests.conftest import lo, dataset, run_tool

pytestmark = pytest.mark.e2e


def fasta_rows():
    ds = dataset("dna_small"); names, seqs = lo.read_fasta(open(ds.aln).read()); return ds, names, seqs


def write_fasta(path, names, seqs):
    with open(path, "w") as fh:
        for n, s in zip(names, seqs): fh.write(f">{n}\n{s}\n")


def parts_text(ds, n=None):
    return "".join(f"DNA, locus{i + 1} = {a}-{b}\n" for i, (a, b) in enumerate(ds.truth["locus_bounds"][:n]))


def failed(r, *words):
    """the run failed with a message that holds every word, and no traceback"""
    assert r.rc != 0, "the run did not fail\n" + r.log
    assert "Traceback" not in r.log, r.log
    for w in words: assert w.lower() in r.log.lower(), (w, r.log)


class TestBrokenAlignments:
    def test_empty_file(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); f = tmp_path / "empty.fasta"; f.write_text("")
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), "empty")

    def test_only_a_header(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); f = tmp_path / "h.fasta"; f.write_text(">a\n")
        failed(run_tool(str(f), tmp_out, "dna", ds.parts))

    def test_unequal_lengths(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); seqs[3] = seqs[3][:-10]; f = tmp_path / "a.fasta"; write_fasta(f, names, seqs)
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), "not all the same length", names[3], f"{len(seqs[3]):,}", f"{len(seqs[0]):,}", "alignment")

    def test_duplicate_names(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); names[5] = names[2]; f = tmp_path / "a.fasta"; write_fasta(f, names, seqs)
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), "duplicate", names[2])

    def test_a_record_with_no_sequence(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); f = tmp_path / "a.fasta"
        with open(f, "w") as fh:
            fh.write(f">{names[0]}\n")
            for n, s in zip(names[1:], seqs[1:]): fh.write(f">{n}\n{s}\n")
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), names[0])

    def test_a_gzipped_alignment_is_refused_with_a_message(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); f = tmp_path / "a.fasta.gz"
        with gzip.open(f, "wb") as fh: fh.write(open(ds.aln, "rb").read())
        failed(run_tool(str(f), tmp_out, "dna", ds.parts))

    def test_a_directory_as_the_alignment(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); failed(run_tool(str(tmp_path), tmp_out, "dna", ds.parts))

    def test_a_missing_alignment(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); failed(run_tool(str(tmp_path / "nope.fasta"), tmp_out, "dna", ds.parts), "nope.fasta")

    def test_protein_letters_with_datatype_dna(self, tmp_out):
        """most letters fall outside the alphabet: the run must refuse and name the likely cause, not analyse the A, C, G and T amino acids as nucleotides"""
        ds = dataset("aa_small"); failed(run_tool(ds.aln, tmp_out, "dna", ds.parts), "illegal characters", "protein alignment", "--datatype aa")

    def test_dna_letters_with_datatype_aa(self, tmp_out):
        """the other way round every letter is a valid amino acid, so the check cannot fire; the run is then the user's responsibility"""
        ds = dataset("dna_small"); r = run_tool(ds.aln, tmp_out, "aa", ds.parts); assert "Traceback" not in r.log
        assert re.search(r"WARNING[^\n]*looks like a nucleotide alignment", r.log), r.log

    def test_many_ambiguity_codes_are_a_warning(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); rng = np.random.default_rng(0)
        seqs = ["".join(c if rng.random() > 0.1 else "N" for c in s) for s in seqs]; f = tmp_path / "a.fasta"; write_fasta(f, names, seqs)
        r = run_tool(str(f), tmp_out, "dna", ds.parts); assert r.rc == 0 and re.search(r"WARNING[^\n]*ambiguity codes", r.log), r.log

    def test_nexus_with_the_wrong_dimensions(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); txt = open(ds.file("nexus")).read().replace("ntax=30", "ntax=31"); f = tmp_path / "a.nex"; f.write_text(txt)
        failed(run_tool(str(f), tmp_out, "dna"), "nexus")

    def test_nexus_truncated(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); txt = open(ds.file("nexus")).read(); f = tmp_path / "a.nex"; f.write_text(txt[:len(txt) // 2])
        failed(run_tool(str(f), tmp_out, "dna"), "nexus")

    def test_nexus_declares_another_datatype(self, tmp_out):
        failed(run_tool(dataset("dna_small").file("nexus"), tmp_out, "aa"), "declares", "dna")

    def test_phylip_with_wrong_counts(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); txt = open(ds.file("phylip")).read().replace("30 ", "29 ", 1); f = tmp_path / "a.phy"; f.write_text(txt)
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), "phylip")

    def test_too_few_taxa_says_how_many(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); f = tmp_path / "a.fasta"; write_fasta(f, names[:4], seqs[:4])
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), "4 taxa", "at least 5")

    def test_crlf_and_lowercase_are_fine(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); f = tmp_path / "a.fasta"
        with open(f, "w", newline="") as fh:
            for n, s in zip(names, seqs): fh.write(f">{n}\r\n{s.lower()}\r\n")
        r = run_tool(str(f), tmp_out, "dna", ds.parts); assert r.rc == 0 and open(r.file("masked.csv")).read() == open(dataset("dna_small").dir + "/../../golden/dna_small/masked.csv").read()


class TestBrokenPartitions:
    def run_parts(self, tmp_path, tmp_out, text, extra=()):
        ds = dataset("dna_small"); p = tmp_path / "p.txt"; p.write_text(text); return run_tool(ds.aln, tmp_out, "dna", str(p), extra)

    def test_overlapping_ranges_name_the_site_and_both_partitions(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); b = ds.truth["locus_bounds"]; text = parts_text(ds).replace(f"locus2 = {b[1][0]}-{b[1][1]}", f"locus2 = {b[1][0] - 5}-{b[1][1]}")
        failed(self.run_parts(tmp_path, tmp_out, text), "locus1", "locus2", str(b[1][0] - 5), "once")

    def test_a_single_site_listed_twice(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); b = ds.truth["locus_bounds"]; text = parts_text(ds) + f"extra = {b[3][0] + 7}\n"
        failed(self.run_parts(tmp_path, tmp_out, text), "locus4", "extra", str(b[3][0] + 7))

    def test_sites_beyond_the_alignment(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); L = ds.truth["sites"]; text = parts_text(ds, 13) + f"locus14 = {L - 50}-{L + 10}\n"
        failed(self.run_parts(tmp_path, tmp_out, text), "locus14", f"{L + 10:,}", f"{L:,}")

    def test_zero_is_not_a_site(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); text = parts_text(ds).replace("locus1 = 1-", "locus1 = 0-")
        failed(self.run_parts(tmp_path, tmp_out, text), "locus1", "1-based")

    def test_reversed_range(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); b = ds.truth["locus_bounds"][0]; text = parts_text(ds).replace(f"locus1 = {b[0]}-{b[1]}", f"locus1 = {b[1]}-{b[0]}")
        failed(self.run_parts(tmp_path, tmp_out, text), "locus1", f"{b[1]}-{b[0]}")

    def test_garbage_in_a_range(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); text = parts_text(ds).replace("locus3 = ", "locus3 = abc, ")
        failed(self.run_parts(tmp_path, tmp_out, text), "abc", "locus3")

    def test_an_empty_partition(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); failed(self.run_parts(tmp_path, tmp_out, parts_text(ds) + "empty = \n"), "empty", "no sites")

    def test_a_name_used_twice(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); text = parts_text(ds).replace("locus5 =", "locus4 =", 1); failed(self.run_parts(tmp_path, tmp_out, text), "locus4", "twice")

    def test_a_partition_file_that_is_not_one(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); failed(self.run_parts(tmp_path, tmp_out, open(ds.aln).read()), "no loci")

    def test_only_comments(self, tmp_path, tmp_out):
        failed(self.run_parts(tmp_path, tmp_out, "# nothing here\n\n# still nothing\n"), "no loci")

    def test_fewer_than_ten_partitions(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); failed(self.run_parts(tmp_path, tmp_out, parts_text(ds, 9)), "9 loci", "at least 10")

    def test_a_missing_partition_file(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); failed(run_tool(ds.aln, tmp_out, "dna", str(tmp_path / "nope.txt")), "nope.txt")

    def test_partitions_with_a_nexus_that_has_charsets(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); failed(run_tool(ds.file("nexus"), tmp_out, "dna", ds.parts), "charsets", "partition")

    def test_overlapping_nexus_charsets_name_the_site_and_the_charsets(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); b = ds.truth["locus_bounds"]; txt = open(ds.file("nexus")).read().replace(f"charset locus2 = {b[1][0]}-{b[1][1]}", f"charset locus2 = {b[1][0] - 5}-{b[1][1]}")
        f = tmp_path / "a.nex"; f.write_text(txt); failed(run_tool(str(f), tmp_out, "dna"), "locus1", "locus2", str(b[1][0] - 5))

    def test_uncovered_sites_are_a_loud_warning_not_an_error(self, tmp_path, tmp_out):
        """sites in no partition are legal (not every column is a locus) but must be announced with their count"""
        ds = dataset("dna_small"); r = self.run_parts(tmp_path, tmp_out, parts_text(ds, 12)); b = ds.truth["locus_bounds"]
        uncovered = ds.truth["sites"] - b[11][1]
        assert r.rc == 0 and re.search(rf"warning[^\n]*{uncovered:,}[^\n]*site", r.log, re.I), r.log
        assert r.report.nloc == 12

    def test_a_locus_too_short_to_test_is_reported_not_tested(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); b = ds.truth["locus_bounds"]; text = parts_text(ds, 13) + f"tiny = {b[13][0]}-{b[13][0] + 4}\n"
        r = self.run_parts(tmp_path, tmp_out, text); assert r.rc == 0 and r.report.loci["status"][13] != "tested" and "tiny" in r.log and "not tested" in r.log


class TestUnaligned:
    """unaligned sequences in every format: the message names the sequence and the lengths"""
    def test_fasta_one_longer(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); seqs[7] = seqs[7] + "ACGT"; f = tmp_path / "a.fasta"; write_fasta(f, names, seqs)
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), "not all the same length", names[7], f"{len(seqs[7]):,}")

    def test_fasta_many_lengths(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); rng = np.random.default_rng(0); seqs = [s[:rng.integers(100, len(s))] for s in seqs]; f = tmp_path / "a.fasta"; write_fasta(f, names, seqs)
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), "not all the same length", "and", "more", "alignment")

    def test_phylip_one_shorter(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); seqs[2] = seqs[2][:-7]; f = tmp_path / "a.phy"
        f.write_text(f"{len(names)} {len(seqs[0])}\n" + "".join(f"{n} {s}\n" for n, s in zip(names, seqs)))
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), "PHYLIP", names[2], f"{len(seqs[2]):,}", "not aligned")

    def test_phylip_header_disagrees_with_every_sequence(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); f = tmp_path / "a.phy"
        f.write_text(f"{len(names)} {len(seqs[0]) + 5}\n" + "".join(f"{n} {s}\n" for n, s in zip(names, seqs)))
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), "PHYLIP")

    def test_nexus_one_shorter(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); txt = open(ds.file("nexus")).read().split("\n"); i = next(i for i, l in enumerate(txt) if l.startswith("t05 ")); txt[i] = txt[i][:-12]
        f = tmp_path / "a.nex"; f.write_text("\n".join(txt)); failed(run_tool(str(f), tmp_out, "dna"), "t05", "nchar", "not aligned")

    def test_nexus_one_longer(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); txt = open(ds.file("nexus")).read().split("\n"); i = next(i for i, l in enumerate(txt) if l.startswith("t05 ")); txt[i] = txt[i] + "ACGTACGT"
        f = tmp_path / "a.nex"; f.write_text("\n".join(txt)); failed(run_tool(str(f), tmp_out, "dna"), "nexus")

    def test_quirky_nexus_one_shorter_through_biopython(self, tmp_path, tmp_out):
        pytest.importorskip("Bio.Nexus")
        ds = dataset("dna_small"); txt = open(ds.file("nexus_quirky")).read().split("\n"); i = max(i for i, l in enumerate(txt) if "'t05'" in l); txt[i] = txt[i][:-5]
        f = tmp_path / "a.nex"; f.write_text("\n".join(txt)); failed(run_tool(str(f), tmp_out, "dna"), "Bio.Nexus", "aligned")


class TestPartitionGarbage:
    @pytest.mark.parametrize("rhs,words", [("-5-10", ["cannot parse", "-5-10"]), ("1.5-10", ["cannot parse", "1.5-10"]), ("1e3", ["cannot parse", "1e3"]), ("1-", ["cannot parse"]), ("5--10", ["cannot parse"]),
                                           ("10-5", ["10-5", "a <= b"]), ("1-10\\0", ["1-10"]), ("1-1000000000", ["runs past", "1,000,000,000"]), ("99999999999999999999-99999999999999999999", ["runs past"]), ("one-ten", ["cannot parse", "one-ten"])])
    def test_garbage_ranges(self, tmp_path, tmp_out, rhs, words):
        ds = dataset("dna_small"); text = parts_text(ds, 13) + f"bad = {rhs}\n"
        p = tmp_path / "p.txt"; p.write_text(text); failed(run_tool(ds.aln, tmp_out, "dna", str(p)), *words)

    def test_a_partition_that_covers_the_whole_alignment_twice(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); L = ds.truth["sites"]; p = tmp_path / "p.txt"; p.write_text(parts_text(ds) + f"all = 1-{L}\n")
        failed(run_tool(ds.aln, tmp_out, "dna", str(p)), "site 1 is in partition all and in partition locus1")

    def test_binary_partition_file(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); p = tmp_path / "p.txt"; p.write_bytes(bytes(range(256)) * 10)
        r = run_tool(ds.aln, tmp_out, "dna", str(p)); assert r.rc != 0 and "Traceback" not in r.log


class TestIllegalCharacters:
    """as Bio.Nexus: anything outside the datatype's symbols, the ambiguity codes, the gap and the missing symbol is an error"""
    def run_with(self, tmp_path, tmp_out, dtype, char, taxon=4, site=100):
        ds = dataset("dna_small" if dtype == "dna" else "aa_small"); names, seqs = lo.read_fasta(open(ds.aln).read())
        s = list(seqs[taxon]); s[site - 1] = char; seqs[taxon] = "".join(s); f = tmp_path / "a.fasta"; write_fasta(f, names, seqs)
        return run_tool(str(f), tmp_out, dtype, ds.parts), names[taxon]

    @pytest.mark.parametrize("char", ["%", '"', "!", "#", "1", ".", "~", "@", "E", "J", "O"])
    def test_dna(self, tmp_path, tmp_out, char):
        r, name = self.run_with(tmp_path, tmp_out, "dna", char); failed(r, "illegal characters", repr(char), name, "site 100", "dna alphabet")

    @pytest.mark.parametrize("char", ["%", '"', "!", "1", ".", "~", "J", "O", "U"])
    def test_protein(self, tmp_path, tmp_out, char):
        r, name = self.run_with(tmp_path, tmp_out, "aa", char); failed(r, "illegal characters", repr(char), name, "site 100", "aa alphabet")

    @pytest.mark.parametrize("dtype,char", [("dna", "N"), ("dna", "R"), ("dna", "n"), ("dna", "u"), ("dna", "?"), ("dna", "-"), ("aa", "X"), ("aa", "B"), ("aa", "Z"), ("aa", "*"), ("aa", "-"), ("aa", "?")])
    def test_legal_characters_run(self, tmp_path, tmp_out, dtype, char):
        r, _ = self.run_with(tmp_path, tmp_out, dtype, char); assert r.rc == 0, r.log

    def test_every_illegal_character_is_listed_with_its_count(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); s = list(seqs[1]); s[10] = "%"; s[20] = "%"; s[30] = "$"; seqs[1] = "".join(s); f = tmp_path / "a.fasta"; write_fasta(f, names, seqs)
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), "'%' x2 (first in t01 at site 11)", "'$' x1 (first in t01 at site 31)")

    def test_non_ascii_bytes(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); seqs[0] = seqs[0][:50] + "é" + seqs[0][51:]; f = tmp_path / "a.fasta"; f.write_text("".join(f">{n}\n{s}\n" for n, s in zip(names, seqs)), encoding="utf-8")
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), "non-ASCII", "é", names[0], "site 51")

    def test_nexus_declared_symbols_are_legal(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); txt = open(ds.file("nexus")).read().replace("missing=? gap=-", "missing=~ gap=.")
        txt = txt.replace("-", ".") if False else txt
        lines = txt.split("\n"); out = []
        for l in lines:
            if l.startswith("t") and " " in l and not l.startswith("taxset"): nm, sq = l.split(" ", 1); out.append(nm + " " + sq.replace("-", "."))
            else: out.append(l)
        f = tmp_path / "a.nex"; f.write_text("\n".join(out)); r = run_tool(str(f), tmp_out, "dna"); assert r.rc == 0, r.log
        f2 = tmp_path / "b.nex"; f2.write_text("\n".join(out).replace("missing=~ gap=.", "missing=? gap=-")); failed(run_tool(str(f2), tmp_out + "2", "dna"), "illegal characters", "'.'")

    def test_quirky_nexus_illegal_character_through_biopython(self, tmp_path, tmp_out):
        pytest.importorskip("Bio.Nexus")
        ds = dataset("dna_small"); txt = open(ds.file("nexus_quirky")).read().replace("'t05'", "'t05'", 1); lines = txt.split("\n"); i = max(i for i, l in enumerate(lines) if "'t05'" in l)
        lines[i] = lines[i][:-1] + "%"; f = tmp_path / "a.nex"; f.write_text("\n".join(lines)); failed(run_tool(str(f), tmp_out, "dna"), "Illegal character %")


class TestNames:
    """taxon names: anything printable is a name; the outputs carry it unchanged and the NEXUS writer quotes what needs quoting"""
    ODD = ["with-dash", "slash/name", "pipe|name", "colon:name", "paren(name)", "quote'name", 'dq"name', "hash#name", "semi;name", "comma,name", "unicode_Æ_名", "123", "x" * 400, "UPPER", "upper"]

    def renamed(self, tmp_path):
        ds, names, seqs = fasta_rows(); names = self.ODD + names[len(self.ODD):]; f = tmp_path / "a.fasta"; write_fasta(f, names, seqs); return ds, names, seqs, str(f)

    def test_fasta_names_survive_every_output(self, tmp_path, tmp_out):
        ds, names, seqs, f = self.renamed(tmp_path); r = run_tool(f, tmp_out, "dna", ds.parts); assert r.rc == 0, r.log
        assert lo.read_fasta(open(r.file("a.masked.fasta")).read())[0] == names and r.report.taxa["names"] == names
        for row in r.masked: assert row["taxon"] in names

    def test_names_differing_only_by_case_are_different_taxa(self, tmp_path, tmp_out):
        ds, names, seqs, f = self.renamed(tmp_path); r = run_tool(f, tmp_out, "dna", ds.parts); assert "UPPER" in r.report.taxa["names"] and "upper" in r.report.taxa["names"]

    def test_nexus_output_quotes_what_biopython_needs(self, tmp_path, tmp_out):
        pytest.importorskip("Bio.Nexus"); from Bio.Nexus import Nexus
        ds, names, seqs, f = self.renamed(tmp_path); p = tmp_path / "a.nex"
        with open(p, "w") as fh:
            fh.write(f"#NEXUS\nbegin data;\ndimensions ntax={len(names)} nchar={len(seqs[0])};\nformat datatype=dna missing=? gap=-;\nmatrix\n")
            for n, s in zip(names, seqs): fh.write(f"{lo._safename(n)} {s}\n")
            fh.write(";\nend;\nbegin sets;\n" + "".join(f"charset locus{i + 1} = {a}-{b};\n" for i, (a, b) in enumerate(ds.truth["locus_bounds"])) + "end;\n")
        r = run_tool(str(p), tmp_out, "dna"); assert r.rc == 0, r.log
        nex = Nexus.Nexus(r.file("a.masked.nex")); assert list(nex.taxlabels) == names

    def test_an_empty_name_is_an_error(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); f = tmp_path / "a.fasta"
        with open(f, "w") as fh:
            fh.write(f"> \n{seqs[0]}\n"); [fh.write(f">{n}\n{s}\n") for n, s in zip(names[1:], seqs[1:])]
        failed(run_tool(str(f), tmp_out, "dna", ds.parts), "no name")

    def test_a_name_that_is_only_whitespace_in_phylip(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); f = tmp_path / "a.phy"
        f.write_text(f"{len(names)} {len(seqs[0])}\n" + f"   {seqs[0]}\n" + "".join(f"{n} {s}\n" for n, s in zip(names[1:], seqs[1:])))
        r = run_tool(str(f), tmp_out, "dna", ds.parts); assert r.rc != 0 and "Traceback" not in r.log

    def test_phylip_long_names(self, tmp_path, tmp_out):
        ds, names, seqs = fasta_rows(); names = [n + "_" + "y" * 60 for n in names]; f = tmp_path / "a.phy"
        f.write_text(f"{len(names)} {len(seqs[0])}\n" + "".join(f"{n} {s}\n" for n, s in zip(names, seqs)))
        r = run_tool(str(f), tmp_out, "dna", ds.parts); assert r.rc == 0 and lo.read_phylip(open(r.file("a.masked.phy")).read())[0] == names
