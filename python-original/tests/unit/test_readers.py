"""The input readers: FASTA, PHYLIP, RAxML partitions, the fast NEXUS reader, and read_input's dispatch and checks.
Every test states what a reader should do with an input, not what it happens to do."""
import os
import numpy as np
import pytest
from tests.conftest import lo, dataset


# ------------------------------------------------------------------ FASTA
class TestReadFasta:
    def test_basic(self):
        names, seqs = lo.read_fasta(">a\nACGT\n>b\nAC-T\n")
        assert names == ["a", "b"] and seqs == ["ACGT", "AC-T"]

    def test_multiline_sequences_are_joined(self):
        names, seqs = lo.read_fasta(">a\nACG\nT\n\nAA\n>b\nC\n")
        assert seqs == ["ACGTAA", "C"]

    def test_name_is_the_first_token_of_the_header(self):
        names, _ = lo.read_fasta(">taxon_1 some description here\nACGT\n")
        assert names == ["taxon_1"]

    def test_whitespace_inside_sequence_lines_is_dropped(self):
        _, seqs = lo.read_fasta(">a\nAC GT\tAC\n")
        assert seqs == ["ACGTAC"]

    def test_crlf_line_endings(self):
        names, seqs = lo.read_fasta(">a\r\nACGT\r\n>b\r\nAAAA\r\n")
        assert names == ["a", "b"] and seqs == ["ACGT", "AAAA"]

    def test_case_is_preserved_for_the_caller(self):
        _, seqs = lo.read_fasta(">a\nacgtN\n")
        assert seqs == ["acgtN"]

    def test_no_trailing_newline(self):
        names, seqs = lo.read_fasta(">a\nACGT")
        assert names == ["a"] and seqs == ["ACGT"]

    def test_empty_input(self):
        assert lo.read_fasta("") == ([], [])

    def test_header_without_a_name_is_an_error(self):
        with pytest.raises(SystemExit):
            lo.read_fasta(">\nACGT\n")
        with pytest.raises(SystemExit):
            lo.read_fasta(">   \nACGT\n")

    def test_names_and_sequences_always_pair_up(self):
        """a record with an empty sequence (a header followed by another header) must not shift every later sequence
        onto the wrong name: it is kept as an empty sequence, which read_input then rejects for its length"""
        names, seqs = lo.read_fasta(">a\n>b\nACGT\n>c\nAAAA\n")
        assert len(names) == len(seqs) == 3
        assert dict(zip(names, seqs)) == {"a": "", "b": "ACGT", "c": "AAAA"}

    def test_text_before_the_first_header_is_an_error(self):
        """sequence text with no header is not a record: it must not become a nameless sequence that misaligns the rest"""
        with pytest.raises(SystemExit, match="before the first header"): lo.read_fasta("ACGT\n>a\nACGT\n>b\nAAAA\n")
        names, seqs = lo.read_fasta("\n\n>a\nACGT\n"); assert names == ["a"] and seqs == ["ACGT"]       # blank lines are fine


# ------------------------------------------------------------------ PHYLIP
class TestReadPhylip:
    def test_relaxed_one_line(self):
        names, seqs = lo.read_phylip("3 8\nalpha ACGTACGT\nbeta  AAAATTTT\ngamma CCCCGGGG\n")
        assert names == ["alpha", "beta", "gamma"] and seqs == ["ACGTACGT", "AAAATTTT", "CCCCGGGG"]

    def test_spaces_inside_a_sequence(self):
        names, seqs = lo.read_phylip("1 8\na ACGT ACGT\n")
        assert seqs == ["ACGTACGT"]

    def test_interleaved_names_in_the_first_block_only(self):
        txt = "2 10\na ACGTA\nb CCCCC\n\nCGTAC\nGGGGG\n"
        names, seqs = lo.read_phylip(txt)
        assert names == ["a", "b"] and seqs == ["ACGTACGTAC", "CCCCCGGGGG"]

    def test_interleaved_with_names_repeated(self):
        txt = "2 10\na ACGTA\nb CCCCC\n\na CGTAC\nb GGGGG\n"
        names, seqs = lo.read_phylip(txt)
        assert names == ["a", "b"] and seqs == ["ACGTACGTAC", "CCCCCGGGGG"]

    def test_interleaved_with_names_repeated_and_spaced_chunks(self):
        txt = "2 20\na ACGTA CGTAC\nb CCCCC GGGGG\n\na ACGTA CGTAC\nb CCCCC GGGGG\n"
        names, seqs = lo.read_phylip(txt); assert names == ["a", "b"] and seqs == ["ACGTACGTACACGTACGTAC", "CCCCCGGGGGCCCCCGGGGG"]

    def test_a_sequence_continued_on_the_next_lines(self):
        names, seqs = lo.read_phylip("2 10\na ACGTA\nCGTAC\nb CCCCC\nGGGGG\n"); assert names == ["a", "b"] and seqs == ["ACGTACGTAC", "CCCCCGGGGG"]

    def test_header_may_carry_extra_tokens(self):
        names, seqs = lo.read_phylip("2 4 I\na ACGT\nb AAAA\n"); assert names == ["a", "b"] and seqs == ["ACGT", "AAAA"]

    def test_a_bad_header_is_a_message(self):
        with pytest.raises(SystemExit, match="first line"): lo.read_phylip("two 4\na ACGT\nb AAAA\n")
        with pytest.raises(SystemExit, match="empty"): lo.read_phylip("\n\n")

    def test_header_whitespace_is_free(self):
        names, seqs = lo.read_phylip("  2\t 4 \na ACGT\nb AAAA\n")
        assert names == ["a", "b"]

    def test_blank_lines_anywhere(self):
        names, seqs = lo.read_phylip("\n2 4\n\na ACGT\n\nb AAAA\n\n")
        assert names == ["a", "b"] and seqs == ["ACGT", "AAAA"]

    def test_wrong_length_is_an_error(self):
        with pytest.raises(SystemExit):
            lo.read_phylip("2 4\na ACGT\nb AAA\n")

    def test_wrong_count_is_an_error(self):
        with pytest.raises(SystemExit):
            lo.read_phylip("3 4\na ACGT\nb AAAA\n")

    def test_strict_phylip_without_a_separator_fails_cleanly(self):
        """strict PHYLIP (a ten-character name glued to the sequence) is not supported, and must say so rather than crash"""
        with pytest.raises(SystemExit):
            lo.read_phylip("2 4\nabcdefghijACGT\nklmnopqrstAAAA\n")

    def test_fixture_phylip_matches_fasta(self):
        ds = dataset("dna_small")
        fn, fs = lo.read_fasta(open(ds.file("fasta")).read())
        pn, ps = lo.read_phylip(open(ds.file("phylip")).read())
        assert (pn, ps) == (fn, fs)
        pn, ps = lo.read_phylip(open(ds.file("phylip_interleaved")).read())
        assert (pn, ps) == (fn, fs)


# ------------------------------------------------------------------ RAxML partitions
def parts(tmp_path, text):
    p = tmp_path / "parts.txt"; p.write_text(text); return str(p)


class TestRaxmlPartitions:
    def test_ranges_are_one_based_inclusive(self, tmp_path):
        loci = lo.parse_raxml_partitions(parts(tmp_path, "DNA, locus1 = 1-100\n"), 200)
        assert [k for k, _ in loci] == ["locus1"] and loci[0][1].tolist() == list(range(0, 100))

    def test_model_prefix_is_optional(self, tmp_path):
        loci = lo.parse_raxml_partitions(parts(tmp_path, "locus1 = 1-10\nWAG, locus2 = 11-20\n"), 20)
        assert [k for k, _ in loci] == ["locus1", "locus2"]

    def test_comma_lists_single_sites_and_steps(self, tmp_path):
        loci = lo.parse_raxml_partitions(parts(tmp_path, "p = 1-3, 7, 10-16\\3\n"), 20)
        assert loci[0][1].tolist() == [0, 1, 2, 6, 9, 12, 15]

    def test_sites_are_sorted_and_unique(self, tmp_path):
        loci = lo.parse_raxml_partitions(parts(tmp_path, "p = 9-10, 2, 2-3\n"), 20)
        assert loci[0][1].tolist() == [1, 2, 8, 9]

    def test_comments_blank_lines_and_semicolons(self, tmp_path):
        loci = lo.parse_raxml_partitions(parts(tmp_path, "# a comment\n\nDNA, a = 1-5;  # trailing\n  \nb = 6-10\n"), 10)
        assert [k for k, _ in loci] == ["a", "b"]

    def test_file_order_is_kept(self, tmp_path):
        loci = lo.parse_raxml_partitions(parts(tmp_path, "z = 6-10\na = 1-5\n"), 10)
        assert [k for k, _ in loci] == ["z", "a"]

    @pytest.mark.parametrize("text", ["a = 1-5\na = 6-10\n", "a = 1-5\nb = 3-8\n", "a = 1-50\n", "a = 10-5\n", "a = 0-5\n", "a = 0\n",
                                      "a = 1-5x\n", "a = \n", "a = 5-5\\0\n"])
    def test_bad_definitions_are_errors(self, tmp_path, text):
        with pytest.raises(SystemExit):
            lo.parse_raxml_partitions(parts(tmp_path, text), 20)

    def test_lines_without_an_equals_sign_are_ignored(self, tmp_path):
        loci = lo.parse_raxml_partitions(parts(tmp_path, "just text\na = 1-5\n"), 10)
        assert len(loci) == 1

    def test_fixture_partitions(self):
        ds = dataset("dna_small")
        loci = lo.parse_raxml_partitions(ds.parts, ds.truth["sites"])
        assert len(loci) == ds.nloci
        for (k, sites), (a, b) in zip(loci, ds.truth["locus_bounds"]):
            assert sites.tolist() == list(range(a - 1, b))


# ------------------------------------------------------------------ NEXUS names and ranges as Bio.Nexus writes them
class TestNexusHelpers:
    @pytest.mark.parametrize("name,expect", [("plain", "plain"), ("with space", "'with space'"), ("a-b", "'a-b'"), ("it's", "'it''s'"), ("a(b)", "'a(b)'"), ("tab\there", "'tab\there'")])
    def test_safename(self, name, expect):
        assert lo._safename(name) == expect

    @pytest.mark.parametrize("sites,expect", [([], ""), ([0], "1"), ([0, 1, 2], "1-3"), ([0, 2, 4], "1-5\\2"), ([0, 1, 2, 5], "1-3 6"), ([0, 2], "1 3"), ([3, 1, 1, 0], "1-2 4")])
    def test_compact4nexus(self, sites, expect):
        assert lo._compact4nexus(sites) == expect

    def test_compact4nexus_round_trips_through_biopython(self):
        """whatever the compaction writes, Bio.Nexus must read back as the same sites"""
        Bio = pytest.importorskip("Bio.Nexus")
        from Bio.Nexus import Nexus
        rng = np.random.default_rng(5)
        for _ in range(40):
            sites = sorted(set(rng.integers(0, 60, int(rng.integers(1, 25))).tolist()))
            txt = lo._compact4nexus(sites)
            nex = Nexus.Nexus(f"#NEXUS\nbegin data;\ndimensions ntax=1 nchar=60;\nformat datatype=dna;\nmatrix\na {'A' * 60}\n;\nend;\nbegin sets;\ncharset x = {txt};\nend;\n")
            assert sorted(nex.charsets["x"]) == sites, (sites, txt)


# ------------------------------------------------------------------ the fast NEXUS reader
def nexus_text(seqs, extra_format=" missing=? gap=-", sets="", head="#NEXUS\n", dtype="dna", block="data", interleave=False, end="end;"):
    names = list(seqs)
    L = len(next(iter(seqs.values())))
    if interleave:
        mat = "".join("".join(f"{t} {seqs[t][s:s + 10]}\n" for t in names) + "\n" for s in range(0, L, 10))
    else:
        mat = "".join(f"{t} {seqs[t]}\n" for t in names)
    txt = (f"{head}begin {block};\ndimensions ntax={len(names)} nchar={L};\nformat datatype={dtype}{extra_format};\nmatrix\n{mat};\n{end}\n")
    if sets: txt += f"begin sets;\n{sets}\nend;\n"
    return txt


class TestReadNexusFast:
    SEQS = {"a": "ACGTACGTACGTACGTACGT", "b": "ACGTACGTACGTACGTACGA", "c": "ACGTACGTACGTACGTAC--"}

    def write(self, tmp_path, txt):
        p = tmp_path / "x.nex"; p.write_text(txt); return str(p)

    def test_plain_file(self, tmp_path):
        names, rows, charsets, info = lo.read_nexus_fast(self.write(tmp_path, nexus_text(self.SEQS, " missing=? gap=-", "charset l1 = 1-10;\ncharset l2 = 11-20;")))
        assert names == ["a", "b", "c"] and [bytes(r).decode() for r in rows] == list(self.SEQS.values())
        assert charsets == {"l1": list(range(10)), "l2": list(range(10, 20))}
        assert info.datatype == "dna" and info.missing == "?" and info.gap == "-"

    def test_missing_and_gap_default_to_none_when_absent(self, tmp_path):
        _, _, _, info = lo.read_nexus_fast(self.write(tmp_path, nexus_text({"a": "ACGT", "b": "ACGA"}, extra_format="")))
        assert info.missing is None and info.gap is None

    def test_interleaved_matrix(self, tmp_path):
        names, rows, _, _ = lo.read_nexus_fast(self.write(tmp_path, nexus_text(self.SEQS, " missing=? gap=- interleave=yes", interleave=True)))
        assert [bytes(r).decode() for r in rows] == list(self.SEQS.values())

    def test_interleave_without_a_value_means_yes(self, tmp_path):
        names, rows, _, _ = lo.read_nexus_fast(self.write(tmp_path, nexus_text(self.SEQS, " missing=? gap=- interleave", interleave=True)))
        assert [bytes(r).decode() for r in rows] == list(self.SEQS.values())

    def test_keywords_are_case_insensitive(self, tmp_path):
        txt = nexus_text(self.SEQS, sets="CHARSET l1 = 1-20;").replace("begin data;", "BEGIN DATA;").replace("matrix", "MATRIX").replace("dimensions", "DIMENSIONS").replace("format", "FORMAT")
        names, rows, charsets, _ = lo.read_nexus_fast(self.write(tmp_path, txt))
        assert names == ["a", "b", "c"] and charsets == {"l1": list(range(20))}

    def test_characters_block_and_endblock(self, tmp_path):
        names, rows, _, _ = lo.read_nexus_fast(self.write(tmp_path, nexus_text(self.SEQS, block="characters", end="endblock;")))
        assert names == ["a", "b", "c"]

    def test_crlf(self, tmp_path):
        names, rows, charsets, _ = lo.read_nexus_fast(self.write(tmp_path, nexus_text(self.SEQS, sets="charset l1 = 1-20;").replace("\n", "\r\n")))
        assert names == ["a", "b", "c"] and [bytes(r).decode() for r in rows] == list(self.SEQS.values()) and charsets == {"l1": list(range(20))}

    @pytest.mark.parametrize("dtype,expect", [("dna", "dna"), ("nucleotide", "nucleotide"), ("rna", "rna"), ("protein", "protein"), ("DNA", "dna")])
    def test_datatypes(self, tmp_path, dtype, expect):
        seqs = {"a": "MKLV", "b": "MKLI"} if dtype == "protein" else ({"a": "ACGU", "b": "ACGA"} if dtype == "rna" else {"a": "ACGT", "b": "ACGA"})
        _, _, _, info = lo.read_nexus_fast(self.write(tmp_path, nexus_text(seqs, dtype=dtype)))
        assert info.datatype == expect

    def test_charset_forms(self, tmp_path):
        _, _, charsets, _ = lo.read_nexus_fast(self.write(tmp_path, nexus_text(self.SEQS, sets="charset a = 1-6\\3 8 10-12;\ncharset b = 13;")))
        assert charsets == {"a": [0, 3, 7, 9, 10, 11], "b": [12]}

    def test_taxset(self, tmp_path):
        _, _, _, info = lo.read_nexus_fast(self.write(tmp_path, nexus_text(self.SEQS, sets="taxset og = c a;")))
        assert info.taxsets == {"og": ["a", "c"]}

    @pytest.mark.parametrize("mutate", [
        lambda t: t.replace("#NEXUS", "[comment]\n#NEXUS"), lambda t: t.replace("a ACGT", "'a' ACGT"), lambda t: t.replace("datatype=dna", "datatype=standard"),
        lambda t: t.replace("a ACGT", "b ACGT"), lambda t: t.replace("ntax=3", "ntax=4"),
        lambda t: t + "begin trees;\ntree t = (a,b,c);\nend;\n",
        lambda t: t.replace("begin data;", "begin data;\n") + nexus_text({"a": "ACGT"}).split("#NEXUS\n")[1],
        lambda t: t.replace("format datatype=dna missing=? gap=-", "format datatype=dna missing=? gap=- symbols=\"01\""), lambda t: t.replace("charset l1 = 1-10;", "charset l1 = 1-10 x;"),
        lambda t: t.replace("charset l1 = 1-10;", "charset l1 = 1-30;"), lambda t: t.replace("charset l1 = 1-10;", "charset l1 = 1-10;\ncharset l1 = 11-20;"),
        lambda t: t.replace("end;\n", "end;\nstray text\n", 1), lambda t: t.replace("#NEXUS", "NEXUS")])
    def test_anything_unsupported_defers_to_biopython(self, tmp_path, mutate):
        """the fast reader handles a narrow, plain dialect; everything else raises _Unsupported so Bio.Nexus takes over"""
        txt = mutate(nexus_text(self.SEQS, " missing=? gap=-", "charset l1 = 1-10;"))
        with pytest.raises(lo._Unsupported):
            lo.read_nexus_fast(self.write(tmp_path, txt))

    def test_illegal_characters_are_an_error_with_the_place(self, tmp_path):
        """as Bio.Nexus: a character outside the datatype's symbols is an error that names it, with the taxon and site"""
        with pytest.raises(SystemExit, match=r"illegal characters.*'Z' x1 \(first in c at site 19\)"):
            lo.read_nexus_fast(self.write(tmp_path, nexus_text(self.SEQS).replace("ACGTACGTACGTACGTAC--", "ACGTACGTACGTACGTACZ-")))

    def test_a_wrong_nchar_is_an_error_naming_the_dimensions_line(self, tmp_path):
        with pytest.raises(SystemExit, match="every sequence has 20 characters but the dimensions line says nchar=19"):
            lo.read_nexus_fast(self.write(tmp_path, nexus_text(self.SEQS).replace("nchar=20", "nchar=19")))

    def test_an_unaligned_row_is_an_error_naming_the_taxon(self, tmp_path):
        with pytest.raises(SystemExit, match="taxon c has 19 characters but nchar is 20"):
            lo.read_nexus_fast(self.write(tmp_path, nexus_text(self.SEQS).replace("c ACGTACGTACGTACGTAC--", "c ACGTACGTACGTACGTAC-")))

    def test_fixture_nexus_matches_fasta(self):
        ds = dataset("dna_small")
        fn, fs = lo.read_fasta(open(ds.file("fasta")).read())
        names, rows, charsets, info = lo.read_nexus_fast(ds.file("nexus"))
        assert names == fn and [bytes(r).decode() for r in rows] == fs
        assert list(charsets) == [f"locus{i + 1}" for i in range(ds.nloci)]
        for (a, b), sites in zip(ds.truth["locus_bounds"], charsets.values()): assert sites == list(range(a - 1, b))

    def test_real_excerpts_read_on_the_fast_path(self):
        for name in ("cannon_aa", "near_dna"):
            ds = dataset(name); names, rows, charsets, info = lo.read_nexus_fast(ds.aln)
            assert len(names) == ds.ntaxa and len(charsets) == ds.nloci and all(len(r) == ds.truth["sites"] for r in rows)


# ------------------------------------------------------------------ read_input: dispatch and the checks every input gets
class TestReadInput:
    def test_fasta_with_partitions(self):
        ds = dataset("dna_small"); names, rows, loci, fmt, declared, nexus = lo.read_input(ds.aln, ds.parts)
        assert fmt == "fasta" and declared is None and nexus is None and len(names) == ds.ntaxa and len(loci) == ds.nloci
        assert isinstance(rows, np.ndarray) and rows.dtype == np.uint8 and rows.ndim == 2 and rows.flags.c_contiguous      # one array, a row per taxon

    def test_phylip_with_partitions(self):
        ds = dataset("dna_small"); names, rows, loci, fmt, declared, nexus = lo.read_input(ds.file("phylip"), ds.parts)
        assert fmt == "phylip" and declared is None and len(loci) == ds.nloci

    def test_nexus_with_charsets_declares_its_datatype(self):
        ds = dataset("dna_small"); names, rows, loci, fmt, declared, nexus = lo.read_input(ds.file("nexus"), None)
        assert fmt == "nexus" and declared == "dna" and isinstance(nexus, lo.NexusInfo) and len(loci) == ds.nloci
        ds = dataset("cannon_aa"); _, _, _, _, declared, _ = lo.read_input(ds.aln, None)
        assert declared == "aa"

    def test_quirky_nexus_through_biopython_equals_the_plain_file(self):
        pytest.importorskip("Bio.Nexus")
        ds = dataset("dna_small")
        pn, pr, pl, _, pd, _ = lo.read_input(ds.file("nexus"), None)
        qn, qr, ql, fmt, qd, nexus = lo.read_input(ds.file("nexus_quirky"), None)
        assert fmt == "nexus" and qd == pd and qn == pn and [bytes(r) for r in qr] == [bytes(r) for r in pr]
        assert [(k, s.tolist()) for k, s in ql] == [(k, s.tolist()) for k, s in pl]
        assert not isinstance(nexus, lo.NexusInfo)

    def test_nexus_charsets_and_partitions_together_is_an_error(self):
        ds = dataset("dna_small")
        with pytest.raises(SystemExit, match="one or the other"):
            lo.read_input(ds.file("nexus"), ds.parts)

    def test_nexus_without_charsets_needs_partitions(self, tmp_path):
        p = tmp_path / "x.nex"; p.write_text(nexus_text({"a": "ACGTACGTAC", "b": "ACGTACGTAA"}))
        with pytest.raises(SystemExit, match="charsets"):
            lo.read_input(str(p), None)
        q = tmp_path / "p.txt"; q.write_text("l1 = 1-5\nl2 = 6-10\n")
        names, rows, loci, fmt, declared, nexus = lo.read_input(str(p), str(q))
        assert fmt == "nexus" and [k for k, _ in loci] == ["l1", "l2"] and declared == "dna"

    def test_fasta_without_partitions_is_an_error(self):
        with pytest.raises(SystemExit, match="partitions"):
            lo.read_input(dataset("dna_small").aln, None)

    def test_unequal_lengths_are_an_error(self, tmp_path):
        p = tmp_path / "x.fasta"; p.write_text(">a\nACGT\n>b\nACG\n"); q = tmp_path / "p.txt"; q.write_text("l = 1-4\n")
        with pytest.raises(SystemExit, match="length"):
            lo.read_input(str(p), str(q))

    def test_duplicate_names_are_an_error(self, tmp_path):
        p = tmp_path / "x.fasta"; p.write_text(">a\nACGT\n>a\nACGA\n"); q = tmp_path / "p.txt"; q.write_text("l = 1-4\n")
        with pytest.raises(SystemExit, match="duplicate"):
            lo.read_input(str(p), str(q))

    def test_no_loci_is_an_error(self, tmp_path):
        p = tmp_path / "x.fasta"; p.write_text(">a\nACGT\n>b\nACGA\n"); q = tmp_path / "p.txt"; q.write_text("# nothing\n")
        with pytest.raises(SystemExit, match="no loci"):
            lo.read_input(str(p), str(q))

    def test_overlapping_charsets_are_an_error(self, tmp_path):
        pytest.importorskip("Bio.Nexus")
        p = tmp_path / "x.nex"; p.write_text("#NEXUS\n[c]\nbegin data;\ndimensions ntax=2 nchar=10;\nformat datatype=dna;\nmatrix\na ACGTACGTAC\nb ACGTACGTAA\n;\nend;\nbegin sets;\ncharset l1 = 1-6;\ncharset l2 = 5-10;\nend;\n")
        with pytest.raises(SystemExit, match="site 5 is in charset l2 and in charset l1"):
            lo.read_input(str(p), None)

    def test_a_missing_file_is_a_clean_error(self, tmp_path):
        """a user's typo in the path must produce the tool's own message, not a traceback"""
        with pytest.raises(SystemExit):
            lo.read_input(str(tmp_path / "nope.fasta"), str(tmp_path / "nope.txt"))

    def test_leading_whitespace_before_nexus_header(self, tmp_path):
        p = tmp_path / "x.nex"; p.write_text("\n  " + nexus_text({"a": "ACGTACGTAC", "b": "ACGTACGTAA"}, sets="charset l = 1-10;"))
        names, rows, loci, fmt, declared, nexus = lo.read_input(str(p), None)
        assert fmt == "nexus" and len(loci) == 1
