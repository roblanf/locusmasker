"""The masked-alignment writer in every format, the progress and log helpers."""
import io, os, sys, time
import numpy as np
import pytest
from tests.conftest import lo, dataset


class TestWriteMasked:
    def rows(self):
        return ["a", "b", "c"], [bytearray(b"ACGTACGT--"), bytearray(b"----ACGTAC"), bytearray(b"ACGTACGTAC")]

    def test_fasta_round_trip(self, tmp_path):
        names, rows = self.rows(); out = lo.write_masked("/x/y/aln.fasta", str(tmp_path), "fasta", names, rows, None)
        assert out == str(tmp_path / "aln.masked.fasta")
        n2, s2 = lo.read_fasta(open(out).read()); assert n2 == names and s2 == [bytes(r).decode() for r in rows]

    def test_phylip_round_trip(self, tmp_path):
        names, rows = self.rows(); out = lo.write_masked("/x/aln.phy", str(tmp_path), "phylip", names, rows, None)
        n2, s2 = lo.read_phylip(open(out).read()); assert n2 == names and s2 == [bytes(r).decode() for r in rows]

    def test_output_name_without_an_extension_takes_the_format(self, tmp_path):
        names, rows = self.rows()
        assert lo.write_masked("/x/aln", str(tmp_path), "fasta", names, rows, None).endswith("aln.masked.fasta")

    def test_nexus_fast_path_round_trip(self, tmp_path):
        names, rows = self.rows(); charsets = {"l1": [0, 1, 2, 3], "l2": [4, 5, 6, 7, 8, 9]}
        info = lo.NexusInfo("dna", "?", "-", charsets, {"og": ["a"]})
        out = lo.write_masked("/x/aln.nex", str(tmp_path), "nexus", names, rows, info)
        n2, r2, c2, i2 = lo.read_nexus_fast(out)
        assert n2 == names and [bytes(r) for r in r2] == [bytes(r) for r in rows] and c2 == charsets
        assert i2.datatype == "dna" and i2.missing == "?" and i2.gap == "-" and i2.taxsets == {"og": ["a"]}

    def test_nexus_fast_path_quotes_names_that_need_it(self, tmp_path):
        pytest.importorskip("Bio.Nexus"); from Bio.Nexus import Nexus
        names = ["plain", "with space", "a-b"]; rows = [bytearray(b"ACGT")] * 3
        info = lo.NexusInfo("dna", None, None, {"l": [0, 1, 2, 3]}, {})
        out = lo.write_masked("/x/aln.nex", str(tmp_path), "nexus", names, rows, info)
        nex = Nexus.Nexus(out); assert list(nex.taxlabels) == names and sorted(nex.charsets["l"]) == [0, 1, 2, 3]

    def test_nexus_through_biopython_round_trip(self, tmp_path):
        pytest.importorskip("Bio.Nexus"); from Bio.Nexus import Nexus
        ds = dataset("dna_small"); names, rows, loci, fmt, declared, nexus = lo.read_input(ds.file("nexus_quirky"), None)
        rows[0, :50] = ord("-")
        out = lo.write_masked(ds.file("nexus_quirky"), str(tmp_path), "nexus", names, rows, nexus)
        nex = Nexus.Nexus(out)
        assert list(nex.taxlabels) == names and str(nex.matrix[names[0]])[:50] == "-" * 50
        assert [sorted(nex.charsets[k]) for k, _ in loci] == [s.tolist() for _, s in loci]

    def test_masked_rows_are_written_as_given(self, tmp_path):
        names, rows = self.rows(); rows[1][:] = b"-" * 10
        out = lo.write_masked("/x/aln.fasta", str(tmp_path), "fasta", names, rows, None)
        assert lo.read_fasta(open(out).read())[1][1] == "-" * 10


class TestProgress:
    def capture(self, mode, fn):
        buf = io.StringIO(); P = lo.Progress(mode); P.out = buf; fn(P); return buf.getvalue()

    def test_none_is_silent(self):
        out = self.capture("none", lambda P: (P.start("s", 10), P.tick(5), P.finish()))
        assert out == ""

    def test_sparse_mode_reports_stage_done_total(self):
        out = self.capture("sparse", lambda P: (P.start("counts", 4), P.tick(), P.tick(3), P.finish()))
        lines = [l.split("\t") for l in out.strip().split("\n")]
        assert all(l[0] == "progress" and l[1] == "counts" for l in lines) and lines[-1][2] == "4" and lines[-1][3] == "4"

    def test_sparse_mode_emits_at_steps(self):
        out = self.capture("sparse", lambda P: (P.start("s", 1000), [P.tick() for _ in range(1000)], P.finish()))
        n = out.count("progress\t"); assert 40 <= n <= 60

    def test_bar_mode_draws_and_finishes_with_a_newline(self):
        out = self.capture("bar", lambda P: (P.start("test", 2, "x"), P.tick(), P.tick(), P.finish()))
        assert out.startswith("\rtest") and out.rstrip("\x1b[K").endswith("\n") or "\n" in out

    def test_durations_recorded_per_stage(self):
        P = lo.Progress("none"); P.start("a", 1); P.tick(); P.start("b", 1); P.finish()
        assert set(P.durations) == {"a", "b"} and all(v >= 0 for v in P.durations.values())

    def test_tick_before_start_is_ignored(self):
        P = lo.Progress("sparse"); P.out = io.StringIO(); P.tick(); assert P.out.getvalue() == ""

    def test_there_is_no_event_protocol(self):
        assert not hasattr(lo.Progress, "event")

    def test_auto_picks_bar_on_a_tty_else_sparse(self, monkeypatch):
        class T:
            def isatty(self): return True
            encoding = "utf-8"
        monkeypatch.setattr(sys, "stderr", T()); assert lo.Progress("auto").mode == "bar"
        monkeypatch.setattr(sys, "stderr", io.StringIO()); assert lo.Progress("auto").mode == "sparse"

    def test_extra_text_is_carried(self):
        out = self.capture("sparse", lambda P: (P.start("s", 2), P.tick(extra="3 masked"), P.finish()))
        assert "3 masked" in out


class TestLog:
    def test_writes_to_stderr_and_file(self, tmp_path, monkeypatch):
        buf = io.StringIO(); L = lo._Log(); L.real = buf; L.open(str(tmp_path / "run.log"))
        L("hello"); L.detail("fine print"); L.close()
        assert buf.getvalue() == "hello\nfine print\n" and open(tmp_path / "run.log").read() == "hello\nfine print\n"

    def test_detail_stays_off_the_terminal_while_a_bar_is_drawn(self):
        buf = io.StringIO(); L = lo._Log(); L.real = buf; L.bar = True
        L.detail("fine print"); L("line")
        assert "fine print" not in buf.getvalue() and buf.getvalue().endswith("line\n") and buf.getvalue().startswith("\r")
