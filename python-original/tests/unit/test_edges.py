"""The fallback paths: Biopython absent, files that cannot be read, OS queries that fail, the viewer template missing,
and the pair stage's thread pool on a dataset big enough to be chunked."""
import argparse, os, subprocess, sys, time
import numpy as np
import pytest
from tests.conftest import lo, dataset, run_tool


class TestWithoutBiopython:
    def test_a_quirky_nexus_names_the_missing_dependency(self, tmp_out, monkeypatch):
        monkeypatch.setitem(sys.modules, "Bio.Nexus", None)
        r = run_tool(dataset("dna_small").file("nexus_quirky"), tmp_out, "dna"); assert r.rc != 0 and "Biopython" in r.log and "Traceback" not in r.log

    def test_writing_through_biopython_needs_it(self, tmp_path, monkeypatch):
        pytest.importorskip("Bio.Nexus")
        ds = dataset("dna_small"); names, rows, loci, fmt, declared, nexus = lo.read_input(ds.file("nexus_quirky"), None)
        monkeypatch.setitem(sys.modules, "Bio.Seq", None)
        with pytest.raises(SystemExit, match="Biopython"): lo.write_masked(ds.file("nexus_quirky"), str(tmp_path), "nexus", names, rows, nexus)


class TestUnreadableFiles:
    def test_a_directory_as_the_partition_file(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); r = run_tool(ds.aln, tmp_out, "dna", str(tmp_path)); assert r.rc != 0 and "Traceback" not in r.log and "partition file" in r.log

    def test_a_latin1_byte_deep_in_a_fasta(self, tmp_path, tmp_out):
        ds = dataset("dna_small"); b = open(ds.aln, "rb").read(); f = tmp_path / "a.fasta"; f.write_bytes(b[:2000] + b"\xe9" + b[2001:])
        r = run_tool(str(f), tmp_out, "dna", ds.parts); assert r.rc != 0 and "Traceback" not in r.log and "0xE9" in r.log and "not UTF-8" in r.log and "site 1,972" in r.log

    @pytest.mark.parametrize("mutate,expect", [
        (lambda t: t.replace("end;\n", "end;\nbegin assumptions;\noptions deftype=ord;\nend;\n", 1), "block"),
        (lambda t: t.replace("nchar=20;", "nchar=20 nblocks=1;"), "dimensions"),
        (lambda t: t.replace("gap=-;", "gap=- matchchar=.;"), "format option"),
        (lambda t: t.replace("matrix", "options gapmode=missing;\nmatrix", 1), "command")])
    def test_fast_nexus_reader_names_what_it_does_not_handle(self, tmp_path, mutate, expect):
        from tests.unit.test_readers import nexus_text
        txt = mutate(nexus_text({"a": "ACGTACGTACGTACGTACGT", "b": "ACGTACGTACGTACGTACGA"}, sets="charset l = 1-20;")); p = tmp_path / "x.nex"; p.write_text(txt)
        with pytest.raises(lo._Unsupported, match=expect): lo.read_nexus_fast(str(p))


class TestOsFallbacks:
    def test_cpus_without_affinity(self, monkeypatch):
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: (_ for _ in ()).throw(OSError()), raising=False); assert lo._cpus() >= 1

    def test_memory_without_proc(self, monkeypatch):
        real_open = open
        def no_proc(path, *a, **k):
            if str(path).startswith("/proc/meminfo"): raise OSError("no proc")
            return real_open(path, *a, **k)
        monkeypatch.setattr("builtins.open", no_proc); assert lo._mem_available() > 0
        monkeypatch.setattr(os, "sysconf", lambda *a: (_ for _ in ()).throw(ValueError()), raising=False); assert lo._mem_available() == 2 << 30

    def test_git_commit_without_git(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())); assert lo._git_commit() == "unknown"


class TestReportTemplate:
    def test_a_missing_viewer_still_writes_the_data(self, tmp_out, tmp_path, monkeypatch):
        monkeypatch.setattr(lo, "__file__", str(tmp_path / "elsewhere" / "locusmasker.py"))
        ds = dataset("dna_clean"); r = run_tool(ds.aln, tmp_out, "dna", ds.parts)
        assert r.rc == 0 and r.exists("report.npz") and not r.exists("report.html") and "report.html not written" in r.log


class TestPairStagePool:
    def test_chunked_threaded_pair_stage_matches_one_thread(self, tmp_path, monkeypatch):
        """300 taxa in one block pair of 90,000 positions: with the memory reported tiny the medians run in several numpy
        chunks on several threads, and the sweep's structures do not change"""
        rng = np.random.default_rng(3); n, nloc, ll = 300, 12, 40; anc = rng.integers(0, 4, nloc * ll); X = np.tile(anc, (n, 1)); mut = rng.random(X.shape) < 0.1
        X[mut] = (X[mut] + rng.integers(1, 4, int(mut.sum()))) % 4; rows = [bytearray(bytes(np.frombuffer(b"ACGT", dtype=np.uint8)[r])) for r in X]
        names = [f"t{i:03d}" for i in range(n)]; loci = [(f"l{i}", np.arange(i * ll, (i + 1) * ll)) for i in range(nloc)]
        P1 = lo.prepare(argparse.Namespace(fit_overlap=20, threads=1), names, rows, loci, "dna", 5, time.time())
        monkeypatch.setattr(lo, "_mem_available", lambda: 16 << 20)
        logs = []; monkeypatch.setattr(lo, "LOG", lambda m: logs.append(m))
        P4 = lo.prepare(argparse.Namespace(fit_overlap=20, threads=4), names, rows, loci, "dna", 5, time.time())
        assert any("medians by numpy chunks" in m for m in logs), logs
        for k in ("m_full", "mfl_full", "rate", "cls"): assert np.array_equal(getattr(P1, k), getattr(P4, k), equal_nan=True), k
        for k in ("cand", "key", "m", "mfl", "nfin"): assert np.array_equal(P1.lists[k], P4.lists[k], equal_nan=True), k
        for x, y in zip(P1.closest, P4.closest): assert np.array_equal(x, y, equal_nan=True)


class TestMemoryRoom:
    def test_room_is_the_budget_less_what_is_resident(self, monkeypatch):
        monkeypatch.setattr(lo, "MEM_BUDGET", None); assert abs(lo._mem_room() - lo._mem_available()) < 0.01 * lo._mem_available()   # the machine's figure moves between the calls
        rss = lo._rss(); assert rss > 0
        monkeypatch.setattr(lo, "MEM_BUDGET", rss + (100 << 20)); assert abs(lo._mem_room() - (100 << 20)) < (16 << 20)
        monkeypatch.setattr(lo, "MEM_BUDGET", 1 << 20); assert lo._mem_room() == (1 << 20) // 16                  # never below a sixteenth of the budget
        monkeypatch.setattr(lo, "MEM_BUDGET", rss + (300 << 20)); monkeypatch.setattr(lo, "MEM_RESIDENT_EST", rss + (200 << 20))
        assert abs(lo._mem_room() - (100 << 20)) < (16 << 20)                                                    # the estimate counts before the arrays are filled

    def test_rss_falls_back_to_the_peak_without_proc(self, monkeypatch):
        real_open = open
        def no_proc(path, *a, **k):
            if str(path).startswith("/proc/self/statm"): raise OSError("no proc")
            return real_open(path, *a, **k)
        monkeypatch.setattr("builtins.open", no_proc); assert lo._rss() > 0
