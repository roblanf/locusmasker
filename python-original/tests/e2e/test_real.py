"""The excerpts of real alignments: the tool runs, the outputs are sane, the NEXUS round trip holds."""
import numpy as np
import pytest
from tests.conftest import lo, dataset, REAL

pytestmark = [pytest.mark.e2e, pytest.mark.real]


@pytest.mark.parametrize("name", REAL)
class TestReal:
    def test_runs_and_masks_a_little(self, runs, name):
        run = runs(name); R = run.report; c = R.man["counts"]
        assert run.rc == 0 and c["cells"] > 0 and 0 <= c["masked"] < 0.1 * c["cells"]
        assert R.man["options"]["datatype"] == dataset(name).dtype and R.man["input"]["format"] == "nexus"

    def test_masked_nexus_round_trips(self, runs, name):
        run = runs(name); ds = dataset(name); names, rows, charsets, info = lo.read_nexus_fast(run.file("alignment.masked.nex"))
        n0, r0, c0, i0 = lo.read_nexus_fast(ds.aln)
        assert names == n0 and charsets == c0 and info.datatype == i0.datatype and len(rows) == len(r0)
        side = run.report.arr("cells/side.npy")
        for g in range(len(names)):
            for li, (k, sites) in enumerate(charsets.items()):
                a = np.frombuffer(rows[g], dtype=np.uint8)[sites]; b = np.frombuffer(r0[g], dtype=np.uint8)[sites]
                if side[g, li] & 3: assert (a == ord("-")).all()
                else: assert np.array_equal(a, b)

    def test_log_is_complete(self, runs, name):
        log = runs(name).log; assert "tests:" in log and "masked" in log and "done" in log and "Traceback" not in log
