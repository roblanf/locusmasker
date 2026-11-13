# The test suite

Run from `python-original/`, in the conda environment of `environment.yml`:

    pytest -n auto                     # everything, in parallel (about 20 s on 96 CPUs, a few minutes on a laptop)
    pytest -n auto -m "not viewer"     # without the headless browser
    pytest tests/unit -n auto          # one group: unit, property, e2e, regression, viewer
    pytest -n auto --cov              # with coverage of the tool (.coveragerc: fails under 95%)

Groups (markers in pytest.ini): `unit` tests state what each function should do and use scipy and Biopython as oracles;
`property` tests (hypothesis) check round trips and the algebra of the tails and medians for any input (and `tests/unit/test_kernel.py`
holds the compiled pair stage, numba, to the bits of the numpy one; its compiled half is skipped where numba is not installed); `e2e` tests run the
tool on the fixture datasets (planted truths, every input format, thread and chunking invariance, broken inputs, degenerate
alignments); `regression` tests pin the current outputs on the fixtures; `viewer` tests open the HTML report in headless
Chromium and read every panel back (skipped when playwright's Chromium is not installed: `python -m playwright install chromium`).

Fixtures live in `tests/data`: synthetic alignments with planted truths (`synth/*`, `truth.json` beside each) in every
input format, excerpts of two real alignments (`real/*`, CC0), an alignment simulated by AliSim with a random gene tree per
locus and nothing planted (`sim/aa_random_trees`, every mask a false positive; written by `tests/data/make_random_trees.py`,
which needs IQ-TREE 3), and the golden outputs (`golden/*`). The rest are written by `tests/data/make_fixtures.py` from fixed seeds; `--check` confirms the committed files are what the generator makes,
`--golden` re-pins the outputs after a deliberate change of the method (say so in the commit).

A test marked `xfail(strict=True)` states behaviour the code does not have yet; its reason starts with BUG, LIMIT or PRECISION.
When the code is fixed the test starts passing and pytest reports the stale marker, so remove it then.

## The Rust tool

The suite is run from `python-original/` (`pytest.ini` is there). With `LOCUSMASKER_BIN` set to the Rust binary (built at
the repository root: `cargo build --release`), `run_tool` and `run_cli` run it instead of the Python script, and a constant a
test changes in the module reaches the binary through its test hooks (`tests/conftest.py`):

    LOCUSMASKER_BIN=$PWD/../target/release/locusmasker pytest tests/e2e tests/viewer -n auto

The `unit` and `property` tests call the Python functions and say nothing about the Rust tool. A few `e2e` tests assert log
lines of the Python's own machinery (the taxon blocks of its sweep, its compiled kernel, Biopython's messages), start with
the Python script, or expect a gzip-compressed alignment to be refused (the binary reads it), and fail against the binary. `validation/compare_ports.py` and `validation/fuzz_ports.py` hold the Rust
tool's outputs to the script's byte for byte; `cargo test --release` at the repository root runs the Rust unit tests.
