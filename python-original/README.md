# The original Python LocusMasker

`locusmasker.py` is the Python script LocusMasker began as (version 0.14.0, by Robert Lanfear). The program at the root
of this repository is a port of it to Rust, and this script is kept as the reference that port is held to: for the same
input and options the two write the same results. It is not needed to build or run LocusMasker, and new work goes into
the Rust source, not here.

What is in this folder:

    locusmasker.py     the script: the method, the report writer, the command line
    environment.yml    the conda environment it runs in (numpy; numba and Biopython optional), with what its tests need
    tests/             its pytest suite and the fixture datasets (tests/README.md); pytest.ini and .coveragerc configure it
    validation/        simulations and benchmarks of the method, and the two scripts that compare this script with the
                       Rust program (compare_ports.py, fuzz_ports.py)
    viewer, LICENSE    links to ../viewer and ../LICENSE: the script looks for both beside itself

## Running the script

    mamba env create -f environment.yml && mamba activate locusmasker
    python locusmasker.py --aln alignment.fasta --partitions partitions.txt --datatype dna --out scan --threads 0

It takes the options of the Rust program (see the main `README.md`), and is slower and heavier on memory. Its report
viewer is the one in `../viewer`, reached through the `viewer` link here, so the script needs the repository around it
(on a system that does not check out symbolic links, copy `../viewer` and `../LICENSE` into this folder).

## What "the same results" means

For the same input and options the Rust program writes, byte for byte, what `locusmasker.py` writes:

- the masked alignment and `masked.csv`, as files;
- `tests.tsv.gz` (`--all-tests`), decompressed;
- every member of `report.npz` (the arrays, `taxa.json`, `loci.json`, the trees and pair records) and `manifest.json`
  apart from the fields listed below;
- every member of the report's core (inline in `report.html`) and of every `report_files/` chunk, decoded, and the page
  around the core.

Not the same, by nature: the compressed containers themselves (`report.npz`, the base64 ZIPs, the gzip stream: their
bytes hold timestamps and depend on the zlib build, as they do between two runs of the script), `run.log` and the
progress lines (timings, and each tool describes its own machinery), and these manifest fields: `date`, `timings`,
`command`, `system`, `commit`, `python` and `numpy` (the Rust program writes its compiler there, so the viewer's run line
stays as it is). And the version: the Rust program is 0.16.0 and writes that where the script writes its 0.14.0 (the
manifest's `version`, `report_files/index.js` and the report's loading line); the method and every result are those of
`locusmasker.py` 0.14.0.

## Checking it

Build the Rust program at the repository root (`cargo build --release`), then from this folder, in the conda
environment:

    python validation/compare_ports.py                      # 42 cases
    python validation/fuzz_ports.py --cases 150 --seed 3

`compare_ports.py` runs both tools on the fixture datasets (every input format, the option combinations, and the size
thresholds forced low so that the sparse report, the no-full-matrix mode and the extended candidate lists run) and
compares all of the above; `fuzz_ports.py` does the same on random alignments with random options. Both find the binary
at `../target/release/locusmasker` (`--rust` names another).

The pytest suite runs against the binary with `LOCUSMASKER_BIN=$PWD/../target/release/locusmasker pytest tests/e2e`
(see `tests/README.md`).

## How the numbers are held to the script

The Rust source mirrors the order of floating-point operations of the numpy expressions it replaces: numpy's pairwise
summation, its medians, percentiles and interpolation, float32 where the Python computes in float32, Python's
`round()`, `repr()` of floats, `json.dumps` and `csv.writer`, the compensated `sum()` of Python 3.12 and later (the
neighbour-joining trees use it), numpy's PCG64 generator and `choice` (the report's fixed sample of cells), and the
tool's own Lanczos log-gamma and exact tails. `exp`, `log`, `pow` and `erfc` come from the C library, as they do in
Python and in numpy on a machine without AVX-512: so the two tools agree to the bit where they share a C library, and
a different libm (or numpy's AVX-512 loops) can move a last digit in either of them. The pinned hashes in
`tests/data/golden` show this for the script itself: `loci.json` (the dispersions at full precision) and a few trees
hash differently on a machine other than the one that pinned them.

## Where the Rust program behaves differently

- **The pair sweep.** The script walks pairs of taxon blocks with matrix products and merges block results (its
  reductions are order-independent); the Rust program counts shared and identical residues on bit planes with
  popcounts (AVX2 where the machine has it), with the arithmetic of the script's compiled kernel. Each pair of taxa
  is counted once and serves both of its rows: the rows of the pair matrix are swept above the diagonal, in chunks
  of about half a megabyte of planes with the columns in the outer loop (a column is read from memory once per
  chunk), and a pair (g, c) is offered to row g and to row c. A row's results (the candidate list so far, the closest
  match at each locus so far) are held behind a lock, and the cuts that decide whether a pair can matter to a row
  are read without it: a cut only tightens, so an old value lets through a pair that no longer matters, never the
  reverse. A pair's medians are computed only where their values are used (a pair that can enter a list or a
  closest-match band): whether m is at most the list's current cut is decided exactly from the weight of the loci
  at or below it, without the median. When the full m matrix is kept (up to 20,000 taxa, sweep 2) every m is
  computed, once per pair. The sweeps of a few given rows (the taxa whose candidate lists ran out) go row by row.
  `--block-taxa` and `--no-numba` are accepted and do nothing. Threads replace the worker processes.
- **The broad-sample test** is computed row by row in three passes over a locus (the draws and the curve; the
  too-close side; the too-divergent side and the direction) instead of on rows x members matrices: the same numbers
  from a fraction of the memory. The pools' pair divergences are computed in sweep 2 as the pairs go by, when the
  full matrix is not kept (a run that starts after the sweeps computes them pair by pair, as the script does).
- **The tests of a locus** spread over the threads themselves (the pairs' exact tails, the rows of the broad sample
  and the bins of its curve, the cells of the best match), and a few loci are in flight at a time, where the script
  tests one locus per worker: the stage's memory does not grow with the thread count. While a locus is tested its
  bit planes of every taxon are held side by side, and its counts and single-site comparisons are made there.
  The broad sample's curve bins its points on its grid as they are produced and takes each bin's medians and
  percentiles by selection: the curve of the points sorted, without the sort.
- **The best match.** The script compares the residues of a challenger, the champion and the focal sequence site by
  site, for every remaining challenger in every round, and computes the pair divergence m of every challenger that
  ties with another to order them. The Rust program counts on the planes (one three-way count per challenger),
  takes the challengers a group of equals at a time and computes the m that orders a group only when a member of it
  wins (a group in which nobody wins is passed whatever its order), reads an m that a candidate list already holds
  from the list, and counts a block of failing cells against every taxon in one pass over the taxa. The best
  matches are the script's.
- **A taxon whose candidate list ran out** gets its full row, of which only the comparators then chosen and the
  first five entries are kept.
- **Reading.** A FASTA file's records are parsed in parallel straight into the matrix of residues (a file with
  anything out of the ordinary is read record by record, with the script's messages), a gzip-compressed file is
  decompressed as it is read, and the input's sha256 is computed on a thread of its own meanwhile. The residues are
  held once as letters and once as bit planes; a sequence's residue codes are read back from the planes where the
  tests look at single sites. `report.npz` is deflated member by member in parallel.
- **NEXUS files with comments or quoted names.** The script hands these to Biopython; the Rust program has its own
  reader for them (comments, quoted names, interleaved matrices, TAXA / DATA / CHARACTERS blocks, a SETS block of
  charset, taxset and charpartition statements; other blocks are ignored) and writes such a file back the way Biopython
  does. Its error messages are its own.
- **Gzip-compressed alignments.** The script refuses them; the Rust program reads a file that starts with the gzip
  magic bytes (whatever its name) and writes the masked alignment gzip-compressed (`x.fasta.gz` gives
  `x.masked.fasta.gz`). Every other output is that of the same alignment uncompressed, apart from the manifest's
  `input` entry (the file's name, size and sha256 are those of the compressed file).
- **The log.** Every line of `run.log` and of the log on stderr, the progress lines included, starts with the local
  time (`[2026-11-13 07:30:01] `), and the last line is `done: total execution time ...` where the script writes
  `done 156s`. The message of a run that stops on an error is as the script writes it, without a time.
- **Restart points** are written and used as in the script (`restart/rates`, `restart/pairs`, `--restart`,
  `LOCUSMASKER_STOP_AFTER`), but hold the Rust program's own structures: a point written by one tool is not used by
  the other.
- **`--memory`** sets the budget the run's estimate is checked against before it starts; the Rust program has no chunk
  sizes to derive from it.
- **Test hooks.** The constants the test suite changes in the Python module are read from the environment:
  `LOCUSMASKER_DENSE_CELLS_MAX`, `LOCUSMASKER_BEST_MATCH_ALL_MAX_TAXA`, `LOCUSMASKER_FULL_M_MAX_TAXA`,
  `LOCUSMASKER_SWEEP_BLOCK_MIN`, `LOCUSMASKER_REPORT_SAMPLE_CELLS`, `LOCUSMASKER_REPORT_SEQ_ALL_MAX`.
  `LOCUSMASKER_NO_AVX2` turns the AVX2 code off (the same numbers from the scalar code),
  `LOCUSMASKER_NO_LAZY` computes every pair's medians, and `LOCUSMASKER_NO_PAIRS_ONCE` sweeps every row against
  every column (each pair twice): the same results, slower.

## Where the script's parts are in the Rust source

    src/main.rs      the command line and the run
    src/input.rs     FASTA, PHYLIP, NEXUS, partitions, the character checks, the masked alignment writers
    src/prepare.rs   residue codes, plateaus, the two pair sweeps, the locus rates, the broad-sample pools
    src/runz.rs      comparators, dispersion, the tests, the decisions, masked.csv, tests.tsv.gz
    src/report.rs    report.npz, report.html, report_files/
    src/tree.rs      neighbour joining and newick
    src/npmath.rs    numpy's reductions, the special functions and exact tails, the hashes
    src/pyfmt.rs     Python's float formatting, round(), json.dumps, csv.writer
    src/npy.rs       .npy members, ZIP, base64          src/rng.rs      numpy's default_rng(0).choice
    src/rep.rs       the report's data                  src/restart.rs  the restart points
    src/log.rs       the log and the progress lines     src/tests.rs    unit tests
