# LocusMasker: masking phylogenetic outliers from a multi-locus alignment

# Warning!

This is work in progress! Please contact me if you are thinking of using it. A preprint will be submitted soon with a full description of the methods. Until then (well, until the preprint has passed muster in peer reivew really), please use carefully and at your own risk.

# What's this?

A set of methods which attempt to highlight and mask potentially misleading sequences (individual loci from individual taxa) from large phylogenomic datasets. Read on if you have a large (>10 loci) dataset, and are intending to use it to estimate a single phylogenetic tree with a method that doesn't account for things like contamination, introgression, horizontal gene transfer, or accidental use of paralogs. (NB: this includes methods in software like IQ-TREE, RAxML, BEAST, PhyloBayes, MCMCTREE, MrBayes, ASTRAL, etc.)

Other methods like this exist. The most similar is [PhylteR](https://github.com/damiendevienne/phylter). The main differences here are:

1. This method is really fast (usually less than a few minutes, even on really big datasets!). PhylteR requires you to estimate gene trees first. 

2. This method gives you a really lovely interactive report to explore the data and judge every decision for yourself.

3. My simulations (see caveat above, preprint coming soon) suggest that this approach often has better precision and recall than PhylteR when there is stochastic error in your gene trees.

4. The tests here are each designed quite specifically to spot issues I see in many datasets. PhylteR on the other hand is a very elegant and much more general method.

My suggestion - try both. This one to help you improve your data and bioinformatic pipelines. Then PhylteR after you've invested lots of compute in estimating gene trees. 

# Why?

A lot of people (me included) are interested in estimating phylogenies from large datasets. Although we all know that some data doesn't fit on a tree, we are often willing to intentionally ignore this and as a first pass estimate a single tree topology that we hope is a good explanation of the entire dataset (e.g. a species tree). 

The problem is that in a large multi-locus alignment not all sequenced loci from all taxa will *fit* on a single tree, and these loci can be problematic for _inferring_ that tree. Loci can mislead tree inference for a lot of reasons, including but not limited to: contamination, mis-labelling, bioinformatic errors (like BUSCO pulling out a different locus from different genomes, errors in exon annotations etc.), hybridisation and introgression, horizontal gene transfer, accidental labelling of a paralog as an ortholog, etc. If the aim is to produce a single estimate of a tree from a concatenated (e.g. IQ-TREE), ILS-aware (e.g. BEAST), or ILS-robust (ASTRAL) analysis, these loci can mislead the analysis. The point of LocusMasker is to try and find as many of these as possible, avoid false positives, and then let you see and explore the data in detail.

# Pros and cons of masking data

The intention of this repo is that one would use this program to mask data before building a tree. An obvious use case would be to detect and remove contaminants and mislabelled sequences from large datasets. These are _obviously_ misleading because they're _wrong_, and nobody wants them in their datasets. Another use case is more nuanced - removing loci that have moved horizontally, due to processes like introgression or HGT. These loci are _not_ wrong in the biological sense, but one might still be interested in masking them if the intention is to use a method that doesn't account for these biological processes when building the tree. 

So, one should be aware that the individual loci from individual taxa highlighted by this method could be highlighted due to any of the following three reasons: (i) the loci are strictly wrong (e.g. contamination during sequencing); (ii) the loci are biologically correct but not accounted for by a downstream method (e.g. HGT in a standard BEAST analysis); (iii) the loci have been incorrectly highlighted. Obviously, LocusMasker is trying to find instances of (i) and (ii), and avoid (iii). 

An important thing to be aware of is that contamination and HGT can leave identical signals in the data. It's literally impossible to confidently tell them apart from an alignment alone. So, if the *focus* of your work is finding signals of HGT, then LocusMasker will _help_ you to infer a good backbone species tree, but it will do that in part by masking precisely the data you'd want to use to find HGT. In this case, the sensible use of the program would be to do it in two stages:

i. use the program to mask data to build a single tree
ii. un-mask data that you think comes from HGT (to which I say, you will need some kind of useful things like replicates to know if you are dealing with HGT or contamination!) to then look for the interesting biological signals w.r.t to the tree estimated in step (i).

# Input/output

## Input

Input either of the following:

1. A single NEXUS format alignment file with a SETS block defining the loci
2. A FASTA, PHYLIP, or NEXUS format alignment with a RAxML-style partitions.txt file defining the loci.

The alignment file may be gzip-compressed (e.g. `alignment.fasta.gz`). It is recognised by its content, not its name, and the masked alignment is then written gzip-compressed too.

NB: your alignment must have at least 10 loci for this method to work.

More details on formats can be found here: https://iqtree.github.io/doc/Advanced-Tutorial

## Output

The output consists of the following:

1. `report.html`, an interactive HTML report showing all results. Note that the data for this are in the `report_files` folder, so if you want to share your report, share this folder alongside it!
2. The alignment in its input format with every outlier sequence (individual locus in a single taxon) masked as gaps, e.g. `alignment.masked.nex` (or `alignment.masked.nex.gz` if the input was gzip-compressed)
3. `masked.csv`, one row per masked cell with the evidence for masking it (see tests below)
4. `run.log`, the log of the run as it was printed: every line starts with the time it was written, and the last line gives the total execution time

# Method overview

The basic idea of the method is to compare each sequence to its relatives in the alignment, and ask if the number of differences you'd _expect_ to see (based on information in the whole alignment) differs in a surprising way from the number of differences you _actually_ see. 

A full paper with a description is forthcoming, but briefly there are three tests that the method performs:

## close-relative test
This test asks whether a sequence is _more divergent_ from the homologous sequences in close relatives than you'd expect given what the rest of the loci look like. This happens with things like contamination, sequencing error, and bioinformatic errors like getting the wrong locus, or an incorrect exon. There's a bit of work involved in calculating the expectations, but suffice to say here that these use information from across the alignment (but without building trees) to develop an expectation of how many mismatches one expects to see between any sequence and it's homologues from close relatives. The default is to use 5 close relatives. For each of these relatives, we ask whether the observed number of mismatches is larger than expected using a beta-binomial test. We then take the median p-value from these tests (after using a Bonferroni correction across ALL of the test we do, which ends up being A LOT). This is equivalent to saying - if more than half of the comparisons fail for a given sequence, we mask that sequence.

## closest-match test
This test asks whether the closest sequence in the dataset is closer than expected (expectations come from the close-relative test, described above). There are two versions of this test. Both can help to mask sequences that are contaminants or recent HGT events.

One thing common to both tests is that when a pair of sequences are surprisingly close, we would like to know _which one_ is the problem. To do that, if either test below is signficant, we repeat the same test on both members of the too close pair, and call the one with the most surprising shift (smallest observed/expected mismatches) the problem and mask that. Then we call the masked one the 'recipient' (e.g. it received contamination) and the unmasked one the 'donor'. To be conservative, we only make this call when one sequence is twice as surprising as the other. If they are more similar than that, we mask both.

As above - these tests are Bonferroni corrected alongside all other tests performed. 

### closest-match test: close-relative
If the closest match is a close relative from the close-relative test, we just ask the opposite question from the close-relative test using the same statistics. I.e. a one-tailed beta-binomial to ask if the sequence is more _similar_ to the closest match than the rest of the alignment would suggest. We do some corrections here because we observed that you can get a lot of false positives if the focal sequence just happens to evolve really slowly. 

### closest-match test: not a close-relative
If the closest match is a not one of the close relatives, then we don't have an expectation of how different the two should be. So we ask a different question: is the focal sequence closer to its closest match than its own close relatives are? Here, we get the expectation from the close relatives, and ask if the focal seuqence is an outlier in this context. 


## broad-sample test
The tests above can miss a couple of important cases. An HGT event deep in the tree can move a whole clade, which undermines all of them. Or if there's a paralog in the data, then this can cause them to fail too, even though if we did have gene trees, it would be obvious. The broad-sample test tries to catch these cases too. To do that, we ask the same two questions as above, but from a random sample of 50 (if there are enough homologues in the data) other homologues for each focal sequence. Because sequences accumulate mismatches more slowly the more different they get (because of multiple changes happening at individual sites) we have to recalculate expected distances here by building a mismatch accumulation curve for each locus. But the mechanics of the test are then similar to those above. I.e. we call a sequence _too divergent_ if it is significantly far from most of the broad sample (after dropping any sequences that fail the other tests above). We call a sequence _too close_ if a sequence is much more similar to one of the distant taxa. This test works by calculating a p-value against all of the broad-samples, correcting them using Bonferroni, and taking the smallest one. If _that_ is significant, it suggests that there's a close relative in the context of this set of broad samples. We then multiply that p value by the number of total tests (i.e. the Bonferroni across all tests) and only mask the cell if that p value (i.e. p_value * n_broad_samples * N_tests_over_all_tests) is below the alpha threshold.

# Usage

LocusMasker is a single program written in Rust. (It began as a Python script, `locusmasker.py`; that script is kept in `python-original/` as the reference this program was ported from, and for the same input the two write the same results.)

## Installation

**A prebuilt binary.** For Linux on x86-64 (glibc 2.28 or later), download `locusmasker-<version>-x86_64-linux-gnu.tar.gz` from the [releases page](https://github.com/donovan-h-parks/locusmasker-rust/releases), unpack it, and put the `locusmasker` file it holds somewhere on your `PATH`. That one file is the whole program: the report viewer is built into it.

**From source.** Install Rust if you don't already have it (https://rustup.rs), then:

```
git clone https://github.com/donovan-h-parks/locusmasker-rust.git
cd locusmasker-rust
cargo build --release
```

This leaves the program at `target/release/locusmasker`. Copy it wherever you like, or let Cargo put it on your `PATH` (in `~/.cargo/bin`):

```
cargo install --path .
```

Check that it runs:

```
locusmasker --help
```

## Running LocusMasker

You need to tell it where the alignment is, whether it's DNA or amino acids, and where to put the output. If you have a nexus file with a SETS block (example filename here is `alignment.nex`):

```
locusmasker --aln alignment.nex --datatype dna --out scan
```

If you have a nexus file with no SETS block, or a FASTA or PHYLIP alignment, and a partitions file:

```
locusmasker --aln alignment.fasta --partitions partitions.txt --datatype dna --out scan
```

Big datasets take a while. Give it more threads and it will use them:

```
locusmasker --aln alignment.nex --datatype dna --out scan --threads 0
```

All of the options are listed below. The defaults are sensible, and most people will only ever need the first four.

Input and output:

- `--aln`: the alignment (required), plain or gzip-compressed.
- `--datatype dna|aa`: DNA or amino acids (required). DNA is analysed on A, C, G, T (U is read as T); amino acids on the 20 standard residues. Everything else counts as missing.
- `--out`: the output folder (required).
- `--partitions`: a RAxML-style partitions file defining the loci. Required unless the alignment is a nexus file with a SETS block.
- `--name`: the dataset name shown in the report. Default is the alignment file's name.

Method options:

- `--n-comp 5`: the number of close relatives each sequence is compared to. Must be odd and at least 3.
- `--fit-overlap 20`: the minimum number of sites a pair must share at a locus to be compared there.
- `--alpha 0.05`: the significance level, applied after Bonferroni correction across every test in the run.
- `--donor-threshold 2`: in the closest-match and broad-sample tests, a too-similar pair gets a direction (copy masked, source not) when one member's deviation from its own expectation is more than this many times the other's. Otherwise the direction is uncertain and both are masked.

Performance options (none of these change the results):

- `--threads 1`: the maximum number of threads to use. `0` means all available.
- `--memory`: the RAM budget in GB. Default is half of what's free when the run starts; `0` means all available RAM. A run whose estimated memory is over the budget stops before it starts.
- `--restart`: ignore the restart points an earlier run left in the output folder and recompute everything. By default a rerun with the same input and options picks up where it left off.
- `--progress auto|bar|none`: a progress bar on a terminal, progress lines otherwise.

Report options:

- `--no-report`: write no report at all.
- `--all-tests`: also write `tests.tsv.gz`, one row per test performed with its statistic, p-value and result. Useful if you want to do your own analysis of the results, but large.

Help

- `-h` or `--help`: print these options

## Development

The repository is an ordinary Cargo project: the source is in `src/`, the report viewer (HTML, CSS and JavaScript, built into the program when it is compiled) in `viewer/`, and `cargo test --release` runs the unit tests. `python-original/` holds the original Python script with its test suite and the scripts that compare the two programs' outputs; see the `README.md` there.


# Computational requirements

Nearly all of the time on a big alignment goes into comparing every pair of taxa, so the number of taxa is what matters most:

- **Running time** grows with the square of the number of taxa and in proportion to the length of the alignment: twice the taxa takes about four times as long, twice the sites about twice as long. The tests that follow the pair comparisons are the smaller part of the run beyond a few thousand taxa (about 3 of the 50 minutes of the largest run below).
- **Threads**: doubling them cuts the running time by 40 to 45% up to the number of physical cores (the table below); hyperthreads beyond that add less.
- **Peak memory** grows in proportion to the size of the alignment (taxa x sites: its letters, and its residues again as bit planes at a little under a byte per site for amino acids) and to the number of sequences. Up to 20,000 taxa a matrix of every pair's divergence is kept as well (16 bytes per pair, so 6 GiB at 20,000 taxa), and filling it costs time too. Above 20,000 taxa the matrix is not kept, and memory no longer grows with the square of anything. More threads cost a little more memory; at most eight loci are tested at a time, whatever the thread count.

Measured with version 0.16.0 on amino-acid alignments of 120 loci and 41,084 sites, on a machine with 48 cores (96 hardware threads):

| Taxa | Threads | Running time | Peak memory |
| ---: | ---: | ---: | ---: |
| 10,424 | 32 | 31 s | 4.0 GiB |
| 20,847 | 8 | 3 min 9 s | 3.7 GiB |
| 20,847 | 16 | 1 min 42 s | 4.0 GiB |
| 20,847 | 32 | 1 min 0 s | 4.4 GiB |
| 20,847 | 96 | 47 s | 5.5 GiB |
| 229,307 | 64 | 50 min 3 s | 41.4 GiB |

The largest of these was read from a gzip-compressed file. The start of `run.log` gives an estimate of the memory the run's long-lived data needs; the peak is higher, as the table shows, by what the tests use while they run.

# License

LocusMasker is released under the MIT License (see `LICENSE`), copyright Robert Lanfear.

Every `report.html` LocusMasker writes embeds four third-party components so that the report is a single self-contained file: [D3](https://d3js.org) (ISC), [d3-sankey](https://github.com/d3/d3-sankey) (BSD 3-clause), [Fuse.js](https://www.fusejs.io) (Apache 2.0) and the [Inter](https://rsms.me/inter/) font (SIL Open Font License 1.1). Their licenses are in `viewer/vendor/`, and every report carries a copy of them in its page source.
