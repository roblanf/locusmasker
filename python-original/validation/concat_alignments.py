#!/usr/bin/env python3
"""Concatenate per-locus alignments into one FASTA plus a RAxML-style partition file, the non-NEXUS input route.

    python concat_alignments.py <dir-or-archive> <out_prefix> [--datatype dna|aa] [--pattern GLOB]

Reads every alignment file found (FASTA or sequential/interleaved PHYLIP; a .tar.gz or .zip is extracted first into
<out_prefix>_extracted/), takes the union of the taxon names, pads a taxon absent from a locus with gaps, and writes
<out_prefix>.fasta and <out_prefix>_partitions.txt ("DNA, <locus> = start-end"). Loci are ordered by file name.
Taxon names are used as written in the files (whitespace in a FASTA header ends the name)."""
import argparse, glob, os, re, sys, tarfile, zipfile

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--datatype", choices=("dna", "aa"), default="dna")
ap.add_argument("--pattern", default="*", help="glob for the alignment files inside the directory (default: every regular file)")
a = ap.parse_args()

src = a.src
if os.path.isfile(src) and (src.endswith(".tar.gz") or src.endswith(".tgz") or src.endswith(".tar")):
    dst = a.out + "_extracted"; os.makedirs(dst, exist_ok=True)
    with tarfile.open(src) as tf: tf.extractall(dst, filter="data")
    src = dst
elif os.path.isfile(src) and src.endswith(".zip"):
    dst = a.out + "_extracted"; os.makedirs(dst, exist_ok=True)
    with zipfile.ZipFile(src) as zf: zf.extractall(dst)
    src = dst

files = sorted(p for p in glob.glob(os.path.join(src, "**", a.pattern), recursive=True) if os.path.isfile(p) and not os.path.basename(p).startswith("."))
if not files: sys.exit(f"no alignment files under {src}")

def read_alignment(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    seqs = {}
    if txt.lstrip().startswith(">"):
        name = None
        for line in txt.splitlines():
            if line.startswith(">"): name = line[1:].split()[0]; seqs[name] = []
            elif name is not None: seqs[name].append(line.strip())
        return {k: "".join(v) for k, v in seqs.items()}
    lines = [l for l in txt.splitlines() if l.strip()]
    m = re.match(r"\s*(\d+)\s+(\d+)", lines[0])
    if not m: sys.exit(f"{path}: neither FASTA nor PHYLIP")
    ntax, nchar = int(m.group(1)), int(m.group(2)); order = []
    for line in lines[1:]:
        parts = line.split()
        if len(order) < ntax:
            name = parts[0]; order.append(name); seqs[name] = "".join(parts[1:])
        else:                                                           # interleaved continuation blocks
            idx = len([n for n in order if len(seqs[n]) >= nchar])      # the next taxon still short
            name = order[idx % ntax]; seqs[name] += "".join(parts)
    bad = [n for n in order if len(seqs[n]) != nchar]
    if bad: sys.exit(f"{path}: sequence length differs from the header for {bad[:3]}")
    return seqs

loci = []
for p in files:
    s = read_alignment(p)
    L = {len(v) for v in s.values()}
    if len(L) != 1: sys.exit(f"{p}: sequences of different lengths {sorted(L)[:5]}")
    loci.append((os.path.splitext(os.path.basename(p))[0], s, L.pop()))
taxa = sorted({t for _, s, _ in loci for t in s})
print(f"{len(loci)} loci, {len(taxa)} taxa, {sum(L for _, _, L in loci):,} sites", file=sys.stderr)
with open(a.out + ".fasta", "w") as fa, open(a.out + "_partitions.txt", "w") as pt:
    pos = 1
    for name, s, L in loci:
        pt.write(f"{'DNA' if a.datatype == 'dna' else 'WAG'}, {name} = {pos}-{pos + L - 1}\n"); pos += L
    for t in taxa:
        fa.write(f">{t}\n"); fa.write("".join(s.get(t, "-" * L) for _, s, L in loci) + "\n")
print(f"wrote {a.out}.fasta and {a.out}_partitions.txt", file=sys.stderr)
