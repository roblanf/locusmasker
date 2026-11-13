#!/usr/bin/env python3
"""The random-gene-tree fixture, tests/data/sim/aa_random_trees: 100 taxa x 100 loci of 300 amino acids simulated by AliSim
(IQ-TREE 3) under LG+R12, each locus on its own random Yule-Harding tree with AliSim's default branch lengths. Nothing is
planted and the loci share no tree, so every sequence is a true sequence of its locus and anything the tool masks is a
false positive. Rob's question (31 October 2026): what does the test do when the gene trees are random?

    python tests/data/make_random_trees.py [--iqtree iqtree3] [--out tests/data/sim/aa_random_trees]

Regenerating needs iqtree3 on the PATH; the suite does not, it reads the committed files. Deterministic: locus i is
simulated with AliSim seed i. truth.json records the command, the rate categories AliSim used and every locus's tree."""
import argparse, json, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
N_LOCI, LENGTH, MODEL, TREE = 100, 300, "LG+R12", "RANDOM{yh,100}"


def simulate(iqtree, tmp, i):
    """one locus: the sequences by taxon, the tree AliSim drew, and the rate categories it reports"""
    pre = os.path.join(tmp, f"locus{i}")
    subprocess.run([iqtree, "--alisim", pre, "-m", MODEL, "-t", TREE, "--length", str(LENGTH), "--seed", str(i), "-af", "fasta"],
                   check=True, capture_output=True, text=True)
    seqs, cur = {}, None
    for line in open(pre + ".fa"):
        line = line.strip()
        if line.startswith(">"): cur = line[1:].split()[0]; seqs[cur] = []
        elif line: seqs[cur].append(line)
    log = open(pre + ".treefile.log").read(); rates = re.search(r"Site proportion and rates:\s*(.*)", log)
    return {k: "".join(v) for k, v in seqs.items()}, open(pre + ".treefile").read().strip(), rates.group(1).strip() if rates else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iqtree", default="iqtree3"); ap.add_argument("--out", default=os.path.join(HERE, "sim", "aa_random_trees")); a = ap.parse_args()
    if shutil.which(a.iqtree) is None: sys.exit(f"{a.iqtree} not found: AliSim (IQ-TREE 3) is needed to regenerate this fixture")
    tmp = tempfile.mkdtemp(); names = None; loci = []; trees = []; parts = []; pos = 1; rates = None
    for i in range(1, N_LOCI + 1):
        seqs, tree, r = simulate(a.iqtree, tmp, i)
        if names is None: names = sorted(seqs, key=lambda s: int(s[1:]))
        assert set(seqs) == set(names) and all(len(s) == LENGTH for s in seqs.values()), f"locus {i}"
        loci.append(seqs); trees.append(tree); parts.append(f"AA, locus{i} = {pos}-{pos + LENGTH - 1}"); pos += LENGTH; rates = rates or r
    shutil.rmtree(tmp); os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/alignment.fasta", "w") as fh:
        for t in names: fh.write(f">{t}\n" + "".join(sq[t] for sq in loci) + "\n")
    with open(f"{a.out}/partitions.txt", "w") as fh: fh.write("\n".join(parts) + "\n")
    version = subprocess.run([a.iqtree, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    meta = dict(name="aa_random_trees", datatype="aa", taxa=len(names), loci=N_LOCI, sites=pos - 1, events=[],
                locus_bounds=[[1 + i * LENGTH, (i + 1) * LENGTH] for i in range(N_LOCI)],
                simulation=dict(tool=version, model=MODEL, rate_categories=rates, length=LENGTH,
                                tree=f"{TREE}: a new random Yule-Harding tree per locus, AliSim's default branch lengths", seeds="locus i has seed i",
                                command=f"{a.iqtree} --alisim locusI -m {MODEL} -t '{TREE}' --length {LENGTH} --seed I -af fasta"),
                trees={f"locus{i + 1}": t for i, t in enumerate(trees)})
    with open(f"{a.out}/truth.json", "w") as fh: json.dump(meta, fh, indent=1)
    print(f"{a.out}: {len(names)} taxa, {N_LOCI} loci, {pos - 1} sites")


if __name__ == "__main__":
    main()
