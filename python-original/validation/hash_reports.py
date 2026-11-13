#!/usr/bin/env python3
"""sha256 of every report.npz member (but the manifest) and of masked.csv for every run under a profile_corpus.py output
directory made with --keep-outputs: the pin of the present engine's outputs, and the comparison of a later run with it.

    python validation/hash_reports.py DIR                 writes DIR/hashes.json
    python validation/hash_reports.py DIR --compare BASE  compares DIR's runs with BASE/hashes.json, member by member"""
import argparse, glob, hashlib, json, os, zipfile


def hashes(run_dir):
    out = {}
    npz = os.path.join(run_dir, "report.npz")
    if os.path.exists(npz):
        z = zipfile.ZipFile(npz); out.update({k: hashlib.sha256(z.read(k)).hexdigest() for k in sorted(z.namelist()) if k != "manifest.json"})
    csv_ = os.path.join(run_dir, "masked.csv")
    if os.path.exists(csv_): out["masked.csv"] = hashlib.sha256(open(csv_, "rb").read()).hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--compare"); a = ap.parse_args()
    runs = {}
    for run_dir in sorted(glob.glob(os.path.join(a.dir, "*", "t*", "run"))):
        ds, t = run_dir.split(os.sep)[-3:-1]; runs[f"{ds}/{t}"] = hashes(run_dir)
    if a.compare:
        if not runs: raise SystemExit(f"no runs found under {a.dir} (looked for */t*/run): nothing was compared")
        base = json.load(open(os.path.join(a.compare, "hashes.json"))); bad = 0
        for key, h in runs.items():
            if key not in base: print(f"{key}: not in the baseline"); continue
            diff = sorted(k for k in set(h) | set(base[key]) if h.get(k) != base[key].get(k))
            if diff: bad += 1; print(f"{key}: {len(diff)} members differ: {', '.join(diff[:8])}{' ...' if len(diff) > 8 else ''}")
            else: print(f"{key}: identical ({len(h)} members)")
        print("all identical" if not bad else f"{bad} runs differ")
    else:
        json.dump(runs, open(os.path.join(a.dir, "hashes.json"), "w"), indent=1); print(f"{len(runs)} runs hashed -> {a.dir}/hashes.json")


if __name__ == "__main__":
    main()
