#!/usr/bin/env python3
"""Score a run on a make_big_sim.py dataset against its truth.json: recall per planted kind (contaminants and far copies
by the too-divergent test; near copies as identified too-similar pairs with the direction; multicopy recipients;
sister copies), what happened to the clone clusters (no member should be masked for its copies) and the excluded taxa
(untested), and the false-positive rates on unplanted cells, in validation/sim_suite.py's conventions.

    python validation/score_big_sim.py test_data/Sim_25K notes/profiling_2026-10-30/m1_corpus/Sim_25K/t96/run"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_suite import read_scan, DIV, SIM


def main(data, run):
    truth = json.load(open(os.path.join(data, "truth.json"))); masked, pairs, closest, man = read_scan(run)
    why = {c: r["reason"] for c, r in masked.items()}
    def source(c):
        r = masked.get(c)
        if r is None: return None
        return r["closest_match_partner"] if r["closest_match_role"] == "copy" else None
    planted = set(); out = {}
    def tally(kind, hit):
        a = out.setdefault(kind, [0, 0]); a[1] += 1; a[0] += bool(hit)
    for ev in truth["events"]:
        k = ev["kind"]
        if k in ("contam", "farcopy", "sistercopy"):
            c = (ev["locus"], ev["taxon"] if "taxon" in ev else ev["recipient"]); planted.add(c)
            tally(k + "_divergent", why.get(c) in DIV); tally(k + "_any", c in why)
            if k != "contam": planted.add((ev["locus"], ev["donor"])); tally(k + "_similar_named", why.get(c) in SIM and source(c) == ev["donor"])
        elif k == "nearcopy":
            l, dn, rc = ev["locus"], ev["donor"], ev["recipient"]; planted |= {(l, dn), (l, rc)}
            called = pairs.get((l, frozenset((dn, rc)))); tally("nearcopy_pair", called is not None); tally("nearcopy_direction", called == rc)
            tally("nearcopy_recipient_any", (l, rc) in why); tally("nearcopy_donor_masked", (l, dn) in why)
        elif k == "multicopy":
            l = ev["locus"]; planted.add((l, ev["donor"]))
            for r in ev["recipients"]: planted.add((l, r)); tally("multicopy_recipient_any", (l, r) in why)
            tally("multicopy_donor_masked", (l, ev["donor"]) in why)
        elif k == "clone":
            members = [ev["donor"]] + list(ev["copies"]); n_masked = sum(1 for (l, t) in why if t in set(members))
            tally("clone_clusters_with_a_mask", n_masked > 0); out.setdefault("clone_cells_masked", [0, 0])[0] += n_masked
        elif k == "excluded":
            tally("excluded_untested", not any(t == ev["taxon"] for _, t in why))
    out["fp_divergent_unplanted"] = [sum(1 for c, w in why.items() if w in DIV and c not in planted), int(man["counts"]["cells"])]
    out["fp_similar_pairs_unplanted"] = [sum(1 for (l, pr), rc in pairs.items() if not any((l, t) in planted for t in pr)), len(pairs)]
    out["masked_total"] = [len(why), int(man["counts"]["cells"])]
    print(f"{os.path.basename(data.rstrip('/'))}: {man['counts']['taxa']:,} taxa x {man['counts']['loci']} loci, {man['counts']['cells']:,} cells, {len(why):,} masked; LocusMasker {man['version']}")
    for k, (a, b) in out.items():
        print(f"  {k:30s} {a:7d} / {b:9d}" + (f"  ({a / b:.1%})" if b else ""))
    return out


if __name__ == "__main__": main(sys.argv[1], sys.argv[2])
