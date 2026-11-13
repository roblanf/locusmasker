"""Neighbour joining, rooting and Newick output for the report's face-to-face trees."""
import re
import numpy as np
import pytest
from tests.conftest import lo


def random_additive(rng, n):
    """a random binary tree on n tips as an adjacency dict with positive lengths, and its additive distance matrix"""
    adj = {0: [], 1: []}; l = float(rng.uniform(0.5, 2)); adj[0].append((1, l)); adj[1].append((0, l)); nxt = 2; tips = [0, 1]
    while len(tips) < n:
        t = tips[int(rng.integers(len(tips)))]; p, lp = adj[t][0]                 # a tip has one neighbour
        u = nxt; nxt += 1; new = nxt; nxt += 1
        f = float(rng.uniform(0.2, 0.8)); adj[t] = [(u, lp * f)]
        adj[p] = [(y, ln) if y != t else (u, lp * (1 - f)) for y, ln in adj[p]]
        adj[u] = [(t, lp * f), (p, lp * (1 - f)), (new, float(rng.uniform(0.5, 2)))]; adj[new] = [(u, adj[u][2][1])]; tips.append(new)
    # relabel so tips are 0..n-1
    order = {old: i for i, old in enumerate(tips)}; inner = [x for x in adj if x not in order]
    for i, old in enumerate(inner): order[old] = n + i
    adj2 = {order[x]: [(order[y], ln) for y, ln in v] for x, v in adj.items()}
    D = np.zeros((n, n))
    for a in range(n):
        stack = [(a, None, 0.0)]
        while stack:
            x, parent, d = stack.pop()
            if x < n: D[a, x] = d
            for y, ln in adj2[x]:
                if y != parent: stack.append((y, x, d + ln))
    return adj2, D


def splits(adj, n):
    """the bipartitions of a tree, each as the frozenset of the smaller side"""
    out = set()
    def side(x, parent):
        s = {x} if x < n else set()
        for y, _ in adj[x]:
            if y != parent: s |= side(y, x)
        return s
    for x in adj:
        for y, _ in adj[x]:
            if x < y:
                s = side(y, x); t = frozenset(range(n)) - s
                if 1 < len(s) < n - 1: out.add(min(frozenset(s), t, key=lambda z: (len(z), sorted(z))))
    return out


def tip_distances(adj, n):
    D = np.zeros((n, n))
    for a in range(n):
        stack = [(a, None, 0.0)]
        while stack:
            x, parent, d = stack.pop()
            if x < n: D[a, x] = d
            for y, ln in adj[x]:
                if y != parent: stack.append((y, x, d + ln))
    return D


class TestNeighbourJoining:
    @pytest.mark.parametrize("n", [3, 4, 5, 8, 12])
    def test_recovers_an_additive_tree(self, n):
        rng = np.random.default_rng(n)
        for _ in range(5):
            adj, D = random_additive(rng, n); t = lo._nj(D)
            assert splits(t, n) == splits(adj, n)
            assert np.allclose(tip_distances(t, n), D, atol=1e-9)

    def test_two_tips(self):
        t = lo._nj(np.array([[0.0, 3.0], [3.0, 0.0]]))
        assert t[0] == [(1, 3.0)] and t[1] == [(0, 3.0)]

    def test_adjacency_is_symmetric_and_internal_nodes_have_degree_three(self):
        rng = np.random.default_rng(4); _, D = random_additive(rng, 9); t = lo._nj(D)
        for x, nb in t.items():
            for y, ln in nb: assert (x, ln) in t[y]
            if x >= 9: assert len(nb) == 3
            else: assert len(nb) == 1

    def test_negative_lengths_are_clamped_to_zero(self):
        D = np.array([[0, 1, 5, 5], [1, 0, 5, 5], [5, 5, 0, 0.1], [5, 5, 0.1, 0]], float)   # not additive: NJ would want a negative branch
        t = lo._nj(D)
        assert all(ln >= 0 for nb in t.values() for _, ln in nb)


class TestRooting:
    def test_root_on_edge_splits_the_length(self):
        rng = np.random.default_rng(5); adj, _ = random_additive(rng, 6); u = 0; v, l = adj[0][0]
        r = lo._root_on_edge(adj, u, v, 0.25)
        assert (r, l * 0.25) in adj[u] and (r, l * 0.75) in adj[v] and sorted(adj[r]) == sorted([(u, l * 0.25), (v, l * 0.75)])
        assert all((x, ln) in adj[y] for x, nb in adj.items() for y, ln in nb)
        assert all(y != v for y, _ in adj[u]) and all(y != u for y, _ in adj[v])

    def test_midpoint_is_equidistant_from_the_two_farthest_tips(self):
        rng = np.random.default_rng(6)
        for n in (3, 5, 9):
            adj, D = random_additive(rng, n); u, v, f = lo._midpoint_edge(adj, n)
            r = lo._root_on_edge(adj, u, v, f); dist = tip_distances(adj, n + len(adj) - n)      # distances from the new root
            dr = {}
            stack = [(r, None, 0.0)]
            while stack:
                x, parent, d = stack.pop()
                if x < n: dr[x] = d
                for y, ln in adj[x]:
                    if y != parent: stack.append((y, x, d + ln))
            far = sorted(dr.values())[-2:]
            assert abs(far[0] - far[1]) < 1e-9 and abs(far[0] - D.max() / 2) < 1e-9

    def test_zero_length_tree_midpoint_is_well_defined(self):
        adj = {0: [(2, 0.0)], 1: [(2, 0.0)], 2: [(0, 0.0), (1, 0.0)]}
        u, v, f = lo._midpoint_edge(adj, 2); assert 0.0 <= f <= 1.0


def parse_newick(s):
    """(topology as nested tuples of labels, lengths) for small trees"""
    i = 0
    def node():
        nonlocal i
        if s[i] == "(":
            i += 1; kids = [node()]
            while s[i] == ",": i += 1; kids.append(node())
            i += 1; name = None; children = kids
        else:
            j = i
            while s[i] not in ",:);": i += 1
            name = s[j:i]; children = []
        ln = None
        if s[i] == ":":
            i += 1; j = i
            while s[i] not in ",);": i += 1
            ln = float(s[j:i])
        return (name, tuple(children), ln)
    return node()


class TestNewick:
    def test_round_trip(self):
        rng = np.random.default_rng(8); adj, D = random_additive(rng, 7); labels = [f"x{i}" for i in range(7)] + [None] * (len(adj) - 7)
        u, v, f = lo._midpoint_edge(adj, 7); r = lo._root_on_edge(adj, u, v, f)
        s = lo._newick(adj, r, labels)
        assert s.endswith(";") and s.count("(") == s.count(")") and all(l in s for l in labels[:7])
        tree = parse_newick(s); assert tree[0] is None and len(tree[1]) == 2

    def test_cell_trees_two_newicks(self):
        rng = np.random.default_rng(9); _, De = random_additive(rng, 5); _, Do = random_additive(rng, 5)
        out = lo.cell_trees(De, Do, ["a", "b", "c", "d", "e"])
        assert len(out) == 2 and all(isinstance(t, str) and t.endswith(";") for t in out)
        for t in out: assert all(re.search(rf"\b{l}:", t) for l in "abcde")

    def test_cell_trees_rooted_on_the_outgroup(self):
        rng = np.random.default_rng(10); _, De = random_additive(rng, 5); _, Do = random_additive(rng, 5)
        out = lo.cell_trees(De, Do, ["a", "b", "c", "d", "og"], og=4)
        for t in out:
            root = parse_newick(t); kids = root[1]
            assert len(kids) == 2 and any(k[0] == "og" for k in kids), t            # the outgroup is one child of the root

    def test_cell_trees_too_few_labels(self):
        assert lo.cell_trees(np.zeros((1, 1)), np.zeros((1, 1)), ["a"]) is None

    def test_cell_trees_lengths_are_non_negative_and_rounded(self):
        rng = np.random.default_rng(12); _, De = random_additive(rng, 6); _, Do = random_additive(rng, 6)
        for t in lo.cell_trees(De, Do, list("abcdef")):
            assert all(float(x) >= 0 for x in re.findall(r":(-?[\d.]+)", t))
