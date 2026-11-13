//! The face-to-face trees of a cell: neighbour joining, rooting on an edge, midpoint, newick. A literal port of the
//! Python (dicts in insertion order become vectors; the built-in sum() of floats is Neumaier-compensated since 3.12).
use crate::npmath::{pymax, pymin};
use std::collections::HashMap;

type Adj = Vec<Vec<(usize, f64)>>;

/// Python's sum() over floats, started from the integer 0
fn py_sum(items: impl Iterator<Item = f64>) -> f64 {
    let mut it = items;
    let Some(x0) = it.next() else { return 0.0 };
    let mut f = 0.0 + x0; let mut c = 0.0f64;
    for x in it {
        let t = f + x;
        if f.abs() >= x.abs() { c += (f - t) + x; } else { c += (x - t) + f; }
        f = t;
    }
    if c != 0.0 && c.is_finite() { f += c; }
    f
}

/// neighbour joining on an n x n distance matrix: an unrooted tree as adjacency lists, tips 0..n-1
fn nj(d_: &[f64], n: usize) -> Adj {
    let mut adj: Adj = vec![vec![]; n]; let mut dist: HashMap<(usize, usize), f64> = HashMap::new();
    for i in 0..n { for j in i + 1..n { dist.insert((i, j), d_[i * n + j]); } }
    let d = |dist: &HashMap<(usize, usize), f64>, i: usize, j: usize| -> f64 { if i < j { dist[&(i, j)] } else { dist[&(j, i)] } };
    let mut active: Vec<usize> = (0..n).collect();
    while active.len() > 2 {
        let m = active.len();
        let r: HashMap<usize, f64> = active.iter().map(|&i| (i, py_sum(active.iter().filter(|&&j| j != i).map(|&j| d(&dist, i, j))))).collect();
        let mut best: Option<(f64, usize, usize)> = None;
        for a in 0..m { for b in a + 1..m {
            let (i, j) = (active[a], active[b]); let q = (m - 2) as f64 * d(&dist, i, j) - r[&i] - r[&j];
            if best.is_none() || q < best.unwrap().0 { best = Some((q, i, j)); }
        } }
        let (_, i, j) = best.unwrap(); let dij = d(&dist, i, j);
        let li = 0.5 * dij + (r[&i] - r[&j]) / (2 * (m - 2)) as f64; let lj = dij - li;
        let (li, lj) = (pymax(li, 0.0), pymax(lj, 0.0));
        let u = adj.len(); adj.push(vec![(i, li), (j, lj)]); adj[i].push((u, li)); adj[j].push((u, lj));
        for &k in &active {
            if k != i && k != j { let v = pymax(0.5 * (d(&dist, i, k) + d(&dist, j, k) - dij), 0.0); dist.insert((u.min(k), u.max(k)), v); }
        }
        active.retain(|&k| k != i && k != j); active.push(u);
    }
    if active.len() == 2 { let (i, j) = (active[0], active[1]); let l = pymax(d(&dist, i, j), 0.0); adj[i].push((j, l)); adj[j].push((i, l)); }
    adj
}

fn root_on_edge(adj: &mut Adj, u: usize, v: usize, f: f64) -> usize {
    let l = adj[u].iter().find(|x| x.0 == v).unwrap().1; let r = adj.len();
    adj[u].retain(|x| x.0 != v); adj[u].push((r, l * f));
    adj[v].retain(|x| x.0 != u); adj[v].push((r, l * (1.0 - f)));
    adj.push(vec![(u, l * f), (v, l * (1.0 - f))]);
    r
}

fn midpoint_edge(adj: &Adj, ntips: usize) -> (usize, usize, f64) {
    let mut best: Option<(f64, Vec<usize>)> = None;
    for a in 0..ntips {
        let mut stack: Vec<(usize, Option<usize>, f64, Vec<usize>)> = vec![(a, None, 0.0, vec![a])];
        while let Some((x, parent, dist_, path)) = stack.pop() {
            if x < ntips && x != a && (best.is_none() || dist_ > best.as_ref().unwrap().0) { best = Some((dist_, path.clone())); }
            for &(y, ln) in &adj[x] {
                if Some(y) != parent { let mut p = path.clone(); p.push(y); stack.push((y, Some(x), dist_ + ln, p)); }
            }
        }
    }
    let (total, path) = best.unwrap(); let half = total / 2.0; let mut cum = 0.0;
    for k in 0..path.len() - 1 {
        let (u, v) = (path[k], path[k + 1]);
        let l = adj[u].iter().find(|x| x.0 == v).unwrap().1;
        if cum + l >= half || k == path.len() - 2 { return (u, v, if l == 0.0 { 0.0 } else { pymin(pymax((half - cum) / l, 0.0), 1.0) }); }
        cum += l;
    }
    (path[0], path[1], 0.5)
}

fn newick(adj: &Adj, root: usize, labels: &[String]) -> String {
    fn rec(adj: &Adj, x: usize, parent: Option<usize>, labels: &[String]) -> String {
        let kids: Vec<&(usize, f64)> = adj[x].iter().filter(|k| Some(k.0) != parent).collect();
        if kids.is_empty() { return labels[x].clone(); }
        format!("({})", kids.iter().map(|(y, ln)| format!("{}:{}", rec(adj, *y, Some(x), labels), crate::pyfmt::fixed(*ln, 3))).collect::<Vec<_>>().join(","))
    }
    rec(adj, root, None, labels) + ";"
}

/// (expected newick, observed newick), each rooted where the outgroup tip attaches (midpoint-rooted without one)
pub fn cell_trees(d_exp: &[f64], d_obs: &[f64], labels: &[String], og: Option<usize>) -> Option<(String, String)> {
    let n = labels.len();
    if n < 2 { return None; }
    let mut out = vec![];
    for d in [d_exp, d_obs] {
        let mut t = nj(d, n);
        let (u, v, f) = match og { None => midpoint_edge(&t, n), Some(o) => (o, t[o][0].0, 1.0) };
        let r = root_on_edge(&mut t, u, v, f); out.push(newick(&t, r, labels));
    }
    let b = out.pop().unwrap(); let a = out.pop().unwrap(); Some((a, b))
}
