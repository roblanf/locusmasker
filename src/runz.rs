//! The counting test: comparators, dispersion, the two relative tests, the broad-sample test, the decisions,
//! masked.csv, tests.tsv.gz and the report's per-cell data. A port of run_z, locus by locus.
use crate::input::Loci;
use crate::npmath::*;
use crate::prepare::*;
use crate::pyfmt::{commas, fixed, fnum, pfmt, pfmt10, round, J};
use crate::rep::*;
use crate::{die, jdict, log, Args};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io::Write;

pub const PHI_MIN_E: f64 = 10.0;
pub const PHI_N0: f64 = 30.0;
pub const DENSE_CELLS_MAX: usize = 8_000_000;
pub const BEST_MATCH_ALL_MAX_TAXA: usize = 2000;
pub const BEST_MATCH_MIN_SITES: f64 = 30.0;
/// the batches of loci tested at once (run_z)
pub const TEST_LOCI_IN_FLIGHT: usize = 8;
pub const CODE_SIMILAR: u8 = 128;
pub const CODE_ABSENT: u8 = 255;
pub const BS_BLANK: usize = 12;
pub const MASKED_COLUMNS: [&str; 35] = ["id", "locus", "taxon", "reason", "mask_test", "mask_p", "residues_masked",
    "close_relative_fails", "close_relative_comparators", "close_relative_observed", "close_relative_expected", "close_relative_excess_m", "close_relative_p",
    "closest_match_partner", "closest_match_arm", "closest_match_role", "closest_match_observed", "closest_match_expected", "closest_match_row_factor", "closest_match_p", "closest_match_relatives", "closest_match_partner_excess_m",
    "broad_sample_side", "broad_sample_role", "broad_sample_partner", "broad_sample_partner_mismatches", "broad_sample_partner_expected", "broad_sample_rate_factor", "broad_sample_close_p",
    "broad_sample_members_failing", "broad_sample_members", "broad_sample_divergent_observed", "broad_sample_divergent_expected", "broad_sample_divergent_p",
    "comparators"];
pub const TESTS_COLUMNS: [&str; 10] = ["test", "locus", "taxon", "comparator", "overlap", "mismatches", "expected", "row_factor", "p", "fail"];

/// a locus's comparators: per taxon its first K candidates with the overlap it needs, the closest match at the locus
/// and whether that match is outside the set
pub struct Sel { pub top: Vec<i32>, pub cnt: Vec<i32>, pub nres: Vec<i64>, pub closest: Vec<i32>, pub fork: Vec<bool>, pub exhausted: Vec<bool> }

fn choose_comparators(p: &Prep, fit: i64, k: usize, li: usize, only: Option<&[usize]>) -> Sel {
    let core = &p.core; let (n, nloc) = (core.n, core.nloc);
    let nres: Vec<i64> = (0..n).map(|g| if core.keep[g] { p.focal_sites[g * nloc + li] as i64 } else { 0 }).collect();
    let mut s = Sel { top: vec![-1; n * k], cnt: vec![0; n], nres, closest: vec![-1; n], fork: vec![false; n], exhausted: vec![false; n] };
    let all: Vec<usize>; let rows: &[usize] = match only { Some(o) => o, None => { all = (0..n).collect(); &all } };
    let mut seen: Vec<usize> = Vec::with_capacity(k);
    for &g in rows {
        if s.nres[g] <= 0 { continue; }
        let need = fit.min(s.nres[g]); let mut cnt = 0usize; seen.clear();
        for &c in p.row_cands(g) {
            if cnt == k { break; }
            let c = c as usize;
            if c == g || (core.at(li, g, c).0 as i64) < need { continue; }
            if p.has_dup { let cj = p.cls[c]; if seen.contains(&cj) { continue; } seen.push(cj); }
            s.top[g * k + cnt] = c as i32; cnt += 1;
        }
        s.cnt[g] = cnt as i32;
    }
    for g in 0..n {
        if (s.cnt[g] as usize) < k && s.nres[g] > 0 && p.lists[g].nfin > p.c_len as i64 && !p.ext.contains_key(&g) { s.exhausted[g] = true; }
    }
    if let Some(o) = only { let keep: HashSet<usize> = o.iter().copied().collect(); for g in 0..n { if !keep.contains(&g) { s.exhausted[g] = false; } } }
    for &g in rows {
        if s.cnt[g] <= 0 { continue; }
        let w = p.cl_win[g * nloc + li]; s.closest[g] = w as i32;
        if w < 0 { continue; }
        let dmin = p.cl_dmin[g * nloc + li]; let mut inset = false;
        for j in 0..k {
            let c = s.top[g * k + j]; let valid = c >= 0;
            let (ov, sm) = core.at(li, g, if valid { c as usize } else { 0 }); let (ov, sm) = (ov as f64, sm as f64);
            let d = if valid && ov >= fit as f64 { (ov - sm + HALF) / ov.max(1.0) } else { f64::INFINITY };
            if d <= dmin { inset = true; }
        }
        s.fork[g] = !inset;
    }
    s
}

#[inline] fn corr(p_: f64, p: f64) -> f64 { -p_ * (1.0 - npmin(p, 0.98 * p_) / p_).ln() }
#[inline] fn expect(p_: f64, rate_c: f64, mv: f64) -> f64 { let mvc = npmax(mv, 1e-4); npmax(p_ * (1.0 - (-rate_c * corr(p_, mvc) / p_).exp()), 1e-4) }
/// Python's float ** 2 (the C library's pow, which the optimiser must not turn into a product)
#[inline] fn pow2(x: f64) -> f64 { x.powf(std::hint::black_box(2.0)) }

pub struct Pairs {
    idx: Vec<usize>, gi: Vec<usize>, ci: Vec<usize>, rk: Vec<usize>,
    o: Vec<f32>, mis: Vec<f32>, ident: Vec<f32>, mp: Vec<f64>, mph: Vec<f64>, fit: Vec<bool>, rate_c: f64, p_: f64,
}
enum LP { Skip(String), Ok(Pairs) }

pub struct Ctx<'a> {
    pub a: &'a Args, pub names: &'a [String], pub loci: &'a Loci, pub p: &'a Prep, pub sel: &'a [Sel], pub bs: Option<&'a BsPools>,
    pub k: usize, pub m_eff: usize, pub phi_l: &'a [f64], pub logbar: f64, pub ladj: f64, pub ladj10: f64, pub ln10: f64, pub lf: &'a [f64],
    pub rows: &'a [u8], pub l: usize, pub every: bool, pub mask_hi: bool, pub mask_lo: bool, pub codes_policy: &'a str, pub seq_policy: &'a str,
    pub bm_all: bool, pub dense: bool,
}
impl Ctx<'_> {
    #[inline] fn adj10(&self, v: f64) -> f64 { npmin(v + self.ladj10, 0.0) }
    #[inline] fn adj10f(&self, v: f32) -> f32 { let x = v + self.ladj10 as f32; if x.is_nan() { x } else if x < 0.0 { x } else { 0.0 } }
}

fn locus_pairs(a: &Args, p: &Prep, sel: &Sel, k: usize, li: usize, lb: Option<&LocusBits>) -> LP {
    let core = &p.core; let n = core.n;
    let idx: Vec<usize> = (0..n).filter(|&g| sel.cnt[g] > 0).collect();
    if idx.is_empty() { return LP::Skip("skipped-no-cells".into()); }
    let (mut gi, mut ci, mut rk) = (vec![], vec![], vec![]);
    for &g in &idx { for j in 0..k { let c = sel.top[g * k + j]; if c >= 0 { gi.push(g); ci.push(c as usize); rk.push(j); } } }
    let np = gi.len();
    let (mut o, mut mis, mut ident, mut dob) = (Vec::with_capacity(np), Vec::with_capacity(np), Vec::with_capacity(np), Vec::with_capacity(np));
    let (mut mp, mut mph, mut fit) = (Vec::with_capacity(np), Vec::with_capacity(np), Vec::with_capacity(np));
    // each pair's counts (a read of the two taxa's planes) and its m from the lists: the pairs in parallel, kept in order
    let per: Vec<(f32, f32, f64, f64)> = (0..np).into_par_iter().with_min_len(256).map(|j| {
        let (oi, si) = match lb { Some(lb) => core.at_l(lb, li, gi[j], ci[j]), None => core.at(li, gi[j], ci[j]) }; let (m_, f_) = p.pair_m_lists(gi[j], ci[j]);
        (oi as f32, si as f32, m_, f_)
    }).collect();
    for (of, sf, m_, f_) in per {
        o.push(of); mis.push(of - sf);
        let id = 100.0f32 * sf / of; ident.push(id); dob.push(100.0f32 - id);
        mp.push(m_); mph.push(npmax(m_, f_)); let mpx = npmax(m_ - f_, 0.0);
        fit.push(of >= a.fit_overlap as f32 && mpx > 0.0);
    }
    let nfit = fit.iter().filter(|x| **x).count();
    if nfit < MIN_FIT_PAIRS && !p.rate[li].is_finite() { return LP::Skip(format!("skipped-few-resolved-pairs({})", nfit)); }
    let p_ = p.plateaus[li];
    let rate_c = if nfit >= MIN_FIT_PAIRS {
        let v: Vec<f64> = (0..np).filter(|&j| fit[j]).map(|j| corr(p_, (dob[j] + 50.0f32 / o[j]) as f64) / corr(p_, mph[j])).collect();
        pymax(median(&v), 1e-3)
    } else { pymax(p.rate[li], 1e-3) };
    LP::Ok(Pairs { idx, gi, ci, rk, o, mis, ident, mp, mph, fit, rate_c, p_ })
}

/// the dispersion of a locus and the tests it will run: (phi, pairs behind it, distinct comparisons, closest-match tests)
fn locus_phi(a: &Args, p: &Prep, sel: &Sel, k: usize, li: usize) -> (f64, f64, i64, i64) {
    let l = match locus_pairs(a, p, sel, k, li, None) { LP::Skip(_) => return (f64::NAN, 0.0, 0, 0), LP::Ok(l) => l };
    let n = p.n(); let core = &p.core;
    let mut keys: Vec<u64> = (0..l.gi.len()).map(|j| (l.gi[j].min(l.ci[j]) * n + l.gi[j].max(l.ci[j])) as u64).collect();
    keys.sort_unstable(); keys.dedup(); let ndist = keys.len() as i64;
    let mut npt = 0i64;
    for g in 0..n {
        if !sel.fork[g] { continue; }
        let y = sel.closest[g]; let mut okr = 0;
        for j in 0..k {
            let t = sel.top[g * k + j];
            let wo = core.at(li, if t >= 0 { t as usize } else { 0 }, y as usize).0 as i64;
            if t >= 0 && t != y && wo >= a.fit_overlap { okr += 1; }
        }
        if okr >= 3 { npt += 1; }
    }
    let np = l.gi.len();
    let e: Vec<f64> = (0..np).map(|j| l.o[j] as f64 * expect(l.p_, l.rate_c, l.mph[j]) / 100.0).collect();
    let used: Vec<usize> = (0..np).filter(|&j| l.fit[j] && e[j] >= PHI_MIN_E).collect();
    if used.len() < 10 { return (f64::NAN, 0.0, ndist, npt); }
    let r: Vec<f64> = used.iter().map(|&j| ((l.mis[j] + 0.5f32) as f64 / e[j]).ln()).collect();
    let mr = median(&r); let dev: Vec<f64> = r.iter().map(|x| (x - mr).abs()).collect();
    let s_ = 1.4826 * median(&dev);
    let inv: Vec<f64> = used.iter().map(|&j| 1.0 / e[j]).collect();
    (pymax(pow2(s_) - median(&inv), 0.0), used.len() as f64, ndist, npt)
}

/// the locus's own distance curve for the broad-sample test: knots (m*), curve values (percent) and dispersions.
/// t: every valid broad comparison of the locus as (m*, mismatches, shared sites), in row order (the sort is stable)
/// the grid of pair divergences the broad sample's curve is binned on
const BS_GRID: [f64; 30] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 26.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 85.0, 100.0, 1e9];
/// the grid cell of a pair divergence: the first grid value it does not exceed (one more cell beyond the grid)
#[inline] fn bs_cell(m: f64) -> usize { BS_GRID.partition_point(|g| *g < m) }
type BsCells = Vec<Vec<(f64, u32, u32)>>;
fn bs_cells_new() -> BsCells { vec![Vec::new(); BS_GRID.len() + 1] }

/// The locus's curve from its comparisons (m*, mismatches, shared sites), given by grid cell (bs_cell). The curve's
/// bins are runs of whole cells and what a bin contributes are medians and percentiles of its points, so the points
/// need no order: this is the curve of the points sorted by m* (the Python sorts them), without the sort
fn bs_curve(cells: BsCells, phi_l: f64) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let (min_points, min_window) = (60usize, 30usize);
    let k: usize = cells.iter().map(|c| c.len()).sum();
    if k < min_points { return None; }
    // a bin ends at a grid value once it holds min_window points: starts in points, cuts in cells
    let mut starts: Vec<usize> = vec![0]; let mut cuts: Vec<usize> = vec![0]; let (mut last, mut cum) = (0usize, 0usize);
    for gi in 0..BS_GRID.len() { cum += cells[gi].len(); if cum - last >= min_window { starts.push(cum); cuts.push(gi + 1); last = cum; } }
    if *starts.last().unwrap() < k {
        if k - starts.last().unwrap() >= min_window { starts.push(k); cuts.push(cells.len()); } else { *starts.last_mut().unwrap() = k; *cuts.last_mut().unwrap() = cells.len(); }
    }
    if starts.len() < 2 { return None; }
    let nb = starts.len() - 1;
    let bin = |b: usize| cells[cuts[b]..cuts[b + 1]].iter().flatten();
    // each bin's medians and dispersion are its own: the bins in parallel, by selection (median_of, percentile_of:
    // the values are positive ratios and their logarithms, without NaN or negative zeros, so these are the numbers
    // the sorts gave)
    let per_bin: Vec<(f64, f64, f64)> = (0..nb).into_par_iter().map(|b| {
        let q: Vec<f64> = bin(b).map(|x| (x.1 as f64 + HALF) / npmax(x.2 as f64, 1.0)).collect();
        (median_of(bin(b).map(|x| x.0).collect::<Vec<f64>>()), 100.0 * median_of(q), (starts[b + 1] - starts[b]) as f64)
    }).collect();
    let xm: Vec<f64> = per_bin.iter().map(|x| x.0).collect(); let yd: Vec<f64> = per_bin.iter().map(|x| x.1).collect(); let cnt: Vec<f64> = per_bin.iter().map(|x| x.2).collect();
    let yd2 = pav(&yd, &cnt);
    let yphi: Vec<f64> = (0..nb).into_par_iter().map(|b| {
        // the points with enough expected mismatches: their log ratio of observed to expected, and 1 / expected
        let (mut r, mut inv): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
        for x in bin(b) { let e = x.2 as f64 * yd2[b] / 100.0; if e >= PHI_MIN_E { r.push(((x.1 as f64 + HALF) / e).ln()); inv.push(1.0 / e); } }
        if r.len() < 10 { return phi_l; }
        let nu = r.len() as f64;
        let s_ = (percentile_of(&mut r, 90.0) - percentile_of(&mut r, 10.0)) / 2.5631;
        let ph = pymax(s_ * s_ - median_of(inv), 0.0);
        (nu * ph + PHI_N0 * phi_l) / (nu + PHI_N0)
    }).collect();
    let mut x = vec![0.0]; x.extend(xm); let mut y = vec![0.0]; y.extend(yd2); let mut ph = vec![yphi[0]]; ph.extend(yphi);
    let mut acc = f64::NEG_INFINITY;
    for (i, v) in x.iter_mut().enumerate() { let t = *v + i as f64 * 1e-9; acc = if i == 0 { t } else { npmax(acc, t) }; *v = acc; }
    Some((x, y, ph))
}

#[derive(Clone)]
struct Rec { cmp_arm: bool, o: f64, mis: f64, e: f64, f: f64, lnp: f64, rel: Vec<usize>, uncertain: bool }
#[derive(Clone, Copy, PartialEq)]
enum Role { Copy, Source, Uncertain }
impl Role { fn s(self) -> &'static str { match self { Role::Copy => "copy", Role::Source => "source", Role::Uncertain => "uncertain" } } }

/// the broad test's per-locus results, one entry per row (a present cell of a kept taxon)
pub struct BsBlk {
    pub rows: Vec<usize>, pub nt: Vec<i16>, pub ntr: Vec<i16>, pub nfail: Vec<i16>,
    pub lo: Vec<f64>, pub lo_j: Vec<i32>, pub lo_o: Vec<f64>, pub lo_w: Vec<f64>, pub lo_e: Vec<f64>, pub lo_f: Vec<f64>, pub lo_p: Vec<f64>,
    pub hi: Vec<f64>, pub hi_j: Vec<i32>, pub hi_o: Vec<f64>, pub hi_w: Vec<f64>, pub hi_e: Vec<f64>, pub hi_p: Vec<f64>,
    pub call: Vec<i8>, pub dirc: Vec<i8>, pub medoe: Vec<f32>, pub rep: Vec<i32>, pub dm: Vec<bool>, pub obs_pct: Vec<f32>, pub exp_pct: Vec<f32>,
    pub cfail: Vec<i16>, pub score: Vec<i16>, pub score_p: Vec<i16>, pub exm_f: Vec<f32>, pub exm_p: Vec<f32>, pub excl: Vec<usize>,
}

pub struct RepBlk {
    pub idx: Vec<usize>, pub n_comp: Vec<u8>, pub n_fail: Vec<u8>, pub n_similar: Vec<u8>, pub side: Vec<u8>, pub cols: Vec<Col>,
    pub comparators: Vec<i32>, pub comp_div: Vec<f32>, pub comp_sim: Vec<f32>, pub fail_q: Vec<bool>,
    pub codes: Option<(Vec<i32>, Vec<u8>)>, pub seq: Option<(Vec<i32>, Vec<u8>)>, pub trees: Option<Trees>, pub similar: Vec<J>,
}
pub struct MRow { pub g: usize, pub reason: &'static str, pub row: Vec<String>, pub mlp: f64 }
pub struct LocusOut { pub skipped: Option<String>, pub log: String, pub mrows: Vec<MRow>, pub tests: String, pub rep: Option<RepBlk>, pub phi: f64, pub rate_c: f64, pub rc: Option<BsBlk> }

/// sorted NaN-last, read beyond the end as NaN: a row of cell_matrix after its sort
struct SortedPad(Vec<f64>);
impl SortedPad {
    fn new(mut v: Vec<f64>) -> Self { sort_f64(&mut v); SortedPad(v) }
    #[inline] fn at(&self, i: usize) -> f64 { if i < self.0.len() { self.0[i] } else { f64::NAN } }
}

/// the residues of taxon g at locus li
/// a taxon's residue codes at a locus (read back from the locus's planes)
#[inline] fn xs(p: &Prep, lb: &LocusBits, g: usize, li: usize) -> Vec<i8> { lb.seq(g, p.core.off[li + 1] - p.core.off[li]) }

/// the site codes of cells (g, partner or -1): how many comparators differ from the focal at each site, the blue of a copy
fn site_codes(cx: &Ctx, lb: &LocusBits, li: usize, cells: &[(usize, i64)]) -> Vec<u8> {
    let p = cx.p; let k = cx.k; let top = &cx.sel[li].top; let ls = cx.loci[li].1.len();
    let mut out = vec![0u8; cells.len() * ls];
    out.par_chunks_mut(ls.max(1)).zip(cells.par_iter()).for_each(|(row, &(g, bq))| {
        if ls == 0 { return; }
        let fx = xs(p, lb, g, li); let mut kk = vec![0u16; ls];
        for j in 0..k {
            let c = top[g * k + j];
            if c < 0 || c as i64 == bq { continue; }
            let xc = xs(p, lb, c as usize, li);
            for s in 0..ls { if xc[s] != fx[s] && xc[s] >= 0 { kk[s] += 1; } }
        }
        for s in 0..ls { row[s] = if fx[s] >= 0 { kk[s].min(127) as u8 } else { CODE_ABSENT }; }
        if bq >= 0 {
            let xb = xs(p, lb, bq as usize, li);
            for s in 0..ls { if fx[s] >= 0 && xb[s] >= 0 && fx[s] == xb[s] && kk[s] >= 2 { row[s] = CODE_SIMILAR + kk[s].min(125) as u8; } }
        }
    });
    out
}

/// the face-to-face trees of cells (focal, tips): the outgroup search, both distance matrices, neighbour joining
fn build_trees(cx: &Ctx, lb: &LocusBits, li: usize, cells_t: Vec<(usize, Vec<usize>)>, p_: f64, rate_c: f64) -> Option<Trees> {
    let p = cx.p; let n = p.n(); let core = &p.core; let fit = cx.a.fit_overlap as f64;
    let mut t_out = Trees::default();
    for (g, mut tips) in cells_t {
        let mut og: i64 = -1;
        if let Some(mf) = &p.m_full {
            let in_tips: HashSet<usize> = tips.iter().copied().collect(); let nt = tips.len();
            let mut best: Option<(f64, usize)> = None; let mut vals = vec![0.0; nt];
            for c in 0..n {
                if in_tips.contains(&c) { continue; }
                let mut ok = true;
                for (ti, &t) in tips.iter().enumerate() {
                    let m = mf[t * n + c]; vals[ti] = m;
                    if !((core.at_l(lb, li, t, c).0 as f64) >= fit) || !m.is_finite() { ok = false; break; }
                }
                if !ok { continue; }
                let mean = pairwise_sum(&vals) / nt as f64;
                if best.is_none() || mean > best.unwrap().0 { best = Some((mean, c)); }
            }
            if let Some((_, c)) = best { og = c as i64; tips.push(c); }
        }
        let nt = tips.len();
        let mut de = vec![f64::NAN; nt * nt]; let mut d_o = vec![f64::NAN; nt * nt];
        let tip_x: Vec<Vec<i8>> = tips.iter().map(|&t| xs(p, lb, t, li)).collect();
        for i in 0..nt { for j in 0..nt {
            let (a, b) = (&tip_x[i], &tip_x[j]); let (mut nb, mut df) = (0u64, 0u64);
            for s in 0..a.len() { let both = a[s] >= 0 && b[s] >= 0; nb += both as u64; df += (both && a[s] != b[s]) as u64; }
            d_o[i * nt + j] = if nb > 0 { 100.0 * df as f64 / (nb.max(1)) as f64 } else { f64::NAN };
            let mv = if i == j { f64::NAN } else { p.pair_m_any(tips[i], tips[j]) };
            de[i * nt + j] = if mv.is_finite() { expect(p_, rate_c, mv) } else { f64::NAN };
        } }
        for i in 0..nt { de[i * nt + i] = 0.0; d_o[i * nt + i] = 0.0; }
        if de.iter().any(|x| x.is_nan()) {
            let mx = de.iter().copied().filter(|x| !x.is_nan()).fold(f64::NEG_INFINITY, f64::max);
            let fill = if mx.is_finite() { mx } else { 50.0 };
            for x in de.iter_mut() { if x.is_nan() { *x = fill; } }
        }
        for i in 0..nt * nt { if d_o[i].is_nan() { d_o[i] = de[i]; } }
        for x in de.iter_mut() { *x = corr(p_, *x); } for x in d_o.iter_mut() { *x = corr(p_, *x); }
        for i in 0..nt { de[i * nt + i] = 0.0; d_o[i * nt + i] = 0.0; }
        let labels: Vec<String> = tips.iter().map(|x| x.to_string()).collect();
        let ogi = if og >= 0 { tips.iter().position(|&t| t as i64 == og) } else { None };
        if let Some((te, to)) = crate::tree::cell_trees(&de, &d_o, &labels, ogi) {
            let cand: Vec<bool> = (0..nt).map(|j| j != 0 && Some(j) != ogi).collect();
            let argmin = |d: &[f64]| -> i64 {
                if !cand.iter().any(|x| *x) { return -1; }
                let mut b = 0usize; let mut bv = if cand[0] { d[0] } else { f64::INFINITY };
                for j in 1..nt { let v = if cand[j] { d[j] } else { f64::INFINITY }; if v < bv || (bv.is_nan() && false) { bv = v; b = j; } }
                b as i64
            };
            let (ce, co) = (argmin(&de[..nt]), argmin(&d_o[..nt]));
            t_out.cells.push(g as i64); t_out.expected.push(te); t_out.observed.push(to);
            t_out.tips.push(tips.iter().map(|x| *x as i64).collect()); t_out.outgroup.push(ogi.map_or(-1, |x| x as i64)); t_out.closest.push([ce, co]);
        }
    }
    if t_out.cells.is_empty() { None } else { Some(t_out) }
}

/// the raw alignment rows of the given taxa at a locus's sites
fn seq_rows(cx: &Ctx, li: usize, rows_s: &[usize]) -> Vec<u8> {
    let sites = &cx.loci[li].1; let mut out = Vec::with_capacity(rows_s.len() * sites.len());
    for &g in rows_s { let r = &cx.rows[g * cx.l..(g + 1) * cx.l]; for &s in sites { out.push(r[s]); } }
    out
}

fn first_argmin(v: impl Iterator<Item = f64>) -> usize {
    let mut b = 0; let mut bv = f64::NAN; let mut first = true;
    for (i, x) in v.enumerate() { if first { bv = x; first = false; if x.is_nan() { return 0; } } else if x.is_nan() { return i; } else if x < bv { bv = x; b = i; } }
    b
}
fn first_argmax(v: impl Iterator<Item = f64>) -> usize {
    let mut b = 0; let mut bv = f64::NAN; let mut first = true;
    for (i, x) in v.enumerate() { if first { bv = x; first = false; if x.is_nan() { return 0; } } else if x.is_nan() { return i; } else if x > bv { bv = x; b = i; } }
    b
}

/// everything the test decides and writes for one locus; touches no shared state
pub fn process_locus(cx: &Ctx, li: usize) -> LocusOut {
    let (name, sites) = (&cx.loci[li].0, &cx.loci[li].1); let a = cx.a; let p = cx.p; let core = &p.core; let n = p.n(); let k = cx.k; let sel = &cx.sel[li];
    // the locus's planes of every taxon, side by side: what the tests below count on
    let lb_own = core.bits.locus(li, n); let lb = &lb_own;
    let l = match locus_pairs(a, p, sel, k, li, Some(lb)) {
        LP::Skip(s) => return LocusOut { skipped: Some(s.clone()), log: format!("{}: {}", name, s), mrows: vec![], tests: String::new(), rep: None, phi: f64::NAN, rate_c: f64::NAN, rc: None },
        LP::Ok(l) => l,
    };
    let Pairs { idx, gi, ci, rk, o, mis, ident, mp, mph, fit: _, rate_c, p_ } = &l; let (rate_c, p_) = (*rate_c, *p_);
    let phi = cx.phi_l[li]; let top = &sel.top; let lf = cx.lf; let logbar = cx.logbar; let fit_f = a.fit_overlap as f64;
    let np = gi.len(); let ncell = idx.len();
    let o64: Vec<f64> = o.iter().map(|x| *x as f64).collect(); let mis64: Vec<f64> = mis.iter().map(|x| *x as f64).collect();
    let dhat: Vec<f64> = mph.iter().map(|&m| expect(p_, rate_c, m)).collect();
    let e: Vec<f64> = (0..np).map(|j| o64[j] * dhat[j] / 100.0).collect();
    let up: Vec<bool> = (0..np).map(|j| mis64[j] - e[j] > 0.0).collect();
    let kc: Vec<usize> = idx.iter().map(|&g| sel.cnt[g] as usize).collect();
    let mut starts = Vec::with_capacity(ncell); { let mut acc = 0; for q in 0..ncell { starts.push(acc); acc += kc[q]; } }
    let mut row = Vec::with_capacity(np); for q in 0..ncell { for _ in 0..kc[q] { row.push(q); } }
    let cell = |v: &[f64], q: usize| -> Vec<f64> { v[starts[q]..starts[q] + kc[q]].to_vec() };
    let cell_median = |v: &[f64]| -> Vec<f64> { (0..ncell).map(|q| { let s = SortedPad::new(cell(v, q)); (s.at((kc[q] - 1) / 2) + s.at(kc[q] / 2)) / 2.0 }).collect() };
    let ident64: Vec<f64> = ident.iter().map(|x| *x as f64).collect();
    let med_i = cell_median(&ident64); let med_e = cell_median(&dhat.iter().map(|d| 100.0 - d).collect::<Vec<_>>());

    // ---- test 2's comparator arm: the row factors
    let mut qpos = vec![-1i64; n]; for (q, &g) in idx.iter().enumerate() { qpos[g] = q as i64; }
    let cpos = |g: usize| -> Option<usize> { if qpos[g] >= 0 { Some(qpos[g] as usize) } else { None } };
    let e_lo: Vec<f64> = (0..np).map(|j| o64[j] * expect(p_, rate_c, mp[j]) / 100.0).collect();
    let oe: Vec<f64> = (0..np).map(|j| (mis[j] + 0.5f32) as f64 / (e_lo[j] + HALF)).collect();
    let med_capped = |mut vals: Vec<f64>, kcount: usize| -> f64 {
        sort_f64(&mut vals); let s = SortedPad(vals); let kk = kcount.max(1);
        let v = (s.at((kk - 1) / 2) + s.at(kk / 2)) / 2.0;
        if kcount >= 2 { npmin(v, 1.0) } else { 1.0 }
    };
    let mut f_all = vec![1.0f64; n]; let mut f_loo = vec![1.0f64; n * k];
    for (q, &g) in idx.iter().enumerate() {
        f_all[g] = med_capped(cell(&oe, q), kc[q]);
        for r_ in 0..k {
            let mut v = cell(&oe, q); let cnt = if kc[q] > r_ { v[r_] = f64::NAN; kc[q] - 1 } else { kc[q] };
            f_loo[g * k + r_] = med_capped(v, cnt);
        }
    }
    let rf = |x: usize, y: usize| -> f64 { match (0..k).find(|&j| top[x * k + j] == y as i32) { Some(j) => f_loo[x * k + j], None => f_all[x] } };
    let ff: Vec<f64> = (0..np).map(|j| npmin(rf(gi[j], ci[j]), rf(ci[j], gi[j]))).collect();
    let dn: Vec<bool> = (0..np).map(|j| mis64[j] < e_lo[j]).collect();

    // ---- test 1, too divergent
    // the exact tails are the work of the comparator tests: the pairs in parallel
    let mut lp_hi: Vec<f64> = (0..np).into_par_iter().with_min_len(256).map(|j| if up[j] { bb_tail(mis64[j], o64[j], e[j], phi, true, lf) } else { f64::NAN }).collect();
    let fail_hi: Vec<bool> = (0..np).map(|j| up[j] && lp_hi[j] < logbar).collect();
    let nf: Vec<usize> = (0..ncell).map(|q| (starts[q]..starts[q] + kc[q]).filter(|&j| fail_hi[j]).count()).collect();
    let flag: Vec<bool> = (0..ncell).map(|q| kc[q] >= k && nf[q] >= cx.m_eff).collect();
    let amb: Vec<bool> = (0..ncell).map(|q| kc[q] < k && nf[q] >= kc[q] / 2 + 1).collect();
    let dflag: Vec<bool> = (0..ncell).map(|q| flag[q] || amb[q]).collect();
    let d_cmp: Vec<f64> = (0..np).map(|j| (corr(p_, (100.0f32 - ident[j]) as f64) - corr(p_, dhat[j])) / rate_c).collect();
    let junk_c: Vec<bool> = (0..np).map(|j| qpos[ci[j]] >= 0 && dflag[qpos[ci[j]] as usize]).collect();
    let med_dcmp = cell_median(&d_cmp);
    let exm: Vec<f64> = (0..ncell).map(|q| {
        let kk = kc[q] - (starts[q]..starts[q] + kc[q]).filter(|&j| junk_c[j]).count();
        if kk > 0 {
            let s = SortedPad::new((starts[q]..starts[q] + kc[q]).map(|j| if junk_c[j] { f64::NAN } else { d_cmp[j] }).collect());
            (s.at((kk - 1) / 2) + s.at(kk.max(1) / 2)) / 2.0
        } else { med_dcmp[q] }
    }).collect();
    let lp_in: Vec<f64> = (0..np).into_par_iter().with_min_len(256).map(|j| {
        if !dn[j] { return 0.0; }
        let v = bb_tail(mis64[j], o64[j], e_lo[j], phi, false, lf);
        if ff[j] < 1.0 { npmax(v, bb_tail(mis64[j], o64[j], e_lo[j] * ff[j], phi, false, lf)) } else { v }
    }).collect();
    let fail_in: Vec<bool> = lp_in.iter().map(|x| *x < logbar).collect();
    let mut pairs: Vec<((usize, usize), Rec)> = vec![]; let mut pair_ix: HashMap<(usize, usize), usize> = HashMap::new();
    let mut record = |key: (usize, usize), rec: Rec| {
        match pair_ix.get(&key) { Some(&i) => { if rec.lnp < pairs[i].1.lnp { pairs[i].1 = rec; } } None => { pair_ix.insert(key, pairs.len()); pairs.push((key, rec)); } }
    };
    for j in 0..np {
        if fail_in[j] { let (g, c) = (gi[j], ci[j]); record((g.min(c), g.max(c)), Rec { cmp_arm: true, o: o64[j], mis: mis64[j], e: e_lo[j], f: ff[j], lnp: lp_in[j], rel: vec![], uncertain: false }); }
    }
    // ---- test 2's closest-match arm
    let y_: Vec<i64> = idx.iter().map(|&g| if sel.fork[g] { sel.closest[g] as i64 } else { -1 }).collect();
    let mut tested2 = vec![false; ncell]; let mut o2 = vec![0.0f64; ncell]; let mut mis2 = vec![0.0f64; ncell]; let mut e2 = vec![f64::NAN; ncell];
    let mut lp_lo = vec![0.0f64; ncell]; let mut rel2: Vec<Vec<usize>> = vec![vec![]; ncell];
    // a cell of the arm reads the planes of its relatives and of its closest match: the cells in parallel
    let arm: Vec<Option<(f64, f64, f64, Vec<usize>, f64)>> = (0..ncell).into_par_iter().with_min_len(64).map(|q| {
        let g = idx[q]; if !sel.fork[g] { return None; }
        let yv = y_[q] as usize; let mut rates = Vec::with_capacity(k); let mut rel = vec![];
        for j in 0..k {
            let c = top[g * k + j]; let c0 = c.max(0) as usize; let qc = qpos[c0];
            let mut valid = c >= 0 && c0 != yv && !(qc >= 0 && dflag[qc.max(0) as usize]);
            let (wo, ws) = core.at_l(lb, li, c0, yv); let (w, ws) = (wo as f64, ws as f64); valid &= w >= fit_f;
            if valid { rates.push((w - ws) / w.max(1.0)); rel.push(c0); }
        }
        if rates.len() < 3 { return None; }
        let (go, gs) = core.at_l(lb, li, g, yv);
        let o = go as f64; let mis = o - gs as f64; let e = o * nanmedian(&rates);
        Some((o, mis, e, rel, if mis < e { bb_tail(mis, o, e, phi, false, lf) } else { 0.0 }))
    }).collect();
    for (q, x) in arm.into_iter().enumerate() { if let Some((o, mis, e, rel, lp)) = x { o2[q] = o; mis2[q] = mis; e2[q] = e; tested2[q] = true; rel2[q] = rel; lp_lo[q] = lp; } }
    let fail2: Vec<bool> = (0..ncell).map(|q| tested2[q] && lp_lo[q] < logbar).collect();
    for q in 0..ncell {
        if fail2[q] { let (g, y) = (idx[q], y_[q] as usize); record((g.min(y), g.max(y)), Rec { cmp_arm: false, o: o2[q], mis: mis2[q], e: e2[q], f: 1.0, lnp: lp_lo[q], rel: rel2[q].clone(), uncertain: false }); }
    }
    // the pairs that failed, each once: the copy is the member with the larger |test-1 excess|
    let mut copy_q = vec![false; ncell]; let mut src_q = vec![false; ncell]; let mut partner_ = vec![-1i32; ncell];
    let mut npair = vec![0usize; ncell]; let mut pair_of: Vec<Option<(usize, usize, Role)>> = vec![None; ncell];
    for pi in 0..pairs.len() {
        let (g, y) = pairs[pi].0;
        let (qg, qy) = (cpos(g), cpos(y));
        let xg = match qg { Some(q) if exm[q].is_finite() => exm[q].abs(), _ => f64::NEG_INFINITY };
        let xy = match qy { Some(q) if exm[q].is_finite() => exm[q].abs(), _ => f64::NEG_INFINITY };
        let (cp, sr) = if xg >= xy { (g, y) } else { (y, g) };
        let uncertain = xg.is_finite() && xy.is_finite() && pymax(xg, xy) <= a.donor_threshold * pymin(xg, xy);
        pairs[pi].1.uncertain = uncertain;
        for (member, other, is_copy) in [(cp, sr, true), (sr, cp, false)] {
            let Some(qm) = cpos(member) else { continue };
            npair[qm] += 1;
            let role = if uncertain { Role::Uncertain } else if is_copy { Role::Copy } else { Role::Source };
            if role == Role::Source { src_q[qm] = true; } else { copy_q[qm] = true; }
            if pair_of[qm].is_none() || pairs[pi].1.lnp < pairs[pair_of[qm].unwrap().0].1.lnp { pair_of[qm] = Some((pi, other, role)); }
        }
    }
    for q in 0..ncell { if let Some((_, other, _)) = pair_of[q] { partner_[q] = other as i32; } }
    let unc_q: Vec<bool> = (0..ncell).map(|q| matches!(pair_of[q], Some((_, _, Role::Uncertain)))).collect();
    for q in 0..ncell { if tested2[q] && partner_[q] < 0 { partner_[q] = y_[q] as i32; } }
    let sflag = &copy_q;

    // ---- the decision
    let masked_q: Vec<bool> = (0..ncell).map(|q| (dflag[q] && cx.mask_hi) || (sflag[q] && cx.mask_lo)).collect();
    let fail_q: Vec<bool> = (0..ncell).map(|q| nf[q] >= 1 || fail2[q] || copy_q[q] || src_q[q] || masked_q[q]).collect();
    lp_hi.par_iter_mut().enumerate().with_min_len(256).for_each(|(j, v)| { if v.is_nan() && (fail_q[row[j]] || cx.every) { *v = bb_tail(mis64[j], o64[j], e[j], phi, true, lf); } });
    let div_lp: Vec<f64> = (0..ncell).map(|q| {
        if !fail_q[q] { return f64::NAN; }
        let need = if kc[q] >= k { cx.m_eff } else { kc[q] / 2 + 1 };
        SortedPad::new(cell(&lp_hi, q)).at(need.max(1) - 1)
    }).collect();
    let ln10 = cx.ln10;
    let mut comp_div_ = vec![f32::NAN; ncell * k]; let mut comp_sim_ = vec![f32::NAN; ncell * k];
    for j in 0..np {
        comp_div_[row[j] * k + rk[j]] = ((if lp_hi[j].is_nan() { 0.0 } else { lp_hi[j] }) / ln10) as f32;
        comp_sim_[row[j] * k + rk[j]] = (lp_in[j] / ln10) as f32;
    }
    let lclos_: Vec<i32> = idx.iter().map(|&g| sel.closest[g]).collect(); let mut lclos_p = vec![f32::NAN; ncell];
    for q in 0..ncell {
        let hasy = lclos_[q] >= 0; let viaarm = hasy && tested2[q] && y_[q] == lclos_[q] as i64;
        if viaarm { lclos_p[q] = (lp_lo[q] / ln10) as f32; }
        else if hasy { if let Some(j) = (0..k).find(|&j| top[idx[q] * k + j] == lclos_[q]) { lclos_p[q] = comp_sim_[q * k + j]; } }
    }

    let names = cx.names; let ladj = cx.ladj;
    let pex_of = |other: usize| -> String { match cpos(other) { Some(qo) => fnum(exm[qo], 3), None => String::new() } };
    let cm_cols = |q: usize| -> Vec<String> {
        if let Some((pi, other, role)) = pair_of[q] {
            let rec = &pairs[pi].1; let in_set = rec.cmp_arm;
            return vec![names[other].clone(), (if in_set { "relative" } else { "outside" }).to_string(), role.s().to_string(),
                        fnum(100.0 * rec.mis / rec.o, 2), fnum(100.0 * rec.e / rec.o, 2), if in_set { fnum(rec.f, 3) } else { String::new() }, pfmt(rec.lnp + ladj),
                        if in_set { String::new() } else { rec.rel.iter().map(|&c| names[c].as_str()).collect::<Vec<_>>().join("; ") }, pex_of(other)];
        }
        if tested2[q] {
            let y = y_[q] as usize;
            return vec![names[y].clone(), "outside".into(), String::new(), fnum(100.0 * mis2[q] / o2[q], 2), fnum(100.0 * e2[q] / o2[q], 2),
                        String::new(), pfmt(lp_lo[q] + ladj), rel2[q].iter().map(|&c| names[c].as_str()).collect::<Vec<_>>().join("; "), pex_of(y)];
        }
        let (s_, k_) = (starts[q], kc[q]);
        if k_ == 0 { return vec![String::new(); 9]; }
        let j = s_ + first_argmin(lp_in[s_..s_ + k_].iter().copied());
        vec![names[ci[j]].clone(), "relative".into(), String::new(), fnum((100.0f32 * mis[j] / o[j]) as f64, 2), fnum(100.0 * e_lo[j] / o64[j], 2),
             fnum(ff[j], 3), pfmt(lp_in[j] + ladj), String::new(), pex_of(ci[j])]
    };
    let mut mrows_out = vec![];
    for q in 0..ncell {
        if !masked_q[q] { continue; }
        let g = idx[q]; let dv = dflag[q] && cx.mask_hi; let sm = sflag[q] && cx.mask_lo;
        let reason = if dv && sm { "both" } else if dv { "divergent" } else { "similar" };
        let mut mc: Vec<(f64, &str)> = vec![];
        if dv && div_lp[q].is_finite() { mc.push((div_lp[q], "close_relative")); }
        if sm { if let Some((pi, _, _)) = pair_of[q] { mc.push((pairs[pi].1.lnp, "closest_match")); } }
        let best = mc.iter().copied().reduce(|x, y| if y.0 < x.0 || (y.0 == x.0 && y.1 < x.1) { y } else { x });
        let (mt, mp_txt, mlp) = match best { Some((v, t)) => (t.to_string(), pfmt(v + ladj), v), None => (String::new(), String::new(), f64::INFINITY) };
        let s = starts[q]; let comps: Vec<&str> = (s..s + kc[q]).map(|j| names[ci[j]].as_str()).collect();
        let mut row_ = vec![name.clone(), names[g].clone(), reason.to_string(), mt, mp_txt, sel.nres[g].to_string(), nf[q].to_string(), kc[q].to_string(),
                            fnum(100.0 - med_i[q], 2), fnum(100.0 - med_e[q], 2), fnum(exm[q], 3), pfmt(div_lp[q] + ladj)];
        row_.extend(cm_cols(q)); row_.extend(vec![String::new(); BS_BLANK]); row_.push(comps.join("; "));
        mrows_out.push(MRow { g, reason, row: row_, mlp });
    }

    // ---- the report's view of this locus
    let side_: Vec<u8> = (0..ncell).map(|q| {
        let mut s = 0u8;
        if dflag[q] { s |= if cx.mask_hi { 1 } else { 16 }; } if amb[q] { s |= 4; }
        if sflag[q] { s |= if cx.mask_lo { 2 } else { 32 }; } if tested2[q] { s |= 64; }
        if src_q[q] { s |= 128; } if unc_q[q] { s |= 8; }
        s
    }).collect();

    // ---- the broad-sample test
    let mut bs_blk = None; let mut bsrep_full = vec![-1i64; n];
    if let Some(bs) = cx.bs {
        let mut dmask = vec![false; n]; let mut cpmask = vec![false; n];
        if cx.mask_hi { for q in 0..ncell { if dflag[q] { dmask[idx[q]] = true; } } }
        if cx.mask_lo { for q in 0..ncell { if matches!(pair_of[q], Some((_, _, Role::Copy))) && sflag[q] { cpmask[idx[q]] = true; } } }
        let nloc = core.nloc;
        let rows_bs: Vec<usize> = (0..n).filter(|&g| core.keep[g] && p.focal_sites[g * nloc + li] > 0).collect();
        if !rows_bs.is_empty() {
            // Row by row, in three passes, so that nothing of size rows x members is held but each member's taxon, its
            // pool slot and whether it is judged: (0) the draws and the valid comparisons' counts, for the locus's curve;
            // (A) each row's expectations, tails and too-close side; (B) the too-divergent side, once the rows that fail
            // it themselves are known; (C) the direction, once every row's score is known. The numbers are those of the
            // whole-matrix expressions of the Python
            let nr = rows_bs.len(); let bk = bs.k; let pn = bs.pn; let nn = nr * bk;
            let cnt = |g: usize, c: usize| -> (f64, f64) { let (sh, sm) = lb.count(g, c); (sh as f64, (sh - sm) as f64) };      // shared sites, mismatches
            let ok0_of = |c: usize, w: f64, mv0: f64| w >= fit_f && mv0.is_finite() && core.keep[c] && !dmask[c] && !cpmask[c];
            // (0) the rows in parallel, each writing its own members; the curve's comparisons go into the cells of its
            // grid as they are counted (the curve needs them by cell, in no order)
            let mut comp = vec![0u32; nn]; let mut pos = vec![0u32; nn];
            let curve: BsCells = comp.par_chunks_mut(bk).zip(pos.par_chunks_mut(bk)).zip(rows_bs.par_iter()).with_min_len(64).fold(bs_cells_new, |mut cells, ((cm, ps_), &g)| {
                for (j, &ps) in hash_draw(bs_keys_cell(g, li), pn, bk).iter().enumerate() {
                    let c = bs.pool[g * pn + ps as usize] as usize; let mv0 = bs.mstar[g * pn + ps as usize];
                    cm[j] = c as u32; ps_[j] = ps as u32;
                    let (w, o) = cnt(g, c);
                    if ok0_of(c, w, mv0) { let m = npmax(mv0, 1e-4); cells[bs_cell(m)].push((m, o as u32, w as u32)); }
                }
                cells
            }).reduce(bs_cells_new, |mut x, mut y| { for (p, q) in x.iter_mut().zip(y.iter_mut()) { if p.len() < q.len() { std::mem::swap(p, q); } p.append(q); } x });
            let bcv = bs_curve(curve, phi);
            let clip = |x: f64, lo: f64, hi: f64| npmin(npmax(x, lo), hi);
            // a member's expected mismatches, expected divergence (%) and dispersion from its m* and shared sites
            let vals = |mv: f64, w: f64| -> (f64, f64, f64) {
                let (d0, ph) = match &bcv {
                    None => (npmax(p_ * (1.0 - (-rate_c * corr(p_, mv) / p_).exp()), 1e-4), phi),
                    Some((bx, by, bp)) => (npmax(interp(mv, bx, by), 1e-4), interp(mv, bx, bp)),
                };
                let e_ = clip(npmax(w * d0 / 100.0, HALF), 1e-9, w * (1.0 - 1e-9));
                (e_, 100.0 * e_ / npmax(w, 1.0), ph)
            };
            let mut m5 = vec![f64::NAN; n];
            for (q, &g) in idx.iter().enumerate() { let mut mx = f64::NAN; for j in starts[q]..starts[q] + kc[q] { if !mph[j].is_nan() && (mx.is_nan() || mph[j] > mx) { mx = mph[j]; } } m5[g] = mx; }
            let mut r_full = vec![1.0f64; n];
            for (q, &g) in idx.iter().enumerate() {
                let r_cell = (100.0 - med_i[q] + 0.5) / (100.0 - med_e[q] + 0.5);
                r_full[g] = npmin(if r_cell.is_finite() && r_cell > 0.0 { r_cell } else { 1.0 }, 1.0);
            }
            let mut exm_full = vec![f64::NAN; n]; for (q, &g) in idx.iter().enumerate() { exm_full[g] = exm[q]; }
            let lnb = |c: i64| ((c.max(1)) as f64).ln(); let lean_ = (0.01f64).ln(); let thr_d = a.donor_threshold;
            let nmed = |v: &[f64], ok: &[bool]| -> f64 { let x: Vec<f64> = (0..bk).filter(|&j| ok[j]).map(|j| v[j]).collect(); if x.is_empty() { f64::NAN } else { nanmedian(&x) } };
            // ---- pass A: a row's results depend on the locus's curve and on the row alone, so the rows go in parallel
            struct RowA { nt: i64, lo_stat: f64, lo_j: i64, lo_o: f64, lo_w: f64, lo_e: f64, lo_f: f64, lo_p: f64, dm: bool, bad: bool, cfail: i64, score: i64, a_e: f64, b_e: f64,
                          rep: i64, medoe: f64, obs_pct: f64, exp_pct: f64, hi: Vec<(u32, f64)> }
            let mut btested = vec![false; nn];
            let scratch = || (vec![0.0f64; bk], vec![0.0f64; bk], vec![0.0f64; bk], vec![false; bk], vec![0.0f64; bk], vec![0.0f64; bk], vec![0.0f64; bk], vec![0.0f64; bk], vec![0.0f64; bk], vec![0.0f64; bk], vec![0.0f64; bk], vec![0.0f64; bk]);
            let rows_a: Vec<RowA> = btested.par_chunks_mut(bk).zip(rows_bs.par_iter()).enumerate().with_min_len(64).map_init(scratch, |(w_, o_, mv, ok0, e_, dexp, ratio, bf, blo, bhi, zhi, obs_v), (r, (bt, &g))| {
                let cm = &comp[r * bk..(r + 1) * bk];
                for j in 0..bk {
                    let c = cm[j] as usize; let mv0 = bs.mstar[g * pn + pos[r * bk + j] as usize]; let (w, o) = cnt(g, c);
                    w_[j] = w; o_[j] = o; mv[j] = npmax(mv0, 1e-4); ok0[j] = ok0_of(c, w, mv0);
                    let (e, d, _) = vals(mv[j], w); e_[j] = e; dexp[j] = d;
                }
                let medm = nmed(mv, ok0); let thr = if m5[g].is_finite() { m5[g] } else { medm }; let thr = if thr.is_finite() { thr } else { f64::INFINITY };
                let mut nt = 0i64;
                for j in 0..bk {
                    bt[j] = ok0[j] && mv[j] > thr; nt += bt[j] as i64;
                    ratio[j] = (o_[j] + HALF) / (e_[j] + HALF); obs_v[j] = 100.0 * o_[j] / npmax(w_[j], 1.0);
                    bf[j] = npmin((r_full[g] * r_full[cm[j] as usize]).sqrt(), 1.0);
                    blo[j] = 0.0; bhi[j] = 0.0; zhi[j] = f64::NAN;
                    if !bt[j] { continue; }
                    let phi_m = vals(mv[j], w_[j]).2;
                    let ef = clip(e_[j] * bf[j], 1e-9, w_[j] * (1.0 - 1e-9)); let df = dexp[j] * bf[j];
                    let z_lo = (o_[j] - ef) / npmax(ef * (1.0 - df / 100.0) + phi_m * ef * ef, 1e-12).sqrt();
                    zhi[j] = (o_[j] - e_[j]) / npmax(e_[j] * (1.0 - dexp[j] / 100.0) + phi_m * e_[j] * e_[j], 1e-12).sqrt();
                    if z_lo < -2.0 { blo[j] = bb_tail(o_[j], w_[j], ef, phi_m, false, lf); }
                    if zhi[j] > 2.0 { bhi[j] = bb_tail(o_[j], w_[j], e_[j], phi_m, true, lf); }
                }
                let lpl = |j: usize| if bt[j] { blo[j] } else { f64::INFINITY };
                let jlo = first_argmin((0..bk).map(lpl));
                let lo_j = if nt > 0 { cm[jlo] as i64 } else { -1 };
                let dm = dmask[g];
                let fh = (0..bk).filter(|&j| bt[j] && bhi[j] < logbar).count() as f64;
                let tb = logbar - lnb(nt);
                let az = |j: usize| if bt[j] && zhi[j].is_finite() { zhi[j].abs() } else { f64::NEG_INFINITY };
                let jr = first_argmax((0..bk).map(az));
                RowA { nt, lo_stat: if nt > 0 { npmin(lpl(jlo) + lnb(nt), 0.0) } else { 0.0 }, lo_j, lo_o: o_[jlo], lo_w: w_[jlo], lo_e: e_[jlo], lo_f: bf[jlo], lo_p: lpl(jlo), dm,
                       bad: fh > nt as f64 / 2.0 && nt >= BS_MIN && !dm,
                       cfail: (0..bk).filter(|&j| bt[j] && blo[j] < tb).count() as i64,
                       score: (0..bk).filter(|&j| bt[j] && blo[j] < lean_).count() as i64,        // the too-divergent half is added in pass B
                       a_e: (if exm_full[g].is_finite() { exm_full[g] } else { 0.0 }).abs(),
                       b_e: { let jx = lo_j.max(0) as usize; (if lo_j >= 0 && exm_full[jx].is_finite() { exm_full[jx] } else { 0.0 }).abs() },
                       rep: if az(jr).is_finite() { cm[jr] as i64 } else { -1 },                  // the called side's partner replaces it in pass B
                       medoe: nmed(ratio, bt), obs_pct: nmed(obs_v, bt), exp_pct: nmed(dexp, bt),
                       hi: (0..bk).filter(|&j| bhi[j] != 0.0).map(|j| (j as u32, bhi[j])).collect() }
            }).collect();
            let mut hi_off: Vec<usize> = Vec::with_capacity(nr + 1); let mut hi_val: Vec<(u32, f64)> = Vec::new();
            let mut bnt = vec![0i64; nr]; let mut lo_stat = vec![0.0f64; nr]; let mut lo_j = vec![-1i64; nr];
            let (mut lo_o, mut lo_w, mut lo_e, mut lo_f, mut lo_p) = (vec![0.0f64; nr], vec![0.0f64; nr], vec![0.0f64; nr], vec![0.0f64; nr], vec![0.0f64; nr]);
            let mut dm_r = vec![false; nr]; let mut cfail = vec![0i64; nr]; let mut score = vec![0i64; nr]; let mut a_e = vec![0.0f64; nr]; let mut b_e = vec![0.0f64; nr];
            let mut rep_ = vec![-1i64; nr]; let (mut medoe, mut obs_pct, mut exp_pct) = (vec![0.0f64; nr], vec![0.0f64; nr], vec![0.0f64; nr]);
            let mut bad_r = vec![false; n];
            for (r, x) in rows_a.into_iter().enumerate() {
                bnt[r] = x.nt; lo_stat[r] = x.lo_stat; lo_j[r] = x.lo_j; lo_o[r] = x.lo_o; lo_w[r] = x.lo_w; lo_e[r] = x.lo_e; lo_f[r] = x.lo_f; lo_p[r] = x.lo_p;
                dm_r[r] = x.dm; if x.bad { bad_r[rows_bs[r]] = true; }
                cfail[r] = x.cfail; score[r] = x.score; a_e[r] = x.a_e; b_e[r] = x.b_e; rep_[r] = x.rep; medoe[r] = x.medoe; obs_pct[r] = x.obs_pct; exp_pct[r] = x.exp_pct;
                hi_off.push(hi_val.len()); hi_val.extend(x.hi);
            }
            hi_off.push(hi_val.len());
            // ---- pass B: with every row's own failure known (bad_r), the rows in parallel again
            struct RowB { ntr: i64, nfail: i64, sc2: i64, hi_j: i64, p_red: f64, call: i8, rep: i64, hi_o: f64, hi_w: f64, hi_e: f64, hi_p: f64 }
            let rows_b: Vec<RowB> = rows_bs.par_iter().enumerate().with_min_len(64).map_init(|| (vec![0.0f64; bk], vec![0.0f64; bk]), |(lph, bhi), (r, &g)| {
                let cm = &comp[r * bk..(r + 1) * bk]; let bt = &btested[r * bk..(r + 1) * bk];
                bhi.fill(0.0); for &(j, v) in &hi_val[hi_off[r]..hi_off[r + 1]] { bhi[j as usize] = v; }
                let (mut nt, mut nf_, mut sc2) = (0i64, 0i64, 0i64);
                for j in 0..bk {
                    let tr = bt[j] && !bad_r[cm[j] as usize];
                    lph[j] = if tr { bhi[j] } else { f64::INFINITY };
                    if tr { nt += 1; if bhi[j] < logbar { nf_ += 1; } if bhi[j] < lean_ { sc2 += 1; } }
                }
                let red = nf_ as f64 > nt as f64 / 2.0 && nt >= BS_MIN && !dm_r[r];
                let jhi = first_argmin(lph.iter().copied());
                let hi_j = if nt > 0 { cm[jhi] as i64 } else { -1 };
                let p_red = if nt > 0 { let mut srt = lph.clone(); sort_f64(&mut srt); npmin(srt[((nt / 2) as usize).min(bk - 1)], 0.0) } else { 0.0 };
                let bclose = lo_stat[r] < logbar && bnt[r] >= BS_MIN;
                let mut call = 0i8;
                if bclose { call = 1; }
                if red && !bclose { call = -1; }
                if bclose && red { call = if lo_stat[r] <= p_red { 1 } else { -1 }; }
                let rep = if call == 1 { lo_j[r] } else if call == -1 { hi_j } else { rep_[r] };
                // the member behind the too-divergent side: its counts and expectation again
                let c = cm[jhi] as usize; let mv0 = bs.mstar[g * pn + pos[r * bk + jhi] as usize]; let (w, o) = cnt(g, c);
                RowB { ntr: nt, nfail: nf_, sc2, hi_j, p_red, call, rep, hi_o: o, hi_w: w, hi_e: vals(npmax(mv0, 1e-4), w).0, hi_p: lph[jhi] }
            }).collect();
            let mut ntr = vec![0i64; nr]; let mut nfail = vec![0i64; nr]; let mut hi_j = vec![-1i64; nr]; let mut p_red = vec![0.0f64; nr]; let mut bcall = vec![0i8; nr];
            let (mut hi_o, mut hi_w, mut hi_e, mut hi_p) = (vec![0.0f64; nr], vec![0.0f64; nr], vec![0.0f64; nr], vec![0.0f64; nr]);
            for (r, x) in rows_b.into_iter().enumerate() {
                ntr[r] = x.ntr; nfail[r] = x.nfail; score[r] += x.sc2; hi_j[r] = x.hi_j; p_red[r] = x.p_red; bcall[r] = x.call; rep_[r] = x.rep;
                hi_o[r] = x.hi_o; hi_w[r] = x.hi_w; hi_e[r] = x.hi_e; hi_p[r] = x.hi_p; bsrep_full[rows_bs[r]] = x.rep;
            }
            // ---- pass C: the direction of a too-close pair
            let mut score_full = vec![0i64; n]; for r in 0..nr { score_full[rows_bs[r]] = score[r]; }
            let sp_: Vec<i64> = (0..nr).map(|r| score_full[lo_j[r].max(0) as usize]).collect();
            let dirc: Vec<i8> = (0..nr).map(|r| {
                if score[r] >= sp_[r] + 2 { 1 } else if sp_[r] >= score[r] + 2 { -1 }
                else if a_e[r] >= thr_d * b_e[r] && a_e[r] > 0.0 { 1 } else if b_e[r] >= thr_d * a_e[r] && b_e[r] > 0.0 { -1 } else { 0 }
            }).collect();
            bs_blk = Some(BsBlk {
                rows: rows_bs.clone(), nt: bnt.iter().map(|x| *x as i16).collect(), ntr: ntr.iter().map(|x| *x as i16).collect(), nfail: nfail.iter().map(|x| *x as i16).collect(),
                lo: lo_stat, lo_j: lo_j.iter().map(|x| *x as i32).collect(), lo_o, lo_w, lo_e, lo_f, lo_p,
                hi: p_red, hi_j: hi_j.iter().map(|x| *x as i32).collect(), hi_o, hi_w, hi_e, hi_p,
                call: bcall, dirc, medoe: medoe.iter().map(|x| (if x.is_finite() { *x } else { 1.0 }) as f32).collect(), rep: rep_.iter().map(|x| *x as i32).collect(), dm: dm_r,
                obs_pct: obs_pct.iter().map(|x| *x as f32).collect(), exp_pct: exp_pct.iter().map(|x| *x as f32).collect(),
                cfail: cfail.iter().map(|x| *x as i16).collect(), score: score.iter().map(|x| *x as i16).collect(), score_p: sp_.iter().map(|x| *x as i16).collect(),
                exm_f: a_e.iter().map(|x| *x as f32).collect(), exm_p: b_e.iter().map(|x| *x as f32).collect(),
                excl: (0..n).filter(|&g| dmask[g] || cpmask[g] || bad_r[g]).collect(),
            });
        }
    }
    let mut simlp_ = vec![f32::NAN; ncell];
    for q in 0..ncell { if tested2[q] { simlp_[q] = (lp_lo[q] / ln10) as f32; } }
    for q in 0..ncell { if let Some((pi, _, _)) = pair_of[q] { simlp_[q] = (pairs[pi].1.lnp / ln10) as f32; } }
    let rep_q: Vec<bool> = match cx.codes_policy { "fails" => fail_q.clone(), "masked" => masked_q.clone(), _ => vec![false; ncell] };

    // ---- best match: per-site divergence compared on the same sites
    let rows_bm: Vec<usize> = if cx.bm_all { (0..ncell).collect() } else { (0..ncell).filter(|&q| fail_q[q]).collect() };
    let mut best_ = vec![-1i32; ncell]; let mut bident_ = vec![f32::NAN; ncell];
    // A cell against every taxon. The cells go in blocks: one pass over the taxa counts every cell of a block against
    // each taxon (a taxon's planes at the locus are read once for the block), the counts kept as integers in buffers
    // each thread keeps; the cells of a block then go in parallel, each reading its tiers and challengers from its
    // own counts. The numbers are those of the whole-row expressions: okc = enough shared sites and not itself,
    // dvg = mismatches / shared sites. A block is small where there are few cells, so that they spread over the threads
    let bm_block = ((32usize << 20) / (8 * n.max(1))).clamp(1, 64).min(rows_bm.len().div_ceil(2 * rayon::current_num_threads().max(1))).max(1);
    let bm: Vec<(usize, i32, f32)> = rows_bm.par_chunks(bm_block).map_init(|| (vec![0u32; bm_block * n], vec![0u32; bm_block * n]), |(wv, sv), qs| {
        let gs: Vec<usize> = qs.iter().map(|&q| idx[q]).collect();
        if core.slab.is_some() { for (bi, &g) in gs.iter().enumerate() { for c in 0..n { let (w, s) = core.at(li, g, c); wv[bi * n + c] = w; sv[bi * n + c] = s; } } }
        else { lb.block(&gs, &core.keep, n, wv, sv); }
        let (wv, sv, gs) = (&*wv, &*sv, &gs);
        qs.par_iter().enumerate().map(|(bi, &q)| {
            let g = gs[bi]; let (wr, sr) = (&wv[bi * n..(bi + 1) * n], &sv[bi * n..(bi + 1) * n]);
            let okc = |c: usize| wr[c] as f64 >= BEST_MATCH_MIN_SITES && c != g;
            let dvg = |c: usize| if okc(c) { (wr[c] as f64 - sr[c] as f64) / (wr[c] as f64).max(1.0) } else { f64::INFINITY };
            let mut mcache: HashMap<usize, f64> = HashMap::new();
            let mut mb = |c: usize| -> f64 { let v = match &p.m_full { Some(m) => m[g * n + c], None => *mcache.entry(c).or_insert_with(|| p.pair_m_any(g, c)) }; if v.is_finite() { v } else { f64::INFINITY } };
            // the tier of the most shared sites, and the lowest divergence in it
            let wmax = (0..n).map(|c| if okc(c) { wr[c] as f64 } else { -1.0 }).fold(f64::NEG_INFINITY, f64::max);
            let lo_t = (0..n).map(|c| if okc(c) && wr[c] as f64 >= wmax { dvg(c) } else { f64::INFINITY }).fold(f64::INFINITY, f64::min);
            if !lo_t.is_finite() { return (q, -1, f32::NAN); }                 // no taxon shares enough sites
            // the tier's taxa at that divergence, and the taxa outside it with a lower one (the challengers)
            let mut tied: Vec<usize> = vec![]; let mut t0: Vec<usize> = vec![];
            for c in 0..n { if !okc(c) { continue; } let d = dvg(c); if wr[c] as f64 >= wmax && d == lo_t { tied.push(c); } if d < lo_t { t0.push(c); } }
            let mut champ = tied.first().copied().unwrap_or(0);
            if tied.len() > 1 {
                let tk: Vec<f64> = tied.iter().map(|&c| mb(c)).collect();
                if tk.iter().any(|x| x.is_finite()) { champ = tied[first_argmin(tk.iter().copied())]; }
            }
            if !t0.is_empty() {
                // The challengers in order of more shared sites, then lower divergence, then (among those equal in
                // both) lower pair divergence m, then taxon; each in turn against the champion on the sites all three
                // have (counted on the planes: the sites, where the challenger differs from the focal, where the
                // champion does), the first that wins taking over and the ones after it going on against it.
                // Taken a group of equals at a time: a group in which nobody wins is passed whatever its order, so
                // the m that orders a group (a pair kernel for each member) is computed only for a group with a
                // winner. The outcome is that of the whole list sorted and walked
                let mut c = champ; let mut mc_: Option<f64> = None;
                let kw: Vec<f64> = t0.iter().map(|&t| -(wr[t] as f64)).collect(); let kd: Vec<f64> = t0.iter().map(|&t| dvg(t)).collect();
                let mut ord: Vec<usize> = (0..t0.len()).collect();
                ord.sort_by(|&x, &y| cmp_nan_last(&kw[x], &kw[y]).then(cmp_nan_last(&kd[x], &kd[y])));
                let mut i0 = 0usize;
                while i0 < ord.len() {
                    let mut i1 = i0 + 1;
                    while i1 < ord.len() && kw[ord[i1]].to_bits() == kw[ord[i0]].to_bits() && kd[ord[i1]].to_bits() == kd[ord[i0]].to_bits() { i1 += 1; }
                    let single = i1 - i0 == 1;
                    let mut rem: Vec<usize> = ord[i0..i1].iter().map(|&x| t0[x]).collect();          // the group's taxa still to come, ascending
                    let mut km: Option<Vec<f64>> = None;                                                // their m, once the order is needed
                    while !rem.is_empty() {
                        let mut wins: Vec<usize> = vec![];
                        for (pos, &t) in rem.iter().enumerate() {
                            let (common, dt_, dc_) = lb.three(g, c, t);
                            if (common as f64) < BEST_MATCH_MIN_SITES { continue; }
                            if dt_ < dc_ || (dt_ == dc_ && { let mt = mb(t); let mcv = *mc_.get_or_insert_with(|| mb(c)); mt < mcv }) { wins.push(pos); }
                        }
                        if wins.is_empty() { break; }
                        if single { c = rem[0]; mc_ = None; break; }
                        let kmv = match km.take() { Some(v) => v, None => rem.iter().map(|&t| mb(t)).collect() };
                        let after = |a: usize, b: usize| cmp_nan_last(&kmv[a], &kmv[b]).then(rem[a].cmp(&rem[b]));
                        let w = wins.iter().copied().min_by(|&a, &b| after(a, b)).unwrap();
                        let stay: Vec<usize> = (0..rem.len()).filter(|&x| after(x, w) == std::cmp::Ordering::Greater).collect();
                        c = rem[w]; mc_ = None;
                        km = Some(stay.iter().map(|&x| kmv[x]).collect()); rem = stay.iter().map(|&x| rem[x]).collect();
                    }
                    i0 = i1;
                }
                champ = c;
            }
            (q, champ as i32, (100.0 * (1.0 - dvg(champ))) as f32)
        }).collect::<Vec<_>>()
    }).flatten().collect();
    for (q, b, bi) in bm { best_[q] = b; bident_[q] = bi; }

    let codes_out = if rep_q.iter().any(|x| *x) {
        let qs: Vec<usize> = (0..ncell).filter(|&q| rep_q[q]).collect();
        let cells: Vec<(usize, i64)> = qs.iter().map(|&q| (idx[q], if copy_q[q] || src_q[q] { partner_[q] as i64 } else { -1 })).collect();
        Some((qs.iter().map(|&q| idx[q] as i32).collect::<Vec<i32>>(), site_codes(cx, lb, li, &cells)))
    } else { None };
    let trees_out = if masked_q.iter().any(|x| *x) {
        let mut cells_t = vec![];
        for q in 0..ncell {
            if !masked_q[q] { continue; }
            let g = idx[q]; let mut tips: Vec<usize> = vec![g]; for j in 0..k { let c = top[g * k + j]; if c >= 0 { tips.push(c as usize); } }
            for t_ in [partner_[q] as i64, lclos_[q] as i64, bsrep_full[g]] { if t_ >= 0 && !tips.contains(&(t_ as usize)) { tips.push(t_ as usize); } }
            if tips.len() >= 2 { cells_t.push((g, tips)); }
        }
        build_trees(cx, lb, li, cells_t, p_, rate_c)
    } else { None };
    let seq_out = if cx.seq_policy != "none" {
        let rows_s: Vec<usize> = if cx.seq_policy == "all" { (0..n).filter(|&g| sel.nres[g] > 0).collect() } else {
            let want = if cx.seq_policy == "fails" { &fail_q } else { &masked_q }; let mut rs: HashSet<usize> = HashSet::new();
            for q in 0..ncell {
                if !want[q] { continue; }
                let g = idx[q]; rs.insert(g); for j in 0..k { let c = top[g * k + j]; if c >= 0 { rs.insert(c as usize); } }
                if partner_[q] >= 0 { rs.insert(partner_[q] as usize); }
                if bsrep_full[g] >= 0 { rs.insert(bsrep_full[g] as usize); }
            }
            let mut v: Vec<usize> = rs.into_iter().collect(); v.sort_unstable(); v
        };
        if rows_s.is_empty() { None } else { Some((rows_s.iter().map(|x| *x as i32).collect::<Vec<i32>>(), seq_rows(cx, li, &rows_s))) }
    } else { None };
    let exm_j = |x: usize| -> J { match cpos(x) { Some(q) if exm[q].is_finite() => J::Float(round(exm[q], 4)), _ => J::Null } };
    let mut sim_json: Vec<J> = pairs.iter().map(|((g, y), rec)| {
        let role_g = cpos(*g).and_then(|q| pair_of[q]).map(|x| x.2);
        jdict! { "test" => J::s(if rec.cmp_arm { "relative" } else { "outside" }), "taxon" => J::Int(*g as i64), "partner" => J::Int(*y as i64),
                 "overlap" => J::Int(rec.o as i64), "mismatches" => J::Int(rec.mis as i64), "expected" => J::Float(round(rec.e, 3)), "row_factor" => J::Float(round(rec.f, 4)),
                 "relatives" => J::List(rec.rel.iter().map(|c| J::Int(*c as i64)).collect()), "log10p" => J::Float(round(cx.adj10(rec.lnp / ln10), 3)), "fail" => J::Bool(true),
                 "copy" => J::Int(if rec.uncertain { -1 } else if role_g == Some(Role::Copy) { *g as i64 } else { *y as i64 }), "uncertain" => J::Bool(rec.uncertain),
                 "excess_m" => J::List(vec![exm_j(*g), exm_j(*y)]) }
    }).collect();
    for q in 0..ncell {
        if tested2[q] && !fail2[q] {
            sim_json.push(jdict! { "test" => J::s("outside"), "taxon" => J::Int(idx[q] as i64), "partner" => J::Int(y_[q]), "overlap" => J::Int(o2[q] as i64), "mismatches" => J::Int(mis2[q] as i64),
                                   "expected" => J::Float(round(e2[q], 3)), "row_factor" => J::Float(1.0), "relatives" => J::List(rel2[q].iter().map(|c| J::Int(*c as i64)).collect()),
                                   "log10p" => J::Float(round(cx.adj10(lp_lo[q] / ln10), 3)), "fail" => J::Bool(false), "copy" => J::Int(-1),
                                   "excess_m" => J::List(vec![if exm[q].is_finite() { J::Float(round(exm[q], 4)) } else { J::Null }, J::Null]) });
        }
    }
    let f32v = |v: &[f64]| -> Col { Col::F32(v.iter().map(|x| *x as f32).collect()) };
    let cols = vec![
        f32v(&med_i), f32v(&med_e), f32v(&exm),
        Col::I32((0..ncell).map(|q| ci[starts[q]] as i32).collect()), Col::U32((0..ncell).map(|q| o[starts[q]] as u32).collect()),
        Col::F32((0..ncell).map(|q| ident[starts[q]]).collect()), Col::F32((0..ncell).map(|q| (100.0 - dhat[starts[q]]) as f32).collect()),
        Col::I32(best_), Col::F32(bident_), Col::I32(partner_.clone()),
        Col::F32(div_lp.iter().map(|x| cx.adj10(x / ln10) as f32).collect()), Col::F32(simlp_.iter().map(|x| cx.adj10f(*x)).collect()),
        Col::I32(lclos_.clone()), Col::F32(lclos_p.iter().map(|x| cx.adj10f(*x)).collect())];
    let mut comparators = Vec::with_capacity(ncell * k); for &g in idx { comparators.extend_from_slice(&top[g * k..(g + 1) * k]); }
    let repblk = RepBlk { idx: idx.clone(), n_comp: kc.iter().map(|x| *x as u8).collect(), n_fail: nf.iter().map(|x| *x as u8).collect(),
                          n_similar: npair.iter().map(|x| (*x).min(255) as u8).collect(), side: side_, cols, comparators,
                          comp_div: comp_div_.iter().map(|x| cx.adj10f(*x)).collect(), comp_sim: comp_sim_.iter().map(|x| cx.adj10f(*x)).collect(),
                          fail_q: fail_q.clone(), codes: codes_out, seq: seq_out, trees: trees_out, similar: sim_json };

    // ---- every test of the locus, for --all-tests
    let mut tests_block = String::new();
    if cx.every {
        let clean = |x: &str| x.replace(['\t', '\n', '\r'], " ");
        let lname = clean(name); let tn = |g: usize| clean(&names[g]);
        let r3 = |v: f64| fixed((v * 1000.0).round_ties_even() / 1000.0 + 0.0, 3);
        let p6 = |v: f64| fixed((npmin(v, 0.0).exp() * 1e6).round_ties_even() / 1e6 + 0.0, 6);
        for j in 0..np {
            tests_block.push_str(&format!("close_relative\t{}\t{}\t{}\t{}\t{}\t{}\t\t{}\t{}\n", lname, tn(gi[j]), tn(ci[j]), o[j] as i64, mis[j] as i64, r3(e[j]), p6(lp_hi[j] + ladj), fail_hi[j] as i64));
        }
        for j in 0..np {
            tests_block.push_str(&format!("closest_match_relative\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", lname, tn(gi[j]), tn(ci[j]), o[j] as i64, mis[j] as i64, r3(e_lo[j]), fixed(ff[j], 4), p6(lp_in[j] + ladj), fail_in[j] as i64));
        }
        for q in 0..ncell {
            if tested2[q] {
                tests_block.push_str(&format!("closest_match_outside\t{}\t{}\t{}\t{}\t{}\t{}\t\t{}\t{}\n", lname, tn(idx[q]), tn(y_[q] as usize), o2[q] as i64, mis2[q] as i64, r3(e2[q]), p6(lp_lo[q] + ladj), fail2[q] as i64));
            }
        }
    }
    let log = format!("{}: {} cells, {} comparator slots, {} closest-match tests; too divergent {} (+{} with fewer comparators), too-similar pairs {} ({} by the closest match)",
                      name, ncell, np, tested2.iter().filter(|x| **x).count(), flag.iter().filter(|x| **x).count(), amb.iter().filter(|x| **x).count(),
                      pairs.len(), pairs.iter().filter(|x| !x.1.cmp_arm).count());
    let _ = sites;
    LocusOut { skipped: None, mrows: mrows_out, tests: tests_block, log, rep: Some(repblk), phi, rate_c, rc: bs_blk }
}

// ------------------------------------------------------------------ the broad-sample verdict pass, per locus
#[derive(Default)]
struct BsOut {
    mask: Vec<(usize, u8, f64, i64)>, note: Vec<(usize, u8, f64, i64)>,          // (taxon, side2 bits, statistic, partner)
    label: Vec<(usize, f64, i64)>, lean: Vec<(usize, f64, f64, i64, i8)>, dlean: Vec<(usize, f64, i64)>,
    codes: Option<(Vec<i32>, Vec<u8>)>, trees: Option<Trees>, seq: Option<(Vec<i32>, Vec<u8>)>, broad: Vec<J>,
}

fn bs_locus_b(cx: &Ctx, rep: &Rep, b: &BsBlk, li: usize, near_ln: f64) -> Option<BsOut> {
    let rate_c = rep.test_rate[li];
    if !rate_c.is_finite() { return None; }
    let p = cx.p; let k = cx.k; let nloc = p.core.nloc; let top = &cx.sel[li].top; let logbar = cx.logbar; let ln10 = cx.ln10; let p_ = p.plateaus[li];
    let mut out = BsOut::default(); let mut tree_cells: Vec<(usize, i64, bool)> = vec![];
    let side_at = |g: usize| rep.side[g * nloc + li];
    for (i, &g) in b.rows.iter().enumerate() {
        let call = b.call[i]; let nt_ = b.nt[i] as i64; let ntr_ = b.ntr[i] as i64;
        if nt_ < BS_MIN { continue; }
        let (lo_, hi_, dm_) = (b.lo[i], b.hi[i], b.dm[i]);
        if dm_ {
            let partner = b.lo_j[i] as i64;
            if lo_ < logbar && partner >= 0 {
                let (w, o, e) = (b.lo_w[i], b.lo_o[i], b.lo_e[i]);
                out.label.push((g, lo_, partner));
                out.broad.push(jdict! { "side" => J::s("close"), "taxon" => J::Int(g as i64), "partner" => J::Int(partner), "overlap" => J::Int(w as i64), "mismatches" => J::Int(o as i64),
                                        "expected" => J::Float(round(e, 3)), "rho" => J::Float(round(b.lo_f[i], 4)), "log10p" => J::Float(round(cx.adj10(lo_ / ln10), 3)), "nf" => J::Int(nt_), "nfail" => J::Int(0),
                                        "direction" => J::s("label"), "pair_log10p" => J::Float(round(b.lo_p[i] / ln10, 3)) });
            } else { out.dlean.push((g, lo_, b.rep[i] as i64)); }
            continue;
        }
        if call == 0 {
            if lo_ < logbar + near_ln || (hi_ < logbar + near_ln && ntr_ >= BS_MIN) { out.lean.push((g, lo_, hi_, b.rep[i] as i64, 0)); }
            continue;
        }
        let close = call == 1; let stat = if close { lo_ } else { hi_ };
        let partner = (if close { b.lo_j[i] } else { b.hi_j[i] }) as i64;
        let (w, o, e) = if close { (b.lo_w[i], b.lo_o[i], b.lo_e[i]) } else { (b.hi_w[i], b.hi_o[i], b.hi_e[i]) };
        out.lean.push((g, lo_, hi_, partner, if close { 1 } else { -1 }));
        if partner < 0 { continue; }
        let dr = if close { match b.dirc[i] { 1 => "copy", -1 => "source", _ => "uncertain" } } else { "" };
        out.broad.push(jdict! { "side" => J::s(if close { "close" } else { "divergent" }), "taxon" => J::Int(g as i64), "partner" => J::Int(partner), "overlap" => J::Int(w as i64), "mismatches" => J::Int(o as i64),
                                "expected" => J::Float(round(e, 3)), "rho" => J::Float(round(b.lo_f[i], 4)), "log10p" => J::Float(round(cx.adj10(stat / ln10), 3)),
                                "nf" => J::Int(if close { nt_ } else { ntr_ }), "nfail" => J::Int(b.nfail[i] as i64), "direction" => J::s(dr),
                                "pair_log10p" => J::Float(round((if close { b.lo_p[i] } else { b.hi_p[i] }) / ln10, 3)),
                                "cfail" => J::Int(b.cfail[i] as i64), "score" => J::Int(b.score[i] as i64), "score_partner" => J::Int(b.score_p[i] as i64),
                                "excess_focal" => J::Float(round(b.exm_f[i] as f64, 3)), "excess_partner" => J::Float(round(b.exm_p[i] as f64, 3)) });
        let sd0 = side_at(g); let b2: u8 = (if close { 4 } else { 8 }) | 2;
        if sd0 & 3 != 0 { out.note.push((g, b2 | 32, stat, partner)); continue; }
        if close && b.dirc[i] == -1 { out.note.push((g, b2, stat, partner)); continue; }
        if !(if close { cx.mask_lo } else { cx.mask_hi }) { out.note.push((g, b2, stat, partner)); continue; }
        out.mask.push((g, b2, stat, partner)); tree_cells.push((g, partner, close));
    }
    let in_tc: HashSet<usize> = tree_cells.iter().map(|x| x.0).collect();
    let mut retree: Vec<(usize, i64)> = out.dlean.iter().filter(|x| x.2 >= 0).map(|x| (x.0, x.2)).collect();
    retree.extend(out.label.iter().filter(|x| x.2 >= 0).map(|x| (x.0, x.2)));
    retree.extend(out.lean.iter().filter(|x| x.3 >= 0 && side_at(x.0) & 3 != 0 && !in_tc.contains(&x.0)).map(|x| (x.0, x.3)));
    if !tree_cells.is_empty() {
        let cells: Vec<(usize, i64)> = tree_cells.iter().map(|&(g, pt, cc)| (g, if cc { pt } else { -1 })).collect();
        out.codes = Some((tree_cells.iter().map(|x| x.0 as i32).collect(), site_codes(cx, &cx.p.core.bits.locus(li, cx.p.n()), li, &cells)));
    }
    if !tree_cells.is_empty() || !retree.is_empty() {
        let mut cells_t = vec![];
        let all: Vec<(usize, i64)> = tree_cells.iter().map(|x| (x.0, x.1)).chain(retree.iter().copied()).collect();
        for (g, extra) in all {
            let mut tips = vec![g]; for j in 0..k { let c = top[g * k + j]; if c >= 0 { tips.push(c as usize); } }
            let (pt, lc) = if cx.dense { (rep.cols[C_PARTNER].get_i(g * nloc + li), rep.cols[C_LOCUS_CLOSEST].get_i(g * nloc + li)) } else { (-1, -1) };
            for t_ in [pt, lc, extra] { if t_ >= 0 && !tips.contains(&(t_ as usize)) { tips.push(t_ as usize); } }
            if tips.len() >= 2 { cells_t.push((g, tips)); }
        }
        out.trees = build_trees(cx, &cx.p.core.bits.locus(li, cx.p.n()), li, cells_t, p_, rate_c);
        if cx.seq_policy != "none" && cx.seq_policy != "all" {
            let mut rs: HashSet<usize> = HashSet::new();
            for &(g, partner, _) in &tree_cells { rs.insert(g); for j in 0..k { let c = top[g * k + j]; if c >= 0 { rs.insert(c as usize); } } if partner >= 0 { rs.insert(partner as usize); } }
            for x in &out.label { if x.2 >= 0 { rs.insert(x.2 as usize); } }
            for x in &out.dlean { if x.2 >= 0 { rs.insert(x.2 as usize); } }
            for x in &out.lean { if x.3 >= 0 && side_at(x.0) & 3 != 0 { rs.insert(x.3 as usize); } }
            if !rs.is_empty() { let mut v: Vec<usize> = rs.into_iter().collect(); v.sort_unstable(); out.seq = Some((v.iter().map(|x| *x as i32).collect(), seq_rows(cx, li, &v))); }
        }
    }
    Some(out)
}

// ------------------------------------------------------------------ the driver
#[allow(clippy::too_many_arguments)]
pub fn run_z(a: &Args, names: &[String], loci: &Loci, p: &mut Prep, masked: &mut [u8], l: usize, k_eff: usize, m_eff: usize,
             seq_policy: &str, codes_policy: &str, uncovered: usize) -> (Rep, usize) {
    let n = names.len(); let nloc = loci.len(); let k = k_eff; let prog = crate::log::prog();
    let mask_hi = !a.mask_similar_only; let mask_lo = !a.mask_different_only;
    let from_sweep = p.bs_pre.is_some();
    let bs = if a.no_broad_sample { None } else { bs_prepare(p, k_eff) };
    if let Some(b) = &bs {
        log!("broad sample: {} per sequence per locus, drawn from each taxon's pool of {}{}", b.k, b.pn,
             if p.m_full.is_some() { String::new() } else if from_sweep { " (the pools' pair divergences from sweep 2)".to_string() }
             else { format!(" (the pools' pair divergences computed: {} pairs)", commas((n * b.pn) as i64)) });
    }
    // ---- comparators
    prog.start("comparators", nloc);
    let mut sel: Vec<Sel> = (0..nloc).into_par_iter().map(|li| { let s = choose_comparators(p, a.fit_overlap, k, li, None); prog.tick(); s }).collect();
    prog.finish();
    let mut exhausted: Vec<usize> = { let mut s: HashSet<usize> = HashSet::new(); for r in &sel { for g in 0..n { if r.exhausted[g] { s.insert(g); } } } let mut v: Vec<usize> = s.into_iter().collect(); v.sort_unstable(); v };
    if !exhausted.is_empty() {
        let t1 = std::time::Instant::now();
        let e = sweep(&p.core, Some(&p.rr), Some(&p.ok_r), Key::Mj, n, a.fit_overlap, false, false, Some(&exhausted), None);
        for (&g, row) in exhausted.iter().zip(e.rows) { p.ext.insert(g, row); }
        let pr: &Prep = p;
        let redo: Vec<Sel> = (0..nloc).into_par_iter().map(|li| choose_comparators(pr, a.fit_overlap, k, li, Some(&exhausted))).collect();
        for li in 0..nloc {
            for &g in &exhausted {
                sel[li].top[g * k..(g + 1) * k].copy_from_slice(&redo[li].top[g * k..(g + 1) * k]);
                sel[li].cnt[g] = redo[li].cnt[g]; sel[li].nres[g] = redo[li].nres[g]; sel[li].closest[g] = redo[li].closest[g]; sel[li].fork[g] = redo[li].fork[g];
            }
        }
        // A full row is every other taxon, and what is read from it from here on are the comparators just chosen
        // (their m, for the tests) and its first five entries (the report's closest taxa): the rest goes. Any other
        // pair of the taxon is computed where it is asked for (Prep::pair_m_any), the same number
        for &g in &exhausted {
            let used: HashSet<i64> = sel.iter().flat_map(|s| s.top[g * k..(g + 1) * k].iter().filter(|c| **c >= 0).map(|c| *c as i64)).collect();
            if let Some(row) = p.ext.get_mut(&g) {
                let keep: Vec<usize> = (0..row.cand.len()).filter(|&j| j < 5 || used.contains(&row.cand[j])).collect();
                row.cand = keep.iter().map(|&j| row.cand[j]).collect(); row.key = keep.iter().map(|&j| row.key[j]).collect();
                row.m = keep.iter().map(|&j| row.m[j]).collect(); row.mfl = keep.iter().map(|&j| row.mfl[j]).collect();
            }
        }
        log!("comparators: {} taxa ran out of their {} candidates at some locus: full rows computed and walked ({:.0}s)", exhausted.len(), p.c_len, t1.elapsed().as_secs_f64());
    }
    let n_exhausted = exhausted.len(); exhausted.clear();
    let p: &Prep = p;
    // ---- dispersion pre-pass
    prog.start("dispersion", nloc);
    let disp: Vec<(f64, f64, i64, i64)> = (0..nloc).into_par_iter().map(|li| { let r = locus_phi(a, p, &sel[li], k, li); prog.tick(); r }).collect();
    prog.finish();
    let phi_hat: Vec<f64> = disp.iter().map(|x| x.0).collect(); let phi_n: Vec<f64> = disp.iter().map(|x| x.1).collect();
    let ndist_l: Vec<i64> = disp.iter().map(|x| x.2).collect(); let npt_l: Vec<i64> = disp.iter().map(|x| x.3).collect();
    if ndist_l.iter().sum::<i64>() == 0 && npt_l.iter().sum::<i64>() == 0 {
        let mut reasons: Vec<(String, usize)> = vec![];
        for li in 0..nloc {
            let st = match locus_pairs(a, p, &sel[li], k, li, None) { LP::Skip(s) => s, LP::Ok(_) => "tested".to_string() };
            if let Some(e) = reasons.iter_mut().find(|x| x.0 == st) { e.1 += 1; } else { reasons.push((st, 1)); }
        }
        reasons.sort();
        let nk = p.core.keep.iter().filter(|x| **x).count() as i64; let fin = p.fin_pairs; let npairs = nk * (nk - 1) / 2; let nrated = p.rate.iter().filter(|x| x.is_finite()).count();
        let mut why = vec![format!("{} of {} taxa excluded (fewer than {} loci with at least {} residues)", p.thin.len(), n, MIN_LOCI_PER_TAXON, a.fit_overlap)];
        if nrated == 0 { why.push(format!("no locus has a rate: at no locus are {} nearest-neighbour pairs resolved beyond their resolution floor (the sequences are too similar, or share too few sites, for their divergence to show), so no pair has a divergence m", MIN_FIT_PAIRS)); }
        else if nk >= 2 { why.push(format!("{} of the {} pairs of the remaining taxa overlap in fewer than {} loci with a rate and are never compared", commas(npairs - fin), commas(npairs), MIN_LOCI_FOR_M)); }
        for (st, c) in &reasons {
            if st == "skipped-no-cells" { why.push(format!("{} loci with no sequence that has a comparator (no taxon pair shares enough sites, or the taxa are excluded)", c)); }
            else if st.starts_with("skipped-few-resolved-pairs") { why.push(format!("{} loci with fewer than {} comparator pairs resolved beyond their resolution floor (the sequences are too similar, or too short, to set a locus rate)", c, MIN_FIT_PAIRS)); }
        }
        die!("nothing could be tested: no sequence has a comparator at any locus. {}. Check --datatype, the partitions and the missing data; --fit-overlap {} sets the sites a pair must share.", why.join("; "), a.fit_overlap);
    }
    let okp: Vec<bool> = phi_n.iter().map(|x| *x > 0.0).collect();
    let (ph_ok, pn_ok): (Vec<f64>, Vec<f64>) = (0..nloc).filter(|&i| okp[i]).map(|i| (phi_hat[i], phi_n[i])).unzip();
    let phi_bar = if ph_ok.is_empty() { 0.0 } else { wmedian1(&ph_ok, &pn_ok) };
    let pct = if ph_ok.is_empty() { "n/a".to_string() } else {
        let fin: Vec<f64> = ph_ok.iter().copied().filter(|x| !x.is_nan()).collect();
        format!("[{}]", [10.0, 50.0, 90.0].iter().map(|q| crate::pyfmt::repr_f64((percentile(&fin, *q) * 1e4).round_ties_even() / 1e4)).collect::<Vec<_>>().join(", "))
    };
    log!("dispersion: dataset-wide phi {:.4} (robust SD of O/E beyond Poisson {:.2}); per-locus phi 10/50/90%: {}", phi_bar, phi_bar.sqrt(), pct);
    let d_tests: i64 = ndist_l.iter().sum(); let s_tests: i64 = npt_l.iter().sum();
    let bs_tests: i64 = if bs.is_some() { 2 * (0..n).map(|g| if p.core.keep[g] { p.focal_sites[g * nloc..(g + 1) * nloc].iter().filter(|x| **x > 0).count() as i64 } else { 0 }).sum::<i64>() } else { 0 };
    let n_tests = 2 * d_tests + s_tests + bs_tests;
    let ln10 = std::hint::black_box(10.0f64).ln();
    let logbar = (a.alpha / n_tests.max(1) as f64).ln();
    let kq = norm_quantile(1.0 - a.alpha / n_tests.max(1) as f64);
    let ladj = (n_tests.max(1) as f64).ln(); let ladj10 = ladj / ln10;
    let nfork: i64 = sel.iter().map(|s| s.fork.iter().filter(|x| **x).count() as i64).sum();
    log!("tests: {} close-relative comparisons + {} closest-match comparisons (relative arm) + {} closest-match tests (outside arm: {} sequences whose closest match at the locus is not among their comparators, {} of them with three relatives to set the expectation){} = N {} tests, alpha {}, all p values adjusted by Bonferroni correction",
         commas(d_tests), commas(d_tests), commas(s_tests), commas(nfork), commas(s_tests),
         if bs.is_some() { format!(" + {} broad-sample tests (two per present sequence: one per side)", commas(bs_tests)) } else { String::new() }, commas(n_tests), crate::pyfmt::g(a.alpha));

    // ---- the report's per-cell columns
    let dense = n * nloc <= a.dense_cells_max;
    let cells = n * nloc;
    let mut rep = Rep {
        dense, focal_sites: p.focal_sites.clone(), thin: p.thin.clone(), present: p.present.clone(),
        seq_policy: seq_policy.to_string(), uncovered_sites: uncovered, restart: p.restart.clone(), lists_extended: (p.n_starved, n_exhausted),
        n_comp: vec![0; cells], n_fail: vec![0; cells], n_similar: vec![0; cells], side: vec![0; cells], side2: Some(vec![0; cells]),
        cols: NUMCOLS.iter().map(|(_, dt, fill)| Col::filled(*dt, *fill, if dense { cells } else { 0 })).collect(),
        comparators: if dense { vec![-1; cells * k] } else { vec![] }, comp_div: if dense { vec![f32::NAN; cells * k] } else { vec![] }, comp_sim: if dense { vec![f32::NAN; cells * k] } else { vec![] },
        sparse_locus: vec![], sparse_taxon: vec![], codes: ByLocus::default(), seqs: ByLocus::default(), trees: ByLocus::default(), similar: ByLocus::default(), bs_records: ByLocus::default(),
        loc_status: vec!["tested".to_string(); nloc], phi: vec![f64::NAN; nloc], test_rate: vec![f64::NAN; nloc], pair_tests: npt_l.clone(),
        bscols: None, bs_rows: None, bs_summary: None, taxa_closest: vec![], taxa_closest_w: 0,
        rate: p.rate.clone(), plateaus: p.plateaus.clone(), kq, phi_bar, tests: [d_tests, d_tests, s_tests, bs_tests, n_tests, nfork], untested: [0; 4],
    };
    let bm_all = n <= a.best_match_all_max_taxa;
    let mut tests_fh = if a.all_tests {
        let f = std::fs::File::create(format!("{}/tests.tsv.gz", a.out)).unwrap_or_else(|e| die!("cannot write in {}: {}", a.out, crate::input::os_err(&e)));
        let mut gz = flate2::write::GzEncoder::new(std::io::BufWriter::new(f), flate2::Compression::new(4));
        let _ = gz.write_all((TESTS_COLUMNS.join("\t") + "\n").as_bytes()); Some(gz)
    } else { None };
    let every = tests_fh.is_some();
    let phi_l: Vec<f64> = (0..nloc).map(|i| (phi_n[i] * (if okp[i] { phi_hat[i] } else { 0.0 }) + PHI_N0 * phi_bar) / (phi_n[i] + PHI_N0)).collect();
    let lf = log_fact(loci.iter().map(|x| x.1.len()).max().unwrap_or(0) + 2);

    let mut mrows: Vec<Vec<String>> = vec![]; let mut tally: HashMap<&'static str, i64> = HashMap::new(); let mut mrow_lnp: HashMap<(String, String), f64> = HashMap::new();
    let mut bs_l: Vec<Option<BsBlk>> = (0..nloc).map(|_| None).collect();
    let mut sparse_at: HashMap<usize, HashSet<usize>> = HashMap::new();
    let ncmp_l: Vec<usize> = sel.iter().map(|s| s.cnt.iter().map(|c| *c as usize).sum()).collect();
    let target = 50_000usize; let mut batches: Vec<Vec<usize>> = vec![]; { let mut cur = vec![]; let mut acc = 0; for li in 0..nloc { cur.push(li); acc += ncmp_l[li]; if acc >= target { batches.push(std::mem::take(&mut cur)); acc = 0; } } if !cur.is_empty() { batches.push(cur); } }
    // A few batches at a time, the results consumed in locus order. A locus spreads its own work over the threads
    // (its pairs, its rows of the broad sample, its cells), so the batches in flight are few: each holds working
    // arrays in proportion to the taxa, and the stage's memory no longer grows with the thread count
    let group = rayon::current_num_threads().clamp(1, TEST_LOCI_IN_FLIGHT);
    prog.start_extra("test", nloc, "0 too divergent, 0 copied");
    macro_rules! mk_ctx { ($rows:expr) => { Ctx { a, names, loci, p, sel: &sel, bs: bs.as_ref(), k, m_eff, phi_l: &phi_l, logbar, ladj, ladj10, ln10, lf: &lf, rows: $rows, l, every, mask_hi, mask_lo, codes_policy, seq_policy, bm_all, dense } }; }
    for chunk in batches.chunks(group) {
        let lis: Vec<usize> = chunk.iter().flatten().copied().collect();
        let outs: Vec<LocusOut> = { let cx = mk_ctx!(&*masked); lis.par_iter().map(|&li| process_locus(&cx, li)).collect() };
        for (li, r) in lis.into_iter().zip(outs) {
            let (name, sites) = (&loci[li].0, &loci[li].1);
            if let Some(st) = r.skipped { crate::log::detail(&r.log); rep.loc_status[li] = st; prog.tick(); continue; }
            for m in r.mrows {
                for &s in sites { masked[m.g * l + s] = 45; }
                mrow_lnp.insert((name.clone(), names[m.g].clone()), m.mlp); *tally.entry(m.reason).or_insert(0) += 1; mrows.push(m.row);
            }
            if let Some(fh) = tests_fh.as_mut() { let _ = fh.write_all(r.tests.as_bytes()); }
            crate::log::detail(&r.log);
            if let Some(b) = r.rc {
                let s2 = rep.side2.as_mut().unwrap();
                for &g in &b.rows { s2[g * nloc + li] |= 1; } for &g in &b.excl { s2[g * nloc + li] |= 16; }
                bs_l[li] = Some(b);
            }
            let rb = r.rep.unwrap(); rep.phi[li] = r.phi; rep.test_rate[li] = r.rate_c;
            for (q, &g) in rb.idx.iter().enumerate() {
                let c = g * nloc + li; rep.n_comp[c] = rb.n_comp[q]; rep.n_fail[c] = rb.n_fail[q]; rep.n_similar[c] = rb.n_similar[q]; rep.side[c] = rb.side[q];
                if dense {
                    for (kk, col) in rep.cols.iter_mut().enumerate() { col.copy_from(c, &rb.cols[kk], q); }
                    rep.comparators[c * k..(c + 1) * k].copy_from_slice(&rb.comparators[q * k..(q + 1) * k]);
                    rep.comp_div[c * k..(c + 1) * k].copy_from_slice(&rb.comp_div[q * k..(q + 1) * k]); rep.comp_sim[c * k..(c + 1) * k].copy_from_slice(&rb.comp_sim[q * k..(q + 1) * k]);
                } else if rb.fail_q[q] {
                    rep.sparse_locus.push(li as i32); rep.sparse_taxon.push(g as i32); sparse_at.entry(li).or_default().insert(g);
                    for (kk, col) in rep.cols.iter_mut().enumerate() { col.push_from(&rb.cols[kk], q); }
                    rep.comparators.extend_from_slice(&rb.comparators[q * k..(q + 1) * k]);
                    rep.comp_div.extend_from_slice(&rb.comp_div[q * k..(q + 1) * k]); rep.comp_sim.extend_from_slice(&rb.comp_sim[q * k..(q + 1) * k]);
                }
            }
            if let Some(c) = rb.codes { rep.codes.set(li, c); }
            if let Some(s) = rb.seq { rep.seqs.set(li, s); }
            if let Some(t) = rb.trees { rep.trees.set(li, t); }
            if !rb.similar.is_empty() { rep.similar.set(li, rb.similar); }
            let t = |k: &str| *tally.get(k).unwrap_or(&0);
            prog.tick_k(1, Some(format!("{} too divergent, {} copied", t("divergent") + t("both"), t("similar") + t("both"))));
        }
    }
    prog.finish();
    if let Some(fh) = tests_fh.take() { if let Err(e) = fh.finish().and_then(|mut w| w.flush()) { die!("cannot write {}/tests.tsv.gz: {}", a.out, crate::input::os_err(&e)); } }

    // ---- the broad-sample verdicts
    let adj10 = |v: f64| npmin(v + ladj10, 0.0);
    if bs.is_some() && bs_l.iter().any(|b| b.is_some()) {
        let bsp = bs.as_ref().unwrap();
        let (mut n_tested, mut n_called) = (0i64, 0i64); let mut callp: HashMap<(usize, usize), f64> = HashMap::new(); let near_ln = 3.0 * ln10; let mut todo = vec![];
        let pc_of = |b: &BsBlk, i: usize| -> f64 { if b.call[i] == 1 { b.lo[i] } else if b.call[i] == -1 { b.hi[i] } else { npmin(b.lo[i], b.hi[i]) } };
        for (li, b) in bs_l.iter().enumerate() {
            let Some(b) = b else { continue }; let mut lean = false;
            for i in 0..b.rows.len() {
                if b.nt[i] as i64 >= BS_MIN { n_tested += 1; }
                let c_ = b.call[i] != 0;
                if c_ { n_called += 1; callp.insert((li, b.rows[i]), pymin(pc_of(b, i) / ln10 + ladj10, 0.0)); }
                if c_ || b.lo[i] < logbar + near_ln || (b.hi[i] < logbar + near_ln && b.ntr[i] as i64 >= BS_MIN) || (b.dm[i] && b.nt[i] as i64 >= BS_MIN) { lean = true; }
            }
            if lean { todo.push(li); }
        }
        if dense {
            let mut bc: Vec<Col> = BSCOLS.iter().map(|(_, dt, fill)| Col::filled(*dt, *fill, cells)).collect();
            for (li, b) in bs_l.iter().enumerate() {
                let Some(b) = b else { continue };
                for i in 0..b.rows.len() {
                    if (b.nt[i] as i64) < BS_MIN { continue; }
                    let c = b.rows[i] * nloc + li;
                    bc[0].set_f(c, adj10(pc_of(b, i) / ln10) as f32); bc[1].set_f(c, b.lo_f[i] as f32); bc[2].set_f(c, b.medoe[i]); bc[3].set_i(c, b.rep[i] as i64);
                    bc[4].set_f(c, adj10(b.lo[i] / ln10) as f32);
                    bc[5].set_f(c, if b.ntr[i] as i64 >= BS_MIN { adj10(b.hi[i] / ln10) as f32 } else { f32::NAN });
                    bc[6].set_i(c, b.lo_j[i] as i64); bc[7].set_i(c, npmax(b.lo_o[i], 0.0) as i64); bc[8].set_f(c, (b.lo_e[i] * b.lo_f[i]) as f32);
                    bc[9].set_i(c, b.ntr[i].max(0) as i64); bc[10].set_i(c, b.nfail[i].max(0) as i64); bc[11].set_f(c, b.obs_pct[i]); bc[12].set_f(c, b.exp_pct[i]);
                }
            }
            rep.bscols = Some(bc);
        }
        let mut mrow_at: HashMap<(String, String), usize> = mrows.iter().enumerate().map(|(j, r)| ((r[0].clone(), r[1].clone()), j)).collect();
        let mut bs_rows_rep: Vec<(usize, usize, f64, f64, i64, i8)> = vec![]; let mut tally_bs: HashMap<&str, i64> = HashMap::new();
        if !todo.is_empty() {
            prog.start("broad sample", todo.len());
            let outs: Vec<Option<BsOut>> = { let cx = mk_ctx!(&*masked); let rp = &rep; todo.par_iter().map(|&li| { let r = bs_locus_b(&cx, rp, bs_l[li].as_ref().unwrap(), li, near_ln); prog.tick(); r }).collect() };
            for (&li, out) in todo.iter().zip(outs) {
                let Some(out) = out else { continue };
                let (name, sites) = (&loci[li].0, &loci[li].1);
                let var_lp10 = |g: usize| -> f64 { *callp.get(&(li, g)).unwrap_or(&f64::NAN) };
                for &(g, sp_i, ep_i, partner, call_i) in &out.lean { bs_rows_rep.push((li, g, sp_i, ep_i, partner, call_i)); }
                for &(g, b2, _lp, _partner) in &out.note {
                    let c = g * nloc + li; rep.side2.as_mut().unwrap()[c] |= b2; *tally_bs.entry("noted").or_insert(0) += 1;
                    let key = (name.clone(), names[g].clone());
                    if b2 & 8 != 0 && b2 & 32 != 0 && mask_hi && rep.side[c] & 3 == 2 {
                        rep.side[c] |= 1;
                        if let Some(&j) = mrow_at.get(&key) { if mrows[j][2] == "similar" { *tally.entry("similar").or_insert(0) -= 1; *tally.entry("both").or_insert(0) += 1; mrows[j][2] = "both".into(); } }
                    }
                    if let Some(&j) = mrow_at.get(&key) {
                        let v10 = var_lp10(g); let cur_ln = *mrow_lnp.get(&key).unwrap_or(&f64::INFINITY) + ladj;
                        if v10.is_finite() && v10 * ln10 < cur_ln - 1e-9 { mrows[j][3] = "broad_sample".into(); mrows[j][4] = pfmt10(v10); mrow_lnp.insert(key, v10 * ln10 - ladj); }
                    }
                }
                for &(g, b2, _lp, partner) in &out.mask {
                    let c = g * nloc + li; rep.side2.as_mut().unwrap()[c] |= b2; rep.side[c] |= if b2 & 4 != 0 { 2 } else { 1 };
                    for &s in sites { masked[g * l + s] = 45; }
                    let reason = if b2 & 4 != 0 { "similar" } else { "divergent" };
                    *tally.entry(reason).or_insert(0) += 1; *tally_bs.entry(if b2 & 4 != 0 { "close" } else { "divergent" }).or_insert(0) += 1;
                    let mut row_ = vec![name.clone(), names[g].clone(), reason.to_string(), "broad_sample".to_string(), pfmt10(var_lp10(g)), p.focal_sites[c].to_string()];
                    row_.extend(vec![String::new(); 6 + 9 + BS_BLANK + 1]);
                    mrows.push(row_); mrow_at.insert((name.clone(), names[g].clone()), mrows.len() - 1);
                    if dense { rep.cols[C_PARTNER].set_i(c, partner); }
                    else if !sparse_at.get(&li).map_or(false, |s| s.contains(&g)) {
                        rep.sparse_locus.push(li as i32); rep.sparse_taxon.push(g as i32);
                        for (kk, col) in rep.cols.iter_mut().enumerate() { col.push_val(if kk == C_PARTNER { partner as f64 } else { NUMCOLS[kk].2 }); }
                        rep.comparators.extend_from_slice(&sel[li].top[g * k..(g + 1) * k]);
                        rep.comp_div.extend(std::iter::repeat(f32::NAN).take(k)); rep.comp_sim.extend(std::iter::repeat(f32::NAN).take(k));
                        sparse_at.entry(li).or_default().insert(g);
                    }
                }
                for &(g, dp_i, rp_) in &out.dlean { bs_rows_rep.push((li, g, dp_i, f64::NAN, rp_, 0)); }
                for &(g, dp_i, partner) in &out.label {
                    rep.side2.as_mut().unwrap()[g * nloc + li] |= 64; bs_rows_rep.push((li, g, dp_i, f64::NAN, partner, 1));
                    if mrow_at.contains_key(&(name.clone(), names[g].clone())) { *tally_bs.entry("labels").or_insert(0) += 1; }
                }
                if !out.broad.is_empty() { match rep.bs_records.get_mut(li) { Some(v) => v.extend(out.broad), None => rep.bs_records.set(li, out.broad) } }
                let ls = sites.len();
                if let Some((qs_, blk_)) = out.codes {
                    match rep.codes.get_mut(li) {
                        Some((c0, b0)) => {
                            let newset: HashSet<i32> = qs_.iter().copied().collect(); let (mut c1, mut b1) = (vec![], vec![]);
                            for (i, &c) in c0.iter().enumerate() { if !newset.contains(&c) { c1.push(c); b1.extend_from_slice(&b0[i * ls..(i + 1) * ls]); } }
                            c1.extend(qs_); b1.extend(blk_); *c0 = c1; *b0 = b1;
                        }
                        None => rep.codes.set(li, (qs_, blk_)),
                    }
                }
                if let Some((rows_s, blk_s)) = out.seq {
                    match rep.seqs.get_mut(li) {
                        Some((r0, b0)) => {
                            let have: HashSet<i32> = r0.iter().copied().collect();
                            let mut items: Vec<(i32, Vec<u8>)> = r0.iter().enumerate().map(|(i, &r)| (r, b0[i * ls..(i + 1) * ls].to_vec())).collect();
                            for (i, &r) in rows_s.iter().enumerate() { if !have.contains(&r) { items.push((r, blk_s[i * ls..(i + 1) * ls].to_vec())); } }
                            items.sort_by_key(|x| x.0);
                            *r0 = items.iter().map(|x| x.0).collect(); *b0 = items.into_iter().flat_map(|x| x.1).collect();
                        }
                        None => rep.seqs.set(li, (rows_s, blk_s)),
                    }
                }
                if let Some(t_) = out.trees {
                    match rep.trees.get_mut(li) {
                        Some(t0) => {
                            let new_: HashSet<i64> = t_.cells.iter().copied().collect(); let mut comb = Trees::default();
                            for j in 0..t0.cells.len() { if !new_.contains(&t0.cells[j]) { comb.push(t0, j); } }
                            for j in 0..t_.cells.len() { comb.push(&t_, j); }
                            let mut order: Vec<usize> = (0..comb.cells.len()).collect(); order.sort_by_key(|&j| comb.cells[j]);
                            let mut fin = Trees::default(); for j in order { fin.push(&comb, j); }
                            *t0 = fin;
                        }
                        None => rep.trees.set(li, t_),
                    }
                }
            }
            prog.finish();
        }
        if !bs_rows_rep.is_empty() {
            bs_rows_rep.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
            let f = |v: f64| adj10(v / ln10) as f32;
            rep.bs_rows = Some(BsRows {
                locus: bs_rows_rep.iter().map(|r| r.0 as i32).collect(), taxon: bs_rows_rep.iter().map(|r| r.1 as i32).collect(),
                var_log10p: bs_rows_rep.iter().map(|r| f(if r.5 == 1 { r.2 } else if r.5 == -1 { r.3 } else { pymin(r.2, if r.3.is_finite() { r.3 } else { r.2 }) })).collect(),
                lo_log10p: bs_rows_rep.iter().map(|r| f(r.2)).collect(), hi_log10p: bs_rows_rep.iter().map(|r| f(r.3)).collect(),
                partner: bs_rows_rep.iter().map(|r| r.4 as i32).collect(), call: bs_rows_rep.iter().map(|r| r.5).collect() });
        }
        let tb = |k: &str| *tally_bs.get(k).unwrap_or(&0);
        rep.bs_summary = Some(jdict! { "tested" => J::Int(n_tested), "bar_passed" => J::Int(n_called), "scale_sd" => J::Float(0.0), "per_cell" => J::Int(bsp.k as i64), "pool" => J::Int(bsp.pn as i64),
                                       "family" => J::Int(bs_tests), "close" => J::Int(tb("close")), "divergent" => J::Int(tb("divergent")), "noted" => J::Int(tb("noted")), "divergent_labels" => J::Int(tb("labels")) });
        log!("broad sample v2: {} sequences tested, {} called; new masks: {} too close + {} too divergent; {} fired on cells already masked, {} divergent cells labelled also-too-close",
             commas(n_tested), commas(n_called), tb("close"), tb("divergent"), tb("noted"), tb("labels"));
    }

    // ---- the report's remaining pieces
    let w5 = 5.min(n); let mut cl = vec![-1i32; n * w5];
    for g in 0..n {
        let cand = p.row_cands(g);
        let lim = if p.ext.contains_key(&g) { 5.min(cand.len()) } else { 5.min(p.c_len).min(cand.len()) };
        for j in 0..lim.min(w5) { cl[g * w5 + j] = cand[j] as i32; }
    }
    rep.taxa_closest = cl; rep.taxa_closest_w = w5;
    let mut lidx: HashMap<&str, usize> = HashMap::new(); for (i, (nm, _)) in loci.iter().enumerate() { lidx.insert(nm, i); }
    let mut tidx: HashMap<&str, usize> = HashMap::new(); for (i, nm) in names.iter().enumerate() { tidx.insert(nm, i); }
    if bs.is_some() {
        let mut rowpos: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
        for row_ in mrows.iter_mut() {
            let (li, g) = (lidx[row_[0].as_str()], tidx[row_[1].as_str()]);
            let Some(b) = bs_l[li].as_ref() else { continue };
            let rp = rowpos.entry(li).or_insert_with(|| b.rows.iter().enumerate().map(|(kk, r)| (*r, kk)).collect());
            let Some(&kk) = rp.get(&g) else { continue };
            if (b.nt[kk] as i64) < BS_MIN { continue; }
            let s2v = rep.side2.as_ref().unwrap()[g * nloc + li];
            let side_ = if s2v & 8 != 0 { "divergent" } else if s2v & (4 | 64) != 0 { "close" } else { "" };
            let role_ = if s2v & 4 != 0 { match b.dirc[kk] { 1 => "copy", -1 => "source", _ => "uncertain" } } else { "" };
            let jl = b.lo_j[kk]; let red_ok = b.ntr[kk] as i64 >= BS_MIN; let e = String::new;
            let vals = [side_.to_string(), role_.to_string(), if jl >= 0 { names[jl as usize].clone() } else { e() }, if jl >= 0 { (b.lo_o[kk] as i64).to_string() } else { e() },
                        if jl >= 0 { fnum(b.lo_e[kk] * b.lo_f[kk], 1) } else { e() }, if jl >= 0 { fnum(b.lo_f[kk], 3) } else { e() }, if jl >= 0 { pfmt(b.lo[kk] + ladj) } else { e() },
                        if red_ok { b.nfail[kk].to_string() } else { e() }, if red_ok { b.ntr[kk].to_string() } else { e() },
                        if red_ok { fnum(b.obs_pct[kk] as f64, 2) } else { e() }, if red_ok { fnum(b.exp_pct[kk] as f64, 2) } else { e() }, if red_ok { pfmt(b.hi[kk] + ladj) } else { e() }];
            for (i, v) in vals.into_iter().enumerate() { row_[21 + i] = v; }
        }
    }
    mrows.sort_by_key(|r| (lidx[r[0].as_str()], tidx[r[1].as_str()]));
    {
        let mut txt = crate::pyfmt::csv_row(&MASKED_COLUMNS.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        for (i, r) in mrows.iter().enumerate() { let mut f = vec![(i + 1).to_string()]; f.extend(r.iter().cloned()); txt.push_str(&crate::pyfmt::csv_row(&f)); }
        if let Err(e) = std::fs::write(format!("{}/masked.csv", a.out), txt) { die!("cannot write {}/masked.csv: {}", a.out, crate::input::os_err(&e)); }
    }
    let skipped: Vec<(usize, &String)> = rep.loc_status.iter().enumerate().filter(|(_, st)| *st != "tested").collect();
    if !skipped.is_empty() {
        log!("{} loci not tested: {}{}", skipped.len(), skipped.iter().take(20).map(|(i, st)| format!("{} ({})", loci[*i].0, st)).collect::<Vec<_>>().join("; "),
             if skipped.len() > 20 { format!("; and {} more", skipped.len() - 20) } else { String::new() });
    }
    let mut thin_t = vec![false; n]; for &t in &rep.thin { thin_t[t] = true; }
    let skipped_l: Vec<bool> = rep.loc_status.iter().map(|s| s != "tested").collect();
    let (mut unt, mut u_thin, mut u_loc, mut present_c) = (0i64, 0i64, 0i64, 0i64);
    for g in 0..n { for li in 0..nloc {
        let c = g * nloc + li; if rep.focal_sites[c] == 0 { continue; } present_c += 1;
        if rep.n_comp[c] > 0 { continue; } unt += 1;
        if thin_t[g] { u_thin += 1; } else if skipped_l[li] { u_loc += 1; }
    } }
    let u_comp = unt - u_thin - u_loc; rep.untested = [unt, u_thin, u_loc, u_comp];
    let all_skipped: String = rep.loc_status.iter().enumerate().filter(|(_, st)| *st != "tested").map(|(i, st)| format!("{} ({})", loci[i].0, st)).collect::<Vec<_>>().join(", ").chars().take(300).collect();
    log!("not tested: {} of {} taxa (excluded: fewer than {} loci with at least {} residues{}); {} of {} loci ({}); {} of {} present sequences ({} of excluded taxa, {} at untested loci, {} with no comparator: no resolved relative sharing enough sites)",
         rep.thin.len(), n, MIN_LOCI_PER_TAXON, a.fit_overlap,
         if rep.thin.is_empty() { String::new() } else { format!(": {}{}", rep.thin.iter().take(10).map(|&i| names[i].as_str()).collect::<Vec<_>>().join(", "), if rep.thin.len() > 10 { " ..." } else { "" }) },
         skipped_l.iter().filter(|x| **x).count(), nloc, if all_skipped.is_empty() { "none".to_string() } else { all_skipped }, commas(unt), commas(present_c), commas(u_thin), commas(u_loc), commas(u_comp));
    let is_fail = |e: &J| matches!(e.get("fail"), Some(J::Bool(true)));
    let npair: i64 = rep.similar.0.iter().map(|(_, v)| v.iter().filter(|e| is_fail(e)).count() as i64).sum();
    let ncl: i64 = rep.similar.0.iter().map(|(_, v)| v.iter().filter(|e| is_fail(e) && matches!(e.get("test"), Some(J::Str(s)) if s == "closest")).count() as i64).sum();
    log!("too-similar pairs: {} ({} against a comparator, {} against the closest match; {} closest-match tests run)", commas(npair), commas(npair - ncl), commas(ncl), commas(s_tests));
    let nunc = rep.side.iter().filter(|s| **s & 8 != 0).count();
    let t = |k: &str| *tally.get(k).unwrap_or(&0);
    log!("masked {} sequences: {} too divergent, {} too similar (copied from another taxon{}), {} both", mrows.len(), t("divergent"), t("similar"),
         if nunc != 0 { format!("; {} of a pair whose direction is uncertain, both members masked", nunc) } else { String::new() }, t("both"));
    let nm = mrows.len();
    (rep, nm)
}
