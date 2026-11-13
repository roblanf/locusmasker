//! Everything the test needs before it runs: the alignment as residue codes, the locus plateaus and rates, and the
//! pair structures (each taxon's candidate list with m, its floor and the ranking key; the closest match at every
//! locus; the classes of identical genomes; the full m up to FULL_M_MAX_TAXA).
//!
//! locusmasker.py walks pairs of taxon blocks with matrix products and reduces block results; its reductions are
//! order-independent by construction, so this port walks the pair matrix directly with the arithmetic of its
//! compiled pair kernel (_pair_medians_py, _closest_rows_py): each pair of taxa once, serving both of its rows
//! (sweep_pairs_once), or a few given rows against every column (sweep_rows).
use crate::input::Loci;
use crate::npmath::*;
use crate::pyfmt::commas;
use crate::{log, Args};
use rayon::prelude::*;
use std::collections::HashMap;

pub const MIN_LOCI_FOR_M: usize = 5;
pub const MIN_LOCI_PER_TAXON: i64 = 10;
pub const MIN_FIT_PAIRS: usize = 8;
pub const BS_PER_COMP: usize = 10;
pub const BS_POOL: usize = 100;
pub const BS_MIN: i64 = 5;
pub const BS_SEED: u64 = 0x5eed1517;
pub const SWEEP_BLOCK_MIN: usize = 1024;
pub const CANDIDATES: usize = 200;
pub const FULL_M_MAX_TAXA: usize = 20000;
pub const BAND_SLOTS: usize = 4;
const SWEEP_CHUNK_BYTES: usize = 512 << 10;       // the planes of a chunk of rows: about a core's share of the cache

/// the residues by locus as bit planes (each taxon's row holds the loci one after another) and, up to
/// SWEEP_BLOCK_MIN taxa, the counts of every pair at every locus. The planes are the only copy of the residue
/// codes: a sequence's codes are read back from them where the tests look at single sites (LocusBits::seq)
pub struct Core {
    pub n: usize, pub nloc: usize,
    pub off: Vec<usize>, pub keep: Vec<bool>,
    pub slab: Option<Slab>, pub bits: Bits,
}

/// the residues as bit planes, for counting: per taxon a presence plane and the planes of the residue code's bits, each
/// locus starting at a word. Two sequences share a site where both presence bits are set, and agree there where every
/// code plane agrees: two popcounts per 64 sites, the same integers as counting site by site
pub struct Bits { pub planes: usize, pub wpr: usize, pub woff: Vec<usize>, pub data: Vec<u64>, popcnt: bool, avx2: bool }
impl Bits {
    /// the layout: the planes of a code, each locus's first word, the words of a row
    fn layout(off: &[usize], a_: usize) -> (usize, Vec<usize>, usize) {
        let planes = if a_ <= 4 { 2 } else { 5 }; let nloc = off.len() - 1;
        let mut woff = Vec::with_capacity(nloc + 1); let mut acc = 0; for li in 0..nloc { woff.push(acc); acc += (off[li + 1] - off[li]).div_ceil(64); } woff.push(acc);
        (planes, woff, acc)
    }
    fn finish(planes: usize, wpr: usize, woff: Vec<usize>, data: Vec<u64>) -> Bits {
        #[cfg(target_arch = "x86_64")]
        let (popcnt, avx2) = (std::is_x86_feature_detected!("popcnt"), std::is_x86_feature_detected!("popcnt") && std::is_x86_feature_detected!("avx2") && std::env::var_os("LOCUSMASKER_NO_AVX2").is_none());
        #[cfg(not(target_arch = "x86_64"))]
        let (popcnt, avx2) = (false, false);
        Bits { planes, wpr, woff, data, popcnt, avx2 }
    }
    /// The planes straight from the alignment's letters (rows: n x l bytes; lut: a letter's code or -1; loci: each
    /// locus's sites), and how often each code occurs at each locus over all taxa (nloc x a_), for the plateaus
    fn from_rows(n: usize, l: usize, rows: &[u8], lut: &[i8; 256], loci: &Loci, off: &[usize], a_: usize) -> (Bits, Vec<u64>) {
        let (planes, woff, wpr) = Bits::layout(off, a_); let nloc = loci.len();
        let stride = (planes + 1) * wpr; let mut data = vec![0u64; n * stride];
        let counts = if stride > 0 {
            data.par_chunks_mut(stride).enumerate().fold(|| vec![0u64; nloc * a_], |mut cnt, (g, row)| {
                let src = &rows[g * l..(g + 1) * l];
                for (li, (_, st)) in loci.iter().enumerate() {
                    for (k, &site) in st.iter().enumerate() {
                        let v = lut[src[site] as usize];
                        if v < 0 { continue; }
                        cnt[li * a_ + v as usize] += 1;
                        let (w, b) = (woff[li] + k / 64, k % 64);
                        row[w] |= 1u64 << b;
                        for p in 0..planes { if (v as u8 >> p) & 1 == 1 { row[(p + 1) * wpr + w] |= 1u64 << b; } }
                    }
                }
                cnt
            }).reduce(|| vec![0u64; nloc * a_], |mut x, y| { for (p, q) in x.iter_mut().zip(&y) { *p += q; } x })
        } else { vec![0u64; nloc * a_] };
        (Bits::finish(planes, wpr, woff, data), counts)
    }
    /// the planes from residue codes laid out by locus (xl: n x lt)
    #[cfg(test)]
    fn build(n: usize, xl: &[i8], lt: usize, off: &[usize], a_: usize) -> Bits {
        let (planes, woff, wpr) = Bits::layout(off, a_); let nloc = off.len() - 1;
        let stride = (planes + 1) * wpr; let mut data = vec![0u64; n * stride];
        if stride > 0 {
            data.par_chunks_mut(stride).enumerate().for_each(|(g, row)| {
                let x = &xl[g * lt..(g + 1) * lt];
                for li in 0..nloc {
                    for (k, &v) in x[off[li]..off[li + 1]].iter().enumerate() {
                        if v < 0 { continue; }
                        let (w, b) = (woff[li] + k / 64, k % 64);
                        row[w] |= 1u64 << b;
                        for p in 0..planes { if (v as u8 >> p) & 1 == 1 { row[(p + 1) * wpr + w] |= 1u64 << b; } }
                    }
                }
            });
        }
        Bits::finish(planes, wpr, woff, data)
    }
    /// a taxon's residue codes at a locus, read back from the planes (-1: no residue)
    #[cfg(test)]
    pub fn seq(&self, g: usize, li: usize, len: usize) -> Vec<i8> {
        let stride = (self.planes + 1) * self.wpr; let row = &self.data[g * stride..(g + 1) * stride]; let w0 = self.woff[li];
        (0..len).map(|k| {
            let (w, b) = (w0 + k / 64, k % 64);
            if (row[w] >> b) & 1 == 0 { return -1; }
            let mut v = 0u8; for p in 0..self.planes { v |= (((row[(p + 1) * self.wpr + w] >> b) & 1) as u8) << p; }
            v as i8
        }).collect()
    }
    /// the residues a taxon has at a locus
    pub fn present(&self, g: usize, li: usize) -> u32 {
        let stride = (self.planes + 1) * self.wpr; let row = &self.data[g * stride..(g + 1) * stride];
        row[self.woff[li]..self.woff[li + 1]].iter().map(|w| w.count_ones()).sum()
    }
    #[inline(always)]
    fn count_impl(&self, g: usize, c: usize, li: usize) -> (u32, u32) {
        let (w0, w1) = (self.woff[li], self.woff[li + 1]); let stride = (self.planes + 1) * self.wpr;
        let (rg, rc) = (&self.data[g * stride..(g + 1) * stride], &self.data[c * stride..(c + 1) * stride]);
        let (mut sh, mut sm) = (0u32, 0u32);
        for w in w0..w1 {
            let pa = rg[w] & rc[w]; let mut eq = pa;
            for p in 1..=self.planes { eq &= !(rg[p * self.wpr + w] ^ rc[p * self.wpr + w]); }
            sh += pa.count_ones(); sm += eq.count_ones();
        }
        (sh, sm)
    }
    #[inline(always)]
    fn all_impl(&self, g: usize, c: usize, w: &mut [u32], s: &mut [u32]) { for li in 0..w.len() { let (a, b) = self.count_impl(g, c, li); w[li] = a; s[li] = b; } }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "popcnt")]
    unsafe fn count_popcnt(&self, g: usize, c: usize, li: usize) -> (u32, u32) { self.count_impl(g, c, li) }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "popcnt")]
    unsafe fn all_popcnt(&self, g: usize, c: usize, w: &mut [u32], s: &mut [u32]) { self.all_impl(g, c, w, s) }
    /// every locus at once: the two masks of the whole row four words at a time, then the popcounts per locus
    /// (pa, eq: scratch of wpr words). The same integers as count_impl
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "popcnt,avx2")]
    unsafe fn all_avx2(&self, g: usize, c: usize, w: &mut [u32], s: &mut [u32], pa: &mut [u64], eq: &mut [u64]) {
        use std::arch::x86_64::*;
        let wpr = self.wpr; let stride = (self.planes + 1) * wpr; debug_assert!(pa.len() >= wpr && eq.len() >= wpr);
        let (rg, rc) = (self.data.as_ptr().add(g * stride), self.data.as_ptr().add(c * stride)); let mut x = 0;
        while x + 4 <= wpr {
            let p = _mm256_and_si256(_mm256_loadu_si256(rg.add(x) as *const __m256i), _mm256_loadu_si256(rc.add(x) as *const __m256i));
            let mut d = _mm256_setzero_si256();
            for pl in 1..=self.planes { d = _mm256_or_si256(d, _mm256_xor_si256(_mm256_loadu_si256(rg.add(pl * wpr + x) as *const __m256i), _mm256_loadu_si256(rc.add(pl * wpr + x) as *const __m256i))); }
            _mm256_storeu_si256(pa.as_mut_ptr().add(x) as *mut __m256i, p); _mm256_storeu_si256(eq.as_mut_ptr().add(x) as *mut __m256i, _mm256_andnot_si256(d, p)); x += 4;
        }
        while x < wpr { let p = *rg.add(x) & *rc.add(x); let mut d = 0u64; for pl in 1..=self.planes { d |= *rg.add(pl * wpr + x) ^ *rc.add(pl * wpr + x); } pa[x] = p; eq[x] = p & !d; x += 1; }
        for li in 0..w.len() {
            let (mut sh, mut sm) = (0u32, 0u32);
            for x in self.woff[li]..self.woff[li + 1] { sh += pa[x].count_ones(); sm += eq[x].count_ones(); }
            w[li] = sh; s[li] = sm;
        }
    }
    /// shared sites and same residues of two taxa at a locus
    #[inline] pub fn count(&self, g: usize, c: usize, li: usize) -> (u32, u32) {
        #[cfg(target_arch = "x86_64")]
        if self.popcnt { return unsafe { self.count_popcnt(g, c, li) }; }
        self.count_impl(g, c, li)
    }
    /// the same at every locus (pa, eq: scratch of wpr words each)
    #[inline] pub fn all(&self, g: usize, c: usize, w: &mut [u32], s: &mut [u32], pa: &mut [u64], eq: &mut [u64]) {
        #[cfg(target_arch = "x86_64")]
        {
            if self.avx2 { return unsafe { self.all_avx2(g, c, w, s, pa, eq) }; }
            if self.popcnt { return unsafe { self.all_popcnt(g, c, w, s) }; }
        }
        let _ = (pa, eq);
        self.all_impl(g, c, w, s)
    }
    /// bytes of one taxon's planes
    pub fn row_bytes(&self) -> usize { (self.planes + 1) * self.wpr * 8 }
    pub fn has_avx2(&self) -> bool { self.avx2 }
}
pub struct Slab { pub w: Vec<u32>, pub s: Vec<u32> }                // [tri(g, c) * nloc + li], g <= c

/// One locus's planes of every taxon, side by side (row g: the presence words, then each code plane's). The tests of
/// a locus read every taxon's residues there many times over (a failing cell against every taxon, a sequence
/// against its broad sample), and in the planes of the whole alignment those are a few words of every row, the rows
/// tens of kilobytes apart: gathered, a locus is a few megabytes that stay in the cache
pub struct LocusBits { wl: usize, planes: usize, data: Vec<u64>, popcnt: bool }
impl Bits {
    pub fn locus(&self, li: usize, n: usize) -> LocusBits {
        let (w0, wl) = (self.woff[li], self.woff[li + 1] - self.woff[li]); let stride = (self.planes + 1) * self.wpr; let ls = (self.planes + 1) * wl;
        let mut data = vec![0u64; n * ls];
        if ls > 0 {
            data.par_chunks_mut(ls).enumerate().with_min_len(256).for_each(|(g, row)| {
                let src = &self.data[g * stride..(g + 1) * stride];
                for p in 0..=self.planes { row[p * wl..(p + 1) * wl].copy_from_slice(&src[p * self.wpr + w0..p * self.wpr + w0 + wl]); }
            });
        }
        LocusBits { wl, planes: self.planes, data, popcnt: self.popcnt }
    }
}
impl LocusBits {
    #[inline(always)]
    fn count_impl(&self, g: usize, c: usize) -> (u32, u32) {
        let ls = (self.planes + 1) * self.wl; let (rg, rc) = (&self.data[g * ls..(g + 1) * ls], &self.data[c * ls..(c + 1) * ls]);
        let (mut sh, mut sm) = (0u32, 0u32);
        for w in 0..self.wl {
            let pa = rg[w] & rc[w]; let mut eq = pa;
            for p in 1..=self.planes { eq &= !(rg[p * self.wl + w] ^ rc[p * self.wl + w]); }
            sh += pa.count_ones(); sm += eq.count_ones();
        }
        (sh, sm)
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "popcnt")]
    unsafe fn count_popcnt(&self, g: usize, c: usize) -> (u32, u32) { self.count_impl(g, c) }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "popcnt")]
    unsafe fn block_popcnt(&self, gs: &[usize], keep: &[bool], n: usize, w: &mut [u32], s: &mut [u32]) { self.block_impl(gs, keep, n, w, s) }
    #[inline(always)]
    fn block_impl(&self, gs: &[usize], keep: &[bool], n: usize, w: &mut [u32], s: &mut [u32]) {
        for c in 0..n {
            for (bi, &g) in gs.iter().enumerate() { let (a, b) = if keep[g] && keep[c] { self.count_impl(g, c) } else { (0, 0) }; w[bi * n + c] = a; s[bi * n + c] = b; }
        }
    }
    #[inline(always)]
    fn three_impl(&self, f: usize, c: usize, t: usize) -> (u32, u32, u32) {
        let ls = (self.planes + 1) * self.wl; let wl = self.wl;
        let (rf, rc, rt) = (&self.data[f * ls..(f + 1) * ls], &self.data[c * ls..(c + 1) * ls], &self.data[t * ls..(t + 1) * ls]);
        let (mut common, mut dt, mut dc) = (0u32, 0u32, 0u32);
        for w in 0..wl {
            let m = rf[w] & rc[w] & rt[w]; let (mut xt, mut xc) = (0u64, 0u64);
            for p in 1..=self.planes { xt |= rf[p * wl + w] ^ rt[p * wl + w]; xc |= rf[p * wl + w] ^ rc[p * wl + w]; }
            common += m.count_ones(); dt += (m & xt).count_ones(); dc += (m & xc).count_ones();
        }
        (common, dt, dc)
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "popcnt")]
    unsafe fn three_popcnt(&self, f: usize, c: usize, t: usize) -> (u32, u32, u32) { self.three_impl(f, c, t) }
    /// Three taxa at the locus, on the sites all three have: how many there are, at how many t differs from f, and
    /// at how many c differs from f (the best match compares a challenger and the champion with the focal sequence
    /// on the same sites)
    #[inline] pub fn three(&self, f: usize, c: usize, t: usize) -> (u32, u32, u32) {
        #[cfg(target_arch = "x86_64")]
        if self.popcnt { return unsafe { self.three_popcnt(f, c, t) }; }
        self.three_impl(f, c, t)
    }
    /// shared sites and same residues of two taxa at the locus: the integers of Bits::count
    #[inline] pub fn count(&self, g: usize, c: usize) -> (u32, u32) {
        #[cfg(target_arch = "x86_64")]
        if self.popcnt { return unsafe { self.count_popcnt(g, c) }; }
        self.count_impl(g, c)
    }
    /// A few taxa against every taxon: shared sites and same residues (0 of each where either is excluded), w and s
    /// as gs.len() rows of n. The taxa of the locus are the outer loop, so each one's planes are read once for the
    /// whole block (one taxon at a time would read the locus through once per taxon)
    pub fn block(&self, gs: &[usize], keep: &[bool], n: usize, w: &mut [u32], s: &mut [u32]) {
        #[cfg(target_arch = "x86_64")]
        if self.popcnt { return unsafe { self.block_popcnt(gs, keep, n, w, s) }; }
        self.block_impl(gs, keep, n, w, s)
    }
    /// a taxon's residue codes at the locus (-1: no residue)
    pub fn seq(&self, g: usize, len: usize) -> Vec<i8> {
        let ls = (self.planes + 1) * self.wl; let row = &self.data[g * ls..(g + 1) * ls];
        (0..len).map(|k| {
            let (w, b) = (k / 64, k % 64);
            if (row[w] >> b) & 1 == 0 { return -1; }
            let mut v = 0u8; for p in 0..self.planes { v |= (((row[(p + 1) * self.wl + w] >> b) & 1) as u8) << p; }
            v as i8
        }).collect()
    }
}

impl Core {
    #[inline] pub fn tri(&self, g: usize, c: usize) -> usize { let (a, b) = if g <= c { (g, c) } else { (c, g) }; a * self.n - a * a.saturating_sub(1) / 2 + (b - a) }
    /// shared sites and same residues of a pair at a locus; a pair with an excluded member counts 0 of each
    #[inline] pub fn at(&self, li: usize, g: usize, c: usize) -> (u32, u32) {
        if let Some(sl) = &self.slab { let k = self.tri(g, c) * self.nloc + li; return (sl.w[k], sl.s[k]); }
        if !(self.keep[g] && self.keep[c]) { return (0, 0); }
        self.bits.count(g, c, li)
    }
    /// the same from a locus's gathered planes
    #[inline] pub fn at_l(&self, lb: &LocusBits, li: usize, g: usize, c: usize) -> (u32, u32) {
        if let Some(sl) = &self.slab { let k = self.tri(g, c) * self.nloc + li; return (sl.w[k], sl.s[k]); }
        if !(self.keep[g] && self.keep[c]) { return (0, 0); }
        lb.count(g, c)
    }
    /// the counts of a pair at every locus
    fn pair_counts(&self, g: usize, c: usize, w: &mut [u32], s: &mut [u32], pa: &mut [u64], eq: &mut [u64]) {
        if let Some(sl) = &self.slab { let k = self.tri(g, c) * self.nloc; w.copy_from_slice(&sl.w[k..k + self.nloc]); s.copy_from_slice(&sl.s[k..k + self.nloc]); return; }
        if !(self.keep[g] && self.keep[c]) { w.fill(0); s.fill(0); return; }
        self.bits.all(g, c, w, s, pa, eq);
    }
    fn build_slab(&mut self) {
        let n = self.n; let nloc = self.nloc; let npair = n * (n + 1) / 2;
        let mut w = vec![0u32; npair * nloc]; let mut s = vec![0u32; npair * nloc];
        // row g owns the pairs (g, g..n): a contiguous run of the triangle
        let this: &Core = self;
        let mut wrest: &mut [u32] = &mut w; let mut srest: &mut [u32] = &mut s; let mut parts = Vec::with_capacity(n);
        for g in 0..n { let len = (n - g) * nloc; let (a, b) = wrest.split_at_mut(len); let (c, d) = srest.split_at_mut(len); parts.push((g, a, c)); wrest = b; srest = d; }
        let wpr = this.bits.wpr;
        parts.into_par_iter().for_each_init(|| (vec![0u64; wpr], vec![0u64; wpr]), |(pa, eq), (g, wg, sg)| {
            for c in g..n {
                let k = (c - g) * nloc;
                if !(this.keep[g] && this.keep[c]) { continue; }
                this.bits.all(g, c, &mut wg[k..k + nloc], &mut sg[k..k + nloc], pa, eq);
            }
        });
        self.slab = Some(Slab { w, s });
    }
}

#[derive(Clone, Default)]
pub struct RowList { pub cand: Vec<i64>, pub key: Vec<f64>, pub m: Vec<f64>, pub mfl: Vec<f64>, pub nfin: i64 }

pub struct Prep {
    pub core: Core,
    pub thin: Vec<usize>, pub present: Vec<i64>, pub focal_sites: Vec<u32>,       // focal_sites: n x nloc
    pub plateaus: Vec<f64>, pub rate: Vec<f64>, pub rr: Vec<f64>, pub ok_r: Vec<bool>,
    pub cls: Vec<usize>, pub has_dup: bool,
    pub lists: Vec<RowList>, pub c_len: usize,
    pub cl_win: Vec<i64>, pub cl_dmin: Vec<f64>,                                  // n x nloc
    pub m_full: Option<Vec<f64>>, pub mfl_full: Option<Vec<f64>>,                 // n x n
    pub fin_pairs: i64, pub ext: HashMap<usize, RowList>, pub n_starved: usize,
    pub bs_pre: Option<BsPools>,                                                  // the broad-sample pools, when sweep 2 filled them
    pub restart: Option<(String, String)>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Key { M0, Mj }

pub struct SweepOut {
    pub rows: Vec<RowList>, pub cl_win: Vec<i64>, pub cl_dmin: Vec<f64>, pub ident: Vec<(usize, usize)>,
    pub counters: (i64, i64, i64), pub full: Option<(Vec<f64>, Vec<f64>)>, pub overflow: i64,
}

struct RowRes { list: RowList, cl_win: Vec<i64>, cl_dmin: Vec<f64>, ident: Vec<usize>, counters: (i64, i64, i64), m: Vec<f64>, mfl: Vec<f64>, overflow: i64 }

/// m and the loci voting for one pair from its per-locus counts: the pair kernel's first median
#[inline]
fn median_m(w: &[u32], s: &[u32], rr: Option<&[f64]>, ok_r: Option<&[bool]>, v: &mut [f64], wv: &mut [f64]) -> (f64, usize) {
    let (one, hundred) = (1.0f32, 100.0f32);
    let mut cnt = 0usize;
    for l in 0..w.len() {
        if let Some(ok) = ok_r { if !ok[l] { continue; } }
        let r = match rr { Some(rr) => rr[l], None => 1.0 };
        let wc = w[l] as f32;
        if wc > 0.0 {
            let sc = s[l] as f32;
            let x = (hundred * (one - sc / wc)) as f64 / r;
            if x.is_finite() { v[cnt] = x; wv[cnt] = wc as f64; cnt += 1; }
        }
    }
    (if cnt >= MIN_LOCI_FOR_M && cnt > 0 { wmed_select(v, wv, cnt) } else { f64::NAN }, cnt)
}

/// the pair's resolution floor: the same median of half a mismatch per locus
#[inline]
fn median_floor(w: &[u32], rr: Option<&[f64]>, ok_r: Option<&[bool]>, f: &mut [f64], wf: &mut [f64]) -> f64 {
    let c50f = 50.0f32; let mut cntf = 0usize;
    for l in 0..w.len() {
        if let Some(ok) = ok_r { if !ok[l] { continue; } }
        let r = match rr { Some(rr) => rr[l], None => 1.0 };
        let wc = w[l] as f32;
        if wc > 0.0 {
            let xf = (c50f / wc) as f64 / r;
            if xf.is_finite() { f[cntf] = xf; wf[cntf] = wc as f64; cntf += 1; }
        }
    }
    if cntf >= MIN_LOCI_FOR_M && cntf > 0 { wmed_select(f, wf, cntf) } else { f64::NAN }
}

/// m, its floor and the loci voting for one pair from its per-locus counts
#[inline]
fn pair_medians(w: &[u32], s: &[u32], rr: Option<&[f64]>, ok_r: Option<&[bool]>, v: &mut [f64], wv: &mut [f64], f: &mut [f64], wf: &mut [f64]) -> (f64, f64, usize) {
    let (m, cnt) = median_m(w, s, rr, ok_r, v, wv);
    (m, median_floor(w, rr, ok_r, f, wf), cnt)
}

/// Without computing it: is the pair's m below thr, and how many loci vote? The weighted median is below thr exactly
/// when the loci whose value is below thr carry at least half the weight, so no selection is needed; and a locus's
/// value x / r is compared with thr through the products tlo = thr r (1 - 1e-12) and thi = thr r (1 + 1e-12), the
/// division kept for the values in between (the same answer as comparing the divided value, since r > 0)
#[inline]
fn m_below(w: &[u32], s: &[u32], rr: Option<&[f64]>, ok_r: Option<&[bool]>, thr: f64, tlo: &[f64], thi: &[f64]) -> (bool, usize) {
    let (one, hundred) = (1.0f32, 100.0f32);
    let (mut cnt, mut tot, mut below) = (0usize, 0u64, 0u64);
    for l in 0..w.len() {
        if let Some(ok) = ok_r { if !ok[l] { continue; } }
        if w[l] == 0 { continue; }
        cnt += 1; tot += w[l] as u64;
        let x = (hundred * (one - s[l] as f32 / w[l] as f32)) as f64;
        let lt = if x < tlo[l] { true } else if x > thi[l] { false } else { x / (match rr { Some(rr) => rr[l], None => 1.0 }) < thr };
        if lt { below += w[l] as u64; }
    }
    (2 * below >= tot, cnt)
}

/// per-thread scratch of a pair: its counts at every locus, the medians' buffers, the kernel's masks
struct Scratch { w: Vec<u32>, s: Vec<u32>, v: Vec<f64>, wv: Vec<f64>, f: Vec<f64>, wf: Vec<f64>, pa: Vec<u64>, eq: Vec<u64>,
                 z: Vec<f64>, wt: Vec<u64>, a: Vec<f64>, om: Vec<f64> }      // what the two directions of a pair share (sweep_pairs_once)
impl Scratch {
    fn new(core: &Core) -> Scratch {
        let (nloc, wpr) = (core.nloc, core.bits.wpr);
        Scratch { w: vec![0; nloc], s: vec![0; nloc], v: vec![0.0; nloc], wv: vec![0.0; nloc], f: vec![0.0; nloc], wf: vec![0.0; nloc], pa: vec![0; wpr], eq: vec![0; wpr],
                  z: vec![f64::NAN; nloc], wt: vec![0; nloc], a: vec![f64::NAN; nloc], om: vec![1.0; nloc] }
    }
}

/// the closest-match band of one (taxon, locus) as the columns go by: the entries (rate d, vals, column) of the
/// distinct rates within 1e-9 of the lowest so far, ascending, each the (vals, column)-smallest of its rate. One more
/// than BAND_SLOTS is kept, so a band with more distinct rates than slots is known (the overflow the log counts);
/// the result is that of the Python's rule over all columns at once, whatever the order the columns come in
#[derive(Clone, Copy)]
struct Band { n: u8, e: [(f64, f64, i64); BAND_SLOTS + 1] }
const BAND_EMPTY: Band = Band { n: 0, e: [(f64::INFINITY, f64::INFINITY, -1); BAND_SLOTS + 1] };

/// a column whose rate d is within the band (d <= *limit): into the band; the limit follows the minimum
#[inline(never)]
fn band_insert(b: &mut Band, limit: &mut f64, d: f64, mj: f64, c: i64) {
    let v = d + 1e-9 * mj / 100.0; let n = b.n as usize; const CAP: usize = BAND_SLOTS + 1;
    if n == 0 || d < b.e[0].0 {                              // a new minimum: the entries still within 1e-9 of it stay
        let band = d + 1e-9; let old = b.e; let mut k = 1; b.e[0] = (d, v, c);
        for e in old.iter().take(n) { if e.0 <= band && k < CAP { b.e[k] = *e; k += 1; } }
        b.n = k as u8; *limit = band; return;
    }
    for i in 0..n {
        if b.e[i].0 == d { if v < b.e[i].1 || (v == b.e[i].1 && c < b.e[i].2) { b.e[i] = (d, v, c); } return; }
        if b.e[i].0 > d {                                    // a new distinct rate before entry i
            let last = if n < CAP { n } else { CAP - 1 };
            for j in (i + 1..=last).rev() { b.e[j] = b.e[j - 1]; }
            b.e[i] = (d, v, c); b.n = (last + 1) as u8; return;
        }
    }
    if n < CAP { b.e[n] = (d, v, c); b.n += 1; }
}

/// the rows of a chunk against every column. The columns are the outer loop: a column's planes are read from memory
/// once for the whole chunk, whose own rows stay in the cache (the sweep is bound by memory traffic otherwise).
/// Nothing is kept per pair but what the results need: the cmax smallest keys so far and, in sweep 2, the band of
/// the closest match at every locus.
/// lazy: a pair's medians are computed only where their values are used. A row's list needs m only for a pair whose
/// key can be among its cmax smallest: once the list has been cut back, thr is its largest kept key, and a pair
/// whose m is not below thr cannot enter (its key is at least m, and at an equal key the earlier column stays), which
/// m_below decides without the median. The closest match needs m only for a pair that enters a band. Everything else
/// (whether m is finite, the counters, the identical pairs) follows from the counts. In lazy mode the nfin of an M0
/// sweep counts the keys computed, not every eligible key (prepare settles the rows where that matters)
#[allow(clippy::too_many_arguments)]
fn sweep_chunk(core: &Core, rows: &[usize], rr: Option<&[f64]>, ok_r: Option<&[bool]>, key: Key, cmax: usize, fit: f64, want_closest: bool, want_m: bool, lazy: bool, sc: &mut Scratch) -> Vec<RowRes> {
    let (n, nloc) = (core.n, core.nloc); let nr = rows.len();
    struct St { m: Vec<f64>, mfl: Vec<f64>, entries: Vec<(f64, usize, f64, f64)>, nfin: i64, ident: Vec<usize>, counters: (i64, i64, i64), thr: f64, tlo: Vec<f64>, thi: Vec<f64> }
    let mut st: Vec<St> = (0..nr).map(|_| St { m: if want_m { vec![f64::NAN; n] } else { vec![] }, mfl: if want_m { vec![f64::NAN; n] } else { vec![] }, entries: Vec::new(), nfin: 0, ident: vec![], counters: (0, 0, 0),
                                               thr: f64::INFINITY, tlo: if lazy { vec![f64::INFINITY; nloc] } else { vec![] }, thi: if lazy { vec![f64::INFINITY; nloc] } else { vec![] } }).collect();
    let by_key = |a: &(f64, usize, f64, f64), b: &(f64, usize, f64, f64)| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1));
    let cap = if cmax >= n { usize::MAX } else { 2 * cmax.max(16) };       // the list is cut back to its cmax smallest whenever it reaches this
    let mut bands = if want_closest { vec![BAND_EMPTY; nr * nloc] } else { vec![] };
    let mut limits = if want_closest { vec![f64::INFINITY; nr * nloc] } else { vec![] };
    let mut limits1 = if want_closest { vec![f64::INFINITY; nr * nloc] } else { vec![] };          // limit (1 + 1e-12): the test without a division
    for c in 0..n {
        for (ri, &g) in rows.iter().enumerate() {
            core.pair_counts(g, c, &mut sc.w, &mut sc.s, &mut sc.pa, &mut sc.eq);
            let (w, s) = (&sc.w, &sc.s); let t = &mut st[ri];
            let diag = c == g;
            let tot_same: f64 = s.iter().map(|x| *x as u64).sum::<u64>() as f64; let tot_shared: f64 = w.iter().map(|x| *x as u64).sum::<u64>() as f64;
            let ani = if tot_shared > 0.0 && !diag { 100.0 * tot_same / tot_shared } else { f64::NAN };
            let is_ident = tot_shared > 0.0 && tot_same == tot_shared && !diag;
            let base = (if ani.is_nan() { 100.0 } else { 100.0 - ani }) * 1e-9;
            // mm: the pair's m, or not yet computed (have_m false) in lazy mode; m_fin: whether it is finite
            let (mut mm, mut mf, mut have_m, m_fin, nl, candidate);
            if !lazy {
                let r = if diag { (f64::NAN, f64::NAN, 0) } else { pair_medians(w, s, rr, ok_r, &mut sc.v, &mut sc.wv, &mut sc.f, &mut sc.wf) };
                mm = r.0; mf = r.1; nl = r.2; have_m = true; m_fin = mm.is_finite(); candidate = true;
            } else {
                let (below, cnt) = if diag { (false, 0) } else { m_below(w, s, rr, ok_r, t.thr, &t.tlo, &t.thi) };
                mm = f64::NAN; mf = f64::NAN; have_m = false; nl = cnt; m_fin = cnt >= MIN_LOCI_FOR_M; candidate = below;
            }
            if want_m { t.m[c] = mm; t.mfl[c] = mf; }
            if is_ident && g < c { t.ident.push(c); }
            if g < c && core.keep[g] && core.keep[c] {
                t.counters.1 += 1; if nl < MIN_LOCI_FOR_M { t.counters.0 += 1; } if m_fin { t.counters.2 += 1; }
            }
            let key_fin = m_fin && !is_ident;                    // the pair has a finite mj (and may have a finite m0)
            if key_fin && candidate {
                if !have_m { mm = median_m(w, s, rr, ok_r, &mut sc.v, &mut sc.wv).0; mf = median_floor(w, rr, ok_r, &mut sc.f, &mut sc.wf); have_m = true; }
                let k = match key {
                    Key::M0 => if mm > mf { mm } else { f64::INFINITY },
                    Key::Mj => mm + (base + c as f64 * 1e-12),
                };
                if k.is_finite() {
                    if !lazy || key == Key::M0 { t.nfin += 1; }
                    if k < t.thr {
                        t.entries.push((k, c, mm, mf));
                        if t.entries.len() >= cap {
                            t.entries.select_nth_unstable_by(cmax - 1, by_key); t.entries.truncate(cmax);
                            if lazy {
                                t.thr = t.entries[cmax - 1].0;
                                for l in 0..nloc { let tr = t.thr * (match rr { Some(rr) => rr[l], None => 1.0 }); t.tlo[l] = tr * (1.0 - 1e-12); t.thi[l] = tr * (1.0 + 1e-12); }
                            }
                        }
                    }
                }
            }
            if lazy && key == Key::Mj && key_fin { t.nfin += 1; }
            if want_closest && key_fin {
                let (bd, lm, lm1) = (&mut bands[ri * nloc..(ri + 1) * nloc], &mut limits[ri * nloc..(ri + 1) * nloc], &mut limits1[ri * nloc..(ri + 1) * nloc]);
                for li in 0..nloc {
                    let o = w[li] as f64;
                    if o >= fit {
                        // d = a / om is above the limit for certain when a is above limit (1 + 1e-12) om: no division then
                        let a = o - s[li] as f64 + HALF; let om = o.max(1.0);
                        if a > lm1[li] * om { continue; }
                        let d = a / om;
                        if d <= lm[li] {
                            if !have_m { mm = median_m(w, s, rr, ok_r, &mut sc.v, &mut sc.wv).0; have_m = true; }
                            band_insert(&mut bd[li], &mut lm[li], d, mm + (base + c as f64 * 1e-12), c as i64);
                            lm1[li] = lm[li] * (1.0 + 1e-12);
                        }
                    }
                }
            }
        }
    }
    let mut out = Vec::with_capacity(nr);
    for (ri, mut t) in st.into_iter().enumerate() {
        t.entries.sort_by(by_key); t.entries.truncate(cmax);
        let list = RowList { cand: t.entries.iter().map(|e| e.1 as i64).collect(), key: t.entries.iter().map(|e| e.0).collect(),
                             m: t.entries.iter().map(|e| e.2).collect(), mfl: t.entries.iter().map(|e| e.3).collect(), nfin: t.nfin };
        let mut cl_win = vec![]; let mut cl_dmin = vec![]; let mut overflow = 0i64;
        if want_closest {
            cl_win = vec![-1i64; nloc]; cl_dmin = vec![f64::INFINITY; nloc];
            for li in 0..nloc {
                let b = &bands[ri * nloc + li]; let kept = (b.n as usize).min(BAND_SLOTS);
                if b.n as usize > BAND_SLOTS { overflow += 1; }
                if kept == 0 { continue; }
                // closest_winners: the (vals, column)-smallest of the kept entries
                let mut best = 0usize;
                for j in 1..kept { if b.e[j].1 < b.e[best].1 || (b.e[j].1 == b.e[best].1 && b.e[j].2 < b.e[best].2) { best = j; } }
                cl_win[li] = b.e[best].2; cl_dmin[li] = b.e[0].0;
            }
        }
        out.push(RowRes { list, cl_win, cl_dmin, ident: t.ident, counters: t.counters, m: t.m, mfl: t.mfl, overflow });
    }
    out
}

/// the sweep over the rows of the pair matrix (rows: the given taxa against everyone, for the second passes; all of
/// them: each pair once, sweep_pairs_once)
#[allow(clippy::too_many_arguments)]
pub fn sweep(core: &Core, rr: Option<&[f64]>, ok_r: Option<&[bool]>, key: Key, cmax: usize, fit: i64, want_closest: bool, want_full: bool, rows: Option<&[usize]>, stage: Option<&str>) -> SweepOut {
    // every row against every column: each pair is computed once and serves both of its rows
    if rows.is_none() && std::env::var_os("LOCUSMASKER_NO_PAIRS_ONCE").is_none() { return sweep_pairs_once(core, rr, ok_r, key, cmax, fit, want_closest, want_full, stage, None); }
    sweep_rows(core, rr, ok_r, key, cmax, fit, want_closest, want_full, rows, stage)
}

/// the sweep row by row: the given rows (or all of them) against every column, each row on its own
#[allow(clippy::too_many_arguments)]
fn sweep_rows(core: &Core, rr: Option<&[f64]>, ok_r: Option<&[bool]>, key: Key, cmax: usize, fit: i64, want_closest: bool, want_full: bool, rows: Option<&[usize]>, stage: Option<&str>) -> SweepOut {
    let (n, nloc) = (core.n, core.nloc);
    let all: Vec<usize> = (0..n).collect(); let rows_v: &[usize] = rows.unwrap_or(&all);
    let prog = crate::log::prog();
    // rows per chunk: about half a megabyte of planes (they stay in the cache while the columns stream by), few enough
    // that every thread has several chunks
    let threads = rayon::current_num_threads().max(1);
    let chunk = (SWEEP_CHUNK_BYTES / core.bits.row_bytes().max(1)).clamp(4, 64).min(rows_v.len().div_ceil(4 * threads)).max(1);
    if let Some(st) = stage {
        log!("pair sweep {}: {} rows of the pair matrix against {} taxa in chunks of {}, {} thread{}, the pair counts {}", st, commas(rows_v.len() as i64), commas(n as i64), chunk, threads,
             if threads > 1 { "s" } else { "" }, if core.slab.is_some() { "kept for the run" } else { "from the residues" });
        prog.start(st, rows_v.len());
    }
    let cmax = cmax.min(n);
    // the medians only where they are used, unless every m is kept (the full matrix) or a locus's rate is so small that
    // a value divided by it could overflow (the shortcut reads the loci that vote from the counts alone)
    let lazy = lazy_ok(want_full, nloc, rr, ok_r);
    let res: Vec<Vec<RowRes>> = rows_v.par_chunks(chunk).map_init(|| Scratch::new(core),
        |sc, ch| { let r = sweep_chunk(core, ch, rr, ok_r, key, cmax, fit as f64, want_closest, want_full, lazy, sc); if stage.is_some() { prog.tick_k(ch.len(), None); } r }).collect();
    if stage.is_some() { prog.finish(); }
    let mut out = SweepOut { rows: Vec::with_capacity(rows_v.len()), cl_win: vec![], cl_dmin: vec![], ident: vec![], counters: (0, 0, 0), full: None, overflow: 0 };
    if want_closest { out.cl_win.reserve(rows_v.len() * nloc); out.cl_dmin.reserve(rows_v.len() * nloc); }
    let mut full = if want_full && rows.is_none() { Some((Vec::with_capacity(n * n), Vec::with_capacity(n * n))) } else { None };
    for (k, r) in res.into_iter().flatten().enumerate() {
        let g = rows_v[k];
        for c in r.ident { out.ident.push((g, c)); }
        out.counters.0 += r.counters.0; out.counters.1 += r.counters.1; out.counters.2 += r.counters.2; out.overflow += r.overflow;
        if want_closest { out.cl_win.extend(r.cl_win); out.cl_dmin.extend(r.cl_dmin); }
        if let Some(f) = full.as_mut() { f.0.extend(r.m); f.1.extend(r.mfl); }
        out.rows.push(r.list);
    }
    out.full = full; out
}

/// the medians only where they are used, unless every m is kept (the full matrix) or a locus's rate is so small or so
/// large that a value divided by it could leave the normal range (the shortcuts read the loci that vote from the
/// counts alone, and compare products against thresholds with a margin that assumes normal numbers)
fn lazy_ok(want_full: bool, nloc: usize, rr: Option<&[f64]>, ok_r: Option<&[bool]>) -> bool {
    !want_full && std::env::var_os("LOCUSMASKER_NO_LAZY").is_none()
        && (0..nloc).all(|l| ok_r.map_or(true, |ok| !ok[l]) || rr.map_or(true, |r| r[l].is_finite() && r[l] >= 1e-300 && r[l] <= 1e300))
}

// ------------------------------------------------------------------ the sweep that computes each pair once
// A pair's counts are the same from either side, so the rows of the pair matrix are swept above the diagonal only:
// the pair (g, c), g < c, is counted once and offered to row g (as its column c) and to row c (as its column g). A
// row's results are therefore gathered by many threads: each row has a state behind a lock (the cmax smallest keys
// so far, the closest-match bands), and the cuts that decide whether a pair can matter to a row at all (the list's
// largest kept key, each band's limit) are published in atomics that are read without the lock. A cut only ever
// tightens, so a reader holding an old value lets through a pair that no longer matters, never the reverse: the lock
// is taken for the few pairs that pass, and the state then decides with what it holds. What a row ends with is a
// function of the set of its columns, not of the order they came in (the smallest keys by (key, column); the band
// rule of band_insert), so the results are those of the row-by-row sweep, at any thread count.

/// an entry of a row's candidate list, ordered by (key, column)
#[derive(Clone, Copy)]
struct Ent { key: f64, c: u32, m: f64, mfl: f64 }
impl PartialEq for Ent { fn eq(&self, o: &Self) -> bool { self.cmp(o) == std::cmp::Ordering::Equal } }
impl Eq for Ent {}
impl PartialOrd for Ent { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for Ent { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.key.partial_cmp(&o.key).unwrap().then(self.c.cmp(&o.c)) } }

const B_NONE: u32 = u32::MAX;            // no column yet at this locus
const B_MULTI: u32 = u32::MAX - 1;       // more than one rate in the band: it is in `multi`

/// a row's state: the cmax smallest entries so far (a heap, the largest on top) and, in sweep 2, the closest-match
/// band of every locus. Nearly every band holds one rate, so one entry per locus is held in line (rate, vals,
/// column) and the bands with more are kept aside
struct RowState { heap: std::collections::BinaryHeap<Ent>, bd: Vec<f64>, bv: Vec<f64>, bc: Vec<u32>, multi: Vec<(u32, Band)> }

impl RowState {
    fn band(&self, li: usize) -> Band {
        match self.bc[li] {
            B_NONE => BAND_EMPTY,
            B_MULTI => self.multi.iter().find(|x| x.0 == li as u32).expect("a band kept aside").1,
            c => { let mut b = BAND_EMPTY; b.n = 1; b.e[0] = (self.bd[li], self.bv[li], c as i64); b }
        }
    }
    /// a column whose rate d may be within the band of locus li: into the band if it is. Returns the band's limit
    fn band_event(&mut self, li: usize, d: f64, mj: f64, col: usize) -> f64 {
        let mut b = self.band(li);
        let mut limit = if b.n == 0 { f64::INFINITY } else { b.e[0].0 + 1e-9 };
        if d <= limit {
            band_insert(&mut b, &mut limit, d, mj, col as i64);
            let was_multi = self.bc[li] == B_MULTI;
            if b.n == 1 {
                if was_multi { self.multi.retain(|x| x.0 != li as u32); }
                self.bd[li] = b.e[0].0; self.bv[li] = b.e[0].1; self.bc[li] = b.e[0].2 as u32;
            } else if was_multi { self.multi.iter_mut().find(|x| x.0 == li as u32).unwrap().1 = b; }
            else { self.multi.push((li as u32, b)); self.bc[li] = B_MULTI; }
        }
        limit
    }
    /// an entry whose key may be among the cmax smallest: into the list if it is. Returns the list's cut (its largest
    /// kept key once it holds cmax entries)
    fn list_event(&mut self, cmax: usize, e: Ent) -> f64 {
        if self.heap.capacity() == 0 && cmax <= 4096 { self.heap.reserve_exact(cmax); }
        if self.heap.len() < cmax { self.heap.push(e); }
        else if let Some(mut top) = self.heap.peek_mut() { if e < *top { *top = e; } }
        if self.heap.len() >= cmax { self.heap.peek().map_or(f64::INFINITY, |t| t.key) } else { f64::INFINITY }
    }
}

/// values written from many threads, each element by one of them (the full m matrix: element (i, j) by the task that
/// holds the pair)
struct SharedF64(std::cell::UnsafeCell<Vec<f64>>);
unsafe impl Sync for SharedF64 {}
impl SharedF64 {
    /// SAFETY: no two threads write the same element, and nothing reads until the writers are done
    #[inline] unsafe fn set(&self, i: usize, v: f64) { *(*self.0.get()).as_mut_ptr().add(i) = v; }
}

/// The broad-sample pools as sweep 2 fills them: each taxon's pool members by column (cols, ascending) with the slot
/// each has in the pool as drawn (slots), and the pool's pair divergences in drawn order (out: n x pn), each written
/// once, by the task that holds the pair
pub struct PoolIndex { pub pn: usize, pub cols: Vec<u32>, pub slots: Vec<u32>, out: SharedF64 }
impl PoolIndex {
    /// every taxon's pool (pool: n x pn as drawn, bs_pool_of)
    pub fn new(n: usize, pn: usize, pool: &[i32]) -> PoolIndex {
        let mut cols = vec![0u32; n * pn]; let mut slots = vec![0u32; n * pn];
        if pn > 0 {
            cols.par_chunks_mut(pn).zip(slots.par_chunks_mut(pn)).enumerate().for_each(|(g, (cg, sg))| {
                let mut v: Vec<(u32, u32)> = pool[g * pn..(g + 1) * pn].iter().enumerate().map(|(j, c)| (*c as u32, j as u32)).collect();
                v.sort_unstable();
                for (k, (c, j)) in v.into_iter().enumerate() { cg[k] = c; sg[k] = j; }
            });
        }
        PoolIndex { pn, cols, slots, out: SharedF64(std::cell::UnsafeCell::new(vec![f64::NAN; n * pn])) }
    }
    pub fn into_mstar(self) -> Vec<f64> { self.out.0.into_inner() }
}

/// What the two directions of a pair share, from its counts: per locus that votes (a rate, shared sites), the
/// value z = x q with x the float32 divergence and q the reciprocal of the locus rate, and its weight; and for the
/// closest match, a = mismatches + 1/2 and om = the shared sites, at the loci with enough of them (NaN elsewhere, so
/// that every comparison fails there). Returns the total weight and the number of loci that vote.
/// okm: -1 where the locus has a rate, 0 elsewhere. The loci from `from` on (the AVX2 version leaves its tail here)
#[inline]
fn derive_scalar(w: &[u32], s: &[u32], q: &[f64], okm: &[i32], fit: f64, bands: bool, z: &mut [f64], wt: &mut [u64], a: &mut [f64], om: &mut [f64], from: usize) -> (u64, usize) {
    let (one, hundred) = (1.0f32, 100.0f32); let (mut tot, mut cnt) = (0u64, 0usize);
    for l in from..w.len() {
        let vote = (okm[l] != 0) & (w[l] > 0);
        let x = (hundred * (one - s[l] as f32 / w[l] as f32)) as f64;
        z[l] = if vote { x * q[l] } else { f64::NAN }; wt[l] = if vote { w[l] as u64 } else { 0 };
        tot += wt[l]; cnt += vote as usize;
        if bands { let o = w[l] as f64; a[l] = if o >= fit { o - s[l] as f64 + HALF } else { f64::NAN }; om[l] = o.max(1.0); }
    }
    (tot, cnt)
}
/// the same operations eight loci at a time (each is the IEEE operation of the scalar code: the same bits)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn derive_avx2(w: &[u32], s: &[u32], q: &[f64], okm: &[i32], fit: f64, bands: bool, z: &mut [f64], wt: &mut [u64], a: &mut [f64], om: &mut [f64]) -> (u64, usize) {
    use std::arch::x86_64::*;
    let n = w.len(); let (one, hundred, nan) = (_mm256_set1_ps(1.0), _mm256_set1_ps(100.0), _mm256_set1_pd(f64::NAN));
    let (vfit, vhalf, vone) = (_mm256_set1_pd(fit), _mm256_set1_pd(HALF), _mm256_set1_pd(1.0));
    let mut acc = _mm256_setzero_si256(); let mut cnt = 0usize; let mut l = 0;
    while l + 8 <= n {
        let wi = _mm256_loadu_si256(w.as_ptr().add(l) as *const __m256i); let si = _mm256_loadu_si256(s.as_ptr().add(l) as *const __m256i);
        // the loci that vote: a rate and shared sites (the counts are below 2^24: exact as float32, and as signed integers)
        let vote = _mm256_and_si256(_mm256_loadu_si256(okm.as_ptr().add(l) as *const __m256i), _mm256_cmpgt_epi32(wi, _mm256_setzero_si256()));
        cnt += (_mm256_movemask_ps(_mm256_castsi256_ps(vote)) as u32).count_ones() as usize;
        let x = _mm256_mul_ps(hundred, _mm256_sub_ps(one, _mm256_div_ps(_mm256_cvtepi32_ps(si), _mm256_cvtepi32_ps(wi))));
        for h in 0..2 {
            let (x4, w4, s4, v4) = if h == 0 { (_mm256_castps256_ps128(x), _mm256_castsi256_si128(wi), _mm256_castsi256_si128(si), _mm256_castsi256_si128(vote)) }
                                   else { (_mm256_extractf128_ps::<1>(x), _mm256_extracti128_si256::<1>(wi), _mm256_extracti128_si256::<1>(si), _mm256_extracti128_si256::<1>(vote)) };
            let at = l + 4 * h; let v64 = _mm256_cvtepi32_epi64(v4);
            let zz = _mm256_mul_pd(_mm256_cvtps_pd(x4), _mm256_loadu_pd(q.as_ptr().add(at)));
            _mm256_storeu_pd(z.as_mut_ptr().add(at), _mm256_blendv_pd(nan, zz, _mm256_castsi256_pd(v64)));
            let wv = _mm256_and_si256(_mm256_cvtepu32_epi64(w4), v64);
            _mm256_storeu_si256(wt.as_mut_ptr().add(at) as *mut __m256i, wv); acc = _mm256_add_epi64(acc, wv);
            if bands {
                let o = _mm256_cvtepi32_pd(w4);
                let av = _mm256_add_pd(_mm256_sub_pd(o, _mm256_cvtepi32_pd(s4)), vhalf);
                _mm256_storeu_pd(a.as_mut_ptr().add(at), _mm256_blendv_pd(nan, av, _mm256_cmp_pd::<_CMP_GE_OQ>(o, vfit)));
                _mm256_storeu_pd(om.as_mut_ptr().add(at), _mm256_max_pd(o, vone));
            }
        }
        l += 8;
    }
    let mut t = [0u64; 4]; _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, acc);
    let (tot2, cnt2) = derive_scalar(w, s, q, okm, fit, bands, z, wt, a, om, l);
    (t[0] + t[1] + t[2] + t[3] + tot2, cnt + cnt2)
}
#[inline]
#[allow(clippy::too_many_arguments)]
fn derive(w: &[u32], s: &[u32], q: &[f64], okm: &[i32], fit: f64, bands: bool, z: &mut [f64], wt: &mut [u64], a: &mut [f64], om: &mut [f64], avx2: bool) -> (u64, usize) {
    #[cfg(target_arch = "x86_64")]
    if avx2 { return unsafe { derive_avx2(w, s, q, okm, fit, bands, z, wt, a, om) }; }
    let _ = avx2; derive_scalar(w, s, q, okm, fit, bands, z, wt, a, om, 0)
}

/// the weight of the loci whose z is below lo for certain, and whether any z lies in [lo, hi] (to be settled exactly)
#[inline]
fn below_scalar(z: &[f64], wt: &[u64], lo: f64, hi: f64) -> (u64, bool) {
    let (mut below, mut unsure) = (0u64, false);
    for l in 0..z.len() { if z[l] < lo { below += wt[l]; } unsure |= z[l] >= lo && z[l] <= hi; }
    (below, unsure)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn below_avx2(z: &[f64], wt: &[u64], lo: f64, hi: f64) -> (u64, bool) {
    use std::arch::x86_64::*;
    let n = z.len(); let (vlo, vhi) = (_mm256_set1_pd(lo), _mm256_set1_pd(hi));
    let mut acc = _mm256_setzero_si256(); let mut un = _mm256_setzero_pd(); let mut l = 0;
    while l + 4 <= n {
        let zv = _mm256_loadu_pd(z.as_ptr().add(l));
        let lt = _mm256_cmp_pd::<_CMP_LT_OQ>(zv, vlo);
        acc = _mm256_add_epi64(acc, _mm256_and_si256(_mm256_castpd_si256(lt), _mm256_loadu_si256(wt.as_ptr().add(l) as *const __m256i)));
        un = _mm256_or_pd(un, _mm256_and_pd(_mm256_cmp_pd::<_CMP_GE_OQ>(zv, vlo), _mm256_cmp_pd::<_CMP_LE_OQ>(zv, vhi)));
        l += 4;
    }
    let mut t = [0u64; 4]; _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, acc);
    let (mut below, mut unsure) = (t[0] + t[1] + t[2] + t[3], _mm256_movemask_pd(un) != 0);
    while l < n { if z[l] < lo { below += wt[l]; } unsure |= z[l] >= lo && z[l] <= hi; l += 1; }
    (below, unsure)
}

/// Without computing it: is the pair's m at most thr? The weighted median is at most thr exactly when the loci
/// whose value is at most thr carry at least half the weight. A locus's value x / r is compared through z = x q:
/// below thr (1 - 1e-12) it is below thr for certain, above thr (1 + 1e-12) above it for certain (the rates are
/// normal numbers, lazy_ok), and when a locus lies in between the divisions are done
#[inline]
fn m_at_most(sc: &Scratch, tot: u64, thr: f64, rr: Option<&[f64]>, okm: &[i32], avx2: bool) -> bool {
    let (lo, hi) = (thr * (1.0 - 1e-12), thr * (1.0 + 1e-12));
    #[cfg(target_arch = "x86_64")]
    let (mut below, unsure) = if avx2 { unsafe { below_avx2(&sc.z, &sc.wt, lo, hi) } } else { below_scalar(&sc.z, &sc.wt, lo, hi) };
    #[cfg(not(target_arch = "x86_64"))]
    let (mut below, unsure) = { let _ = avx2; below_scalar(&sc.z, &sc.wt, lo, hi) };
    if unsure {
        let (one, hundred) = (1.0f32, 100.0f32); below = 0;
        for l in 0..sc.w.len() {
            if okm[l] == 0 || sc.w[l] == 0 { continue; }
            let x = (hundred * (one - sc.s[l] as f32 / sc.w[l] as f32)) as f64;
            if x / (match rr { Some(rr) => rr[l], None => 1.0 }) <= thr { below += sc.w[l] as u64; }
        }
    }
    2 * below >= tot
}

/// is any locus's a at most lim1 om: can the pair's rate be within a band of this row?
#[inline]
fn band_hit_scalar(a: &[f64], om: &[f64], lim1: &[f64]) -> bool { let mut hit = false; for l in 0..a.len() { hit |= a[l] <= lim1[l] * om[l]; } hit }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn band_hit_avx2(a: &[f64], om: &[f64], lim1: &[f64]) -> bool {
    use std::arch::x86_64::*;
    let n = a.len(); let mut hit = _mm256_setzero_pd(); let mut l = 0;
    while l + 4 <= n {
        let p = _mm256_mul_pd(_mm256_loadu_pd(lim1.as_ptr().add(l)), _mm256_loadu_pd(om.as_ptr().add(l)));
        hit = _mm256_or_pd(hit, _mm256_cmp_pd::<_CMP_LE_OQ>(_mm256_loadu_pd(a.as_ptr().add(l)), p)); l += 4;
    }
    let mut h = _mm256_movemask_pd(hit) != 0;
    while l < n { h |= a[l] <= lim1[l] * om[l]; l += 1; }
    h
}
#[inline]
fn band_hit(a: &[f64], om: &[f64], lim1: &[f64], avx2: bool) -> bool {
    #[cfg(target_arch = "x86_64")]
    if avx2 { return unsafe { band_hit_avx2(a, om, lim1) }; }
    let _ = avx2; band_hit_scalar(a, om, lim1)
}

#[allow(clippy::too_many_arguments)]
fn sweep_pairs_once(core: &Core, rr: Option<&[f64]>, ok_r: Option<&[bool]>, key: Key, cmax: usize, fit: i64, want_closest: bool, want_full: bool, stage: Option<&str>, pools: Option<&PoolIndex>) -> SweepOut {
    use std::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};
    use std::sync::Mutex;
    let (n, nloc) = (core.n, core.nloc); let prog = crate::log::prog();
    assert!(n < B_MULTI as usize, "too many taxa for a 32-bit column index");
    let threads = rayon::current_num_threads().max(1);
    let chunk = (SWEEP_CHUNK_BYTES / core.bits.row_bytes().max(1)).clamp(4, 64).min(n.div_ceil(4 * threads)).max(1);
    if let Some(st) = stage {
        log!("pair sweep {}: each pair of {} taxa once, {} rows at a time, {} thread{}, the pair counts {}", st, commas(n as i64), chunk, threads,
             if threads > 1 { "s" } else { "" }, if core.slab.is_some() { "kept for the run" } else { "from the residues" });
        prog.start(st, n);
    }
    let cmax = cmax.min(n).max(1); let fit_f = fit as f64; let avx2 = core.bits.has_avx2();
    let lazy = lazy_ok(want_full, nloc, rr, ok_r);
    let okm: Vec<i32> = (0..nloc).map(|l| if ok_r.map_or(true, |ok| ok[l]) { -1 } else { 0 }).collect();
    let q: Vec<f64> = (0..nloc).map(|l| match rr { Some(r) if okm[l] != 0 => 1.0 / r[l], _ => 1.0 }).collect();
    let inf = f64::INFINITY.to_bits();
    let thr: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(inf)).collect();
    let lim1: Vec<AtomicU64> = if want_closest { (0..n * nloc).map(|_| AtomicU64::new(inf)).collect() } else { vec![] };
    let nfin: Vec<AtomicI64> = (0..n).map(|_| AtomicI64::new(0)).collect();
    let states: Vec<Mutex<RowState>> = (0..n).map(|_| Mutex::new(RowState { heap: std::collections::BinaryHeap::new(),
        bd: if want_closest { vec![0.0; nloc] } else { vec![] }, bv: if want_closest { vec![0.0; nloc] } else { vec![] }, bc: if want_closest { vec![B_NONE; nloc] } else { vec![] }, multi: vec![] })).collect();
    let full: Option<(SharedF64, SharedF64)> = if want_full { Some((SharedF64(std::cell::UnsafeCell::new(vec![f64::NAN; n * n])), SharedF64(std::cell::UnsafeCell::new(vec![f64::NAN; n * n])))) } else { None };
    // progress in rows, in proportion to the pairs done (row g holds n - 1 - g pairs)
    let total_pairs = (n as u64) * (n as u64 - 1) / 2; let done_pairs = AtomicU64::new(0); let ticked = AtomicU64::new(0);
    let starts: Vec<usize> = (0..n).step_by(chunk).collect();
    let parts: Vec<(Vec<(usize, usize)>, (i64, i64, i64))> = starts.par_iter().with_max_len(1).map_init(|| Scratch::new(core), |sc, &a| {
        let b = (a + chunk).min(n); let nr = b - a;
        let mut ident: Vec<(usize, usize)> = vec![]; let mut counters = (0i64, 0i64, 0i64);
        // this chunk's rows: their cuts as last seen (other threads tighten them too; refreshed every few columns)
        let mut lthr = vec![f64::INFINITY; nr]; let mut llim = if want_closest { vec![f64::INFINITY; nr * nloc] } else { vec![] };
        let mut own_nfin = vec![0i64; nr]; let mut clim = if want_closest { vec![f64::INFINITY; nloc] } else { vec![] };
        // the broad-sample pools: for each of this chunk's rows, where it stands among its pool's columns (they come in
        // ascending order); for a column, which of the chunk's rows are in its pool
        let pn = pools.map_or(0, |p| p.pn);
        let mut pcur: Vec<usize> = match pools { Some(p) => (a..b).map(|g| p.cols[g * pn..(g + 1) * pn].partition_point(|x| (*x as usize) <= g)).collect(), None => vec![] };
        let mut cmem: Vec<(usize, usize)> = vec![];           // (row of the chunk, slot in the column's pool)
        for c in a + 1..n {
            let gmax = b.min(c);
            if let Some(p) = pools {
                cmem.clear(); let pc = &p.cols[c * pn..(c + 1) * pn];
                let mut i = pc.partition_point(|x| (*x as usize) < a);
                while i < pn && (pc[i] as usize) < gmax { cmem.push((pc[i] as usize - a, p.slots[c * pn + i] as usize)); i += 1; }
            }
            if (c - a) & 7 == 1 {
                for ri in 0..nr { lthr[ri] = f64::from_bits(thr[a + ri].load(Relaxed)); }
                if want_closest { for (k, v) in llim.iter_mut().enumerate() { *v = f64::from_bits(lim1[a * nloc + k].load(Relaxed)); } }
            }
            // the column as a row: its cuts as they stand
            let mut cthr = f64::from_bits(thr[c].load(Relaxed));
            if want_closest { for l in 0..nloc { clim[l] = f64::from_bits(lim1[c * nloc + l].load(Relaxed)); } }
            let mut c_nfin = 0i64;
            for g in a..gmax {
                let ri = g - a;
                core.pair_counts(g, c, &mut sc.w, &mut sc.s, &mut sc.pa, &mut sc.eq);
                let tot_same: u64 = sc.s.iter().map(|x| *x as u64).sum(); let tot_shared: u64 = sc.w.iter().map(|x| *x as u64).sum();
                let is_ident = tot_shared > 0 && tot_same == tot_shared;
                if is_ident { ident.push((g, c)); }
                // mm, mf: the pair's m and floor once computed (have_m, have_f); m_fin: whether m is finite
                let (mut mm, mut mf, mut have_m, mut have_f) = (f64::NAN, f64::NAN, false, false);
                // nl: the loci that vote; tot_w: their weight (the lazy tests read it)
                let mut tot_w = 0u64;
                let (nl, m_fin) = if lazy {
                    let (tw, cnt) = derive(&sc.w, &sc.s, &q, &okm, fit_f, want_closest, &mut sc.z, &mut sc.wt, &mut sc.a, &mut sc.om, avx2);
                    tot_w = tw; (cnt, cnt >= MIN_LOCI_FOR_M)
                } else {
                    let r = pair_medians(&sc.w, &sc.s, rr, ok_r, &mut sc.v, &mut sc.wv, &mut sc.f, &mut sc.wf);
                    mm = r.0; mf = r.1; have_m = true; have_f = true;
                    if let Some((fm, ff)) = &full { unsafe { fm.set(g * n + c, mm); fm.set(c * n + g, mm); ff.set(g * n + c, mf); ff.set(c * n + g, mf); } }
                    if want_closest { for l in 0..nloc { let o = sc.w[l] as f64; sc.a[l] = if o >= fit_f { o - sc.s[l] as f64 + HALF } else { f64::NAN }; sc.om[l] = o.max(1.0); } }
                    (r.2, mm.is_finite())
                };
                if core.keep[g] && core.keep[c] { counters.1 += 1; if nl < MIN_LOCI_FOR_M { counters.0 += 1; } if m_fin { counters.2 += 1; } }
                if let Some(p) = pools {
                    // a pair of a broad-sample pool (c in g's, g in c's): its divergence as Prep::pair_m gives it
                    let own = pcur[ri] < pn && p.cols[g * pn + pcur[ri]] as usize == c;
                    let col = cmem.iter().find(|x| x.0 == ri).map(|x| x.1);
                    if own || col.is_some() {
                        if !have_m { mm = median_m(&sc.w, &sc.s, rr, ok_r, &mut sc.v, &mut sc.wv).0; have_m = true; }
                        if !have_f { mf = median_floor(&sc.w, rr, ok_r, &mut sc.f, &mut sc.wf); have_f = true; }
                        let v = if mf.is_finite() { mm.max(mf) } else { mm };
                        if own { unsafe { p.out.set(g * pn + p.slots[g * pn + pcur[ri]] as usize, v); } pcur[ri] += 1; }
                        if let Some(slot) = col { unsafe { p.out.set(c * pn + slot, v); } }
                    }
                }
                if !m_fin || is_ident { continue; }                  // no finite mj: the pair enters no list and no band
                let base = (100.0 - 100.0 * tot_same as f64 / tot_shared as f64) * 1e-9;
                if key == Key::Mj { own_nfin[ri] += 1; c_nfin += 1; }
                // the two directions: row g with column c, row c with column g
                for dir in 0..2 {
                    let (row, col) = if dir == 0 { (g, c) } else { (c, g) };
                    let t = if dir == 0 { lthr[ri] } else { cthr };
                    if !lazy || m_at_most(sc, tot_w, t, rr, &okm, avx2) {
                        if !have_m { mm = median_m(&sc.w, &sc.s, rr, ok_r, &mut sc.v, &mut sc.wv).0; have_m = true; }
                        if key == Key::M0 && !have_f { mf = median_floor(&sc.w, rr, ok_r, &mut sc.f, &mut sc.wf); have_f = true; }
                        let k = match key { Key::M0 => if mm > mf { mm } else { f64::INFINITY }, Key::Mj => mm + (base + col as f64 * 1e-12) };
                        if k.is_finite() {
                            if key == Key::M0 && !lazy { if dir == 0 { own_nfin[ri] += 1; } else { c_nfin += 1; } }
                            if k <= t {
                                if !have_f { mf = median_floor(&sc.w, rr, ok_r, &mut sc.f, &mut sc.wf); have_f = true; }
                                let mut st = states[row].lock().unwrap();
                                let t2 = st.list_event(cmax, Ent { key: k, c: col as u32, m: mm, mfl: mf });
                                thr[row].store(t2.to_bits(), Relaxed); drop(st);
                                if dir == 0 { lthr[ri] = t2; } else { cthr = t2; }
                            }
                        }
                    }
                    if want_closest {
                        let lim: &mut [f64] = if dir == 0 { &mut llim[ri * nloc..(ri + 1) * nloc] } else { &mut clim };
                        if band_hit(&sc.a, &sc.om, lim, avx2) {
                            if !have_m { mm = median_m(&sc.w, &sc.s, rr, ok_r, &mut sc.v, &mut sc.wv).0; have_m = true; }
                            let mj = mm + (base + col as f64 * 1e-12);
                            let mut st = states[row].lock().unwrap();
                            for li in 0..nloc {
                                // d = a / om is above the limit for certain when a is above limit (1 + 1e-12) om
                                if !(sc.a[li] <= lim[li] * sc.om[li]) { continue; }
                                let l1 = st.band_event(li, sc.a[li] / sc.om[li], mj, col) * (1.0 + 1e-12);
                                lim[li] = l1; lim1[row * nloc + li].store(l1.to_bits(), Relaxed);
                            }
                        }
                    }
                }
            }
            if c_nfin != 0 { nfin[c].fetch_add(c_nfin, Relaxed); }
        }
        for ri in 0..nr { if own_nfin[ri] != 0 { nfin[a + ri].fetch_add(own_nfin[ri], Relaxed); } }
        if stage.is_some() && total_pairs > 0 {
            let mine: u64 = (a..b).map(|g| (n - 1 - g) as u64).sum();
            let d = done_pairs.fetch_add(mine, Relaxed) + mine;
            let want = (n as u128 * d as u128 / total_pairs as u128) as u64; let had = ticked.fetch_max(want, Relaxed);
            if want > had { prog.tick_k((want - had) as usize, None); }
        }
        (ident, counters)
    }).collect();
    if stage.is_some() { prog.finish(); }
    let mut out = SweepOut { rows: vec![], cl_win: vec![], cl_dmin: vec![], ident: vec![], counters: (0, 0, 0), full: None, overflow: 0 };
    for (id, cn) in parts { out.ident.extend(id); out.counters.0 += cn.0; out.counters.1 += cn.1; out.counters.2 += cn.2; }
    out.ident.sort_unstable();
    // each row's list in order, and the closest match at every locus from its bands
    let count_all = !(lazy && key == Key::M0);            // a lazy M0 sweep does not count every finite key: its nfin is the list's length
    let rows: Vec<(RowList, Vec<i64>, Vec<f64>, i64)> = states.into_par_iter().enumerate().map(|(g, st)| {
        let st = st.into_inner().unwrap();
        let (cl_win, cl_dmin, overflow) = if want_closest {
            let (mut win, mut dmin, mut over) = (vec![-1i64; nloc], vec![f64::INFINITY; nloc], 0i64);
            for li in 0..nloc {
                let b = st.band(li); let kept = (b.n as usize).min(BAND_SLOTS);
                if b.n as usize > BAND_SLOTS { over += 1; }
                if kept == 0 { continue; }
                // closest_winners: the (vals, column)-smallest of the kept entries
                let mut best = 0usize;
                for j in 1..kept { if b.e[j].1 < b.e[best].1 || (b.e[j].1 == b.e[best].1 && b.e[j].2 < b.e[best].2) { best = j; } }
                win[li] = b.e[best].2; dmin[li] = b.e[0].0;
            }
            (win, dmin, over)
        } else { (vec![], vec![], 0) };
        let e = st.heap.into_sorted_vec();
        let list = RowList { cand: e.iter().map(|x| x.c as i64).collect(), key: e.iter().map(|x| x.key).collect(), m: e.iter().map(|x| x.m).collect(), mfl: e.iter().map(|x| x.mfl).collect(),
                             nfin: if count_all { nfin[g].load(Relaxed) } else { e.len() as i64 } };
        (list, cl_win, cl_dmin, overflow)
    }).collect();
    out.rows.reserve(n); if want_closest { out.cl_win.reserve(n * nloc); out.cl_dmin.reserve(n * nloc); }
    for (list, w, d, o) in rows { out.rows.push(list); out.cl_win.extend(w); out.cl_dmin.extend(d); out.overflow += o; }
    out.full = full.map(|(m, f)| (m.0.into_inner(), f.0.into_inner()));
    out
}

/// each taxon's class of identical genomes: the lowest index among the genomes it is linked to
pub fn classes_from_pairs(n: usize, pairs: &[(usize, usize)]) -> Vec<usize> {
    let mut cls: Vec<usize> = (0..n).collect();
    loop {
        let mut new = cls.clone();
        for &(g, c) in pairs { if cls[c] < new[g] { new[g] = cls[c]; } }
        for &(g, c) in pairs { if cls[g] < new[c] { new[c] = cls[g]; } }
        if new == cls { return cls; }
        cls = new;
    }
}

/// a letter's residue code (-1: not a residue), each locus's offset in a row laid out by locus, and the warning when
/// many letters are ambiguity codes
pub fn alphabet_lut(rows: &[u8], dtype: &str, loci: &Loci) -> ([i8; 256], Vec<usize>) {
    let alphabet = if dtype == "dna" { crate::input::DNA } else { crate::input::PROTEIN };
    let mut lut = [-1i8; 256];
    for (i, c) in alphabet.bytes().enumerate() { lut[c as usize] = i as i8; lut[c.to_ascii_lowercase() as usize] = i as i8; }
    if dtype == "dna" { lut[b'U' as usize] = 3; lut[b'u' as usize] = 3; }
    let letters = crate::input::byte_counts(rows);
    let (mut n_letters, mut n_in) = (0i64, 0i64);
    for c in (65..91).chain(97..123) { n_letters += letters[c]; if lut[c] >= 0 { n_in += letters[c]; } }
    if n_letters != 0 && (n_in as f64) < 0.95 * n_letters as f64 {
        log!("WARNING: {} of the {} letters ({:.1}%) are ambiguity codes and count as missing data", commas(n_letters - n_in), commas(n_letters), 100.0 * (n_letters - n_in) as f64 / n_letters as f64);
    }
    let mut off = Vec::with_capacity(loci.len() + 1); let mut acc = 0; for (_, st) in loci { off.push(acc); acc += st.len(); } off.push(acc);
    (lut, off)
}

/// each locus's plateau from how often each residue occurs there (counts: nloc x a_)
pub fn locus_plateaus(counts: &[u64], nloc: usize, a_: usize) -> Vec<f64> {
    (0..nloc).map(|li| {
        let cntf: Vec<f64> = counts[li * a_..(li + 1) * a_].iter().map(|c| *c as f64).collect();
        let tot = pymax(pairwise_sum(&cntf), 1.0);
        let f2: Vec<f64> = cntf.iter().map(|c| { let f = c / tot; f * f }).collect();
        pymax(100.0 * (1.0 - pairwise_sum(&f2)), 1.0)
    }).collect()
}

impl Prep {
    #[inline] pub fn n(&self) -> usize { self.core.n }
    /// m and its floor for a pair: from the full matrix when it is kept, else by the kernel on the pair's counts (NaN for a taxon with itself)
    pub fn pair_m(&self, g: usize, c: usize) -> (f64, f64) {
        if let (Some(m), Some(f)) = (&self.m_full, &self.mfl_full) { return (m[g * self.core.n + c], f[g * self.core.n + c]); }
        if g == c { return (f64::NAN, f64::NAN); }
        thread_local! { static SC: std::cell::RefCell<Option<Scratch>> = const { std::cell::RefCell::new(None) }; }
        SC.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.as_ref().map_or(true, |sc| sc.w.len() != self.core.nloc || sc.pa.len() != self.core.bits.wpr) { *slot = Some(Scratch::new(&self.core)); }
            let sc = slot.as_mut().unwrap();
            self.core.pair_counts(g, c, &mut sc.w, &mut sc.s, &mut sc.pa, &mut sc.eq);
            let (m, mf, _) = pair_medians(&sc.w, &sc.s, Some(&self.rr), Some(&self.ok_r), &mut sc.v, &mut sc.wv, &mut sc.f, &mut sc.wf);
            (m, mf)
        })
    }
    /// m and its floor for a comparator pair, from the focal taxon's list (or its extended row, or the full matrix)
    pub fn pair_m_lists(&self, g: usize, c: usize) -> (f64, f64) {
        if let (Some(m), Some(f)) = (&self.m_full, &self.mfl_full) { return (m[g * self.core.n + c], f[g * self.core.n + c]); }
        let l = &self.lists[g];
        if let Some(p) = l.cand.iter().position(|x| *x == c as i64) { return (l.m[p], l.mfl[p]); }
        if let Some(e) = self.ext.get(&g) { if let Some(p) = e.cand.iter().position(|x| *x == c as i64) { return (e.m[p], e.mfl[p]); } }
        (f64::NAN, f64::NAN)
    }
    /// m for any pair: from either taxon's list or extended row when the pair is there (the value the sweep computed:
    /// the same bits), else by the kernel on the pair's counts, a read of both taxa's planes
    pub fn pair_m_any(&self, g: usize, c: usize) -> f64 {
        if let Some(m) = &self.m_full { return m[g * self.core.n + c]; }
        for (a, b) in [(g, c), (c, g)] {
            let l = &self.lists[a];
            if let Some(p) = l.cand.iter().position(|x| *x == b as i64) { return l.m[p]; }
            if let Some(e) = self.ext.get(&a) { if let Some(p) = e.cand.iter().position(|x| *x == b as i64) { return e.m[p]; } }
        }
        self.pair_m_only(g, c)
    }
    /// m alone for a pair by the kernel (pair_m's m, without the floor's median)
    fn pair_m_only(&self, g: usize, c: usize) -> f64 {
        if g == c { return f64::NAN; }
        thread_local! { static SC: std::cell::RefCell<Option<Scratch>> = const { std::cell::RefCell::new(None) }; }
        SC.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.as_ref().map_or(true, |sc| sc.w.len() != self.core.nloc || sc.pa.len() != self.core.bits.wpr) { *slot = Some(Scratch::new(&self.core)); }
            let sc = slot.as_mut().unwrap();
            self.core.pair_counts(g, c, &mut sc.w, &mut sc.s, &mut sc.pa, &mut sc.eq);
            median_m(&sc.w, &sc.s, Some(&self.rr), Some(&self.ok_r), &mut sc.v, &mut sc.wv).0
        })
    }
    /// the candidates of a taxon in order of pair divergence: its extended row when it has one, else its list
    pub fn row_cands(&self, g: usize) -> &[i64] { match self.ext.get(&g) { Some(e) => &e.cand, None => &self.lists[g].cand } }
}

pub fn prepare(a: &Args, names: &[String], n: usize, l: usize, rows: &[u8], loci: &Loci, dtype: &str, k_eff: usize, t0: std::time::Instant) -> Prep {
    let a_ = if dtype == "dna" { 4 } else { 20 }; let nloc = loci.len();
    let (lut, off) = alphabet_lut(rows, dtype, loci);
    let (bits, counts) = Bits::from_rows(n, l, rows, &lut, loci, &off, a_);
    let mut core = Core { n, nloc, off, keep: vec![true; n], slab: None, bits };
    let mut focal_sites = vec![0u32; n * nloc];
    focal_sites.par_chunks_mut(nloc).enumerate().for_each(|(g, fs)| { for li in 0..nloc { fs[li] = core.bits.present(g, li); } });
    let present: Vec<i64> = (0..n).map(|g| focal_sites[g * nloc..(g + 1) * nloc].iter().filter(|x| **x as i64 >= a.fit_overlap).count() as i64).collect();
    let thin: Vec<usize> = (0..n).filter(|&g| present[g] < MIN_LOCI_PER_TAXON).collect();
    for &g in &thin { core.keep[g] = false; }
    if !thin.is_empty() {
        log!("{} taxa with fewer than {} loci present are excluded from the test and left unmasked: {}{}", thin.len(), MIN_LOCI_PER_TAXON,
             thin.iter().take(20).map(|&i| names[i].as_str()).collect::<Vec<_>>().join(", "),
             if thin.len() > 20 { format!(" and {} more (the report lists them)", thin.len() - 20) } else { String::new() });
    }
    let plateaus = locus_plateaus(&counts, nloc, a_); drop(counts);
    let longest = loci.iter().map(|x| x.1.len()).max().unwrap_or(0);
    assert!(longest < (1 << 24), "a locus of 2^24 sites or more: the float32 products would not be exact");
    let c = a.candidates.unwrap_or(CANDIDATES).max(1);
    let full = n <= a.full_m_max_taxa;
    log!("pair sweep: rows of the pair matrix in {} thread{}, {} candidates per taxon, the full m matrix {}", a.threads, if a.threads > 1 { "s" } else { "" }, c, if full { "kept" } else { "not kept" });
    let kn = k_eff.min(n - 1);
    if n <= a.sweep_block_min {
        core.build_slab();
        let bytes = core.slab.as_ref().map(|s| 8 * s.w.len()).unwrap_or(0);
        log!("the pair counts of every locus ({}) are kept for the run", crate::fmem(bytes as u64));
    }
    let header = crate::restart::header(a, n, nloc, l, dtype, kn, c, &thin, full);
    let mut pair_names = vec!["ident", "rate", "ok_r", "cand", "key", "m", "mfl", "nfin", "cl_win", "cl_dmin", "counters"]; if full { pair_names.extend(["m_full", "mfl_full"]); }
    let pt2 = crate::restart::read(a, "pairs", &header, &pair_names);
    let pt1 = if pt2.is_none() { crate::restart::read(a, "rates", &header, &["ident", "rate", "ok_r"]) } else { None };
    let c_len = c.min(n);
    let mut restart = None; let mut n_starved = 0usize; let mut bs_pre: Option<BsPools> = None;
    let (ident, mut rate, ok_r): (Vec<(usize, usize)>, Vec<f64>, Vec<bool>);
    let from_point = pt2.as_ref().or(pt1.as_ref());
    if let Some(pt) = from_point {
        restart = Some(((if pt2.is_some() { "the pair sweeps" } else { "the locus rates" }).to_string(), pt.written.clone()));
        ident = pt.i64s("ident").chunks_exact(2).map(|x| (x[0] as usize, x[1] as usize)).collect(); rate = pt.f64s("rate"); ok_r = pt.bools("ok_r");
    } else {
        // ---- sweep 1: the raw divergences, for the rate-vote neighbours
        let s1 = sweep(&core, None, None, Key::M0, c, a.fit_overlap, false, false, None, Some("pairs0"));
        ident = s1.ident.clone();
        let cls0 = classes_from_pairs(n, &ident);
        let rep: Vec<bool> = (0..n).map(|i| cls0[i] == i).collect();
        // the locus rates, voted by each taxon's nearest resolved non-identical neighbours
        let neighbours = |list: &RowList, g: usize, out: &mut Vec<(usize, usize, f64)>| -> bool {
            let mut picked = 0;
            for (j, &cd) in list.cand.iter().enumerate() {
                if picked == kn { break; }
                if rep[cd as usize] { out.push((g, cd as usize, npmax(list.m[j], list.mfl[j]))); picked += 1; }
            }
            picked < kn
        };
        // a taxon whose list ran out before Kn representatives although it has more candidates than the list holds gets
        // its full row. The sweep does not count every candidate (it skips the pairs that cannot enter the list), so
        // the rows that are short with a full list are swept in full: more entries than the list means it ran out
        let mut votes: Vec<(usize, usize, f64)> = vec![]; let mut short_full: Vec<usize> = vec![]; let mut spans: Vec<(usize, usize)> = vec![];
        for g in 0..n {
            let before = votes.len();
            let short = neighbours(&s1.rows[g], g, &mut votes);
            if short && s1.rows[g].cand.len() >= c_len { short_full.push(g); spans.push((before, votes.len())); }
        }
        if !short_full.is_empty() {
            let e = sweep(&core, None, None, Key::M0, n, a.fit_overlap, false, false, Some(&short_full), None);
            let mut drop = vec![false; votes.len()]; let mut extra: Vec<(usize, usize, f64)> = vec![];
            for (k, &g) in short_full.iter().enumerate() {
                if e.rows[k].cand.len() > c_len { n_starved += 1; for d in drop[spans[k].0..spans[k].1].iter_mut() { *d = true; } neighbours(&e.rows[k], g, &mut extra); }
            }
            if n_starved > 0 {
                log!("rate neighbours: {} taxa ran out of candidates (copies of one genome fill the list): full rows computed", n_starved);
                votes = votes.into_iter().zip(drop).filter(|(_, d)| !d).map(|(v, _)| v).collect(); votes.extend(extra);
            }
        }
        drop(s1);
        let prog = crate::log::prog();
        prog.start("rates", nloc);
        let fit_f = a.fit_overlap as f32;
        rate = (0..nloc).into_par_iter().map(|li| {
            let p_ = plateaus[li];
            let corr = |p: f64| -> f64 { -p_ * (1.0 - npmin(p, 0.98 * p_) / p_).ln() };
            let mut vals: Vec<f64> = Vec::with_capacity(votes.len());
            for &(g, cc, m0) in &votes {
                let (o, s) = core.at(li, g, cc); let (o_, s_) = (o as f32, s as f32);
                let d_ = if o_ > 0.0 { 100.0f32 * (1.0f32 - s_ / o_.max(1.0)) } else { f32::NAN };
                if d_.is_finite() && o_ >= fit_f { vals.push(corr((d_ + 50.0f32 / o_) as f64) / corr(m0)); }
            }
            prog.tick();
            if vals.len() < MIN_FIT_PAIRS { f64::NAN } else { median(&vals) }
        }).collect();
        prog.finish();
        let lsites: Vec<f64> = loci.iter().map(|x| x.1.len() as f64).collect();
        ok_r = rate.iter().map(|r| r.is_finite() && *r > 0.0).collect();
        if ok_r.iter().filter(|x| **x).count() >= 3 {
            let (rv, lv): (Vec<f64>, Vec<f64>) = (0..nloc).filter(|&i| ok_r[i]).map(|i| (rate[i], lsites[i])).unzip();
            let scale = wmedian_sorted(&rv, &lv);
            let div = pymax(scale, 1e-6);
            for r in rate.iter_mut() { *r /= div; }
        }
    }
    let cls = classes_from_pairs(n, &ident); let has_dup = !ident.is_empty();
    if has_dup {
        log!("{} genomes are identical to another at every shared site: an identical genome is never a comparator, and no two comparators are identical to each other", (0..n).filter(|&i| cls[i] != i).count());
    }
    if restart.is_none() { log!("per-locus identities done, {:.0}s", t0.elapsed().as_secs_f64()); }
    let rr: Vec<f64> = (0..nloc).map(|i| if ok_r[i] { rate[i] } else { f64::NAN }).collect();
    let ident_flat: Vec<i64> = ident.iter().flat_map(|x| [x.0 as i64, x.1 as i64]).collect();
    use crate::restart::Arr;
    if restart.is_none() { crate::restart::write(a, "rates", &header, &[("ident", Arr::I64(&ident_flat, vec![ident.len(), 2])), ("rate", Arr::F64(&rate, vec![nloc])), ("ok_r", Arr::Bool(&ok_r))]); }
    let s2 = if let Some(pt) = &pt2 {
        let (cand, key, m, mfl, nfin) = (pt.i64s("cand"), pt.f64s("key"), pt.f64s("m"), pt.f64s("mfl"), pt.i64s("nfin"));
        let rows = (0..n).map(|g| { let k = (0..c_len).take_while(|&j| cand[g * c_len + j] >= 0).count(); let r = g * c_len..g * c_len + k;
            RowList { cand: cand[r.clone()].to_vec(), key: key[r.clone()].to_vec(), m: m[r.clone()].to_vec(), mfl: mfl[r].to_vec(), nfin: nfin[g] } }).collect();
        let cn = pt.i64s("counters");
        SweepOut { rows, cl_win: pt.i64s("cl_win"), cl_dmin: pt.f64s("cl_dmin"), ident: vec![], counters: (cn[0], cn[1], cn[2]), full: if full { Some((pt.f64s("m_full"), pt.f64s("mfl_full"))) } else { None }, overflow: 0 }
    } else {
        // ---- sweep 2: the rate-normalised divergences, the lists, the closest match at every locus
        // the broad-sample pools' pair divergences come out of this sweep when the full matrix is not kept: the pairs
        // go by anyway (computed afterwards they are a read of a column from memory each)
        let pools = match bs_sizes(n, k_eff) {
            Some((k, pn)) if !a.no_broad_sample && !full && std::env::var_os("LOCUSMASKER_NO_PAIRS_ONCE").is_none() => { let pool = bs_pools(n, pn); Some((k, PoolIndex::new(n, pn, &pool), pool)) }
            _ => None,
        };
        let s2 = match &pools {
            Some((_, ix, _)) => sweep_pairs_once(&core, Some(&rr), Some(&ok_r), Key::Mj, c, a.fit_overlap, true, full, Some("pairs"), Some(ix)),
            None => sweep(&core, Some(&rr), Some(&ok_r), Key::Mj, c, a.fit_overlap, true, full, None, Some("pairs")),
        };
        bs_pre = pools.map(|(k, ix, pool)| BsPools { k, pn: ix.pn, pool, mstar: ix.into_mstar() });
        let pad_i = |f: &dyn Fn(&RowList) -> &Vec<i64>| -> Vec<i64> { let mut v = vec![-1i64; n * c_len]; for g in 0..n { let x = f(&s2.rows[g]); v[g * c_len..g * c_len + x.len()].copy_from_slice(x); } v };
        let pad_f = |f: &dyn Fn(&RowList) -> &Vec<f64>, fill: f64| -> Vec<f64> { let mut v = vec![fill; n * c_len]; for g in 0..n { let x = f(&s2.rows[g]); v[g * c_len..g * c_len + x.len()].copy_from_slice(x); } v };
        let (cand, key, m, mfl) = (pad_i(&|r| &r.cand), pad_f(&|r| &r.key, f64::INFINITY), pad_f(&|r| &r.m, f64::NAN), pad_f(&|r| &r.mfl, f64::NAN));
        let nfin: Vec<i64> = s2.rows.iter().map(|r| r.nfin).collect(); let cn = [s2.counters.0, s2.counters.1, s2.counters.2];
        let mut arrays = vec![("ident", Arr::I64(&ident_flat, vec![ident.len(), 2])), ("rate", Arr::F64(&rate, vec![nloc])), ("ok_r", Arr::Bool(&ok_r)),
                              ("cand", Arr::I64(&cand, vec![n, c_len])), ("key", Arr::F64(&key, vec![n, c_len])), ("m", Arr::F64(&m, vec![n, c_len])), ("mfl", Arr::F64(&mfl, vec![n, c_len])),
                              ("nfin", Arr::I64(&nfin, vec![n])), ("cl_win", Arr::I64(&s2.cl_win, vec![n, nloc])), ("cl_dmin", Arr::F64(&s2.cl_dmin, vec![n, nloc])), ("counters", Arr::I64(&cn, vec![3]))];
        if let Some((mf, ff)) = &s2.full { arrays.push(("m_full", Arr::F64(mf, vec![n, n]))); arrays.push(("mfl_full", Arr::F64(ff, vec![n, n]))); }
        crate::restart::write(a, "pairs", &header, &arrays);
        s2
    };
    let (nl_low, npk, fin_pairs) = s2.counters; let fb = if npk != 0 { nl_low as f64 / npk as f64 } else { 0.0 };
    let n_ok = ok_r.iter().filter(|x| **x).count();
    log!("pair divergences done, {:.0}s{}", t0.elapsed().as_secs_f64(),
         if n_ok == 0 { "; note: no locus has a rate (no nearest-neighbour pairs resolved beyond their resolution floor), so no pair has a divergence".to_string() }
         else if fb > 0.05 { format!("; note: {:.0}% of taxon pairs overlap in <{} loci with a rate and are never compared", 100.0 * fb, MIN_LOCI_FOR_M) } else { String::new() });
    if s2.overflow != 0 { log!("note: {} cells had more distinct closest-match rates within 1e-9 of the minimum than {} slots", commas(s2.overflow), BAND_SLOTS); }
    let (m_full, mfl_full) = match s2.full { Some((m, f)) => (Some(m), Some(f)), None => (None, None) };
    Prep { core, thin, present, focal_sites, plateaus, rate, rr, ok_r, cls, has_dup, lists: s2.rows, c_len, cl_win: s2.cl_win, cl_dmin: s2.cl_dmin,
           m_full, mfl_full, fin_pairs, ext: HashMap::new(), n_starved, bs_pre, restart }
}

// ------------------------------------------------------------------ the broad-sample pools
pub struct BsPools { pub k: usize, pub pn: usize, pub pool: Vec<i32>, pub mstar: Vec<f64> }      // pool, mstar: n x pn

pub fn bs_keys_taxa(i: usize) -> u64 { sm64((i as u64 + 1) ^ BS_SEED) }
pub fn bs_keys_cell(g: usize, li: usize) -> u64 {
    let mix = (li as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
    sm64(sm64((g as u64).wrapping_add(1)) ^ mix ^ BS_SEED)
}

/// the members drawn per sequence and the size of each taxon's pool, or None when there are too few taxa for the test
pub fn bs_sizes(n: usize, k_eff: usize) -> Option<(usize, usize)> {
    let k = (BS_PER_COMP * k_eff).min(n - 1); let pn = (BS_POOL * k_eff).min(n - 1);
    if (k as i64) < BS_MIN { None } else { Some((k, pn)) }
}
/// a taxon's pool, as drawn: pn of the other taxa
pub fn bs_pool_of(g: usize, n: usize, pn: usize) -> Vec<i64> {
    let mut pool: Vec<i64> = if pn >= n - 1 { (0..(n - 1) as i64).collect() } else { hash_draw(bs_keys_taxa(g), n - 1, pn) };
    for v in pool.iter_mut() { if *v >= g as i64 { *v += 1; } }
    pool
}
/// every taxon's pool (n x pn)
fn bs_pools(n: usize, pn: usize) -> Vec<i32> {
    let mut pool = vec![0i32; n * pn];
    if pn > 0 { pool.par_chunks_mut(pn).enumerate().for_each(|(g, row)| { for (x, v) in row.iter_mut().zip(bs_pool_of(g, n, pn)) { *x = v as i32; } }); }
    pool
}

/// The pools and their pair divergences. Sweep 2 has left them when it ran without the full matrix (Prep::bs_pre);
/// otherwise they are read from the full matrix, or computed pair by pair (a run that starts after the sweeps)
pub fn bs_prepare(p: &mut Prep, k_eff: usize) -> Option<BsPools> {
    let n = p.n();
    let (k, pn) = bs_sizes(n, k_eff)?;
    if let Some(b) = p.bs_pre.take() { if b.k == k && b.pn == pn { return Some(b); } }
    let p: &Prep = p;
    let prog = crate::log::prog(); let on_demand = p.m_full.is_none();
    if on_demand { prog.start("broad-sample pool m", n); }
    let pool = bs_pools(n, pn);
    let mut mstar = vec![f64::NAN; n * pn];
    if pn > 0 {
        mstar.par_chunks_mut(pn).enumerate().for_each(|(g, ms)| {
            for (j, m) in ms.iter_mut().enumerate() { let (m_, f_) = p.pair_m(g, pool[g * pn + j] as usize); *m = if f_.is_finite() { m_.max(f_) } else { m_ }; }
            if on_demand { prog.tick(); }
        });
    }
    if on_demand { prog.finish(); }
    Some(BsPools { k, pn, pool, mstar })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a small random alignment with relatives, gaps, identical genomes and a taxon with almost nothing
    fn random_core(seed: u64, n: usize, nloc: usize, a_: usize) -> Core {
        let mut x = seed; let mut rnd = move || { x = sm64(x.wrapping_add(0x9E3779B97F4A7C15)); x };
        let lens: Vec<usize> = (0..nloc).map(|_| 20 + (rnd() % 150) as usize).collect();
        let mut off = vec![0usize]; for l in &lens { off.push(off.last().unwrap() + l); }
        let lt = *off.last().unwrap(); let mut xl = vec![-1i8; n * lt];
        for g in 0..n {
            let parent = if g > 0 && rnd() % 4 != 0 { Some((rnd() % g as u64) as usize) } else { None };
            let rate = [0u64, 1, 3, 10, 40][(rnd() % 5) as usize];           // percent of sites that differ from the parent
            for k in 0..lt {
                let v = match parent { Some(p) if rnd() % 100 >= rate => xl[p * lt + k], _ => (rnd() % a_ as u64) as i8 };
                xl[g * lt + k] = if rnd() % 12 == 0 { -1 } else { v };
            }
            if g % 17 == 5 { for li in 0..nloc { if rnd() % 3 != 0 { for k in off[li]..off[li + 1] { xl[g * lt + k] = -1; } } } }
        }
        for k in 0..lt { let v = xl[k]; xl[(n - 1) * lt + k] = v; }      // the last genome is a copy of the first
        let bits = Bits::build(n, &xl, lt, &off, a_);
        // a locus's gathered planes give the counts and the codes of the whole planes
        for li in 0..nloc.min(4) { let lb = bits.locus(li, n); let len = off[li + 1] - off[li];
            for f in (0..n).step_by(5) { for c in (1..n).step_by(11) { for t in (2..n).step_by(13) {
                let (xf, xc, xt) = (lb.seq(f, len), lb.seq(c, len), lb.seq(t, len)); let (mut common, mut dt, mut dc) = (0u32, 0u32, 0u32);
                for s in 0..len { if xf[s] >= 0 && xc[s] >= 0 && xt[s] >= 0 { common += 1; dt += (xt[s] != xf[s]) as u32; dc += (xc[s] != xf[s]) as u32; } }
                assert_eq!(lb.three(f, c, t), (common, dt, dc));
            } } } }
        for li in 0..nloc { let lb = bits.locus(li, n); for g in 0..n { assert_eq!(lb.seq(g, off[li + 1] - off[li]), bits.seq(g, li, off[li + 1] - off[li])); for c in (0..n).step_by(7) { assert_eq!(lb.count(g, c), bits.count(g, c, li)); } } }
        // the planes hold the codes: what is read back is what went in
        for g in 0..n { for li in 0..nloc { assert_eq!(bits.seq(g, li, off[li + 1] - off[li]), &xl[g * lt + off[li]..g * lt + off[li + 1]]); assert_eq!(bits.present(g, li) as usize, xl[g * lt + off[li]..g * lt + off[li + 1]].iter().filter(|v| **v >= 0).count()); } }
        Core { n, nloc, off, keep: (0..n).map(|g| g % 23 != 7).collect(), slab: None, bits }
    }

    fn same(a: &SweepOut, b: &SweepOut, nfin: bool) {
        assert_eq!(a.rows.len(), b.rows.len());
        for (g, (x, y)) in a.rows.iter().zip(&b.rows).enumerate() {
            assert_eq!(x.cand, y.cand, "row {g}: candidates");
            let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<u64>>();
            assert_eq!(bits(&x.key), bits(&y.key), "row {g}: keys"); assert_eq!(bits(&x.m), bits(&y.m), "row {g}: m"); assert_eq!(bits(&x.mfl), bits(&y.mfl), "row {g}: floors");
            if nfin { assert_eq!(x.nfin, y.nfin, "row {g}: nfin"); }
        }
        assert_eq!(a.cl_win, b.cl_win); assert_eq!(a.cl_dmin.iter().map(|f| f.to_bits()).collect::<Vec<_>>(), b.cl_dmin.iter().map(|f| f.to_bits()).collect::<Vec<_>>());
        assert_eq!(a.ident, b.ident); assert_eq!(a.counters, b.counters); assert_eq!(a.overflow, b.overflow);
        match (&a.full, &b.full) {
            (Some((m1, f1)), Some((m2, f2))) => { assert!(m1.iter().zip(m2).all(|(p, q)| p.to_bits() == q.to_bits())); assert!(f1.iter().zip(f2).all(|(p, q)| p.to_bits() == q.to_bits())); }
            (None, None) => {} _ => panic!("one sweep kept the full matrix"),
        }
    }

    /// the sweep that computes each pair once gives what the row-by-row sweep gives: lists, closest matches, identical
    /// pairs, counters and the full matrix, in each of its modes
    #[test]
    fn pairs_once_is_the_row_sweep() {
        for (seed, n, nloc, a_, cmax) in [(1u64, 90usize, 14usize, 20usize, 7usize), (2, 61, 9, 4, 200), (3, 140, 23, 20, 12), (4, 33, 12, 4, 3)] {
            let core = random_core(seed, n, nloc, a_);
            let mut x = seed ^ 99; let rr: Vec<f64> = (0..nloc).map(|_| { x = sm64(x); 0.3 + (x % 1000) as f64 / 400.0 }).collect();
            let ok: Vec<bool> = (0..nloc).map(|l| l % 5 != 3).collect();
            // sweep 1 (the lazy M0 sweep's nfin is not compared: neither sweep counts every key there)
            same(&sweep_rows(&core, None, None, Key::M0, cmax, 20, false, false, None, None), &sweep_pairs_once(&core, None, None, Key::M0, cmax, 20, false, false, None, None), false);
            // sweep 2, the medians where they are used
            same(&sweep_rows(&core, Some(&rr), Some(&ok), Key::Mj, cmax, 20, true, false, None, None), &sweep_pairs_once(&core, Some(&rr), Some(&ok), Key::Mj, cmax, 20, true, false, None, None), true);
            // sweep 2 with the full matrix
            same(&sweep_rows(&core, Some(&rr), Some(&ok), Key::Mj, cmax, 20, true, true, None, None), &sweep_pairs_once(&core, Some(&rr), Some(&ok), Key::Mj, cmax, 20, true, true, None, None), true);
        }
    }

    /// the pools' pair divergences that sweep 2 leaves are those computed pair by pair, and the sweep's other results
    /// are what they are without the pools
    #[test]
    fn pools_from_the_sweep_are_pair_m() {
        for (seed, n, nloc, pn) in [(6u64, 70usize, 12usize, 9usize), (7, 45, 10, 44), (8, 120, 15, 30)] {
            let core = random_core(seed, n, nloc, 20);
            let rr: Vec<f64> = (0..nloc).map(|l| 0.5 + l as f64 / 9.0).collect(); let ok: Vec<bool> = (0..nloc).map(|l| l % 4 != 1).collect();
            let pool = bs_pools(n, pn); let ix = PoolIndex::new(n, pn, &pool);
            let with = sweep_pairs_once(&core, Some(&rr), Some(&ok), Key::Mj, 8, 20, true, false, None, Some(&ix));
            same(&with, &sweep_pairs_once(&core, Some(&rr), Some(&ok), Key::Mj, 8, 20, true, false, None, None), true);
            let got = ix.into_mstar(); let mut sc = Scratch::new(&core);
            for g in 0..n { for j in 0..pn {
                let c = pool[g * pn + j] as usize; assert_ne!(c, g);
                core.pair_counts(g, c, &mut sc.w, &mut sc.s, &mut sc.pa, &mut sc.eq);
                let (m, f, _) = pair_medians(&sc.w, &sc.s, Some(&rr), Some(&ok), &mut sc.v, &mut sc.wv, &mut sc.f, &mut sc.wf);
                let want = if f.is_finite() { m.max(f) } else { m };
                assert!(got[g * pn + j].to_bits() == want.to_bits() || (got[g * pn + j].is_nan() && want.is_nan()), "taxon {g}, pool member {c}: {} and {}", got[g * pn + j], want);
            } }
        }
    }

    /// the shared per-pair values eight loci at a time are those of the scalar code, bit for bit
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn derive_avx2_is_derive_scalar() {
        if !std::is_x86_feature_detected!("avx2") { return; }
        let mut x = 7u64;
        for nloc in [1usize, 7, 8, 9, 53, 120] {
            for _ in 0..200 {
                let w: Vec<u32> = (0..nloc).map(|_| { x = sm64(x); if x % 7 == 0 { 0 } else { (x >> 8) as u32 % 900 } }).collect();
                let s: Vec<u32> = w.iter().map(|&wv| { x = sm64(x); if wv == 0 { 0 } else { (x >> 8) as u32 % (wv + 1) } }).collect();
                let q: Vec<f64> = (0..nloc).map(|_| { x = sm64(x); 0.2 + (x % 1000) as f64 / 300.0 }).collect();
                let okm: Vec<i32> = (0..nloc).map(|_| { x = sm64(x); if x % 6 == 0 { 0 } else { -1 } }).collect();
                let mk = || (vec![0.0f64; nloc], vec![0u64; nloc], vec![0.0f64; nloc], vec![0.0f64; nloc]);
                let (mut z1, mut t1, mut a1, mut o1) = mk(); let (mut z2, mut t2, mut a2, mut o2) = mk();
                let r1 = derive_scalar(&w, &s, &q, &okm, 20.0, true, &mut z1, &mut t1, &mut a1, &mut o1, 0);
                let r2 = unsafe { derive_avx2(&w, &s, &q, &okm, 20.0, true, &mut z2, &mut t2, &mut a2, &mut o2) };
                let b = |v: &[f64]| v.iter().map(|f| if f.is_nan() { u64::MAX } else { f.to_bits() }).collect::<Vec<u64>>();
                assert_eq!(r1, r2); assert_eq!(b(&z1), b(&z2)); assert_eq!(t1, t2); assert_eq!(b(&a1), b(&a2)); assert_eq!(b(&o1), b(&o2));
                for thr in [0.0, 3.0, 40.0, f64::INFINITY, z1.iter().copied().find(|v| v.is_finite()).unwrap_or(1.0)] {
                    let (lo, hi) = (thr * (1.0 - 1e-12), thr * (1.0 + 1e-12));
                    assert_eq!(below_scalar(&z1, &t1, lo, hi), unsafe { below_avx2(&z1, &t1, lo, hi) });
                }
                let lim: Vec<f64> = (0..nloc).map(|_| { x = sm64(x); if x % 3 == 0 { f64::INFINITY } else { (x % 100) as f64 / 400.0 } }).collect();
                assert_eq!(band_hit_scalar(&a1, &o1, &lim), unsafe { band_hit_avx2(&a1, &o1, &lim) });
            }
        }
    }

    /// m_at_most says whether the weighted median is at most the threshold, also when the threshold is a locus's value
    #[test]
    fn m_at_most_is_the_median_test() {
        let core = random_core(5, 40, 16, 20); let nloc = core.nloc; let mut sc = Scratch::new(&core);
        let rr: Vec<f64> = (0..nloc).map(|l| 0.4 + l as f64 / 7.0).collect(); let ok: Vec<bool> = (0..nloc).map(|l| l != 2).collect();
        let okm: Vec<i32> = ok.iter().map(|o| if *o { -1 } else { 0 }).collect(); let q: Vec<f64> = (0..nloc).map(|l| 1.0 / rr[l]).collect();
        for g in 0..core.n { for c in g + 1..core.n {
            core.pair_counts(g, c, &mut sc.w, &mut sc.s, &mut sc.pa, &mut sc.eq);
            let (w, s) = (sc.w.clone(), sc.s.clone());
            let (m, cnt) = median_m(&w, &s, Some(&rr), Some(&ok), &mut sc.v, &mut sc.wv);
            let (tot, cnt2) = derive(&w, &s, &q, &okm, 20.0, false, &mut sc.z, &mut sc.wt, &mut sc.a, &mut sc.om, core.bits.has_avx2());
            assert_eq!(cnt, cnt2);
            if cnt < MIN_LOCI_FOR_M { continue; }
            for thr in [m, m * (1.0 - 1e-15), m * (1.0 + 1e-15), m / 2.0, m * 2.0, 0.0, f64::INFINITY] {
                assert_eq!(m_at_most(&sc, tot, thr, Some(&rr), &okm, core.bits.has_avx2()), m <= thr, "pair ({g}, {c}), m {m}, threshold {thr}");
            }
        } }
    }

    /// a row's bands held one entry in line and the rest aside are the bands of band_insert, whatever the order
    #[test]
    fn row_state_bands_are_band_insert() {
        let mut x = 11u64;
        for _ in 0..300 {
            let k = 1 + (sm64(x) % 12) as usize; x = sm64(x);
            // rates that tie, nearly tie (within 1e-9) and differ
            let ev: Vec<(f64, f64, usize)> = (0..k).map(|i| { x = sm64(x); let d = [0.01, 0.01 + 2e-10, 0.01 + 4e-10, 0.01 + 6e-10, 0.01 + 8e-10, 0.01 + 9.5e-10, 0.02, 0.005][(x % 8) as usize]; x = sm64(x); (d, (x % 50) as f64, i) }).collect();
            let mut plain = BAND_EMPTY; let mut lim = f64::INFINITY;
            for &(d, mj, c) in &ev { if d <= lim { band_insert(&mut plain, &mut lim, d, mj, c as i64); } }
            let mut st = RowState { heap: std::collections::BinaryHeap::new(), bd: vec![0.0; 2], bv: vec![0.0; 2], bc: vec![B_NONE; 2], multi: vec![] };
            let mut order = ev.clone(); order.reverse();
            for &(d, mj, c) in &order { st.band_event(1, d, mj, c); }
            let b = st.band(1);
            assert_eq!(b.n.min(BAND_SLOTS as u8), plain.n.min(BAND_SLOTS as u8));
            let kept = (b.n as usize).min(BAND_SLOTS);
            let win = |b: &Band| (0..kept).map(|j| (b.e[j].1, b.e[j].2)).fold((f64::INFINITY, i64::MAX), |a, v| if v.0 < a.0 || (v.0 == a.0 && v.1 < a.1) { v } else { a });
            assert_eq!(b.e[0].0.to_bits(), plain.e[0].0.to_bits()); assert_eq!(win(&b), win(&plain));
            assert_eq!(st.bc[0], B_NONE);
        }
    }
}
