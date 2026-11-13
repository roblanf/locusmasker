//! The numerics of locusmasker.py, operation for operation: numpy's pairwise summation, medians, percentiles and
//! interpolation, the Lanczos log-gamma, the exact binomial / beta-binomial tails, the hashes behind the broad-sample
//! draws, and the weighted-median selection of the pair kernel. Each function mirrors the order of floating-point
//! operations of the numpy expression it replaces, because the outputs are compared to the bit.
use std::cmp::Ordering;

pub const HALF: f64 = 0.5;

/// numpy's pairwise summation of a contiguous float64 run (what np.sum does along the last axis)
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = -0.0f64;
        for &x in a { r += x; }
        return r;
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 { r[k] += a[i + k]; }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n { res += a[i]; i += 1; }
        return res;
    }
    let mut n2 = n / 2; n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// total order with NaN last, as numpy sorts
#[inline]
pub fn cmp_nan_last(a: &f64, b: &f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => a.partial_cmp(b).unwrap(),
        (true, true) => Ordering::Equal, (true, false) => Ordering::Greater, (false, true) => Ordering::Less,
    }
}
pub fn sort_f64(v: &mut [f64]) { v.sort_by(cmp_nan_last); }

/// np.median of a 1-D array: NaN when any element is NaN or the array is empty
pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() || v.iter().any(|x| x.is_nan()) { return f64::NAN; }
    let mut s = v.to_vec(); sort_f64(&mut s);
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// the k-th smallest of s (which is rearranged) and the smallest above it: the order statistics k and k + 1
fn two_order_stats(s: &mut [f64], k: usize) -> (f64, f64) {
    let (_, a, right) = s.select_nth_unstable_by(k, cmp_nan_last);
    (*a, right.iter().copied().fold(f64::INFINITY, f64::min))
}
/// np.median by selection, for values without NaN and without negative zeros (where the value of an order statistic
/// does not depend on how equal elements are arranged): the number median() gives, without the sort
pub fn median_of(mut s: Vec<f64>) -> f64 {
    let n = s.len(); if n == 0 { return f64::NAN; }
    if n % 2 == 1 { *s.select_nth_unstable_by(n / 2, cmp_nan_last).1 } else { let (a, b) = two_order_stats(&mut s, n / 2 - 1); (a + b) / 2.0 }
}
/// np.percentile by selection, under the same conditions: the number percentile() gives
pub fn percentile_of(s: &mut [f64], q: f64) -> f64 {
    let n = s.len(); if n == 0 { return f64::NAN; }
    let vi = (n - 1) as f64 * (q / 100.0);
    let (a, b, t);
    if vi >= (n - 1) as f64 { let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max); a = m; b = m; t = vi + 1.0; }
    else if vi < 0.0 { let m = s.iter().copied().fold(f64::INFINITY, f64::min); a = m; b = m; t = vi; }
    else { let p = vi.floor(); let (x, y) = two_order_stats(s, p as usize); a = x; b = y; t = vi - p; }
    let diff = b - a;
    if t >= 0.5 { b - diff * (1.0 - t) } else { a + diff * t }
}

/// np.nanmedian of a 1-D array
pub fn nanmedian(v: &[f64]) -> f64 {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if s.is_empty() { return f64::NAN; }
    sort_f64(&mut s);
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// np.percentile(v, q) with the default linear method, v without NaN
pub fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec(); sort_f64(&mut s);
    let n = s.len(); if n == 0 { return f64::NAN; }
    let vi = (n - 1) as f64 * (q / 100.0);
    let (a, b, t);
    if vi >= (n - 1) as f64 { a = s[n - 1]; b = s[n - 1]; t = vi + 1.0; }
    else if vi < 0.0 { a = s[0]; b = s[0]; t = vi; }
    else { let p = vi.floor(); a = s[p as usize]; b = s[p as usize + 1]; t = vi - p; }
    let diff = b - a;
    if t >= 0.5 { b - diff * (1.0 - t) } else { a + diff * t }
}

/// np.interp(x, xp, fp) for increasing xp
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x.is_nan() { return f64::NAN; }
    if x < xp[0] { return fp[0]; }
    if x > xp[n - 1] { return fp[n - 1]; }
    if n == 1 { return fp[0]; }
    // the last j with xp[j] <= x
    let mut lo = 0usize; let mut hi = n;                    // invariant: xp[lo] <= x, (hi == n or xp[hi] > x)
    while hi - lo > 1 { let mid = (lo + hi) / 2; if xp[mid] <= x { lo = mid; } else { hi = mid; } }
    let j = lo;
    if j == n - 1 { return fp[j]; }
    if xp[j] == x { return fp[j]; }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut r = slope * (x - xp[j]) + fp[j];
    if r.is_nan() {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if r.is_nan() && fp[j] == fp[j + 1] { r = fp[j]; }
    }
    r
}

/// np.logaddexp
#[inline]
pub fn logaddexp(x: f64, y: f64) -> f64 {
    if x == y { return x + std::f64::consts::LN_2; }
    let tmp = x - y;
    if tmp > 0.0 { x + (-tmp).exp().ln_1p() } else if tmp <= 0.0 { y + tmp.exp().ln_1p() } else { tmp }
}

const LANCZOS: [f64; 9] = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
                           12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];

/// the tool's lgamma_v: Lanczos (g = 7, 9 terms), arguments below 1 shifted up by one
#[inline]
pub fn lgamma_v(x: f64) -> f64 {
    let small = x < 1.0;
    let xs = (if small { x + 1.0 } else { x }) - 1.0; let t = xs + 7.5;
    let a = [LANCZOS[1] / (xs + 1.0), LANCZOS[2] / (xs + 2.0), LANCZOS[3] / (xs + 3.0), LANCZOS[4] / (xs + 4.0),
             LANCZOS[5] / (xs + 5.0), LANCZOS[6] / (xs + 6.0), LANCZOS[7] / (xs + 7.0), LANCZOS[8] / (xs + 8.0)];
    let s = LANCZOS[0] + (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7])));
    let out = 0.5 * (2.0 * std::f64::consts::PI).ln() + (xs + 0.5) * t.ln() - t + s.ln();
    if small { out - x.ln() } else { out }
}

/// log k! for k = 0..=upto: the cumulative sum of log 1, log 2, ...
pub fn log_fact(upto: usize) -> Vec<f64> {
    let mut lf = Vec::with_capacity(upto + 1); lf.push(0.0);
    let mut acc = 0.0f64;
    for k in 1..=upto { if k == 1 { acc = (1.0f64).ln(); } else { acc += (k as f64).ln(); } lf.push(acc); }
    lf
}

pub fn log_floor() -> f64 { (1e-300f64).ln() }

/// log tail probability of the mismatch count o among n shared sites with expectation E and dispersion phi:
/// the scalar form of the tool's bb_tails (upper: log P(X >= o), else log P(X <= o)). lf: the log-factorial table,
/// at least n + 1 long
pub fn bb_tail(o: f64, n: f64, e_in: f64, phi: f64, upper: bool, lf: &[f64]) -> f64 {
    let floor = log_floor();
    if e_in.is_nan() || n.is_nan() || o.is_nan() { return f64::NAN; }
    if e_in <= 1e-12 { return if upper { if o <= 0.0 { 0.0 } else { floor } } else { 0.0 }; }
    let e = e_in.min(n * (1.0 - 1e-9)); let d = e / n;
    let mut binom = !(phi >= 1e-8);
    let (mut a_, mut b_, mut cpair) = (0.0, 0.0, 0.0);
    if !binom {
        let rho = if n >= 2.0 { ((d + phi * n * d) / ((1.0 - d) * (n - 1.0).max(1.0))).min(1.0 - 1e-6) } else { 0.5 };
        a_ = d * (1.0 / rho - 1.0); b_ = (1.0 - d) * (1.0 / rho - 1.0);
        if a_ + b_ > 1e9 {
            if e <= 1e-12 { return if upper { if o <= 0.0 { 0.0 } else { floor } } else { 0.0 }; }
            binom = true;
        } else {
            cpair = -lgamma_v(n + a_ + b_) - (lgamma_v(a_) + lgamma_v(b_) - lgamma_v(a_ + b_));
        }
    }
    let oi = o as i64; let nint = n as i64; const B: usize = 64;
    let mut out = f64::NEG_INFINITY;
    let (mut kstart, last, mut top_k) = (oi, nint, oi.min(nint));
    let active = if upper { oi <= nint } else { oi >= 0 };
    if !active { return floor; }
    let ld = d.ln(); let l1d = (1.0 - d).ln();
    let mut lp = [0.0f64; B]; let mut ex = [0.0f64; B];
    loop {
        let (cnt, k0) = if upper { let c = (B as i64).min(last - kstart + 1); (c, kstart) } else { let c = (B as i64).min(top_k + 1); (c, top_k - c + 1) };
        if cnt <= 0 { break; }
        // _block_terms
        let k0f = k0 as f64;
        let small = n < 2.0;
        let (base_a, base_b, xb) = if binom { (0.0, 0.0, 0.0) } else { (lgamma_v(k0f + a_), lgamma_v(n - k0f + b_), n - k0f + b_) };
        let mut cs_a = 0.0f64; let mut cs_b = 0.0f64; let mut cs_b_prev = 0.0f64;
        for j in 0..B {
            let valid = (j as i64) < cnt;
            let ksafe = if valid { k0 + j as i64 } else { 0 }; let nk = if valid { nint - (k0 + j as i64) } else { 0 };
            let lchoose = lf[nint as usize] - lf[ksafe as usize] - lf[nk as usize];
            let kf = ksafe as f64;
            let v;
            if binom { v = lchoose + kf * ld + (n - kf) * l1d; }
            else {
                let inc_a = ((k0f + a_) + j as f64).ln();
                cs_a = if j == 0 { inc_a } else { cs_a + inc_a };
                let lg_a = base_a + cs_a - inc_a;
                let inc_b = (xb - (j as f64 + 1.0)).max(1e-300).ln();
                let lg_b = base_b - if j == 0 { 0.0 } else { cs_b_prev };
                cs_b = if j == 0 { inc_b } else { cs_b + inc_b }; cs_b_prev = cs_b;
                v = if small { lchoose + kf * ld + (n - kf) * l1d } else { lchoose + lg_a + lg_b + cpair };
            }
            lp[j] = if valid { v } else { f64::NEG_INFINITY };
        }
        // _logsumexp_rows
        let mut mx = lp[0];
        for &v in &lp[1..] { if v.is_nan() { mx = v; break; } if v > mx { mx = v; } }
        let mx = if mx.is_finite() { mx } else { 0.0 };
        for j in 0..B { ex[j] = (lp[j] - mx).exp(); }
        let blk = mx + pairwise_sum(&ex).ln();
        out = logaddexp(out, blk);
        let c = cnt as usize;
        let (done, falling);
        if upper { kstart += cnt; done = kstart > last; falling = lp[c - 1] < lp[0]; }
        else { top_k -= cnt; done = top_k < 0; falling = lp[0] < lp[c - 1]; }
        if done || (falling && blk < out - 46.0) { break; }
    }
    out.min(0.0).max(floor)
}

extern "C" { fn erfc(x: f64) -> f64; }

/// inverse standard normal CDF by bisection on erfc, as the tool does it
pub fn norm_quantile(p: f64) -> f64 {
    if p < 0.5 { return -norm_quantile(1.0 - p); }
    let (mut lo, mut hi) = (0.0f64, 50.0f64);
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if 0.5 * unsafe { erfc(mid / 2f64.sqrt()) } > 1.0 - p { lo = mid; } else { hi = mid; }
    }
    (lo + hi) / 2.0
}

// ------------------------------------------------------------------ the broad-sample draws
#[inline]
pub fn sm64(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// `need` distinct values in [0, m) for one key by hashing (key, slot): the tool's _hash_draw for one row
pub fn hash_draw(key: u64, m: usize, need: usize) -> Vec<i64> {
    if m < 2 * need {
        let mut h: Vec<(u64, usize)> = (0..m).map(|j| (sm64(key ^ j as u64), j)).collect();
        h.sort();                                               // by hash, then slot: the stable order
        return h[..need].iter().map(|x| x.1 as i64).collect();
    }
    let mut s = (need as f64 + 4.0 * (need as f64).sqrt() + (need * need) as f64 / (m as i64 - need as i64).max(1) as f64 + 16.0) as usize;
    let mut start = 0usize;
    loop {
        let mut out: Vec<i64> = Vec::with_capacity(need);
        if m <= 4096 {
            // the values seen so far as a bit each (the broad sample draws from pools of a few hundred, per sequence)
            let mut seen = [0u64; 64];
            for slot in start..start + s {
                let v = (sm64(key ^ slot as u64) % m as u64) as usize; let (w, b) = (v / 64, 1u64 << (v % 64));
                if seen[w] & b == 0 { seen[w] |= b; out.push(v as i64); if out.len() == need { return out; } }
            }
        } else {
            let mut seen = std::collections::HashSet::with_capacity(2 * need);
            for slot in start..start + s {
                let v = (sm64(key ^ slot as u64) % m as u64) as i64;
                if seen.insert(v) { out.push(v); if out.len() == need { return out; } }
            }
        }
        start += s; s *= 2;
        if start > (1 << 20) { panic!("broad-sample draw could not find {need} distinct of {m}"); }
    }
}

/// pool-adjacent-violators: the non-decreasing sequence closest to y (weights w)
pub fn pav(y: &[f64], w: &[f64]) -> Vec<f64> {
    let mut blocks: Vec<(f64, f64, usize)> = Vec::new();
    for (&yi, &wi) in y.iter().zip(w) {
        blocks.push((yi, wi, 1));
        while blocks.len() > 1 && blocks[blocks.len() - 2].0 > blocks[blocks.len() - 1].0 {
            let b2 = blocks.pop().unwrap(); let b1 = blocks.pop().unwrap(); let wt = b1.1 + b2.1;
            blocks.push(((b1.0 * b1.1 + b2.0 * b2.1) / wt, wt, b1.2 + b2.2));
        }
    }
    let mut out = Vec::with_capacity(y.len());
    for (v, _, c) in blocks { for _ in 0..c { out.push(v); } }
    out
}

/// the weighted median of v[..l] under the integer-valued weights w[..l], by selection (the pair kernel's
/// _wmed_select): the value at the first position of the sorted order whose prefix weight reaches half the total
pub fn wmed_select(v: &mut [f64], w: &mut [f64], l: usize) -> f64 {
    let mut tot = 0.0f64;
    for t in 0..l { tot += w[t]; }
    if tot <= 0.0 { return f64::NAN; }
    let half = tot / 2.0; let mut lo = 0isize; let mut hi = l as isize - 1; let mut wleft = 0.0f64;
    while hi - lo >= 16 {
        let mid = (lo + hi) >> 1; let mut a = v[lo as usize]; let mut b = v[mid as usize]; let c = v[hi as usize];
        if a > b { std::mem::swap(&mut a, &mut b); }
        if b > c { b = c; }
        if a > b { b = a; }
        let piv = b; let mut i = lo; let mut j = lo; let mut k = hi; let mut wl = 0.0f64; let mut we = 0.0f64;
        while j <= k {
            let x = v[j as usize];
            if x < piv {
                v.swap(i as usize, j as usize); w.swap(i as usize, j as usize); wl += w[i as usize]; i += 1; j += 1;
            } else if x > piv {
                v.swap(j as usize, k as usize); w.swap(j as usize, k as usize); k -= 1;
            } else { we += w[j as usize]; j += 1; }
        }
        if wleft + wl >= half { hi = i - 1; }
        else if wleft + wl + we >= half { return piv; }
        else { wleft += wl + we; lo = j; }
    }
    for s in (lo + 1)..=hi {
        let x = v[s as usize]; let y = w[s as usize]; let mut t = s - 1;
        while t >= lo && v[t as usize] > x { v[(t + 1) as usize] = v[t as usize]; w[(t + 1) as usize] = w[t as usize]; t -= 1; }
        v[(t + 1) as usize] = x; w[(t + 1) as usize] = y;
    }
    let mut cw = wleft;
    for t in lo..=hi { cw += w[t as usize]; if cw >= half { return v[t as usize]; } }
    v[hi as usize]
}

/// the tool's wmedian_stack for one column: the weighted median of the finite values with positive weight
pub fn wmedian_sorted(vals: &[f64], wts: &[f64]) -> f64 {
    let mut vw: Vec<(f64, f64)> = vals.iter().zip(wts).map(|(&v, &w)| if v.is_finite() && w > 0.0 { (v, w) } else { (f64::INFINITY, 0.0) }).collect();
    vw.sort_by(|a, b| cmp_nan_last(&a.0, &b.0));
    let mut cw = Vec::with_capacity(vw.len()); let mut acc = 0.0;
    for (i, x) in vw.iter().enumerate() { acc = if i == 0 { x.1 } else { acc + x.1 }; cw.push(acc); }
    let tot = *cw.last().unwrap_or(&0.0);
    if !(tot > 0.0) { return f64::NAN; }
    let idx = cw.iter().filter(|&&c| c < tot / 2.0).count().min(vw.len() - 1);
    vw[idx].0
}

/// the tool's wmedian1
pub fn wmedian1(v: &[f64], w: &[f64]) -> f64 {
    let mut vw: Vec<(f64, f64)> = v.iter().copied().zip(w.iter().copied()).collect();
    vw.sort_by(|a, b| cmp_nan_last(&a.0, &b.0));
    let mut cw = Vec::with_capacity(vw.len()); let mut acc = 0.0;
    for (i, x) in vw.iter().enumerate() { acc = if i == 0 { x.1 } else { acc + x.1 }; cw.push(acc); }
    let tot = *cw.last().unwrap();
    if tot > 0.0 { vw[cw.iter().filter(|&&c| c < tot / 2.0).count().min(vw.len() - 1)].0 } else { median(v) }
}

/// np.maximum / np.minimum: NaN when either is NaN
#[inline] pub fn npmax(a: f64, b: f64) -> f64 { if a.is_nan() || b.is_nan() { f64::NAN } else if a > b { a } else { b } }
#[inline] pub fn npmin(a: f64, b: f64) -> f64 { if a.is_nan() || b.is_nan() { f64::NAN } else if a < b { a } else { b } }
/// Python's max(a, b) / min(a, b): the first unless the second beats it
#[inline] pub fn pymax(a: f64, b: f64) -> f64 { if b > a { b } else { a } }
#[inline] pub fn pymin(a: f64, b: f64) -> f64 { if b < a { b } else { a } }
