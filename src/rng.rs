//! numpy's default_rng(seed).choice(n, k, replace=False), bit for bit: SeedSequence, PCG64 (XSL-RR 128/64), the
//! buffered 32-bit draws, Lemire's bounded integers, the tail shuffle and Floyd's algorithm.
const INIT_A: u32 = 0x43b0d7e5; const MULT_A: u32 = 0x931e8875; const INIT_B: u32 = 0x8b51f9dd; const MULT_B: u32 = 0x58f38ded;
const MIX_MULT_L: u32 = 0xca01f9dd; const MIX_MULT_R: u32 = 0x4973f715; const XSHIFT: u32 = 16;
const PCG_MULT: u128 = 0x2360ED051FC65DA44385DF649FCCF645;

fn hashmix(mut value: u32, hash_const: &mut u32) -> u32 {
    value ^= *hash_const; *hash_const = hash_const.wrapping_mul(MULT_A); value = value.wrapping_mul(*hash_const); value ^= value >> XSHIFT; value
}
fn mix(x: u32, y: u32) -> u32 { let mut r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y)); r ^= r >> XSHIFT; r }

/// SeedSequence(entropy).generate_state(4, uint64)
fn seed_state(entropy: u32) -> [u64; 4] {
    let ent = [entropy]; let mut pool = [0u32; 4]; let mut hc = INIT_A;
    for i in 0..4 { pool[i] = hashmix(if i < ent.len() { ent[i] } else { 0 }, &mut hc); }
    for i_src in 0..4 { for i_dst in 0..4 { if i_src != i_dst { let h = hashmix(pool[i_src], &mut hc); pool[i_dst] = mix(pool[i_dst], h); } } }
    let mut hc = INIT_B; let mut w = [0u32; 8];
    for (i, x) in w.iter_mut().enumerate() { let mut d = pool[i % 4]; d ^= hc; hc = hc.wrapping_mul(MULT_B); d = d.wrapping_mul(hc); d ^= d >> XSHIFT; *x = d; }
    [w[0] as u64 | (w[1] as u64) << 32, w[2] as u64 | (w[3] as u64) << 32, w[4] as u64 | (w[5] as u64) << 32, w[6] as u64 | (w[7] as u64) << 32]
}

pub struct Pcg64 { state: u128, inc: u128, has32: bool, buf32: u32 }
impl Pcg64 {
    pub fn new(seed: u32) -> Pcg64 {
        let s = seed_state(seed);
        let initstate = (s[0] as u128) << 64 | s[1] as u128; let initseq = (s[2] as u128) << 64 | s[3] as u128;
        let mut r = Pcg64 { state: 0, inc: (initseq << 1) | 1, has32: false, buf32: 0 };
        r.step(); r.state = r.state.wrapping_add(initstate); r.step(); r
    }
    #[inline] fn step(&mut self) { self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc); }
    #[inline] pub fn next64(&mut self) -> u64 { self.step(); let hi = (self.state >> 64) as u64; let lo = self.state as u64; (hi ^ lo).rotate_right((hi >> 58) as u32) }
    #[inline] pub fn next32(&mut self) -> u32 {
        if self.has32 { self.has32 = false; return self.buf32; }
        let n = self.next64(); self.has32 = true; self.buf32 = (n >> 32) as u32; n as u32
    }
    /// random_bounded_uint64(bitgen, 0, rng, 0, false): uniform on [0, rng]
    pub fn bounded(&mut self, rng: u64) -> u64 {
        if rng == 0 { return 0; }
        if rng <= 0xFFFF_FFFF {
            if rng == 0xFFFF_FFFF { return self.next32() as u64; }
            let rng = rng as u32; let rng_excl = rng.wrapping_add(1);
            let mut m = self.next32() as u64 * rng_excl as u64; let mut leftover = m as u32;
            if leftover < rng_excl {
                let threshold = (u32::MAX - rng) % rng_excl;
                while leftover < threshold { m = self.next32() as u64 * rng_excl as u64; leftover = m as u32; }
            }
            return m >> 32;
        }
        if rng == u64::MAX { return self.next64(); }
        let rng_excl = rng + 1;
        let mut m = self.next64() as u128 * rng_excl as u128; let mut leftover = m as u64;
        if leftover < rng_excl {
            let threshold = (u64::MAX - rng) % rng_excl;
            while leftover < threshold { m = self.next64() as u128 * rng_excl as u128; leftover = m as u64; }
        }
        (m >> 64) as u64
    }
}

/// sorted(default_rng(seed).choice(pop, size, replace=False))
pub fn choice_sorted(seed: u32, pop: usize, size: usize) -> Vec<usize> {
    if size >= pop { return (0..pop).collect(); }
    let mut rng = Pcg64::new(seed);
    let mut out: Vec<usize>;
    if pop > 10000 && size > pop / 50 {
        let mut idx: Vec<usize> = (0..pop).collect();
        let first = (pop - size).max(1);
        for i in (first..pop).rev() { let j = rng.bounded(i as u64) as usize; idx.swap(i, j); }
        out = idx[pop - size..].to_vec();
    } else {
        let mut set_size = (1.2 * size as f64) as u64;
        let mut mask = set_size; mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16; mask |= mask >> 32;
        set_size = 1 + mask; let mut hash = vec![u64::MAX; set_size as usize]; out = Vec::with_capacity(size);
        for j in (pop - size)..pop {
            let val = rng.bounded(j as u64); let mut loc = (val & mask) as usize;
            while hash[loc] != u64::MAX && hash[loc] != val { loc = (loc + 1) & mask as usize; }
            if hash[loc] == u64::MAX { hash[loc] = val; out.push(val as usize); }
            else { let mut loc = (j as u64 & mask) as usize; while hash[loc] != u64::MAX { loc = (loc + 1) & mask as usize; } hash[loc] = j as u64; out.push(j); }
        }
    }
    out.sort_unstable(); out
}
