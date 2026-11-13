//! The report's data as the test stage fills it: per-cell state arrays, the numeric columns (dense taxa x loci, or
//! sparse rows), and the per-locus members (site codes, sequence rows, trees, pair records).
use crate::npy::npy;
use crate::pyfmt::J;

#[derive(Clone, Copy, PartialEq)]
pub enum Dt { F32, I32, U32, U16 }

#[derive(Clone)]
pub enum Col { F32(Vec<f32>), I32(Vec<i32>), U32(Vec<u32>), U16(Vec<u16>) }

impl Col {
    pub fn filled(dt: Dt, fill: f64, len: usize) -> Col {
        match dt { Dt::F32 => Col::F32(vec![fill as f32; len]), Dt::I32 => Col::I32(vec![fill as i32; len]), Dt::U32 => Col::U32(vec![fill as u32; len]), Dt::U16 => Col::U16(vec![fill as u16; len]) }
    }
    pub fn itemsize(&self) -> usize { match self { Col::U16(_) => 2, _ => 4 } }
    /// self[dst] = src[i]
    pub fn copy_from(&mut self, dst: usize, src: &Col, i: usize) {
        match (self, src) {
            (Col::F32(a), Col::F32(b)) => a[dst] = b[i], (Col::I32(a), Col::I32(b)) => a[dst] = b[i],
            (Col::U32(a), Col::U32(b)) => a[dst] = b[i], (Col::U16(a), Col::U16(b)) => a[dst] = b[i],
            _ => panic!("column types differ"),
        }
    }
    pub fn push_from(&mut self, src: &Col, i: usize) {
        match (self, src) {
            (Col::F32(a), Col::F32(b)) => a.push(b[i]), (Col::I32(a), Col::I32(b)) => a.push(b[i]),
            (Col::U32(a), Col::U32(b)) => a.push(b[i]), (Col::U16(a), Col::U16(b)) => a.push(b[i]),
            _ => panic!("column types differ"),
        }
    }
    pub fn push_val(&mut self, v: f64) {
        match self { Col::F32(a) => a.push(v as f32), Col::I32(a) => a.push(v as i32), Col::U32(a) => a.push(v as u32), Col::U16(a) => a.push(v as u16) }
    }
    pub fn set_i(&mut self, i: usize, v: i64) { match self { Col::I32(a) => a[i] = v as i32, Col::U32(a) => a[i] = v as u32, Col::U16(a) => a[i] = v as u16, Col::F32(a) => a[i] = v as f32 } }
    pub fn set_f(&mut self, i: usize, v: f32) { if let Col::F32(a) = self { a[i] = v; } else { panic!("not a float column") } }
    pub fn get_i(&self, i: usize) -> i64 { match self { Col::I32(a) => a[i] as i64, Col::U32(a) => a[i] as i64, Col::U16(a) => a[i] as i64, Col::F32(a) => a[i] as i64 } }
    pub fn f32s(&self) -> &[f32] { if let Col::F32(a) = self { a } else { panic!("not a float column") } }
    /// the column at the given rows (None: the fill)
    pub fn gather(&self, idx: &[Option<usize>], fill: f64) -> Col {
        match self {
            Col::F32(a) => Col::F32(idx.iter().map(|i| i.map_or(fill as f32, |k| a[k])).collect()),
            Col::I32(a) => Col::I32(idx.iter().map(|i| i.map_or(fill as i32, |k| a[k])).collect()),
            Col::U32(a) => Col::U32(idx.iter().map(|i| i.map_or(fill as u32, |k| a[k])).collect()),
            Col::U16(a) => Col::U16(idx.iter().map(|i| i.map_or(fill as u16, |k| a[k])).collect()),
        }
    }
    pub fn npy(&self, shape: &[usize]) -> Vec<u8> {
        match self { Col::F32(a) => npy(shape, a), Col::I32(a) => npy(shape, a), Col::U32(a) => npy(shape, a), Col::U16(a) => npy(shape, a) }
    }
}

/// the per-cell numeric columns: name, dtype, fill
pub const NUMCOLS: [(&str, Dt, f64); 14] = [
    ("med_identity", Dt::F32, f64::NAN), ("med_expected", Dt::F32, f64::NAN), ("excess_m", Dt::F32, f64::NAN),
    ("closest", Dt::I32, -1.0), ("closest_overlap", Dt::U32, 0.0), ("closest_identity", Dt::F32, f64::NAN), ("closest_expected", Dt::F32, f64::NAN),
    ("best_match", Dt::I32, -1.0), ("best_match_identity", Dt::F32, f64::NAN), ("partner", Dt::I32, -1.0),
    ("div_log10p", Dt::F32, f64::NAN), ("sim_log10p", Dt::F32, f64::NAN), ("locus_closest", Dt::I32, -1.0), ("locus_closest_log10p", Dt::F32, f64::NAN)];
pub const C_PARTNER: usize = 9;
pub const C_LOCUS_CLOSEST: usize = 12;
/// the broad test's per-cell members (dense runs): the first four, then the cards' rows
pub const BSCOLS: [(&str, Dt, f64); 13] = [
    ("bs_log10p", Dt::F32, f64::NAN), ("bs_rho", Dt::F32, f64::NAN), ("bs_stat", Dt::F32, f64::NAN), ("bs_rep", Dt::I32, -1.0),
    ("bs_close_log10p", Dt::F32, f64::NAN), ("bs_div_log10p", Dt::F32, f64::NAN), ("bs_closest", Dt::I32, -1.0), ("bs_closest_mis", Dt::U32, 0.0),
    ("bs_closest_exp", Dt::F32, f64::NAN), ("bs_nt", Dt::U16, 0.0), ("bs_nfail", Dt::U16, 0.0), ("bs_obs_pct", Dt::F32, f64::NAN), ("bs_exp_pct", Dt::F32, f64::NAN)];

#[derive(Clone, Default)]
pub struct Trees { pub cells: Vec<i64>, pub expected: Vec<String>, pub observed: Vec<String>, pub tips: Vec<Vec<i64>>, pub outgroup: Vec<i64>, pub closest: Vec<[i64; 2]> }
impl Trees {
    pub fn json(&self) -> J {
        J::Dict(vec![
            ("cells".into(), J::ints(&self.cells)), ("expected".into(), J::List(self.expected.iter().map(|s| J::s(s)).collect())),
            ("observed".into(), J::List(self.observed.iter().map(|s| J::s(s)).collect())), ("tips".into(), J::List(self.tips.iter().map(|t| J::ints(t)).collect())),
            ("outgroup".into(), J::ints(&self.outgroup)), ("closest".into(), J::List(self.closest.iter().map(|c| J::ints(&c[..])).collect()))])
    }
    pub fn push(&mut self, o: &Trees, j: usize) {
        self.cells.push(o.cells[j]); self.expected.push(o.expected[j].clone()); self.observed.push(o.observed[j].clone());
        self.tips.push(o.tips[j].clone()); self.outgroup.push(o.outgroup[j]); self.closest.push(o.closest[j]);
    }
}

/// a dict keyed by locus, in insertion order
pub struct ByLocus<T>(pub Vec<(usize, T)>);
impl<T> Default for ByLocus<T> { fn default() -> Self { ByLocus(vec![]) } }
impl<T> ByLocus<T> {
    pub fn get_mut(&mut self, li: usize) -> Option<&mut T> { self.0.iter_mut().find(|x| x.0 == li).map(|x| &mut x.1) }
    pub fn set(&mut self, li: usize, v: T) { if let Some(e) = self.0.iter_mut().find(|x| x.0 == li) { e.1 = v; } else { self.0.push((li, v)); } }
    pub fn index(&self) -> std::collections::HashMap<usize, usize> { self.0.iter().enumerate().map(|(k, x)| (x.0, k)).collect() }
}

pub struct Rep {
    pub dense: bool,
    pub focal_sites: Vec<u32>, pub thin: Vec<usize>, pub present: Vec<i64>,
    pub seq_policy: String, pub uncovered_sites: usize,
    pub restart: Option<(String, String)>, pub lists_extended: (usize, usize),
    pub n_comp: Vec<u8>, pub n_fail: Vec<u8>, pub n_similar: Vec<u8>, pub side: Vec<u8>, pub side2: Option<Vec<u8>>,
    pub cols: Vec<Col>,                                   // NUMCOLS: n x nloc when dense, else one row per carried cell
    pub comparators: Vec<i32>, pub comp_div: Vec<f32>, pub comp_sim: Vec<f32>,      // rows x k_eff
    pub sparse_locus: Vec<i32>, pub sparse_taxon: Vec<i32>,
    pub codes: ByLocus<(Vec<i32>, Vec<u8>)>, pub seqs: ByLocus<(Vec<i32>, Vec<u8>)>, pub trees: ByLocus<Trees>,
    pub similar: ByLocus<Vec<J>>, pub bs_records: ByLocus<Vec<J>>,
    pub loc_status: Vec<String>, pub phi: Vec<f64>, pub test_rate: Vec<f64>, pub pair_tests: Vec<i64>,
    pub bscols: Option<Vec<Col>>,                         // BSCOLS, n x nloc (dense runs with a broad test)
    pub bs_rows: Option<BsRows>, pub bs_summary: Option<J>,
    pub taxa_closest: Vec<i32>, pub taxa_closest_w: usize,
    pub rate: Vec<f64>, pub plateaus: Vec<f64>, pub kq: f64, pub phi_bar: f64,
    pub tests: [i64; 6],                                  // close_relative, closest_match_relative, closest_match_outside, broad_sample, total, locus_sets
    pub untested: [i64; 4],                               // sequences, of_excluded_taxa, at_untested_loci, without_a_comparator
}

pub struct BsRows { pub locus: Vec<i32>, pub taxon: Vec<i32>, pub var_log10p: Vec<f32>, pub lo_log10p: Vec<f32>, pub hi_log10p: Vec<f32>, pub partner: Vec<i32>, pub call: Vec<i8> }
