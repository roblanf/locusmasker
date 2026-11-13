//! Restart points: after the locus rates and after the pair sweeps the run leaves what it has computed in
//! <out>/restart/<point>/ (arrays as .npy, a header naming the input and every option the results depend on), so a
//! rerun of the same input with the same options in the same --out picks up there. The points hold this port's own
//! structures (each taxon's list and the closest match per cell): a point written by locusmasker.py is not used.
use crate::npy::{npy, npy1, read_npy};
use crate::pyfmt::{json_parse, J};
use crate::{die, jdict, log, Args};
use std::collections::HashMap;

pub struct Point { pub arrays: HashMap<String, (Vec<usize>, Vec<u8>)>, pub written: String }
impl Point {
    fn raw(&self, k: &str) -> &[u8] { &self.arrays[k].1 }
    pub fn f64s(&self, k: &str) -> Vec<f64> { self.raw(k).chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect() }
    pub fn i64s(&self, k: &str) -> Vec<i64> { self.raw(k).chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect() }
    pub fn bools(&self, k: &str) -> Vec<bool> { self.raw(k).iter().map(|b| *b != 0).collect() }
}

/// what a restart point must have been made from
#[allow(clippy::too_many_arguments)]
pub fn header(a: &Args, n: usize, nloc: usize, l: usize, dtype: &str, kn: usize, c: usize, thin: &[usize], full: bool) -> J {
    let base = |p: &str| std::path::Path::new(p).file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
    jdict! { "tool" => J::s("LocusMasker"), "version" => J::s(crate::VERSION),
             "input" => jdict! { "file" => J::Str(base(&a.aln)), "bytes" => J::Int(std::fs::metadata(&a.aln).map(|m| m.len() as i64).unwrap_or(0)), "sha256" => J::Str(crate::input_sha256(a)) },
             "partitions_sha256" => match &a.partitions { Some(p) => J::Str(crate::report::sha256_file(p)), None => J::Null },
             "datatype" => J::s(dtype), "fit_overlap" => J::Int(a.fit_overlap), "K_eff" => J::Int(kn as i64), "candidates" => J::Int(c as i64),
             "full_m_max_taxa" => J::Int(a.full_m_max_taxa as i64), "full" => J::Bool(full), "taxa" => J::Int(n as i64), "loci" => J::Int(nloc as i64), "sites" => J::Int(l as i64),
             "thin" => J::List(thin.iter().map(|t| J::Int(*t as i64)).collect()), "port" => J::s("rust") }
}

pub enum Arr<'a> { F64(&'a [f64], Vec<usize>), I64(&'a [i64], Vec<usize>), Bool(&'a [bool]) }

/// a restart point: the arrays and then the header, into a .tmp directory that is renamed into place
pub fn write(a: &Args, point: &str, header: &J, arrays: &[(&str, Arr)]) {
    let d = format!("{}/restart/{}", a.out, point); let tmp = format!("{d}.tmp");
    let r = (|| -> std::io::Result<()> {
        let _ = std::fs::remove_dir_all(&tmp); std::fs::create_dir_all(&tmp)?;
        for (k, v) in arrays {
            let bytes = match v { Arr::F64(x, sh) => npy(sh, x), Arr::I64(x, sh) => npy(sh, x), Arr::Bool(x) => npy1(x) };
            std::fs::write(format!("{tmp}/{k}.npy"), bytes)?;
        }
        let mut h = header.clone(); h.set("point", J::s(point)); h.set("written", J::Str(crate::now_string()));
        std::fs::write(format!("{tmp}/header.json"), h.indent1())?;
        let _ = std::fs::remove_dir_all(&d); std::fs::rename(&tmp, &d)
    })();
    match r {
        Ok(()) => log!("restart point written: {} ({}): a rerun of this input with these options in this --out starts from it", d, if point == "rates" { "after the locus rates" } else { "after the pair sweeps" }),
        Err(e) => log!("WARNING: the restart point {} could not be written ({}); the run goes on without it", d, crate::input::os_err(&e)),
    }
    if a.stop_after.as_deref() == Some(point) { die!("stopped after the {point} restart point (LOCUSMASKER_STOP_AFTER)"); }
}

/// the arrays of a matching restart point and when it was written, or None: absent, incomplete (said so and removed)
/// or written by locusmasker.py. One from another input or other options stops the run naming what differs
pub fn read(a: &Args, point: &str, header: &J, names: &[&str]) -> Option<Point> {
    if a.restart { return None; }
    let d = format!("{}/restart/{}", a.out, point);
    if !std::path::Path::new(&d).is_dir() { return None; }
    let load = || -> Result<(J, HashMap<String, (Vec<usize>, Vec<u8>)>), String> {
        let txt = std::fs::read_to_string(format!("{d}/header.json")).map_err(|e| format!("header.json: {}", crate::input::os_err(&e)))?;
        let got = json_parse(&txt).ok_or("header.json is not JSON")?;
        if got.get("port").is_none() && got.get("tool").is_some() { return Err("py".into()); }
        let mut arrays = HashMap::new();
        for k in names {
            let b = std::fs::read(format!("{d}/{k}.npy")).map_err(|e| format!("{k}.npy: {}", crate::input::os_err(&e)))?;
            let (_, shape, raw) = read_npy(&b).ok_or(format!("{k}.npy is not an array"))?;
            arrays.insert(k.to_string(), (shape, raw.to_vec()));
        }
        Ok((got, arrays))
    };
    let (got, arrays) = match load() {
        Ok(x) => x,
        Err(e) if e == "py" => { log!("note: the restart point {} was written by locusmasker.py, whose points this port does not read: the sweeps run", d); return None; }
        Err(e) => { log!("note: the restart point {} is incomplete ({}): removed; the sweeps run", d, e); let _ = std::fs::remove_dir_all(&d); return None; }
    };
    let J::Dict(h) = header else { return None };
    let diff: Vec<&str> = h.iter().filter(|(k, v)| got.get(k) != Some(v)).map(|(k, _)| k.as_str()).collect();
    if !diff.is_empty() { die!("the restart point {} is from a different run ({} differ): --restart ignores it, or use another --out", d, diff.join(", ")); }
    let written = match got.get("written") { Some(J::Str(s)) => s.clone(), _ => String::new() };
    log!("restarted after {} from the restart point written {}: {} not repeated", if point == "rates" { "the locus rates" } else { "the pair sweeps" }, written,
         if point == "rates" { "sweep 1 and the rates are" } else { "the sweeps are" });
    Some(Point { arrays, written })
}
