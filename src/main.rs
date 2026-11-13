//! LocusMasker: per-locus outlier detection and masking in a multi-locus alignment.
//! A Rust port of locusmasker.py that writes the same outputs; the report viewer (HTML, CSS, JavaScript) is the
//! Python tool's, embedded unchanged.
mod input;
mod log;
mod npmath;
mod npy;
mod prepare;
mod pyfmt;
mod rep;
mod report;
mod restart;
mod rng;
mod runz;
mod tree;
#[cfg(test)]
mod tests;

use pyfmt::commas;

const REPORT_SEQ_ALL_MAX_DEFAULT: usize = 32 << 20;

pub const VERSION: &str = "0.16.0";
const USAGE: &str = "usage: locusmasker --aln FILE --datatype dna|aa --out DIR [options]";

pub struct Args {
    pub aln: String, pub datatype: String, pub out: String, pub partitions: Option<String>, pub name: Option<String>,
    pub n_comp: i64, pub fit_overlap: i64, pub alpha: f64, pub donor_threshold: f64,
    pub threads: usize, pub memory: Option<f64>, pub restart: bool, pub progress: String,
    pub no_report: bool, pub all_tests: bool, pub mask_similar_only: bool, pub mask_different_only: bool, pub no_broad_sample: bool,
    pub candidates: Option<usize>, pub block_taxa: Option<i64>, pub no_numba: bool, pub report_codes: String, pub report_sequences: String,
    pub stop_after: Option<String>, pub dense_cells_max: usize, pub best_match_all_max_taxa: usize, pub full_m_max_taxa: usize, pub sweep_block_min: usize,
}

static SHA: std::sync::OnceLock<String> = std::sync::OnceLock::new();
static SHA_JOB: std::sync::Mutex<Option<std::thread::JoinHandle<Option<String>>>> = std::sync::Mutex::new(None);
/// start the input's sha256 on a thread of its own, while the alignment is read (a file that cannot be read is
/// left to the reader to report)
fn start_input_sha256(path: &str) {
    let p = path.to_string();
    *SHA_JOB.lock().unwrap() = Some(std::thread::spawn(move || report::sha256_of(&p).ok()));
}
/// the input's sha256, computed once
pub fn input_sha256(a: &Args) -> String {
    SHA.get_or_init(|| {
        let job = SHA_JOB.lock().unwrap().take();
        job.and_then(|h| h.join().ok().flatten()).unwrap_or_else(|| report::sha256_file(&a.aln))
    }).clone()
}

pub fn fmem(b: u64) -> String {
    if b >= (1 << 30) { format!("{:.1} GB", b as f64 / (1u64 << 30) as f64) } else { format!("{:.0} MB", b as f64 / (1u64 << 20) as f64) }
}

/// a duration for a person: 34.2 s, 2 min 36 s (156 s), 3 h 12 min 5 s (11,525 s)
pub fn elapsed_string(sec: f64) -> String {
    if sec < 59.95 { return format!("{:.1} s", sec); }
    let t = sec.round() as u64; let (h, m, s) = (t / 3600, t % 3600 / 60, t % 60);
    if h > 0 { format!("{} h {} min {} s ({} s)", h, m, s, pyfmt::commas(t as i64)) } else { format!("{} min {} s ({} s)", m, s, t) }
}

pub fn now_string() -> String {
    unsafe {
        let t = libc::time(std::ptr::null_mut()); let mut tm: libc::tm = std::mem::zeroed(); libc::localtime_r(&t, &mut tm);
        format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
    }
}

pub fn system_string() -> String {
    unsafe {
        let mut u: libc::utsname = std::mem::zeroed();
        if libc::uname(&mut u) != 0 { return std::env::consts::OS.to_string(); }
        let s = |p: &[libc::c_char]| std::ffi::CStr::from_ptr(p.as_ptr()).to_string_lossy().into_owned();
        let base = format!("{}-{}-{}", s(&u.sysname), s(&u.release), s(&u.machine));
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        { return format!("{}-with-glibc{}", base, std::ffi::CStr::from_ptr(libc::gnu_get_libc_version()).to_string_lossy()); }
        #[allow(unreachable_code)]
        base
    }
}

fn mem_available() -> u64 {
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for line in s.lines() { if line.starts_with("MemAvailable:") { if let Some(v) = line.split_whitespace().nth(1).and_then(|x| x.parse::<u64>().ok()) { return v * 1024; } } }
    }
    2 << 30
}

fn arg_error(msg: &str) -> ! { eprintln!("{USAGE}\nlocusmasker: error: {msg}"); std::process::exit(2) }

fn parse_args(argv: &[String]) -> Args {
    // (name, takes a value)
    const OPTS: [(&str, bool); 24] = [("--aln", true), ("--datatype", true), ("--out", true), ("--partitions", true), ("--name", true), ("--n-comp", true), ("--fit-overlap", true),
        ("--alpha", true), ("--donor-threshold", true), ("--threads", true), ("--memory", true), ("--restart", false), ("--progress", true), ("--no-report", false), ("--all-tests", false),
        ("--help", false), ("--mask-similar-only", false), ("--mask-different-only", false), ("--no-broad-sample", false), ("--candidates", true), ("--block-taxa", true), ("--no-numba", false),
        ("--report-codes", true), ("--report-sequences", true)];
    let mut vals: std::collections::HashMap<&str, String> = std::collections::HashMap::new(); let mut flags: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut extra: Vec<String> = vec![]; let mut i = 1;
    while i < argv.len() {
        let tok = &argv[i]; i += 1;
        if tok == "-h" { print!("{}", include_str!("help.txt")); std::process::exit(0); }
        if !tok.starts_with("--") || tok == "--" { extra.push(tok.clone()); continue; }
        let (key, inline) = match tok.split_once('=') { Some((k, v)) => (k, Some(v.to_string())), None => (tok.as_str(), None) };
        let hit: Vec<&(&str, bool)> = match OPTS.iter().find(|o| o.0 == key) { Some(o) => vec![o], None => OPTS.iter().filter(|o| o.0.starts_with(key)).collect() };
        if hit.is_empty() { extra.push(tok.clone()); continue; }
        if hit.len() > 1 { arg_error(&format!("ambiguous option: {} could match {}", key, hit.iter().map(|o| o.0).collect::<Vec<_>>().join(", "))); }
        let (name, takes) = *hit[0];
        if name == "--help" { print!("{}", include_str!("help.txt")); std::process::exit(0); }
        if takes {
            let v = match inline { Some(v) => v, None => { if i < argv.len() && !(argv[i].starts_with('-') && argv[i].len() > 1 && argv[i][1..].parse::<f64>().is_err()) { i += 1; argv[i - 1].clone() } else { arg_error(&format!("argument {}: expected one argument", name)) } } };
            vals.insert(name, v);
        } else {
            if inline.is_some() { arg_error(&format!("argument {}: ignored explicit argument {}", name, pyfmt::repr_str(&inline.unwrap()))); }
            flags.insert(name);
        }
    }
    let missing: Vec<&str> = ["--aln", "--datatype", "--out"].into_iter().filter(|k| !vals.contains_key(k)).collect();
    if !missing.is_empty() { arg_error(&format!("the following arguments are required: {}", missing.join(", "))); }
    if !extra.is_empty() { arg_error(&format!("unrecognized arguments: {}", extra.join(" "))); }
    let choice = |k: &str, opts: &[&str], default: &str| -> String {
        match vals.get(k) { None => default.to_string(), Some(v) => { if opts.contains(&v.as_str()) { v.clone() } else { arg_error(&format!("argument {}: invalid choice: {} (choose from {})", k, pyfmt::repr_str(v), opts.iter().map(|o| pyfmt::repr_str(o)).collect::<Vec<_>>().join(", "))) } } }
    };
    let int = |k: &str| -> Option<i64> { vals.get(k).map(|v| v.trim().replace('_', "").parse::<i64>().unwrap_or_else(|_| arg_error(&format!("argument {}: invalid int value: {}", k, pyfmt::repr_str(v))))) };
    let float = |k: &str| -> Option<f64> { vals.get(k).map(|v| { let t = v.trim().to_lowercase().replace('_', ""); let t = match t.as_str() { "infinity" | "+infinity" | "+inf" => "inf".to_string(), "-infinity" => "-inf".to_string(), _ => t };
        t.parse::<f64>().unwrap_or_else(|_| arg_error(&format!("argument {}: invalid float value: {}", k, pyfmt::repr_str(v)))) }) };
    let env_usize = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    Args {
        aln: vals["--aln"].clone(), datatype: choice("--datatype", &["dna", "aa"], ""), out: vals["--out"].clone(), partitions: vals.get("--partitions").cloned(), name: vals.get("--name").cloned(),
        n_comp: int("--n-comp").unwrap_or(5), fit_overlap: int("--fit-overlap").unwrap_or(20), alpha: float("--alpha").unwrap_or(0.05), donor_threshold: float("--donor-threshold").unwrap_or(2.0),
        threads: 0, memory: float("--memory"), restart: flags.contains("--restart"), progress: choice("--progress", &["auto", "bar", "none"], "auto"),
        no_report: flags.contains("--no-report"), all_tests: flags.contains("--all-tests"), mask_similar_only: flags.contains("--mask-similar-only"), mask_different_only: flags.contains("--mask-different-only"),
        no_broad_sample: flags.contains("--no-broad-sample"), candidates: int("--candidates").map(|v| if v < 1 { 0 } else { v as usize }), block_taxa: int("--block-taxa"), no_numba: flags.contains("--no-numba"),
        report_codes: choice("--report-codes", &["fails", "masked", "none"], "fails"), report_sequences: choice("--report-sequences", &["auto", "all", "fails", "masked", "none"], "auto"),
        stop_after: std::env::var("LOCUSMASKER_STOP_AFTER").ok().filter(|v| v == "rates" || v == "pairs"),
        dense_cells_max: env_usize("LOCUSMASKER_DENSE_CELLS_MAX", runz::DENSE_CELLS_MAX), best_match_all_max_taxa: env_usize("LOCUSMASKER_BEST_MATCH_ALL_MAX_TAXA", runz::BEST_MATCH_ALL_MAX_TAXA),
        full_m_max_taxa: env_usize("LOCUSMASKER_FULL_M_MAX_TAXA", prepare::FULL_M_MAX_TAXA), sweep_block_min: env_usize("LOCUSMASKER_SWEEP_BLOCK_MIN", prepare::SWEEP_BLOCK_MIN),
    }.with_threads(int("--threads").unwrap_or(1))
}

struct Asked(i64);
static ASKED: std::sync::OnceLock<Asked> = std::sync::OnceLock::new();
impl Args { fn with_threads(mut self, asked: i64) -> Args { let _ = ASKED.set(Asked(asked)); self.threads = asked.max(0) as usize; self } }

fn memory_estimate(a: &Args, n: usize, l: usize, loci: &input::Loci, c: Option<usize>, k_eff: usize, rc: bool) -> [u64; 7] {
    let nloc = loci.len() as u64; let n64 = n as u64; let c = c.unwrap_or(prepare::CANDIDATES).min(n) as u64; let full = n <= a.full_m_max_taxa;
    let longest = loci.iter().map(|x| x.1.len()).max().unwrap_or(0);
    let slab = if n <= a.sweep_block_min { 2 * nloc * n64 * n64 * (if longest < 65536 { 2 } else { 4 }) } else { 0 };
    // the alignment's letters, and its residues as bit planes (a presence plane and the planes of the code, each locus from a word)
    let planes: u64 = if a.datatype == "dna" { 3 } else { 6 }; let words: u64 = loci.iter().map(|x| x.1.len().div_ceil(64) as u64).sum();
    let x = n64 * l as u64 + n64 * planes * words * 8; let lists = n64 * c * 32; let closest = n64 * nloc * prepare::BAND_SLOTS as u64 * 24; let mfull = if full { 2 * n64 * n64 * 8 } else { 0 };
    let mrc = if rc { n64 * ((prepare::BS_POOL * k_eff) as u64).min(n64.saturating_sub(1)) * 12 } else { 0 };
    [x + lists + closest + mfull + slab + mrc, x, lists, closest, mfull, slab, mrc]
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = parse_args(&argv); let k_in = a.n_comp;
    if let Err(e) = std::fs::create_dir_all(&a.out) { die!("cannot write in {}: {}", a.out, input::os_err(&e)); }
    let t0 = std::time::Instant::now();
    let exists = |f: &str| std::path::Path::new(&format!("{}/{}", a.out, f)).exists();
    let mut stale: Vec<&str> = vec![];
    if !a.all_tests && exists("tests.tsv.gz") { stale.push("tests.tsv.gz"); }
    for f in ["report.html", "report.npz", "report_files"] { if a.no_report && exists(f) { stale.push(f); } }
    for f in ["cells.tsv", "pairs.npy", "similar.tsv", "per_locus.tsv", "report_data.js", "taxa.tsv", "loci.tsv", "summary.json"] { if exists(f) { stale.push(f); } }
    for f in ["report.html.part", "report.npz.part"] { if exists(f) { let _ = std::fs::remove_file(format!("{}/{}", a.out, f)); } }
    if let Err(e) = log::open(&format!("{}/run.log", a.out)) { die!("cannot write in {}: {}", a.out, input::os_err(&e)); }
    let _ = log::PROG.set(log::Progress::new(&a.progress, t0)); log::set_bar(log::prog().mode() == log::Mode::Bar);
    log!("LocusMasker {} ({}) on {}; Rust port ({}), {}", VERSION, env!("LM_GIT_COMMIT"), now_string(), env!("LM_RUSTC"), system_string());
    log!("command: {}", argv.join(" "));
    if !stale.is_empty() { log!("note: {} holds files this run does not write ({}): they are from an earlier run", a.out, stale.join(", ")); }
    let ncpu = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1); let asked = ASKED.get().map(|x| x.0).unwrap_or(1);
    a.threads = if asked <= 0 { ncpu } else { (asked as usize).min(ncpu).max(1) };
    if asked > ncpu as i64 { log!("threads: {} asked, {} CPUs available to this process: using {}", asked, ncpu, a.threads); }
    let _ = rayon::ThreadPoolBuilder::new().num_threads(a.threads).stack_size(16 << 20).build_global();
    let avail = mem_available();
    let budget: Option<u64> = match a.memory { Some(m) if m <= 0.0 => None, Some(m) => Some((m * (1u64 << 30) as f64) as u64), None => Some(avail / 2) };
    if let (Some(b), Some(m)) = (budget, a.memory) { if b > avail { log!("WARNING: --memory {} GB is more than the {:.1} GB available now", pyfmt::g(m), avail as f64 / (1u64 << 30) as f64); } }
    if k_in < 3 || k_in % 2 == 0 { die!("--n-comp must be an odd number of at least 3"); }
    if a.candidates == Some(0) { die!("--candidates must be at least 1"); }
    if let Some(b) = a.block_taxa { if b < 1 { die!("--block-taxa must be at least 1"); } }
    if a.mask_similar_only && a.mask_different_only { die!("--mask-similar-only and --mask-different-only are mutually exclusive"); }
    if let Some(p) = &a.partitions { if !std::path::Path::new(p).is_file() { die!("the partition file {} does not exist", p); } }
    let prog = log::prog();
    prog.start("read", 1);
    start_input_sha256(&a.aln);
    let mut inp = input::read_input(&a.aln, a.partitions.as_deref());
    prog.finish();
    let (n, l) = (inp.n, inp.l);
    let mut covered = vec![false; l];
    for (_, st) in &inp.loci { for &s in st { covered[s] = true; } }
    let nuncov = covered.iter().filter(|c| !**c).count();
    if nuncov != 0 {
        let unc: Vec<usize> = (0..l).filter(|&i| !covered[i]).collect(); let runs = report::ranges(&unc);
        let shown: Vec<String> = runs.iter().take(6).map(|r| if r[1] > r[0] { format!("{}-{}", r[0] + 1, r[1] + 1) } else { (r[0] + 1).to_string() }).collect();
        log!("WARNING: {} of the {} sites are in no {} and are ignored by every test and never masked (sites {}{})", commas(nuncov as i64), commas(l as i64),
             if inp.fmt == "nexus" && a.partitions.is_none() { "charset" } else { "partition" }, shown.join(", "), if runs.len() > 6 { ", ..." } else { "" });
    }
    if let Some(d) = inp.declared { if d != a.datatype { die!("--datatype {} was given but the NEXUS file declares {}", a.datatype, d); } }
    let dtype = a.datatype.clone();
    if inp.nexus.as_ref().map_or(true, |nx| nx.general) {
        let (gap, missing) = match &inp.nexus { Some(nx) => (nx.gap.clone(), nx.missing.clone()), None => (None, None) };
        let rows: Vec<&[u8]> = (0..n).map(|i| &inp.rows[i * l..(i + 1) * l]).collect();
        input::check_characters(&a.aln, &inp.names, &rows, &dtype, &gap, &missing);
    }
    let nloc = inp.loci.len();
    if nloc < 10 { die!("only {} loci defined: this is a phylogenomic test and needs at least 10", nloc); }
    if n < 5 { die!("only {} taxa: the test needs at least 5", n); }
    let mut k_eff = (k_in as usize).min(n - 1); k_eff -= 1 - k_eff % 2;
    let m_eff = (k_eff + 1) / 2;
    if (k_eff as i64) < k_in { log!("n-comp shrunk to {} ({} taxa)", k_eff, n); }
    log!("{} taxa, {} sites, {} loci, {}, {}{}", n, l, nloc, dtype, inp.fmt, if inp.gz { ", gzip-compressed" } else { "" });
    let me = memory_estimate(&a, n, l, &inp.loci, a.candidates, k_eff, !a.no_broad_sample);
    let parts = format!("{} of alignment (the letters and the bit planes), {} of candidate lists for {} taxa, {} of closest matches for {} taxa x {} loci{}{}{}",
                        fmem(me[1]), fmem(me[2]), commas(n as i64), fmem(me[3]), commas(n as i64), commas(nloc as i64),
                        if me[4] != 0 { format!(", {} of the full m matrix (kept up to {} taxa)", fmem(me[4]), commas(a.full_m_max_taxa as i64)) } else { String::new() },
                        if me[5] != 0 { format!(", {} of pair counts at every locus (kept up to {} taxa)", fmem(me[5]), commas(a.sweep_block_min as i64)) } else { String::new() },
                        if me[6] != 0 { format!(", {} of broad-sample pools", fmem(me[6])) } else { String::new() });
    match budget {
        None => log!("memory: about {} resident ({}); no budget (--memory 0)", fmem(me[0]), parts),
        Some(b) => {
            log!("memory: about {} resident ({}); budget {}{}", fmem(me[0]), parts, fmem(b), if a.memory.is_some() { "" } else { " (half of the memory available now; --memory sets it)" });
            if me[0] > b {
                die!("this run needs about {} of memory ({}), more than the budget of {}{}: --memory raises the budget (--memory 0 removes it), or give fewer taxa or loci",
                     fmem(me[0]), parts, fmem(b), if a.memory.is_some() { "" } else { ", half of what is available now" });
            }
        }
    }
    let mut p = prepare::prepare(&a, &inp.names, n, l, &inp.rows, &inp.loci, &dtype, k_eff, t0);
    let seq_policy = if a.report_sequences != "auto" { a.report_sequences.clone() } else if n * l <= std::env::var("LOCUSMASKER_REPORT_SEQ_ALL_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(REPORT_SEQ_ALL_MAX_DEFAULT) { "all".to_string() } else { "fails".to_string() };
    let seq_policy = if a.no_report { "none".to_string() } else { seq_policy };
    let codes_policy = if a.no_report { "none".to_string() } else { a.report_codes.clone() };
    let mut masked = std::mem::take(&mut inp.rows);
    let (rep, _nmasked) = runz::run_z(&a, &inp.names, &inp.loci, &mut p, &mut masked, l, k_eff, m_eff, &seq_policy, &codes_policy, nuncov);
    prog.start("write", 2);
    let out = input::write_masked(&a.aln, &a.out, &inp, &masked);
    prog.tick();
    if !a.no_report { report::write_report(&a, &inp.names, &inp.loci, &rep, &dtype, inp.fmt, inp.gz, k_in as usize, k_eff, m_eff, t0, &argv); }
    prog.finish();
    let _ = std::fs::remove_dir_all(format!("{}/restart", a.out));
    log!("outputs: {}, {}/masked.csv{}{}, {}/run.log", out, a.out, if a.all_tests { format!(", {}/tests.tsv.gz", a.out) } else { String::new() },
         if a.no_report { String::new() } else { format!(", {0}/report.html (+ report_files/), {0}/report.npz", a.out) }, a.out);
    log!("done: total execution time {}", elapsed_string(t0.elapsed().as_secs_f64()));
    log::close();
}
