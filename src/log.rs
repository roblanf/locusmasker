//! stderr prose (and run.log) with every line behind its time, stage progress, and the fatal exit of sys.exit(message)
use std::fs::File;
use std::io::{IsTerminal, Write};
use std::sync::Mutex;
use std::time::Instant;

pub struct LogState { pub fh: Option<File>, pub bar: bool }
pub static LOG: Mutex<LogState> = Mutex::new(LogState { fh: None, bar: false });

pub fn open(path: &str) -> std::io::Result<()> { LOG.lock().unwrap().fh = Some(File::create(path)?); Ok(()) }
pub fn close() { LOG.lock().unwrap().fh = None; }
pub fn set_bar(b: bool) { LOG.lock().unwrap().bar = b; }

/// every line of a message behind the local time: [2026-11-13 07:30:01] line
fn stamped(msg: &str) -> String { let t = crate::now_string(); msg.split('\n').map(|l| format!("[{t}] {l}")).collect::<Vec<_>>().join("\n") }

pub fn log(msg: &str) {
    let msg = &stamped(msg);
    let mut l = LOG.lock().unwrap(); let mut e = std::io::stderr().lock();
    if l.bar { let _ = e.write_all(b"\r\x1b[K"); }
    let _ = writeln!(e, "{msg}"); let _ = e.flush();
    if let Some(f) = l.fh.as_mut() { let _ = writeln!(f, "{msg}"); let _ = f.flush(); }
}
pub fn detail(msg: &str) {
    let msg = &stamped(msg);
    let mut l = LOG.lock().unwrap();
    if let Some(f) = l.fh.as_mut() { let _ = writeln!(f, "{msg}"); }
    if !l.bar { let _ = writeln!(std::io::stderr().lock(), "{msg}"); }
}
#[macro_export]
macro_rules! log { ($($a:tt)*) => { $crate::log::log(&format!($($a)*)) }; }

/// sys.exit(message): the message on stderr, exit status 1
pub fn die(msg: &str) -> ! {
    { let mut l = LOG.lock().unwrap(); if let Some(f) = l.fh.as_mut() { let _ = f.flush(); } }
    eprintln!("{msg}");
    std::process::exit(1)
}
#[macro_export]
macro_rules! die { ($($a:tt)*) => { $crate::log::die(&format!($($a)*)) }; }

#[derive(PartialEq, Clone, Copy)]
pub enum Mode { None, Bar, Sparse }

struct P { mode: Mode, t0: Instant, stage: Option<String>, total: u64, done: u64, extra: String, ts: Instant, last: f64, next_frac: f64,
           durations: Vec<(String, f64)>, fill: char, empty: char }
pub struct Progress { p: Mutex<P> }
pub static PROG: std::sync::OnceLock<Progress> = std::sync::OnceLock::new();
pub fn prog() -> &'static Progress { PROG.get_or_init(|| Progress::new("none", Instant::now())) }

impl Progress {
    pub fn new(mode: &str, t0: Instant) -> Progress {
        let mode = match mode { "auto" => if std::io::stderr().is_terminal() { Mode::Bar } else { Mode::Sparse }, "bar" => Mode::Bar, "sparse" => Mode::Sparse, _ => Mode::None };
        Progress { p: Mutex::new(P { mode, t0, stage: None, total: 1, done: 0, extra: String::new(), ts: t0, last: 0.0, next_frac: 0.02, durations: vec![], fill: '\u{2588}', empty: '\u{2591}' }) }
    }
    pub fn mode(&self) -> Mode { self.p.lock().unwrap().mode }
    pub fn durations(&self) -> Vec<(String, f64)> { self.p.lock().unwrap().durations.clone() }
    pub fn start(&self, stage: &str, total: usize) { self.start_extra(stage, total, "") }
    pub fn start_extra(&self, stage: &str, total: usize, extra: &str) {
        let mut p = self.p.lock().unwrap();
        if p.stage.is_some() { Self::fin(&mut p); }
        p.stage = Some(stage.to_string()); p.total = (total as u64).max(1); p.done = 0; p.extra = extra.to_string();
        p.ts = Instant::now(); p.last = 0.0; p.next_frac = 0.02;
        let now = p.ts; Self::emit(&mut p, now, false);
    }
    pub fn tick(&self) { self.tick_k(1, None) }
    pub fn tick_k(&self, k: usize, extra: Option<String>) {
        let mut p = self.p.lock().unwrap();
        if p.stage.is_none() { return; }
        p.done += k as u64;
        if let Some(e) = extra { p.extra = e; }
        let now = Instant::now();
        if p.mode == Mode::Sparse {
            if p.done >= p.total || p.done as f64 / p.total as f64 >= p.next_frac {
                p.next_frac = (p.done as f64 / p.total as f64 * 50.0 + 1.0).floor() / 50.0; Self::emit(&mut p, now, false);
            }
        } else if now.duration_since(p.t0).as_secs_f64() - p.last >= 0.5 || p.done >= p.total { Self::emit(&mut p, now, false); }
    }
    pub fn finish(&self) { let mut p = self.p.lock().unwrap(); Self::fin(&mut p); }
    fn fin(p: &mut P) {
        let Some(stage) = p.stage.clone() else { return };
        p.done = p.total; let now = Instant::now();
        let d = crate::pyfmt::round(now.duration_since(p.ts).as_secs_f64(), 2);
        if let Some(e) = p.durations.iter_mut().find(|x| x.0 == stage) { e.1 = d; } else { p.durations.push((stage, d)); }
        Self::emit(p, now, true); p.stage = None;
    }
    fn emit(p: &mut P, now: Instant, last: bool) {
        if p.mode == Mode::None { return; }
        let el = now.duration_since(p.t0).as_secs_f64(); let st = now.duration_since(p.ts).as_secs_f64();
        let stage = p.stage.clone().unwrap_or_default(); let mut e = std::io::stderr().lock();
        if p.mode == Mode::Sparse {
            let _ = writeln!(e, "[{}] progress\t{}\t{}\t{}\t{:.1}\t{}", crate::now_string(), stage, p.done, p.total, el, p.extra); let _ = e.flush(); p.last = el; return;
        }
        let frac = p.done as f64 / p.total as f64; let w = 28usize; let filled = ((frac * w as f64).round_ties_even() as usize).min(w);
        let eta = if frac > 0.0 && frac < 1.0 { st / frac - st } else { 0.0 };
        let left = if last { String::new() } else if frac > 0.0 { format!("{:4.0} s left", eta) } else { String::new() };
        let line = format!("\r{:<12} {}{} {:>8}/{:<8} {:5.0} s {:>11}  {}", stage, p.fill.to_string().repeat(filled), p.empty.to_string().repeat(w - filled),
                           crate::pyfmt::commas(p.done as i64), crate::pyfmt::commas(p.total as i64), st, left, p.extra);
        let line: String = line.chars().take(220).collect();
        let _ = write!(e, "{}\x1b[K{}", line, if last { "\n" } else { "" }); let _ = e.flush(); p.last = el;
    }
}
