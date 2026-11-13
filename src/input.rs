//! Input and output of alignments: FASTA, relaxed PHYLIP, NEXUS (a fast path for plain files, a general reader for
//! comments and quoted names), RAxML-style partitions, the character checks, and the masked alignment writers.
use crate::pyfmt::{commas, repr_bytes, repr_str};
use crate::{die, log};
use std::collections::{HashMap, HashSet};

pub const PROTEIN: &str = "ACDEFGHIKLMNPQRSTVWY";
pub const DNA: &str = "ACGT";

pub type Loci = Vec<(String, Vec<usize>)>;

/// what the writer needs from a NEXUS file
pub struct NexusInfo {
    pub datatype: String, pub missing: Option<String>, pub gap: Option<String>,
    pub charsets: Vec<(String, Vec<usize>)>, pub taxsets: Vec<(String, Vec<String>)>,
    pub charpartitions: Vec<(String, Vec<(String, Vec<usize>)>)>,
    pub general: bool,                 // read by the general reader (written back the way Bio.Nexus writes)
}

pub struct Input {
    pub names: Vec<String>, pub n: usize, pub l: usize, pub rows: Vec<u8>,     // rows: n x l bytes
    pub loci: Loci, pub fmt: &'static str, pub declared: Option<&'static str>, pub nexus: Option<NexusInfo>,
    pub gz: bool,                                                               // the file is gzip-compressed: the masked alignment is written so too
}

pub fn os_err(e: &std::io::Error) -> String {
    match e.raw_os_error() {
        Some(code) => unsafe { std::ffi::CStr::from_ptr(libc::strerror(code)).to_string_lossy().into_owned() },
        None => e.to_string(),
    }
}

fn is_ws(b: u8) -> bool { matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) }
fn strip(b: &[u8]) -> &[u8] {
    let mut s = 0; let mut e = b.len();
    while s < e && is_ws(b[s]) { s += 1; }
    while e > s && is_ws(b[e - 1]) { e -= 1; }
    &b[s..e]
}
fn lstrip(b: &[u8]) -> &[u8] { let mut s = 0; while s < b.len() && is_ws(b[s]) { s += 1; } &b[s..] }
/// bytes.split(None, 1): the first word and the rest (leading whitespace of the rest removed), or one or no word
fn split1(b: &[u8]) -> Vec<&[u8]> {
    let b = lstrip(b); if b.is_empty() { return vec![]; }
    let mut i = 0; while i < b.len() && !is_ws(b[i]) { i += 1; }
    let rest = lstrip(&b[i..]);
    if rest.is_empty() { vec![&b[..i]] } else { vec![&b[..i], rest] }
}
fn squash(b: &[u8]) -> Vec<u8> { b.iter().copied().filter(|c| !is_ws(*c)).collect() }
/// bytes.splitlines(): at \n, \r and \r\n
fn first_line(b: &[u8]) -> &[u8] { let e = b.iter().position(|c| *c == b'\n' || *c == b'\r').unwrap_or(b.len()); &b[..e] }

// ------------------------------------------------------------------ FASTA
pub fn read_fasta(txt: &[u8]) -> (Vec<String>, Vec<Vec<u8>>) {
    let body = lstrip(txt);
    if body.is_empty() { return (vec![], vec![]); }
    if body[0] != b'>' {
        let first = strip(first_line(body)); let first = &first[..first.len().min(40)];
        die!("FASTA: text before the first header line ('>'): {} (a sequence whose header line is missing?)", repr_str(&String::from_utf8_lossy(first)));
    }
    let body = &body[1..];
    let mut names = vec![]; let mut seqs = vec![];
    let mut start = 0usize;
    let mut cuts = vec![];
    let mut i = 0;
    while i + 1 < body.len() { if body[i] == b'\n' && body[i + 1] == b'>' { cuts.push(i); i += 2; } else { i += 1; } }
    cuts.push(body.len());
    for &c in &cuts {
        let rec = &body[start..c]; start = c + 2;
        let (header, seq) = match rec.iter().position(|x| *x == b'\n') { Some(p) => (&rec[..p], &rec[p + 1..]), None => (rec, &rec[rec.len()..]) };
        let w = split1(header);
        if w.is_empty() { die!("FASTA: a header line ('>') has no name"); }
        let nm = match std::str::from_utf8(w[0]) { Ok(s) => s.to_string(), Err(_) => die!("FASTA: the name {} is not UTF-8 text", repr_bytes(&w[0][..w[0].len().min(40)])) };
        let mut sq: Vec<u8> = seq.iter().copied().filter(|c| *c != b'\n' && *c != b'\r').collect();
        if sq.iter().any(|c| *c == b' ' || *c == b'\t') { sq = squash(&sq); }
        names.push(nm); seqs.push(sq);
    }
    (names, seqs)
}

// ------------------------------------------------------------------ PHYLIP
fn py_space(c: char) -> bool { c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c) }
fn py_lines(s: &str) -> Vec<&str> {
    let mut out = vec![]; let mut start = 0; let b: Vec<(usize, char)> = s.char_indices().collect(); let mut i = 0;
    while i < b.len() {
        let (p, c) = b[i];
        let brk = matches!(c, '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}');
        if brk {
            out.push(&s[start..p]);
            if c == '\r' && i + 1 < b.len() && b[i + 1].1 == '\n' { i += 1; }
            start = if i + 1 < b.len() { b[i + 1].0 } else { s.len() };
        }
        i += 1;
    }
    if start < s.len() { out.push(&s[start..]); }
    out
}
fn py_split1(s: &str) -> Vec<&str> {
    let s = s.trim_start_matches(py_space); if s.is_empty() { return vec![]; }
    match s.find(py_space) { None => vec![s], Some(p) => { let r = s[p..].trim_start_matches(py_space); if r.is_empty() { vec![&s[..p]] } else { vec![&s[..p], r] } } }
}
fn py_squash(s: &str) -> String { s.chars().filter(|c| !py_space(*c)).collect() }
fn py_isdigit(s: &str) -> bool { !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) }
fn trunc(s: &str, n: usize) -> String { s.chars().take(n).collect() }

pub fn read_phylip(txt: &str) -> (Vec<String>, Vec<String>) {
    let lines: Vec<&str> = py_lines(txt).into_iter().filter(|l| !l.trim_matches(py_space).is_empty()).collect();
    let hdr: Vec<&str> = if lines.is_empty() { vec![] } else { lines[0].split(py_space).filter(|x| !x.is_empty()).collect() };
    if hdr.len() < 2 || !py_isdigit(hdr[0]) || !py_isdigit(hdr[1]) {
        if lines.is_empty() { die!("PHYLIP: the file is empty"); }
        die!("cannot parse PHYLIP: the first line must give the number of sequences and the number of sites, not {}", repr_str(&trunc(lines[0].trim_matches(py_space), 40)));
    }
    let n: usize = hdr[0].parse().unwrap_or(usize::MAX); let l: usize = hdr[1].parse().unwrap_or(usize::MAX); let body = &lines[1..];
    let sequential = || {
        let mut names: Vec<String> = vec![]; let mut seqs: Vec<String> = vec![];
        for line in body {
            let p = py_split1(line);
            if p.len() == 2 && names.len() < n { names.push(p[0].to_string()); seqs.push(py_squash(p[1])); }
            else if !names.is_empty() { let k = seqs.len() - 1; seqs[k].push_str(&py_squash(line)); }
        }
        (names, seqs)
    };
    let interleaved = || {
        let mut names: Vec<String> = vec![]; let mut seqs: Vec<String> = vec![];
        for (i, line) in body.iter().enumerate() {
            let p = py_split1(line);
            if i < n { names.push(p[0].to_string()); seqs.push(if p.len() == 2 { py_squash(p[1]) } else { String::new() }); }
            else { let k = i % n; let add = py_squash(if p.len() == 2 && p[0] == names[k] { p[1] } else { line }); seqs[k].push_str(&add); }
        }
        (names, seqs)
    };
    let slen = |s: &String| s.chars().count();
    let (mut names, mut seqs) = sequential();
    if names.len() != n || seqs.iter().any(|s| slen(s) != l) { let r = interleaved(); names = r.0; seqs = r.1; }
    if names.len() == n && seqs.iter().any(|s| slen(s) != l) {
        let k = seqs.iter().position(|s| slen(s) != l).unwrap();
        die!("PHYLIP: the header says {} sites but {} has {}: the sequences are not aligned (every sequence must have {} characters), \
              or the file is laid out in a form the reader does not handle (relaxed one-line and interleaved layouts are supported; try FASTA plus a partition file)",
             commas(l as i64), names[k], commas(slen(&seqs[k]) as i64), commas(l as i64));
    }
    if names.len() != n || seqs.iter().any(|s| slen(s) != l) {
        die!("cannot parse PHYLIP: the header says {} sequences of {} sites, {} were found; relaxed one-line or interleaved layouts are supported; \
              try FASTA plus a RAxML-style partition file", n, commas(l as i64), names.len());
    }
    (names, seqs)
}

// ------------------------------------------------------------------ partitions
fn read_text_universal(path: &str) -> Result<String, std::io::Error> {
    let b = std::fs::read(path)?;
    let s = String::from_utf8(b).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "not utf-8"))?;
    Ok(s.replace("\r\n", "\n").replace('\r', "\n"))
}

fn plural(k: usize) -> (&'static str, &'static str) { if k > 1 { ("s", "are") } else { ("", "is") } }

pub fn parse_raxml_partitions(path: &str, l: usize) -> Loci {
    let mut loci: Loci = vec![]; let mut seen: HashSet<String> = HashSet::new();
    let mut owner: Vec<i64> = vec![-1; l];
    let txt = match read_text_universal(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::InvalidData => die!("the partition file {path} is not a text file"),
        Err(e) => die!("cannot read the partition file {path}: {}", os_err(&e)),
    };
    let re_range = regex::Regex::new(r"^(\d+)\s*-\s*(\d+)(?:\s*\\\s*(\d+))?$").unwrap();
    let re_int = regex::Regex::new(r"^\d+$").unwrap();
    let int = |s: &str| -> u64 { s.parse::<u64>().unwrap_or(u64::MAX) };
    for line in py_lines(&txt) {
        let line = line.split('#').next().unwrap().trim_matches(py_space);
        if line.is_empty() || !line.contains('=') { continue; }
        let (lhs, rhs) = line.split_once('=').unwrap();
        let name = lhs.rsplit(',').next().unwrap().trim_matches(py_space).to_string();
        if seen.contains(&name) { die!("partition name {} appears twice", repr_str(&name)); }
        seen.insert(name.clone());
        let mut sites: Vec<usize> = vec![];
        for part in rhs.replace(';', "").split(',') {
            let part = part.trim_matches(py_space);
            if part.is_empty() { continue; }
            if let Some(m) = re_range.captures(part) {
                let (a, b, s) = (int(&m[1]), int(&m[2]), m.get(3).map(|x| int(x.as_str())).unwrap_or(1));
                if a < 1 || b < a || s < 1 { die!("bad range {} in partition {}: sites are 1-based and a-b needs a <= b", repr_str(part), name); }
                if b > l as u64 { die!("partition {} runs past the alignment ({} > {} sites)", name, commas(b as i128), commas(l as i64)); }
                let mut x = a - 1; while x < b { sites.push(x as usize); x += s; }
            } else if re_int.is_match(part) {
                let v = int(part);
                if v < 1 { die!("bad site {} in partition {}: sites are 1-based", repr_str(part), name); }
                if v > l as u64 { die!("partition {} runs past the alignment ({} > {} sites)", name, commas(v as i128), commas(l as i64)); }
                sites.push(v as usize - 1);
            } else {
                die!("cannot parse partition range {} in {}", repr_str(part), repr_str(line));
            }
        }
        if sites.is_empty() { die!("partition {} has no sites", name); }
        sites.sort_unstable(); sites.dedup();
        let dup: Vec<usize> = sites.iter().copied().filter(|&x| owner[x] >= 0).collect();
        if !dup.is_empty() {
            let first = dup[0]; let mut others: Vec<String> = dup.iter().map(|&x| loci[owner[x] as usize].0.clone()).collect(); others.sort(); others.dedup();
            let (s, are) = plural(dup.len());
            die!("site {} is in partition {} and in partition {}: a site may be in one partition only ({} site{} of {} {} already in {}; the alignment has {} sites, a site may appear only once)",
                 first + 1, name, others[0], dup.len(), s, name, are, others.join(", "), l);
        }
        for &x in &sites { owner[x] = loci.len() as i64; }
        loci.push((name, sites));
    }
    loci
}

// ------------------------------------------------------------------ the characters an alignment may hold
fn alphabet_chars(dtype: &str) -> &'static str { if dtype == "dna" { "GATCURYWSMKHBVDN" } else { "BZXACDEFGHIKLMNPQRSTVWY*" } }
fn ambiguity_chars(dtype: &str) -> &'static str { if dtype == "dna" { "RYWSMKHBVDN" } else { "BZX*" } }
fn spaced(s: &str) -> String { s.replace('*', "").chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" ") }
fn py_repr_opt(s: &Option<String>) -> String { match s { Some(x) => repr_str(x), None => "None".into() } }

/// exit with every illegal character, how often it occurs and where it first occurs; warn when the data look like the other datatype
/// how often each byte occurs in the alignment
pub fn byte_counts(flat: &[u8]) -> [i64; 256] {
    use rayon::prelude::*;
    flat.par_chunks(1 << 22).map(|ch| { let mut c = [0i64; 256]; for &b in ch { c[b as usize] += 1; } c })
        .reduce(|| [0i64; 256], |mut x, y| { for (p, q) in x.iter_mut().zip(y.iter()) { *p += q; } x })
}

pub fn check_characters(path: &str, names: &[String], rows: &[&[u8]], dtype: &str, gap: &Option<String>, missing: &Option<String>) {
    let mut legal: String = alphabet_chars(dtype).to_string(); legal = format!("{}{}-?{}{}", legal, legal.to_lowercase(), gap.clone().unwrap_or_default(), missing.clone().unwrap_or_default());
    let mut ok = [false; 256]; for c in legal.chars() { if (c as u32) < 256 { ok[c as usize] = true; } }
    let counts = { use rayon::prelude::*;
        rows.par_iter().map(|r| { let mut c = [0i64; 256]; for &b in r.iter() { c[b as usize] += 1; } c }).reduce(|| [0i64; 256], |mut x, y| { for (p, q) in x.iter_mut().zip(y.iter()) { *p += q; } x }) };
    let bad: Vec<usize> = (0..256).filter(|&c| counts[c] > 0 && !ok[c]).collect();
    let sumc = |s: &str| -> i64 { s.bytes().map(|c| counts[c as usize]).sum() };
    let letters = sumc("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"); let acgt = sumc("ACGTUNacgtun");
    if !bad.is_empty() {
        let mut first: HashMap<usize, (String, usize)> = HashMap::new();
        for (t, r) in names.iter().zip(rows) {
            for &c in &bad {
                if !first.contains_key(&c) { if let Some(p) = r.iter().position(|&b| b as usize == c) { first.insert(c, (t.clone(), p + 1)); } }
            }
            if first.len() == bad.len() { break; }
        }
        let mut shown: Vec<String> = bad.iter().take(8).map(|&c| {
            let ch = if (32..127).contains(&c) { (c as u8 as char).to_string() } else { format!("0x{:02x}", c) };
            format!("{} x{} (first in {} at site {})", repr_str(&ch), commas(counts[c]), first[&c].0, commas(first[&c].1 as i64))
        }).collect();
        if bad.len() > 8 { shown.push(format!("and {} more", bad.len() - 8)); }
        let hint = if dtype == "dna" && letters != 0 && (acgt as f64) < 0.5 * letters as f64 { " Fewer than half the letters are A, C, G, T or N: is this a protein alignment (--datatype aa)?" } else { "" };
        let decl = if gap.as_deref().map_or(false, |g| !g.is_empty() && g != "-") || missing.as_deref().map_or(false, |m| !m.is_empty() && m != "?") {
            format!(" (this NEXUS file declares gap={} and missing={})", py_repr_opt(gap), py_repr_opt(missing)) } else { String::new() };
        let base = std::path::Path::new(path).file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
        die!("illegal characters in the alignment {}: {}. The {} alphabet is {}{}, with the ambiguity codes {} as missing data, '-' for a gap and '?' for missing data{}.{}",
             base, shown.join("; "), dtype, spaced(alphabet_chars(dtype)), if dtype == "dna" { " (U is read as T)" } else { " and * for a stop" },
             spaced(ambiguity_chars(dtype)), decl, hint);
    }
    if dtype == "aa" && letters != 0 && acgt as f64 > 0.95 * letters as f64 {
        log!("WARNING: {:.1}% of the letters are A, C, G, T, U or N: this looks like a nucleotide alignment, but --datatype aa was given", 100.0 * acgt as f64 / letters as f64);
    }
}

// ------------------------------------------------------------------ NEXUS
const NEX_PUNCT: &str = "()[]{}\\,;:=*\\'\"`+-<>";

/// a taxon or set name as Bio.Nexus writes it: quoted when it holds whitespace or punctuation
pub fn safename(name: &str) -> String {
    let safe = name.replace('\'', "''");
    if safe.chars().any(|c| c == ' ' || c == '\t' || c == '\n' || NEX_PUNCT.contains(c)) { format!("'{}'", safe) } else { safe }
}

/// site list (0-based) -> NEXUS ranges (1-based), exactly as Bio.Nexus writes them
pub fn compact4nexus(orig: &[usize]) -> String {
    if orig.is_empty() { return String::new(); }
    let mut l: Vec<f64> = { let mut v = orig.to_vec(); v.sort_unstable(); v.dedup(); v.into_iter().map(|x| x as f64).collect() };
    let last = *l.last().unwrap(); l.push(last + 0.5);
    let mut short: Vec<String> = vec![]; let mut clist: &[f64] = &l;
    while clist.len() > 1 {
        let mut step = 1.0f64; let mut cut = None;
        for (i, &x) in clist.iter().enumerate() {
            if x == clist[0] + i as f64 * step { continue; }
            else if i == 1 && clist.len() > 3 && clist[i + 1] - x == x - clist[0] { step = x - clist[0]; }
            else {
                let sub = &clist[..i];
                if sub.len() == 1 { short.push(format!("{}", sub[0] as i64 + 1)); }
                else if step == 1.0 { short.push(format!("{}-{}", sub[0] as i64 + 1, sub[sub.len() - 1] as i64 + 1)); }
                else { short.push(format!("{}-{}\\{}", sub[0] as i64 + 1, sub[sub.len() - 1] as i64 + 1, step as i64)); }
                cut = Some(i); break;
            }
        }
        match cut { Some(i) => clist = &clist[i..], None => break }
    }
    short.join(" ")
}

enum NexErr { Unsupported, }

fn basename(path: &str) -> String { std::path::Path::new(path).file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default() }

struct NexRead { names: Vec<String>, rows: Vec<Vec<u8>>, info: NexusInfo }

fn find(hay: &[u8], needle: u8, from: usize, to: usize) -> Option<usize> { if from >= to { return None; } hay[from..to].iter().position(|c| *c == needle).map(|p| p + from) }

fn read_nexus_fast(path: &str, data: &[u8]) -> Result<NexRead, NexErr> {
    use regex::bytes::Regex;
    let un = || Err(NexErr::Unsupported);
    if data.iter().any(|c| matches!(c, b'[' | b'\'' | b'"' | b'(' | b'{')) { return un(); }
    let re_head = Regex::new(r"(?i-u)^\s*#NEXUS[ \t]*\r?\n").unwrap();
    let re_begin = Regex::new(r"(?i-u)^\s*begin\s+(\w+)\s*;").unwrap();
    let re_end = Regex::new(r"(?im-u)^[ \t]*end(?:block)?[ \t]*;").unwrap();
    let re_matrix = Regex::new(r"(?i-u)\bmatrix\b").unwrap();
    let Some(m) = re_head.find(data) else { return un() };
    let mut pos = m.end(); let mut datab: Option<(usize, usize)> = None; let mut setsb: Option<&[u8]> = None;
    loop {
        let Some(c) = re_begin.captures(&data[pos..]) else {
            if !strip(&data[pos..]).is_empty() { return un(); }
            break;
        };
        let kind = String::from_utf8_lossy(&c[1]).to_lowercase(); let body_start = pos + c.get(0).unwrap().end();
        let mut from = body_start;
        if kind == "data" || kind == "characters" {
            if let Some(mi) = re_matrix.find_at(data, body_start) { if let Some(semi) = find(data, b';', mi.end(), data.len()) { from = semi; } }
        }
        let Some(e) = re_end.find_at(data, from) else { return un() };
        let span = (body_start, e.start());
        if kind == "data" || kind == "characters" { if datab.is_some() { return un(); } datab = Some(span); }
        else if kind == "sets" { if setsb.is_some() { return un(); } setsb = Some(&data[span.0..span.1]); }
        else { return un(); }
        pos = e.end();
    }
    let Some(datab) = datab else { return un() };
    let Some(mi) = re_matrix.find_at(&data[..datab.1], datab.0) else { return un() };
    let Ok(header) = std::str::from_utf8(&data[datab.0..mi.start()]) else { return un() };
    if !header.is_ascii() { return un(); }
    let mstart = mi.end();
    let (mut ntax, mut nchar): (Option<usize>, Option<usize>) = (None, None);
    let mut opts: Vec<(String, String)> = vec![];
    let re_dim = regex::Regex::new(r"(\w+)\s*=\s*(\d+)").unwrap(); let re_dimk = regex::Regex::new(r"\w+\s*=").unwrap();
    let re_fmt = regex::Regex::new(r"\w+(?:\s*=\s*\S+)?").unwrap();
    for c in header.split(';').map(|c| c.trim_matches(py_space)).filter(|c| !c.is_empty()) {
        let w = py_split1(c); let key = w[0].to_lowercase(); let rest = if w.len() > 1 { w[1] } else { "" };
        if key == "dimensions" {
            for cap in re_dim.captures_iter(rest) {
                let k = cap[1].to_lowercase(); let v: usize = cap[2].parse().unwrap_or(usize::MAX);
                if k == "ntax" { ntax = Some(v); } else if k == "nchar" { nchar = Some(v); } else { return un(); }
            }
            if re_dimk.find_iter(rest).count() != rest.split(py_space).filter(|x| !x.is_empty()).count() { return un(); }
        } else if key == "format" {
            for tok in re_fmt.find_iter(rest) {
                let tok = tok.as_str();
                let (k, v) = match tok.split_once('=') { Some((k, v)) => (k.trim_matches(py_space).to_lowercase(), Some(v.trim_matches(py_space).to_string())), None => (tok.trim_matches(py_space).to_lowercase(), None) };
                let set = |opts: &mut Vec<(String, String)>, k: &str, v: String| { if let Some(e) = opts.iter_mut().find(|x| x.0 == k) { e.1 = v; } else { opts.push((k.to_string(), v)); } };
                if (k == "datatype" || k == "missing" || k == "gap") && v.is_some() { set(&mut opts, &k, v.unwrap()); }
                else if k == "interleave" { set(&mut opts, &k, v.filter(|x| !x.is_empty()).unwrap_or_else(|| "yes".into()).to_lowercase()); }
                else { return un(); }
            }
        } else { return un(); }
    }
    let (Some(ntax), Some(nchar)) = (ntax, nchar) else { return un() };
    let opt = |k: &str| opts.iter().find(|x| x.0 == k).map(|x| x.1.clone());
    let dt = opt("datatype").unwrap_or_default().to_lowercase();
    if !matches!(dt.as_str(), "dna" | "nucleotide" | "rna" | "protein") { return un(); }
    let first_char = |s: String| s.chars().next().map(|c| c.to_string()).unwrap_or_default();
    let missing = opt("missing").map(first_char); let gap = opt("gap").map(first_char);
    let interleave = matches!(opt("interleave").as_deref().unwrap_or("no"), "yes" | "");
    let Some(end) = find(data, b';', mstart, datab.1) else { return un() };
    if !strip(&data[end + 1..datab.1]).is_empty() { return un(); }
    let lines = || {
        let mut out: Vec<&[u8]> = vec![]; let mut p = mstart;
        while p < end {
            let q = find(data, b'\n', p, end).unwrap_or(end);
            let ln = strip(&data[p..q]); if !ln.is_empty() { out.push(ln); }
            p = q + 1;
        }
        out
    };
    let mut names: Vec<String> = vec![]; let mut index: HashMap<String, usize> = HashMap::new(); let mut rows: Vec<Vec<u8>> = vec![];
    let ascii = |b: &[u8]| -> Result<String, NexErr> { if b.is_ascii() { Ok(String::from_utf8_lossy(b).into_owned()) } else { Err(NexErr::Unsupported) } };
    if interleave {
        for ln in lines() {
            let w = split1(ln);
            if w.len() != 2 { return un(); }
            let nm = ascii(w[0])?;
            let k = *index.entry(nm.clone()).or_insert_with(|| { names.push(nm.clone()); rows.push(vec![]); rows.len() - 1 });
            rows[k].extend(squash(w[1]));
        }
    } else {
        let mut pending: Option<usize> = None;
        for ln in lines() {
            match pending {
                None => {
                    let w = split1(ln); let nm = ascii(w[0])?;
                    if index.contains_key(&nm) { return un(); }
                    index.insert(nm.clone(), rows.len()); names.push(nm);
                    rows.push(if w.len() > 1 { squash(w[1]) } else { vec![] });
                    pending = if rows[rows.len() - 1].len() < nchar { Some(rows.len() - 1) } else { None };
                }
                Some(k) => { rows[k].extend(squash(ln)); if rows[k].len() >= nchar { pending = None; } }
            }
        }
    }
    if names.len() != ntax || rows.iter().any(|r| r.len() != nchar) {
        if !interleave {
            let mut n2: Vec<String> = vec![]; let mut r2: Vec<usize> = vec![]; let mut seen: HashSet<String> = HashSet::new();
            for ln in lines() {
                let w = split1(ln); let nm = String::from_utf8_lossy(w[0]).into_owned();
                if seen.contains(&nm) { break; }
                seen.insert(nm.clone()); n2.push(nm); r2.push(if w.len() > 1 { squash(w[1]).len() } else { 0 });
            }
            if n2.len() == ntax {
                // the most common length (the first to reach the highest count, as Counter.most_common)
                let mut order: Vec<usize> = vec![]; let mut cnt: HashMap<usize, usize> = HashMap::new();
                for &x in &r2 { if !cnt.contains_key(&x) { order.push(x); } *cnt.entry(x).or_insert(0) += 1; }
                let mut l2 = order[0]; for &x in &order { if cnt[&x] > cnt[&l2] { l2 = x; } }
                let off: Vec<usize> = (0..n2.len()).filter(|&i| r2[i] != nchar).collect();
                if !off.is_empty() && off.len() == ntax && cnt.len() == 1 {
                    die!("NEXUS {}: every sequence has {} characters but the dimensions line says nchar={}: fix nchar", basename(path), commas(l2 as i64), commas(nchar as i64));
                }
                if !off.is_empty() && (l2 == nchar || cnt[&l2] >= 2) {
                    let t = off[0];
                    die!("NEXUS {}: taxon {} has {} characters but nchar is {} ({} of {} taxa differ): the sequences are not aligned \
                          (every sequence must have nchar characters), or the matrix is interleaved without 'interleave' in its format line",
                         basename(path), n2[t], commas(r2[t] as i64), commas(nchar as i64), off.len(), ntax);
                }
            }
        }
        return un();
    }
    {
        let rr: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        check_characters(path, &names, &rr, if dt == "protein" { "aa" } else { "dna" }, &gap, &missing);
    }
    let mut charsets: Vec<(String, Vec<usize>)> = vec![]; let mut taxsets: Vec<(String, Vec<String>)> = vec![];
    let sets_txt = match setsb { Some(b) => { if !b.is_ascii() { return un(); } String::from_utf8_lossy(b).into_owned() } None => String::new() };
    let re_set = regex::Regex::new(r"(?is)^(charset|taxset)\s+(\S+)\s*=\s*(.*)$").unwrap();
    let re_tok = regex::Regex::new(r"^(\d+)(?:-(\d+))?(?:\\(\d+))?$").unwrap();
    for c in sets_txt.split(';') {
        let c = c.trim_matches(py_space);
        if c.is_empty() { continue; }
        let Some(m) = re_set.captures(c) else { return un() };
        let kind = m[1].to_lowercase(); let nm = m[2].to_string(); let body: Vec<&str> = m.get(3).unwrap().as_str().split(py_space).filter(|x| !x.is_empty()).collect();
        if kind == "charset" {
            let mut sites: Vec<usize> = vec![];
            for tok in body {
                let Some(r) = re_tok.captures(tok) else { return un() };
                let a: usize = r[1].parse().unwrap_or(usize::MAX); let b: usize = r.get(2).map(|x| x.as_str().parse().unwrap_or(usize::MAX)).unwrap_or(a);
                let st: usize = r.get(3).map(|x| x.as_str().parse().unwrap_or(usize::MAX)).unwrap_or(1);
                if a < 1 || b < a || b > nchar || st < 1 { return un(); }
                let mut x = a - 1; while x < b { sites.push(x); x += st; }
            }
            if charsets.iter().any(|x| x.0 == nm) { return un(); }
            sites.sort_unstable(); sites.dedup(); charsets.push((nm, sites));
        } else {
            if body.iter().any(|t| !index.contains_key(*t)) { return un(); }
            let mut b: Vec<String> = body.iter().map(|x| x.to_string()).collect(); b.sort(); b.dedup();
            if let Some(e) = taxsets.iter_mut().find(|x| x.0 == nm) { e.1 = b; } else { taxsets.push((nm, b)); }
        }
    }
    Ok(NexRead { names, rows, info: NexusInfo { datatype: dt, missing, gap, charsets, taxsets, charpartitions: vec![], general: false } })
}

// ---- the general NEXUS reader: comments, quoted names, interleaved matrices, charsets, taxsets and charpartitions.
// It stands where locusmasker.py hands the file to Bio.Nexus, and the writer below writes such a file back the way
// Bio.Nexus does.
fn nexus_fail(path: &str, msg: &str) -> ! {
    die!("the NEXUS reader could not parse {}:\n  {}\n\
          The general NEXUS reader accepts comments, quoted names, interleaved matrices and a SETS block of charset, taxset and charpartition statements. Common causes: \
          commas inside CHARSET site lists (NEXUS wants whitespace: 'CHARSET x = 1-10 21-30;'), \
          taxon names with '-' or other punctuation that are not quoted, or unusual block formatting. \
          Check for commas in the SETS block, and/or supply the alignment as FASTA with a RAxML-style \
          partition file (--partitions).", path, msg)
}

/// the text without its [comments] (brackets inside quoted names are kept)
fn strip_comments(path: &str, s: &str) -> String {
    let mut out = String::with_capacity(s.len()); let mut depth = 0usize; let mut quote: Option<char> = None;
    for c in s.chars() {
        if depth > 0 { if c == '[' { depth += 1; } else if c == ']' { depth -= 1; } continue; }
        match quote {
            Some(q) => { out.push(c); if c == q { quote = None; } }
            None => { if c == '[' { depth = 1; } else { if c == '\'' || c == '"' { quote = Some(c); } out.push(c); } }
        }
    }
    if depth > 0 { nexus_fail(path, "a comment ('[') is not closed"); }
    out
}

/// the commands of a block: split at ';' outside quotes
fn split_commands(s: &str) -> Vec<String> {
    let mut out = vec![]; let mut cur = String::new(); let mut quote: Option<char> = None;
    for c in s.chars() {
        match quote {
            Some(q) => { cur.push(c); if c == q { quote = None; } }
            None => { if c == ';' { out.push(std::mem::take(&mut cur)); } else { if c == '\'' || c == '"' { quote = Some(c); } cur.push(c); } }
        }
    }
    if !cur.trim().is_empty() { out.push(cur); }
    out
}

/// the next NEXUS word of s from byte p: a quoted name (quotes removed, '' read as ') or a run without whitespace
fn next_word(s: &str, p: &mut usize, stop: &str) -> Option<String> {
    let b = s.as_bytes();
    while *p < b.len() && (b[*p] as char).is_whitespace() { *p += 1; }
    if *p >= b.len() { return None; }
    let q = b[*p] as char;
    if q == '\'' || q == '"' {
        let mut out = String::new(); let mut i = *p + 1; let rest: Vec<(usize, char)> = s[i..].char_indices().map(|(k, c)| (k + i, c)).collect(); let mut k = 0;
        while k < rest.len() {
            let (pos, c) = rest[k];
            if c == q {
                if k + 1 < rest.len() && rest[k + 1].1 == q { out.push(q); k += 2; continue; }
                i = pos + 1; *p = i; return Some(out);
            }
            out.push(c); k += 1;
        }
        *p = b.len(); return Some(out);
    }
    let start = *p;
    while *p < b.len() && !(b[*p] as char).is_whitespace() && !stop.contains(b[*p] as char) { *p += 1; }
    if *p == start { *p += 1; }                             // a stop character on its own is a word
    Some(s[start..*p].to_string())
}

fn parse_sites(path: &str, body: &str, nchar: usize, what: &str) -> Vec<usize> {
    // ranges 'a', 'a-b', 'a-b\s', with optional blanks around '-' and '\'; '.' is the last character
    let norm = regex::Regex::new(r"\s*-\s*").unwrap().replace_all(body, "-").into_owned();
    let norm = regex::Regex::new(r"\s*\\\s*").unwrap().replace_all(&norm, "\\").into_owned();
    let re_tok = regex::Regex::new(r"^(\d+|\.)(?:-(\d+|\.))?(?:\\(\d+))?$").unwrap();
    let mut sites = vec![];
    for tok in norm.split_whitespace() {
        let Some(r) = re_tok.captures(tok) else { nexus_fail(path, &format!("cannot read the site range {} of {}", repr_str(tok), what)) };
        let val = |x: &str| -> usize { if x == "." { nchar } else { x.parse().unwrap_or(usize::MAX) } };
        let a = val(&r[1]); let b = r.get(2).map(|x| val(x.as_str())).unwrap_or(a); let st = r.get(3).map(|x| x.as_str().parse().unwrap_or(1)).unwrap_or(1usize).max(1);
        if a < 1 || b < a || b > nchar { nexus_fail(path, &format!("the site range {} of {} is outside the alignment (1-{})", tok, what, nchar)); }
        let mut x = a - 1; while x < b { sites.push(x); x += st; }
    }
    sites.sort_unstable(); sites.dedup(); sites
}

fn read_nexus_general(path: &str, data: &[u8]) -> NexRead {
    let Ok(text) = std::str::from_utf8(data) else { nexus_fail(path, "the file is not UTF-8 text") };
    let text = strip_comments(path, &text.replace("\r\n", "\n").replace('\r', "\n"));
    let t = text.trim_start();
    if t.len() < 6 || !t[..6].eq_ignore_ascii_case("#NEXUS") { nexus_fail(path, "the file does not start with #NEXUS"); }
    let cmds = split_commands(&t[6..]);
    let mut blocks: Vec<(String, Vec<String>)> = vec![]; let mut cur: Option<(String, Vec<String>)> = None;
    for c in cmds {
        let ct = c.trim(); if ct.is_empty() { continue; }
        let lower = ct.to_lowercase(); let mut w = lower.split_whitespace();
        let first = w.next().unwrap_or("");
        if first == "begin" { if let Some(b) = cur.take() { blocks.push(b); } cur = Some((w.next().unwrap_or("").to_string(), vec![])); }
        else if first == "end" || first == "endblock" { if let Some(b) = cur.take() { blocks.push(b); } }
        else if let Some(b) = cur.as_mut() { b.1.push(c); }
        else { nexus_fail(path, &format!("text outside a block: {}", repr_str(&trunc(ct, 40)))); }
    }
    if let Some(b) = cur.take() { blocks.push(b); }
    let (mut ntax, mut nchar): (Option<usize>, Option<usize>) = (None, None);
    let mut datatype = String::new(); let (mut missing, mut gap) = ("?".to_string(), "-".to_string()); let mut interleave = false;
    let mut names: Vec<String> = vec![]; let mut index: HashMap<String, usize> = HashMap::new(); let mut rows: Vec<Vec<u8>> = vec![]; let mut have_matrix = false;
    let mut taxlabels: Vec<String> = vec![];
    for (kind, cmds) in &blocks {
        if kind == "taxa" {
            for c in cmds {
                let mut p = 0; let Some(key) = next_word(c, &mut p, "") else { continue };
                if key.eq_ignore_ascii_case("taxlabels") { while let Some(w) = next_word(c, &mut p, "") { taxlabels.push(w); } }
                else if key.eq_ignore_ascii_case("dimensions") { for cap in regex::Regex::new(r"(?i)(ntax|nchar)\s*=\s*(\d+)").unwrap().captures_iter(&c[p..]) { if cap[1].eq_ignore_ascii_case("ntax") { ntax = cap[2].parse().ok(); } } }
            }
        }
        if kind != "data" && kind != "characters" { continue; }
        if have_matrix { nexus_fail(path, "two data blocks"); }
        for c in cmds {
            let mut p = 0; let Some(key) = next_word(c, &mut p, "") else { continue };
            let key = key.to_lowercase(); let rest = &c[p..];
            match key.as_str() {
                "dimensions" => {
                    for cap in regex::Regex::new(r"(?i)(ntax|nchar)\s*=\s*(\d+)").unwrap().captures_iter(rest) {
                        if cap[1].eq_ignore_ascii_case("ntax") { ntax = cap[2].parse().ok(); } else { nchar = cap[2].parse().ok(); }
                    }
                }
                "format" => {
                    for cap in regex::Regex::new(r#"(?i)(\w+)(?:\s*=\s*("[^"]*"|'[^']*'|\S+))?"#).unwrap().captures_iter(rest) {
                        let k = cap[1].to_lowercase(); let v = cap.get(2).map(|x| x.as_str().trim_matches(|c| c == '"' || c == '\'').to_string());
                        match k.as_str() {
                            "datatype" => datatype = v.unwrap_or_default().to_lowercase(),
                            "missing" => missing = v.unwrap_or_default().chars().take(1).collect(),
                            "gap" => gap = v.unwrap_or_default().chars().take(1).collect(),
                            "interleave" => interleave = v.map_or(true, |x| !x.eq_ignore_ascii_case("no")),
                            "respectcase" | "symbols" => {}
                            other => nexus_fail(path, &format!("the format option {} is not supported", other)),
                        }
                    }
                }
                "matrix" => {
                    have_matrix = true;
                    let (Some(_), Some(nc)) = (ntax, nchar) else { nexus_fail(path, "the dimensions (ntax, nchar) must come before the matrix") };
                    let mut pending: Option<usize> = None;
                    for line in rest.split('\n') {
                        if line.trim().is_empty() { continue; }
                        if let (false, Some(k)) = (interleave, pending) {
                            rows[k].extend(line.bytes().filter(|b| !is_ws(*b)));
                            if rows[k].len() >= nc { pending = None; }
                            continue;
                        }
                        let mut q = 0; let nm = next_word(line, &mut q, "").unwrap();
                        let seq: Vec<u8> = line[q..].bytes().filter(|b| !is_ws(*b)).collect();
                        let k = match index.get(&nm) {
                            Some(&k) => { if !interleave { nexus_fail(path, &format!("the taxon {} appears twice in the matrix", nm)); } k }
                            None => { index.insert(nm.clone(), rows.len()); names.push(nm); rows.push(vec![]); rows.len() - 1 }
                        };
                        rows[k].extend(seq);
                        if !interleave && rows[k].len() < nc { pending = Some(k); }
                    }
                }
                "taxlabels" | "charlabels" | "charstatelabels" | "statelabels" | "options" | "title" | "link" => {}
                other => nexus_fail(path, &format!("the command {} of the data block is not supported", other)),
            }
        }
    }
    if !have_matrix { nexus_fail(path, "no matrix was found"); }
    let (ntax, nchar) = (ntax.unwrap_or(names.len()), nchar.unwrap());
    if names.len() != ntax { nexus_fail(path, &format!("the dimensions line says ntax={} but the matrix has {} taxa", ntax, names.len())); }
    if let Some(k) = rows.iter().position(|r| r.len() != nchar) {
        nexus_fail(path, &format!("the sequence of taxon {} has {} characters but nchar is {} (a sequence shorter or longer than nchar: the sequences are not aligned?)", names[k], rows[k].len(), nchar));
    }
    let mut charsets: Vec<(String, Vec<usize>)> = vec![]; let mut taxsets: Vec<(String, Vec<String>)> = vec![]; let mut charpartitions = vec![];
    for (kind, cmds) in &blocks {
        if kind != "sets" { continue; }
        for c in cmds {
            let mut p = 0; let Some(key) = next_word(c, &mut p, "") else { continue };
            let key = key.to_lowercase();
            if !matches!(key.as_str(), "charset" | "taxset" | "charpartition") { continue; }
            let Some(nm) = next_word(c, &mut p, "=") else { nexus_fail(path, &format!("a {} without a name", key)) };
            let rest = c[p..].trim_start(); let Some(body) = rest.strip_prefix('=') else { nexus_fail(path, &format!("the {} {} has no '='", key, nm)) };
            match key.as_str() {
                "charset" => {
                    let sites = parse_sites(path, body, nchar, &format!("charset {}", nm));
                    if let Some(e) = charsets.iter_mut().find(|x| x.0 == nm) { e.1 = sites; } else { charsets.push((nm, sites)); }
                }
                "taxset" => {
                    let mut q = 0; let mut members = vec![];
                    while let Some(w) = next_word(body, &mut q, "") { if !index.contains_key(&w) { nexus_fail(path, &format!("the taxset {} names {}, which is not in the matrix", nm, w)); } members.push(w); }
                    members.sort(); members.dedup();
                    if let Some(e) = taxsets.iter_mut().find(|x| x.0 == nm) { e.1 = members; } else { taxsets.push((nm, members)); }
                }
                _ => {
                    let mut parts: Vec<(String, Vec<usize>)> = vec![];
                    for sub in body.split(',') {
                        let Some((sn, sv)) = sub.split_once(':') else { nexus_fail(path, &format!("the charpartition {} needs name:sites entries", nm)) };
                        let mut q = 0; let sn = next_word(sn, &mut q, "").unwrap_or_default(); let sv = sv.trim();
                        let mut q2 = 0; let ref_name = next_word(sv, &mut q2, "").unwrap_or_default();
                        let sites = match charsets.iter().find(|x| x.0 == ref_name) { Some(cs) if sv[q2..].trim().is_empty() => cs.1.clone(), _ => parse_sites(path, sv, nchar, &format!("charpartition {}", nm)) };
                        parts.push((sn, sites));
                    }
                    charpartitions.push((nm, parts));
                }
            }
        }
    }
    NexRead { names, rows, info: NexusInfo { datatype, missing: Some(missing), gap: Some(gap), charsets, taxsets, charpartitions, general: true } }
}

// ------------------------------------------------------------------ read_input
fn check_rows(path: &str, names: &[String], rows: &[Vec<u8>]) {
    if rows.is_empty() { die!("no sequences in {path}"); }
    for (t, r) in names.iter().zip(rows) {
        if let Some(h) = r.iter().position(|b| *b >= 128) {
            let tail = &r[h..(h + 4).min(r.len())];
            let ch = String::from_utf8_lossy(tail).chars().next().unwrap_or('\u{fffd}');
            // a character cut by the four-byte window still decodes when the bytes are a whole character
            let what = if ch != '\u{fffd}' { format!("non-ASCII character {}", repr_str(&ch.to_string())) } else { format!("byte 0x{:02X} (not UTF-8 text)", r[h]) };
            die!("{} in the sequence of taxon {} at site {} of {}: sequences may hold only ASCII letters, '-' and '?'", what, t, commas(h as i64 + 1), basename(path));
        }
    }
    let mut order: Vec<usize> = vec![]; let mut cnt: HashMap<usize, usize> = HashMap::new();
    for r in rows { if !cnt.contains_key(&r.len()) { order.push(r.len()); } *cnt.entry(r.len()).or_insert(0) += 1; }
    let mut l = order[0]; for &x in &order { if cnt[&x] > cnt[&l] { l = x; } }
    let bad: Vec<(&String, usize)> = names.iter().zip(rows).filter(|(_, r)| r.len() != l).map(|(t, r)| (t, r.len())).collect();
    if !bad.is_empty() {
        let mut shown: Vec<String> = bad.iter().take(5).map(|(t, k)| format!("{} has {}", t, commas(*k as i64))).collect();
        if bad.len() > 5 { shown.push(format!("and {} more", bad.len() - 5)); }
        let shown = if bad.len() > 5 { let k = shown.len(); format!("{}, {}", shown[..k - 1].join(", "), shown[k - 1]) } else { shown.join(", ") };
        die!("the sequences in {} are not all the same length: {}, while {} of the {} sequences have {} sites. \
              The input must be an alignment: every sequence padded with gaps to the same length", basename(path), shown, cnt[&l], rows.len(), commas(l as i64));
    }
}

/// a file that starts with the gzip magic bytes, whatever its name
pub fn is_gzip(data: &[u8]) -> bool { data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b }

/// the alignment's file name as the outputs are named after it: without the .gz of a gzip-compressed file
pub fn plain_name(path: &str, gz: bool) -> String {
    let f = basename(path);
    if gz && f.len() > 3 && f.to_ascii_lowercase().ends_with(".gz") { f[..f.len() - 3].to_string() } else { f }
}

/// the alignment file's text and whether the file is gzip-compressed: a compressed file is decompressed as it is
/// read, so its compressed bytes are never held beside the text
fn read_bytes(path: &str) -> (Vec<u8>, bool) {
    use std::io::Read;
    let mut f = match std::fs::File::open(path) { Ok(f) => f, Err(e) => die!("cannot read the alignment {path}: {}", os_err(&e)) };
    let mut magic = [0u8; 2]; let mut got = 0;
    while got < 2 {
        match f.read(&mut magic[got..]) { Ok(0) => break, Ok(k) => got += k, Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}, Err(e) => die!("cannot read the alignment {path}: {}", os_err(&e)) }
    }
    if is_gzip(&magic[..got]) {
        let mut plain = Vec::new();
        let src = std::io::BufReader::with_capacity(1 << 20, (&magic[..]).chain(f));
        if let Err(e) = flate2::bufread::MultiGzDecoder::new(src).read_to_end(&mut plain) {
            die!("cannot read the alignment {path}: it starts as a gzip file but does not decompress ({e})");
        }
        (plain, true)
    } else {
        let size = f.metadata().map(|m| m.len() as usize).unwrap_or(0);
        let mut data = Vec::with_capacity(size.max(got) + 1); data.extend_from_slice(&magic[..got]);
        if let Err(e) = f.read_to_end(&mut data) { die!("cannot read the alignment {path}: {}", os_err(&e)); }
        (data, false)
    }
}

/// A FASTA file straight into the n x l matrix of its residues, the records in parallel: (names, rows, l). None when
/// anything is out of the ordinary (text before the first header, a header without a name or with a name that is not
/// UTF-8, sequences of different lengths or of none, blanks or non-ASCII bytes inside a sequence): read_fasta then
/// reads the file record by record, which gives the same rows for a file that is in order and says what is wrong
/// with one that is not
pub fn read_fasta_flat(txt: &[u8]) -> Option<(Vec<String>, Vec<u8>, usize)> {
    use rayon::prelude::*;
    let body = lstrip(txt);
    if body.first() != Some(&b'>') { return None; }
    let body = &body[1..];
    // the records: between a '>' at the start of a line and the next
    let mut recs: Vec<(usize, usize)> = vec![]; let mut start = 0usize;
    for c in memchr::memmem::find_iter(body, b"\n>") { recs.push((start, c)); start = c + 2; }
    recs.push((start, body.len()));
    let seq_of = |&(a, b): &(usize, usize)| -> (&[u8], &[u8]) { let rec = &body[a..b]; match memchr::memchr(b'\n', rec) { Some(p) => (&rec[..p], &rec[p + 1..]), None => (rec, &rec[rec.len()..]) } };
    let l = seq_of(&recs[0]).1.iter().filter(|c| **c != b'\n' && **c != b'\r').count();
    if l == 0 || recs.len().checked_mul(l).is_none() { return None; }
    let mut flat = vec![0u8; recs.len() * l];
    let names: Option<Vec<String>> = flat.par_chunks_mut(l).zip(recs.par_iter()).map(|(row, rec)| {
        let (header, seq) = seq_of(rec);
        let name = std::str::from_utf8(split1(header).first()?).ok()?.to_string();
        let mut at = 0usize;
        for line in seq.split(|c| *c == b'\n') {
            let line = match line.last() { Some(b'\r') => &line[..line.len() - 1], _ => line };
            if memchr::memchr(b'\r', line).is_some() {              // carriage returns inside a line: byte by byte
                for &c in line { if c != b'\r' { if at == l { return None; } row[at] = c; at += 1; } }
            } else {
                if at + line.len() > l { return None; }
                row[at..at + line.len()].copy_from_slice(line); at += line.len();
            }
        }
        if at != l || row.iter().any(|&c| c >= 128 || c == b' ' || c == b'\t' || c == 0x0b || c == 0x0c) { return None; }
        Some(name)
    }).collect();
    Some((names?, flat, l))
}

pub fn read_input(path: &str, partitions: Option<&str>) -> Input {
    let (data, gz) = read_bytes(path);
    let raw: Vec<u8> = data[..data.len().min(64)].to_vec();
    if strip(&raw).is_empty() { die!("the alignment {path} is empty"); }
    let head: String = match std::str::from_utf8(&raw) {
        Ok(s) => s.trim_start_matches(py_space).chars().take(7).collect::<String>().to_uppercase(),
        Err(_) => die!("the alignment {path} is not a text file (compressed? binary?): give a plain NEXUS, FASTA or PHYLIP file"),
    };
    let mut nexus: Option<NexusInfo> = None; let mut declared: Option<&'static str> = None;
    let names: Vec<String>; let rows: Vec<Vec<u8>>; let loci: Loci; let fmt: &'static str;
    let mut ready: Option<(Vec<u8>, usize)> = None;          // the rows already as one matrix, and checked (a FASTA file in order)
    if head.starts_with("#NEXUS") {
        let nr = match read_nexus_fast(path, &data) { Ok(r) => r, Err(NexErr::Unsupported) => read_nexus_general(path, &data) };
        drop(data);
        names = nr.names; rows = nr.rows; let info = nr.info; let dt = info.datatype.clone(); fmt = "nexus";
        if rows.is_empty() { die!("no sequences in {path}"); }
        if !info.charsets.is_empty() && partitions.is_some() { die!("the NEXUS file has charsets and a partition file was given: use one or the other, not both"); }
        if !info.charsets.is_empty() {
            let l = rows[0].len(); let mut owner: Vec<i64> = vec![-1; l]; let mut lc: Loci = info.charsets.clone();
            for v in lc.iter_mut() { v.1.sort_unstable(); }
            for j in 0..lc.len() {
                let dup: Vec<usize> = lc[j].1.iter().copied().filter(|&x| x < l && owner[x] >= 0).collect();
                if !dup.is_empty() {
                    let first = dup[0]; let mut others: Vec<String> = dup.iter().map(|&x| lc[owner[x] as usize].0.clone()).collect(); others.sort(); others.dedup();
                    let (s, are) = plural(dup.len());
                    die!("site {} is in charset {} and in charset {}: a site may be in one charset only ({} site{} of {} {} already in {})",
                         first + 1, lc[j].0, others[0], dup.len(), s, lc[j].0, are, others.join(", "));
                }
                for &x in &lc[j].1 { if x < l { owner[x] = j as i64; } }
            }
            loci = lc;
        } else if let Some(p) = partitions { loci = parse_raxml_partitions(p, rows[0].len()); }
        else { die!("the NEXUS file has no charsets: add a SETS block or give --partitions"); }
        if matches!(dt.as_str(), "dna" | "rna" | "nucleotide") { declared = Some("dna"); } else if dt == "protein" { declared = Some("aa"); }
        nexus = Some(info);
    } else {
        let Some(parts) = partitions else { die!("a FASTA or PHYLIP alignment needs --partitions") };
        if head.starts_with('>') {
            fmt = "fasta";
            if let Some((nm, flat, l)) = read_fasta_flat(&data) { drop(data); names = nm; rows = vec![]; ready = Some((flat, l)); }
            else {
                let (nm, seqs) = read_fasta(&data); drop(data);
                let empty: Vec<&String> = nm.iter().zip(&seqs).filter(|(_, s)| s.is_empty()).map(|(t, _)| t).collect();
                if !empty.is_empty() {
                    die!("FASTA: {} record{} with no sequence under the header in {}: {}{}", empty.len(), if empty.len() > 1 { "s" } else { "" }, path,
                         empty.iter().take(5).map(|s| s.as_str()).collect::<Vec<_>>().join(", "), if empty.len() > 5 { " ..." } else { "" });
                }
                names = nm; rows = seqs;
            }
        } else {
            if !head.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                die!("cannot tell the format of {path}: a NEXUS file starts with #NEXUS, a FASTA file with '>', a PHYLIP file with its counts");
            }
            let txt = match String::from_utf8(data) { Ok(t) => t.replace("\r\n", "\n").replace('\r', "\n"), Err(_) => die!("the alignment {path} is not a UTF-8 text file (compressed? binary?)") };
            let (nm, seqs) = read_phylip(&txt); fmt = "phylip";
            names = nm; rows = seqs.into_iter().map(|s| s.into_bytes()).collect();
        }
        if ready.is_none() { check_rows(path, &names, &rows); }
        loci = parse_raxml_partitions(parts, match &ready { Some((_, l)) => *l, None => rows[0].len() });
    }
    if ready.is_none() { check_rows(path, &names, &rows); }
    let mut cnt: HashMap<&String, usize> = HashMap::new(); for t in &names { *cnt.entry(t).or_insert(0) += 1; }
    let mut dup: Vec<&String> = cnt.iter().filter(|(_, c)| **c > 1).map(|(t, _)| *t).collect(); dup.sort();
    if !dup.is_empty() {
        die!("duplicate taxon names: [{}]{}", dup.iter().take(5).map(|s| repr_str(s)).collect::<Vec<_>>().join(", "), if dup.len() > 5 { " ..." } else { "" });
    }
    if loci.is_empty() { die!("no loci defined"); }
    let n = names.len();
    let (flat, l) = match ready {
        Some(x) => x,
        None => { let l = rows[0].len(); let mut flat = Vec::with_capacity(n * l); for r in rows { flat.extend_from_slice(&r); } (flat, l) }
    };
    Input { names, n, l, rows: flat, loci, fmt, declared, nexus, gz }
}

// ------------------------------------------------------------------ output
pub fn write_masked(path: &str, out_dir: &str, inp: &Input, rows: &[u8]) -> String {
    let name = plain_name(path, inp.gz); let p = std::path::Path::new(&name);
    let stem = p.file_stem().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = p.extension().map(|x| format!(".{}", x.to_string_lossy())).unwrap_or_else(|| format!(".{}", inp.fmt));
    let out = format!("{}/{}.masked{}{}", out_dir, stem, ext, if inp.gz { ".gz" } else { "" });
    let fh = std::fs::File::create(&out).unwrap_or_else(|e| die!("cannot write {}: {}", out, os_err(&e)));
    let mut fh = std::io::BufWriter::with_capacity(1 << 20, fh);
    if let Err(e) = write_masked_to(&mut fh, inp, rows, inp.gz, MASKED_CHUNK_BYTES) { die!("cannot write {}: {}", out, os_err(&e)); }
    out
}

/// the masked alignment goes out in chunks of about this much text: each a block of a gzip-compressed file
const MASKED_CHUNK_BYTES: usize = 4 << 20;

/// The masked alignment in the input's format, plain or gzip-compressed. A gzip-compressed one is one gzip member
/// (as gzip itself writes: any reader reads it), its chunks of rows deflated in parallel and joined the way pigz
/// joins them: each chunk ends on a byte boundary (a sync flush), the last one ends the stream, and the checksums of
/// the chunks are combined. The chunks depend on the alignment alone, so the bytes are the same at any thread count,
/// and the header holds no name or time, so they are the same each run.
pub fn write_masked_to<W: std::io::Write>(fh: &mut W, inp: &Input, rows: &[u8], gz: bool, chunk_bytes: usize) -> std::io::Result<()> {
    use rayon::prelude::*;
    let (n, l) = (inp.n, inp.l);
    // the text before the rows, each row's label, and the text after the rows
    let (head, labels, tail): (Vec<u8>, Vec<Vec<u8>>, Vec<u8>) = match (inp.fmt, &inp.nexus) {
        ("nexus", Some(nx)) => {
            let mut head = format!("#NEXUS\nbegin data;\ndimensions ntax={} nchar={};\n", n, l);
            head += &format!("format datatype={}{}{};\nmatrix\n", nx.datatype,
                             nx.missing.as_ref().filter(|m| !m.is_empty()).map(|m| format!(" missing={m}")).unwrap_or_default(),
                             nx.gap.as_ref().filter(|g| !g.is_empty()).map(|g| format!(" gap={g}")).unwrap_or_default());
            let safe: Vec<String> = inp.names.iter().map(|t| safename(t)).collect();
            let pad = safe.iter().map(|s| s.chars().count()).max().unwrap_or(0);
            let labels = safe.iter().map(|s| { let mut b = s.as_bytes().to_vec(); for _ in s.chars().count()..pad + 1 { b.push(b' '); } b }).collect();
            let mut tail = b";\nend;\n".to_vec();
            if !nx.charsets.is_empty() || !nx.taxsets.is_empty() || !nx.charpartitions.is_empty() {
                let mut lines = vec!["\nbegin sets".to_string()];
                for (k, v) in &nx.charsets { if !v.is_empty() { lines.push(format!("charset {} = {}", safename(k), compact4nexus(v))); } }
                for (k, v) in &nx.taxsets { if !v.is_empty() { lines.push(format!("taxset {} = {}", safename(k), v.iter().map(|t| safename(t)).collect::<Vec<_>>().join(" "))); } }
                for (k, parts) in &nx.charpartitions {
                    let mut ps: Vec<&(String, Vec<usize>)> = parts.iter().collect(); ps.sort_by(|a, b| a.1.cmp(&b.1));
                    lines.push(format!("charpartition {} = {}", safename(k), ps.iter().map(|(sn, sv)| format!("{}: {}", safename(sn), compact4nexus(sv))).collect::<Vec<_>>().join(", ")));
                }
                lines.push("end;\n".to_string());
                if lines.len() > 2 { tail.extend_from_slice(lines.join(";\n").as_bytes()); }
            }
            (head.into_bytes(), labels, tail)
        }
        ("fasta", _) => (vec![], inp.names.iter().map(|t| format!(">{}\n", t).into_bytes()).collect(), vec![]),
        _ => (format!("{} {}\n", n, l).into_bytes(), inp.names.iter().map(|t| format!("{} ", t).into_bytes()).collect(), vec![]),
    };
    // chunk c holds rows c*per .. (c+1)*per, the first one behind the head, the last one before the tail
    let per = (chunk_bytes / (l + 1 + labels.iter().map(|b| b.len()).max().unwrap_or(0)).max(1)).max(1);
    let nchunk = n.div_ceil(per).max(1);
    let text = |c: usize| -> Vec<u8> {
        let (lo, hi) = ((c * per).min(n), ((c + 1) * per).min(n));
        let mut t = Vec::with_capacity((hi - lo) * (l + 1) + labels[lo..hi].iter().map(|b| b.len()).sum::<usize>() + if c == 0 { head.len() } else { 0 } + if c + 1 == nchunk { tail.len() } else { 0 });
        if c == 0 { t.extend_from_slice(&head); }
        for i in lo..hi { t.extend_from_slice(&labels[i]); t.extend_from_slice(&rows[i * l..(i + 1) * l]); t.push(b'\n'); }
        if c + 1 == nchunk { t.extend_from_slice(&tail); }
        t
    };
    if !gz {
        for c in 0..nchunk { fh.write_all(&text(c))?; }
        return fh.flush();
    }
    fh.write_all(&[0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff])?;       // gzip, deflate, no name, no time, no extra flags, unknown system
    let mut crc = crc32fast::Hasher::new(); let mut size: u64 = 0;
    let group = 4 * rayon::current_num_threads().max(1);
    for g0 in (0..nchunk).step_by(group) {
        let blocks: Vec<std::io::Result<(Vec<u8>, crc32fast::Hasher, usize)>> = (g0..(g0 + group).min(nchunk)).into_par_iter().map(|c| {
            let t = text(c); let mut h = crc32fast::Hasher::new(); h.update(&t);
            Ok((deflate_block(&t, c + 1 == nchunk)?, h, t.len()))
        }).collect();
        for b in blocks { let (z, h, k) = b?; fh.write_all(&z)?; crc.combine(&h); size += k as u64; }
    }
    fh.write_all(&crc.finalize().to_le_bytes())?; fh.write_all(&(size as u32).to_le_bytes())?;
    fh.flush()
}

/// one chunk of a deflate stream: its bytes end the stream when it is the last, and otherwise end on a byte boundary
/// (a sync flush) so that the next chunk's bytes follow
fn deflate_block(data: &[u8], last: bool) -> std::io::Result<Vec<u8>> {
    use flate2::{Compress, Compression, FlushCompress, Status};
    let mut z = Compress::new(Compression::default(), false);
    let mut out: Vec<u8> = Vec::with_capacity(data.len() / 2 + 4096);
    let flush = if last { FlushCompress::Finish } else { FlushCompress::Sync };
    loop {
        let at = z.total_in() as usize;
        let st = z.compress_vec(&data[at..], &mut out, flush).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        if st == Status::StreamEnd { break; }
        // a sync flush is complete when the input is used up and the output had room to spare
        if !last && z.total_in() as usize == data.len() && out.len() < out.capacity() { break; }
        out.reserve((out.capacity() / 2).max(1 << 16));
    }
    Ok(out)
}
