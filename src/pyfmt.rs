//! Python-compatible text: float repr, round(), fixed decimals, thousands separators, str repr, json.dumps and csv.writer.
//! Every text output of the tool is produced through these so that it matches locusmasker.py byte for byte.

/// `repr(float)`: the shortest digits that round-trip, in Python's layout (exponent form below 1e-4 and from 1e16)
pub fn repr_f64(x: f64) -> String {
    if x.is_nan() { return "nan".into(); }
    if x.is_infinite() { return if x > 0.0 { "inf".into() } else { "-inf".into() }; }
    if x == 0.0 { return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() }; }
    let s = format!("{:e}", x.abs());                       // shortest round-trip digits: d[.ddd]e[-]x
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1;                                    // position of the decimal point relative to the digits
    let nd = digits.len() as i32;
    let mut out = String::new();
    if x < 0.0 { out.push('-'); }
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if nd > 1 { out.push('.'); out.push_str(&digits[1..]); }
        let e = decpt - 1;
        out.push('e'); out.push(if e < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", e.abs()));
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..(-decpt) { out.push('0'); }
        out.push_str(&digits);
    } else if decpt >= nd {
        out.push_str(&digits);
        for _ in 0..(decpt - nd) { out.push('0'); }
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]); out.push('.'); out.push_str(&digits[decpt as usize..]);
    }
    out
}

/// `round(x, nd)`: correctly rounded (half to even on the exact binary value), as CPython does it through dtoa
pub fn round(x: f64, nd: usize) -> f64 {
    if !x.is_finite() { return x; }
    format!("{:.*}", nd, x).parse().unwrap()
}

/// `f"{x:.{nd}f}"`
pub fn fixed(x: f64, nd: usize) -> String {
    if x.is_nan() { return "nan".into(); }
    if x.is_infinite() { return if x > 0.0 { "inf".into() } else { "-inf".into() }; }
    format!("{:.*}", nd, x)
}

/// the tool's fnum: fixed decimals, no negative zero, empty when not finite
pub fn fnum(v: f64, nd: usize) -> String {
    if !v.is_finite() { return String::new(); }
    fixed(round(v, nd) + 0.0, nd)
}

/// the tool's pfmt: a p-value from a natural-log input, six decimals
pub fn pfmt(lnp: f64) -> String {
    if !lnp.is_finite() { return String::new(); }
    fixed(lnp.exp().min(1.0), 6)
}

/// the tool's pfmt10: the same from a log10 input
pub fn pfmt10(l10: f64) -> String {
    if !l10.is_finite() { return String::new(); }
    fixed(10f64.powf(l10).min(1.0), 6)
}

/// `f"{n:,}"`
pub fn commas<T: Into<i128>>(n: T) -> String {
    let n: i128 = n.into();
    let s = n.abs().to_string(); let b = s.as_bytes(); let mut out = String::new();
    if n < 0 { out.push('-'); }
    for (i, c) in b.iter().enumerate() {
        if i > 0 && (b.len() - i) % 3 == 0 { out.push(','); }
        out.push(*c as char);
    }
    out
}

/// `f"{x:g}"` (six significant digits)
pub fn g(x: f64) -> String {
    if x == 0.0 { return if x.is_sign_negative() { "-0".into() } else { "0".into() }; }
    if !x.is_finite() { return repr_f64(x); }
    let s = format!("{:.5e}", x); let (mant, exp) = s.split_once('e').unwrap(); let exp: i32 = exp.parse().unwrap();
    let strip = |m: &str| -> String { if m.contains('.') { m.trim_end_matches('0').trim_end_matches('.').to_string() } else { m.to_string() } };
    if exp < -4 || exp >= 6 {
        format!("{}e{}{:02}", strip(mant), if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        strip(&format!("{:.*}", (5 - exp).max(0) as usize, x))
    }
}

/// `repr(str)` for the messages that quote user text
pub fn repr_str(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new(); out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"), '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"),
            c if c == q => { out.push('\\'); out.push(c); }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(q); out
}

/// `repr(bytes)`
pub fn repr_bytes(b: &[u8]) -> String {
    let q = if b.contains(&b'\'') && !b.contains(&b'"') { '"' } else { '\'' };
    let mut out = String::from("b"); out.push(q);
    for &c in b {
        match c {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"), b'\r' => out.push_str("\\r"), b'\t' => out.push_str("\\t"),
            c if c as char == q => { out.push('\\'); out.push(c as char); }
            c if c < 0x20 || c >= 0x7f => out.push_str(&format!("\\x{:02x}", c)),
            c => out.push(c as char),
        }
    }
    out.push(q); out
}

// ------------------------------------------------------------------ JSON, as json.dumps writes it
#[derive(Clone, Debug)]
pub enum J { Null, Bool(bool), Int(i64), Float(f64), Str(String), List(Vec<J>), Dict(Vec<(String, J)>) }

impl J {
    pub fn s(v: &str) -> J { J::Str(v.to_string()) }
    pub fn ints<T: Copy + Into<i64>>(v: &[T]) -> J { J::List(v.iter().map(|x| J::Int((*x).into())).collect()) }
    pub fn get(&self, k: &str) -> Option<&J> {
        if let J::Dict(d) = self { d.iter().find(|(a, _)| a == k).map(|(_, v)| v) } else { None }
    }
    pub fn set(&mut self, k: &str, v: J) {
        if let J::Dict(d) = self {
            if let Some(e) = d.iter_mut().find(|(a, _)| a == k) { e.1 = v; } else { d.push((k.to_string(), v)); }
        }
    }
    /// `json.dumps(obj, separators=(",", ":"))`
    pub fn compact(&self) -> String { let mut o = String::new(); self.write(&mut o, None, 0, ",", ":"); o }
    /// `json.dumps(obj, indent=1)`
    pub fn indent1(&self) -> String { let mut o = String::new(); self.write(&mut o, Some(1), 0, ",", ": "); o }
    /// `json.dumps(obj)`
    pub fn plain(&self) -> String { let mut o = String::new(); self.write(&mut o, None, 0, ", ", ": "); o }

    fn write(&self, o: &mut String, indent: Option<usize>, level: usize, isep: &str, ksep: &str) {
        match self {
            J::Null => o.push_str("null"),
            J::Bool(b) => o.push_str(if *b { "true" } else { "false" }),
            J::Int(i) => o.push_str(&i.to_string()),
            J::Float(f) => {
                if f.is_nan() { o.push_str("NaN") } else if f.is_infinite() { o.push_str(if *f > 0.0 { "Infinity" } else { "-Infinity" }) }
                else { o.push_str(&repr_f64(*f)) }
            }
            J::Str(s) => json_str(o, s),
            J::List(v) => {
                if v.is_empty() { o.push_str("[]"); return; }
                o.push('[');
                for (i, x) in v.iter().enumerate() {
                    if i > 0 { o.push_str(isep); }
                    if let Some(n) = indent { o.push('\n'); for _ in 0..n * (level + 1) { o.push(' '); } }
                    x.write(o, indent, level + 1, isep, ksep);
                }
                if let Some(n) = indent { o.push('\n'); for _ in 0..n * level { o.push(' '); } }
                o.push(']');
            }
            J::Dict(v) => {
                if v.is_empty() { o.push_str("{}"); return; }
                o.push('{');
                for (i, (k, x)) in v.iter().enumerate() {
                    if i > 0 { o.push_str(isep); }
                    if let Some(n) = indent { o.push('\n'); for _ in 0..n * (level + 1) { o.push(' '); } }
                    json_str(o, k); o.push_str(ksep);
                    x.write(o, indent, level + 1, isep, ksep);
                }
                if let Some(n) = indent { o.push('\n'); for _ in 0..n * level { o.push(' '); } }
                o.push('}');
            }
        }
    }
}

fn json_str(o: &mut String, s: &str) {
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"),
            '\u{8}' => o.push_str("\\b"), '\u{c}' => o.push_str("\\f"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c if (c as u32) < 0x7f => o.push(c),
            c => { let mut buf = [0u16; 2]; for u in c.encode_utf16(&mut buf) { o.push_str(&format!("\\u{:04x}", u)); } }
        }
    }
    o.push('"');
}

#[macro_export]
macro_rules! jdict {
    ($($k:expr => $v:expr),* $(,)?) => { $crate::pyfmt::J::Dict(vec![$(($k.to_string(), $v)),*]) };
}

// ------------------------------------------------------------------ a minimal JSON reader (restart headers)
pub fn json_parse(s: &str) -> Option<J> {
    let b = s.as_bytes(); let mut p = 0usize; let v = jp(b, &mut p)?; Some(v)
}
fn ws(b: &[u8], p: &mut usize) { while *p < b.len() && (b[*p] as char).is_ascii_whitespace() { *p += 1; } }
fn jp(b: &[u8], p: &mut usize) -> Option<J> {
    ws(b, p); if *p >= b.len() { return None; }
    match b[*p] {
        b'{' => {
            *p += 1; let mut d = vec![]; ws(b, p);
            if b.get(*p) == Some(&b'}') { *p += 1; return Some(J::Dict(d)); }
            loop {
                ws(b, p); let k = if let J::Str(k) = jp(b, p)? { k } else { return None }; ws(b, p);
                if b.get(*p) != Some(&b':') { return None; } *p += 1;
                let v = jp(b, p)?; d.push((k, v)); ws(b, p);
                match b.get(*p) { Some(b',') => { *p += 1; } Some(b'}') => { *p += 1; return Some(J::Dict(d)); } _ => return None }
            }
        }
        b'[' => {
            *p += 1; let mut l = vec![]; ws(b, p);
            if b.get(*p) == Some(&b']') { *p += 1; return Some(J::List(l)); }
            loop {
                l.push(jp(b, p)?); ws(b, p);
                match b.get(*p) { Some(b',') => { *p += 1; } Some(b']') => { *p += 1; return Some(J::List(l)); } _ => return None }
            }
        }
        b'"' => {
            *p += 1; let mut out = String::new();
            loop {
                let c = *b.get(*p)?; *p += 1;
                match c {
                    b'"' => return Some(J::Str(out)),
                    b'\\' => {
                        let e = *b.get(*p)?; *p += 1;
                        match e {
                            b'n' => out.push('\n'), b'r' => out.push('\r'), b't' => out.push('\t'), b'b' => out.push('\u{8}'), b'f' => out.push('\u{c}'),
                            b'u' => {
                                let h = u32::from_str_radix(std::str::from_utf8(b.get(*p..*p + 4)?).ok()?, 16).ok()?; *p += 4;
                                if (0xD800..0xDC00).contains(&h) && b.get(*p) == Some(&b'\\') && b.get(*p + 1) == Some(&b'u') {
                                    let l = u32::from_str_radix(std::str::from_utf8(b.get(*p + 2..*p + 6)?).ok()?, 16).ok()?; *p += 6;
                                    out.push(char::from_u32(0x10000 + ((h - 0xD800) << 10) + (l - 0xDC00))?);
                                } else { out.push(char::from_u32(h)?); }
                            }
                            other => out.push(other as char),
                        }
                    }
                    _ => {
                        let start = *p - 1; let mut end = *p;
                        while end < b.len() && b[end] != b'"' && b[end] != b'\\' { end += 1; }
                        out.push_str(std::str::from_utf8(&b[start..end]).ok()?); *p = end;
                    }
                }
            }
        }
        b't' => { *p += 4; Some(J::Bool(true)) }
        b'f' => { *p += 5; Some(J::Bool(false)) }
        b'n' => { *p += 4; Some(J::Null) }
        b'N' => { *p += 3; Some(J::Float(f64::NAN)) }
        _ => {
            let start = *p;
            while *p < b.len() && (b[*p] == b'-' || b[*p] == b'+' || b[*p] == b'.' || b[*p] == b'e' || b[*p] == b'E' || b[*p].is_ascii_digit()) { *p += 1; }
            let t = std::str::from_utf8(&b[start..*p]).ok()?;
            if t.contains(['.', 'e', 'E']) { t.parse().ok().map(J::Float) } else { t.parse().ok().map(J::Int) }
        }
    }
}

impl PartialEq for J {
    fn eq(&self, o: &J) -> bool {
        match (self, o) {
            (J::Null, J::Null) => true, (J::Bool(a), J::Bool(b)) => a == b, (J::Int(a), J::Int(b)) => a == b,
            (J::Float(a), J::Float(b)) => a == b, (J::Int(a), J::Float(b)) | (J::Float(b), J::Int(a)) => *a as f64 == *b,
            (J::Str(a), J::Str(b)) => a == b, (J::List(a), J::List(b)) => a == b,
            (J::Dict(a), J::Dict(b)) => a.len() == b.len() && a.iter().all(|(k, v)| o.get(k) == Some(v)),
            _ => false,
        }
    }
}

// ------------------------------------------------------------------ csv.writer (the excel dialect)
pub fn csv_row(fields: &[String]) -> String {
    let mut o = String::new();
    for (i, f) in fields.iter().enumerate() {
        if i > 0 { o.push(','); }
        if f.contains([',', '"', '\r', '\n']) {
            o.push('"'); o.push_str(&f.replace('"', "\"\"")); o.push('"');
        } else { o.push_str(f); }
    }
    if fields.len() == 1 && fields[0].is_empty() { o.push_str("\"\""); }
    o.push_str("\r\n"); o
}
