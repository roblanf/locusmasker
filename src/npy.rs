//! .npy members (format 1.0, as np.lib.format.write_array writes them), ZIP archives of them, and base64.
use std::io::{Seek, Write};

pub trait Dtype: Copy { const DESCR: &'static str; fn put(self, out: &mut Vec<u8>); }
macro_rules! dt { ($t:ty, $d:expr) => { impl Dtype for $t { const DESCR: &'static str = $d; #[inline] fn put(self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); } } }; }
dt!(u8, "|u1"); dt!(i8, "|i1"); dt!(u16, "<u2"); dt!(i16, "<i2"); dt!(u32, "<u4"); dt!(i32, "<i4"); dt!(i64, "<i8"); dt!(f32, "<f4"); dt!(f64, "<f8");
impl Dtype for bool { const DESCR: &'static str = "|b1"; #[inline] fn put(self, out: &mut Vec<u8>) { out.push(self as u8); } }

fn header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let sh = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut d = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", descr, sh);
    let unpadded = 10 + d.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    for _ in 0..pad { d.push(' '); }
    d.push('\n');
    let mut out = Vec::with_capacity(10 + d.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(d.len() as u16).to_le_bytes());
    out.extend_from_slice(d.as_bytes());
    out
}

/// the bytes of an .npy file holding `data` with the given shape (C order)
pub fn npy<T: Dtype>(shape: &[usize], data: &[T]) -> Vec<u8> {
    debug_assert_eq!(shape.iter().product::<usize>(), data.len());
    let mut out = header(T::DESCR, shape);
    out.reserve(data.len() * std::mem::size_of::<T>());
    if std::mem::size_of::<T>() == 1 {
        // SAFETY: one-byte plain types (u8, i8, bool) have the layout of their bytes
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) });
    } else {
        for &v in data { v.put(&mut out); }
    }
    out
}
pub fn npy1<T: Dtype>(data: &[T]) -> Vec<u8> { npy(&[data.len()], data) }

/// a ZIP of (name, bytes) members, deflated
pub struct Zip<W: Write + Seek> { zw: zip::ZipWriter<W> }
impl<W: Write + Seek> Zip<W> {
    pub fn new(w: W) -> Self { Zip { zw: zip::ZipWriter::new(w) } }
    pub fn put(&mut self, name: &str, data: &[u8]) -> std::io::Result<()> {
        let opt = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated).compression_level(Some(4))
            .large_file(data.len() as u64 >= 0xFFFF_FFFF);
        self.zw.start_file(name, opt).map_err(std::io::Error::other)?;
        self.zw.write_all(data)
    }
    pub fn finish(self) -> std::io::Result<W> { self.zw.finish().map_err(std::io::Error::other) }
}

/// a member of an archive whose bytes are made when asked for
pub type Member<'a> = (String, Box<dyn Fn() -> Vec<u8> + Sync + Send + 'a>);

/// A ZIP of members deflated in parallel: each is made and compressed on its own (into a one-member archive in
/// memory), and the compressed members are then copied into the file in order. At most a thread's worth of
/// uncompressed members exist at a time
pub fn zip_parallel<W: Write + Seek>(w: W, members: &[Member]) -> std::io::Result<W> {
    use rayon::prelude::*;
    let other = |e: zip::result::ZipError| std::io::Error::other(e);
    let parts: Vec<std::io::Result<Vec<u8>>> = members.par_iter().map(|(name, make)| {
        let data = make(); let mut z = Zip::new(std::io::Cursor::new(Vec::new()));
        z.put(name, &data)?; Ok(z.finish()?.into_inner())
    }).collect();
    let mut zw = zip::ZipWriter::new(w);
    for part in parts { zw.merge_archive(zip::ZipArchive::new(std::io::Cursor::new(part?)).map_err(other)?).map_err(other)?; }
    zw.finish().map_err(other)
}

pub fn zip_bytes(members: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut z = Zip::new(std::io::Cursor::new(Vec::new()));
    for (n, d) in members { z.put(n, d).expect("zip in memory"); }
    z.finish().expect("zip in memory").into_inner()
}

pub fn b64(data: &[u8]) -> String {
    use base64::Engine; base64::engine::general_purpose::STANDARD.encode(data)
}

/// a ZIP of members as base64 text (what a chunk file hands the page)
pub fn zip_b64(members: &[(String, Vec<u8>)]) -> String { b64(&zip_bytes(members)) }

/// read a little-endian .npy file written by this module: (descr, shape, raw data)
pub fn read_npy(bytes: &[u8]) -> Option<(String, Vec<usize>, &[u8])> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" { return None; }
    let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let h = std::str::from_utf8(bytes.get(10..10 + hl)?).ok()?;
    let d0 = h.find("'descr': '")? + 10; let d1 = d0 + h[d0..].find('\'')?;
    let s0 = h.find("'shape': (")? + 10; let s1 = s0 + h[s0..].find(')')?;
    let shape = h[s0..s1].split(',').filter_map(|t| t.trim().parse().ok()).collect();
    Some((h[d0..d1].to_string(), shape, &bytes[10 + hl..]))
}
