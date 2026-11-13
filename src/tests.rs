//! Unit tests: the Python- and numpy-compatible pieces against values taken from Python 3.15 / numpy 2.5
//! (the whole tool is held to locusmasker.py by python-original/validation/compare_ports.py and fuzz_ports.py).
use crate::npmath::*;
use crate::pyfmt::*;

#[test]
fn float_repr_is_pythons() {
    let cases = [(0.1, "0.1"), (1e16, "1e+16"), (1e15, "1000000000000000.0"), (1e-5, "1e-05"), (0.0001, "0.0001"), (123456789.123, "123456789.123"), (-0.0, "-0.0"),
                 (5e-324, "5e-324"), (1.7976931348623157e308, "1.7976931348623157e+308"), (0.30000000000000004, "0.30000000000000004"), (100.0, "100.0"), (2.5e-7, "2.5e-07")];
    for (x, want) in cases { assert_eq!(repr_f64(x), want); }
}

#[test]
fn round_and_formats_are_pythons() {
    assert_eq!(round(2.675, 2), 2.67); assert_eq!(round(1.0000005, 6), 1.000001); assert_eq!(round(123.456789, 4), 123.4568);
    assert!(round(-0.00049, 3) == 0.0 && round(-0.00049, 3).is_sign_negative());
    assert_eq!(fnum(-0.00049, 3), "0.000"); assert_eq!(fnum(f64::NAN, 2), ""); assert_eq!(pfmt(f64::NEG_INFINITY), ""); assert_eq!(pfmt(0.5), "1.000000");
    for (x, want) in [(0.05, "0.05"), (1e-6, "1e-06"), (123456789.0, "1.23457e+08"), (0.0001, "0.0001"), (1.5, "1.5"), (100000.0, "100000"), (1e6, "1e+06")] { assert_eq!(g(x), want); }
    assert_eq!(commas(1234567i64), "1,234,567"); assert_eq!(commas(-1000i64), "-1,000"); assert_eq!(commas(999i64), "999");
    assert_eq!(repr_str("it's"), "\"it's\""); assert_eq!(repr_str("a\tb"), "'a\\tb'");
}

#[test]
fn json_and_csv_are_pythons() {
    let j = crate::jdict! { "a" => J::List(vec![J::Int(1), J::Float(2.5), J::Null, J::Bool(true), J::s("x\"y\u{e9}\n")]), "b" => J::Dict(vec![]), "c" => J::List(vec![]), "d" => J::Float(f64::NAN) };
    assert_eq!(j.indent1().replace('\n', "|"), "{| \"a\": [|  1,|  2.5,|  null,|  true,|  \"x\\\"y\\u00e9\\n\"| ],| \"b\": {},| \"c\": [],| \"d\": NaN|}");
    assert_eq!(j.compact(), "{\"a\":[1,2.5,null,true,\"x\\\"y\\u00e9\\n\"],\"b\":{},\"c\":[],\"d\":NaN}");
    assert!(json_parse(&j.compact()).is_some());
    let row: Vec<String> = ["1", "a,b", "q\"r", "x y", "", " lead", "l\nf"].iter().map(|s| s.to_string()).collect();
    assert_eq!(csv_row(&row), "1,\"a,b\",\"q\"\"r\",x y,, lead,\"l\nf\"\r\n");
}

#[test]
fn npy_header_is_numpys() {
    let b = crate::npy::npy(&[3, 4], &[0f32; 12]);
    assert_eq!(b.len(), 176); assert_eq!(&b[..10], b"\x93NUMPY\x01\x00v\x00"); assert!(b[10..].starts_with(b"{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), } "));
    assert!(crate::npy::npy1(&[1i64, 2]).windows(13).any(|w| w == b"'shape': (2,)")); assert_eq!(crate::npy::read_npy(&b).unwrap().1, vec![3, 4]);
}

#[test]
fn choice_is_numpys() {
    let first6 = |pop, size| crate::rng::choice_sorted(0, pop, size)[..6].to_vec();
    assert_eq!(first6(50, 7), [2, 3, 12, 14, 23, 28]); assert_eq!(first6(20000, 500), [54, 106, 125, 158, 165, 196]);          // Floyd's algorithm; the tail shuffle
    assert_eq!(first6(20000, 100), [54, 106, 165, 329, 440, 564]); assert_eq!(first6(2_000_000, 30), [5476, 33054, 67171, 81946, 150478, 350530]);
    assert_eq!(crate::rng::choice_sorted(0, 9, 9), (0..9).collect::<Vec<_>>());
}

#[test]
fn hashes_and_draws_are_the_tools() {
    assert_eq!([sm64(0), sm64(1), sm64(12345)], [16294208416658607535, 10451216379200822465, 2454886589211414944]);
    use crate::prepare::{bs_keys_cell, bs_keys_taxa};
    assert_eq!(hash_draw(bs_keys_taxa(0), 30, 4), [28, 11, 29, 24]); assert_eq!(hash_draw(bs_keys_taxa(1), 30, 4), [17, 27, 28, 22]);
    assert_eq!(hash_draw(bs_keys_cell(3, 2), 500, 50)[..5], [483, 342, 434, 237, 350]); assert_eq!(hash_draw(bs_keys_cell(9, 2), 500, 50)[..5], [216, 334, 189, 431, 340]);
}

#[test]
fn special_functions_match_to_the_bit() {
    let lg = [(0.3, 1.0957979948180763), (1.0, 0.0), (7.5, 7.534364236758733), (123.25, 468.6144829505166), (4000.0, 29172.970494954596)];
    for (x, want) in lg { assert_eq!(lgamma_v(x).to_bits(), f64::to_bits(want), "lgamma_v({x})"); }
    let lf = log_fact(1002);
    let (o, n, e, phi) = ([30., 2., 0., 55., 10., 3.], [200., 300., 50., 60., 1000., 40.], [10., 12., 3., 20., 25.5, 0.0], [0.0, 0.02, 0.5, 0.001, 1e-12, 0.1]);
    let up = [-16.256292659923375, -0.00023410280570779202, 0.0, -31.97441295682957, -0.00013788904382700174, -690.7755278982137];
    let dn = [-2.478249783166575e-08, -6.6671309609495415, -1.7868390023785707, 0.0, -7.881158662602448, 0.0];
    for i in 0..6 {
        assert_eq!(bb_tail(o[i], n[i], e[i], phi[i], true, &lf).to_bits(), f64::to_bits(up[i]), "upper tail {i}");
        assert_eq!(bb_tail(o[i], n[i], e[i], phi[i], false, &lf).to_bits(), f64::to_bits(dn[i]), "lower tail {i}");
    }
    assert_eq!(norm_quantile(1.0 - 0.05 / 1234.0), 3.9413115842417996);
}

#[test]
fn numpy_reductions() {
    let v = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3];
    assert_eq!(percentile(&v, 10.0), 1.3); assert_eq!(percentile(&v, 90.0), 6.780000000000001); assert_eq!(interp(2.2, &[0.0, 1.0, 3.0], &[0.0, 10.0, 11.0]), 10.6);
    assert_eq!(median(&v), 3.0); assert_eq!(median(&v[..6]), 2.8); assert!(median(&[1.0, f64::NAN]).is_nan()); assert_eq!(nanmedian(&[1.0, f64::NAN, 3.0]), 2.0);
    let a: Vec<f64> = (0..300).map(|i| 0.1 * i as f64 + 1e-9 * (i * i) as f64).collect();
    assert_ne!(pairwise_sum(&a).to_bits(), a.iter().sum::<f64>().to_bits());          // numpy's pairwise order is not the running sum
    let mut vals = vec![5.0, 1.0, 3.0, 2.0, 4.0]; let mut w = vec![1.0, 1.0, 1.0, 1.0, 10.0];
    assert_eq!(wmed_select(&mut vals, &mut w, 5), 4.0); assert_eq!(wmedian1(&[5.0, 1.0, 3.0], &[1.0, 1.0, 1.0]), 3.0);
    assert!(npmax(f64::NAN, 1.0).is_nan() && pymax(f64::NAN, 1.0).is_nan() && pymax(1.0, f64::NAN) == 1.0);
}

#[test]
fn nexus_helpers_write_as_biopython() {
    use crate::input::{compact4nexus, safename};
    assert_eq!([compact4nexus(&[0, 1, 2, 3]), compact4nexus(&[0, 2, 4, 6, 9]), compact4nexus(&[5]), compact4nexus(&[0, 1, 2, 10, 11, 20]), compact4nexus(&[])], ["1-4", "1-7\\2 10", "6", "1-3 11-12 21", ""]);
    assert_eq!([safename("abc"), safename("a b"), safename("it's"), safename("x-y")], ["abc", "'a b'", "'it''s'", "'x-y'"]);
    let r = crate::report::ranges;
    assert_eq!(r(&[0, 1, 2, 3]), [[0, 3, 1]]); assert_eq!(r(&[0, 2, 4, 6, 9]), [[0, 6, 2], [9, 9, 1]]); assert_eq!(r(&[5]), [[5, 5, 1]]);
    assert_eq!(r(&[0, 3, 4, 5]), [[0, 0, 1], [3, 5, 1]]); assert_eq!(r(&[1, 4]), [[1, 1, 1], [4, 4, 1]]);
}

#[test]
fn readers() {
    let (names, seqs) = crate::input::read_fasta(b">a desc\nAC\nGT\n>b\r\nAC GT\r\n");
    assert_eq!(names, ["a", "b"]); assert_eq!(seqs, [b"ACGT".to_vec(), b"ACGT".to_vec()]);
    let (names, seqs) = crate::input::read_phylip("2 4\na ACGT\nb AC\nGT\n"); assert_eq!(names, ["a", "b"]); assert_eq!(seqs, ["ACGT", "ACGT"]);
    let (names, seqs) = crate::input::read_phylip("2 8\na ACGT\nb ACGA\n\nTTTT\nCCCC\n"); assert_eq!(names, ["a", "b"]); assert_eq!(seqs, ["ACGTTTTT", "ACGACCCC"]);
}

#[test]
fn gzip_alignments() {
    use crate::input::{is_gzip, plain_name};
    assert!(is_gzip(&[0x1f, 0x8b, 8, 0]) && !is_gzip(b">a\nACGT\n") && !is_gzip(&[0x1f]) && !is_gzip(b""));
    // the outputs are named after the file without its .gz, when the file is gzip-compressed
    assert_eq!(plain_name("/d/x.fasta.gz", true), "x.fasta"); assert_eq!(plain_name("x.FAA.GZ", true), "x.FAA"); assert_eq!(plain_name("x.fasta", true), "x.fasta");
    assert_eq!(plain_name("x.fasta.gz", false), "x.fasta.gz"); assert_eq!(plain_name(".gz", true), ".gz");
    // an alignment read from a gzip file is the alignment, and its masked copy is written gzip-compressed and reads back the same
    let dir = std::env::temp_dir().join(format!("locusmasker_gz_test_{}", std::process::id())); std::fs::create_dir_all(&dir).unwrap();
    let text = ">a\nACGTACGTAC\n>b\nACGTAC--AC\n"; let parts = dir.join("p.txt"); std::fs::write(&parts, "DNA, l1 = 1-5\nDNA, l2 = 6-10\n").unwrap();
    let plain = dir.join("x.fasta"); std::fs::write(&plain, text).unwrap();
    let gz = dir.join("x.fasta.gz");
    { use std::io::Write; let mut e = flate2::write::GzEncoder::new(std::fs::File::create(&gz).unwrap(), flate2::Compression::default()); e.write_all(text.as_bytes()).unwrap(); e.finish().unwrap(); }
    let a = crate::input::read_input(plain.to_str().unwrap(), parts.to_str()); let b = crate::input::read_input(gz.to_str().unwrap(), parts.to_str());
    assert!(!a.gz && b.gz && a.names == b.names && a.rows == b.rows && a.fmt == "fasta" && b.fmt == "fasta" && a.loci == b.loci);
    let out = dir.to_str().unwrap();
    let pa = crate::input::write_masked(plain.to_str().unwrap(), out, &a, &a.rows); let pb = crate::input::write_masked(gz.to_str().unwrap(), out, &b, &b.rows);
    assert!(pa.ends_with("/x.masked.fasta") && pb.ends_with("/x.masked.fasta.gz"));
    let back = { use std::io::Read; let mut s = Vec::new(); flate2::read::MultiGzDecoder::new(std::fs::File::open(&pb).unwrap()).read_to_end(&mut s).unwrap(); s };
    assert_eq!(back, std::fs::read(&pa).unwrap()); assert_eq!(back, text.as_bytes());
    // gzip output in many chunks (13 bytes of text each) is still one gzip member that holds the plain text
    for chunk in [1usize, 13, 20, 1 << 20] {
        let mut z = Vec::new(); crate::input::write_masked_to(&mut z, &b, &b.rows, true, chunk).unwrap();
        let mut plain = Vec::new(); crate::input::write_masked_to(&mut plain, &b, &b.rows, false, chunk).unwrap();
        let mut one = Vec::new(); { use std::io::Read; let mut d = flate2::read::GzDecoder::new(&z[..]); d.read_to_end(&mut one).unwrap(); }
        assert_eq!(one, text.as_bytes()); assert_eq!(plain, text.as_bytes());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn elapsed_times() {
    let e = crate::elapsed_string;
    assert_eq!(e(0.04), "0.0 s"); assert_eq!(e(34.21), "34.2 s"); assert_eq!(e(59.96), "1 min 0 s (60 s)"); assert_eq!(e(156.3), "2 min 36 s (156 s)");
    assert_eq!(e(3600.0), "1 h 0 min 0 s (3,600 s)"); assert_eq!(e(11525.4), "3 h 12 min 5 s (11,525 s)");
}

/// a FASTA file in order read straight into the matrix is the file read record by record; anything out of the
/// ordinary is left to the record-by-record reader
#[test]
fn fasta_straight_into_the_matrix() {
    use crate::input::{read_fasta, read_fasta_flat};
    for txt in [&b">a desc\nACGT\nACGT\n>b\nAC--ACGT\n>c x y\nACGTACGT"[..], b"\n\n>a\r\nACGT\r\nAC\r\n>b\r\nACGTAC\r\n", b">a\nAC\rGT\n>b\nACGT\n", b">a\nA\n\n\nC\n>b\n\nAC\n\n"] {
        let (names, seqs) = read_fasta(txt); let (n2, flat, l) = read_fasta_flat(txt).expect("a file in order");
        assert_eq!(names, n2); assert!(seqs.iter().all(|s| s.len() == l)); assert_eq!(seqs.concat(), flat);
    }
    // text before the first header, no name, different lengths, no sequence, blanks or non-ASCII bytes in a sequence, a name that is not UTF-8
    for txt in [&b"x\n>a\nACGT\n"[..], b">\nACGT\n>b\nACGT\n", b">a\nACGT\n>b\nACG\n", b">a\nACGT\n>b\nACGTA\n", b">a\n>b\nACGT\n", b">a\nAC GT\n>b\nAC GT\n", b">a\nAC\tGT\n>b\nACGTA\n",
                b">a\nAC\xc3\xa9T\n>b\nACGT\n", b">\xff\nACGT\n>b\nACGT\n", b">a\nACGT\n>", b""] {
        assert!(read_fasta_flat(txt).is_none(), "{:?}", String::from_utf8_lossy(txt));
    }
}

/// the medians and percentiles by selection are those by sorting
#[test]
fn selection_statistics() {
    let mut x = 3u64;
    for n in [1usize, 2, 3, 4, 5, 10, 11, 100, 1001] {
        for _ in 0..20 {
            let v: Vec<f64> = (0..n).map(|_| { x = sm64(x); ((x % 50) as f64 / 7.0).ln_1p() + (x % 3) as f64 }).collect();
            assert_eq!(median_of(v.clone()).to_bits(), median(&v).to_bits());
            for q in [0.0, 10.0, 50.0, 90.0, 99.9, 100.0] { let mut w = v.clone(); assert_eq!(percentile_of(&mut w, q).to_bits(), percentile(&v, q).to_bits(), "n {n}, q {q}"); }
        }
    }
}

/// the draws from a small pool (seen values as bits) and from a large one (as a set) follow the same rule
#[test]
fn hash_draws_small_and_large() {
    for (key, m, need) in [(1u64, 500usize, 50usize), (99, 4096, 300), (7, 130, 60), (12345, 64, 10)] {
        let got = hash_draw(key, m, need);
        // the rule, spelled out: slots in order, a value kept the first time it appears, the window doubled when it runs out
        let want = {
            let mut s = (need as f64 + 4.0 * (need as f64).sqrt() + (need * need) as f64 / (m as i64 - need as i64).max(1) as f64 + 16.0) as usize; let mut start = 0usize;
            loop {
                let mut out: Vec<i64> = vec![];
                for slot in start..start + s { let v = (sm64(key ^ slot as u64) % m as u64) as i64; if !out.contains(&v) { out.push(v); if out.len() == need { break; } } }
                if out.len() == need { break out; }
                start += s; s *= 2;
            }
        };
        if m >= 2 * need { assert_eq!(got, want, "m {m}, need {need}"); }
        let mut d = got.clone(); d.sort_unstable(); d.dedup(); assert_eq!(d.len(), need); assert!(got.iter().all(|v| (*v as usize) < m));
    }
    assert_eq!(hash_draw(5, 5000, 40).len(), 40);
}
