//! report.npz, report.html (the viewer with the core data inline) and report_files/ (one chunk per block of loci and
//! per block of taxa). The viewer's HTML, CSS, JavaScript, fonts and icons are embedded in the binary unchanged.
use crate::input::Loci;
use crate::npy::{b64, npy, npy1, zip_b64};
use crate::pyfmt::{round, J};
use crate::rep::*;
use crate::{die, jdict, log, Args};
use std::collections::HashMap;

pub const REPORT_FORMAT: i64 = 9;
const REPORT_TAXA_BLOCK: usize = 64;
const REPORT_CHUNK_BYTES: usize = 8 << 20;
const REPORT_SAMPLE_CELLS: usize = 30_000;

static VIEWER: &[(&str, &[u8])] = &[
    ("viewer.html", include_bytes!("../viewer/viewer.html")), ("viewer.js", include_bytes!("../viewer/viewer.js")),
    ("strings.js", include_bytes!("../viewer/strings.js")), ("viewer.css", include_bytes!("../viewer/viewer.css")),
    ("fonts.css", include_bytes!("../viewer/fonts.css")),
    ("vendor/d3.v7.min.js", include_bytes!("../viewer/vendor/d3.v7.min.js")), ("vendor/d3-sankey.min.js", include_bytes!("../viewer/vendor/d3-sankey.min.js")),
    ("vendor/fuse.min.js", include_bytes!("../viewer/vendor/fuse.min.js")),
    ("icons/favicon-16.png", include_bytes!("../viewer/icons/favicon-16.png")), ("icons/favicon-32.png", include_bytes!("../viewer/icons/favicon-32.png")),
    ("icons/favicon-16.svg", include_bytes!("../viewer/icons/favicon-16.svg")), ("icons/logo.svg", include_bytes!("../viewer/icons/logo.svg")),
    ("vendor/D3_LICENSE", include_bytes!("../viewer/vendor/D3_LICENSE")), ("vendor/D3_SANKEY_LICENSE", include_bytes!("../viewer/vendor/D3_SANKEY_LICENSE")),
    ("vendor/FUSE_LICENSE", include_bytes!("../viewer/vendor/FUSE_LICENSE")), ("vendor/INTER_LICENSE", include_bytes!("../viewer/vendor/INTER_LICENSE")),
    ("LICENSE", include_bytes!("../LICENSE")),
];
fn asset(name: &str) -> &'static [u8] { VIEWER.iter().find(|x| x.0 == name).unwrap_or_else(|| die!("the viewer file {name} is not embedded in this build")).1 }
fn text(name: &str) -> String { String::from_utf8_lossy(asset(name)).replace("\r\n", "\n").replace('\r', "\n") }

fn licences_text() -> String {
    let parts = [("LocusMasker (this page and the tool that wrote it): MIT License", "LICENSE"), ("D3 (d3.v7.min.js): ISC License", "vendor/D3_LICENSE"),
                 ("d3-sankey (d3-sankey.min.js): BSD 3-Clause License", "vendor/D3_SANKEY_LICENSE"), ("Fuse.js (fuse.min.js): Apache License 2.0", "vendor/FUSE_LICENSE"),
                 ("Inter font (InterVariable.woff2, embedded in the stylesheet): SIL Open Font License 1.1", "vendor/INTER_LICENSE")];
    let mut out = vec!["This report embeds the following software and font. Their licence texts follow; the same files are in the tool's viewer/vendor folder.".to_string()];
    let bar = "=".repeat(78);
    for (head, path) in parts {
        let body = text(path); let body = body.trim_matches(|c: char| c.is_whitespace()).replace("</script", "<\\/script");
        out.push(format!("{bar}\n{head}\n{bar}\n{body}"));
    }
    out.join("\n\n")
}

/// a locus's site indices as [start, end, step] runs (0-based, inclusive)
pub fn ranges(sites: &[usize]) -> Vec<[i64; 3]> {
    let s: Vec<i64> = sites.iter().map(|x| *x as i64).collect(); let n = s.len(); let mut runs = vec![]; let mut i = 0;
    while i < n {
        let mut j = i + 1; let step = if j < n { s[j] - s[i] } else { 1 };
        while j + 1 < n && s[j + 1] - s[j] == step { j += 1; }
        if j == i + 1 && step != 1 && j < n { runs.push([s[i], s[i], 1]); i += 1; continue; }
        if j >= n { j = n - 1; }
        runs.push([s[i], s[j], if j > i { step } else { 1 }]); i = j + 1;
    }
    runs
}

pub fn sha256_of(path: &str) -> std::io::Result<String> {
    use sha2::Digest; use std::io::Read;
    let mut h = sha2::Sha256::new(); let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    loop { let k = f.read(&mut buf)?; if k == 0 { break; } h.update(&buf[..k]); }
    Ok(h.finalize().iter().map(|b| format!("{:02x}", b)).collect())
}
pub fn sha256_file(path: &str) -> String { sha256_of(path).unwrap_or_else(|e| die!("cannot read {path}: {}", crate::input::os_err(&e))) }

fn opt_round(v: f64, nd: usize) -> J { if v.is_finite() { J::Float(round(v, nd)) } else { J::Null } }

#[allow(clippy::too_many_arguments)]
pub fn write_report(a: &Args, names: &[String], loci: &Loci, rep: &Rep, dtype: &str, fmt: &str, gz: bool, k_in: usize, k_eff: usize, m_eff: usize, t0: std::time::Instant, argv: &[String]) -> bool {
    let (n, nloc) = (names.len(), loci.len()); let dense = rep.dense; let k = k_eff;
    let npz = format!("{}/report.npz", a.out);
    let io = |r: std::io::Result<()>, what: &str| { if let Err(e) = r { die!("cannot write {}: {}", what, crate::input::os_err(&e)); } };
    let cell_shape: Vec<usize> = if dense { vec![n, nloc] } else { vec![rep.sparse_locus.len()] };
    let mut with_k = cell_shape.clone(); with_k.push(k);
    let trees_json: Vec<(usize, String)> = rep.trees.0.iter().map(|(li, t)| (*li, t.json().compact())).collect();
    let similar_json: Vec<(usize, String)> = rep.similar.0.iter().map(|(li, v)| (*li, J::List(v.clone()).compact())).collect();
    let broad_json: Vec<(usize, String)> = rep.bs_records.0.iter().map(|(li, v)| (*li, J::List(v.clone()).compact())).collect();
    // ---- the tables and the manifest
    let tested = |c: usize| rep.n_comp[c] > 0;
    let cnt = |f: &dyn Fn(usize) -> bool| -> i64 { (0..n * nloc).filter(|&c| f(c)).count() as i64 };
    let side = &rep.side;
    let near_miss = J::Dict((1..m_eff).map(|j| (j.to_string(), J::Int(cnt(&|c| rep.n_fail[c] as usize == j && tested(c) && side[c] & 1 == 0)))).collect());
    let similar_pairs: i64 = rep.similar.0.iter().map(|(_, v)| v.iter().filter(|e| matches!(e.get("fail"), Some(J::Bool(true)))).count() as i64).sum();
    let untested = jdict! { "sequences" => J::Int(rep.untested[0]), "of_excluded_taxa" => J::Int(rep.untested[1]), "at_untested_loci" => J::Int(rep.untested[2]), "without_a_comparator" => J::Int(rep.untested[3]) };
    let counts = jdict! {
        "taxa" => J::Int(n as i64), "loci" => J::Int(nloc as i64), "cells" => J::Int(cnt(&tested)), "present" => J::Int(rep.focal_sites.iter().filter(|x| **x > 0).count() as i64),
        "masked" => J::Int(cnt(&|c| side[c] & 3 > 0)), "masked_divergent" => J::Int(cnt(&|c| side[c] & 1 > 0)), "masked_similar" => J::Int(cnt(&|c| side[c] & 2 > 0)),
        "masked_both" => J::Int(cnt(&|c| side[c] & 3 == 3)), "ambiguous" => J::Int(cnt(&|c| side[c] & 4 > 0)), "similar_tested" => J::Int(cnt(&|c| side[c] & 64 > 0)),
        "similar_pairs" => J::Int(similar_pairs), "similar_sources" => J::Int(cnt(&|c| side[c] & 128 > 0)), "similar_uncertain" => J::Int(cnt(&|c| side[c] & 8 > 0)),
        "near_miss" => near_miss, "excluded_taxa" => J::Int(rep.thin.len() as i64), "skipped_loci" => J::Int(rep.loc_status.iter().filter(|s| *s != "tested").count() as i64),
        "untested_sequences" => J::Int(rep.untested[0]), "untested" => untested, "uncovered_sites" => J::Int(rep.uncovered_sites as i64) };
    let thin: std::collections::HashSet<usize> = rep.thin.iter().copied().collect();
    let taxa_json = jdict! {
        "names" => J::List(names.iter().map(|s| J::s(s)).collect()),
        "status" => J::List((0..n).map(|i| if thin.contains(&i) { J::Str(format!("excluded: {} loci with at least {} residues, {} needed", rep.present[i], a.fit_overlap, crate::prepare::MIN_LOCI_PER_TAXON)) } else { J::s("tested") }).collect()),
        "loci_present" => J::List((0..n).map(|g| J::Int(rep.focal_sites[g * nloc..(g + 1) * nloc].iter().filter(|x| **x > 0).count() as i64)).collect()) };
    let loci_json = jdict! {
        "names" => J::List(loci.iter().map(|x| J::s(&x.0)).collect()), "length" => J::List(loci.iter().map(|x| J::Int(x.1.len() as i64)).collect()),
        "ranges" => J::List(loci.iter().map(|x| J::List(ranges(&x.1).iter().map(|r| J::ints(&r[..])).collect())).collect()),
        "status" => J::List(rep.loc_status.iter().map(|s| J::s(s)).collect()), "rate" => J::List(rep.rate.iter().map(|v| opt_round(*v, 5)).collect()),
        "plateau" => J::List(rep.plateaus.iter().map(|v| J::Float(round(*v, 3))).collect()), "phi" => J::List(rep.phi.iter().map(|v| if v.is_finite() { J::Float(*v) } else { J::Null }).collect()),
        "test_rate" => J::List(rep.test_rate.iter().map(|v| opt_round(*v, 5)).collect()), "similar_tests" => J::List(rep.pair_tests.iter().map(|v| J::Int(*v)).collect()) };
    let aln_path = std::path::Path::new(&a.aln);
    let base = aln_path.file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
    // the dataset is named after the alignment file, a gzip-compressed one without its .gz
    let stem = std::path::Path::new(&crate::input::plain_name(&a.aln, gz)).file_stem().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
    let parent = std::fs::canonicalize(aln_path).ok().or_else(|| std::env::current_dir().ok().map(|d| d.join(aln_path)))
        .and_then(|p| p.parent().and_then(|d| d.file_name().map(|x| x.to_string_lossy().into_owned()))).unwrap_or_default();
    let dataset = match &a.name { Some(nm) if !nm.is_empty() => nm.clone(), _ => if stem.to_lowercase() == "alignment" && !matches!(parent.as_str(), "" | "." | "work") { parent } else { stem } };
    let now = crate::now_string();
    let mut timings: Vec<(String, J)> = crate::log::prog().durations().into_iter().map(|(s, d)| (s, J::Float(d))).collect();
    timings.retain(|x| x.0 != "total"); timings.push(("total".into(), J::Float(round(t0.elapsed().as_secs_f64(), 2))));
    let mut manifest = jdict! {
        "format_version" => J::Int(REPORT_FORMAT), "tool" => J::s("LocusMasker"), "version" => J::s(crate::VERSION), "commit" => J::s(env!("LM_GIT_COMMIT")),
        "python" => J::Str(format!("n/a (Rust port, {})", env!("LM_RUSTC"))), "numpy" => J::s("n/a"),
        "system" => J::Str(crate::system_string()), "date" => J::Str(now), "command" => J::Str(argv.join(" ")),
        "options" => jdict! { "datatype" => J::s(dtype), "n_comp" => J::Int(k_in as i64), "K_eff" => J::Int(k_eff as i64), "M_eff" => J::Int(m_eff as i64), "fit_overlap" => J::Int(a.fit_overlap),
                              "alpha" => J::Float(a.alpha), "donor_threshold" => J::Float(a.donor_threshold), "mask_similar_only" => J::Bool(a.mask_similar_only), "mask_different_only" => J::Bool(a.mask_different_only),
                              "threads" => J::Int(a.threads as i64), "all_tests" => J::Bool(a.all_tests), "report_codes" => J::s(&a.report_codes), "report_sequences" => J::s(&rep.seq_policy) },
        "input" => jdict! { "file" => J::Str(base), "bytes" => J::Int(std::fs::metadata(&a.aln).map(|m| m.len() as i64).unwrap_or(0)), "sha256" => J::Str(crate::input_sha256(a)), "format" => J::s(fmt),
                            "partitions" => match &a.partitions { Some(p) => J::Str(std::path::Path::new(p).file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default()), None => J::Null } },
        "dataset" => J::Str(dataset.clone()), "counts" => counts,
        "test" => jdict! { "tests" => J::Int(rep.tests[4]), "close_relative_tests" => J::Int(rep.tests[0]), "broad_sample_tests" => J::Int(rep.tests[3]),
                           "closest_match_relative_tests" => J::Int(rep.tests[1]), "closest_match_outside_tests" => J::Int(rep.tests[2]), "similar_candidates" => J::Int(rep.tests[5]),
                           "comparisons" => J::Int(rep.tests[0]), "bonferroni_k" => J::Float(round(rep.kq, 4)), "log_bar" => J::Float(round(a.alpha.ln(), 4)), "log10_bar" => J::Float(round(a.alpha.log10(), 4)),
                           "p_values" => J::s("adjusted by Bonferroni correction: p x N, capped at 1"), "phi_dataset" => J::Float(round(rep.phi_bar, 6)), "half" => J::Float(0.5) },
        "carried" => jdict! { "numeric_columns" => J::s(if dense { "dense" } else { "sparse" }), "codes" => J::s(&a.report_codes), "sequences" => J::s(&rep.seq_policy), "trees" => J::s("masked"),
                              "tree_partners" => J::s("the pair's partner"), "similar" => J::s("every too-similar test and every source, per locus (similar/<locus>.json)"),
                              "tests" => J::s("every cell's comparators and both of each comparator's tests (cells/comp_div_log10p, cells/comp_sim_log10p, log10 p by comparator rank), the closest match at the locus and its too-similar test (cells/locus_closest, cells/locus_closest_log10p)"),
                              "best_match_cells" => J::s(if n <= a.best_match_all_max_taxa { "all" } else { "fails" }) },
        "code_map" => jdict! { "mismatch_count_max" => J::Int(k_eff as i64), "similar_base" => J::Int(128), "tick" => J::Int(254), "absent" => J::Int(255) },
        "side_bits" => jdict! { "masked_divergent" => J::Int(1), "masked_similar" => J::Int(2), "ambiguous" => J::Int(4), "similar_uncertain" => J::Int(8), "flagged_not_masked" => J::Int(16), "similar_not_masked" => J::Int(32), "similar_tested" => J::Int(64), "similar_source" => J::Int(128) },
        "side2_bits" => jdict! { "broad_tested" => J::Int(1), "broad_past_bar" => J::Int(2), "broad_close" => J::Int(4), "broad_divergent" => J::Int(8), "broad_excluded_sample" => J::Int(16), "broad_fired_not_masked" => J::Int(32), "divergent_also_close_label" => J::Int(64) },
        "broad_sample" => rep.bs_summary.clone().unwrap_or(J::Null),
        "restart" => match &rep.restart { Some((after, written)) => jdict! { "after" => J::s(after), "written" => J::s(written) }, None => J::Null },
        "lists_extended" => jdict! { "rate_neighbours" => J::Int(rep.lists_extended.0 as i64), "comparators" => J::Int(rep.lists_extended.1 as i64) },
        "timings" => J::Dict(timings) };

    // ---- report.npz
    {
        let part = format!("{npz}.part");
        let fh = std::fs::File::create(&part).unwrap_or_else(|e| die!("cannot write {}: {}", part, crate::input::os_err(&e)));
        // the members are listed here and made and deflated in parallel (zip_parallel), in this order in the file
        let mut jobs: Vec<crate::npy::Member> = vec![];
        let (cell_shape, with_k, taxa_json, loci_json, manifest) = (&cell_shape, &with_k, &taxa_json, &loci_json, &manifest);
        macro_rules! put { ($name:expr, $data:expr) => { jobs.push(($name.to_string(), Box::new(move || $data))) }; }
        put!("cells/n_comp.npy", npy(&[n, nloc], &rep.n_comp)); put!("cells/n_fail.npy", npy(&[n, nloc], &rep.n_fail)); put!("cells/n_similar.npy", npy(&[n, nloc], &rep.n_similar));
        put!("cells/side.npy", npy(&[n, nloc], &rep.side)); put!("cells/focal_sites.npy", npy(&[n, nloc], &rep.focal_sites));
        if let Some(s2) = &rep.side2 {
            put!("cells/side2.npy", npy(&[n, nloc], s2));
            if let Some(b) = &rep.bs_rows {
                put!("cells/bs_locus.npy", npy1(&b.locus)); put!("cells/bs_taxon.npy", npy1(&b.taxon)); put!("cells/bs_var_log10p.npy", npy1(&b.var_log10p));
                put!("cells/bs_lo_log10p.npy", npy1(&b.lo_log10p)); put!("cells/bs_hi_log10p.npy", npy1(&b.hi_log10p)); put!("cells/bs_partner.npy", npy1(&b.partner)); put!("cells/bs_call.npy", npy1(&b.call));
            }
            if let Some(bc) = &rep.bscols { for j in 0..4 { put!(&format!("cells/{}.npy", BSCOLS[j].0), bc[j].npy(&[n, nloc])); } }
        }
        if !dense { put!("cells/sparse_locus.npy", npy1(&rep.sparse_locus)); put!("cells/sparse_taxon.npy", npy1(&rep.sparse_taxon)); }
        for (j, (name, _, _)) in NUMCOLS.iter().enumerate() { put!(&format!("cells/{name}.npy"), rep.cols[j].npy(cell_shape)); }
        put!("cells/comparators.npy", npy(with_k, &rep.comparators)); put!("cells/comp_div_log10p.npy", npy(with_k, &rep.comp_div)); put!("cells/comp_sim_log10p.npy", npy(with_k, &rep.comp_sim));
        for (li, (cells_, blk)) in &rep.codes.0 { put!(&format!("codes/{li}.npy"), npy(&[cells_.len(), loci[*li].1.len()], blk)); put!(&format!("codes/{li}_cells.npy"), npy1(cells_)); }
        for (li, t) in &trees_json { put!(&format!("trees/{li}.json"), t.clone().into_bytes()); }
        for (li, t) in &similar_json { put!(&format!("similar/{li}.json"), t.clone().into_bytes()); }
        for (li, t) in &broad_json { put!(&format!("broadsample/{li}.json"), t.clone().into_bytes()); }
        for (li, (rows_, blk)) in &rep.seqs.0 { put!(&format!("seq/{li}.npy"), npy(&[rows_.len(), loci[*li].1.len()], blk)); put!(&format!("seq/{li}_taxa.npy"), npy1(rows_)); }
        put!("taxa/closest.npy", npy(&[n, rep.taxa_closest_w], &rep.taxa_closest));
        put!("taxa.json", taxa_json.indent1().into_bytes()); put!("loci.json", loci_json.indent1().into_bytes()); put!("manifest.json", manifest.indent1().into_bytes());
        match crate::npy::zip_parallel(std::io::BufWriter::with_capacity(1 << 20, fh), &jobs) { Ok(mut w) => { use std::io::Write; io(w.flush(), &part) } Err(e) => die!("cannot write {}: {}", part, crate::input::os_err(&e)) }
        drop(jobs);
        io(std::fs::rename(&part, &npz), &npz);
    }
    let size = std::fs::metadata(&npz).map(|m| m.len()).unwrap_or(0);

    // ---- the per-cell columns, read at any set of cells whether the run kept them dense or sparse
    let sp: HashMap<(i32, i32), usize> = if dense { HashMap::new() } else {
        let mut m = HashMap::new(); for i in (0..rep.sparse_locus.len()).rev() { m.insert((rep.sparse_locus[i], rep.sparse_taxon[i]), i); } m };
    let rows_of = |gs: &[usize], lis: &[usize]| -> Vec<Option<usize>> {
        if dense { gs.iter().zip(lis).map(|(g, l)| Some(g * nloc + l)).collect() } else { gs.iter().zip(lis).map(|(g, l)| sp.get(&(*l as i32, *g as i32)).copied()).collect() } };
    let bsc = rep.bscols.as_ref();
    let per_cell: usize = NUMCOLS.iter().map(|_| 4).sum::<usize>() + 4 + bsc.map_or(0, |b| b.iter().map(|c| c.itemsize()).sum()) + 4 * k * 3;
    let cell_members = |prefix: &str, gs: &[usize], lis: &[usize], shape: &[usize]| -> Vec<(String, Vec<u8>)> {
        let rows = rows_of(gs, lis); let mut out = vec![];
        for (j, (name, _, fill)) in NUMCOLS.iter().enumerate() { out.push((format!("{prefix}{name}.npy"), rep.cols[j].gather(&rows, *fill).npy(shape))); }
        let fs: Vec<u32> = gs.iter().zip(lis).map(|(g, l)| rep.focal_sites[g * nloc + l]).collect();
        out.push((format!("{prefix}focal_sites.npy"), npy(shape, &fs)));
        if let Some(b) = bsc {
            let drows: Vec<Option<usize>> = gs.iter().zip(lis).map(|(g, l)| Some(g * nloc + l)).collect();
            for (j, (name, _, fill)) in BSCOLS.iter().enumerate() { out.push((format!("{prefix}{name}.npy"), b[j].gather(&drows, *fill).npy(shape))); }
        }
        let mut shk = shape.to_vec(); shk.push(k);
        let gk = |src: &[i32]| -> Vec<i32> { let mut v = Vec::with_capacity(rows.len() * k); for r in &rows { match r { Some(i) => v.extend_from_slice(&src[i * k..(i + 1) * k]), None => v.extend(std::iter::repeat(-1).take(k)) } } v };
        let gf = |src: &[f32]| -> Vec<f32> { let mut v = Vec::with_capacity(rows.len() * k); for r in &rows { match r { Some(i) => v.extend_from_slice(&src[i * k..(i + 1) * k]), None => v.extend(std::iter::repeat(f32::NAN).take(k)) } } v };
        out.push((format!("{prefix}comparators.npy"), npy(&shk, &gk(&rep.comparators))));
        out.push((format!("{prefix}comp_div_log10p.npy"), npy(&shk, &gf(&rep.comp_div)))); out.push((format!("{prefix}comp_sim_log10p.npy"), npy(&shk, &gf(&rep.comp_sim))));
        out
    };
    // ---- the core: states, the masked or flagged cells' numbers, the ribbon sample and its axes
    let (mut cg, mut cl) = (vec![], vec![]);
    for g in 0..n { for li in 0..nloc { if side[g * nloc + li] & (1 | 2 | 16 | 32) > 0 { cg.push(g); cl.push(li); } } }
    let mut pool_mask = vec![false; n * nloc];
    if dense { for c in 0..n * nloc { pool_mask[c] = tested(c); } }
    else { for i in 0..rep.sparse_locus.len() { let c = rep.sparse_taxon[i] as usize * nloc + rep.sparse_locus[i] as usize; pool_mask[c] = tested(c); } }
    let (mut pg, mut pl) = (vec![], vec![]);
    for g in 0..n { for li in 0..nloc { if pool_mask[g * nloc + li] { pg.push(g); pl.push(li); } } }
    let sample_cells: usize = std::env::var("LOCUSMASKER_REPORT_SAMPLE_CELLS").ok().and_then(|v| v.parse().ok()).unwrap_or(REPORT_SAMPLE_CELLS);   // a test hook
    let take = crate::rng::choice_sorted(0, pg.len(), sample_cells.min(pg.len()));
    let sg: Vec<usize> = take.iter().map(|&i| pg[i]).collect(); let sl: Vec<usize> = take.iter().map(|&i| pl[i]).collect();
    let tb = 1.max(REPORT_TAXA_BLOCK.min(REPORT_CHUNK_BYTES / 1.max(per_cell * nloc)));
    let lb = 1.max(REPORT_TAXA_BLOCK.min(REPORT_CHUNK_BYTES / 1.max(per_cell * n)));
    let (ntb, nlb) = ((n + tb - 1) / tb, (nloc + lb - 1) / lb);
    let layout = jdict! { "kind" => J::s("folder"), "dir" => J::s("report_files"), "taxa_block" => J::Int(tb as i64), "taxa_blocks" => J::Int(ntb as i64), "locus_block" => J::Int(lb as i64),
                          "loci_blocks" => J::Int(nlb as i64), "loci" => J::Int(nloc as i64), "core_cells" => J::Int(cg.len() as i64), "sample_cells" => J::Int(sg.len() as i64) };
    let ex = rep.cols[2].f32s(); let fin: Vec<f32> = ex.iter().copied().filter(|x| x.is_finite()).collect();
    let (lo, hi) = if fin.is_empty() { (J::Null, J::Null) } else { (J::Float(fin.iter().copied().fold(f32::INFINITY, f32::min) as f64), J::Float(fin.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64)) };
    let bmax = match bsc { Some(b) if dense => { let v: Vec<f32> = b[2].f32s().iter().copied().filter(|x| x.is_finite() && *x > 0.0).collect(); if v.is_empty() { J::Null } else { J::Float(v.iter().map(|x| x.log2().abs()).fold(f32::NEG_INFINITY, f32::max) as f64) } } _ => J::Null };
    manifest.set("layout", layout.clone()); manifest.set("ribbons", jdict! { "excess_min" => lo, "excess_max" => hi, "bs_log2_max" => bmax });
    let mut core: Vec<(String, Vec<u8>)> = vec![("manifest.json".into(), manifest.indent1().into_bytes()), ("taxa.json".into(), taxa_json.indent1().into_bytes()), ("loci.json".into(), loci_json.indent1().into_bytes())];
    core.push(("cells/n_comp.npy".into(), npy(&[n, nloc], &rep.n_comp))); core.push(("cells/n_fail.npy".into(), npy(&[n, nloc], &rep.n_fail)));
    core.push(("cells/n_similar.npy".into(), npy(&[n, nloc], &rep.n_similar))); core.push(("cells/side.npy".into(), npy(&[n, nloc], &rep.side)));
    core.push(("cells/present.npy".into(), npy(&[n, nloc], &rep.focal_sites.iter().map(|x| (*x > 0) as u8).collect::<Vec<u8>>())));
    if let Some(s2) = &rep.side2 {
        core.push(("cells/side2.npy".into(), npy(&[n, nloc], s2)));
        if let Some(b) = &rep.bs_rows {
            core.push(("cells/bs_locus.npy".into(), npy1(&b.locus))); core.push(("cells/bs_taxon.npy".into(), npy1(&b.taxon))); core.push(("cells/bs_var_log10p.npy".into(), npy1(&b.var_log10p)));
            core.push(("cells/bs_lo_log10p.npy".into(), npy1(&b.lo_log10p))); core.push(("cells/bs_hi_log10p.npy".into(), npy1(&b.hi_log10p))); core.push(("cells/bs_partner.npy".into(), npy1(&b.partner))); core.push(("cells/bs_call.npy".into(), npy1(&b.call)));
        }
    }
    core.push(("taxa/closest.npy".into(), npy(&[n, rep.taxa_closest_w], &rep.taxa_closest)));
    core.push(("core/locus.npy".into(), npy1(&cl.iter().map(|x| *x as i32).collect::<Vec<i32>>()))); core.push(("core/taxon.npy".into(), npy1(&cg.iter().map(|x| *x as i32).collect::<Vec<i32>>())));
    core.extend(cell_members("core/", &cg, &cl, &[cg.len()]));
    core.push(("sample/locus.npy".into(), npy1(&sl.iter().map(|x| *x as i32).collect::<Vec<i32>>()))); core.push(("sample/taxon.npy".into(), npy1(&sg.iter().map(|x| *x as i32).collect::<Vec<i32>>())));
    let srows = rows_of(&sg, &sl);
    for j in [2usize, 11, 13] { core.push((format!("sample/{}.npy", NUMCOLS[j].0), rep.cols[j].gather(&srows, f64::NAN).npy(&[sg.len()]))); }
    if let (Some(b), true) = (bsc, dense) { core.push(("sample/bs_log10p.npy".into(), b[0].gather(&srows, f64::NAN).npy(&[sg.len()]))); }
    let mcs: Vec<f32> = srows.iter().map(|r| match r {
        Some(i) => { let v = &rep.comp_sim[i * k..(i + 1) * k]; if v.iter().any(|x| x.is_finite()) { v.iter().copied().filter(|x| x.is_finite()).fold(f32::INFINITY, f32::min) } else { f32::NAN } }
        None => f32::NAN }).collect();
    core.push(("sample/min_comp_sim.npy".into(), npy1(&mcs)));
    if let Some(b) = bsc { let drows: Vec<Option<usize>> = sg.iter().zip(&sl).map(|(g, l)| Some(g * nloc + l)).collect(); core.push(("sample/bs_stat.npy".into(), b[2].gather(&drows, f64::NAN).npy(&[sg.len()]))); }
    let core_b64 = zip_b64(&core); drop(core);

    // ---- the chunks: one file per block of loci, one per block of taxa
    let fdir = format!("{}/report_files", a.out); let fpart = format!("{fdir}.part");
    let _ = std::fs::remove_dir_all(&fdir); let _ = std::fs::remove_dir_all(&fpart);
    io(std::fs::create_dir_all(&fpart), &fpart);
    let write_chunk = |name: &str, kind: &str, idx: usize, members: Vec<(String, Vec<u8>)>| {
        let tmp = format!("{fpart}/{name}.part");
        io(std::fs::write(&tmp, format!("LM_CHUNK(\"{}\",{},\"{}\");\n", kind, idx, zip_b64(&members))), &tmp);
        io(std::fs::rename(&tmp, format!("{fpart}/{name}")), name);
    };
    let (codes_ix, seqs_ix) = (rep.codes.index(), rep.seqs.index());
    let find = |v: &[(usize, String)], li: usize| v.iter().find(|x| x.0 == li).map(|x| x.1.clone());
    use rayon::prelude::*;
    (0..nlb).into_par_iter().for_each(|b| {
        let (l0, l1) = (b * lb, nloc.min((b + 1) * lb)); let bw = l1 - l0;
        let gs: Vec<usize> = (0..n).flat_map(|g| std::iter::repeat(g).take(bw)).collect(); let lis: Vec<usize> = (0..n).flat_map(|_| l0..l1).collect();
        let mut members = cell_members("cells/", &gs, &lis, &[n, bw]);
        for li in l0..l1 {
            if let Some(&i) = codes_ix.get(&li) { let (cells_, blk) = &rep.codes.0[i].1; members.push((format!("codes/{li}.npy"), npy(&[cells_.len(), loci[li].1.len()], blk))); members.push((format!("codes/{li}_cells.npy"), npy1(cells_))); }
            if let Some(&i) = seqs_ix.get(&li) { let (rows_, blk) = &rep.seqs.0[i].1; members.push((format!("seq/{li}.npy"), npy(&[rows_.len(), loci[li].1.len()], blk))); members.push((format!("seq/{li}_taxa.npy"), npy1(rows_))); }
            if let Some(t) = find(&trees_json, li) { members.push((format!("trees/{li}.json"), t.into_bytes())); }
            if let Some(t) = find(&similar_json, li) { members.push((format!("similar/{li}.json"), t.into_bytes())); }
            if let Some(t) = find(&broad_json, li) { members.push((format!("broadsample/{li}.json"), t.into_bytes())); }
        }
        write_chunk(&format!("loci_{b}.js"), "loci", b, members);
    });
    (0..ntb).into_par_iter().for_each(|b| {
        let (g0, g1) = (b * tb, n.min((b + 1) * tb)); let bw = g1 - g0;
        let gs: Vec<usize> = (g0..g1).flat_map(|g| std::iter::repeat(g).take(nloc)).collect(); let lis: Vec<usize> = (g0..g1).flat_map(|_| 0..nloc).collect();
        write_chunk(&format!("taxa_{b}.js"), "taxa", b, cell_members("cells/", &gs, &lis, &[bw, nloc]));
    });
    let mut index = layout.clone(); index.set("version", J::s(crate::VERSION)); index.set("dataset", J::Str(dataset.clone()));
    io(std::fs::write(format!("{fpart}/index.js"), format!("window.LM_INDEX={};\n", index.compact())), "index.js");
    io(std::fs::rename(&fpart, &fdir), &fdir);

    // ---- the page: the viewer with the core inline
    let inlined = |part: &str| -> String {
        let re_s = regex::Regex::new(r#"<script src="([^"]+)"></script>"#).unwrap();
        let s = re_s.replace_all(part, |c: &regex::Captures| format!("<script>\n{}\n</script>", text(&c[1]))).into_owned();
        let re_l = regex::Regex::new(r#"<link rel="stylesheet" href="([^"]+)">"#).unwrap();
        let s = re_l.replace_all(&s, |c: &regex::Captures| format!("<style>\n{}\n</style>", text(&c[1]))).into_owned();
        let re_i = regex::Regex::new(r#"(href|src)="(icons/[^"]+)""#).unwrap();
        re_i.replace_all(&s, |c: &regex::Captures| { let kind = if c[2].ends_with(".svg") { "image/svg+xml" } else { "image/png" }; format!("{}=\"data:{};base64,{}\"", &c[1], kind, b64(asset(&c[2]))) }).into_owned()
    };
    let tpl = text("viewer.html");
    let (before, after) = tpl.split_once("@@BUNDLE@@").unwrap_or_else(|| die!("the viewer template has no bundle marker"));
    let before = before.replacen("@@LICENCES@@", &licences_text(), 1);
    let meta_js = jdict! { "dataset" => J::Str(dataset.clone()), "bundle_bytes" => J::Int((core_b64.len() * 3 / 4) as i64), "version" => J::s(crate::VERSION) }.plain();
    let before = before.replacen("@@META@@", &meta_js, 1);
    let page = format!("{}/report.html", a.out); let ppart = format!("{page}.part");
    {
        use std::io::Write;
        let fh = std::fs::File::create(&ppart).unwrap_or_else(|e| die!("cannot write {}: {}", ppart, crate::input::os_err(&e)));
        let mut w = std::io::BufWriter::with_capacity(1 << 20, fh);
        io(w.write_all(inlined(&before).as_bytes()).and_then(|_| w.write_all(core_b64.as_bytes())).and_then(|_| w.write_all(inlined(after).as_bytes())).and_then(|_| w.flush()), &ppart);
    }
    io(std::fs::rename(&ppart, &page), &page);
    let fsize: u64 = std::fs::read_dir(&fdir).map(|d| d.filter_map(|e| e.ok()).filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum()).unwrap_or(0);
    log!("report: report.html ({} MB, {} cells' numbers inline), report_files/ ({} MB in {} files: {} blocks of {} loci, {} of {} taxa), report.npz ({} MB)",
         std::fs::metadata(&page).map(|m| m.len()).unwrap_or(0) >> 20, crate::pyfmt::commas(cg.len() as i64), fsize >> 20, nlb + ntb + 1, nlb, lb, ntb, tb, size >> 20);
    true
}
