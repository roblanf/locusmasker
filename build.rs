use std::process::Command;
fn out(cmd: &str, args: &[&str]) -> Option<String> {
    let o = Command::new(cmd).args(args).output().ok()?;
    if !o.status.success() { return None; }
    let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}
fn main() {
    let commit = out("git", &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let rustc = std::env::var("RUSTC").ok().and_then(|r| out(&r, &["--version"])).unwrap_or_else(|| "rustc".into());
    println!("cargo:rustc-env=LM_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=LM_RUSTC={rustc}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=.git/HEAD");
}
