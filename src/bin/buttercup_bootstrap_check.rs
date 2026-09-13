//! Structural bootstrap preflight; no Python, camera, LibTorch or CUDA.
#[path = "../bootstrapability.rs"]
mod bootstrapability;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let first = args
        .next()
        .ok_or("usage: buttercup_bootstrap_check --describe-source|GRAPH.json [--repo DIR]")?;
    let mut repo = std::env::current_dir().map_err(|e| e.to_string())?;
    if let Some(flag) = args.next() {
        if flag != "--repo" {
            return Err("expected --repo DIR".into());
        }
        repo = args.next().ok_or("missing repository directory")?.into();
    }
    if args.next().is_some() {
        return Err("unexpected arguments".into());
    }
    let source = bootstrapability::current_source(&repo)?;
    if first == "--describe-source" {
        println!(
            "{}",
            serde_json::to_string_pretty(&source).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    let metadata = std::fs::metadata(&first).map_err(|e| e.to_string())?;
    if metadata.len() > 8 * 1024 * 1024 {
        return Err("graph exceeds 8 MiB".into());
    }
    let bytes = std::fs::read(first).map_err(|e| e.to_string())?;
    let result = bootstrapability::parse(&bytes)
        .and_then(|manifest| bootstrapability::validate(&manifest, &source));
    match result {
        Ok(certificate) => {
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"valid":true,"certificate":certificate})
                )
                .map_err(|e| e.to_string())?
            );
            Ok(())
        }
        Err(error) => Err(serde_json::json!({"valid":false,"error":error}).to_string()),
    }
}
