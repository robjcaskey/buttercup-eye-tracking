//! JSON-driven orchestration of the shared RAW selection, SAM and renderer.
use super::{data, pupil_prompts, sam_export, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{fs, path::Path, time::Instant};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    prompts: Vec<String>,
    title: String,
    #[serde(default)]
    selection: Option<String>,
    #[serde(default)]
    area_first: Option<String>,
    #[serde(default)]
    seed: Option<String>,
    #[serde(default)]
    sampling: Option<String>,
    #[serde(default)]
    export: Option<String>,
    #[serde(default)]
    checkpoint: Option<String>,
}

fn absolute(path: &str) -> Result<String> {
    Ok(fs::canonicalize(path)?
        .to_str()
        .ok_or("non-UTF8 path")?
        .to_owned())
}

pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 3 {
        return Err(
            "roi-contact-sheet REQUEST_JSON NEW_OUTPUT_DIR; see docs/sam-roi-contact-sheets.md"
                .into(),
        );
    }
    let mut request: Request = serde_json::from_slice(&fs::read(&args[1])?)?;
    if request.prompts.len() != 6 || request.prompts.iter().any(|p| p.trim().is_empty()) {
        return Err("exactly six nonempty literal prompts required".into());
    }
    if request.title.trim().is_empty() || request.title.chars().count() > 96 {
        return Err("title must contain 1-96 characters".into());
    }
    if request.selection.is_some() == request.area_first.is_some() {
        return Err("provide exactly one of selection or area_first".into());
    }
    if request.selection.is_some() && (request.seed.is_some() || request.sampling.is_some()) {
        return Err("selection reuses its frozen seed and sampling; omit seed and sampling".into());
    }
    if request.area_first.is_some() {
        if request.seed.as_deref().unwrap_or("").trim().is_empty() {
            return Err("new corpus selection requires an explicit nonempty seed".into());
        }
        let mode = request
            .sampling
            .get_or_insert_with(|| "matched-random".into());
        if !["matched-random", "matched", "natural"].contains(&mode.as_str()) {
            return Err("sampling must be matched-random, matched or natural".into());
        }
    }
    if request.export.is_some() && request.checkpoint.is_some() {
        return Err("provide export or checkpoint, not both".into());
    }
    for field in [
        &mut request.selection,
        &mut request.area_first,
        &mut request.export,
    ] {
        if let Some(path) = field {
            *path = absolute(path)?;
        }
    }
    if request.export.is_none() {
        request.checkpoint = Some(absolute(
            request
                .checkpoint
                .as_deref()
                .unwrap_or("data/models/sam31_multiplex.pt"),
        )?);
    } else {
        let receipt: serde_json::Value = serde_json::from_slice(&fs::read(
            Path::new(request.export.as_ref().unwrap()).join("export.json"),
        )?)?;
        if receipt["prompts"] != json!(request.prompts) {
            return Err(
                "cached export prompts must exactly match requested prompts, including order"
                    .into(),
            );
        }
    }
    let out = data::output(&args[2])?;
    let out = fs::canonicalize(out)?;
    data::write(out.join("request.json"), &request)?;
    data::write(out.join("prompts.json"), &request.prompts)?;
    let started = Instant::now();
    let mut receipt = json!({
        "schema":"buttercup-roi-contact-sheet-run-v1", "complete":false,
        "request_sha256":sam_export::hash(&out.join("request.json"))?,
        "executable_sha256":sam_export::hash(Path::new("/proc/self/exe"))?,
        "device":"cpu", "cached_export":request.export.is_some()
    });
    data::write(out.join("run.json"), &receipt)?;
    let selection = if let Some(selection) = &request.selection {
        selection.clone()
    } else {
        let selection = out
            .join("selection")
            .to_str()
            .ok_or("selection path")?
            .to_owned();
        pupil_prompts::select(&[
            "pupil-prompt-select".into(),
            request.area_first.clone().unwrap(),
            selection.clone(),
            request.sampling.clone().unwrap(),
            request.seed.clone().unwrap(),
        ])?;
        selection
    };
    let export = if let Some(export) = &request.export {
        export.clone()
    } else {
        let export = out.join("export").to_str().ok_or("export path")?.to_owned();
        sam_export::run(&[
            "sam-export-anatomy-cpu".into(),
            request.checkpoint.clone().unwrap(),
            export.clone(),
            out.join("prompts.json")
                .to_str()
                .ok_or("prompt path")?
                .to_owned(),
        ])?;
        export
    };
    let results = out.join("results");
    let prompts: Vec<_> = request.prompts.iter().map(String::as_str).collect();
    pupil_prompts::run_prompts(
        Path::new(&selection),
        Path::new(&export),
        results.to_str().ok_or("results path")?,
        &prompts,
        &request.title,
        "custom",
    )?;
    let readme = fs::read_to_string(results.join("README.md"))?;
    let description = readme
        .split("Reproduce with new output paths")
        .next()
        .unwrap();
    fs::write(results.join("README.md"), format!("{description}Reproduce from the repository root with a new output directory:\n\n    buttercup_calibration_sign roi-contact-sheet {} NEW_OUTPUT_DIR\n\nSee docs/sam-roi-contact-sheets.md for runtime setup and review guidance.\n", out.join("request.json").display()))?;
    receipt["complete"] = json!(true);
    receipt["seconds"] = json!(started.elapsed().as_secs_f64());
    receipt["selection"] = json!(selection);
    receipt["export"] = json!(export);
    receipt["results"] = json!(results);
    receipt["summary_sha256"] = json!(sam_export::hash(&results.join("summary.json"))?);
    data::write(out.join("run.json"), &receipt)?;
    eprintln!(
        "CONTACT SHEET READY: {}",
        results.join("contact-sheet.png").display()
    );
    Ok(())
}
