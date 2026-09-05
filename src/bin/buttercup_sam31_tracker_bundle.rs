//! Extract the official SAM3.1 tracker tensors into a compact native runtime
//! bundle. Model files remain external data and are never added to source.

use std::fs;
use std::path::{Path, PathBuf};
use tch::{Device, Tensor};

const TRACKER_PREFIX: &str = "tracker.model.";
const REQUIRED_GROUPS: [&str; 6] = [
    "transformer.encoder.layers.0.",
    "maskmem_backbone.mask_downsampler.",
    "maskmem_backbone.fuser.",
    "sam_mask_decoder.",
    "no_obj_ptr_linear.",
    "maskmem_tpos_enc",
];

fn tracker_name(name: &str) -> Option<&str> {
    name.strip_prefix(TRACKER_PREFIX)
}

fn temporary_output(output: &Path) -> Result<PathBuf, String> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("output filename is not UTF-8: {}", output.display()))?;
    Ok(output.with_file_name(format!(".{name}.building")))
}

fn extract_tracker_bundle(checkpoint: &Path, output: &Path) -> Result<usize, String> {
    if !checkpoint.is_file() {
        return Err(format!("checkpoint unavailable: {}", checkpoint.display()));
    }
    if output.exists() {
        return Err(format!("refusing to overwrite {}", output.display()));
    }
    let temporary = temporary_output(output)?;
    if temporary.exists() {
        return Err(format!(
            "stale temporary output must be removed explicitly: {}",
            temporary.display()
        ));
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let loaded = Tensor::loadz_multi_with_device(checkpoint, Device::Cpu)
        .map_err(|error| format!("load {}: {error}", checkpoint.display()))?;
    let tracker = loaded
        .into_iter()
        .filter_map(|(name, tensor)| tracker_name(&name).map(|name| (name.to_string(), tensor)))
        .collect::<Vec<_>>();
    if tracker.len() < 400 {
        return Err(format!(
            "checkpoint exposed only {} tracker tensors; expected at least 400",
            tracker.len()
        ));
    }
    for group in REQUIRED_GROUPS {
        if !tracker.iter().any(|(name, _)| name.starts_with(group)) {
            return Err(format!("tracker checkpoint lacks required group {group}"));
        }
    }
    let borrowed = tracker
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect::<Vec<_>>();
    Tensor::save_multi(&borrowed, &temporary)
        .map_err(|error| format!("write {}: {error}", temporary.display()))?;
    fs::rename(&temporary, output).map_err(|error| {
        format!(
            "publish {} -> {}: {error}",
            temporary.display(),
            output.display()
        )
    })?;
    Ok(tracker.len())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() != 2 {
        return Err("usage: buttercup-sam31-tracker-bundle CHECKPOINT.pt OUTPUT.pt".into());
    }
    let checkpoint = PathBuf::from(&arguments[0]);
    let output = PathBuf::from(&arguments[1]);
    let count = extract_tracker_bundle(&checkpoint, &output)?;
    println!("tracker_bundle={}", output.display());
    println!("tracker_tensors={count}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_only_the_exact_tracker_namespace() {
        assert_eq!(
            tracker_name("tracker.model.maskmem_tpos_enc"),
            Some("maskmem_tpos_enc")
        );
        assert_eq!(tracker_name("detector.model.weight"), None);
        assert_eq!(tracker_name("tracker.modelish.weight"), None);
    }
}
