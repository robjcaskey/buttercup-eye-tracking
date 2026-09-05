//! Extend Buttercup's detector-only SAM3.1 TorchScript archive with native
//! feature outputs needed by the stateful video tracker.
//!
//! This is deliberately an archive-to-archive Rust transformation. It neither
//! imports Python nor deserializes proprietary capture data. Model weights stay
//! external to the source tree and are copied byte-for-byte.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use zip::write::FileOptions;
use zip::{ZipArchive, ZipWriter};

const FORWARD_SIGNATURE: &str = "text_ids: Tensor) -> Tuple[Tensor, Tensor]:";
const FEATURE_SIGNATURE: &str =
    "text_ids: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:";
const WRAPPER_CALL: &str = "_1, _2, = (detector).forward(_0, language_features, language_mask, img_ids, text_ids, )\n    return (_1, _2)";
const FEATURE_WRAPPER_CALL: &str = "_1, _2, _3, _4, _5, _6, = (detector).forward(_0, language_features, language_mask, img_ids, text_ids, )\n    return (_1, _2, _3, _4, _5, _6)";
const DETECTOR_RETURN: &str = "return (_47, _44)";
const FEATURE_DETECTOR_RETURN: &str = "return (_47, _44, _4, _6, _8, hs)";

fn transform_root_code(source: &str) -> Result<String, String> {
    let signature_count = source.matches(FORWARD_SIGNATURE).count();
    if signature_count != 2 {
        return Err(format!(
            "expected two detector forward signatures, found {signature_count}"
        ));
    }
    if source.matches(WRAPPER_CALL).count() != 1 {
        return Err("detector wrapper forward pattern is absent or ambiguous".to_string());
    }
    if source.matches(DETECTOR_RETURN).count() != 1 {
        return Err("detector feature return pattern is absent or ambiguous".to_string());
    }
    let transformed = source
        .replace(FORWARD_SIGNATURE, FEATURE_SIGNATURE)
        .replace(WRAPPER_CALL, FEATURE_WRAPPER_CALL)
        .replace(DETECTOR_RETURN, FEATURE_DETECTOR_RETURN);
    if transformed.matches(FEATURE_SIGNATURE).count() != 2
        || transformed.matches(FEATURE_WRAPPER_CALL).count() != 1
        || transformed.matches(FEATURE_DETECTOR_RETURN).count() != 1
    {
        return Err("post-transform feature graph validation failed".to_string());
    }
    Ok(transformed)
}

/// Export a second text-conditioned decode over the exact same backbone
/// features. No weights, prompt tensors, masks, or arithmetic are changed.
fn add_shared_feature_prompt(source: &str) -> Result<String, String> {
    const METHOD_START: &str = "  def forward(self: __torch__.FixedOuter,\n    image: Tensor,\n";
    const SHARED_START: &str = "  def prompt_from_features(self: __torch__.FixedOuter,\n    feature0: Tensor,\n    feature1: Tensor,\n    feature2: Tensor,\n";
    const BACKBONE: &str = "    trunk = self.trunk\n    _3 = (trunk).forward(image, )\n    _4 = (_0).forward(_3, )\n";
    if source.matches(METHOD_START).count() != 1
        || source.matches(BACKBONE).count() != 1
        || source.matches("class FixedOuter(Module):").count() != 1
        || source.contains("prompt_from_features")
    {
        return Err("shared-feature export requires the known, unextended native detector graph".into());
    }
    let start = source.find(METHOD_START).unwrap();
    let end = source[start..].find(FEATURE_DETECTOR_RETURN)
        .ok_or("shared-feature export lacks the detector feature return")?
        + start + FEATURE_DETECTOR_RETURN.len();
    let mut shared = source[start..end].replace(METHOD_START, SHARED_START)
        .replace(BACKBONE, "    _4 = feature0\n");
    for (old, new) in [
        ("    _6 = (_1).forward(_3, )", "    _6 = feature1"),
        ("    _8 = (_2).forward(_3, )", "    _8 = feature2"),
    ] {
        if shared.matches(old).count() != 1 {
            return Err(format!("shared-feature export lacks unique feature assignment: {old}"));
        }
        shared = shared.replace(old, new);
    }
    if shared.contains("(trunk).forward") || shared.contains("forward(image") {
        return Err("shared-feature prompt still invokes the image backbone".into());
    }
    let wrapper = r#"  def prompt_from_features(self: __torch__.U8Filmstrip,
    feature0: Tensor,
    feature1: Tensor,
    feature2: Tensor,
    language_features: Tensor,
    language_mask: Tensor,
    img_ids: Tensor,
    text_ids: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    detector = self.detector
    _0, _1, _2, _3, _4, _5, = (detector).prompt_from_features(feature0, feature1, feature2, language_features, language_mask, img_ids, text_ids, )
    return (_0, _1, _2, _3, _4, _5)
"#;
    let root = source.replacen("class FixedOuter(Module):",
        &format!("{wrapper}\nclass FixedOuter(Module):"), 1);
    Ok(format!("{root}\n{shared}\n"))
}

fn temporary_output(output: &Path) -> Result<PathBuf, String> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("output filename is not UTF-8: {}", output.display()))?;
    Ok(output.with_file_name(format!(".{name}.building")))
}

fn extend_archive(input: &Path, output: &Path, shared_features: bool) -> Result<(), String> {
    if !input.is_file() {
        return Err(format!("input graph is unavailable: {}", input.display()));
    }
    if output.exists() {
        return Err(format!(
            "refusing to overwrite existing output: {}",
            output.display()
        ));
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create output directory {}: {error}", parent.display()))?;
    }
    let temporary = temporary_output(output)?;
    if temporary.exists() {
        return Err(format!(
            "stale temporary output must be removed explicitly: {}",
            temporary.display()
        ));
    }
    let input_file = File::open(input)
        .map_err(|error| format!("open input graph {}: {error}", input.display()))?;
    let mut archive = ZipArchive::new(BufReader::new(input_file))
        .map_err(|error| format!("read input graph archive: {error}"))?;
    let output_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("create {}: {error}", temporary.display()))?;
    let mut writer = ZipWriter::new(BufWriter::new(output_file));
    let mut transformed_code = 0usize;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| format!("read graph archive entry {index}: {error}"))?;
        let name = entry.name().to_string();
        let options = FileOptions::default()
            .compression_method(entry.compression())
            .large_file(entry.size() > u32::MAX as u64);
        if entry.is_dir() {
            writer
                .add_directory(name, options)
                .map_err(|error| format!("copy graph directory: {error}"))?;
            continue;
        }
        writer
            .start_file(&name, options)
            .map_err(|error| format!("start graph entry {name}: {error}"))?;
        if name.ends_with("/code/__torch__.py") {
            let mut source = String::new();
            entry
                .read_to_string(&mut source)
                .map_err(|error| format!("read root TorchScript code {name}: {error}"))?;
            let transformed = transform_root_code(&source)?;
            let transformed = if shared_features {
                add_shared_feature_prompt(&transformed)?
            } else { transformed };
            writer
                .write_all(transformed.as_bytes())
                .map_err(|error| format!("write transformed TorchScript code: {error}"))?;
            transformed_code += 1;
        } else {
            std::io::copy(&mut entry, &mut writer)
                .map_err(|error| format!("copy graph entry {name}: {error}"))?;
        }
    }
    if transformed_code != 1 {
        return Err(format!(
            "expected one root TorchScript code entry, transformed {transformed_code}"
        ));
    }
    let mut output_file = writer
        .finish()
        .map_err(|error| format!("finish transformed graph archive: {error}"))?;
    output_file
        .flush()
        .map_err(|error| format!("flush transformed graph archive: {error}"))?;
    output_file
        .get_ref()
        .sync_all()
        .map_err(|error| format!("sync transformed graph archive: {error}"))?;
    fs::rename(&temporary, output).map_err(|error| {
        format!(
            "publish transformed graph {} -> {}: {error}",
            temporary.display(),
            output.display()
        )
    })?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if !(2..=3).contains(&arguments.len())
        || arguments.get(2).is_some_and(|argument| argument != "--shared-features")
    {
        return Err("usage: buttercup-sam31-video-graph INPUT.pt OUTPUT.pt [--shared-features]".into());
    }
    let input = PathBuf::from(&arguments[0]);
    let output = PathBuf::from(&arguments[1]);
    extend_archive(&input, &output, arguments.len() == 3)?;
    println!("video_feature_graph={}", output.display());
    println!("outputs=scores,masks,fpn_level_0,fpn_level_1,fpn_level_2,decoder_queries");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_only_the_expected_detector_contract() {
        let source =
            format!("{FORWARD_SIGNATURE}\n{WRAPPER_CALL}\n{FORWARD_SIGNATURE}\n{DETECTOR_RETURN}");
        let transformed = transform_root_code(&source).unwrap();
        assert_eq!(transformed.matches(FEATURE_SIGNATURE).count(), 2);
        assert!(transformed.contains(FEATURE_WRAPPER_CALL));
        assert!(transformed.contains(FEATURE_DETECTOR_RETURN));
    }

    #[test]
    fn refuses_an_unknown_graph_shape() {
        assert!(transform_root_code("different graph").is_err());
        assert!(add_shared_feature_prompt("different graph").is_err());
    }

    #[test]
    fn shared_prompt_preserves_full_forward_and_removes_only_backbone_calls() {
        let detector = format!(
            "class FixedOuter(Module):\n  def forward(self: __torch__.FixedOuter,\n    image: Tensor,\n    {FEATURE_SIGNATURE}\n    trunk = self.trunk\n    _3 = (trunk).forward(image, )\n    _4 = (_0).forward(_3, )\n    _6 = (_1).forward(_3, )\n    _8 = (_2).forward(_3, )\n    {FEATURE_DETECTOR_RETURN}\n"
        );
        let source = format!("class U8Filmstrip(Module):\n{detector}");
        let shared = add_shared_feature_prompt(&source).unwrap();
        assert!(shared.contains(&detector), "original full inference changed");
        assert_eq!(shared.matches("(trunk).forward(image, )").count(), 1);
        assert!(shared.contains("    _4 = feature0\n"));
        assert!(shared.contains("    _6 = feature1\n"));
        assert!(shared.contains("    _8 = feature2\n"));
        assert!(shared.contains("\n    feature0: Tensor,\n"), "TorchScript indentation changed");
        assert!(add_shared_feature_prompt(&shared).is_err(), "duplicate export was accepted");
        assert!(add_shared_feature_prompt(&source.replace("    _8 = (_2).forward(_3, )", "changed")).is_err());
    }
}
