//! Bind a reusable SAM-only input cache to its actual preparation recipe.
//! Mapper/renderer edits need not rerun an unchanged foundation-model input path.
use sha2::{Digest, Sha256};
pub fn stamp() -> Result<String, std::io::Error> {
    let mut hash = Sha256::new();
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "src/raw10.rs",
        "src/raw_preview.rs",
        "src/recorded_bundle.rs",
        "src/bootstrapability.rs",
        "src/sclera_splat_input_recipe.rs",
        "src/bin/buttercup_calibration_sign.rs",
        "src/bin/buttercup_calibration_sign/data.rs",
        "src/bin/buttercup_calibration_sign/roi_anatomy.rs",
        "src/bin/buttercup_calibration_sign/sam_native.rs",
        "src/bin/buttercup_calibration_sign/sam_export.rs",
        "src/bin/buttercup_calibration_sign/sclera_splat_inputs.rs",
    ] {
        hash.update(path.as_bytes());
        hash.update([0]);
        hash.update(Sha256::digest(std::fs::read(path)?));
    }
    Ok(format!("{:x}", hash.finalize()))
}
