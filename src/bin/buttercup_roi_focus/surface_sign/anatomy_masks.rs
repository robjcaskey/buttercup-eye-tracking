//! Verified fresh SAM sclera proposals, separate from the historical Obelisk masks.
use super::*;
pub(super) struct AnatomyMasks {
    pub meta: Value,
    frames: BTreeMap<u64, (Value, Vec<u8>)>,
}
impl AnatomyMasks {
    pub fn open(path: &str, area_summary_sha: &str) -> Result<Self> {
        let path = Path::new(path);
        let summary = load(&path.join("summary.json"))?;
        let input = load(&path.join("inputs.json"))?;
        if summary["complete"] != true
            || summary["device"] != "cpu"
            || input["area_summary_sha256"] != area_summary_sha
        {
            return Err("anatomy masks must come from this completed area-first CPU run".into());
        }
        let bytes = fs::read(path.join("frames.jsonl"))?;
        let mut frames = BTreeMap::new();
        for row in rows(&path.join("frames.jsonl"))? {
            let prompt = row["prompts"]
                .as_array()
                .ok_or("anatomy prompts")?
                .iter()
                .find(|v| v["prompt"] == "white of the eye")
                .ok_or("explicit white of the eye prompt required")?;
            let choice = &prompt["candidates"][0];
            let mut probabilities = vec![0u8; 6 * 384 * 256];
            if let Some(name) = choice["mask"].as_str() {
                let mask = fs::read(path.join(name))?;
                let w = n(&choice["width"]) as usize;
                let h = n(&choice["height"]) as usize;
                if (w, h) != (420, 280)
                    || mask.len() != w * h
                    || archive::digest(&mask) != choice["mask_sha256"]
                {
                    return Err("anatomy mask hash/shape mismatch".into());
                }
                // Existing mask sampler uses a 384x256 grid. Nearest-center
                // sampling changes only the mask; photometry still reads each
                // individual native RAW green photosite without averaging.
                for y in 0..256 {
                    for x in 0..384 {
                        let xx = ((x * 2 + 1) * w / (384 * 2)).min(w - 1);
                        let yy = ((y * 2 + 1) * h / (256 * 2)).min(h - 1);
                        probabilities[3 * 384 * 256 + y * 384 + x] = mask[yy * w + xx];
                    }
                }
            }
            let mut provenance = row.clone();
            provenance["selected_sclera_proposal"] = choice.clone();
            if frames
                .insert(n(&row["record"]), (provenance, probabilities))
                .is_some()
            {
                return Err("duplicate RAW record in anatomy run".into());
            }
        }
        Ok(Self {
            meta: json!({"source":path,"frames_sha256":archive::digest(&bytes),"summary":summary,
            "selection":"highest-scoring nonempty white of the eye proposal, before observing either sign; missing proposal supplies no samples",
            "resampling":"nearest-center mask sampling 420x280 to 384x256; native RAW photometry unchanged",
            "lid_veto":"disabled: neither tested model supplies reliable lid boundaries; mask contamination remains a failure mode"}),
            frames,
        })
    }
    pub fn contains(&self, id: u64) -> bool {
        self.frames.contains_key(&id)
    }
    pub fn get(&self, row: &Value) -> Result<Vec<u8>> {
        let (saved, mask) = self
            .frames
            .get(&n(&row["record"]))
            .ok_or("missing anatomy input")?;
        if saved["raw_sha256"] != row["raw_sha256"]
            || saved["frame"] != row["frame"]
            || !saved["area_admitted_providers"]
                .as_array()
                .ok_or("provider eligibility")?
                .contains(&row["provider"])
        {
            return Err("anatomy mask does not match this admitted RAW/provider".into());
        }
        Ok(mask.clone())
    }
}
