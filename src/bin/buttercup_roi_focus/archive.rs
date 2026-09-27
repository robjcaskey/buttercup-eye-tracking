use crate::Result;
use buttercup_eye_tracking::geometry::Ellipse;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};
pub const RECORD_BYTES: usize = 160;
#[derive(Clone, Copy, Debug)]
pub struct DiskAreaRecord {
    pub projected_disk_px2: f64,
    pub frontal_equivalent_disk_px2: f64,
    pub independent_linear_scale: f64,
    pub sn_feida: f64,
    pub scale_reference: u32,
    pub flags: u32,
}
impl DiskAreaRecord {
    pub fn missing() -> Self {
        Self {
            projected_disk_px2: f64::NAN,
            frontal_equivalent_disk_px2: f64::NAN,
            independent_linear_scale: f64::NAN,
            sn_feida: f64::NAN,
            scale_reference: u32::MAX,
            flags: 0,
        }
    }
    pub fn from_shape(shape: Option<Ellipse>) -> Self {
        let mut area = Self::missing();
        if let Some(e) = shape.filter(|e| {
            e.major_radius.is_finite()
                && e.minor_radius.is_finite()
                && e.major_radius >= e.minor_radius
                && e.minor_radius > 0.
        }) {
            area.projected_disk_px2 = std::f64::consts::PI * e.major_radius * e.minor_radius;
            area.frontal_equivalent_disk_px2 = std::f64::consts::PI * e.major_radius.powi(2);
            area.flags = 1;
        }
        area
    }
    /// Scale must be independent of this ellipse and its inferred gaze region.
    #[allow(dead_code)]
    pub fn with_independent_scale(mut self, scale: f64, reference: u32) -> Option<Self> {
        if self.flags & 1 == 0 || !scale.is_finite() || scale <= 0. || reference == u32::MAX {
            return None;
        }
        self.independent_linear_scale = scale;
        self.sn_feida = self.frontal_equivalent_disk_px2 / scale.powi(2);
        self.scale_reference = reference;
        self.flags |= 2;
        Some(self)
    }
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"projected_disk_area_px2":self.projected_disk_px2.is_finite().then_some(self.projected_disk_px2),"frontal_equivalent_disk_area_px2":self.frontal_equivalent_disk_px2.is_finite().then_some(self.frontal_equivalent_disk_px2),"independent_linear_scale":self.independent_linear_scale.is_finite().then_some(self.independent_linear_scale),"sn_feida":self.sn_feida.is_finite().then_some(self.sn_feida),"scale_reference":(self.scale_reference!=u32::MAX).then_some(self.scale_reference)})
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Source {
    pub path: String,
    pub prefix: String,
    pub frames_sha256: String,
    pub predictions_sha256: Option<String>,
    pub streams: Vec<String>,
    pub frames: usize,
    pub ellipse_counts: [usize; 4],
    pub unmatched_predictions: usize,
    pub conflicting_ellipses: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: String,
    pub sources: Vec<Source>,
    pub epochs: Vec<String>,
    pub excluded: Vec<serde_json::Value>,
    pub native_conics_path: String,
    pub native_conics_sha256: String,
    pub assumptions: Vec<String>,
    #[serde(default)]
    pub aliases: Vec<serde_json::Value>,
    #[serde(default)]
    pub scale_references: Vec<serde_json::Value>,
}
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub source: u32,
    pub epoch: u32,
    pub eye: u16,
    pub provider: u8,
    pub flags: u8,
    pub sequence: u64,
    pub ns: u64,
    pub host_ns: u64,
    pub origin: [u32; 2],
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub offset: u64,
    pub length: u32,
    pub stream: u32,
    pub ellipse: [f64; 5],
    pub quality: f32,
    pub index: u32,
    pub area: DiskAreaRecord,
}
impl Frame {
    pub fn shape(&self) -> Option<Ellipse> {
        (self.flags & 1 != 0).then(|| Ellipse {
            center: (self.ellipse[0], self.ellipse[1]),
            major_radius: self.ellipse[2],
            minor_radius: self.ellipse[3],
            angle: self.ellipse[4],
        })
    }
    pub fn key(&self) -> (u32, u16, u64, u64) {
        (self.epoch, self.eye, self.ns, self.sequence)
    }
    pub fn encode(&self) -> [u8; RECORD_BYTES] {
        let mut b = Vec::with_capacity(RECORD_BYTES);
        macro_rules! push {($($v:expr),*)=>{$(b.extend_from_slice(&$v.to_le_bytes());)*}}
        push!(
            self.source,
            self.epoch,
            self.eye,
            self.provider,
            self.flags,
            self.sequence,
            self.ns,
            self.host_ns,
            self.origin[0],
            self.origin[1],
            self.width,
            self.height,
            self.stride,
            self.offset,
            self.length,
            self.stream
        );
        for x in self.ellipse {
            push!(x);
        }
        push!(self.quality, self.index);
        push!(
            self.area.projected_disk_px2,
            self.area.frontal_equivalent_disk_px2,
            self.area.independent_linear_scale,
            self.area.sn_feida,
            self.area.scale_reference,
            self.area.flags
        );
        b.resize(RECORD_BYTES, 0);
        b.try_into().unwrap()
    }
    fn decode(b: &[u8]) -> Self {
        let mut p = 0;
        macro_rules! read {
            ($t:ty) => {{
                let n = std::mem::size_of::<$t>();
                let a = b[p..p + n].try_into().unwrap();
                p += n;
                <$t>::from_le_bytes(a)
            }};
        }
        let mut r = Self {
            source: read!(u32),
            epoch: read!(u32),
            eye: read!(u16),
            provider: read!(u8),
            flags: read!(u8),
            sequence: read!(u64),
            ns: read!(u64),
            host_ns: read!(u64),
            origin: [read!(u32), read!(u32)],
            width: read!(u32),
            height: read!(u32),
            stride: read!(u32),
            offset: read!(u64),
            length: read!(u32),
            stream: read!(u32),
            ellipse: std::array::from_fn(|_| read!(f64)),
            quality: read!(f32),
            index: read!(u32),
            area: DiskAreaRecord::missing(),
        };
        debug_assert_eq!(p, 120);
        r.area = if b.len() == RECORD_BYTES {
            DiskAreaRecord {
                projected_disk_px2: read!(f64),
                frontal_equivalent_disk_px2: read!(f64),
                independent_linear_scale: read!(f64),
                sn_feida: read!(f64),
                scale_reference: read!(u32),
                flags: read!(u32),
            }
        } else {
            DiskAreaRecord::from_shape(r.shape())
        };
        debug_assert!(p == 120 || p == 160);
        r
    }
}
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn write(path: &Path, manifest: &Manifest, frames: &[Frame]) -> Result<()> {
    let meta = serde_json::to_vec(manifest)?;
    let mut out = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?,
    );
    out.write_all(b"BCROI002")?;
    out.write_all(&2u32.to_le_bytes())?;
    out.write_all(&(RECORD_BYTES as u32).to_le_bytes())?;
    out.write_all(&(meta.len() as u64).to_le_bytes())?;
    out.write_all(&(frames.len() as u64).to_le_bytes())?;
    out.write_all(&meta)?;
    for f in frames {
        out.write_all(&f.encode())?;
    }
    out.flush()?;
    Ok(())
}
pub fn read(path: &Path) -> Result<(Manifest, Vec<Frame>)> {
    let mut input = BufReader::new(fs::File::open(path)?);
    let mut header = [0; 32];
    input.read_exact(&mut header)?;
    let stride = u32::from_le_bytes(header[12..16].try_into()?) as usize;
    let legacy =
        &header[..8] == b"BCROI001" && header[8..12] == 1u32.to_le_bytes() && stride == 128;
    let current = &header[..8] == b"BCROI002"
        && header[8..12] == 2u32.to_le_bytes()
        && stride == RECORD_BYTES;
    if !legacy && !current {
        return Err("unknown ROI archive header".into());
    }
    let size = u64::from_le_bytes(header[16..24].try_into()?);
    let n = u64::from_le_bytes(header[24..32].try_into()?);
    if size > 128_000_000
        || n > 10_000_000
        || fs::metadata(path)?.len() != 32 + size + n * stride as u64
    {
        return Err("truncated/oversized ROI archive".into());
    }
    let mut meta = vec![0; size as usize];
    input.read_exact(&mut meta)?;
    let manifest: Manifest = serde_json::from_slice(&meta)?;
    let mut frames = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let mut b = [0; RECORD_BYTES];
        input.read_exact(&mut b[..stride])?;
        let f = Frame::decode(&b[..stride]);
        if f.source as usize >= manifest.sources.len()
            || f.epoch as usize >= manifest.epochs.len()
            || f.provider > 3
            || f.stream as usize >= manifest.sources[f.source as usize].streams.len()
        {
            return Err("invalid ROI record reference".into());
        }
        frames.push(f);
    }
    Ok((manifest, frames))
}

pub fn compact(manifest: &mut Manifest, frames: &mut Vec<Frame>) {
    use serde_json::json;
    use std::collections::BTreeMap;
    let mut excluded =
        BTreeMap::<(String, String, String), (usize, Option<u64>, Option<u64>)>::new();
    for r in &manifest.excluded {
        let key = (
            r["path"].as_str().unwrap_or("").into(),
            r["prefix"].as_str().unwrap_or("").into(),
            r["reason"].as_str().unwrap_or("").into(),
        );
        let entry = excluded.entry(key).or_default();
        entry.0 += r["count"].as_u64().unwrap_or(1) as usize;
        if let Some(n) = r["row"].as_u64() {
            entry.1 = Some(entry.1.unwrap_or(n).min(n));
            entry.2 = Some(entry.2.unwrap_or(n).max(n));
        }
    }
    manifest.excluded=excluded.into_iter().map(|((path,prefix,reason),(count,first,last))|json!({"path":path,"prefix":prefix,"reason":reason,"count":count,"first_row":first,"last_row":last})).collect();
    let mut digests = vec![Sha256::new(); manifest.sources.len()];
    for f in frames.iter() {
        let mut copy = *f;
        copy.source = 0;
        digests[f.source as usize].update(copy.encode());
    }
    let mut groups = BTreeMap::new();
    let mut remap = vec![];
    let mut sources = vec![];
    let mut canonical = vec![];
    for (s, digest) in manifest.sources.iter().zip(digests) {
        let key = (
            s.frames_sha256.clone(),
            s.predictions_sha256.clone(),
            format!("{:x}", digest.finalize()),
        );
        if let Some(&id) = groups.get(&key) {
            remap.push(id);
            canonical.push(false);
            manifest.aliases.push(json!({"canonical_source":id,"alias":s,"equivalence":"identical per-frame ellipse/clock/crop records and input metadata hashes; RAW payload equality is not claimed"}));
        } else {
            let id = sources.len() as u32;
            groups.insert(key, id);
            remap.push(id);
            canonical.push(true);
            sources.push(s.clone());
        }
    }
    frames.retain_mut(|f| {
        let old = f.source as usize;
        if canonical[old] {
            f.source = remap[old];
            true
        } else {
            false
        }
    });
    manifest.sources = sources;
    manifest.schema = "buttercup-roi-ellipse-binary-v2".into();
    manifest.assumptions.retain(|s| {
        !s.starts_with("128-byte") && !s.starts_with("160-byte") && !s.starts_with("Disk area:")
    });
    manifest.assumptions.push("160-byte little-endian records; v2 explicitly stores projected disk area pi*a*b, frontal-equivalent disk area pi*a^2, independent linear scale, SN-FEIDA, scale reference and validity flags. Missing area/scale is NaN plus flags, never zero.".into());
    manifest.assumptions.push("Disk area: weak-perspective outer-limbus disk, not visible mask or curved tissue; same for both signs. SN-FEIDA=pi*a^2/s^2 requires independent scale. This historical export has no verified independent scale and leaves normalized area unavailable.".into());
    for note in &mut manifest.assumptions {
        if note.contains("duplicate capture copies remain separate") {
            *note="Every indexed ROI frame is retained, including missing ellipses; identical ellipse-history/metadata copies are explicit aliases and never pooled as independent evidence".into();
        }
    }
}

pub fn compact_file(input: &str, output: &str) -> Result<()> {
    let (mut manifest, mut frames) = read(Path::new(input))?;
    let before = frames.len();
    compact(&mut manifest, &mut frames);
    write(Path::new(output), &manifest, &frames)?;
    let (_, restored) = read(Path::new(output))?;
    if restored
        .iter()
        .zip(&frames)
        .any(|(a, b)| a.encode() != b.encode())
        || restored.len() != frames.len()
    {
        return Err("compacted roundtrip failed".into());
    }
    let summary = serde_json::json!({"binary":output,"sha256":digest(&fs::read(output)?),"bytes":fs::metadata(output)?.len(),"records":frames.len(),"before_alias_deduplication":before,"roundtrip_exact":true,"manifest":manifest});
    fs::write(
        Path::new(output).with_extension("manifest.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    eprintln!(
        "COMPACT {} -> {} frame records, {} recordings and {} metadata-identical aliases",
        before,
        frames.len(),
        manifest.sources.len(),
        manifest.aliases.len()
    );
    Ok(())
}
