//! Small, fixed-vocabulary CUDA mask student. SAM supplies pseudo-labels only;
//! the shared observed-contour/RAW/3D solver remains the geometry authority.
//! Weights, teacher exports, and reports belong under the runtime data links.
use super::*;
#[path = "sam31_student_raw.rs"]
pub mod raw;
#[cfg(feature = "sam31")]
#[path = "bootstrapability.rs"]
mod bootstrap;

pub const ARCHITECTURE: &str = "buttercup-eye-mask-unet-v1";
pub const LUMA_CONTEXT_ARCHITECTURE: &str = "buttercup-eye-mask-unet-luma-context-v2";
pub const RAW_ARCHITECTURE: &str = "buttercup-eye-mask-unet-raw16-v1";
pub const RAW_DISPLAY_NAME: &str = "Butter Obelisk";

/// Inference placement only. Shared offline training retains its explicit CUDA
/// recipe; choosing CPU must never enter a CUDA stream or synchronize CUDA.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InferenceDevice {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

impl InferenceDevice {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            _ => Err(format!("Obelisk device must be auto, cpu, or gpu; got {value:?}")),
        }
    }
    pub fn configured() -> Result<Self, String> {
        Self::parse(&std::env::var("BUTTERCUP_OBELISK_DEVICE").unwrap_or_else(|_| "auto".into()))
    }
    pub fn label(self) -> &'static str {
        match self { Self::Auto => "auto", Self::Cpu => "cpu", Self::Gpu => "gpu" }
    }
    pub fn select_gpu(self, available: bool) -> Result<bool, String> {
        match self {
            Self::Cpu => Ok(false),
            Self::Auto => Ok(available),
            Self::Gpu if available => Ok(true),
            Self::Gpu => Err("Obelisk GPU was forced but CUDA is unavailable; use --obelisk-device cpu or auto".into()),
        }
    }
}

#[cfg(test)]
mod inference_device_tests {
    use super::*;
    #[test]
    fn inference_device_selection_is_explicit_and_auto_falls_back() {
        assert_eq!(InferenceDevice::default(), InferenceDevice::Auto);
        for available in [false, true] {
            assert!(!InferenceDevice::Cpu.select_gpu(available).unwrap());
            assert_eq!(InferenceDevice::Auto.select_gpu(available).unwrap(), available);
        }
        assert!(InferenceDevice::Gpu.select_gpu(true).unwrap());
        assert!(InferenceDevice::Gpu.select_gpu(false).is_err());
        for device in [InferenceDevice::Auto, InferenceDevice::Cpu, InferenceDevice::Gpu] {
            assert_eq!(InferenceDevice::parse(device.label()).unwrap(), device);
        }
        assert!(InferenceDevice::parse("gup").is_err());
    }
}

fn display_name_for_architecture(architecture: &str) -> &'static str {
    match architecture {
        RAW_ARCHITECTURE => RAW_DISPLAY_NAME,
        ARCHITECTURE => "Eye Student RGB",
        LUMA_CONTEXT_ARCHITECTURE => "Eye RGB+Luma",
        _ => "Student missing",
    }
}

/// A process selects one immutable student model path at startup. Cache only
/// its validated display identity; never read model files in the render loop.
/// Stable acquisition/control identifiers remain `eye-student`.
pub fn configured_display_name() -> &'static str {
    static NAME: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        validate_model(&default_model_path()).ok()
            .and_then(|meta|meta["architecture"].as_str().map(display_name_for_architecture))
            .unwrap_or("Student missing")
    })
}

#[cfg(test)]
mod display_name_tests {
    use super::*;
    #[test]
    fn raw_and_rgb_assets_have_distinct_human_names() {
        assert_eq!(display_name_for_architecture(RAW_ARCHITECTURE),"Butter Obelisk");
        assert_eq!(display_name_for_architecture(ARCHITECTURE),"Eye Student RGB");
        assert_eq!(display_name_for_architecture(LUMA_CONTEXT_ARCHITECTURE),"Eye RGB+Luma");
        assert_eq!(display_name_for_architecture("unknown"),"Student missing");
        for architecture in [RAW_ARCHITECTURE,ARCHITECTURE,LUMA_CONTEXT_ARCHITECTURE,"unknown"] {
            assert!((6+display_name_for_architecture(architecture).len())*24<=520);
        }
    }
}

fn luma_context_contract() -> serde_json::Value {
    serde_json::json!({"shape":[2,16,24],"source":"current RGB input only",
        "channels":["block_mean_luma","block_luma_standard_deviation"],
        "luma_weights":[0.25,0.5,0.25],"block_side":16,
        "meaning":"appearance context, not measured physical illumination"})
}

fn known_architecture_manifest(meta: &serde_json::Value) -> bool {
    meta["architecture"] == ARCHITECTURE
        || (meta["architecture"] == RAW_ARCHITECTURE && meta["raw_input"] == raw::contract())
        || (meta["architecture"] == LUMA_CONTEXT_ARCHITECTURE
            && meta["derived_context"] == luma_context_contract())
}

pub fn default_model_path() -> PathBuf {
    std::env::var_os("BUTTERCUP_EYE_STUDENT_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/models/eye_student_raw_v5.ot"))
}

pub fn metadata_path(model: &Path) -> PathBuf {
    model.with_extension("json")
}

pub fn validate_model(model: &Path) -> Result<serde_json::Value, String> {
    if !cfg!(feature = "sam31") {
        return Err("eye student requires CUDA/LibTorch (--features sam31)".into());
    }
    if !model.is_file() {
        return Err(format!("eye student weights missing: {}", model.display()));
    }
    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(metadata_path(model)).map_err(|e| format!("student manifest: {e}"))?,
    )
    .map_err(|e| e.to_string())?;
    let raw_native=meta["architecture"]==RAW_ARCHITECTURE;
    let shape=if raw_native {serde_json::json!([raw::CHANNELS,raw::HEIGHT,raw::WIDTH])}
        else {serde_json::json!([3,FRAME_HEIGHT,FRAME_WIDTH])};
    if !known_architecture_manifest(&meta)
        || meta["input_shape"] != shape
        || meta["prompts"] != serde_json::json!(SEMANTIC_PROMPT_LABELS)
        || meta["preprocess"] != if raw_native {raw::CONTRACT} else {PreprocessRegime::configured_live()?.label()}
    {
        return Err("eye student manifest/shape/preprocessing does not match this runtime".into());
    }
    Ok(meta)
}

#[cfg(feature = "sam31")]
mod cuda {
    use super::*;
    use serde_json::{json, Value};
    use std::fs::{File, OpenOptions};
    use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
    use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
    use sha2::{Digest,Sha256};

    fn verify_source(row:&Value,bytes:&[u8])->Result<(),String> {
        let expected=row["raw_sha256"].as_str().ok_or("missing RAW hash")?;
        if format!("{:x}",Sha256::digest(bytes))!=expected {return Err("RAW source hash mismatch".into());}
        if row["frame"]["pixel_format"]!="RAW10_LE40_1X1" {return Err("source readout is not declared native Quad Bayer RAW10_LE40_1X1".into());}
        Ok(())
    }

    /// No pose regression, completed-ellipse raster, or future-frame input.
    /// Separate sigmoid heads preserve overlapping outer disk/pupil masks.
    #[derive(Debug)]
    pub struct Net {
        stem: nn::Sequential,
        down1: nn::Sequential,
        down2: nn::Sequential,
        bottom: nn::Sequential,
        up2: nn::Sequential,
        up1: nn::Sequential,
        head: nn::Conv2D,
        luma_context: Option<nn::Sequential>,
        raw_native: bool,
    }
    fn block(p: nn::Path, input: i64, output: i64, stride: i64) -> nn::Sequential {
        nn::seq()
            .add(nn::conv2d(
                &p / "a",
                input,
                output,
                3,
                nn::ConvConfig {
                    stride,
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add(nn::group_norm(&p / "an", 4, output, Default::default()))
            .add_fn(Tensor::silu)
            .add(nn::conv2d(
                &p / "b",
                output,
                output,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add(nn::group_norm(&p / "bn", 4, output, Default::default()))
            .add_fn(Tensor::silu)
    }
    impl Net {
        pub fn new(p: &nn::Path) -> Self {
            Self::with_luma_context(p, false)
        }
        pub fn with_luma_context(p: &nn::Path, enabled: bool) -> Self {
            Self::with_input(p,enabled,false)
        }
        pub fn with_input(p: &nn::Path, enabled: bool, raw_native: bool) -> Self {
            Self {
                stem: block(p / "stem", if raw_native {16} else {3}, 12, if raw_native {1} else {2}),
                down1: block(p / "down1", 12, 24, 2),
                down2: block(p / "down2", 24, 48, 2),
                bottom: block(p / "bottom", 48, 72, 2),
                up2: block(p / "up2", 72 + 48 + if enabled { 8 } else { 0 }, 48, 1),
                up1: block(p / "up1", 48 + 24, 24, 1),
                head: nn::conv2d(
                    p / "head",
                    24 + 12,
                    SEMANTIC_PROMPT_COUNT as i64,
                    1,
                    Default::default(),
                ),
                luma_context: enabled.then(|| {
                    let config = nn::ConvConfig {
                        padding: 1,
                        ..Default::default()
                    };
                    nn::seq()
                        .add(nn::conv2d(p / "luma_context" / "a", 2, 8, 3, config))
                        .add_fn(Tensor::silu)
                        .add(nn::conv2d(p / "luma_context" / "b", 8, 8, 3, config))
                        .add_fn(Tensor::silu)
                }),
                raw_native,
            }
        }
    }

    /// One small map per current exposure, before any fitted-center or gaze
    /// input. The original image remains present. This is not a Retinex
    /// decomposition or measured incident light; anatomy also affects luma.
    fn coarse_luma_context(rgb: &Tensor) -> Tensor {
        let y = rgb.narrow(1, 0, 1) * 0.25 + rgb.narrow(1, 1, 1) * 0.5 + rgb.narrow(1, 2, 1) * 0.25;
        let pool = |value: &Tensor| value.avg_pool2d([16, 16], [16, 16], [0, 0], false, true, None);
        let mean = pool(&y);
        let variance = (pool(&(&y * &y)) - &mean * &mean).clamp_min(0.0);
        Tensor::cat(&[mean, variance.sqrt()], 1)
    }
    impl Module for Net {
        fn forward(&self, input: &Tensor) -> Tensor {
            let rgb = input.to_kind(Kind::Float) / if self.raw_native {1.0} else {255.0};
            let a = self.stem.forward(&rgb);
            let b = self.down1.forward(&a);
            let c = self.down2.forward(&b);
            let mut d = self.bottom.forward(&c);
            if let Some(context) = &self.luma_context {
                d = Tensor::cat(&[d, context.forward(&coarse_luma_context(&rgb))], 1);
            }
            let up = |x: &Tensor, y: &Tensor| {
                x.upsample_bilinear2d([y.size()[2], y.size()[3]], false, None, None)
            };
            let e = self.up2.forward(&Tensor::cat(&[up(&d, &c), c], 1));
            let f = self.up1.forward(&Tensor::cat(&[up(&e, &b), b], 1));
            self.head
                .forward(&Tensor::cat(&[up(&f, &a), a], 1))
                .upsample_bilinear2d([FRAME_HEIGHT as i64, FRAME_WIDTH as i64], false, None, None)
        }
    }
    pub struct Model {
        pub store: nn::VarStore,
        pub net: Net,
        pub raw_native: bool,
    }
    impl Model {
        pub fn load(path: &Path) -> Result<Self, String> {
            Self::load_on_device(path, runtime::student_inference_device()?)
        }
        pub fn load_on_device(path: &Path, device: Device) -> Result<Self, String> {
            let metadata = validate_model(path)?;
            let mut store = nn::VarStore::new(device);
            let raw_native=metadata["architecture"]==RAW_ARCHITECTURE;
            let net = Net::with_input(
                &store.root(),
                metadata["architecture"] == LUMA_CONTEXT_ARCHITECTURE,
                raw_native,
            );
            store.load(path).map_err(|e| format!("load student: {e}"))?;
            store.freeze();
            Ok(Self { store, net, raw_native })
        }
        pub fn infer(&self, image: &[u8]) -> Result<Tensor, String> {
            if self.raw_native {return Err("RAW student rejects display RGB input".into());}
            if image.len() != FRAME_WIDTH * FRAME_HEIGHT * 3 {
                return Err("invalid student image shape".into());
            }
            tch::no_grad(|| {
                Ok(self.net.forward(
                    &Tensor::from_slice(image)
                        .reshape([1, 3, FRAME_HEIGHT as i64, FRAME_WIDTH as i64])
                        .to_device(self.store.device()),
                ))
            })
        }

        /// The live worker and offline evaluation use this exact preparation.
        /// Synchronization is opt-in for stage benchmarks (whole device, so
        /// claims must disclose other CUDA work); normal workers synchronize
        /// their own stream through existing mask materialization.
        pub fn infer_source(&self, source:&Arc<RawFrame>, timing:bool)->Result<(Tensor,[f64;3]),String> {
            let start=Instant::now();
            let input=if self.raw_native {
                let values=prepare_raw(source)?;
                Tensor::from_slice(&values).reshape([1,16,raw::HEIGHT as i64,raw::WIDTH as i64])
            } else {
                let mut values=vec![0;3*FRAME_HEIGHT*FRAME_WIDTH];
                write_preprocessed_filmstrip(std::slice::from_ref(source),PreprocessRegime::configured_live()?,&mut values)?;
                Tensor::from_slice(&values).reshape([1,3,FRAME_HEIGHT as i64,FRAME_WIDTH as i64])
            };
            let prepare_ms=start.elapsed().as_secs_f64()*1000.;
            let start=Instant::now();
            let input=input.to_device(self.store.device());
            if timing {if let Device::Cuda(index)=self.store.device(){tch::Cuda::synchronize(index as i64);}}
            let transfer_ms=start.elapsed().as_secs_f64()*1000.;
            let start=Instant::now();
            let logits=tch::no_grad(||self.net.forward(&input));
            if timing {if let Device::Cuda(index)=self.store.device(){tch::Cuda::synchronize(index as i64);}}
            Ok((logits,[prepare_ms,transfer_ms,start.elapsed().as_secs_f64()*1000.]))
        }
    }
    pub(crate) fn prepare_raw(source:&RawFrame)->Result<Vec<f32>,String> {
        raw::prepare(raw::Source {samples:&source.pixels,width:source.width,height:source.height,
            sensor_x:source.sensor_x,sensor_y:source.sensor_y,pixel_format:"RAW10_LE40_1X1"})
    }
    pub(crate) struct TeacherSample {
        pub image: Vec<u8>,
        pub raw_image: Vec<f32>,
        pub masks: Vec<u8>,
        pub weights: Vec<f32>,
        pub report: Value,
    }
    fn integer(v: &Value, key: &str) -> Result<u64, String> {
        v[key]
            .as_u64()
            .or_else(|| v[key].as_str()?.parse().ok())
            .ok_or_else(|| format!("missing {key}"))
    }
    pub fn source_frames(
        path: &Path,
    ) -> Result<impl Iterator<Item = Result<(Value, Arc<RawFrame>), String>>, String> {
        let rows = BufReader::new(File::open(path).map_err(|e| e.to_string())?).lines();
        Ok(rows.map(|line| {
            let row: Value = serde_json::from_str(&line.map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            let mut file = File::open(row["raw_file"].as_str().ok_or("missing RAW file")?)
                .map_err(|e| e.to_string())?;
            file.seek(SeekFrom::Start(integer(&row, "raw_offset")?))
                .map_err(|e| e.to_string())?;
            let length = integer(&row, "raw_length")? as usize;
            if length > 64 * 1024 * 1024 {
                return Err("oversized RAW frame".into());
            }
            let mut bytes = vec![0; length];
            file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
            verify_source(&row,&bytes)?;
            let f = &row["frame"];
            let width = integer(f, "width")? as usize;
            let height = integer(f, "height")? as usize;
            let raw = Arc::new(RawFrame {
                eye_index: integer(f, "eye_id")?
                    .checked_sub(1)
                    .filter(|v| *v < 2)
                    .ok_or("invalid eye")? as usize,
                sequence: integer(f, "sequence")?,
                timestamp_ns: integer(f, "timestamp_ns")?,
                sensor_x: integer(f, "sensor_x")? as u32,
                sensor_y: integer(f, "sensor_y")? as u32,
                width,
                height,
                registration_anchor: None,
                pupil_component_seed: None,
                pixels: Arc::new(crate::raw10::try_unpack_raw10(
                    &bytes,
                    width,
                    height,
                    integer(f, "stride")? as usize,
                )?),
            });
            Ok((row, raw))
        }))
    }
    fn runtime_output(path: &Path) -> Result<(), String> {
        let root = std::fs::canonicalize("data").map_err(|e| e.to_string())?;
        let parent = std::fs::canonicalize(path.parent().ok_or("output requires parent")?)
            .map_err(|e| e.to_string())?;
        if !parent.starts_with(root) {
            return Err("student artifacts must be beneath data/outputs".into());
        }
        if path.exists() {
            return Err(format!("refusing to replace {}", path.display()));
        }
        Ok(())
    }
    fn new_writer(path: &Path) -> Result<BufWriter<File>, String> {
        Ok(BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|e| e.to_string())?,
        ))
    }
    fn write_json(path: &Path, value: &Value) -> Result<(), String> {
        serde_json::to_writer_pretty(new_writer(path)?, value).map_err(|e| e.to_string())
    }
    fn hash_file(path:&Path)->Result<String,String> {
        let mut file=File::open(path).map_err(|e|e.to_string())?;
        let mut hash=Sha256::new();let mut buffer=vec![0;1024*1024];
        loop {let n=file.read(&mut buffer).map_err(|e|e.to_string())?;if n==0 {break;}hash.update(&buffer[..n]);}
        Ok(format!("{:x}",hash.finalize()))
    }

    fn export_preflight(index:&Path,model:&Path,out:&Path)->Result<Value,String> {
        let source=bootstrap::current_source(Path::new("."))?;
        let prompt=prompt_bundle_path(model);
        let mut nodes=vec![json!({"id":"source","kind":"source","sha256":source.tree_sha256,"dependencies":[]})];
        for (id,kind,path) in [("raw-index","raw",index),("sam3","sam3",model),("prompts","sam3",prompt.as_path())] {
            nodes.push(json!({"id":id,"kind":kind,"sha256":hash_file(path)?,"dependencies":[]}));
        }
        nodes.push(json!({"id":"teacher","kind":"derived_data","planned":true,"sha256":null,
            "dependencies":["source","raw-index","sam3","prompts"]}));
        let graph=json!({"schema":bootstrap::SCHEMA,"source":source,"targets":["teacher"],"nodes":nodes});
        let parsed=bootstrap::parse(&serde_json::to_vec(&graph).map_err(|e|e.to_string())?).map_err(|e|format!("{e:?}"))?;
        let certificate=bootstrap::validate(&parsed,&source).map_err(|e|format!("{e:?}"))?;
        write_json(&out.join("bootstrap-graph.json"),&graph)?;
        write_json(&out.join("bootstrap-preflight.json"),&json!({"certificate":certificate,
            "raw_inventory":index,"verification":"every native RAW byte hash checked before teacher execution",
            "sam3_graph":model,"sam3_prompts":prompt,"scope":"fresh SAM-derived cache; no custom-model ancestry"}))?;
        Ok(graph)
    }
    fn export(index: &Path, out: &Path) -> Result<(), String> {
        runtime_output(out)?;
        std::fs::create_dir(out).map_err(|e| e.to_string())?;
        let mut images = new_writer(&out.join("images.u8"))?;
        let mut raw_images = new_writer(&out.join("raw-images.f32le"))?;
        let mut masks = new_writer(&out.join("masks.u8"))?;
        let mut records = new_writer(&out.join("records.jsonl"))?;
        let model = std::env::var_os("BUTTERCUP_SAM31_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(super::super::default_model_path);
        let graph=export_preflight(index,&model,out)?;
        let count =
            runtime::export_student_teacher(&model, source_frames(index)?, |row, sample| {
                images.write_all(&sample.image).map_err(|e| e.to_string())?;
                let raw_bytes:Vec<_>=sample.raw_image.iter().flat_map(|v|v.to_le_bytes()).collect();
                raw_images.write_all(&raw_bytes).map_err(|e|e.to_string())?;
                masks.write_all(&sample.masks).map_err(|e| e.to_string())?;
                serde_json::to_writer(
                    &mut records,
                    &json!({"input":row,"weights":sample.weights,"teacher":sample.report}),
                )
                .map_err(|e| e.to_string())?;
                records.write_all(b"\n").map_err(|e| e.to_string())?;
                Ok(())
            })?;
        images.flush().map_err(|e| e.to_string())?;
        raw_images.flush().map_err(|e| e.to_string())?;
        masks.flush().map_err(|e| e.to_string())?;
        records.flush().map_err(|e| e.to_string())?;
        let source_after=bootstrap::current_source(Path::new("."))?;
        write_json(
            &out.join("manifest.json"),
            &json!({"schema":"buttercup-eye-student-teacher-v2",
            "raw_input":raw::contract(),"raw_images":"raw-images.f32le",
            "source_before":graph["source"],"source_after":source_after,
            "payload_hashes":{"images.u8":hash_file(&out.join("images.u8"))?,
                "raw-images.f32le":hash_file(&out.join("raw-images.f32le"))?,"masks.u8":hash_file(&out.join("masks.u8"))?,
                "records.jsonl":hash_file(&out.join("records.jsonl"))?},
            "count":count,"input_shape":[3,FRAME_HEIGHT,FRAME_WIDTH],"prompts":SEMANTIC_PROMPT_LABELS,
            "preprocess":PreprocessRegime::configured_live()?.label(),"teacher_model":model,
            "index":index,"supervision":"SAM masks, never completed ellipses or 3D ground truth"}),
        )
    }
    struct Dataset {
        images: Tensor,
        masks: Tensor,
        weights: Tensor,
        rows: Vec<Value>,
        manifest: Value,
    }
    impl Dataset {
        fn load(path: &Path, raw_native:bool) -> Result<Self, String> {
            let manifest: Value = serde_json::from_reader(
                File::open(path.join("manifest.json")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            if manifest["schema"]=="buttercup-eye-student-teacher-v2" {
                let current=serde_json::to_value(bootstrap::current_source(Path::new("."))?).map_err(|e|e.to_string())?;
                if manifest["source_before"]!=manifest["source_after"] || manifest["source_after"]!=current {
                    return Err("teacher cache was not generated on the unchanged current source; regenerate export".into());
                }
                for name in ["images.u8","raw-images.f32le","masks.u8","records.jsonl"] {
                    if manifest["payload_hashes"][name]!=hash_file(&path.join(name))? {return Err(format!("teacher cache hash mismatch: {name}"));}
                }
            }
            let rows =
                BufReader::new(File::open(path.join("records.jsonl")).map_err(|e| e.to_string())?)
                    .lines()
                    .map(|line| {
                        serde_json::from_str(&line.map_err(|e| e.to_string())?)
                            .map_err(|e| e.to_string())
                    })
                    .collect::<Result<Vec<Value>, String>>()?;
            let n = rows.len();
            if n == 0 || n > 100_000 || manifest["count"].as_u64() != Some(n as u64) {
                return Err("invalid dataset count".into());
            }
            if raw_native && (manifest["raw_input"]!=raw::contract() || manifest["schema"]!="buttercup-eye-student-teacher-v2") {
                return Err("RAW dataset contract missing or incompatible; regenerate teacher export".into());
            }
            let images = std::fs::read(path.join(if raw_native {"raw-images.f32le"}else {"images.u8"})).map_err(|e| e.to_string())?;
            let masks = std::fs::read(path.join("masks.u8")).map_err(|e| e.to_string())?;
            let plane = FRAME_HEIGHT * FRAME_WIDTH;
            if images.len() != n * if raw_native {raw::VALUES*4}else {3*plane} || masks.len() != n * SEMANTIC_PROMPT_COUNT * plane {
                return Err("dataset payload length mismatch".into());
            }
            let mut weights = Vec::new();
            let mut splits = HashMap::new();
            let mut hashes = HashSet::new();
            for row in &rows {
                let split = row["input"]["student_split"]
                    .as_str()
                    .ok_or("missing split")?;
                if !["train", "validation", "test"].contains(&split) {
                    return Err("unknown split".into());
                }
                let lineage = row["input"]["clock_lineage"]
                    .as_str()
                    .ok_or("missing lineage")?;
                if splits
                    .insert(lineage, split)
                    .is_some_and(|old| old != split)
                {
                    return Err("source-session leakage across splits".into());
                }
                if !hashes.insert(
                    row["input"]["raw_sha256"]
                        .as_str()
                        .ok_or("missing RAW hash")?,
                ) {
                    return Err("duplicate RAW in dataset".into());
                }
                for weight in row["weights"]
                    .as_array()
                    .filter(|a| a.len() == SEMANTIC_PROMPT_COUNT)
                    .ok_or("invalid weights")?
                {
                    let w = weight
                        .as_f64()
                        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
                        .ok_or("invalid label weight")?;
                    weights.push(w as f32);
                }
            }
            Ok(Self {
                images: if raw_native {
                    let values:Vec<_>=images.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();
                    if values.iter().any(|v|!v.is_finite() || !(0.0..=1.0).contains(v)) {return Err("invalid RAW cache values".into());}
                    Tensor::from_slice(&values).reshape([n as i64,16,raw::HEIGHT as i64,raw::WIDTH as i64])
                } else {Tensor::from_slice(&images).reshape([
                    n as i64,
                    3,
                    FRAME_HEIGHT as i64,
                    FRAME_WIDTH as i64,
                ])},
                masks: Tensor::from_slice(&masks).reshape([
                    n as i64,
                    SEMANTIC_PROMPT_COUNT as i64,
                    FRAME_HEIGHT as i64,
                    FRAME_WIDTH as i64,
                ]),
                weights: Tensor::from_slice(&weights)
                    .reshape([n as i64, SEMANTIC_PROMPT_COUNT as i64]),
                rows,
                manifest,
            })
        }
        fn indices(&self, split: &str) -> Vec<i64> {
            self.rows
                .iter()
                .enumerate()
                .filter(|(_, r)| r["input"]["student_split"] == split)
                .map(|(i, _)| i as i64)
                .collect()
        }
        fn batch(&self, ids: &[i64], device: Device) -> (Tensor, Tensor, Tensor) {
            let ids = Tensor::from_slice(ids);
            (
                self.images
                    .index_select(0, &ids)
                    .to_device(device)
                    .to_kind(Kind::Float),
                self.masks
                    .index_select(0, &ids)
                    .to_device(device)
                    .to_kind(Kind::Float),
                self.weights.index_select(0, &ids).to_device(device),
            )
        }
    }
    fn loss(logits: &Tensor, truth: &Tensor, weights: &Tensor) -> Tensor {
        loss_with_valid_pixels(logits, truth, weights, None)
    }

    fn loss_with_valid_pixels(
        logits: &Tensor,
        truth: &Tensor,
        weights: &Tensor,
        valid: Option<&Tensor>,
    ) -> Tensor {
        let axes = [2i64, 3];
        let p = logits.sigmoid();
        let per_pixel = logits.clamp_min(0.0) - logits * truth + (-logits.abs()).exp().log1p();
        let (bce, intersection, total) = if let Some(valid) = valid {
            let sum = |value: Tensor| value.sum_dim_intlist(axes.as_slice(), false, Kind::Float);
            (
                sum(per_pixel * valid) / sum(valid.shallow_clone()).clamp_min(1.0),
                sum(&p * truth * valid),
                sum((&p + truth) * valid),
            )
        } else {
            (
                per_pixel.mean_dim(axes.as_slice(), false, Kind::Float),
                (&p * truth).sum_dim_intlist(axes.as_slice(), false, Kind::Float),
                (&p + truth).sum_dim_intlist(axes.as_slice(), false, Kind::Float),
            )
        };
        let dice: Tensor = 1.0 - (intersection * 2.0 + 1.0) / (total + 1.0);
        let priority =
            Tensor::from_slice(&[3.0f32, 0.5, 2.0, 0.5, 0.5, 0.5]).to_device(logits.device());
        let weights = weights * priority;
        ((bce + dice) * &weights).sum(Kind::Float) / weights.sum(Kind::Float).clamp_min(1.0)
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TrainingAugmentation {
        Legacy,
        ShadowCrop,
        MixedShadowCrop,
        /// ShadowCrop plus physically modeled reduced light on RAW input.
        ShadowCropNoise,
        /// ShadowCropNoise plus RAW-domain optics: per-block light colour,
        /// glare, defocus/motion blur and uncompensated dim light.
        ShadowCropOptics,
    }

    impl TrainingAugmentation {
        fn parse(label: &str) -> Result<Self, String> {
            match label {
                "legacy" => Ok(Self::Legacy),
                "shadow-crop-v1" => Ok(Self::ShadowCrop),
                "mixed-shadow-crop-v1" => Ok(Self::MixedShadowCrop),
                "shadow-crop-noise-v1" => Ok(Self::ShadowCropNoise),
                "shadow-crop-optics-v1" => Ok(Self::ShadowCropOptics),
                _ => Err(format!("unknown Student training augmentation: {label}")),
            }
        }
        fn label(self) -> &'static str {
            match self {
                Self::Legacy => "legacy",
                Self::ShadowCrop => "shadow-crop-v1",
                Self::MixedShadowCrop => "mixed-shadow-crop-v1",
                Self::ShadowCropNoise => "shadow-crop-noise-v1",
                Self::ShadowCropOptics => "shadow-crop-optics-v1",
            }
        }
    }

    fn training_luma_context(label: &str) -> Result<bool, String> {
        match label {
            "rgb" => Ok(false),
            "luma-context-v2" => Ok(true),
            _ => Err(format!("unknown Student training architecture: {label}")),
        }
    }

    fn soft_shadow_field(
        height: i64,
        width: i64,
        angle: &Tensor,
        offset: &Tensor,
        softness: &Tensor,
        depth: &Tensor,
    ) -> Tensor {
        let options = (Kind::Float, angle.device());
        let x = Tensor::linspace(-1.0, 1.0, width, options).reshape([1, 1, 1, width]);
        let y = Tensor::linspace(-1.0, 1.0, height, options).reshape([1, 1, height, 1]);
        let ramp =
            ((x * angle.cos() + y * angle.sin() - offset) / softness.clamp_min(0.02)).sigmoid();
        1.0 - depth.clamp(0.0, 0.75) * ramp
    }

    fn augment_spatial_shadow(image: &Tensor) -> Tensor {
        let shape = [image.size()[0], 1, 1, 1];
        let random = || Tensor::rand(shape, (Kind::Float, image.device()));
        let angle = random() * std::f64::consts::TAU;
        let offset = (random() - 0.5) * 1.2;
        let softness = random() * 0.25 + 0.05;
        // Keep half the examples as unshadowed controls. This changes only
        // appearance, never the segmentation truth or boundary visibility.
        let depth = (random() * 0.5 + 0.2) * random().ge(0.5).to_kind(Kind::Float);
        let field = soft_shadow_field(
            image.size()[2],
            image.size()[3],
            &angle,
            &offset,
            &softness,
            &depth,
        );
        // RGB-space diagnostic augmentation, not calibrated RAW sensor noise.
        (image * field + Tensor::randn_like(image) * random() * 1.5).clamp(0.0, 255.0)
    }

    fn resample_training_pair(
        image: &Tensor,
        truth: &Tensor,
        grid: &Tensor,
        exclude_unknown: bool,
    ) -> (Tensor, Tensor, Option<Tensor>) {
        let valid = exclude_unknown.then(|| {
            // Replicated context outside the native crop is not a known
            // negative label. Only fully observed pixels vote in the loss.
            Tensor::ones(
                [image.size()[0], 1, image.size()[2], image.size()[3]],
                (Kind::Float, image.device()),
            )
            .grid_sampler(grid, 0, 0, false)
            .ge(0.999)
            .to_kind(Kind::Float)
        });
        (
            image.grid_sampler(grid, 0, 1, false),
            truth.grid_sampler(grid, 0, 0, false),
            valid,
        )
    }

    fn score(net: &Net, data: &Dataset, ids: &[i64], device: Device) -> (f64, Value) {
        tch::no_grad(|| {
            let mut sums = [0.0; SEMANTIC_PROMPT_COUNT];
            let mut counts = [0u64; SEMANTIC_PROMPT_COUNT];
            for ids in ids.chunks(8) {
                let (image, truth, weights) = data.batch(ids, device);
                let pred = net.forward(&image).gt(0.0).to_kind(Kind::Float);
                let axes = [2i64, 3];
                let overlap = (&pred * &truth).sum_dim_intlist(axes.as_slice(), false, Kind::Float);
                let union = (&pred + &truth - &pred * &truth).sum_dim_intlist(
                    axes.as_slice(),
                    false,
                    Kind::Float,
                );
                let iou = (&overlap + 1.0) / (&union + 1.0);
                for b in 0..ids.len() {
                    for c in 0..SEMANTIC_PROMPT_COUNT {
                        if weights.double_value(&[b as i64, c as i64]) >= 0.5 {
                            sums[c] += iou.double_value(&[b as i64, c as i64]);
                            counts[c] += 1;
                        }
                    }
                }
            }
            let means = std::array::from_fn::<_, SEMANTIC_PROMPT_COUNT, _>(|c| {
                if counts[c] > 0 {
                    sums[c] / counts[c] as f64
                } else {
                    0.0
                }
            });
            (
                (means[0] + means[2]) * 0.5,
                json!({"mask_iou_against_teacher":means,"supported_frames_per_head":counts,
                "frames":ids.len(),"not_human_label_accuracy":true}),
            )
        })
    }
    /// Measured sensor shot-noise model for reduced-light augmentation, per
    /// quad-RGGB sensor phase block (R, G top, G bottom, B): variance in RAW10
    /// codes = K * (code - black). Loaded from a measurement file, never
    /// invented; recorded verbatim in the trained model's metadata.
    struct LightNoiseModel {
        k: Tensor,
        black: Tensor,
        light_range: [f64; 2],
        /// Optional lowest simulated signal (RAW codes above black): each
        /// frame's light factor reaches down to floor / its own mean signal,
        /// so bright frames can be dimmed to real low-light levels while dim
        /// frames are not pushed below anything recorded.
        signal_floor: Option<f64>,
        source: Value,
    }

    impl LightNoiseModel {
        fn load(device: Device) -> Result<Self, String> {
            let path = std::env::var("BUTTERCUP_EYE_STUDENT_TRAIN_NOISE_MODEL")
                .map_err(|_| "shadow-crop-noise-v1 requires BUTTERCUP_EYE_STUDENT_TRAIN_NOISE_MODEL")?;
            let source: Value = serde_json::from_str(&std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?)
                .map_err(|e| format!("{path}: {e}"))?;
            let block = |name: &str, key: &str| source[name][key].as_f64().filter(|v| v.is_finite())
                .ok_or_else(|| format!("{path}: missing {name}.{key}"));
            let names = ["R", "G_top", "G_bottom", "B"];
            let mut k = [0f32; 16];
            let mut black = [0f32; 16];
            // Channel c = 4*(sensor_y mod 4)+(sensor_x mod 4); 2x2 phase blocks.
            for c in 0..16 {
                let name = names[((c / 4) / 2) * 2 + (c % 4) / 2];
                k[c] = block(name, "K")? as f32;
                black[c] = block(name, "black")? as f32;
            }
            let range = std::env::var("BUTTERCUP_EYE_STUDENT_TRAIN_LIGHT_RANGE").unwrap_or_else(|_| "0.25,1".into());
            let parts: Vec<f64> = range.split(',').map(str::parse).collect::<Result<_, _>>()
                .map_err(|e| format!("light range {range}: {e}"))?;
            if parts.len() != 2 || !(0.0 < parts[0] && parts[0] <= parts[1] && parts[1] <= 1.0) {
                return Err(format!("light range must be LOW,HIGH within (0,1]: {range}"));
            }
            let signal_floor = match std::env::var("BUTTERCUP_EYE_STUDENT_TRAIN_SIGNAL_FLOOR") {
                Ok(v) => Some(v.parse::<f64>().ok().filter(|f| f.is_finite() && *f > 0.0)
                    .ok_or_else(|| format!("signal floor must be a positive number of codes: {v}"))?),
                Err(_) => None,
            };
            Ok(Self {
                k: Tensor::from_slice(&k).reshape([1, 16, 1, 1]).to_device(device),
                black: Tensor::from_slice(&black).reshape([1, 16, 1, 1]).to_device(device),
                light_range: [parts[0], parts[1]],
                signal_floor,
                source,
            })
        }

        /// Light reduced by k (log-uniform in light_range) with gain 1/k, as
        /// auto exposure compensates: the frame gains shot noise of std
        /// sqrt(K*signal*(1/k-1)) codes. Noise is drawn per photosite of each
        /// phase lattice and resampled like the RAW fields, so its variance
        /// and neighbor correlation match the registered input.
        /// Log-uniform light factor per frame in [low, high]; with a signal
        /// floor, low = max(range low, floor / frame mean signal), capped at high.
        fn light_factor(&self, signal: &Tensor) -> Tensor {
            let n = signal.size()[0];
            let device = signal.device();
            let [low, high] = self.light_range;
            let low = match self.signal_floor {
                Some(floor) => {
                    let mean = signal.mean_dim([1i64, 2, 3].as_slice(), true, Kind::Float).clamp_min(1e-3);
                    (mean.reciprocal() * floor).clamp(low, high)
                }
                None => Tensor::full([n, 1, 1, 1], low, (Kind::Float, device)),
            };
            let (log_low, log_high) = (low.log(), high.ln());
            (Tensor::rand([n, 1, 1, 1], (Kind::Float, device)) * (-&log_low + log_high) + &log_low).exp()
        }

        fn apply(&self, image: &Tensor) -> Tensor {
            let size = image.size();
            let (n, h, w) = (size[0], size[2], size[3]);
            let device = image.device();
            let codes = image * 1023.0;
            let signal = (&codes - &self.black).clamp_min(0.0);
            let k = self.light_factor(&signal);
            // 420x280 sources have 105x70 photosites per phase lattice.
            let lattice = Tensor::randn([n, 16, 70, 105], (Kind::Float, device))
                .upsample_bilinear2d([h, w], false, None, None);
            let std = (&self.k * signal * (k.reciprocal() - 1.0)).clamp_min(0.0).sqrt();
            ((codes + lattice * std) / 1023.0).clamp(0.0, 1.0)
        }

        /// As apply, but half of the frames keep the reduced light without
        /// gain: black + k*signal plus the shot noise of that dimmer signal.
        fn apply_mixed(&self, image: &Tensor) -> Tensor {
            let size = image.size();
            let (n, h, w) = (size[0], size[2], size[3]);
            let device = image.device();
            let compensated = Tensor::rand([n, 1, 1, 1], (Kind::Float, device)).ge(0.5).to_kind(Kind::Float);
            let codes = image * 1023.0;
            let signal = (&codes - &self.black).clamp_min(0.0);
            let k = self.light_factor(&signal);
            let lattice = Tensor::randn([n, 16, 70, 105], (Kind::Float, device))
                .upsample_bilinear2d([h, w], false, None, None);
            let gained = &codes + &lattice * (&self.k * &signal * (k.reciprocal() - 1.0)).clamp_min(0.0).sqrt();
            let dim = &self.black + &k * &signal
                + &lattice * (&self.k * &signal * (&k - &k * &k)).clamp_min(0.0).sqrt();
            let uncompensated: Tensor = compensated.ones_like() - &compensated;
            ((&compensated * gained + uncompensated * dim) / 1023.0).clamp(0.0, 1.0)
        }
    }

    /// Input pixels per photosite of one phase lattice (4 native px resampled
    /// from a 420 px source to the 192 px RAW field width).
    const INPUT_PX_PER_PHOTOSITE: f64 = 4.0 * 192.0 / 420.0;

    /// Independent light-colour gain per quad phase block, log-uniform in
    /// [1/2, 2]: the per-recording channel ratios measured across the corpus
    /// span R/G 0.9-1.8 and B/G 0.35-1.6.
    fn optics_block_gains(n: i64, device: Device) -> Tensor {
        let blocks: Vec<i64> = (0..16).map(|c| ((c / 4) / 2) * 2 + (c % 4) / 2).collect();
        (Tensor::rand([n, 4, 1, 1], (Kind::Float, device)) * 2.0 - 1.0)
            .multiply_scalar(std::f64::consts::LN_2).exp()
            .index_select(1, &Tensor::from_slice(&blocks).to_device(device))
    }

    /// Glare: 0 (half of frames) or 1-3 soft Gaussian blobs of white light,
    /// sigma 1.5..15 native px (glint to lens reflection), peak 0.3..1.5 of
    /// full scale before clipping. Half are centred on the outer-iris
    /// boundary taken from the (already warped) label.
    fn optics_glare(image: &Tensor, truth: &Tensor) -> Tensor {
        let size = image.size();
        let (n, h, w) = (size[0], size[2], size[3]);
        let device = image.device();
        let options = (Kind::Float, device);
        let iris = truth.narrow(1, 0, 1);
        let boundary = (iris.avg_pool2d([3, 3], [1, 1], [1, 1], false, true, None::<i64>) - &iris).abs().gt(0.05).to_kind(Kind::Float);
        let th = truth.size()[2];
        let tw = truth.size()[3];
        let picks = (boundary.reshape([n, th * tw]) + 1e-6).multinomial(3, true);
        let ys = Tensor::linspace(0.0, 1.0, h, options).reshape([1, 1, h, 1]);
        let xs = Tensor::linspace(0.0, 1.0, w, options).reshape([1, 1, 1, w]);
        let mut light = Tensor::zeros([n, 1, h, w], options);
        for blob in 0..3 {
            let on = Tensor::rand([n, 1, 1, 1], options).lt(if blob == 0 { 0.5 } else { 0.25 }).to_kind(Kind::Float);
            let on_boundary = Tensor::rand([n, 1, 1, 1], options).lt(0.5).to_kind(Kind::Float);
            let index = picks.select(1, blob).to_kind(Kind::Float);
            let by = (&index / tw as f64).floor() / (th - 1).max(1) as f64;
            let bx = (index.fmod(tw as f64)) / (tw - 1).max(1) as f64;
            let ry = Tensor::rand([n], options);
            let rx = Tensor::rand([n], options);
            let pick: Tensor = on_boundary.reshape([n]);
            let free: Tensor = pick.ones_like() - &pick;
            let cy: Tensor = (&pick * &by + &free * &ry).reshape([n, 1, 1, 1]);
            let cx: Tensor = (&pick * &bx + &free * &rx).reshape([n, 1, 1, 1]);
            let sigma_native = (Tensor::rand([n, 1, 1, 1], options) * (10.0f64).ln() + (1.5f64).ln()).exp();
            let sy = &sigma_native / 280.0;
            let sx = &sigma_native / 420.0;
            let peak = Tensor::rand([n, 1, 1, 1], options) * 1.2 + 0.3;
            let d2 = ((&ys - &cy) / &sy).square() + ((&xs - &cx) / &sx).square();
            light = light + on * peak * (d2 * -0.5).exp();
        }
        image + light
    }

    /// Optical blur shared by a batch: defocus (Gaussian, sigma up to 1.5
    /// photosites) for half of batches, motion smear (line up to 2
    /// photosites, random direction) for a quarter, none otherwise. Applied
    /// per RAW field, like light spreading before the photosite lattices.
    fn optics_blur(image: &Tensor, random: &mut impl FnMut() -> f64) -> Tensor {
        let choice = random();
        let radius = 7i64;
        let size = (2 * radius + 1) as usize;
        let mut kernel = vec![0f32; size * size];
        if choice < 0.5 {
            let sigma = (random() * 1.5 * INPUT_PX_PER_PHOTOSITE).max(0.05);
            for y in 0..size { for x in 0..size {
                let (dy, dx) = (y as f64 - radius as f64, x as f64 - radius as f64);
                kernel[y * size + x] = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp() as f32;
            }}
        } else if choice < 0.75 {
            let length = random() * 2.0 * INPUT_PX_PER_PHOTOSITE;
            let angle = random() * std::f64::consts::PI;
            for step in 0..=32 {
                let t = (step as f64 / 32.0 - 0.5) * length;
                let (x, y) = ((radius as f64 + t * angle.cos()).round() as usize, (radius as f64 + t * angle.sin()).round() as usize);
                kernel[y * size + x] += 1.0;
            }
        } else {
            return image.shallow_clone();
        }
        let total: f32 = kernel.iter().sum();
        let kernel = Tensor::from_slice(&kernel.iter().map(|v| v / total).collect::<Vec<_>>())
            .reshape([1, 1, size as i64, size as i64]).repeat([16, 1, 1, 1]).to_device(image.device());
        image.pad([radius, radius, radius, radius], "replicate", None)
            .conv2d(&kernel, None::<Tensor>, [1, 1], [0, 0], [1, 1], 16)
    }

    fn train(data_path: &Path, model_path: &Path, epochs: usize) -> Result<(), String> {
        runtime_output(model_path)?;
        runtime_output(&metadata_path(model_path))?;
        if epochs == 0 || epochs > 2000 {
            return Err("epochs must be 1..2000".into());
        }
        let augmentation = TrainingAugmentation::parse(
            &std::env::var("BUTTERCUP_EYE_STUDENT_TRAIN_AUGMENTATION")
                .unwrap_or_else(|_| "legacy".into()),
        )?;
        let architecture=std::env::var("BUTTERCUP_EYE_STUDENT_TRAIN_ARCHITECTURE").unwrap_or_else(|_|"rgb".into());
        let raw_native=architecture=="raw16-v1";
        let luma_context=if raw_native {false} else {training_luma_context(&architecture)?};
        if matches!(augmentation, TrainingAugmentation::ShadowCropNoise | TrainingAugmentation::ShadowCropOptics) && !raw_native {
            return Err("shadow-crop-noise-v1 models RAW sensor noise; use raw16-v1".into());
        }
        let initial_model =
            std::env::var_os("BUTTERCUP_EYE_STUDENT_TRAIN_INITIAL_MODEL").map(PathBuf::from);
        runtime::student_cuda_init()?;
        tch::set_num_threads(2);
        tch::manual_seed(17091);
        let device = Device::Cuda(0);
        let light_noise = matches!(augmentation, TrainingAugmentation::ShadowCropNoise | TrainingAugmentation::ShadowCropOptics)
            .then(|| LightNoiseModel::load(device)).transpose()?;
        let data = Dataset::load(data_path,raw_native)?;
        if data.manifest["schema"]!="buttercup-eye-student-teacher-v2" {
            return Err("training requires a fresh hash-verified current-checkout teacher export".into());
        }
        if initial_model.is_some() {
            return Err("this cold RAW/RGB experiment must start from seeded initialization, not an unproven custom ancestor".into());
        }
        let mut graph:Value=serde_json::from_reader(File::open(data_path.join("bootstrap-graph.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        for node in graph["nodes"].as_array_mut().ok_or("missing graph nodes")? {
            if node["id"]=="teacher" {node["planned"]=json!(false);node["sha256"]=json!(hash_file(&data_path.join("manifest.json"))?);}
        }
        graph["nodes"].as_array_mut().unwrap().push(json!({"id":"model","kind":"custom_model","planned":true,
            "sha256":null,"dependencies":["teacher","source"]}));
        graph["targets"]=json!(["model"]);
        let parsed=bootstrap::parse(&serde_json::to_vec(&graph).map_err(|e|e.to_string())?).map_err(|e|format!("{e:?}"))?;
        let source=bootstrap::current_source(Path::new("."))?;
        let certificate=bootstrap::validate(&parsed,&source).map_err(|e|format!("{e:?}"))?;
        write_json(&model_path.with_extension("bootstrap.json"),&json!({"graph":graph,"certificate":certificate}))?;
        let mut train = data.indices("train");
        let validation = data.indices("validation");
        let test = data.indices("test");
        if train.len() < 16 || validation.len() < 4 || test.len() < 4 {
            return Err("need train/validation/test sessions with at least 16/4/4 frames".into());
        }
        let mut store = nn::VarStore::new(device);
        let net = Net::with_input(&store.root(), luma_context,raw_native);
        if let Some(path) = initial_model.as_ref() {
            let initial = validate_model(path)?;
            let architecture = if raw_native { RAW_ARCHITECTURE } else if luma_context {
                LUMA_CONTEXT_ARCHITECTURE
            } else {
                ARCHITECTURE
            };
            if initial["architecture"] != architecture
                || initial["preprocess"] != if raw_native {json!(raw::CONTRACT)}else {data.manifest["preprocess"].clone()}
            {
                return Err("Student warm start architecture/preprocessing mismatch".into());
            }
            store
                .load(path)
                .map_err(|e| format!("Student warm start: {e}"))?;
        }
        let parameters: usize = store.trainable_variables().iter().map(Tensor::numel).sum();
        let mut optimizer = nn::AdamW::default()
            .build(&store, 1e-3)
            .map_err(|e| e.to_string())?;
        let mut best = -1.0;
        let mut best_epoch = 0;
        let mut history = Vec::new();
        let started = Instant::now();
        let mut random = 0x67c9f48bu64;
        let mut random_u = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        for epoch in 1..=epochs {
            for i in (1..train.len()).rev() {
                train.swap(i, (random_u() as usize) % (i + 1));
            }
            let (minimum_lr, lr_span) = if initial_model.is_some() {
                (0.00002, 0.00018)
            } else {
                (0.00005, 0.00095)
            };
            optimizer.set_lr(
                minimum_lr
                    + lr_span
                        * 0.5
                        * (1.0 + (std::f64::consts::PI * epoch as f64 / epochs as f64).cos()),
            );
            let mut loss_sum = 0.0;
            let mut batches = 0;
            for ids in train.chunks(12) {
                let (mut image, mut truth, weights) = data.batch(ids, device);
                // RAW channels are registered continuous fields, not a Bayer
                // mosaic. Their channel identity survives spatial transforms.
                if random_u() % 2 == 0 {
                    image = image.flip([3]);
                    truth = truth.flip([3]);
                }
                // Same affine for image and labels; never a completed-ellipse target.
                let n = ids.len() as i64;
                let angles = (Tensor::rand([n], (Kind::Float, device)) - 0.5) * 0.25;
                let spatial = matches!(augmentation, TrainingAugmentation::ShadowCrop | TrainingAugmentation::ShadowCropNoise | TrainingAugmentation::ShadowCropOptics)
                    || (augmentation == TrainingAugmentation::MixedShadowCrop
                        && random_u() % 4 == 0);
                let scales = Tensor::rand([n], (Kind::Float, device))
                    * if spatial { 0.4 } else { 0.2 }
                    + if spatial { 0.8 } else { 0.9 };
                let c = angles.cos() * &scales;
                let s = angles.sin() * scales;
                // +/- half the normalized grid means +/- one quarter ROI:
                // covers the observed 70px vertical reframe in a 280px ROI.
                let shift = (Tensor::rand([n, 2], (Kind::Float, device)) - 0.5)
                    * if spatial { 1.0 } else { 0.16 };
                let theta = Tensor::stack(
                    &[
                        Tensor::stack(&[c.shallow_clone(), -&s, shift.select(1, 0)], 1),
                        Tensor::stack(&[s, c, shift.select(1, 1)], 1),
                    ],
                    1,
                );
                let grid = Tensor::affine_grid_generator(&theta, truth.size(), false);
                let valid;
                if raw_native {
                    let input_grid=Tensor::affine_grid_generator(&theta,image.size(),false);
                    image=image.grid_sampler(&input_grid,0,1,false);
                    let (_, warped_truth, mask)=resample_training_pair(&truth,&truth,&grid,true);
                    truth=warped_truth; valid=mask;
                }else{(image, truth, valid) = resample_training_pair(&image, &truth, &grid, spatial);}
                let mut gain = Tensor::rand([n,3,1,1],(Kind::Float,device))*0.35+0.80;
                if augmentation == TrainingAugmentation::ShadowCropOptics {
                    gain = optics_block_gains(n, device);
                } else if raw_native {
                    let colors:Vec<i64>=(0..16).map(|c|match (c/4<2,c%4<2) {(true,true)=>0,(false,false)=>2,_=>1}).collect();
                    gain=gain.index_select(1,&Tensor::from_slice(&colors).to_device(device));
                }
                image = (image * gain
                    + (Tensor::rand([n, 1, 1, 1], (Kind::Float, device)) - 0.5) * if raw_native {12./1023.}else{12.0})
                    .clamp(0.0, if raw_native {1.0}else{255.0});
                if spatial {
                    image = if raw_native {augment_spatial_shadow(&(image*255.))/255.}else{augment_spatial_shadow(&image)};
                }
                // Light before the lens (glare), then optics (blur), then the
                // sensor (photon noise, clipping): the physical order.
                let mut saturated = None;
                if augmentation == TrainingAugmentation::ShadowCropOptics {
                    image = optics_glare(&image, &truth);
                    let mut random = || random_u() as f64 / u64::MAX as f64;
                    image = optics_blur(&image, &mut random);
                    // Clipped photosites carry no boundary information.
                    saturated = Some(image.ge(1.0).any_dim(1, true).to_kind(Kind::Float));
                    image = image.clamp(0.0, 1.0);
                }
                // Photon counts follow the final illumination: noise last.
                if let Some(model) = &light_noise {
                    image = if augmentation == TrainingAugmentation::ShadowCropOptics {model.apply_mixed(&image)} else {model.apply(&image)};
                }
                let valid = match (valid, saturated) {
                    (Some(valid), Some(sat)) => {
                        let sat = sat.upsample_nearest2d([valid.size()[2], valid.size()[3]], None, None);
                        let keep: Tensor = sat.ones_like() - &sat;
                        Some(valid * keep)
                    }
                    (valid, _) => valid,
                };
                let value =
                    loss_with_valid_pixels(&net.forward(&image), &truth, &weights, valid.as_ref());
                let scalar = value.double_value(&[]);
                if !scalar.is_finite() {
                    return Err("nonfinite student loss".into());
                }
                optimizer.backward_step_clip_norm(&value, 2.0);
                loss_sum += scalar;
                batches += 1;
            }
            if epoch == 1 || epoch % 5 == 0 || epoch == epochs {
                let (metric, report) = score(&net, &data, &validation, device);
                if metric > best {
                    best = metric;
                    best_epoch = epoch;
                    store.save(model_path).map_err(|e| e.to_string())?;
                }
                let row = json!({"epoch":epoch,"training_loss":loss_sum/batches as f64,"validation":report,
                    "best_epoch":best_epoch,"elapsed_seconds":started.elapsed().as_secs_f64()});
                eprintln!("STUDENT_TRAIN {row}");
                history.push(row);
            }
        }
        drop(optimizer);
        // Loading updates the existing parameter tensors referenced by net.
        store.load(model_path).map_err(|e| e.to_string())?;
        let (_, test_report) = score(&net, &data, &test, device);
        let (_, validation_report) = score(&net, &data, &validation, device);
        if bootstrap::current_source(Path::new("."))? != source {
            return Err("source changed during training; weights are uncertified experimental output, not a completed cold bootstrap".into());
        }
        write_json(
            &metadata_path(model_path),
            &json!({"architecture":if raw_native {RAW_ARCHITECTURE}else if luma_context {LUMA_CONTEXT_ARCHITECTURE} else {ARCHITECTURE},
            "display_name":if raw_native {RAW_DISPLAY_NAME}else{"Eye Student RGB"},
            "raw_input":if raw_native {raw::contract()}else{Value::Null},
            "training_source":source,"weights_sha256":hash_file(model_path)?,
            "derived_context":if luma_context {luma_context_contract()} else {Value::Null},
            "training_augmentation":augmentation.label(),"training_seed":17091,
            "light_noise_model":light_noise.as_ref().map(|m|json!({"light_range":m.light_range,"signal_floor_codes":m.signal_floor,"model":m.source,
                "uncompensated_fraction":if augmentation==TrainingAugmentation::ShadowCropOptics {0.5} else {0.0}})),
            "optics_augmentation":(augmentation==TrainingAugmentation::ShadowCropOptics).then(||json!({
                "block_gain_range":[0.5,2.0],"glare_sigma_native_px":[1.5,15.0],"glare_peak_full_scale":[0.3,1.5],
                "glare_on_iris_boundary_fraction":0.5,"defocus_sigma_photosites_max":1.5,"motion_photosites_max":2.0,
                "saturated_pixels":"excluded from loss"})),
            "initial_model":initial_model,
            "learning_rate_range":if initial_model.is_some() {json!([0.00002,0.0002])} else {json!([0.00005,0.001])},
            "input_shape":if raw_native {json!([16,raw::HEIGHT,raw::WIDTH])}else{json!([3,FRAME_HEIGHT,FRAME_WIDTH])},
            "prompts":SEMANTIC_PROMPT_LABELS,"preprocess":if raw_native {json!(raw::CONTRACT)}else{data.manifest["preprocess"].clone()},"parameters":parameters,
            "training_dataset":data_path,"training_frames":train.len(),"validation_frames":validation.len(),"test_frames":test.len(),
            "best_epoch":best_epoch,"epochs":epochs,"validation":validation_report,"test":test_report,
            "elapsed_seconds":started.elapsed().as_secs_f64(),"history":history,"experimental":true,
            "contract":"fixed semantic mask student, not a learned 3D pose or calibrated confidence; shared RAW/conic/gaze gates remain authoritative"}),
        )
    }
    fn evaluate(data_path: &Path, model_path: &Path, output: &Path) -> Result<(), String> {
        runtime_output(output)?;
        runtime::student_cuda_init()?;
        tch::set_num_threads(2);
        let model = Model::load(model_path)?;
        let data = Dataset::load(data_path,model.raw_native)?;
        let mut rows = new_writer(output)?;
        for (i, row) in data.rows.iter().enumerate() {
            let raw = source_frames_from_row(row["input"].clone())?;
            let started = Instant::now();
            let logits=tch::no_grad(||model.net.forward(&data.images.get(i as i64).unsqueeze(0).to_device(model.store.device())));
            let sample = runtime::student_evaluation(&raw, &logits)?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            serde_json::to_writer(&mut rows,&json!({"input":row["input"],"teacher":row["teacher"],"student":sample,"student_ms":elapsed})).map_err(|e|e.to_string())?;
            rows.write_all(b"\n").map_err(|e| e.to_string())?;
        }
        rows.flush().map_err(|e| e.to_string())
    }
    fn source_frames_from_row(row: Value) -> Result<Arc<RawFrame>, String> {
        let f = &row["frame"];
        let width = integer(f, "width")? as usize;
        let height = integer(f, "height")? as usize;
        let mut file = File::open(row["raw_file"].as_str().ok_or("missing source")?)
            .map_err(|e| e.to_string())?;
        file.seek(SeekFrom::Start(integer(&row, "raw_offset")?))
            .map_err(|e| e.to_string())?;
        let length = integer(&row, "raw_length")?;
        if length > 64 * 1024 * 1024 { return Err("oversized RAW frame".into()); }
        let mut bytes = vec![0; length as usize];
        file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        verify_source(&row,&bytes)?;
        Ok(Arc::new(RawFrame {
            eye_index: integer(f, "eye_id")?.checked_sub(1).filter(|v| *v < 2).ok_or("invalid eye")? as usize,
            sequence: integer(f, "sequence")?,
            timestamp_ns: integer(f, "timestamp_ns")?,
            sensor_x: integer(f, "sensor_x")? as u32,
            sensor_y: integer(f, "sensor_y")? as u32,
            width,
            height,
            registration_anchor: None,
            pupil_component_seed: None,
            pixels: Arc::new(crate::raw10::try_unpack_raw10(
                &bytes,
                width,
                height,
                integer(f, "stride")? as usize,
            )?),
        }))
    }

    fn ellipse_json(ellipse: Ellipse) -> Value {
        json!({"center":ellipse.center,"major_radius":ellipse.major_radius,
            "minor_radius":ellipse.minor_radius,"angle":ellipse.angle})
    }

    /// Native observed contour evidence for the shared 3D replay harness.
    /// Do not synthesize rim samples from the fitted ellipse: missing arcs and
    /// flat-tire gaps must remain missing evidence for the conic solver.
    fn conic_replay_evidence(proposal: &ProposalMasks, admitted: bool) -> Value {
        let fit = proposal.outer_fit.as_ref();
        let selected_mask = proposal.semantic.as_ref().and_then(|semantic| {
            semantic.selected_query.and_then(|selected| {
                semantic
                    .masks
                    .iter()
                    .find(|mask| mask.query == selected)
                    .map(|mask| (semantic, mask))
            })
        });
        let score = selected_mask.map(|(_, mask)| mask.score);
        // Keep the original ordered mask boundary even when a complete fit
        // was rejected. This is diagnostic segmentation evidence, not an
        // admitted limbus and not a rim synthesized from the fitted ellipse.
        let outline = selected_mask
            .filter(|(semantic, _)| semantic.prompt_index == OUTER_IRIS_PROMPT)
            .map(|(semantic, mask)| native_outline_points(
                &mask.pixels, semantic.width, semantic.height,
                proposal.source_width, proposal.source_height,
            ))
            .unwrap_or_default();
        let mut record = json!({"selected_query":fit.map(|_|0),
            "candidates":[{"query":0,"semantic_score":score,
                "baseline_raw_admitted":admitted,
                "baseline_ellipse":fit.map(|f|ellipse_json(f.ellipse)),
                "baseline_retained":fit.map(|f|f.retained_points.as_ref()),
                "baseline_retained_segments":fit.map(|f|f.conic_segments.as_ref()),
                "baseline_censored":fit.map(|f|f.flat_tire_points.as_ref()),
                "outline":outline,
                "scope":"selected live proposal; original ordered mask outline is unfiltered diagnostic evidence; retained arcs preserve fit censorship; no synthetic completed rim"}],
            "pupil_void":proposal.inner_pupil_fit.map(|p|json!({"ellipse":ellipse_json(p.ellipse)}))});
        proposal.export_boundary_logits(&mut record);
        record
    }

    fn source_groups(rows: Vec<Value>, paired: bool) -> Result<Vec<Vec<Value>>, String> {
        if rows.len() > 100_000 {
            return Err("replay index exceeds 100000 exposures".into());
        }
        if !paired {
            return Ok(rows.into_iter().map(|row| vec![row]).collect());
        }
        let mut keyed = rows
            .into_iter()
            .map(|row| {
                let lineage = row["clock_lineage"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or("missing source clock")?
                    .to_owned();
                let timestamp = integer(&row["frame"], "timestamp_ns")?;
                let eye = integer(&row["frame"], "eye_id")?;
                if !(1..=2).contains(&eye) {
                    return Err("invalid source eye".into());
                }
                Ok((lineage, timestamp, eye, row))
            })
            .collect::<Result<Vec<_>, String>>()?;
        keyed.sort_by(|a, b| (&a.0, a.1, a.2).cmp(&(&b.0, b.1, b.2)));
        let mut groups: Vec<Vec<Value>> = Vec::new();
        let mut previous = None;
        for (clock, time, eye, row) in keyed {
            if previous
                .as_ref()
                .is_some_and(|(c, t, _)| *c == clock && *t == time)
            {
                if previous.as_ref().unwrap().2 == eye || groups.last().unwrap().len() >= 2 {
                    return Err("duplicate eye/source in replay".into());
                }
                groups.last_mut().unwrap().push(row);
            } else {
                groups.push(vec![row]);
            }
            previous = Some((clock, time, eye));
        }
        Ok(groups)
    }

    fn replay(index: &Path, backend: &str, output: &Path, paired: bool) -> Result<(), String> {
        runtime_output(output)?;
        // Retain only the offline metadata index; decode at most two current
        // RAW buffers. Equal timestamps from unrelated clocks never pair.
        let rows = BufReader::new(File::open(index).map_err(|e| e.to_string())?)
            .lines()
            .map(|line| {
                serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<Value>, String>>()?;
        let groups = source_groups(rows, paired)?;
        let client = match backend {
            "student" => Client::start_student(default_model_path())?,
            "sam" => Client::start(super::super::default_model_path())?,
            _ => return Err("replay backend must be student or sam".into()),
        };
        let mut writer = new_writer(output)?;
        let mut lineage = Value::Null;
        let mut epoch = 0;
        for rows in groups {
            let frames = rows
                .iter()
                .map(|row| source_frames_from_row(row.clone()))
                .collect::<Result<Vec<_>, _>>()?;
            if lineage != rows[0]["clock_lineage"] {
                lineage = rows[0]["clock_lineage"].clone();
                epoch += 1;
            }
            let before = client.status().completed_batches;
            let started = Instant::now();
            let mut proposals: [Option<Arc<ProposalMasks>>; 2] = [None, None];
            let mut results: [Option<OuterResult>; 2] = [None, None];
            let outcome = if frames.len() == 2 {
                client.submit_source_group(
                    [Arc::clone(&frames[0]), Arc::clone(&frames[1])],
                    Target::OuterLimbusAndInnerPupilVoid,
                    0,
                    0,
                    [epoch; 2],
                    [None, None],
                )
            } else {
                client.submit_history(
                    &VecDeque::from([Arc::clone(&frames[0])]),
                    Target::OuterLimbusAndInnerPupilVoid,
                    0,
                    0,
                    epoch,
                )
            };
            if outcome != SubmitOutcome::Accepted {
                return Err(format!("replay submission: {outcome:?}"));
            }
            loop {
                let status = client.status();
                for value in client.drain_results() {
                    let eye = value.eye_index;
                    results[eye] = Some(value);
                }
                for value in client.drain_proposal_masks() {
                    let eye = value.eye_index;
                    proposals[eye] = Some(value);
                }
                if status.completed_batches >= before + frames.len() as u64 {
                    break;
                }
                if status.state == "error" {
                    return Err(status.detail);
                }
                if started.elapsed() > std::time::Duration::from_secs(60) {
                    return Err("replay worker timed out".into());
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            for (row, raw) in rows.iter().zip(&frames) {
                let status = client.status_for_eye(raw.eye_index);
                let result = &results[raw.eye_index];
                let p = proposals[raw.eye_index]
                    .as_ref()
                    .ok_or("worker completed without current source proposal")?;
                if (
                    p.eye_index,
                    p.tracking_epoch,
                    p.source_sequence,
                    p.source_timestamp_ns,
                ) != (raw.eye_index, epoch, raw.sequence, raw.timestamp_ns)
                {
                    return Err("replay source identity mismatch".into());
                }
                if p.source_group_roi_count != frames.len() as u8
                    || result.as_ref().is_some_and(|r| {
                        (r.tracking_epoch, r.source_sequence, r.source_timestamp_ns)
                            != (epoch, raw.sequence, raw.timestamp_ns)
                    })
                {
                    return Err("replay result or atomic source-group provenance mismatch".into());
                }
                let mut record = conic_replay_evidence(p, result.is_some());
                let measurements = json!({"input":row,"backend":backend,"accepted":result.is_some(),
                "student_stages_ms":status.student_stages_ms,
                "stage_order":["prepare_cpu","h2d_synchronized","forward_synchronized","mask_materialization","remaining_downstream"],
                "elapsed_ms":status.last_elapsed_ms,"encode_ms":status.last_encode_ms,"track_ms":status.last_track_ms,
                "queue_ms":status.last_queue_ms,"state":status.state,"detail":status.detail,
                "source_identity_verified":true,"source_group_roi_count":p.source_group_roi_count,
                "outer_ellipse":p.outer_fit.as_ref().map(|r|json!({"center":r.ellipse.center,
                    "major_radius":r.ellipse.major_radius,"minor_radius":r.ellipse.minor_radius,"angle":r.ellipse.angle})),
                "pupil_present":p.inner_pupil_fit.is_some(),"scope":"completion-paced replay, not offered-load camera/display latency"});
                record
                    .as_object_mut()
                    .unwrap()
                    .extend(measurements.as_object().unwrap().clone());
                serde_json::to_writer(&mut writer, &record).map_err(|e| e.to_string())?;
                writer.write_all(b"\n").map_err(|e| e.to_string())?;
            }
        }
        writer.flush().map_err(|e| e.to_string())
    }
    pub fn run_cli(mut args: impl Iterator<Item = String>) -> Result<(), String> {
        let command=args.next().ok_or("expected export INDEX DIRECTORY | train DATA MODEL EPOCHS | evaluate DATA MODEL OUTPUT | replay[-paired] INDEX sam|student OUTPUT")?;
        let a = PathBuf::from(args.next().ok_or("missing input")?);
        let b = PathBuf::from(args.next().ok_or("missing output/model")?);
        let c = args.next();
        if args.next().is_some() {
            return Err("unexpected extra argument".into());
        }
        match command.as_str() {
            "export" if c.is_none() => export(&a, &b),
            "train" => train(
                &a,
                &b,
                c.ok_or("missing epochs")?
                    .parse()
                    .map_err(|_| "invalid epochs")?,
            ),
            "evaluate" => evaluate(&a, &b, Path::new(&c.ok_or("missing report")?)),
            "replay" | "replay-paired" => replay(
                &a,
                b.to_str().ok_or("invalid backend")?,
                Path::new(&c.ok_or("missing replay output")?),
                command == "replay-paired",
            ),
            _ => Err("unknown student command".into()),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn raw_student_manifest_is_explicit_and_cannot_be_loaded_as_rgb() {
            let mut meta=json!({"architecture":RAW_ARCHITECTURE,"raw_input":raw::contract()});
            assert!(known_architecture_manifest(&meta));
            meta["raw_input"]["cfa"]=json!("RGGB");
            assert!(!known_architecture_manifest(&meta));
            meta["raw_input"]=Value::Null;
            assert!(!known_architecture_manifest(&meta));
        }

        #[test]
        fn raw_student_keeps_the_six_full_crop_masks_and_small_parameter_budget() {
            tch::set_num_threads(2);
            let store=nn::VarStore::new(Device::Cpu);
            let net=Net::with_input(&store.root(),false,true);
            let parameters:usize=store.trainable_variables().iter().map(Tensor::numel).sum();
            assert_eq!(parameters,214566);
            let output=tch::no_grad(||net.forward(&Tensor::ones([1,16,128,192],(Kind::Float,Device::Cpu))));
            assert_eq!(output.size(),[1,6,256,384]);
            assert_eq!(output.isfinite().all().int64_value(&[]),1);
        }

        #[test]
        fn raw_student_source_hash_and_readout_must_match_before_any_teacher_use() {
            let bytes=[1u8,2,3,4,5];
            let mut row=json!({"raw_sha256":format!("{:x}",Sha256::digest(bytes)),"frame":{"pixel_format":"RAW10_LE40_1X1"}});
            verify_source(&row,&bytes).unwrap();
            assert!(verify_source(&row,&[1,2,3,4,6]).is_err());
            row["frame"]["pixel_format"]=json!("GRAY10");
            assert!(verify_source(&row,&bytes).is_err());
        }

        #[test]
        fn versioned_context_manifest_keeps_legacy_models_and_rejects_unknown_context() {
            assert!(known_architecture_manifest(
                &json!({"architecture":ARCHITECTURE})
            ));
            let mut meta = json!({"architecture":LUMA_CONTEXT_ARCHITECTURE,
                                  "derived_context":luma_context_contract()});
            assert!(known_architecture_manifest(&meta));
            meta["derived_context"]["shape"] = json!([2, 24, 16]);
            assert!(!known_architecture_manifest(&meta));
            assert!(!known_architecture_manifest(
                &json!({"architecture":LUMA_CONTEXT_ARCHITECTURE})
            ));
            assert!(!known_architecture_manifest(
                &json!({"architecture":"guess"})
            ));
        }

        #[test]
        fn training_variants_are_explicit_and_reject_unknown_names() {
            assert_eq!(
                TrainingAugmentation::parse("legacy").unwrap(),
                TrainingAugmentation::Legacy
            );
            assert_eq!(
                TrainingAugmentation::parse("shadow-crop-v1").unwrap(),
                TrainingAugmentation::ShadowCrop
            );
            assert_eq!(
                TrainingAugmentation::parse("mixed-shadow-crop-v1").unwrap(),
                TrainingAugmentation::MixedShadowCrop
            );
            assert_eq!(
                TrainingAugmentation::parse("shadow-crop-noise-v1").unwrap(),
                TrainingAugmentation::ShadowCropNoise
            );
            assert_eq!(TrainingAugmentation::ShadowCropNoise.label(), "shadow-crop-noise-v1");
            assert!(TrainingAugmentation::parse("shdaow").is_err());
            assert!(!training_luma_context("rgb").unwrap());
            assert!(training_luma_context("luma-context-v2").unwrap());
            assert!(training_luma_context("auto").is_err());
        }

        #[test]
        fn signal_floor_dims_bright_frames_further_than_dim_ones() {
            let model = LightNoiseModel {
                k: Tensor::from_slice(&[2.0f32; 16]).reshape([1, 16, 1, 1]),
                black: Tensor::from_slice(&[50.0f32; 16]).reshape([1, 16, 1, 1]),
                light_range: [0.01, 1.0],
                signal_floor: Some(20.0),
                source: Value::Null,
            };
            let bright = Tensor::full([256, 16, 4, 4], 350.0, (Kind::Float, Device::Cpu));
            let dim = Tensor::full([256, 16, 4, 4], 80.0, (Kind::Float, Device::Cpu));
            let kb = model.light_factor(&bright);
            let kd = model.light_factor(&dim);
            // floor/signal: 20/350 for bright, 20/80 for dim; never above 1.
            assert!(kb.min().double_value(&[]) >= 20.0 / 350.0 - 1e-6 && kb.max().double_value(&[]) <= 1.0);
            assert!(kd.min().double_value(&[]) >= 20.0 / 80.0 - 1e-6 && kd.max().double_value(&[]) <= 1.0);
            assert!(kb.min().double_value(&[]) < 20.0 / 80.0, "bright frames reach darker than dim frames' floor");
        }

        #[test]
        fn optics_augmentations_are_physical_and_bounded() {
            let device = Device::Cpu;
            let gains = optics_block_gains(64, device);
            let (lo, hi) = (gains.min().double_value(&[]), gains.max().double_value(&[]));
            assert!(lo >= 0.5 - 1e-6 && hi <= 2.0 + 1e-6, "gain range {lo}..{hi}");
            // Channels of one phase block (c=0,1,4,5) share a gain.
            for c in [1i64, 4, 5] {
                assert!((gains.select(1, c) - gains.select(1, 0)).abs().max().double_value(&[]) < 1e-6);
            }
            let image = Tensor::full([32, 16, 128, 192], 0.3, (Kind::Float, device));
            let mut truth = Tensor::zeros([32, 6, 256, 384], (Kind::Float, device));
            let _ = truth.narrow(2, 64, 128).narrow(3, 128, 128).narrow(1, 0, 1).fill_(1.0);
            let lit = optics_glare(&image, &truth);
            assert!((&lit - &image).min().double_value(&[]) >= 0.0, "glare only adds light");
            assert!((&lit - &image).max().double_value(&[]) > 0.25, "some glare is strong");
            let mut sequence = [0.1, 0.6, 0.6, 0.9].into_iter().cycle();
            let mut random = || sequence.next().unwrap();
            for _ in 0..3 {
                let blurred = optics_blur(&image, &mut random);
                assert!((blurred - &image).abs().max().double_value(&[]) < 1e-5, "blur keeps a flat field");
            }
            let mut none = || 0.9;
            assert!(optics_blur(&lit, &mut none).equal(&lit));
            let model = LightNoiseModel {
                k: Tensor::from_slice(&[2.0f32; 16]).reshape([1, 16, 1, 1]),
                black: Tensor::from_slice(&[50.0f32; 16]).reshape([1, 16, 1, 1]),
                light_range: [0.25, 0.25],
                signal_floor: None,
                source: Value::Null,
            };
            let mixed = model.apply_mixed(&image) * 1023.0;
            let means = mixed.mean_dim([1i64, 2, 3].as_slice(), false, Kind::Float);
            let bright = means.ge(1023.0 * 0.3 - 8.0).sum(Kind::Int64).int64_value(&[]);
            let dark = means.le(50.0 + 0.25 * (1023.0 * 0.3 - 50.0) + 8.0).sum(Kind::Int64).int64_value(&[]);
            assert_eq!(bright + dark, 32, "each frame is either gain-compensated or left dim");
            assert!(bright > 0 && dark > 0);
        }

        #[test]
        fn light_noise_is_identity_at_full_light_and_adds_modeled_shot_noise_when_dimmed() {
            let model = |range: [f64; 2]| LightNoiseModel {
                k: Tensor::from_slice(&[2.0f32; 16]).reshape([1, 16, 1, 1]),
                black: Tensor::from_slice(&[50.0f32; 16]).reshape([1, 16, 1, 1]),
                light_range: range,
                signal_floor: None,
                source: Value::Null,
            };
            let image = Tensor::full([8, 16, 128, 192], 300.0 / 1023.0, (Kind::Float, Device::Cpu));
            let same = model([1.0, 1.0]).apply(&image);
            assert!((same - &image).abs().max().double_value(&[]) < 1e-6);
            let dim = model([0.25, 0.25]).apply(&image);
            let added = (dim - &image) * 1023.0;
            assert!(added.mean(Kind::Float).double_value(&[]).abs() < 1.0, "zero-mean noise");
            // Photosite std sqrt(K*signal*(1/k-1)) = sqrt(2*250*3) ~ 38.7 codes,
            // reduced but not removed by resampling 105x70 lattices to 192x128.
            let std = added.std(true).double_value(&[]);
            assert!((0.4 * 38.7..=38.7).contains(&std), "std {std}");
        }

        #[test]
        fn spatial_shadow_is_bounded_directional_and_has_an_identity_control() {
            let scalar = |v: f32| Tensor::from_slice(&[v]).reshape([1, 1, 1, 1]);
            let (angle, offset, softness) = (scalar(0.0), scalar(0.0), scalar(0.1));
            let field = soft_shadow_field(8, 12, &angle, &offset, &softness, &scalar(0.6));
            assert_eq!(field.size(), [1, 1, 8, 12]);
            assert!(field.min().double_value(&[]) >= 0.3999);
            assert!(field.max().double_value(&[]) <= 1.0);
            assert!(field.double_value(&[0, 0, 3, 0]) > field.double_value(&[0, 0, 3, 11]) + 0.5);
            assert_eq!(
                field.double_value(&[0, 0, 0, 5]),
                field.double_value(&[0, 0, 7, 5])
            );
            let control = soft_shadow_field(8, 12, &angle, &offset, &softness, &scalar(0.0));
            assert_eq!((control - 1.0).abs().max().double_value(&[]), 0.0);
        }

        #[test]
        fn cropped_unknown_pixels_never_train_negative_labels() {
            let logits =
                Tensor::zeros([1, 6, 4, 4], (Kind::Float, Device::Cpu)).set_requires_grad(true);
            let truth = Tensor::ones_like(&logits);
            let weights = Tensor::ones([1, 6], (Kind::Float, Device::Cpu));
            let valid = Tensor::ones([1, 1, 4, 4], (Kind::Float, Device::Cpu));
            let _ = valid.narrow(3, 2, 2).fill_(0.0);
            loss_with_valid_pixels(&logits, &truth, &weights, Some(&valid)).backward();
            assert_eq!(
                logits.grad().narrow(3, 2, 2).abs().max().double_value(&[]),
                0.0
            );
            assert!(logits.grad().narrow(3, 0, 2).abs().max().double_value(&[]) > 0.0);
            let full = Tensor::ones_like(&valid);
            assert!(
                (loss(&logits, &truth, &weights).double_value(&[])
                    - loss_with_valid_pixels(&logits, &truth, &weights, Some(&full))
                        .double_value(&[]))
                .abs()
                    < 1e-6
            );
            assert_eq!(
                loss_with_valid_pixels(&logits, &truth, &weights, Some(&valid.zeros_like()))
                    .double_value(&[]),
                0.0
            );
        }

        #[test]
        fn reframing_keeps_image_and_labels_aligned_and_marks_missing_source() {
            let plane = Tensor::arange(12, (Kind::Float, Device::Cpu))
                .reshape([1, 1, 1, 12])
                .repeat([1, 1, 8, 1])
                / 12.0;
            let image = plane.repeat([1, 3, 1, 1]) * 255.0;
            let truth = plane.repeat([1, 6, 1, 1]);
            let theta = Tensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0]).reshape([1, 2, 3]);
            let grid = Tensor::affine_grid_generator(&theta, image.size(), false);
            let (shifted_image, shifted_truth, valid) =
                resample_training_pair(&image, &truth, &grid, true);
            let valid = valid.unwrap();
            assert_eq!(valid.narrow(3, 0, 6).min().double_value(&[]), 1.0);
            assert_eq!(valid.narrow(3, 6, 6).max().double_value(&[]), 0.0);
            assert!(shifted_image.narrow(3, 6, 6).min().double_value(&[]) > 200.0);
            assert_eq!(shifted_truth.narrow(3, 6, 6).max().double_value(&[]), 0.0);
            let aligned =
                (shifted_image.narrow(1, 0, 1) / 255.0 - shifted_truth.narrow(1, 0, 1)) * valid;
            assert!(aligned.abs().max().double_value(&[]) < 1e-6);
            assert!(resample_training_pair(&image, &truth, &grid, false)
                .2
                .is_none());
        }

        #[test]
        fn coarse_luma_map_keeps_sources_separate_and_records_local_contrast() {
            let rgb = Tensor::zeros(
                [2, 3, FRAME_HEIGHT as i64, FRAME_WIDTH as i64],
                (Kind::Float, Device::Cpu),
            );
            let _ = rgb.get(0).fill_(0.25);
            let _ = rgb.get(1).fill_(0.75);
            let context = coarse_luma_context(&rgb);
            assert_eq!(context.size(), [2, 2, 16, 24]);
            assert!((context.double_value(&[0, 0, 5, 7]) - 0.25).abs() < 1e-6);
            assert!((context.double_value(&[1, 0, 5, 7]) - 0.75).abs() < 1e-6);
            assert_eq!(context.narrow(1, 1, 1).max().double_value(&[]), 0.0);
            let _ = rgb.get(0).narrow(2, 0, 8).fill_(0.75);
            let changed = coarse_luma_context(&rgb);
            assert!(changed.double_value(&[0, 1, 0, 0]) > 0.24);
            assert_eq!(
                (changed.get(1) - context.get(1))
                    .abs()
                    .max()
                    .double_value(&[]),
                0.0
            );
        }

        #[test]
        fn luma_context_network_preserves_rgb_and_six_mask_contract() {
            let store = nn::VarStore::new(Device::Cpu);
            let net = Net::with_luma_context(&store.root(), true);
            let parameters: usize = store.trainable_variables().iter().map(Tensor::numel).sum();
            assert!(parameters < 225_000, "{parameters}");
            let output = tch::no_grad(|| {
                net.forward(&Tensor::zeros(
                    [1, 3, FRAME_HEIGHT as i64, FRAME_WIDTH as i64],
                    (Kind::Uint8, Device::Cpu),
                ))
            });
            assert_eq!(
                output.size(),
                [
                    1,
                    SEMANTIC_PROMPT_COUNT as i64,
                    FRAME_HEIGHT as i64,
                    FRAME_WIDTH as i64
                ]
            );
            assert_eq!(output.isfinite().all().int64_value(&[]), 1);
        }

        #[test]
        fn paired_replay_uses_exact_clocks_and_retains_missing_eyes() {
            let row = |clock: &str, time, eye| json!({"clock_lineage":clock,"frame":{"timestamp_ns":time,"eye_id":eye}});
            let groups = source_groups(
                vec![
                    row("a", 1, 2),
                    row("b", 1, 1),
                    row("a", 2, 1),
                    row("a", 1, 1),
                ],
                true,
            )
            .unwrap();
            assert_eq!(
                groups.iter().map(Vec::len).collect::<Vec<_>>(),
                vec![2, 1, 1]
            );
            assert_eq!(groups[0][0]["frame"]["eye_id"], 1);
            assert_eq!(groups[0][1]["frame"]["eye_id"], 2);
            assert!(source_groups(vec![row("a", 1, 1), row("a", 1, 1)], true).is_err());
        }

        #[test]
        fn conic_replay_keeps_observed_native_arcs_and_does_not_complete_gaps() {
            let retained = vec![(10.0, 20.0), (12.0, 21.0), (70.0, 40.0), (73.0, 38.0)];
            let segments = vec![vec![0, 1], vec![2, 3]];
            let mut proposal = ProposalMasks::default();
            proposal.outer_fit = Some(OuterMaskFitReview {
                ellipse: Ellipse {
                    center: (40.0, 30.0),
                    major_radius: 35.0,
                    minor_radius: 15.0,
                    angle: 0.2,
                },
                source_component_area_px: 123.0,
                retained_points: Arc::new(retained.clone()),
                conic_segments: Arc::new(segments.clone()),
                flat_tire_points: Arc::new(vec![(40.0, 12.0)]),
                upper_flat_tire: true,
                lower_flat_tire: false,
            });
            let record = conic_replay_evidence(&proposal, false);
            let candidate = &record["candidates"][0];
            assert_eq!(candidate["baseline_retained"], json!(retained));
            assert_eq!(candidate["baseline_retained_segments"], json!(segments));
            assert_eq!(candidate["baseline_raw_admitted"], false);
            assert_eq!(candidate["outline"], json!([]));
            assert!(record["pupil_void"].is_null());
        }

        #[test]
        fn conic_replay_retains_selected_mask_outline_when_complete_fit_is_missing() {
            let mut pixels = vec![0; 64 * 64];
            for y in 12..48 {
                for x in 10..50 { pixels[y * 64 + x] = 1; }
            }
            let mut proposal = ProposalMasks {
                source_width: 420,
                source_height: 280,
                semantic: Some(SemanticProposalMasks {
                    prompt_index: OUTER_IRIS_PROMPT,
                    width: 64,
                    height: 64,
                    selected_query: Some(7),
                    tile_slice: None,
                    masks: vec![
                        ProposalMask { query: 0, score: 0.99,
                            pixels: Arc::new(vec![0; 64 * 64]),
                            boundary_pixels: Arc::new(Vec::new()) },
                        ProposalMask { query: 7, score: 0.8,
                            pixels: Arc::new(pixels),
                            boundary_pixels: Arc::new(Vec::new()) },
                    ],
                }),
                ..ProposalMasks::default()
            };
            let record = conic_replay_evidence(&proposal, false);
            let candidate = &record["candidates"][0];
            assert!(record["selected_query"].is_null());
            assert!(candidate["baseline_ellipse"].is_null());
            assert!(candidate["baseline_retained"].is_null());
            assert_eq!(candidate["baseline_raw_admitted"], false);
            let outline = candidate["outline"].as_array().unwrap();
            assert_eq!(outline.len(), 256);
            // Native pixel centers, including the rectangular corners. An
            // ellipse completion would not preserve all four mask edges.
            let bounds = [(68.40625, 324.34375), (54.1875, 207.3125)];
            for (axis, (lo, hi)) in bounds.into_iter().enumerate() {
                let values = outline.iter().map(|p| p[axis].as_f64().unwrap()).collect::<Vec<_>>();
                assert!((values.iter().copied().fold(f64::INFINITY, f64::min) - lo).abs() < 0.1);
                assert!((values.iter().copied().fold(f64::NEG_INFINITY, f64::max) - hi).abs() < 0.1);
            }
            proposal.semantic.as_mut().unwrap().selected_query = None;
            assert_eq!(conic_replay_evidence(&proposal, false)["candidates"][0]["outline"], json!([]));
            let semantic = proposal.semantic.as_mut().unwrap();
            semantic.selected_query = Some(7);
            semantic.prompt_index = OUTER_IRIS_PROMPT + 1;
            assert_eq!(conic_replay_evidence(&proposal, false)["candidates"][0]["outline"], json!([]));
        }
        #[test]
        fn student_network_is_small_finite_and_preserves_the_six_head_contract() {
            let store = nn::VarStore::new(Device::Cpu);
            let net = Net::new(&store.root());
            let parameters: usize = store.trainable_variables().iter().map(Tensor::numel).sum();
            assert!(parameters < 300_000, "{parameters}");
            let output = tch::no_grad(|| {
                net.forward(&Tensor::zeros(
                    [1, 3, FRAME_HEIGHT as i64, FRAME_WIDTH as i64],
                    (Kind::Uint8, Device::Cpu),
                ))
            });
            assert_eq!(
                output.size(),
                [
                    1,
                    SEMANTIC_PROMPT_COUNT as i64,
                    FRAME_HEIGHT as i64,
                    FRAME_WIDTH as i64
                ]
            );
            assert_eq!(output.isfinite().all().int64_value(&[]), 1);
        }
        #[test]
        fn unknown_teacher_heads_do_not_become_negative_training_labels() {
            let logits =
                Tensor::zeros([1, 6, 4, 4], (Kind::Float, Device::Cpu)).set_requires_grad(true);
            let truth = Tensor::ones_like(&logits);
            let weights = Tensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0, 0.0, 0.0]).reshape([1, 6]);
            loss(&logits, &truth, &weights).backward();
            let gradient = logits.grad();
            assert!(
                gradient
                    .select(1, 0)
                    .abs()
                    .sum(Kind::Float)
                    .double_value(&[])
                    > 0.0
            );
            for head in [1, 3, 4, 5] {
                assert_eq!(
                    gradient
                        .select(1, head)
                        .abs()
                        .sum(Kind::Float)
                        .double_value(&[]),
                    0.0
                );
            }
        }
    }
}
#[cfg(feature = "sam31")]
pub(super) use cuda::{TeacherSample,prepare_raw};
#[cfg(feature = "sam31")]
pub use cuda::{run_cli, Model};
