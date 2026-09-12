//! Small, fixed-vocabulary CUDA mask student. SAM supplies pseudo-labels only;
//! the shared observed-contour/RAW/3D solver remains the geometry authority.
//! Weights, teacher exports, and reports belong under the runtime data links.
use super::*;

pub const ARCHITECTURE: &str = "buttercup-eye-mask-unet-v1";
pub fn default_model_path() -> PathBuf {
    std::env::var_os("BUTTERCUP_EYE_STUDENT_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/models/eye_student_v1.ot"))
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
    if meta["architecture"] != ARCHITECTURE
        || meta["input_shape"] != serde_json::json!([3, FRAME_HEIGHT, FRAME_WIDTH])
        || meta["prompts"] != serde_json::json!(SEMANTIC_PROMPT_LABELS)
        || meta["preprocess"] != PreprocessRegime::configured_live()?.label()
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
            Self {
                stem: block(p / "stem", 3, 12, 2),
                down1: block(p / "down1", 12, 24, 2),
                down2: block(p / "down2", 24, 48, 2),
                bottom: block(p / "bottom", 48, 72, 2),
                up2: block(p / "up2", 72 + 48, 48, 1),
                up1: block(p / "up1", 48 + 24, 24, 1),
                head: nn::conv2d(
                    p / "head",
                    24 + 12,
                    SEMANTIC_PROMPT_COUNT as i64,
                    1,
                    Default::default(),
                ),
            }
        }
    }
    impl Module for Net {
        fn forward(&self, input: &Tensor) -> Tensor {
            let a = self.stem.forward(&(input.to_kind(Kind::Float) / 255.0));
            let b = self.down1.forward(&a);
            let c = self.down2.forward(&b);
            let d = self.bottom.forward(&c);
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
    }
    impl Model {
        pub fn load(path: &Path) -> Result<Self, String> {
            validate_model(path)?;
            let mut store = nn::VarStore::new(Device::Cuda(0));
            let net = Net::new(&store.root());
            store.load(path).map_err(|e| format!("load student: {e}"))?;
            store.freeze();
            Ok(Self { store, net })
        }
        pub fn infer(&self, image: &[u8]) -> Result<Tensor, String> {
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
    }
    pub(crate) struct TeacherSample {
        pub image: Vec<u8>,
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
    fn export(index: &Path, out: &Path) -> Result<(), String> {
        runtime_output(out)?;
        std::fs::create_dir(out).map_err(|e| e.to_string())?;
        let mut images = new_writer(&out.join("images.u8"))?;
        let mut masks = new_writer(&out.join("masks.u8"))?;
        let mut records = new_writer(&out.join("records.jsonl"))?;
        let model = std::env::var_os("BUTTERCUP_SAM31_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(super::super::default_model_path);
        let count =
            runtime::export_student_teacher(&model, source_frames(index)?, |row, sample| {
                images.write_all(&sample.image).map_err(|e| e.to_string())?;
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
        masks.flush().map_err(|e| e.to_string())?;
        records.flush().map_err(|e| e.to_string())?;
        write_json(
            &out.join("manifest.json"),
            &json!({"schema":"buttercup-eye-student-teacher-v1",
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
        fn load(path: &Path) -> Result<Self, String> {
            let manifest: Value = serde_json::from_reader(
                File::open(path.join("manifest.json")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
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
            let images = std::fs::read(path.join("images.u8")).map_err(|e| e.to_string())?;
            let masks = std::fs::read(path.join("masks.u8")).map_err(|e| e.to_string())?;
            let plane = FRAME_HEIGHT * FRAME_WIDTH;
            if images.len() != n * 3 * plane || masks.len() != n * SEMANTIC_PROMPT_COUNT * plane {
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
                images: Tensor::from_slice(&images).reshape([
                    n as i64,
                    3,
                    FRAME_HEIGHT as i64,
                    FRAME_WIDTH as i64,
                ]),
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
        let axes = [2i64, 3];
        let p = logits.sigmoid();
        let bce = (logits.clamp_min(0.0) - logits * truth + (-logits.abs()).exp().log1p())
            .mean_dim(axes.as_slice(), false, Kind::Float);
        let intersection = (&p * truth).sum_dim_intlist(axes.as_slice(), false, Kind::Float);
        let total = (&p + truth).sum_dim_intlist(axes.as_slice(), false, Kind::Float);
        let dice: Tensor = 1.0 - (intersection * 2.0 + 1.0) / (total + 1.0);
        let priority =
            Tensor::from_slice(&[3.0f32, 0.5, 2.0, 0.5, 0.5, 0.5]).to_device(logits.device());
        let weights = weights * priority;
        ((bce + dice) * &weights).sum(Kind::Float) / weights.sum(Kind::Float).clamp_min(1.0)
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
    fn train(data_path: &Path, model_path: &Path, epochs: usize) -> Result<(), String> {
        runtime_output(model_path)?;
        runtime_output(&metadata_path(model_path))?;
        if epochs == 0 || epochs > 2000 {
            return Err("epochs must be 1..2000".into());
        }
        runtime::student_cuda_init()?;
        tch::set_num_threads(2);
        tch::manual_seed(17091);
        let device = Device::Cuda(0);
        let data = Dataset::load(data_path)?;
        let mut train = data.indices("train");
        let validation = data.indices("validation");
        let test = data.indices("test");
        if train.len() < 16 || validation.len() < 4 || test.len() < 4 {
            return Err("need train/validation/test sessions with at least 16/4/4 frames".into());
        }
        let mut store = nn::VarStore::new(device);
        let net = Net::new(&store.root());
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
            optimizer.set_lr(
                0.00005
                    + 0.00095
                        * 0.5
                        * (1.0 + (std::f64::consts::PI * epoch as f64 / epochs as f64).cos()),
            );
            let mut loss_sum = 0.0;
            let mut batches = 0;
            for ids in train.chunks(12) {
                let (mut image, mut truth, weights) = data.batch(ids, device);
                if random_u() % 2 == 0 {
                    image = image.flip([3]);
                    truth = truth.flip([3]);
                }
                // Same affine for image and labels; never a completed-ellipse target.
                let n = ids.len() as i64;
                let angles = (Tensor::rand([n], (Kind::Float, device)) - 0.5) * 0.25;
                let scales = Tensor::rand([n], (Kind::Float, device)) * 0.20 + 0.90;
                let c = angles.cos() * &scales;
                let s = angles.sin() * scales;
                let shift = (Tensor::rand([n, 2], (Kind::Float, device)) - 0.5) * 0.16;
                let theta = Tensor::stack(
                    &[
                        Tensor::stack(&[c.shallow_clone(), -&s, shift.select(1, 0)], 1),
                        Tensor::stack(&[s, c, shift.select(1, 1)], 1),
                    ],
                    1,
                );
                let grid = Tensor::affine_grid_generator(&theta, image.size(), false);
                image = image.grid_sampler(&grid, 0, 1, false);
                truth = truth.grid_sampler(&grid, 0, 0, false);
                let gain = Tensor::rand([n, 3, 1, 1], (Kind::Float, device)) * 0.35 + 0.80;
                image = (image * gain
                    + (Tensor::rand([n, 1, 1, 1], (Kind::Float, device)) - 0.5) * 12.0)
                    .clamp(0.0, 255.0);
                let value = loss(&net.forward(&image), &truth, &weights);
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
        write_json(
            &metadata_path(model_path),
            &json!({"architecture":ARCHITECTURE,"input_shape":[3,FRAME_HEIGHT,FRAME_WIDTH],
            "prompts":SEMANTIC_PROMPT_LABELS,"preprocess":data.manifest["preprocess"],"parameters":parameters,
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
        let data = Dataset::load(data_path)?;
        let model = Model::load(model_path)?;
        let mut rows = new_writer(output)?;
        for (i, row) in data.rows.iter().enumerate() {
            let raw = source_frames_from_row(row["input"].clone())?;
            let image = data.images.get(i as i64).contiguous();
            let mut bytes = vec![0; image.numel()];
            let count = bytes.len();
            image.copy_data(&mut bytes, count);
            let started = Instant::now();
            let logits = model.infer(&bytes)?;
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
        let mut bytes = vec![0; integer(&row, "raw_length")? as usize];
        file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        Ok(Arc::new(RawFrame {
            eye_index: integer(f, "eye_id")? as usize - 1,
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
        let score = proposal.semantic.as_ref().and_then(|semantic| {
            semantic.selected_query.and_then(|selected| {
                semantic
                    .masks
                    .iter()
                    .find(|mask| mask.query == selected)
                    .map(|mask| mask.score)
            })
        });
        json!({"selected_query":fit.map(|_|0),
            "candidates":[{"query":0,"semantic_score":score,
                "baseline_raw_admitted":admitted,
                "baseline_ellipse":fit.map(|f|ellipse_json(f.ellipse)),
                "baseline_retained":fit.map(|f|f.retained_points.as_ref()),
                "baseline_retained_segments":fit.map(|f|f.conic_segments.as_ref()),
                "baseline_censored":fit.map(|f|f.flat_tire_points.as_ref()),
                "outline":[],
                "scope":"selected live proposal; only measured retained arcs, no synthetic completed rim"}],
            "pupil_void":proposal.inner_pupil_fit.map(|p|json!({"ellipse":ellipse_json(p.ellipse)}))})
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
pub(super) use cuda::TeacherSample;
#[cfg(feature = "sam31")]
pub use cuda::{run_cli, Model};
