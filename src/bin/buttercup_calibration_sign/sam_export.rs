//! Recreate the SAM3.1 detector export from the upstream checkpoint.
//! Rust owns asset verification, dependency invocation and provenance. The
//! small adapter below calls the pinned upstream PyTorch API for TorchScript
//! tracing; it contains no RAW preparation, label selection or custom training.
use super::{bootstrapability as boot, data, Result};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path, process::Command, time::Instant};

pub const REVISION: &str = "847e1a3b15115a04c87c0760297f044f0555d970";
pub const CHECKPOINT_REVISION: &str = "daa63191845a41281374e725f4c9e51c7a824460";
pub const CHECKPOINT_SHA: &str = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6";
pub const PROMPTS: [&str; 2] = ["iris", "pupil"];
pub const ANATOMY_PROMPTS: [&str; 6] = [
    "eye",
    "sclera",
    "iris",
    "upper eyelid",
    "lower eyelid",
    "eyelashes",
];
pub fn adapter_hash() -> String {
    data::digest(ADAPTER.as_bytes())
}
pub fn cpu_adapter_hash() -> String {
    data::digest(cpu_adapter().as_bytes())
}
/// Recreate the same pinned upstream detector directly on CPU, in FP32.
/// Moving the BF16 CUDA trace to CPU failed the declared numerical parity
/// check. This is a distinct arithmetic variant with its own eager/reload
/// checks; it is not advertised as bit-equivalent to that CUDA graph.
fn cpu_adapter() -> String {
    ADAPTER
        .replace("import sam3\n", CPU_PRELUDE)
        .replace("detector.cuda().eval()", "detector.cpu().eval()")
        .replace("device='cuda'", "device='cpu'")
        .replace("map_location='cuda'", "map_location='cpu'")
        .replace("torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False)", "torch.autocast('cpu',enabled=False)")
        .replace("torch.autocast('cuda',enabled=False)", "torch.autocast('cpu',enabled=False)")
        .replace("return ((image-.5)/.5).to(torch.bfloat16)", "return ((image-.5)/.5).float()")
        .replace("'device':torch.cuda.get_device_name()", "'device':'cpu'")
        .replace("'exported_dtype':'bfloat16'", "'exported_dtype':'float32'")
        .replace("(x-.5)/.5 -> BF16", "(x-.5)/.5 -> FP32")
        .replace("upstream BF16 image backbone; FP32 semantic encoder, decoder and mask head; no TF32, cached casts or MHA fastpath", "official weights with FP32 operations throughout, directly traced on CPU; forced BF16 fused linear-activation replaced by FP32 linear and identical activation; no autocast or MHA fastpath")
}
const CPU_PRELUDE: &str = r#"import sam3
import sam3.model.vitdet as vitdet
# The upstream fused helper unconditionally casts its operands to BF16 even
# when autocast is disabled. Preserve the linear/GELU algebra in FP32 instead.
def fp32_linear_activation(activation, linear, x):
    y = F.linear(x.float(), linear.weight.float(), linear.bias.float())
    if activation in [F.relu, nn.ReLU]: return F.relu(y)
    if activation in [F.gelu, nn.GELU]: return F.gelu(y)
    raise RuntimeError('unsupported FP32 activation')
vitdet.addmm_act = fp32_linear_activation
# The upstream optional positional cache is constructed on CUDA and is not a
# registered buffer. Disable that optimization; forward builds identical sine
# positions on the input's CPU device.
original_position_encoding = mb._create_position_encoding
def cpu_position_encoding(precompute_resolution=None):
    return original_position_encoding(precompute_resolution=None)
mb._create_position_encoding = cpu_position_encoding
# The box-relative-position cache has the same explicit CUDA preallocation.
from sam3.model.decoder import TransformerDecoder
original_get_coords = TransformerDecoder._get_coords
def cpu_get_coords(h, w, device):
    return original_get_coords(h, w, torch.device('cpu'))
TransformerDecoder._get_coords = staticmethod(cpu_get_coords)
"#;
pub fn hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 3 && !(args.len() == 4 && args[0] == "sam-export-anatomy-cpu") {
        return Err("sam-export[-anatomy[-cpu]] OFFICIAL_CHECKPOINT NEW_OUTPUT_DIR [SIX_PROMPTS_JSON for anatomy CPU]".into());
    }
    let checkpoint = fs::canonicalize(&args[1])?;
    let cpu = args[0] == "sam-export-anatomy-cpu";
    let adapter = if cpu {
        cpu_adapter()
    } else {
        ADAPTER.to_string()
    };
    let prompts: Vec<String> = if let Some(path) = args.get(3) {
        let values: Vec<String> = serde_json::from_slice(&fs::read(path)?)?;
        if values.len() != 6 || values.iter().any(|s| s.trim().is_empty()) {
            return Err("anatomy export requires six nonempty literal prompts".into());
        }
        values
    } else if args[0].starts_with("sam-export-anatomy") {
        ANATOMY_PROMPTS.iter().map(|s| s.to_string()).collect()
    } else {
        PROMPTS.iter().map(|s| s.to_string()).collect()
    };
    let out = data::output(&args[2])?;
    let start = Instant::now();
    if hash(&checkpoint)? != CHECKPOINT_SHA {
        return Err("SAM3.1 upstream checkpoint hash mismatch".into());
    }
    let source = boot::current_source(Path::new("."))?;
    let graph = json!({"schema":boot::SCHEMA,"source":source,"targets":["export"],"nodes":[
        {"id":"source","kind":"source","sha256":source.tree_sha256,"dependencies":[]},
        {"id":"sam31","kind":"sam3","sha256":CHECKPOINT_SHA,"dependencies":[]},
        {"id":"export","kind":"export","planned":true,"sha256":null,"dependencies":["source","sam31"]}]});
    let planned: boot::Manifest = serde_json::from_value(graph.clone())?;
    let preflight =
        boot::validate(&planned, &source).map_err(|e| format!("SAM export preflight: {e:?}"))?;
    data::write(out.join("bootstrap-preflight.json"), &preflight)?;
    fs::write(out.join("upstream-adapter.py"), &adapter)?;
    data::write(out.join("prompts.json"), &prompts)?;
    let package = format!("sam3 @ git+https://github.com/facebookresearch/sam3.git@{REVISION}");
    let mut command = Command::new("uv");
    command.args([
        "run",
        "--no-project",
        "--isolated",
        "--python",
        "3.12.11",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu128",
        "--index-strategy",
        "unsafe-best-match",
    ]);
    for dependency in [
        &*package,
        "torch==2.9.0+cu128",
        "torchvision==0.24.0+cu128",
        "numpy==1.26.4",
        "timm==1.0.20",
        "einops==0.8.1",
        "decord==0.6.0",
        "pycocotools==2.0.10",
        "setuptools==80.9.0",
        "psutil==6.1.1",
        "scipy==1.16.2",
        "anyio==4.15.1",
        "certifi==2026.7.22",
        "click==8.5.0",
        "filelock==3.32.7",
        "fsspec==2026.7.0",
        "ftfy==6.1.1",
        "h11==0.16.0",
        "hf-xet==1.6.0",
        "httpcore==1.0.9",
        "httpx==0.28.1",
        "huggingface_hub==1.31.0",
        "idna==3.19",
        "iopath==0.1.10",
        "Jinja2==3.1.6",
        "MarkupSafe==3.0.3",
        "mpmath==1.3.0",
        "networkx==3.6.1",
        "packaging==26.3",
        "pillow==12.3.0",
        "portalocker==4.3.2",
        "PyYAML==6.0.3",
        "regex==2026.9.10",
        "safetensors==0.8.0",
        "sympy==1.14.0",
        "tqdm==4.70.1",
        "triton==3.5.0",
        "typing_extensions==4.16.0",
        "wcwidth==0.8.3",
        "nvidia-cublas-cu12==12.8.4.1",
        "nvidia-cuda-cupti-cu12==12.8.90",
        "nvidia-cuda-nvrtc-cu12==12.8.93",
        "nvidia-cuda-runtime-cu12==12.8.90",
        "nvidia-cudnn-cu12==9.10.2.21",
        "nvidia-cufft-cu12==11.3.3.83",
        "nvidia-cufile-cu12==1.13.1.3",
        "nvidia-curand-cu12==10.3.9.90",
        "nvidia-cusolver-cu12==11.7.3.90",
        "nvidia-cusparse-cu12==12.5.8.93",
        "nvidia-cusparselt-cu12==0.7.1",
        "nvidia-nccl-cu12==2.27.5",
        "nvidia-nvjitlink-cu12==12.8.93",
        "nvidia-nvshmem-cu12==3.3.20",
        "nvidia-nvtx-cu12==12.8.90",
    ] {
        command.args(["--with", dependency]);
    }
    command
        .arg("python")
        .arg(out.join("upstream-adapter.py"))
        .arg(&checkpoint)
        .arg(&out);
    eprintln!("Recreating detector from official SAM3.1 weights; no old graph is read");
    let status = command.status()?;
    if !status.success() {
        return Err(format!("upstream SAM3.1 export failed: {status}").into());
    }
    if boot::current_source(Path::new("."))? != source {
        return Err("source changed during SAM export".into());
    }
    let model_sha = hash(&out.join("detector.pt"))?;
    let upstream: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("upstream-receipt.json"))?)?;
    if upstream["sam_package_source"]["vcs_info"]["commit_id"] != REVISION {
        return Err("installed SAM source does not match the pinned revision".into());
    }
    let receipt = json!({"schema":"buttercup-sam31-cold-export-v1","source":source,"checkpoint_sha256":CHECKPOINT_SHA,"checkpoint_revision":CHECKPOINT_REVISION,"checkpoint_repository":"https://huggingface.co/facebook/sam3.1","checkpoint_acquisition":"Request upstream access, then download sam3.1_multiplex.pt at the pinned revision and verify its SHA-256. The existing user-authorized checkpoint may be supplied.","upstream_repository":"https://github.com/facebookresearch/sam3","upstream_revision":REVISION,"adapter_sha256":data::digest(adapter.as_bytes()),"prompts":prompts,"model_sha256":model_sha,"upstream":upstream,"seconds":start.elapsed().as_secs_f64(),"role":"allowed external SAM3 dependency; not custom sign truth or a custom model cold-training certificate"});
    data::write(out.join("export.json"), &receipt)?;
    let mut graph = graph;
    graph["nodes"][2]["sha256"] = json!(model_sha);
    graph["nodes"][2]["planned"] = json!(false);
    let complete: boot::Manifest = serde_json::from_value(graph.clone())?;
    data::write(out.join("bootstrap-graph.json"), &graph)?;
    data::write(
        out.join("bootstrap-structural-check.json"),
        &boot::validate(&complete, &source).map_err(|e| format!("SAM export graph: {e:?}"))?,
    )?;
    eprintln!("SAM EXPORT DONE: {}", out.display());
    Ok(())
}

// Adapter for the external PyTorch/SAM API, maintained inside the Rust recipe.
// Strict checkpoint loading prevents random missing weights from being traced.
const ADAPTER: &str = r#"
import sys, json, hashlib, pathlib, importlib.metadata, time
import torch
from torch import nn
from torch.nn import functional as F
from sam3 import model_builder as mb
from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
from sam3.model.data_misc import FindStage
import sam3

torch.set_num_threads(4)
torch.manual_seed(829416)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.mha.set_fastpath_enabled(False)
torch._C._jit_set_autocast_mode(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_override_can_fuse_on_gpu(False)
checkpoint, output = sys.argv[1], pathlib.Path(sys.argv[2])
def runtime_versions():
    return {d.metadata['Name']:d.version for d in importlib.metadata.distributions()
            if d.metadata['Name'] and '_vendor' not in pathlib.Path(d.locate_file('')).parts}
versions=runtime_versions()
(output/'runtime-packages.json').write_text(json.dumps({'python':sys.version,'dependencies':versions},indent=2))
prompts = json.loads((output/'prompts.json').read_text())
bpe = pathlib.Path(sam3.__file__).parent/'assets/bpe_simple_vocab_16e6.txt.gz'
tri = mb._create_multiplex_tri_backbone(compile_mode=None, use_fa3=False, use_rope_real=False)
backbone = mb.SAM3VLBackboneTri(scalp=0, visual=tri, text=mb._create_text_encoder(str(bpe)))
detector = Sam3MultiplexDetector(
    num_feature_levels=1, backbone=backbone,
    transformer=mb._create_sam3_transformer(use_fa3=False),
    segmentation_head=mb._create_segmentation_head(use_fa3=False),
    semantic_segmentation_head=None, input_geometry_encoder=mb._create_geometry_encoder(),
    use_early_fusion=True, use_dot_prod_scoring=True,
    dot_prod_scoring=mb._create_dot_product_scoring(), supervise_joint_box_scores=True,
    is_multiplex=True)
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
if 'model' in state: state=state['model']
state = {k[len('detector.'):]:v for k,v in state.items() if k.startswith('detector.')}
detector.load_state_dict(state, strict=True)
del state
detector.cuda().eval().requires_grad_(False)
# Activation checkpointing is a training memory optimization. Its Python
# autograd node is not exportable and changes no inference arithmetic.
for module in detector.modules():
    if hasattr(module,'act_ckpt'): module.act_ckpt=False
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
    text = detector.backbone.forward_text(prompts, device='cuda')

class Export(nn.Module):
    def __init__(self, detector, text):
        super().__init__()
        self.detector=detector
        self.text_keys=list(text.keys())
        for key,value in text.items(): self.register_buffer(key,value)
        self.register_buffer('img_ids',torch.zeros(len(prompts),device='cuda',dtype=torch.long))
        self.register_buffer('text_ids',torch.arange(len(prompts),device='cuda',dtype=torch.long))
    def forward(self, image):
        normalized=image
        features=self.detector.backbone.forward_image(normalized)
        features.update({key:getattr(self,key) for key in self.text_keys})
        # Keep the semantic encoder, iterative box decoder and mask head in
        # FP32. Their BF16 rounding magnifies tiny eager/JIT arithmetic
        # differences near quantization boundaries on the diagnostic inputs.
        for key,value in list(features.items()):
            if isinstance(value,torch.Tensor) and value.is_floating_point(): features[key]=value.float()
        for level in features['backbone_fpn']: level.tensors=level.tensors.float()
        features['vision_pos_enc']=[p.float() for p in features['vision_pos_enc']]
        stage=FindStage(img_ids=self.img_ids,text_ids=self.text_ids,
            input_boxes=None,input_boxes_mask=None,input_boxes_label=None,input_points=None,input_points_mask=None)
        with torch.autocast('cuda',enabled=False):
            result=self.detector.forward_grounding(features,stage,None,self.detector._get_dummy_prompt(len(prompts)))
        score=result['pred_logits'].sigmoid().squeeze(-1)
        # Multiplex already folds presence into pred_logits because
        # supervise_joint_box_scores=True; do not multiply it twice.
        return score, result['pred_masks']

model=Export(detector,text).eval()
# Language features are fixed prompt buffers; the original transformer is not
# needed in image inference and all of its output was regenerated above.
del model.detector.backbone.language_backbone
def input_sample():
    raw=torch.randint(0,256,(1,3,1120,1260),device='cuda',dtype=torch.uint8)
    image=F.interpolate(raw.float()/255.,(1008,1008),mode='bilinear',align_corners=False,antialias=True)
    return ((image-.5)/.5).to(torch.bfloat16)
example=input_sample()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
    reference=model(example)
    traced=torch.jit.trace(model,(example,),check_trace=False,strict=False)
    torch._C._jit_pass_inline(traced.graph)
    # Drop uncalled geometry/tracker methods, including torchvision operators
    # that the fixed text-only detector does not use in native inference.
    traced=torch.jit.freeze(traced,optimize_numerics=False)
    # JIT may change intermediate strides in packed multi-head attention.
    # Reshape preserves element order and copies only when view cannot alias;
    # parity below checks the resulting serialized graph against upstream.
    graph=traced.graph
    def portable_views(block):
        for node in list(block.nodes()):
            for child in node.blocks(): portable_views(child)
            if node.kind()=='torchvision::roi_align':
                # Fixed text prompts have no geometric boxes. SAM still calls
                # ROIAlign on a constant empty box tensor. Its empty result is
                # exact, and needs no TorchVision extension in the Rust runtime.
                boxes=node.inputsAt(1).toIValue()
                if not isinstance(boxes,torch.Tensor) or boxes.ndim!=2 or boxes.shape[0]!=0:
                    raise RuntimeError('nonempty/dynamic ROIAlign cannot be removed')
                size=node.output().type().sizes()
                if size is None or any(s is None for s in size):
                    size=[0,detector.geometry_encoder.boxes_pool_project.in_channels,
                          int(node.inputsAt(3).toIValue()),int(node.inputsAt(4).toIValue())]
                dtype={'Float':torch.float32,'BFloat16':torch.bfloat16,'Half':torch.float16}.get(node.output().type().scalarType(),torch.float32)
                value=graph.insertConstant(torch.empty(size,device='cuda',dtype=dtype))
                value.node().moveBefore(node)
                node.output().replaceAllUsesWith(value)
                node.destroy()
                continue
            if node.kind()=='aten::view':
                replacement=graph.create('aten::reshape',list(node.inputs()),1)
                replacement.output().setType(node.output().type())
                replacement.insertBefore(node)
                node.output().replaceAllUsesWith(replacement.output())
                node.destroy()
    portable_views(graph)
    traced.save(str(output/'detector.pt'))
    torch._C._set_graph_executor_optimize(False)
    reload=torch.jit.load(str(output/'detector.pt'),map_location='cuda').eval()
    errors=[]
    for seed in [829416,829417]:
        torch.manual_seed(seed)
        image=input_sample()
        a=model(image)
        # The trace already contains the casts performed by upstream autocast,
        # including its explicitly full-precision subregions. Applying an
        # outer autocast again would recast those operations during replay.
        with torch.autocast('cuda',enabled=False): b=reload(image)
        errors.append([float((x.float()-y.float()).abs().max()) for x,y in zip(a,b)])
    print('EXPORT PARITY',errors,flush=True)
    if any(e[0]>0.01 or e[1]>0.15 for e in errors):
        raise RuntimeError('export/reload parity outside predeclared tolerances: '+str(errors))
versions=runtime_versions()
receipt={'torch':torch.__version__,'python':sys.version,'cuda':torch.version.cuda,
    'device':torch.cuda.get_device_name(),'dependencies':versions,
    'sam_package_source':json.loads(importlib.metadata.distribution('sam3').read_text('direct_url.json')),
    'bpe_sha256':hashlib.sha256(bpe.read_bytes()).hexdigest(),
    'checkpoint_detector_strict_load':True,'prompts':prompts,
    'exported_input':[1,3,1008,1008],'exported_dtype':'bfloat16','input_contract':'uint8 RGB/255 -> antialiased bilinear 1008 square -> (x-.5)/.5 -> BF16, outside the serialized graph',
    'arithmetic':'upstream BF16 image backbone; FP32 semantic encoder, decoder and mask head; no TF32, cached casts or MHA fastpath',
    'outputs':[list(t.shape) for t in reference],
    'reload_max_absolute_errors_scores_masks':errors,
    'limits':'Parity on seeded inputs verifies export arithmetic only; RAW segmentation and sign accuracy are separate evaluations.'}
(output/'upstream-receipt.json').write_text(json.dumps(receipt,indent=2))
print('UPSTREAM EXPORT FINISHED',errors,flush=True)
"#;
