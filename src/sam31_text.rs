//! Native SAM 3.1 text prompting.
//!
//! SAM's image detector consumes CLIP-like token features rather than strings.
//! This module reproduces the upstream tokenizer and text transformer with
//! `tch`, so prompt bundles can be generated without Python or libpython.

#![cfg(feature = "sam31")]

use flate2::read::GzDecoder;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use tch::{Device, Kind, Tensor};

const CONTEXT_LENGTH: usize = 32;
const VOCAB_SIZE: usize = 49_408;
const WIDTH: i64 = 1_024;
const HEADS: i64 = 16;
const HEAD_WIDTH: i64 = WIDTH / HEADS;
const LAYERS: usize = 24;
const BPE_MERGES: usize = 49_152 - 256 - 2;
const WEIGHT_PREFIX: &str = "detector.backbone.language_backbone.";
const RTLD_NOW: c_int = 0x0002;
const RTLD_GLOBAL: c_int = 0x0100;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

fn load_cuda_dispatch_library() -> Result<(), String> {
    let library = CString::new("libtorch_cuda.so").expect("literal has no NUL");
    let handle = unsafe { dlopen(library.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
    if !handle.is_null() {
        return Ok(());
    }
    let detail = unsafe {
        let error = dlerror();
        if error.is_null() {
            "unknown dlopen error".to_string()
        } else {
            CStr::from_ptr(error).to_string_lossy().into_owned()
        }
    };
    Err(format!("load libtorch CUDA dispatch kernels: {detail}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticPrompt {
    pub key: &'static str,
    pub label: &'static str,
    pub text: &'static str,
}

/// A deliberately small review vocabulary.  Each entry is an independent
/// semantic question; detector queries within one answer remain object
/// candidates and must never be labeled as additional prompts.
pub const SEMANTIC_PROMPTS: [SemanticPrompt; 6] = [
    SemanticPrompt {
        key: "outer_iris",
        label: "OUTER IRIS DISK",
        text: "the complete dark circular iris disk surrounding the pupil, excluding the eyelids and sclera",
    },
    SemanticPrompt {
        key: "iris_annulus",
        label: "IRIS ANNULUS",
        text: "the colored iris annulus of the eye between the black pupil and the white sclera, completed behind the eyelids",
    },
    SemanticPrompt {
        key: "pupil",
        label: "PUPIL DISK",
        text: "the complete black pupil disk, including any portion hidden behind an eyelid, eyelashes, or a reflection",
    },
    SemanticPrompt {
        key: "sclera",
        label: "VISIBLE SCLERA",
        text: "the visible white sclera of the eye outside the iris and between the eyelids",
    },
    SemanticPrompt {
        key: "upper_eyelid",
        label: "UPPER EYELID",
        text: "the upper eyelid of the eye, excluding the eyebrow, iris, pupil, and sclera",
    },
    SemanticPrompt {
        key: "lower_eyelid",
        label: "LOWER EYELID",
        text: "the lower eyelid of the eye, excluding the cheek, iris, pupil, and sclera",
    },
];

/// Temporary prompt-engineering vocabulary for the offline limbus-arc trial.
/// The first question supplies the completed outer-disk hypothesis.  Every
/// other question is deliberately about directly visible material, so the
/// trial can compare one-, two-, and three-question evidence gates without
/// asking the language model to manufacture a hidden ellipse.
pub const ARC_TRIAL_PROMPTS: [SemanticPrompt; 9] = [
    SemanticPrompt {
        key: "outer_iris",
        label: "OUTER IRIS DISK",
        // Keep the anchor byte-for-byte identical to the live prompt. Prompt
        // trials must vary only the follow-on evidence questions, otherwise a
        // changed anchor confounds every comparison.
        text: "the complete dark circular iris disk surrounding the pupil, excluding the eyelids and sclera",
    },
    SemanticPrompt {
        key: "visible_iris",
        label: "VISIBLE IRIS MATERIAL",
        text: "only the clearly visible colored iris surface, excluding pupil, sclera, eyelids, eyelashes, glare, and reflection",
    },
    SemanticPrompt {
        key: "adjacent_sclera",
        label: "ADJACENT VISIBLE SCLERA",
        text: "the visible white sclera immediately beside the iris, excluding skin, eyelids, eyelashes, and reflection",
    },
    SemanticPrompt {
        key: "visible_limbus_arcs",
        label: "VISIBLE LIMBUS ARCS",
        text: "only the unobstructed visible iris to sclera boundary arcs, excluding eyelids, eyelashes, glasses, nose, and image border",
    },
    SemanticPrompt {
        key: "iris_occluders",
        label: "IRIS OCCLUDERS",
        text: "foreground eyelids, eyelashes, glasses, glare, and skin that cover or interrupt the iris boundary",
    },
    SemanticPrompt {
        key: "left_limbus_slice",
        label: "LEFT PIZZA CRUST",
        text: "the clearly visible left outer edge slice of the iris where colored iris meets white sclera",
    },
    SemanticPrompt {
        key: "right_limbus_slice",
        label: "RIGHT PIZZA CRUST",
        text: "the clearly visible right outer edge slice of the iris where colored iris meets white sclera",
    },
    SemanticPrompt {
        key: "upper_iris_occlusion",
        label: "UPPER IRIS OCCLUSION",
        text: "the upper eyelid and eyelashes directly covering the iris, excluding eyebrow, forehead, and sclera",
    },
    SemanticPrompt {
        key: "lower_iris_occlusion",
        label: "LOWER IRIS OCCLUSION",
        text: "the lower eyelid directly covering the iris, excluding cheek, jaw, nose, and sclera",
    },
];

#[derive(Debug)]
pub struct PromptBundle {
    /// Sequence first, as consumed by the SAM detector: `[32, prompts, 256]`.
    pub language_features: Tensor,
    /// True for padding, as consumed by PyTorch multi-head attention:
    /// `[prompts, 32]`.
    pub language_mask: Tensor,
    /// Retained for exact tokenizer and bundle provenance checks.
    pub token_ids: Tensor,
}

struct SimpleTokenizer {
    byte_encoder: [char; 256],
    encoder: HashMap<String, i64>,
    bpe_ranks: HashMap<(String, String), usize>,
    cache: HashMap<String, Vec<String>>,
    pattern: Regex,
    sot: i64,
    eot: i64,
}

impl SimpleTokenizer {
    fn load(path: &Path) -> Result<Self, String> {
        let byte_encoder = bytes_to_unicode();
        let source = File::open(path)
            .map_err(|error| format!("open SAM3 BPE vocabulary {}: {error}", path.display()))?;
        let mut lines = BufReader::new(GzDecoder::new(source)).lines();
        let _header = lines
            .next()
            .ok_or_else(|| format!("SAM3 BPE vocabulary {} is empty", path.display()))?
            .map_err(|error| format!("read SAM3 BPE header: {error}"))?;
        let mut merges = Vec::with_capacity(BPE_MERGES);
        for line in lines.take(BPE_MERGES) {
            let line = line.map_err(|error| format!("read SAM3 BPE merge: {error}"))?;
            let mut fields = line.split_whitespace();
            let Some(first) = fields.next() else { continue };
            let Some(second) = fields.next() else {
                continue;
            };
            merges.push((first.to_string(), second.to_string()));
        }
        if merges.len() != BPE_MERGES {
            return Err(format!(
                "SAM3 BPE vocabulary has {} usable merges; expected {BPE_MERGES}",
                merges.len()
            ));
        }

        // CLIP assigns vocabulary IDs in its historical `bs` order (visible
        // bytes first, then the remaining bytes), not numeric byte order.
        // `byte_encoder` is indexed numerically for fast token conversion, so
        // iterating that array here silently produces valid-looking but wrong
        // token IDs. Preserve the original order explicitly.
        let byte_order = clip_byte_order();
        let mut vocabulary = byte_order
            .iter()
            .map(|byte| byte_encoder[*byte as usize].to_string())
            .collect::<Vec<_>>();
        vocabulary.extend(
            byte_order
                .iter()
                .map(|byte| format!("{}</w>", byte_encoder[*byte as usize])),
        );
        vocabulary.extend(
            merges
                .iter()
                .map(|(first, second)| format!("{first}{second}")),
        );
        vocabulary.push("<start_of_text>".to_string());
        vocabulary.push("<end_of_text>".to_string());
        if vocabulary.len() != VOCAB_SIZE {
            return Err(format!(
                "SAM3 tokenizer constructed {} tokens; expected {VOCAB_SIZE}",
                vocabulary.len()
            ));
        }
        let encoder = vocabulary
            .into_iter()
            .enumerate()
            .map(|(index, token)| (token, index as i64))
            .collect::<HashMap<_, _>>();
        let bpe_ranks = merges
            .into_iter()
            .enumerate()
            .map(|(rank, pair)| (pair, rank))
            .collect();
        let pattern = Regex::new(
            r"(?i)<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+",
        )
        .map_err(|error| format!("compile SAM3 tokenizer expression: {error}"))?;
        let sot = *encoder
            .get("<start_of_text>")
            .ok_or("SAM3 tokenizer lacks start token")?;
        let eot = *encoder
            .get("<end_of_text>")
            .ok_or("SAM3 tokenizer lacks end token")?;
        Ok(Self {
            byte_encoder,
            encoder,
            bpe_ranks,
            cache: HashMap::new(),
            pattern,
            sot,
            eot,
        })
    }

    fn tokenize(&mut self, prompts: &[SemanticPrompt]) -> Result<Tensor, String> {
        let mut all = vec![0i64; prompts.len() * CONTEXT_LENGTH];
        for (prompt_index, prompt) in prompts.iter().enumerate() {
            let mut ids = Vec::with_capacity(CONTEXT_LENGTH);
            ids.push(self.sot);
            let clean = prompt
                .text
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            // Own the matches before mutating the BPE cache.
            let lexical = self
                .pattern
                .find_iter(&clean)
                .map(|matched| matched.as_str().to_string())
                .collect::<Vec<_>>();
            for token in lexical {
                let encoded = token
                    .as_bytes()
                    .iter()
                    .map(|byte| self.byte_encoder[*byte as usize])
                    .collect::<String>();
                for piece in self.bpe(&encoded) {
                    ids.push(*self.encoder.get(&piece).ok_or_else(|| {
                        format!(
                            "SAM3 BPE produced unknown token {piece:?} for {}",
                            prompt.key
                        )
                    })?);
                }
            }
            ids.push(self.eot);
            if ids.len() > CONTEXT_LENGTH {
                ids.truncate(CONTEXT_LENGTH);
                ids[CONTEXT_LENGTH - 1] = self.eot;
            }
            let offset = prompt_index * CONTEXT_LENGTH;
            all[offset..offset + ids.len()].copy_from_slice(&ids);
        }
        Ok(Tensor::from_slice(&all).view([prompts.len() as i64, CONTEXT_LENGTH as i64]))
    }

    fn bpe(&mut self, token: &str) -> Vec<String> {
        if let Some(cached) = self.cache.get(token) {
            return cached.clone();
        }
        let mut word = token
            .chars()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        loop {
            if word.len() < 2 {
                break;
            }
            let pairs = word
                .windows(2)
                .map(|pair| (pair[0].clone(), pair[1].clone()))
                .collect::<HashSet<_>>();
            let Some((first, second)) = pairs
                .into_iter()
                .filter_map(|pair| self.bpe_ranks.get(&pair).copied().map(|rank| (rank, pair)))
                .min_by_key(|(rank, _)| *rank)
                .map(|(_, pair)| pair)
            else {
                break;
            };
            let mut merged = Vec::with_capacity(word.len());
            let mut index = 0;
            while index < word.len() {
                if index + 1 < word.len() && word[index] == first && word[index + 1] == second {
                    merged.push(format!("{}{}", first, second));
                    index += 2;
                } else {
                    merged.push(word[index].clone());
                    index += 1;
                }
            }
            word = merged;
        }
        self.cache.insert(token.to_string(), word.clone());
        word
    }
}

fn bytes_to_unicode() -> [char; 256] {
    let mut bytes = clip_byte_order();
    let direct_count = bytes
        .iter()
        .take_while(|byte| matches!(**byte, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff))
        .count();
    let direct = bytes[..direct_count].to_vec();
    let direct_set = direct.iter().copied().collect::<HashSet<_>>();
    // `clip_byte_order` already contains the missing bytes after the visible
    // set; rebuild codepoints in exactly the same two phases as upstream.
    bytes.truncate(direct_count);
    let mut codepoints = direct.iter().map(|value| *value as u32).collect::<Vec<_>>();
    let mut next = 0u32;
    for byte in 0u16..=255 {
        let byte = byte as u8;
        if !direct_set.contains(&byte) {
            bytes.push(byte);
            codepoints.push(256 + next);
            next += 1;
        }
    }
    let mut result = ['\0'; 256];
    for (byte, codepoint) in bytes.into_iter().zip(codepoints) {
        result[byte as usize] = char::from_u32(codepoint).expect("CLIP byte codepoint");
    }
    result
}

fn clip_byte_order() -> Vec<u8> {
    let mut direct = Vec::with_capacity(256);
    direct.extend(b'!'..=b'~');
    direct.extend(0xa1..=0xac);
    direct.extend(0xae..=0xff);
    let direct_set = direct.iter().copied().collect::<HashSet<_>>();
    direct.extend(
        (0u16..=255)
            .map(|byte| byte as u8)
            .filter(|byte| !direct_set.contains(byte)),
    );
    direct
}

fn weight<'a>(weights: &'a HashMap<String, Tensor>, suffix: &str) -> Result<&'a Tensor, String> {
    weights
        .get(&format!("{WEIGHT_PREFIX}{suffix}"))
        .ok_or_else(|| format!("SAM3 checkpoint lacks text weight {suffix}"))
}

fn layer_norm(
    input: &Tensor,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
) -> Result<Tensor, String> {
    Ok(input.layer_norm(
        [WIDTH],
        Some(weight(weights, &format!("{prefix}.weight"))?),
        Some(weight(weights, &format!("{prefix}.bias"))?),
        1e-5,
        false,
    ))
}

/// Encode all review prompts directly from the official SAM3.1 checkpoint.
/// The returned tensors are CPU resident and safe to cache as a tiny bundle.
pub fn encode_semantic_prompts(checkpoint: &Path, bpe_path: &Path) -> Result<PromptBundle, String> {
    encode_prompt_set(checkpoint, bpe_path, &SEMANTIC_PROMPTS)
}

/// Encode a caller-selected static prompt set.  This is intentionally exposed
/// for bounded offline prompt trials; the live viewer continues to load only
/// `SEMANTIC_PROMPTS` and therefore cannot accidentally change semantics when
/// an experiment bundle is regenerated.
pub fn encode_prompt_set(
    checkpoint: &Path,
    bpe_path: &Path,
    prompts: &[SemanticPrompt],
) -> Result<PromptBundle, String> {
    if prompts.is_empty() {
        return Err("SAM3 prompt set must not be empty".to_string());
    }
    let compute_kind = if std::env::var_os("BUTTERCUP_SAM31_TEXT_BF16").is_some() {
        Kind::BFloat16
    } else {
        Kind::Float
    };
    let compute_device = if std::env::var_os("BUTTERCUP_SAM31_TEXT_CUDA").is_some() {
        load_cuda_dispatch_library()?;
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let loaded = Tensor::loadz_multi_with_device(checkpoint, Device::Cpu)
        .map_err(|error| format!("load SAM3.1 checkpoint {}: {error}", checkpoint.display()))?;
    let weights = loaded
        .into_iter()
        .filter(|(name, _)| name.starts_with(WEIGHT_PREFIX))
        .map(|(name, tensor)| {
            (
                name,
                tensor.to_device_(compute_device, compute_kind, false, false),
            )
        })
        .collect::<HashMap<_, _>>();
    if weights.len() < 24 * 12 + 5 {
        return Err(format!(
            "SAM3.1 checkpoint exposed only {} text tensors",
            weights.len()
        ));
    }

    let mut tokenizer = SimpleTokenizer::load(bpe_path)?;
    let token_ids = tokenizer.tokenize(prompts)?;
    let compute_token_ids = token_ids.to_device(compute_device);
    let batch = prompts.len() as i64;
    let flat_ids = compute_token_ids.view([-1]);
    let mut hidden = weight(&weights, "encoder.token_embedding.weight")?
        .index_select(0, &flat_ids)
        .view([batch, CONTEXT_LENGTH as i64, WIDTH]);
    hidden += weight(&weights, "encoder.positional_embedding")?.unsqueeze(0);
    for layer in 0..LAYERS {
        let prefix = format!("encoder.transformer.resblocks.{layer}");
        let normalized = layer_norm(&hidden, &weights, &format!("{prefix}.ln_1"))?;
        let qkv = normalized.linear(
            weight(&weights, &format!("{prefix}.attn.in_proj_weight"))?,
            Some(weight(&weights, &format!("{prefix}.attn.in_proj_bias"))?),
        );
        let chunks = qkv.chunk(3, -1);
        let reshape = |value: &Tensor| {
            value
                .view([batch, CONTEXT_LENGTH as i64, HEADS, HEAD_WIDTH])
                .transpose(1, 2)
        };
        let query = reshape(&chunks[0]);
        let key = reshape(&chunks[1]);
        let value = reshape(&chunks[2]);
        let attention = Tensor::scaled_dot_product_attention(
            &query,
            &key,
            &value,
            None::<&Tensor>,
            0.0,
            true,
            None,
            false,
        )
        .transpose(1, 2)
        .contiguous()
        .view([batch, CONTEXT_LENGTH as i64, WIDTH])
        .linear(
            weight(&weights, &format!("{prefix}.attn.out_proj.weight"))?,
            Some(weight(&weights, &format!("{prefix}.attn.out_proj.bias"))?),
        );
        hidden += attention;

        let normalized = layer_norm(&hidden, &weights, &format!("{prefix}.ln_2"))?;
        let feed_forward = normalized
            .linear(
                weight(&weights, &format!("{prefix}.mlp.c_fc.weight"))?,
                Some(weight(&weights, &format!("{prefix}.mlp.c_fc.bias"))?),
            )
            .gelu("none")
            .linear(
                weight(&weights, &format!("{prefix}.mlp.c_proj.weight"))?,
                Some(weight(&weights, &format!("{prefix}.mlp.c_proj.bias"))?),
            );
        hidden += feed_forward;
    }

    hidden = layer_norm(&hidden, &weights, "encoder.ln_final")?;
    let resized = hidden.linear(
        weight(&weights, "resizer.weight")?,
        Some(weight(&weights, "resizer.bias")?),
    );
    Ok(PromptBundle {
        language_features: resized.transpose(0, 1).contiguous().to_device_(
            Device::Cpu,
            Kind::BFloat16,
            false,
            false,
        ),
        language_mask: token_ids.eq(0),
        token_ids,
    })
}

pub fn save_prompt_bundle(bundle: &PromptBundle, path: &Path) -> Result<(), String> {
    Tensor::save_multi(
        &[
            ("language_features", &bundle.language_features),
            ("language_mask", &bundle.language_mask),
            ("token_ids", &bundle.token_ids),
        ],
        path,
    )
    .map_err(|error| {
        format!(
            "save SAM3 semantic prompt bundle {}: {error}",
            path.display()
        )
    })
}

#[allow(dead_code)]
pub fn load_prompt_bundle(path: &Path) -> Result<PromptBundle, String> {
    let mut tensors = Tensor::load_multi_with_device(path, Device::Cpu)
        .map_err(|error| format!("load SAM3 prompt bundle {}: {error}", path.display()))?
        .into_iter()
        .collect::<HashMap<_, _>>();
    let language_features = tensors
        .remove("language_features")
        .ok_or_else(|| "SAM3 prompt bundle lacks language_features".to_string())?;
    let language_mask = tensors
        .remove("language_mask")
        .ok_or_else(|| "SAM3 prompt bundle lacks language_mask".to_string())?;
    let token_ids = tensors
        .remove("token_ids")
        .ok_or_else(|| "SAM3 prompt bundle lacks token_ids".to_string())?;
    let prompt_count = SEMANTIC_PROMPTS.len() as i64;
    if language_features.size() != [CONTEXT_LENGTH as i64, prompt_count, 256]
        || language_mask.size() != [prompt_count, CONTEXT_LENGTH as i64]
        || token_ids.size() != [prompt_count, CONTEXT_LENGTH as i64]
    {
        return Err(format!(
            "SAM3 prompt bundle has incompatible shapes: features={:?} mask={:?} tokens={:?}",
            language_features.size(),
            language_mask.size(),
            token_ids.size()
        ));
    }
    Ok(PromptBundle {
        language_features,
        language_mask,
        token_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_prompt_keys_are_unique_and_outer_iris_stays_first() {
        let keys = SEMANTIC_PROMPTS
            .iter()
            .map(|prompt| prompt.key)
            .collect::<HashSet<_>>();
        assert_eq!(keys.len(), SEMANTIC_PROMPTS.len());
        assert_eq!(SEMANTIC_PROMPTS[0].key, "outer_iris");
    }
}
