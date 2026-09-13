//! CPU-only structural provenance checks. A valid declared DAG is necessary,
//! not sufficient, evidence for a successful, truthful cold bootstrap.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::Path;
use std::process::Command;

pub const SCHEMA: &str = "buttercup-bootstrap-graph-v1";
const MAX_NODES: usize = 10_000;
const MAX_EDGES: usize = 100_000;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceVersion {
    pub revision: String,
    pub tree_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Raw,
    HumanLabels,
    Measurements,
    Sam3,
    Source,
    DerivedData,
    Features,
    Selector,
    CustomModel,
    LastMileRefinement,
    Export,
    Evaluation,
}

impl Kind {
    fn root(self) -> bool {
        matches!(
            self,
            Self::Raw | Self::HumanLabels | Self::Measurements | Self::Sam3 | Self::Source
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    User,
    Scenario,
}

#[derive(Clone, PartialEq, Eq)]
enum ScopeAncestry {
    Unscoped,
    Single(Scope, String),
    Mixed,
}

impl ScopeAncestry {
    fn merge(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Unscoped, value) | (value, Self::Unscoped) => value.clone(),
            (a, b) if a == b => a.clone(),
            _ => Self::Mixed,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Refinement {
    pub scope: Scope,
    pub scope_key: String,
    pub base_model: String,
    pub training_device: String,
    pub inference_device: String,
    pub max_samples: u64,
    pub max_update_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub id: String,
    pub kind: Kind,
    pub sha256: Option<String>,
    #[serde(default)]
    pub planned: bool,
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub refinement: Option<Refinement>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub source: SourceVersion,
    pub targets: Vec<String>,
    pub nodes: Vec<Node>,
}

#[derive(Debug, Serialize)]
pub struct GraphError {
    pub code: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cycle: Vec<String>,
}

impl GraphError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            cycle: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Certificate {
    pub scope: &'static str,
    pub source: SourceVersion,
    pub topological_order: Vec<String>,
    pub targets: Vec<String>,
    pub roots: Vec<String>,
    pub planned_nodes: Vec<String>,
    pub nodes: usize,
    pub edges: usize,
    pub maximum_dependency_depth: usize,
    pub limitation: &'static str,
}

fn hash_text_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_.:-".contains(&c))
}

pub fn parse(bytes: &[u8]) -> Result<Manifest, GraphError> {
    if bytes.len() > 8 * 1024 * 1024 {
        return Err(GraphError::new("manifest_limit", "manifest exceeds 8 MiB"));
    }
    // Typed structs reject duplicate fields as well as unknown fields. Do not
    // first parse into Value, which could silently replace duplicate keys.
    serde_json::from_slice(bytes).map_err(|e| GraphError::new("manifest_schema", e.to_string()))
}

pub fn validate(manifest: &Manifest, current: &SourceVersion) -> Result<Certificate, GraphError> {
    if manifest.schema != SCHEMA {
        return Err(GraphError::new(
            "manifest_schema",
            "unknown bootstrap graph schema",
        ));
    }
    if !hash_text_valid(&manifest.source.tree_sha256) || manifest.source != *current {
        return Err(GraphError::new(
            "source_mismatch",
            "graph is not bound to the current checkout revision/tree",
        ));
    }
    if manifest.nodes.is_empty() || manifest.nodes.len() > MAX_NODES {
        return Err(GraphError::new(
            "graph_limit",
            "graph requires 1..10000 nodes",
        ));
    }
    let mut by_id = BTreeMap::new();
    let mut by_hash = BTreeMap::new();
    let mut edges = 0;
    for (i, node) in manifest.nodes.iter().enumerate() {
        if !valid_id(&node.id) || by_id.insert(node.id.as_str(), i).is_some() {
            return Err(GraphError::new(
                "node_identity",
                format!("invalid or duplicate node id: {}", node.id),
            ));
        }
        edges += node.dependencies.len();
        if edges > MAX_EDGES {
            return Err(GraphError::new(
                "graph_limit",
                "graph exceeds 100000 dependency edges",
            ));
        }
        match (&node.sha256, node.planned) {
            (Some(hash), false) if hash_text_valid(hash) => {
                if let Some(prior) = by_hash.insert(hash.as_str(), node.id.as_str()) {
                    return Err(GraphError::new(
                        "artifact_alias",
                        format!(
                            "{} and {} name the same content; use one canonical artifact node",
                            prior, node.id
                        ),
                    ));
                }
            }
            (None, true) if !node.kind.root() => (),
            _ => {
                return Err(GraphError::new(
                    "artifact_identity",
                    format!(
                        "{} needs a SHA-256, or must be an explicitly planned non-root artifact",
                        node.id
                    ),
                ))
            }
        }
        if node.kind.root() != node.dependencies.is_empty() {
            return Err(GraphError::new("bootstrap_root", format!("{}: only permitted roots may have no dependencies; roots cannot hide derived parents", node.id)));
        }
        if node.kind == Kind::Source && node.sha256.as_deref() != Some(current.tree_sha256.as_str())
        {
            return Err(GraphError::new(
                "source_mismatch",
                "source root digest differs from checkout fingerprint",
            ));
        }
        if (node.kind == Kind::LastMileRefinement) != node.refinement.is_some() {
            return Err(GraphError::new(
                "refinement_policy",
                format!("{} has missing or misplaced refinement policy", node.id),
            ));
        }
    }
    let mut targets = BTreeSet::new();
    if manifest.targets.is_empty()
        || manifest
            .targets
            .iter()
            .any(|id| !by_id.contains_key(id.as_str()) || !targets.insert(id))
    {
        return Err(GraphError::new(
            "targets",
            "targets must be nonempty, known and unique",
        ));
    }
    let mut parents = Vec::with_capacity(manifest.nodes.len());
    for node in &manifest.nodes {
        let mut seen = BTreeSet::new();
        let mut indices = Vec::new();
        for id in &node.dependencies {
            let index = by_id.get(id.as_str()).ok_or_else(|| {
                GraphError::new(
                    "missing_dependency",
                    format!("{} requires undeclared {}", node.id, id),
                )
            })?;
            if !seen.insert(id) {
                return Err(GraphError::new(
                    "duplicate_edge",
                    format!("{} declares {} more than once", node.id, id),
                ));
            }
            indices.push(*index);
        }
        parents.push(indices);
    }

    // Iterative three-color DFS: grey means on this exact path, black means
    // fully proven. Shared ancestors in a diamond are not cycles. All nodes
    // are checked, including disconnected components outside target closure.
    let n = manifest.nodes.len();
    let mut color = vec![0u8; n];
    let mut positions = vec![0usize; n];
    let mut path = Vec::new();
    let mut stack = Vec::new();
    let mut order = Vec::with_capacity(n);
    for start in 0..n {
        if color[start] != 0 {
            continue;
        }
        positions[start] = path.len();
        path.push(start);
        color[start] = 1;
        stack.push((start, 0usize));
        while let Some(&(node, next)) = stack.last() {
            if next == parents[node].len() {
                stack.pop();
                path.pop();
                color[node] = 2;
                order.push(node);
                continue;
            }
            stack.last_mut().unwrap().1 += 1;
            let parent = parents[node][next];
            match color[parent] {
                0 => {
                    color[parent] = 1;
                    positions[parent] = path.len();
                    path.push(parent);
                    stack.push((parent, 0));
                }
                1 => {
                    let mut cycle: Vec<String> = path[positions[parent]..]
                        .iter()
                        .map(|&i| manifest.nodes[i].id.clone())
                        .collect();
                    cycle.push(manifest.nodes[parent].id.clone());
                    return Err(GraphError {
                        code: "dependency_cycle",
                        message: format!("dependency cycle: {}", cycle.join(" -> ")),
                        cycle,
                    });
                }
                _ => (),
            }
        }
    }

    let mut depth = vec![0; n];
    let mut local_ancestor = vec![false; n];
    let mut source_ancestor = vec![false; n];
    let mut raw_ancestor = vec![false; n];
    // Bounded per-node state, not an O(N^2) transitive set of user identities.
    let mut scopes = vec![ScopeAncestry::Unscoped; n];
    for &i in &order {
        let node = &manifest.nodes[i];
        depth[i] = parents[i].iter().map(|&p| depth[p] + 1).max().unwrap_or(0);
        local_ancestor[i] =
            node.kind == Kind::LastMileRefinement || parents[i].iter().any(|&p| local_ancestor[p]);
        source_ancestor[i] =
            node.kind == Kind::Source || parents[i].iter().any(|&p| source_ancestor[p]);
        raw_ancestor[i] = node.kind == Kind::Raw || parents[i].iter().any(|&p| raw_ancestor[p]);
        for &parent in &parents[i] {
            scopes[i] = scopes[i].merge(&scopes[parent]);
        }
        if matches!(node.kind, Kind::CustomModel | Kind::LastMileRefinement)
            && !(source_ancestor[i] && raw_ancestor[i])
        {
            return Err(GraphError::new(
                "model_roots",
                format!("{} must trace to current source and native RAW", node.id),
            ));
        }
        if node.kind == Kind::CustomModel && local_ancestor[i] {
            return Err(GraphError::new("local_refinement_leakage", format!("{} depends on a local refinement, directly or through derived data; the shared base must remain independent", node.id)));
        }
        if let Some(policy) = &node.refinement {
            let base = by_id.get(policy.base_model.as_str()).copied();
            if policy.training_device != "cpu"
                || policy.inference_device != "cpu"
                || policy.scope_key.is_empty()
                || policy.scope_key.len() > 128
                || policy.max_samples == 0
                || policy.max_update_ms == 0
                || !base.is_some_and(|b| {
                    parents[i].contains(&b)
                        && matches!(manifest.nodes[b].kind, Kind::CustomModel | Kind::Sam3)
                })
            {
                return Err(GraphError::new("refinement_policy", format!("{} requires CPU-only training/inference, positive budgets, a scope key and a direct shared-base/SAM3 dependency", node.id)));
            }
            let own_scope = ScopeAncestry::Single(policy.scope, policy.scope_key.clone());
            if scopes[i] != ScopeAncestry::Unscoped && scopes[i] != own_scope {
                return Err(GraphError::new("cross_scope_refinement", format!("{} imports another user/scenario refinement, possibly through derived data", node.id)));
            }
            scopes[i] = own_scope;
        }
    }
    Ok(Certificate {
        scope: "declared_dependency_graph_only",
        source: current.clone(),
        topological_order: order.iter().map(|&i| manifest.nodes[i].id.clone()).collect(),
        targets: manifest.targets.clone(),
        roots: manifest.nodes.iter().filter(|n| n.kind.root()).map(|n| n.id.clone()).collect(),
        planned_nodes: manifest.nodes.iter().filter(|n| n.planned).map(|n| n.id.clone()).collect(),
        nodes: n, edges, maximum_dependency_depth: depth.into_iter().max().unwrap_or(0),
        limitation: "Checks declared identities, dependencies and policy only; does not verify material bytes, undeclared reads, successful regeneration, numerical accuracy or onboarding quality.",
    })
}

/// Content stamp of Git-known source paths plus non-ignored untracked paths.
/// Symlinks are hashed as links, never followed into captures/runtime trees.
pub fn current_source(repo: &Path) -> Result<SourceVersion, String> {
    // `git ls-files` invoked below a worktree root only covers that subtree.
    // Always fingerprint the whole checkout, even when called from src/bin.
    let root = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|e| e.to_string())?;
    if !root.status.success() {
        return Err(String::from_utf8_lossy(&root.stderr).into_owned());
    }
    let root = String::from_utf8(root.stdout).map_err(|e| e.to_string())?;
    let repo = Path::new(root.trim_end_matches('\n'));
    let git = |args: &[&str]| -> Result<Vec<u8>, String> {
        let result = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .output()
            .map_err(|e| e.to_string())?;
        if !result.status.success() {
            return Err(String::from_utf8_lossy(&result.stderr).into_owned());
        }
        Ok(result.stdout)
    };
    let revision = String::from_utf8(git(&["rev-parse", "HEAD"])?)
        .map_err(|e| e.to_string())?
        .trim()
        .to_owned();
    let paths = git(&[
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ])?;
    let paths = String::from_utf8(paths).map_err(|e| e.to_string())?;
    let paths: BTreeSet<_> = paths.split('\0').filter(|s| !s.is_empty()).collect();
    let mut digest = Sha256::new();
    digest.update(b"buttercup-source-tree-v1\0");
    for name in paths {
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        let path = repo.join(name);
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(value) => value,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                digest.update(b"missing\0");
                continue;
            }
            Err(e) => return Err(e.to_string()),
        };
        if metadata.file_type().is_symlink() {
            use std::os::unix::ffi::OsStrExt;
            let target = std::fs::read_link(path).map_err(|e| e.to_string())?;
            let bytes = target.as_os_str().as_bytes();
            digest.update(b"link\0");
            digest.update((bytes.len() as u64).to_le_bytes());
            digest.update(bytes);
        } else if metadata.is_file() {
            use std::os::unix::fs::PermissionsExt;
            digest.update(b"file\0");
            digest.update((metadata.permissions().mode() & 0o111).to_le_bytes());
            let mut content = Sha256::new();
            let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
            let mut buffer = [0u8; 65536];
            loop {
                let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
                if count == 0 {
                    break;
                }
                content.update(&buffer[..count]);
            }
            digest.update(content.finalize());
        } else {
            return Err(format!("unsupported source entry: {name}"));
        }
    }
    Ok(SourceVersion {
        revision,
        tree_sha256: format!("{:x}", digest.finalize()),
    })
}

#[cfg(test)]
#[path = "bootstrapability/tests.rs"]
mod tests;
