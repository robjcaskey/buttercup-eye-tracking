use super::*;

fn source() -> SourceVersion {
    SourceVersion {
        revision: "a".repeat(40),
        tree_sha256: format!("{:x}", Sha256::digest(b"source")),
    }
}

fn node(id: &str, kind: Kind, parents: &[&str]) -> Node {
    Node {
        id: id.into(),
        kind,
        sha256: Some(format!("{:x}", Sha256::digest(id.as_bytes()))),
        planned: false,
        dependencies: parents.iter().map(|p| p.to_string()).collect(),
        refinement: None,
    }
}

fn fixture() -> Manifest {
    Manifest {
        schema: SCHEMA.into(),
        source: source(),
        targets: vec!["base".into()],
        nodes: vec![
            node("source", Kind::Source, &[]),
            node("raw", Kind::Raw, &[]),
            node("labels", Kind::HumanLabels, &[]),
            node("sam3", Kind::Sam3, &[]),
            node("masks", Kind::DerivedData, &["source", "raw", "sam3"]),
            node("base", Kind::CustomModel, &["masks", "labels", "source"]),
        ],
    }
}

fn refinement() -> Node {
    let mut n = node(
        "local",
        Kind::LastMileRefinement,
        &["base", "raw", "source"],
    );
    n.refinement = Some(Refinement {
        scope: Scope::User,
        scope_key: "test-user".into(),
        base_model: "base".into(),
        training_device: "cpu".into(),
        inference_device: "cpu".into(),
        max_samples: 16,
        max_update_ms: 100,
    });
    n
}

fn failure(m: &Manifest, code: &str) -> GraphError {
    let error = validate(m, &source()).unwrap_err();
    assert_eq!(error.code, code, "{error:?}");
    error
}

#[test]
fn sam3_root_and_shared_ancestor_diamond_have_a_topological_certificate() {
    let m = fixture();
    let cert = validate(&m, &source()).unwrap();
    let rank: BTreeMap<_, _> = cert
        .topological_order
        .iter()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    for node in &m.nodes {
        for parent in &node.dependencies {
            assert!(rank[parent] < rank[&node.id]);
        }
    }
    assert_eq!(cert.maximum_dependency_depth, 2);
    assert_eq!(cert.scope, "declared_dependency_graph_only");
}

#[test]
fn rejects_self_dependency_with_closed_cycle_witness() {
    let mut m = fixture();
    m.nodes[5].dependencies.push("base".into());
    assert_eq!(failure(&m, "dependency_cycle").cycle, ["base", "base"]);
}

#[test]
fn rejects_cycle_through_cached_masks_features_and_selector() {
    let mut m = fixture();
    m.nodes[4].dependencies.push("selector".into());
    m.nodes.push(node("features", Kind::Features, &["base"]));
    m.nodes
        .push(node("selector", Kind::Selector, &["features"]));
    let cycle = failure(&m, "dependency_cycle").cycle;
    assert_eq!(cycle.first(), cycle.last());
    for id in ["masks", "selector", "features", "base"] {
        assert!(cycle.iter().any(|entry| entry == id));
    }
}

#[test]
fn accepts_versioned_warm_start_but_not_retroactive_ancestry() {
    let mut m = fixture();
    m.nodes.push(node(
        "pseudo-v1",
        Kind::DerivedData,
        &["raw", "base", "source"],
    ));
    m.nodes.push(node(
        "model-v2",
        Kind::CustomModel,
        &["base", "pseudo-v1", "source"],
    ));
    m.targets = vec!["model-v2".into()];
    validate(&m, &source()).unwrap();
    m.nodes[5].dependencies.push("model-v2".into());
    failure(&m, "dependency_cycle");
}

#[test]
fn rejects_disconnected_cycle_even_outside_selected_targets() {
    let mut m = fixture();
    m.nodes
        .push(node("hidden-a", Kind::DerivedData, &["hidden-b"]));
    m.nodes
        .push(node("hidden-b", Kind::Features, &["hidden-a"]));
    failure(&m, "dependency_cycle");
}

#[test]
fn missing_dependencies_do_not_become_external_roots() {
    let mut m = fixture();
    m.nodes[5].dependencies.push("forgotten-checkpoint".into());
    failure(&m, "missing_dependency");
    m.nodes[5].dependencies.clear();
    failure(&m, "bootstrap_root");
}

#[test]
fn digest_aliases_cannot_relabel_custom_models_as_trusted_roots() {
    let mut m = fixture();
    let mut alias = node("innocent-cache", Kind::Raw, &[]);
    alias.sha256 = m.nodes[5].sha256.clone();
    m.nodes.push(alias);
    failure(&m, "artifact_alias");
}

#[test]
fn root_with_derived_parents_is_rejected() {
    let mut m = fixture();
    m.nodes[1].dependencies.push("base".into());
    failure(&m, "bootstrap_root");
}

#[test]
fn strict_schema_rejects_duplicate_fields_unknown_kinds_and_unknown_fields() {
    let bytes = serde_json::to_vec(&fixture()).unwrap();
    parse(&bytes).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    for broken in [
        text.replacen("\"schema\":", "\"schema\":\"discarded\",\"schema\":", 1),
        text.replacen(
            "\"kind\":\"sam3\"",
            "\"kind\":\"arbitrary_external_model\"",
            1,
        ),
        text.replacen("\"nodes\":", "\"hidden_teacher\":\"x\",\"nodes\":", 1),
    ] {
        assert!(parse(broken.as_bytes()).is_err());
    }
}

#[test]
fn stale_checkout_and_bad_source_digest_fail_closed() {
    let mut m = fixture();
    m.source.revision = "b".repeat(40);
    failure(&m, "source_mismatch");
    m.source = source();
    m.nodes[0].sha256 = Some("c".repeat(64));
    failure(&m, "source_mismatch");
}

#[test]
fn planned_artifacts_are_explicit_and_cannot_be_roots() {
    let mut m = fixture();
    m.nodes[5].planned = true;
    m.nodes[5].sha256 = None;
    let cert = validate(&m, &source()).unwrap();
    assert_eq!(cert.planned_nodes, ["base"]);
    m.nodes[1].planned = true;
    m.nodes[1].sha256 = None;
    failure(&m, "artifact_identity");
}

#[test]
fn duplicate_ids_edges_unknown_targets_and_invalid_hashes_fail() {
    let mut m = fixture();
    m.nodes.push(m.nodes[0].clone());
    failure(&m, "node_identity");
    let mut m = fixture();
    m.nodes[5].dependencies.push("source".into());
    failure(&m, "duplicate_edge");
    let mut m = fixture();
    m.targets.push("unknown".into());
    failure(&m, "targets");
    let mut m = fixture();
    m.nodes[2].sha256 = Some("not-a-hash".into());
    failure(&m, "artifact_identity");
}

#[test]
fn local_refinement_is_cpu_only_scoped_and_bounded() {
    let mut m = fixture();
    m.nodes.push(refinement());
    m.targets = vec!["local".into()];
    validate(&m, &source()).unwrap();
    for (training, inference, samples, budget) in [
        ("cuda", "cpu", 16, 100),
        ("cpu", "cuda", 16, 100),
        ("cpu", "cpu", 0, 100),
        ("cpu", "cpu", 16, 0),
    ] {
        let p = m.nodes.last_mut().unwrap().refinement.as_mut().unwrap();
        p.training_device = training.into();
        p.inference_device = inference.into();
        p.max_samples = samples;
        p.max_update_ms = budget;
        failure(&m, "refinement_policy");
    }
}

#[test]
fn local_refinement_cannot_leak_back_into_a_shared_base_through_features() {
    let mut m = fixture();
    m.nodes.push(refinement());
    m.nodes
        .push(node("local-features", Kind::Features, &["local"]));
    m.nodes.push(node(
        "future-base",
        Kind::CustomModel,
        &["raw", "source", "local-features"],
    ));
    // This is structurally acyclic but violates the last-mile boundary.
    failure(&m, "local_refinement_leakage");
}

#[test]
fn optional_refinement_predictions_may_be_evaluated_without_retraining_base() {
    let mut m = fixture();
    m.nodes.push(refinement());
    m.nodes.push(node(
        "review",
        Kind::Evaluation,
        &["base", "local", "labels"],
    ));
    m.targets = vec!["review".into()];
    validate(&m, &source()).unwrap();
}

#[test]
fn models_require_current_source_and_raw_ancestry() {
    let mut m = fixture();
    m.nodes[5].dependencies = vec!["sam3".into()];
    failure(&m, "model_roots");
}

#[test]
fn long_sparse_chain_does_not_recurse_on_the_thread_stack() {
    let mut m = fixture();
    let mut parent = "base".to_string();
    for i in 0..4000 {
        let id = format!("cache-{i}");
        m.nodes.push(node(&id, Kind::DerivedData, &[&parent]));
        parent = id;
    }
    m.targets = vec![parent];
    assert_eq!(
        validate(&m, &source()).unwrap().maximum_dependency_depth,
        4002
    );
}

#[test]
fn refinements_cannot_import_another_scope_through_features() {
    let mut m = fixture();
    m.nodes.push(refinement());
    m.nodes
        .push(node("local-features", Kind::Features, &["local"]));
    let mut other = refinement();
    other.id = "other".into();
    other.sha256 = node("other", Kind::LastMileRefinement, &[]).sha256;
    other.dependencies.push("local-features".into());
    other.refinement.as_mut().unwrap().scope_key = "another-user".into();
    m.nodes.push(other);
    failure(&m, "cross_scope_refinement");
    // Even identical keys cannot cross the user/scenario boundary.
    let p = m.nodes.last_mut().unwrap().refinement.as_mut().unwrap();
    p.scope_key = "test-user".into();
    p.scope = Scope::Scenario;
    failure(&m, "cross_scope_refinement");
}

#[test]
fn same_scope_refinement_can_reuse_its_own_versioned_features() {
    let mut m = fixture();
    m.nodes.push(refinement());
    m.nodes
        .push(node("local-features", Kind::Features, &["local"]));
    let mut next = refinement();
    next.id = "local-v2".into();
    next.sha256 = node("local-v2", Kind::LastMileRefinement, &[]).sha256;
    next.dependencies.push("local-features".into());
    m.nodes.push(next);
    m.targets = vec!["local-v2".into()];
    validate(&m, &source()).unwrap();
}

struct TemporarySource(std::path::PathBuf);

impl TemporarySource {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let root = std::fs::canonicalize("outputs").unwrap();
        assert!(root.starts_with("/mnt/bulk_data/buttercup-eye-tracking"));
        let path = root.join(format!(
            "bootstrap-source-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let fixture = Self(path);
        fixture.git(&["init", "-q"]);
        std::fs::create_dir(fixture.0.join("sub")).unwrap();
        std::fs::write(fixture.0.join("top.rs"), "initial source\n").unwrap();
        std::fs::write(fixture.0.join("sub/leaf.rs"), "leaf\n").unwrap();
        std::fs::write(fixture.0.join(".gitignore"), "runtime/\n").unwrap();
        fixture.git(&["add", "."]);
        fixture.git(&[
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ]);
        fixture
    }

    fn git(&self, args: &[&str]) {
        let result = Command::new("git")
            .arg("-C")
            .arg(&self.0)
            .args(args)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

impl Drop for TemporarySource {
    fn drop(&mut self) {
        // Only remove the specific unique fixture created above, never inputs.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn source_stamp_covers_whole_checkout_and_dirty_untracked_deleted_files() {
    let f = TemporarySource::new();
    let initial = current_source(&f.0).unwrap();
    assert_eq!(initial, current_source(&f.0.join("sub")).unwrap());
    std::fs::write(f.0.join("top.rs"), "dirty source\n").unwrap();
    let dirty = current_source(&f.0.join("sub")).unwrap();
    assert_eq!(initial.revision, dirty.revision);
    assert_ne!(initial.tree_sha256, dirty.tree_sha256);
    std::fs::write(f.0.join("new.rs"), "untracked\n").unwrap();
    let untracked = current_source(&f.0).unwrap();
    assert_ne!(dirty.tree_sha256, untracked.tree_sha256);
    std::fs::remove_file(f.0.join("top.rs")).unwrap();
    assert_ne!(
        untracked.tree_sha256,
        current_source(&f.0).unwrap().tree_sha256
    );
}

#[test]
fn source_stamp_tracks_executable_bits_and_links_without_following_runtime() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let f = TemporarySource::new();
    let initial = current_source(&f.0).unwrap();
    let path = f.0.join("top.rs");
    let mut permissions = std::fs::metadata(&path).unwrap().permissions();
    permissions.set_mode(permissions.mode() ^ 0o100);
    std::fs::set_permissions(path, permissions).unwrap();
    assert_ne!(
        initial.tree_sha256,
        current_source(&f.0).unwrap().tree_sha256
    );
    std::fs::create_dir(f.0.join("runtime")).unwrap();
    std::fs::write(f.0.join("runtime/raw"), "initial raw").unwrap();
    symlink("runtime/raw", f.0.join("data")).unwrap();
    let linked = current_source(&f.0).unwrap();
    std::fs::write(f.0.join("runtime/raw"), "changed raw").unwrap();
    assert_eq!(linked, current_source(&f.0).unwrap());
    std::fs::remove_file(f.0.join("data")).unwrap();
    symlink("runtime/other", f.0.join("data")).unwrap();
    assert_ne!(
        linked.tree_sha256,
        current_source(&f.0).unwrap().tree_sha256
    );
}
