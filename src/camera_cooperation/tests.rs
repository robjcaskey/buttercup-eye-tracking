use super::*;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("outputs").canonicalize().unwrap()
            .join(format!("upc-test-{}", unique().unwrap()));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap();
        Self(path)
    }
    fn root(&self) -> PathBuf { self.0.join("runtime") }
    fn endpoints(&self) -> (SocketAddr, SocketAddr) {
        ("127.0.0.1:5001".parse().unwrap(), "127.0.0.1:5002".parse().unwrap())
    }
    fn setup(&self) -> Dir {
        for path in [self.root(), self.root().join("cameras"),
            self.root().join("cameras/test-camera"),
            self.root().join("cameras/test-camera/requests")] {
            fs::DirBuilder::new().mode(0o700).create(path).unwrap();
        }
        let root = Dir::open(&self.root()).unwrap();
        let mut version = root.open_file("version", libc::O_CREAT | libc::O_WRONLY).unwrap();
        version.write_all(b"1\n").unwrap();
        let dir = root.child("cameras").unwrap().child("test-camera").unwrap();
        dir.open_file("control.lock", libc::O_CREAT | libc::O_RDWR).unwrap();
        dir.publish("descriptor.json", &json!({
            "schema":"upc.camera.v1", "camera_id":"test-camera",
            "attachment_id":"attachment-test", "publisher_instance":"publisher-test",
            "present":true,"device_paths":[],"ttl_ms":15000,
            "transport":{"kind":"tcp","protocol":"podbay-raw-v1",
                "host":"127.0.0.1","port":5001,"control_port":5002}
        })).unwrap();
        dir
    }
    fn acquire(&self, wait: Duration) -> io::Result<Session> {
        let (capture, control) = self.endpoints();
        Session::acquire(self.root(), capture, control, wait)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); }
}

#[test]
fn absent_runtime_is_standalone_but_appearance_is_not_bypassed() {
    let f = Fixture::new();
    let session = f.acquire(Duration::ZERO).unwrap();
    assert!(session.lease.is_none());
    assert!(session.check(Some(f.endpoints().0)).is_ok());
    assert!(!f.root().exists());
    f.setup();
    assert!(session.check(None).is_err());
}

#[test]
fn valid_session_retains_lock_and_releases_without_replacing_inode() {
    let f = Fixture::new(); let dir = f.setup();
    let competitor = dir.open_file("control.lock", libc::O_RDWR).unwrap();
    let before = competitor.metadata().unwrap().ino();
    let session = f.acquire(Duration::ZERO).unwrap();
    assert!(session.lease.is_some());
    assert!(session.check(Some(f.endpoints().0)).is_ok());
    assert!(session.check(Some(f.endpoints().1)).is_ok());
    assert!(session.check(Some("127.0.0.1:5003".parse().unwrap())).is_err());
    assert_eq!(unsafe { libc::flock(competitor.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) }, -1);
    assert!(dir.child("requests").unwrap().entries().unwrap().is_empty());
    drop(session);
    assert_eq!(unsafe { libc::flock(competitor.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) }, 0);
    assert_eq!(dir.open_file("control.lock", libc::O_RDWR).unwrap().metadata().unwrap().ino(), before);
    let owner: Value = serde_json::from_slice(&dir.read("owner.json").unwrap().0).unwrap();
    assert_eq!(owner["state"], "released");
}

#[test]
fn held_lock_is_not_stolen_and_own_request_is_cleaned_on_timeout() {
    let f = Fixture::new(); let dir = f.setup();
    let held = dir.open_file("control.lock", libc::O_RDWR).unwrap();
    assert_eq!(unsafe { libc::flock(held.as_raw_fd(), libc::LOCK_EX) }, 0);
    assert!(f.acquire(Duration::from_millis(50)).is_err());
    assert!(dir.child("requests").unwrap().entries().unwrap().is_empty());
    assert!(dir.read("owner.json").is_err());
}

#[test]
fn invalid_runtime_and_descriptor_never_fall_back_to_standalone() {
    let f = Fixture::new(); let dir = f.setup();
    fs::set_permissions(f.root(), fs::Permissions::from_mode(0o755)).unwrap();
    assert!(f.acquire(Duration::ZERO).is_err());
    fs::set_permissions(f.root(), fs::Permissions::from_mode(0o700)).unwrap();
    let mut d: Value = serde_json::from_slice(&dir.read("descriptor.json").unwrap().0).unwrap();
    d["present"] = json!(false); dir.publish("descriptor.json", &d).unwrap();
    assert!(f.acquire(Duration::ZERO).is_err());
    d["present"] = json!(true); d["transport"]["control_port"] = json!(5003);
    dir.publish("descriptor.json", &d).unwrap();
    assert!(f.acquire(Duration::ZERO).is_err());
    d["transport"]["control_port"] = json!(5002); d["ttl_ms"] = json!(1);
    dir.publish("descriptor.json", &d).unwrap();
    thread::sleep(Duration::from_millis(5));
    assert!(f.acquire(Duration::ZERO).is_err());
}

#[test]
fn dangling_runtime_symlink_is_not_absence() {
    let f = Fixture::new();
    std::os::unix::fs::symlink(f.0.join("missing"), f.root()).unwrap();
    assert!(f.acquire(Duration::ZERO).is_err());
}

#[test]
fn changed_attachment_invalidates_retained_session() {
    let f = Fixture::new(); let dir = f.setup();
    let session = f.acquire(Duration::ZERO).unwrap();
    let mut d: Value = serde_json::from_slice(&dir.read("descriptor.json").unwrap().0).unwrap();
    d["attachment_id"] = json!("attachment-replaced");
    dir.publish("descriptor.json", &d).unwrap();
    assert!(session.check(None).is_err());
}

#[test]
fn waiting_request_is_renewed_without_changing_fifo_identity() {
    let f = Fixture::new(); let dir = f.setup();
    let held = dir.open_file("control.lock", libc::O_RDWR).unwrap();
    assert_eq!(unsafe { libc::flock(held.as_raw_fd(), libc::LOCK_EX) }, 0);
    thread::scope(|scope| {
        let waiter = scope.spawn(|| f.acquire(Duration::from_secs(8)));
        let requests = dir.child("requests").unwrap();
        let start = Instant::now();
        let name = loop {
            if let Some(name) = requests.entries().unwrap().into_iter().find(|n| n.ends_with(".json")) { break name; }
            assert!(start.elapsed() < Duration::from_secs(2));
            thread::sleep(Duration::from_millis(10));
        };
        let (original, meta) = requests.read(&name).unwrap();
        loop {
            let (current, updated) = requests.read(&name).unwrap();
            if updated.modified().unwrap() > meta.modified().unwrap() {
                assert_eq!(current, original); break;
            }
            assert!(start.elapsed() < Duration::from_secs(6), "request not renewed within ttl/3");
            thread::sleep(Duration::from_millis(20));
        }
        drop(held);
        let session = waiter.join().unwrap().unwrap();
        assert!(session.check(None).is_ok());
    });
}

#[test]
fn atomic_publication_replacement_reopens_instead_of_trusting_an_unlinked_snapshot() {
    let f=Fixture::new();let dir=f.setup();
    let mut opens=0;
    let (bytes,meta)=dir.read_snapshot("descriptor.json",|old| {
        opens+=1;
        if opens==1 {
            let mut next:Value=serde_json::from_slice(&dir.read("descriptor.json").unwrap().0).unwrap();
            next["publisher_instance"]=json!("publisher-refreshed");
            dir.publish("descriptor.json",&next).unwrap();
            assert_eq!(old.metadata().unwrap().nlink(),0);
            assert!(private(old,false).is_err(),"the former reader would reject this normal refresh");
        }
    }).unwrap();
    assert_eq!(opens,2);assert_eq!(meta.nlink(),1);
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap()["publisher_instance"],"publisher-refreshed");
}

#[test]
fn atomic_publication_retry_still_rejects_unsafe_or_missing_replacement_and_is_bounded() {
    for kind in ["unsafe","missing","churn"] {
        let f=Fixture::new();let dir=f.setup();let mut opens=0;
        let result=dir.read_snapshot("descriptor.json",|_| {
            opens+=1;
            if opens==1 || kind=="churn" {
                let next:Value=serde_json::from_slice(&dir.read("descriptor.json").unwrap().0).unwrap();
                dir.publish("descriptor.json",&next).unwrap();
                if kind=="unsafe" {fs::set_permissions(f.root().join("cameras/test-camera/descriptor.json"),fs::Permissions::from_mode(0o644)).unwrap();}
                if kind=="missing" {dir.remove("descriptor.json").unwrap();}
            }
        });
        assert!(result.is_err(),"{kind}");assert!(opens<=4);
        if kind=="churn" {assert_eq!(opens,4);}
    }
}
