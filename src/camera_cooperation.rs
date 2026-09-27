//! Mandatory camera ownership boundary. A missing lease is a programming
//! invariant violation, never permission to try the external camera directly.
use serde_json::{json, Value};
use std::{
    env,
    ffi::CString,
    fs::{self, File},
    io::{self, Read, Write},
    net::{SocketAddr, TcpStream},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::MetadataExt,
    },
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

static SESSION: Mutex<Option<Arc<Session>>> = Mutex::new(None);
static STOP: AtomicBool = AtomicBool::new(false);
static WATCHDOG: Mutex<Option<thread::JoinHandle<()>>> = Mutex::new(None);
const LIMIT: u64 = 65536;

mod diagnostics;

fn err(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}
fn text<'a>(v: &'a Value, key: &str) -> io::Result<&'a str> {
    v[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| err(format!("missing {key}")))
}
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && !id.contains("..")
        && !id.bytes().all(|b| b.is_ascii_digit())
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

fn entry_exists(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true), // A dangling symlink is an invalid runtime, not absence.
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
fn unique() -> io::Result<String> {
    let mut bytes = [0u8; 16];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(format!(
        "buttercup-{}",
        bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
    ))
}

/// Pinned directory-relative access. No symlinked runtime components or files.
struct Dir(File);
impl Dir {
    fn open(path: &Path) -> io::Result<Self> {
        if !path.is_absolute() {
            return Err(err("UPC root must be absolute"));
        }
        let mut dir = Self(File::open("/")?);
        for part in path.components() {
            match part {
                Component::RootDir => (),
                Component::Normal(name) => {
                    let name = name.to_str().ok_or_else(|| err("invalid UPC path"))?;
                    dir = Self(dir.open_file(name, libc::O_RDONLY | libc::O_DIRECTORY)?);
                }
                _ => return Err(err("invalid UPC path component")),
            }
        }
        dir.private()?;
        let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstatfs(dir.0.as_raw_fd(), &mut stat) } != 0 {
            return Err(io::Error::last_os_error());
        }
        if matches!(
            stat.f_type as i64,
            0x6969 | 0xff534d42 | 0xfe534d42 | 0x73757245 | 0x5346414f | 0x01021997
        ) {
            return Err(err("UPC locks require a local filesystem"));
        }
        Ok(dir)
    }
    fn private(&self) -> io::Result<()> {
        private(&self.0, true)
    }
    fn open_file(&self, name: &str, flags: i32) -> io::Result<File> {
        if name.contains('/') || name == "." || name == ".." {
            return Err(err("invalid relative filename"));
        }
        let name = CString::new(name).map_err(|_| err("invalid filename"))?;
        let fd = unsafe {
            libc::openat(
                self.0.as_raw_fd(),
                name.as_ptr(),
                flags | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(unsafe { File::from_raw_fd(fd) })
    }
    fn child(&self, name: &str) -> io::Result<Self> {
        let dir = Self(self.open_file(name, libc::O_RDONLY | libc::O_DIRECTORY)?);
        dir.private()?;
        Ok(dir)
    }
    fn read(&self, name: &str) -> io::Result<(Vec<u8>, fs::Metadata)> {
        self.read_snapshot(name, |_| {})
    }
    fn read_snapshot(&self, name: &str, mut after_open: impl FnMut(&File)) -> io::Result<(Vec<u8>, fs::Metadata)> {
        for _ in 0..4 {
            let mut file = self.open_file(name, libc::O_RDONLY | libc::O_NONBLOCK)?;
            after_open(&file);
            let meta = file.metadata()?;
            // Atomic publication may unlink the old inode after openat but
            // before fstat. Never authorize from that retired snapshot: open
            // the new publication and repeat every ownership/content check.
            // This exception applies ONLY to read-only publications, never
            // the retained camera directory or exclusive control.lock.
            if meta.nlink()==0 && meta.is_file() && meta.uid()==unsafe {libc::geteuid()}
                && meta.mode() & 0o777 == 0o600 {continue;}
            private_metadata(&meta, false).map_err(|e|err(&format!("{name}: {e}")))?;
            if meta.len() > LIMIT {return Err(err("oversized UPC file"));}
            let mut bytes = Vec::new();
            (&mut file).take(LIMIT + 1).read_to_end(&mut bytes)?;
            if bytes.len() as u64 > LIMIT {return Err(err("oversized UPC file"));}
            // Also catch replacement during the read, before returning any
            // bytes to descriptor/owner/TTL validation.
            let after=file.metadata()?;
            if after.nlink()==0 {continue;}
            private_metadata(&after, false).map_err(|e|err(&format!("{name}: {e}")))?;
            return Ok((bytes, meta));
        }
        Err(err(&format!("UPC publication kept changing during read: {name}")))
    }
    fn fresh(&self, name: &str) -> io::Result<Value> {
        let (bytes, meta) = self.read(name)?;
        let v: Value = serde_json::from_slice(&bytes)?;
        let ttl = v["ttl_ms"]
            .as_u64()
            .filter(|t| (1..=300000).contains(t))
            .ok_or_else(|| err("invalid UPC TTL"))?;
        let age = SystemTime::now()
            .duration_since(meta.modified()?)
            .map_err(|_| err("future UPC file"))?;
        if age >= Duration::from_millis(ttl) {
            return Err(err("stale UPC file"));
        }
        Ok(v)
    }
    fn entries(&self) -> io::Result<Vec<String>> {
        // /proc points at our already pinned directory, never a runtime symlink.
        fs::read_dir(format!("/proc/self/fd/{}", self.0.as_raw_fd()))?
            .map(|e| e.map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect()
    }
    fn remove(&self, name: &str) -> io::Result<()> {
        let name = CString::new(name).map_err(|_| err("invalid filename"))?;
        if unsafe { libc::unlinkat(self.0.as_raw_fd(), name.as_ptr(), 0) } != 0 {
            let e = io::Error::last_os_error();
            if e.kind() != io::ErrorKind::NotFound {
                return Err(e);
            }
        }
        Ok(())
    }
    fn publish(&self, name: &str, value: &Value) -> io::Result<()> {
        let temporary = format!(".{}.tmp", unique()?);
        let result = (|| {
            let mut file =
                self.open_file(&temporary, libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL)?;
            serde_json::to_writer(&mut file, value)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            drop(file);
            let from = CString::new(temporary.as_str()).unwrap();
            let to = CString::new(name).map_err(|_| err("invalid filename"))?;
            if unsafe {
                libc::renameat(
                    self.0.as_raw_fd(),
                    from.as_ptr(),
                    self.0.as_raw_fd(),
                    to.as_ptr(),
                )
            } != 0
            {
                return Err(io::Error::last_os_error());
            }
            self.0.sync_all()
        })();
        if result.is_err() {
            let _ = self.remove(&temporary);
        }
        result
    }
}
fn private(file: &File, directory: bool) -> io::Result<()> {
    let m = file.metadata()?;
    private_metadata(&m,directory)
}
fn private_metadata(m: &fs::Metadata, directory: bool) -> io::Result<()> {
    if m.uid() != unsafe { libc::geteuid() }
        || m.mode() & 0o777 != if directory { 0o700 } else { 0o600 }
        || (directory && !m.is_dir())
        || (!directory && (!m.is_file() || m.nlink() != 1))
    {
        return Err(err(&format!("unsafe UPC ownership, permissions, or file type (uid={} mode={:o} links={} directory={directory})",m.uid(),m.mode() & 0o777,m.nlink())));
    }
    Ok(())
}
fn same_file(a: &File, b: &File) -> io::Result<bool> {
    let (a, b) = (a.metadata()?, b.metadata()?);
    Ok(a.dev() == b.dev() && a.ino() == b.ino())
}

struct Lease {
    root: Dir,
    camera_dir: Dir,
    lock: File,
    descriptor: Value,
    owner: Value,
}
struct Session {
    root_path: PathBuf,
    capture: SocketAddr,
    control: SocketAddr,
    lease: Option<Lease>,
}

fn descriptor(dir: &Dir, id: &str, capture: SocketAddr, control: SocketAddr) -> io::Result<Value> {
    let d = dir.fresh("descriptor.json")?;
    let t = &d["transport"];
    if d["schema"] != "upc.camera.v1"
        || d["camera_id"] != id
        || d["present"] != true
        || !valid_id(text(&d, "attachment_id")?)
        || !valid_id(text(&d, "publisher_instance")?)
        || d["device_paths"].as_array().is_none_or(|a| !a.is_empty())
        || t["kind"] != "tcp"
        || t["protocol"] != "podbay-raw-v1"
        || text(t, "host")?.parse::<std::net::IpAddr>().ok() != Some(capture.ip())
        || control.ip() != capture.ip()
        || t["port"].as_u64() != Some(capture.port() as u64)
        || t["control_port"].as_u64() != Some(control.port() as u64)
    {
        return Err(err(
            "camera descriptor does not authorize the exact capture/control endpoints",
        ));
    }
    Ok(d)
}
fn same_attachment(a: &Value, b: &Value) -> bool {
    [
        "camera_id",
        "attachment_id",
        "publisher_instance",
        "transport",
    ]
    .into_iter()
    .all(|k| a[k] == b[k])
}
fn first_request(requests: &Dir, camera: &str, attachment: &str) -> io::Result<Option<String>> {
    let mut queue = Vec::new();
    for name in requests.entries()? {
        let Some(id) = name.strip_suffix(".json").filter(|id| valid_id(id)) else {
            continue;
        };
        let Ok(r) = requests.fresh(&name) else {
            continue;
        };
        if r["schema"] == "upc.request.v1"
            && r["request_id"] == id
            && r["camera_id"] == camera
            && r["attachment_id"] == attachment
            && r["requester_instance"].as_str().is_some_and(valid_id)
            && r["pid"].as_u64().is_some_and(|p| p > 0)
        {
            if let Some(created) = r["created_unix_ms"].as_u64() {
                queue.push((created, id.to_owned()));
            }
        }
    }
    queue.sort();
    Ok(queue.into_iter().next().map(|(_, id)| id))
}

impl Session {
    fn acquire(
        root_path: PathBuf,
        capture: SocketAddr,
        control: SocketAddr,
        wait: Duration,
    ) -> io::Result<Self> {
        if !entry_exists(&root_path)? {
            return Ok(Self {
                root_path,
                capture,
                control,
                lease: None,
            });
        }
        let root = Dir::open(&root_path)?;
        if root.read("version")?.0 != b"1\n" {
            return Err(err("unsupported UPC version"));
        }
        let cameras = root.child("cameras")?;
        let mut matching = Vec::new();
        for id in cameras.entries()?.into_iter().filter(|s| valid_id(s)) {
            let Ok(dir) = cameras.child(&id) else {
                continue;
            };
            if let Ok(d) = descriptor(&dir, &id, capture, control) {
                matching.push((id, dir, d));
            }
        }
        if matching.len() != 1 {
            return Err(err("presence runtime exists but no unique fresh camera descriptor matches; start/recover its publisher, never bypass UPC"));
        }
        let (id, camera_dir, d) = matching.pop().unwrap();
        let attachment = text(&d, "attachment_id")?;
        let requests = camera_dir.child("requests")?;
        let instance = unique()?;
        let request_id = unique()?;
        let filename = format!("{request_id}.json");
        let request = json!({"schema":"upc.request.v1","request_id":request_id,"requester_instance":instance,
            "pid":std::process::id(),"camera_id":id,"attachment_id":attachment,"purpose":"eye tracking camera session",
            "created_unix_ms":now_ms(),"ttl_ms":15000,"expected_max_hold_ms":3600000,"please_yield_within_ms":1500});
        requests.publish(&filename, &request)?;
        let acquired = (|| {
            let end = Instant::now() + wait;
            let mut renewed = Instant::now();
            loop {
                // Preserve FIFO identity/time, refreshing only file freshness.
                if renewed.elapsed() >= Duration::from_secs(4) {
                    requests.publish(&filename, &request)?;
                    renewed = Instant::now();
                }
                let current = descriptor(&camera_dir, &id, capture, control)?;
                if !same_attachment(&d, &current) {
                    return Err(err("UPC attachment changed during acquisition"));
                }
                if first_request(&requests, &id, attachment)?.as_deref() == Some(&request_id) {
                    let lock = camera_dir.open_file("control.lock", libc::O_RDWR)?;
                    private(&lock, false)?;
                    let result =
                        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
                    if result == 0 {
                        let current = descriptor(&camera_dir, &id, capture, control)?;
                        let retained = camera_dir.open_file("control.lock", libc::O_RDWR)?;
                        if !same_file(&lock, &retained)?
                            || !same_attachment(&d, &current)
                            || first_request(&requests, &id, attachment)?.as_deref()
                                != Some(&request_id)
                        {
                            return Err(err(
                                "UPC attachment, queue, or lock changed after acquisition",
                            ));
                        }
                        let owner = json!({"schema":"upc.owner.v1","camera_id":id,"attachment_id":attachment,
                            "owner_instance":instance,"pid":std::process::id(),"request_id":request_id,
                            "purpose":"eye tracking camera session","acquired_unix_ms":now_ms()});
                        camera_dir.publish("owner.json", &owner)?;
                        return Ok((lock, owner));
                    }
                    let e = io::Error::last_os_error();
                    if e.raw_os_error() != Some(libc::EWOULDBLOCK) {
                        return Err(e);
                    }
                }
                if Instant::now() >= end {
                    return Err(err("UPC acquisition timed out without camera ownership"));
                }
                thread::sleep(Duration::from_millis(25));
            }
        })();
        requests.remove(&filename)?;
        let (lock, owner) = acquired?;
        Ok(Self {
            root_path,
            capture,
            control,
            lease: Some(Lease {
                root,
                camera_dir,
                lock,
                descriptor: d,
                owner,
            }),
        })
    }
    fn check(&self, endpoint: Option<SocketAddr>) -> io::Result<()> {
        if endpoint.is_some_and(|a| a != self.capture && a != self.control) {
            return Err(err(
                "camera endpoint changed without a new cooperative session",
            ));
        }
        let Some(lease) = &self.lease else {
            return if entry_exists(&self.root_path)? {
                Err(err(
                    "presence runtime appeared without an acquired camera lease",
                ))
            } else {
                Ok(())
            };
        };
        let root = Dir::open(&self.root_path)?;
        if !same_file(&root.0, &lease.root.0)? || root.read("version")?.0 != b"1\n" {
            return Err(err("UPC root/version changed while camera was owned"));
        }
        let id = text(&lease.descriptor, "camera_id")?;
        let camera = root.child("cameras")?.child(id)?;
        let lock = camera.open_file("control.lock", libc::O_RDWR)?;
        if !same_file(&camera.0, &lease.camera_dir.0)? || !same_file(&lock, &lease.lock)? {
            return Err(err(
                "retained UPC camera directory or lock inode was replaced",
            ));
        }
        let d = descriptor(&camera, id, self.capture, self.control)?;
        if !same_attachment(&lease.descriptor, &d) {
            return Err(err("camera attachment changed while camera was owned"));
        }
        let (bytes, _) = camera.read("owner.json")?;
        let owner: Value = serde_json::from_slice(&bytes)?;
        if owner != lease.owner {
            return Err(err(
                "UPC owner record no longer matches this process's retained lease",
            ));
        }
        Ok(())
    }
}

/// Runs before any camera setup or worker creation. Contention waits inside
/// this call; failure is a startup error, never a direct-camera fallback.
pub(crate) fn start(capture: &str, control: &str) -> Result<(), String> {
    let root = runtime_root();
    let session = Session::acquire(
        root.clone(),
        capture
            .parse()
            .map_err(|e| format!("camera endpoint: {e}"))?,
        control
            .parse()
            .map_err(|e| format!("control endpoint: {e}"))?,
        Duration::from_secs(10),
    )
    .map_err(|e| {
        eprintln!(
            "CAMERA OWNERSHIP STARTUP REFUSED: runtime={} capture={capture} control={control}: {e}",
            root.display(),
        );
        eprintln!("CAMERA RECOVERY: restore the UPC camera publisher and its fresh descriptor, or wait for the current owner to release the camera. An existing unavailable runtime cannot authorize standalone access. Do not delete the runtime or replace its lock. See docs/camera-startup.md.");
        format!("UPC camera startup refused: {e}")
    })?;
    eprintln!(
        "CAMERA SESSION: ownership={} capture={capture} control={control} runtime={}",
        if session.lease.is_some() { "UPC retained lease" } else { "standalone; UPC runtime absent" },
        root.display(),
    );
    let mut current = SESSION
        .lock()
        .unwrap_or_else(|_| fatal("camera session mutex poisoned"));
    if current.is_some() {
        fatal("camera session initialized twice");
    }
    *current = Some(Arc::new(session));
    drop(current);
    STOP.store(false, Ordering::Release);
    // Detect revocation even when a stream remains open and no new connection
    // is attempted. A worker panic cannot swallow this process-wide abort.
    let watchdog = thread::Builder::new()
        .name("upc-camera-invariant".into())
        .spawn(|| {
            while !STOP.load(Ordering::Acquire) {
                assert_access(None);
                thread::sleep(Duration::from_millis(100));
            }
        })
        .map_err(|e| format!("UPC watchdog startup failed: {e}"))?;
    *WATCHDOG
        .lock()
        .unwrap_or_else(|_| fatal("watchdog mutex poisoned")) = Some(watchdog);
    Ok(())
}

/// Call after all camera/control workers have joined. Releasing the retained
/// lock sooner could let the next owner race a surviving capture socket.
pub(crate) fn finish() {
    STOP.store(true, Ordering::Release);
    if let Some(watchdog) = WATCHDOG
        .lock()
        .unwrap_or_else(|_| fatal("watchdog mutex poisoned"))
        .take()
    {
        if watchdog.join().is_err() {
            fatal("camera ownership watchdog panicked");
        }
    }
    SESSION
        .lock()
        .unwrap_or_else(|_| fatal("camera session mutex poisoned"))
        .take();
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(lease) = &self.lease {
            let mut owner = lease.owner.clone();
            owner["state"] = json!("released");
            // Still holding the kernel lock. On abnormal process termination
            // the kernel releases it; owner metadata is diagnostic, not a lease.
            if let Err(e) = lease.camera_dir.publish("owner.json", &owner) {
                eprintln!("UPC release metadata failed: {e}");
            }
        }
    }
}

/// Headless diagnostic using exactly the viewer's startup/ownership guard.
/// No camera commands, GUI, raw frames, or inference are needed to test handoff.
pub(crate) fn check_startup() -> Result<(), String> {
    let capture =
        env::var("BUTTERCUP_CAMERA_ADDRESS").unwrap_or_else(|_| "192.168.88.10:5001".into());
    let control = env::var("BUTTERCUP_VCM_ADDRESS").unwrap_or_else(|_| "192.168.88.10:5002".into());
    start(&capture, &control)?;
    assert_access(Some(capture.parse().map_err(|e| format!("camera: {e}"))?));
    assert_access(Some(control.parse().map_err(|e| format!("control: {e}"))?));
    println!(
        "{}",
        json!({"protocol":"buttercup_camera_cooperation_check_v1","ownership":"validated","camera_commands_sent":0})
    );
    Ok(())
}

fn assert_access(endpoint: Option<SocketAddr>) {
    let session = SESSION
        .lock()
        .unwrap_or_else(|_| fatal("camera session mutex poisoned"))
        .clone()
        .unwrap_or_else(|| {
            fatal("camera access attempted before cooperative session initialization")
        });
    if let Err(e) = session.check(endpoint) {
        fatal(&e.to_string());
    }
}

fn runtime_root() -> PathBuf {
    env::var_os("UPC_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            env::var_os("XDG_RUNTIME_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(format!("/tmp/user-presence-camera-{}", unsafe {
                        libc::geteuid()
                    }))
                })
                .join(if env::var_os("XDG_RUNTIME_DIR").is_some() {
                    "user-presence-camera/v1"
                } else {
                    "v1"
                })
        })
}

/// Abort rather than panic: a caught/unwound worker panic must not leave other
/// camera threads alive. Avoid dumping a potentially huge model-bearing core.
fn fatal(reason: &str) -> ! {
    eprintln!("FATAL ASSERTION: UPC camera ownership invariant violated: {reason}");
    eprintln!("CAMERA RECOVERY: this process is aborting all camera workers. Check initialization order and the UPC publisher/attachment, then restart the viewer; do not reuse the old session or remove ownership files. See docs/camera-startup.md.");
    unsafe {
        let limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        libc::setrlimit(libc::RLIMIT_CORE, &limit);
    }
    std::process::abort()
}

pub(crate) fn connect_timeout(address: &SocketAddr, timeout: Duration) -> io::Result<TcpStream> {
    assert_access(Some(*address));
    let result = TcpStream::connect_timeout(address, timeout);
    // Ownership loss during a failed TCP attempt is still process-fatal. Only
    // an ordinary transport failure may reach the caller or recovery hints.
    assert_access(Some(*address));
    if let Err(error) = &result {
        diagnostics::connection_failed(*address, timeout, error);
    }
    result
}

#[cfg(test)]
#[path = "camera_cooperation/tests.rs"]
mod integration_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        net::TcpListener,
        os::unix::{
            fs::{symlink, OpenOptionsExt, PermissionsExt},
            process::ExitStatusExt,
        },
        process::Command,
    };

    struct Fixture {
        path: PathBuf,
        capture: SocketAddr,
        control: SocketAddr,
    }
    impl Fixture {
        fn new() -> Self {
            let path = env::temp_dir().join(unique().unwrap());
            fs::create_dir(&path).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
            for name in [
                "cameras",
                "cameras/fixture-camera",
                "cameras/fixture-camera/requests",
            ] {
                fs::create_dir(path.join(name)).unwrap();
                fs::set_permissions(path.join(name), fs::Permissions::from_mode(0o700)).unwrap();
            }
            for (name, data) in [
                ("version", "1\n"),
                ("cameras/fixture-camera/control.lock", ""),
            ] {
                let mut f = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(path.join(name))
                    .unwrap();
                f.write_all(data.as_bytes()).unwrap();
            }
            let f = Self {
                path,
                capture: "127.0.0.1:5001".parse().unwrap(),
                control: "127.0.0.1:5002".parse().unwrap(),
            };
            f.publish("attachment-original");
            f
        }
        fn camera(&self) -> Dir {
            Dir::open(&self.path)
                .unwrap()
                .child("cameras")
                .unwrap()
                .child("fixture-camera")
                .unwrap()
        }
        fn publish(&self, attachment: &str) {
            self.camera().publish("descriptor.json", &json!({"schema":"upc.camera.v1","camera_id":"fixture-camera",
                "attachment_id":attachment,"publisher_instance":"fixture-publisher","present":true,"ttl_ms":5000,
                "published_unix_ms":now_ms(),"device_paths":[],"orientation":{"known":false},"capabilities":["video"],
                "transport":{"kind":"tcp","protocol":"podbay-raw-v1","host":self.capture.ip().to_string(),
                "port":self.capture.port(),"control_port":self.control.port()}})).unwrap();
        }
        fn acquire(&self) -> io::Result<Session> {
            Session::acquire(
                self.path.clone(),
                self.capture,
                self.control,
                Duration::from_millis(80),
            )
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn retained_lease_excludes_other_clients_and_releases_without_replacing_inode() {
        let f = Fixture::new();
        let session = f.acquire().unwrap();
        session.check(Some(f.capture)).unwrap();
        session.check(Some(f.control)).unwrap();
        let lock = f.camera().open_file("control.lock", libc::O_RDWR).unwrap();
        let inode = lock.metadata().unwrap().ino();
        assert_ne!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        assert_eq!(
            io::Error::last_os_error().raw_os_error(),
            Some(libc::EWOULDBLOCK)
        );
        assert!(f
            .camera()
            .child("requests")
            .unwrap()
            .entries()
            .unwrap()
            .is_empty());
        assert!(f.acquire().is_err());
        drop(session);
        assert_eq!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        assert_eq!(
            f.camera()
                .open_file("control.lock", libc::O_RDWR)
                .unwrap()
                .metadata()
                .unwrap()
                .ino(),
            inode
        );
        let owner: Value =
            serde_json::from_slice(&f.camera().read("owner.json").unwrap().0).unwrap();
        assert_eq!(owner["state"], "released");
    }
    #[test]
    fn descriptor_or_owner_changes_and_wrong_endpoints_revoke_authorization() {
        let f = Fixture::new();
        let session = f.acquire().unwrap();
        assert!(session
            .check(Some("127.0.0.1:6001".parse().unwrap()))
            .is_err());
        f.publish("attachment-replaced");
        assert!(session.check(None).is_err());
        f.publish("attachment-original");
        session.check(None).unwrap();
        f.camera()
            .publish("owner.json", &json!({"pid":std::process::id()}))
            .unwrap();
        assert!(session.check(None).is_err());
    }
    #[test]
    fn lock_replacement_cannot_create_a_second_authority() {
        let f = Fixture::new();
        let session = f.acquire().unwrap();
        f.camera().remove("control.lock").unwrap();
        let _new = f
            .camera()
            .open_file("control.lock", libc::O_RDWR | libc::O_CREAT | libc::O_EXCL)
            .unwrap();
        assert!(session.check(None).is_err());
    }
    #[test]
    fn missing_stale_malformed_and_symlinked_runtime_never_fall_back() {
        let f = Fixture::new();
        fs::write(f.path.join("version"), "2\n").unwrap();
        assert!(f.acquire().is_err());
        fs::write(f.path.join("version"), "1\n").unwrap();
        f.camera()
            .publish("descriptor.json", &json!({"ttl_ms":1}))
            .unwrap();
        thread::sleep(Duration::from_millis(3));
        assert!(f.acquire().is_err());
        f.publish("attachment-original");
        let link = f.path.join("linked-runtime");
        symlink(f.path.join("missing"), &link).unwrap();
        assert!(Session::acquire(link, f.capture, f.control, Duration::ZERO).is_err());
        f.camera().remove("control.lock").unwrap();
        symlink(
            f.path.join("version"),
            f.path.join("cameras/fixture-camera/control.lock"),
        )
        .unwrap();
        assert!(f.acquire().is_err());
    }
    #[test]
    fn earlier_request_is_respected_and_own_timed_out_request_is_removed() {
        let f = Fixture::new();
        let requests = f.camera().child("requests").unwrap();
        requests.publish("request-earlier.json",&json!({"schema":"upc.request.v1","request_id":"request-earlier",
            "camera_id":"fixture-camera","attachment_id":"attachment-original","requester_instance":"other-client",
            "pid":123,"created_unix_ms":1,"ttl_ms":5000})).unwrap();
        assert!(f.acquire().is_err());
        assert_eq!(requests.entries().unwrap(), vec!["request-earlier.json"]);
    }
    #[test]
    fn fatal_child() {
        let Ok(mode) = env::var("BUTTERCUP_UPC_TEST_CHILD") else {
            return;
        };
        match mode.as_str() {
            "transport" => {
                let endpoint = env::var("BUTTERCUP_UPC_TEST_ENDPOINT").unwrap();
                start(&endpoint, "127.0.0.1:1").unwrap();
                for _ in 0..2 {
                    let error = connect_timeout(&endpoint.parse().unwrap(), Duration::from_millis(100))
                        .unwrap_err();
                    assert_eq!(error.kind(), io::ErrorKind::ConnectionRefused);
                    assert_eq!(error.raw_os_error(), Some(libc::ECONNREFUSED));
                }
                finish();
                return;
            }
            "uninitialized" => {
                let _: io::Result<_> = connect_timeout(
                    &env::var("BUTTERCUP_UPC_TEST_ENDPOINT")
                        .unwrap()
                        .parse()
                        .unwrap(),
                    Duration::from_millis(50),
                );
            }
            "watchdog" => {
                start("127.0.0.1:5001", "127.0.0.1:5002").unwrap();
                let root = runtime_root();
                fs::write(root.join("version"), "2\n").unwrap();
                thread::sleep(Duration::from_secs(2));
            }
            "appeared" => {
                let root = runtime_root();
                start("127.0.0.1:5001", "127.0.0.1:5002").unwrap();
                fs::create_dir(root).unwrap();
                thread::sleep(Duration::from_secs(2));
            }
            _ => panic!("unknown child scenario"),
        }
        panic!("camera invariant violation was not process-fatal");
    }
    #[test]
    fn ordinary_connection_refusal_keeps_original_error_and_reports_recovery_once() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        drop(listener);
        let f = Fixture::new();
        let output = Command::new(env::current_exe().unwrap())
            .args(["--exact", "camera_cooperation::tests::fatal_child", "--nocapture"])
            .env("UPC_ROOT", f.path.join("absent"))
            .env("BUTTERCUP_UPC_TEST_CHILD", "transport")
            .env("BUTTERCUP_UPC_TEST_ENDPOINT", endpoint.to_string())
            .output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stderr}");
        assert_eq!(stderr.matches("CAMERA TCP UNAVAILABLE:").count(), 1, "{stderr}");
        assert!(stderr.contains(&format!("endpoint={endpoint}")), "{stderr}");
        assert!(stderr.contains("tools/deploy.py"), "{stderr}");
        assert!(!stderr.contains("FATAL ASSERTION"), "{stderr}");
    }
    #[test]
    fn violations_abort_the_process_before_connecting_and_during_existing_sessions() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        for mode in ["uninitialized", "watchdog", "appeared"] {
            let f = Fixture::new();
            let root = if mode == "appeared" {
                f.path.join("later")
            } else {
                f.path.clone()
            };
            let output = Command::new(env::current_exe().unwrap())
                .args([
                    "--exact",
                    "camera_cooperation::tests::fatal_child",
                    "--nocapture",
                ])
                .env("UPC_ROOT", root)
                .env("BUTTERCUP_UPC_TEST_CHILD", mode)
                .env(
                    "BUTTERCUP_UPC_TEST_ENDPOINT",
                    listener.local_addr().unwrap().to_string(),
                )
                .output()
                .unwrap();
            assert_eq!(
                output.status.signal(),
                Some(libc::SIGABRT),
                "{mode}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stderr)
                .contains("FATAL ASSERTION: UPC camera ownership invariant violated"));
            assert!(!String::from_utf8_lossy(&output.stderr).contains("CAMERA TCP UNAVAILABLE:"),
                "ownership failures must never become recoverable transport errors");
            assert_eq!(
                listener.accept().unwrap_err().kind(),
                io::ErrorKind::WouldBlock
            );
        }
    }
    #[test]
    fn production_camera_connections_cannot_bypass_the_guard() {
        let source = include_str!("main.rs")
            .split("\nmod tests {")
            .next()
            .unwrap();
        assert!(
            !source.contains("TcpStream::connect"),
            "route camera TCP connections through camera_cooperation"
        );
        assert_eq!(
            source
                .matches("camera_cooperation::connect_timeout(")
                .count(),
            4
        );
        let start = source
            .find("camera_cooperation::start(&config.camera, &config.vcm)?")
            .unwrap();
        assert!(start < source[start..].find("apply_eye_config(&config)?").unwrap() + start);
    }
}
