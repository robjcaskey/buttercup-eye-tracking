//! Optional Keyboard Peeper Protocol (KPP/1) publisher.
//!
//! This intentionally duplicates the tiny wire encoder instead of depending on
//! another checkout. If Keyboard Peeper is absent, the viewer keeps running and
//! silently retries in the background.

use std::env;
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const MAGIC: [u8; 4] = *b"KPP\x01";
const HEADER_LEN: usize = 20;
const REPLACE: u8 = 1;
const CLEAR: u8 = 2;
const ACK: u8 = 0x80;
const ACK_APPLIED: u8 = 0;
const RETRY_INTERVAL: Duration = Duration::from_millis(25);
const IO_TIMEOUT: Duration = Duration::from_millis(250);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Binding {
    pub modifiers: u8,
    pub enabled: bool,
    pub key: &'static str,
    pub label: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HotkeyMap {
    pub title: &'static str,
    pub bindings: Vec<Binding>,
}

pub struct Registration {
    sender: Option<mpsc::Sender<Option<HotkeyMap>>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    last: Option<HotkeyMap>,
}

impl Registration {
    pub fn new(initial: Option<HotkeyMap>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let enabled = env::var_os("BUTTERCUP_KEYBOARD_PEEPER").is_none_or(|value| value != "0");
        if !enabled {
            return Self {
                sender: None,
                stop,
                worker: None,
                last: initial,
            };
        }

        let (sender, receiver) = mpsc::channel();
        let worker_stop = Arc::clone(&stop);
        let last = initial.clone();
        let worker = thread::Builder::new()
            .name("keyboard-peeper-publisher".to_string())
            .spawn(move || publish_loop(initial, receiver, worker_stop))
            .ok();
        Self {
            sender: worker.as_ref().map(|_| sender),
            stop,
            worker,
            last,
        }
    }

    pub fn replace_if_changed(&mut self, map: Option<HotkeyMap>) {
        if map == self.last {
            return;
        }
        self.last = map.clone();
        if let Some(sender) = &self.sender {
            let _ = sender.send(map);
        }
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub fn buttercup_map(virtual_mouse: bool, driving: bool) -> HotkeyMap {
    let mut bindings = Vec::with_capacity(37);
    let mut add = |key, label, enabled| {
        bindings.push(Binding {
            modifiers: 0,
            enabled,
            key,
            label,
        });
    };

    add("Esc", "Exit", true);
    add("Q", "Exit", true);
    add("M", "Mouse mode", true);
    add("\\", "Accuracy check (20 targets)", !virtual_mouse);
    add("B", "Lightbox", true);
    add(
        "[",
        if virtual_mouse {
            "Lightbox -"
        } else {
            "Iris min -"
        },
        true,
    );
    add(
        "]",
        if virtual_mouse {
            "Lightbox +"
        } else {
            "Iris min +"
        },
        true,
    );
    add("-", "Iris max -", !virtual_mouse);
    add("=", "Iris max +", !virtual_mouse);
    add("N", "Pattern", true);
    add("V", "View", true);
    add("F", "Scoped view", true);
    add("Tab", "ROI / linked / global", !virtual_mouse);
    add(",", "Inspector panel", !virtual_mouse);
    add("F2", "AF reference = selected ROI", !virtual_mouse);
    add("Space", "Object search start/stop", !virtual_mouse);
    add("PageUp", "Inspector scroll up", !virtual_mouse);
    add("PageDown", "Inspector scroll down", !virtual_mouse);
    add("X", "Lighthouse", true);
    add("Z", "Screen clock", true);
    add("K", "Edge map", true);
    add("J", "Eye laser", true);
    add("W", "ROI/sensor follow", true);
    add("3", "Second ROI on/off", true);
    add("G", "Segment", true);
    add("Y", "Pupil source", true);
    add("U", "Reticle", true);
    add("T", "Drive mode", driving);
    add("C", "Calibrate", true);
    add(";", "Focus -", true);
    add("'", "Focus +", true);
    add("L", "Auto focus", true);
    add("D", "RAW still", true);
    add("S", "RAW record", true);
    add("H", "RAW record", true);
    add("0", "Auto iris", !virtual_mouse);
    add("1", "Left eye", true);
    add("2", "Right eye", true);
    add("R", "Reacquire", true);
    add("E", "Lock origin", true);
    add("O", "Contrast -", true);
    add("P", "Contrast +", true);
    add("I", "Contrast reset", true);
    add(".", "Brighter", true);
    add("/", "Darker", true);
    add("A", "Auto exposure", true);
    add("←", "Pupil min -", true);
    add("→", "Pupil min +", true);
    add("↓", "Pupil max -", true);
    add("↑", "Pupil max +", true);

    HotkeyMap {
        title: "Buttercup Eye Viewer",
        bindings,
    }
}

fn socket_path() -> PathBuf {
    env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir)
        .join("keyboard-peeper.sock")
}

fn publish_loop(
    mut desired: Option<HotkeyMap>,
    receiver: mpsc::Receiver<Option<HotkeyMap>>,
    stop: Arc<AtomicBool>,
) {
    let mut stream: Option<UnixStream> = None;
    let mut revision = 0_u64;
    let mut dirty = true;
    let mut last_send = Instant::now() - HEARTBEAT_INTERVAL;

    while !stop.load(Ordering::Relaxed) {
        while let Ok(newest) = receiver.try_recv() {
            desired = newest;
            dirty = true;
        }

        if stream.is_none() {
            stream = UnixStream::connect(socket_path()).ok().and_then(|stream| {
                stream.set_read_timeout(Some(IO_TIMEOUT)).ok()?;
                stream.set_write_timeout(Some(IO_TIMEOUT)).ok()?;
                Some(stream)
            });
            dirty = true;
        }

        if let Some(active) = stream.as_mut() {
            if dirty || last_send.elapsed() >= HEARTBEAT_INTERVAL {
                revision = revision.wrapping_add(1).max(1);
                if replace(active, revision, &desired).is_ok() {
                    dirty = false;
                    last_send = Instant::now();
                } else {
                    stream = None;
                }
            }
        }
        thread::park_timeout(RETRY_INTERVAL);
    }
}

fn replace(stream: &mut UnixStream, revision: u64, map: &Option<HotkeyMap>) -> io::Result<()> {
    let (kind, payload) = match map {
        Some(map) => (REPLACE, encode_map(map)?),
        None => (CLEAR, Vec::new()),
    };
    write_frame(stream, kind, revision, &payload)?;

    let mut header = [0_u8; HEADER_LEN];
    stream.read_exact(&mut header)?;
    if header[0..4] != MAGIC || header[4] != ACK || header[5..8] != [0, 0, 0] {
        return Err(invalid("invalid KPP acknowledgement"));
    }
    if u64::from_le_bytes(header[8..16].try_into().unwrap()) != revision
        || u32::from_le_bytes(header[16..20].try_into().unwrap()) != 1
    {
        return Err(invalid("mismatched KPP acknowledgement"));
    }
    let mut status = [0_u8; 1];
    stream.read_exact(&mut status)?;
    if status[0] != ACK_APPLIED {
        return Err(invalid("KPP replacement was not applied"));
    }
    Ok(())
}

fn encode_map(map: &HotkeyMap) -> io::Result<Vec<u8>> {
    let mut payload = Vec::new();
    put_u16_string(&mut payload, map.title)?;
    let count = u16::try_from(map.bindings.len()).map_err(|_| invalid("too many bindings"))?;
    payload.extend_from_slice(&count.to_le_bytes());
    for binding in &map.bindings {
        payload.push(binding.modifiers);
        payload.push(u8::from(binding.enabled));
        put_u8_string(&mut payload, binding.key)?;
        put_u16_string(&mut payload, binding.label)?;
    }
    Ok(payload)
}

fn write_frame(writer: &mut impl Write, kind: u8, revision: u64, payload: &[u8]) -> io::Result<()> {
    let payload_len = u32::try_from(payload.len()).map_err(|_| invalid("payload too large"))?;
    let mut header = [0_u8; HEADER_LEN];
    header[0..4].copy_from_slice(&MAGIC);
    header[4] = kind;
    header[8..16].copy_from_slice(&revision.to_le_bytes());
    header[16..20].copy_from_slice(&payload_len.to_le_bytes());
    writer.write_all(&header)?;
    writer.write_all(payload)?;
    writer.flush()
}

fn put_u8_string(output: &mut Vec<u8>, value: &str) -> io::Result<()> {
    let len = u8::try_from(value.len()).map_err(|_| invalid("key name too long"))?;
    output.push(len);
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_u16_string(output: &mut Vec<u8>, value: &str) -> io::Result<()> {
    let len = u16::try_from(value.len()).map_err(|_| invalid("text too long"))?;
    output.extend_from_slice(&len.to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding<'a>(map: &'a HotkeyMap, key: &str) -> &'a Binding {
        map.bindings
            .iter()
            .find(|binding| binding.key == key)
            .unwrap()
    }

    #[test]
    fn contextual_buttons_report_enabled_state_and_current_action() {
        let normal = buttercup_map(false, false);
        assert!(binding(&normal, "B").enabled);
        assert!(!binding(&normal, "T").enabled);
        assert!(binding(&normal, "-").enabled);
        assert_eq!(binding(&normal, "[").label, "Iris min -");
        assert_eq!(binding(&normal, "F").label, "Scoped view");
        assert!(binding(&normal, ",").enabled);
        assert_eq!(binding(&normal, ",").label, "Inspector panel");
        assert!(normal.bindings.iter().all(|binding| binding.key != "F1"));

        let mouse_driving = buttercup_map(true, true);
        assert!(binding(&mouse_driving, "B").enabled);
        assert!(binding(&mouse_driving, "T").enabled);
        assert!(!binding(&mouse_driving, "-").enabled);
        assert!(!binding(&mouse_driving, ",").enabled);
        assert_eq!(binding(&mouse_driving, "[").label, "Lightbox -");
    }

    #[test]
    fn state_byte_is_part_of_the_language_neutral_wire_format() {
        let map = HotkeyMap {
            title: "Test",
            bindings: vec![Binding {
                modifiers: 0,
                enabled: false,
                key: "B",
                label: "Brush",
            }],
        };
        let payload = encode_map(&map).unwrap();
        // title length + title + binding count + modifier byte
        let state_offset = 2 + map.title.len() + 2 + 1;
        assert_eq!(payload[state_offset], 0);
    }
}
