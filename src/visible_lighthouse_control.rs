use std::env;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const IO_TIMEOUT: Duration = Duration::from_millis(150);
const START_TIMEOUT: Duration = Duration::from_secs(3);

pub fn socket_path() -> PathBuf {
    if let Some(path) = env::var_os("VISIBLE_LIGHTHOUSE_SOCKET") {
        return PathBuf::from(path);
    }
    if let Some(runtime) = env::var_os("XDG_RUNTIME_DIR") {
        return PathBuf::from(runtime).join("visible-lighthouse.sock");
    }
    PathBuf::from(format!("/run/user/{}/visible-lighthouse.sock", unsafe {
        libc_getuid()
    }))
}

// Avoid adding a crate dependency merely to obtain the user-scoped runtime fallback.
unsafe extern "C" {
    #[link_name = "getuid"]
    fn libc_getuid() -> u32;
}

pub fn command(path: &Path, request: &str) -> Result<String, String> {
    let mut stream = UnixStream::connect(path)
        .map_err(|error| format!("connect {}: {error}", path.display()))?;
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .map_err(|error| format!("set lighthouse read timeout: {error}"))?;
    stream
        .set_write_timeout(Some(IO_TIMEOUT))
        .map_err(|error| format!("set lighthouse write timeout: {error}"))?;
    stream
        .write_all(format!("{request}\n").as_bytes())
        .map_err(|error| format!("send lighthouse {request}: {error}"))?;
    let mut response = [0u8; 128];
    let count = stream
        .read(&mut response)
        .map_err(|error| format!("read lighthouse {request}: {error}"))?;
    let response = String::from_utf8_lossy(&response[..count])
        .trim()
        .to_string();
    if response.starts_with("error") || response.is_empty() {
        Err(format!("lighthouse {request} returned {response:?}"))
    } else {
        Ok(response)
    }
}

pub fn enabled(path: &Path) -> Result<bool, String> {
    match command(path, "status")?.as_str() {
        "on" => Ok(true),
        "off" => Ok(false),
        response => Err(format!("unexpected lighthouse status {response:?}")),
    }
}

pub fn toggle_or_start() -> Result<bool, String> {
    let path = socket_path();
    match enabled(&path) {
        Ok(true) => {
            command(&path, "off")?;
            Ok(false)
        }
        Ok(false) => {
            command(&path, "on")?;
            Ok(true)
        }
        Err(_) => {
            let launcher = env::var_os("BUTTERCUP_VISIBLE_LIGHTHOUSE_LAUNCHER")
                .or_else(|| env::var_os("VISIBLE_LIGHTHOUSE_LAUNCHER"))
                .map(PathBuf::from)
                .ok_or_else(|| {
                    format!(
                        "lighthouse service is unavailable at {}; start it externally or set BUTTERCUP_VISIBLE_LIGHTHOUSE_LAUNCHER",
                        path.display()
                    )
                })?;
            Command::new(&launcher)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|error| format!("start {}: {error}", launcher.display()))?;
            let deadline = Instant::now() + START_TIMEOUT;
            loop {
                if enabled(&path).is_ok() {
                    command(&path, "on")?;
                    return Ok(true);
                }
                if Instant::now() >= deadline {
                    return Err(format!(
                        "lighthouse did not bind {} within {:.1}s",
                        path.display(),
                        START_TIMEOUT.as_secs_f64()
                    ));
                }
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

pub fn toggle_or_start_async(origin: &'static str) {
    let _ = thread::Builder::new()
        .name("visible-lighthouse-toggle".to_string())
        .spawn(move || match toggle_or_start() {
            Ok(active) => eprintln!(
                "visible lighthouse {} by {origin}",
                if active { "enabled" } else { "disabled" }
            ),
            Err(error) => eprintln!("visible lighthouse toggle by {origin} failed: {error}"),
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_socket_path_wins() {
        let previous = env::var_os("VISIBLE_LIGHTHOUSE_SOCKET");
        env::set_var(
            "VISIBLE_LIGHTHOUSE_SOCKET",
            "/tmp/test-visible-lighthouse.sock",
        );
        assert_eq!(
            socket_path(),
            PathBuf::from("/tmp/test-visible-lighthouse.sock")
        );
        match previous {
            Some(value) => env::set_var("VISIBLE_LIGHTHOUSE_SOCKET", value),
            None => env::remove_var("VISIBLE_LIGHTHOUSE_SOCKET"),
        }
    }
}
