//! Sway IPC boundary. Only a dwell-approved current-workspace `focus` is a
//! mutating request; startup, geometry and visibility checks are read-only.
use super::{Backend, Hit, Rect, Target};
use serde_json::Value;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

pub(super) struct Sway {
    socket: PathBuf,
    pub output: String,
}

fn connect_bounded(path: &std::path::Path) -> std::io::Result<UnixStream> {
    // A synchronous Unix connect can block when the compositor accept queue
    // is full. Nonblocking connect makes emergency OFF's worker join bounded.
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    let bytes = path.as_os_str().as_bytes();
    if bytes.len() >= address.sun_path.len() || bytes.contains(&0) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid Sway socket path",
        ));
    }
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (slot, byte) in address.sun_path.iter_mut().zip(bytes) {
        *slot = *byte as libc::c_char;
    }
    // SAFETY: socket returns a new owned fd; FromRawFd takes ownership once.
    let fd = unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
            0,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let stream = unsafe { UnixStream::from_raw_fd(fd) };
    let length =
        (std::mem::offset_of!(libc::sockaddr_un, sun_path) + bytes.len() + 1) as libc::socklen_t;
    // SAFETY: initialized sockaddr of the specified length; fd stays live.
    if unsafe { libc::connect(fd, (&address as *const libc::sockaddr_un).cast(), length) } < 0 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EINPROGRESS) {
            return Err(error);
        }
        let mut poll = libc::pollfd {
            fd: stream.as_raw_fd(),
            events: libc::POLLOUT,
            revents: 0,
        };
        // SAFETY: one initialized pollfd and finite millisecond deadline.
        let ready = unsafe { libc::poll(&mut poll, 1, 250) };
        if ready < 0 {
            return Err(std::io::Error::last_os_error());
        }
        if ready == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "Sway connect timed out",
            ));
        }
        if let Some(error) = stream.take_error()? {
            return Err(error);
        }
    }
    stream.set_nonblocking(false)?;
    Ok(stream)
}

fn request(socket: &std::path::Path, kind: u32, payload: &str) -> Result<Value, String> {
    let send = || -> Result<Value, Box<dyn std::error::Error>> {
        let mut stream = connect_bounded(socket)?;
        stream.set_read_timeout(Some(Duration::from_millis(250)))?;
        stream.set_write_timeout(Some(Duration::from_millis(250)))?;
        let mut header = *b"i3-ipc\0\0\0\0\0\0\0\0";
        header[6..10].copy_from_slice(&(payload.len() as u32).to_ne_bytes());
        header[10..14].copy_from_slice(&kind.to_ne_bytes());
        stream.write_all(&header)?;
        stream.write_all(payload.as_bytes())?;
        stream.read_exact(&mut header)?;
        if &header[..6] != b"i3-ipc" || u32::from_ne_bytes(header[10..14].try_into()?) != kind {
            return Err("invalid Sway IPC header".into());
        }
        let length = u32::from_ne_bytes(header[6..10].try_into()?) as usize;
        if length > 8 * 1024 * 1024 {
            return Err("oversized Sway IPC reply".into());
        }
        let mut body = vec![0; length];
        stream.read_exact(&mut body)?;
        Ok(serde_json::from_slice(&body)?)
    };
    send().map_err(|e| format!("Sway focus connection failed: {e}"))
}

impl Sway {
    pub(super) fn connect() -> Result<Self, String> {
        let socket = std::env::var_os("SWAYSOCK")
            .map(PathBuf::from)
            .ok_or("focus mode requires a running Sway session (SWAYSOCK)")?;
        let outputs = request(&socket, 3, "")?;
        let requested = std::env::var("BUTTERCUP_FOCUS_OUTPUT").ok();
        let output = select_output(&outputs, requested.as_deref())?;
        Ok(Self { socket, output })
    }
}

fn select_output(outputs: &Value, requested: Option<&str>) -> Result<String, String> {
    let active: Vec<_> = outputs
        .as_array()
        .into_iter()
        .flatten()
        .filter(|o| o["active"] == true)
        .filter_map(|o| o["name"].as_str())
        .filter(|name| requested.is_none_or(|wanted| wanted == *name))
        .collect();
    if active.len() != 1 {
        return Err(
            "focus mode needs one active monitor; set BUTTERCUP_FOCUS_OUTPUT for multiple monitors"
                .into(),
        );
    }
    Ok(active[0].into())
}

fn rect(node: &Value) -> Option<Rect> {
    let r = &node["rect"];
    let rect = Rect {
        x: r["x"].as_f64()?,
        y: r["y"].as_f64()?,
        w: r["width"].as_f64()?,
        h: r["height"].as_f64()?,
    };
    (rect.w > 0.0
        && rect.h > 0.0
        && [rect.x, rect.y, rect.w, rect.h]
            .into_iter()
            .all(f64::is_finite))
    .then_some(rect)
}

fn children<'a>(node: &'a Value, key: &str) -> &'a [Value] {
    node[key].as_array().map(Vec::as_slice).unwrap_or(&[])
}

fn focused(node: &Value) -> Option<u64> {
    if node["focused"] == true {
        return node["id"].as_u64();
    }
    children(node, "nodes")
        .iter()
        .chain(children(node, "floating_nodes"))
        .find_map(focused)
}

fn leaves(node: &Value, out: &mut Vec<Target>) {
    if children(node, "nodes").is_empty() && children(node, "floating_nodes").is_empty() {
        if node["visible"] != true
            || (node["pid"].is_null() && node["app_id"].is_null() && node["window"].is_null())
        {
            return;
        }
        if let (Some(id), Some(rect)) = (node["id"].as_u64(), rect(node)) {
            out.push(Target { id, rect });
        }
    } else {
        for child in children(node, "nodes")
            .iter()
            .chain(children(node, "floating_nodes"))
        {
            leaves(child, out);
        }
    }
}

fn hit_test(tree: &Value, output_name: &str, uv: (f64, f64)) -> Option<Hit> {
    if ![uv.0, uv.1]
        .into_iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(&v))
    {
        return None;
    }
    let output = children(tree, "nodes")
        .iter()
        .find(|n| n["type"] == "output" && n["name"] == output_name)?;
    let output_rect = rect(output)?;
    let workspace = children(output, "nodes").iter().find(|w| {
        w["type"] == "workspace"
            && w["name"].as_str().is_some()
            && w["name"] == output["current_workspace"]
    })?;
    // Stay on the focused output/workspace, even if another monitor is visible.
    let current_focus = focused(workspace)?;
    let point = (
        output_rect.x + uv.0 * output_rect.w,
        output_rect.y + uv.1 * output_rect.h,
    );
    let mut floating = vec![];
    for node in children(workspace, "floating_nodes") {
        leaves(node, &mut floating);
    }
    let mut candidates: Vec<_> = floating
        .into_iter()
        .filter(|t| t.rect.contains(point, 0.0))
        .collect();
    if candidates.is_empty() {
        for node in children(workspace, "nodes") {
            leaves(node, &mut candidates);
        }
        candidates.retain(|t| t.rect.contains(point, 0.0));
    }
    // Overlapping floating views can have ambiguous stacking in GET_TREE;
    // abstain instead of focusing through an occluder. Also avoid boundary jitter.
    let target = (candidates.len() == 1).then(|| candidates[0]).filter(|t| {
        t.rect
            .contains(point, 12.0_f64.min(t.rect.w.min(t.rect.h) * 0.1))
    });
    Some(Hit {
        workspace: workspace["id"].as_u64()?,
        focused: current_focus,
        target,
    })
}

fn focus_command(target: Target) -> String {
    // The dynamic workspace criterion is rechecked by Sway at execution time:
    // a workspace switch between GET_TREE and this command cannot pull it back.
    // No pointer warp is allowed, including with a user-set WARP_CONTAINER.
    format!(
        "mouse_warping none; [con_id={} workspace=__focused__] focus",
        target.id
    )
}

impl Backend for Sway {
    fn hit(&mut self, point: (f64, f64)) -> Result<Hit, String> {
        let binding = request(&self.socket, 12, "")?;
        if binding["name"] != "default" {
            return Ok(Hit {
                workspace: 0,
                focused: 0,
                target: None,
            });
        }
        let tree = request(&self.socket, 4, "")?;
        Ok(hit_test(&tree, &self.output, point).unwrap_or(Hit {
            workspace: 0,
            focused: 0,
            target: None,
        }))
    }

    fn focus(&mut self, target: Target) -> Result<(), String> {
        let response = request(&self.socket, 0, &focus_command(target))?;
        if response
            .as_array()
            .is_some_and(|r| !r.is_empty() && r.iter().all(|v| v["success"] == true))
        {
            Ok(())
        } else {
            Err("Sway rejected window focus; focus mode disabled".into())
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use serde_json::json;
    fn view(id: u64, x: f64, visible: bool, focused: bool) -> Value {
        json!({"id":id,"pid":100,"visible":visible,"focused":focused,"type":"con",
            "rect":{"x":x,"y":0,"width":500,"height":500},"nodes":[],"floating_nodes":[]})
    }
    fn tree() -> Value {
        json!({"nodes":[{"name":"DP-3","type":"output","current_workspace":"one",
            "rect":{"x":100,"y":0,"width":1000,"height":500},
            "nodes":[{"id":10,"name":"one","type":"workspace","nodes":[view(1,100.0,true,true),view(2,600.0,true,false)],"floating_nodes":[]},
                {"id":20,"name":"hidden","type":"workspace","nodes":[view(3,600.0,false,false)]}]}]})
    }
    #[test]
    fn maps_output_logical_coordinates_without_using_window_size() {
        assert_eq!(
            hit_test(&tree(), "DP-3", (0.75, 0.5))
                .unwrap()
                .target
                .unwrap()
                .id,
            2
        );
        assert_eq!(
            hit_test(&tree(), "DP-3", (0.25, 0.5))
                .unwrap()
                .target
                .unwrap()
                .id,
            1
        );
    }
    #[test]
    fn edges_gaps_offscreen_and_hidden_tabs_are_not_targets() {
        assert!(hit_test(&tree(), "DP-3", (0.501, 0.5))
            .unwrap()
            .target
            .is_none());
        assert!(hit_test(&tree(), "DP-3", (-0.1, 0.5)).is_none());
        assert!(hit_test(&tree(), "DP-3", (f64::NAN, 0.5)).is_none());
        let mut t = tree();
        t["nodes"][0]["nodes"][0]["nodes"][1]["visible"] = json!(false);
        assert!(hit_test(&t, "DP-3", (0.75, 0.5)).unwrap().target.is_none());
    }
    #[test]
    fn floating_views_occlude_tiling_and_ambiguous_overlap_abstains() {
        let mut t = tree();
        t["nodes"][0]["nodes"][0]["floating_nodes"] = json!([view(4, 600.0, true, false)]);
        assert_eq!(
            hit_test(&t, "DP-3", (0.75, 0.5))
                .unwrap()
                .target
                .unwrap()
                .id,
            4
        );
        t["nodes"][0]["nodes"][0]["floating_nodes"] =
            json!([view(4, 600.0, true, false), view(5, 600.0, true, false)]);
        assert!(hit_test(&t, "DP-3", (0.75, 0.5)).unwrap().target.is_none());
    }
    #[test]
    fn inactive_workspace_and_other_output_focus_cannot_be_recruited() {
        let mut t = tree();
        t["nodes"][0]["current_workspace"] = json!("hidden");
        assert!(hit_test(&t, "DP-3", (0.75, 0.5)).is_none());
        assert!(hit_test(&tree(), "DP-4", (0.75, 0.5)).is_none());
    }
    #[test]
    fn focus_command_disables_warp_and_cannot_switch_workspace_or_move_pointer() {
        let command = focus_command(Target {
            id: 42,
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 100.0,
            },
        });
        assert_eq!(
            command,
            "mouse_warping none; [con_id=42 workspace=__focused__] focus"
        );
        for forbidden in [
            "cursor",
            " move ",
            " press",
            "release",
            "workspace number",
            "exec",
        ] {
            assert!(!command.contains(forbidden));
        }
    }
    #[test]
    fn monitor_selection_refuses_ambiguity() {
        let o = json!([{"name":"DP-3","active":true},{"name":"DP-4","active":true}]);
        assert!(select_output(&o, None).is_err());
        assert_eq!(select_output(&o, Some("DP-4")).unwrap(), "DP-4");
    }

    #[test]
    #[ignore = "read-only inspection of the current Sway layout; never changes focus"]
    fn inspect_live_sway_without_focus_commands() {
        let mut backend = Sway::connect().unwrap();
        for x in [0.2, 0.5, 0.8] {
            println!("gaze {x},0.5: {:?}", backend.hit((x, 0.5)).unwrap());
        }
    }
}
