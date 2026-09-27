//! Advice for ordinary TCP failures only. This never probes the camera,
//! changes networking, initializes firmware, or recovers an ownership failure.
use std::{
    io,
    net::SocketAddr,
    sync::Mutex,
    time::{Duration, Instant},
};

const REPEAT_INTERVAL: Duration = Duration::from_secs(30);
// A cooperative session has exactly two allowed endpoints. Keep retry logging
// bounded even when capture/control workers fail concurrently.
static LAST_FAILURES: Mutex<[Option<Failure>; 2]> = Mutex::new([None, None]);

#[derive(Clone, Copy)]
struct Failure {
    endpoint: SocketAddr,
    kind: io::ErrorKind,
    os_error: Option<i32>,
    reported_at: Instant,
}

fn should_report(slots: &mut [Option<Failure>; 2], next: Failure) -> bool {
    let index = slots
        .iter()
        .position(|slot| slot.is_some_and(|last| last.endpoint == next.endpoint));
    if let Some(last) = index.and_then(|i| slots[i]) {
        if last.kind == next.kind
            && last.os_error == next.os_error
            && next.reported_at.saturating_duration_since(last.reported_at) < REPEAT_INTERVAL
        {
            return false;
        }
    }
    let index = index
        .or_else(|| slots.iter().position(Option::is_none))
        .unwrap_or_else(|| {
            usize::from(slots[1].unwrap().reported_at < slots[0].unwrap().reported_at)
        });
    slots[index] = Some(next);
    true
}

fn recovery_hint(error: &io::Error) -> &'static str {
    match error.kind() {
        io::ErrorKind::ConnectionRefused =>
            "The endpoint refused TCP; the Podbay RAW/control service may not be running. Network/SSH bootstrap alone does not start it. In the separate Podbay checkout, use tools/deploy.py deploy with the prepared kernel tree and --accept-camera-changes (full command in docs/camera-startup.md).",
        io::ErrorKind::TimedOut | io::ErrorKind::AddrNotAvailable =>
            "Check the camera route, cable/power and host address. After a camera power cycle its temporary USB network, SSH and RAW service must be restored: in the separate Podbay checkout run python3 tools/pw203_bootstrap_ssh.py --accept-camera-bootstrap, then the documented RAM-only deployment. A timeout alone does not identify which stage is missing.",
        io::ErrorKind::PermissionDenied =>
            "Check local socket permissions and firewall policy. This error does not establish that the camera needs reinitialization.",
        _ if matches!(error.raw_os_error(), Some(libc::ENETUNREACH | libc::EHOSTUNREACH | libc::ENETDOWN)) =>
            "The camera network is unavailable. Restore its USB Ethernet interface and host address with Podbay tools/pw203_bootstrap_ssh.py --accept-camera-bootstrap, then deploy the RAM-only RAW service if the camera was power-cycled. See docs/camera-startup.md for the separate Podbay workflow.",
        _ =>
            "Inspect the transport error and camera service log before retrying. Camera networking, the Podbay RAW service and UPC ownership are separate startup stages; see docs/camera-startup.md.",
    }
}

pub(super) fn connection_failed(endpoint: SocketAddr, timeout: Duration, error: &io::Error) {
    let next = Failure {
        endpoint,
        kind: error.kind(),
        os_error: error.raw_os_error(),
        reported_at: Instant::now(),
    };
    // A poisoned diagnostic throttle cannot suppress the original I/O error.
    let report = should_report(
        &mut LAST_FAILURES.lock().unwrap_or_else(|e| e.into_inner()),
        next,
    );
    if !report {
        return;
    }
    eprintln!("CAMERA TCP UNAVAILABLE: endpoint={endpoint} kind={:?} os_error={:?} connect_timeout_ms={} error={error}",
        error.kind(), error.raw_os_error(), timeout.as_millis());
    eprintln!("CAMERA RECOVERY: {}", recovery_hint(error));
    eprintln!("CAMERA NETWORK CHECK: ip -brief address; ip route get {}. No recovery command was executed; the original transport error is returned. See docs/camera-startup.md.", endpoint.ip());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_worker_failures_are_bounded_without_hiding_new_causes_or_endpoints() {
        let mut slots = [None, None];
        let first = Failure {
            endpoint: "127.0.0.1:5001".parse().unwrap(),
            kind: io::ErrorKind::TimedOut,
            os_error: Some(libc::ETIMEDOUT),
            reported_at: Instant::now(),
        };
        assert!(should_report(&mut slots, first));
        for ms in 1..30_000 {
            assert!(!should_report(
                &mut slots,
                Failure {
                    reported_at: first.reported_at + Duration::from_millis(ms),
                    ..first
                }
            ));
        }
        assert!(should_report(
            &mut slots,
            Failure {
                endpoint: "127.0.0.1:5002".parse().unwrap(),
                ..first
            }
        ));
        assert!(should_report(
            &mut slots,
            Failure {
                kind: io::ErrorKind::ConnectionRefused,
                os_error: Some(libc::ECONNREFUSED),
                ..first
            }
        ));
        assert!(should_report(
            &mut slots,
            Failure {
                reported_at: first.reported_at + REPEAT_INTERVAL,
                ..first
            }
        ));
    }
}
