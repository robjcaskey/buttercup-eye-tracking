//! Minimal userspace Linux uinput ABI. The kernel supplies the driver.
//! ABS_X/Y + BTN_LEFT classifies as an absolute mouse in udev/libinput. The
//! button capability is for classification ONLY: we never send button events.
use super::{Pointer, AXIS_MAX, DEVICE_NAME};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;

const UINPUT: u32 = b'U' as u32;
const EV_SYN: u16 = 0;
const EV_KEY: u16 = 1;
const EV_ABS: u16 = 3;
const SYN_REPORT: u16 = 0;
const ABS_X: u16 = 0;
const ABS_Y: u16 = 1;
const BTN_LEFT: u16 = 0x110;
const BUS_VIRTUAL: u16 = 0x06;

pub(crate) struct UinputPointer(File);

impl UinputPointer {
    pub(crate) fn create() -> io::Result<Self> {
        // Actually opening is the permission check: this respects ACLs,
        // supplementary groups and device policies, unlike checking mode bits.
        let file = OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open("/dev/uinput")?;
        for (operation, bit) in [
            (100, EV_KEY),
            (100, EV_ABS),
            (101, BTN_LEFT),
            (103, ABS_X),
            (103, ABS_Y),
            (110, libc::INPUT_PROP_POINTER),
        ] {
            // SAFETY: owned live fd; these ioctls accept an integer bit value.
            checked(unsafe {
                libc::ioctl(
                    file.as_raw_fd(),
                    libc::_IOW::<libc::c_int>(UINPUT, operation),
                    bit as libc::c_int,
                )
            })?;
        }
        for code in [ABS_X, ABS_Y] {
            // Zero-initialize the ABI padding as well as the defined fields.
            let mut setup: libc::uinput_abs_setup = unsafe { std::mem::zeroed() };
            setup.code = code;
            setup.absinfo.maximum = AXIS_MAX;
            // SAFETY: correct kernel ABI type, initialized and alive for ioctl.
            checked(unsafe {
                libc::ioctl(
                    file.as_raw_fd(),
                    libc::_IOW::<libc::uinput_abs_setup>(UINPUT, 4),
                    &setup,
                )
            })?;
        }
        let mut setup: libc::uinput_setup = unsafe { std::mem::zeroed() };
        setup.id.bustype = BUS_VIRTUAL;
        setup.id.version = 1;
        for (slot, byte) in setup.name.iter_mut().zip(DEVICE_NAME.bytes()) {
            *slot = byte as libc::c_char;
        }
        // SAFETY: initialized ABI buffer then argumentless create on owned fd.
        checked(unsafe {
            libc::ioctl(
                file.as_raw_fd(),
                libc::_IOW::<libc::uinput_setup>(UINPUT, 3),
                &setup,
            )
        })?;
        checked(unsafe { libc::ioctl(file.as_raw_fd(), libc::_IO(UINPUT, 1)) })?;
        Ok(Self(file))
    }
}

fn checked(result: libc::c_int) -> io::Result<()> {
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn position_events(point: [i32; 2]) -> [libc::input_event; 3] {
    // The kernel ignores uinput timestamps. Zero also initializes ABI padding.
    let mut events: [libc::input_event; 3] = unsafe { std::mem::zeroed() };
    for (event, (kind, code, value)) in events.iter_mut().zip([
        (EV_ABS, ABS_X, point[0]),
        (EV_ABS, ABS_Y, point[1]),
        (EV_SYN, SYN_REPORT, 0),
    ]) {
        event.type_ = kind;
        event.code = code;
        event.value = value;
    }
    events
}

impl Pointer for UinputPointer {
    fn position(&mut self, point: [i32; 2]) -> io::Result<()> {
        let events = position_events(point);
        // SAFETY: fully initialized contiguous kernel ABI records, alive until
        // the synchronous write returns. Never serialize keys/buttons here.
        let bytes = unsafe {
            std::slice::from_raw_parts(events.as_ptr().cast::<u8>(), std::mem::size_of_val(&events))
        };
        match self.0.write(bytes) {
            Ok(n) if n == bytes.len() => Ok(()),
            Ok(_) => Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "incomplete absolute pointer report",
            )),
            Err(e) => Err(e),
        }
    }
}

impl Drop for UinputPointer {
    fn drop(&mut self) {
        // SAFETY: argumentless destroy on an owned created uinput descriptor.
        // Closing the fd also destroys the device if explicit destroy fails.
        unsafe {
            libc::ioctl(self.0.as_raw_fd(), libc::_IO(UINPUT, 2));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn report_is_absolute_xy_and_sync_only() {
        let events = position_events([100, 200]);
        let triples: Vec<_> = events.iter().map(|e| (e.type_, e.code, e.value)).collect();
        assert_eq!(
            triples,
            [
                (EV_ABS, ABS_X, 100),
                (EV_ABS, ABS_Y, 200),
                (EV_SYN, SYN_REPORT, 0)
            ]
        );
    }
    #[test]
    #[ignore = "creates a real uinput device without emitting events; inspect with swaymsg get_inputs"]
    fn register_device_without_moving_pointer() {
        let _device = UinputPointer::create().unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}
