# Camera startup and recovery

Buttercup consumes an external Podbay TCP service. Its usual endpoints are
`192.168.88.10:5001` for RAW and `192.168.88.10:5002` for control. A host reboot
stops the viewer and can lose host interface configuration. A camera power
cycle also removes Podbay's temporary network, SSH and RAW service. Restoring
SSH alone does not restore the RAW service.

## Read the failure stage first

| Diagnostic | What it establishes | Next step |
| --- | --- | --- |
| `CAMERA OWNERSHIP STARTUP REFUSED` | UPC could not authorize a session; no camera TCP fallback is allowed. | Restore the UPC publisher and a fresh matching camera descriptor, or let the current owner finish. Use the printed runtime path and underlying error. |
| `FATAL ASSERTION: UPC camera ownership invariant violated` | Initialization or live ownership is invalid; the entire process aborts. | Restore the publisher/attachment and restart the viewer with a new session. Never delete the runtime, replace its lock, or bypass ownership to reconnect. |
| `CAMERA TCP UNAVAILABLE` with connection refused | The selected endpoint refused the connection; its service may not be listening. | Check the Podbay RAW/control service and the configured ports. If camera power was lost, restore the RAM-only service after network/SSH bootstrap. |
| `CAMERA TCP UNAVAILABLE` with timeout or unreachable network | The endpoint could not be reached; this does not prove which startup stage failed. | Check power/cable, the camera network interface, its host address, route and filtering. Restore the missing stages below. |
| TCP works but protocol/setup fails | A listening socket alone does not prove a compatible RAW service. | Read the original setup/protocol error and Podbay service log; verify the supported service version before relaunching. |

Connection diagnostics include the endpoint, error kind, OS error and configured
timeout. Identical failures are reported at most once per endpoint every 30
seconds; a changed error is reported immediately. The original I/O error remains
unchanged. Ownership assertions run before and after each connection attempt,
including unsuccessful attempts. Diagnostics execute no recovery commands.

## Restore the external service after a power cycle

Start with host network inspection:

```sh
ip -brief address
ip route get 192.168.88.10
```

For the usual direct USB network, the host address is `192.168.88.20/24` on the
camera interface. A route through ordinary Wi-Fi/Ethernet is a reason to inspect
that configuration; it is not proof of camera health. If an override selects a
different camera address, inspect that address instead.

If the interface exists but ARP fails and its RX packet count stays at zero,
inspect the host USB network driver as well as the address. On this workstation
after the September 14 reboot, `cdc_subset` had claimed only the camera's data
interface, leaving its RNDIS control interface unbound. Bootstrap completed the
USB switch but timed out waiting for HTTP, despite the correct host IP and route.
The camera descriptors and the installed `rndis_host` module alias both matched
class/subclass/protocol `02/02/ff`.

The successful host-side repair released only that camera's data interface from
`cdc_subset` and bound its control interface to the already installed
`rndis_host`; RNDIS then claimed both interfaces. The network name and MAC changed,
so bootstrap was rerun to configure the newly created interface. ARP, HTTP and
strict SSH then succeeded. This is a host driver-binding repair, not another
camera power cycle or firmware deployment. Confirm the actual USB identity,
interface descriptors and current binding before applying that repair; bus,
interface and network names vary between connections. No host-driver control
code or persistent driver blacklist is part of Buttercup.

Read-only checks, using the current camera interface name:

```sh
ip -s link show dev CAMERA_INTERFACE
ip neigh show 192.168.88.10
readlink -f /sys/class/net/CAMERA_INTERFACE/device/driver
lsusb -t
```

Read the **separate Podbay repository's README.md** and run its documented tools
from that checkout. They own camera initialization; Buttercup never imports or
executes them and does not require that source checkout at runtime.

```sh
python3 tools/pw203_bootstrap_ssh.py --dry-run
python3 tools/pw203_bootstrap_ssh.py --accept-camera-bootstrap
```

This restores USB networking and temporary key-only SSH. It requires an unowned,
healthy camera and pins the temporary SSH host key in
`/tmp/podbay-pw203-known-hosts`. If the host rebooted while camera SSH survived but
that trust file did not, use a camera power cycle and bootstrap again; do not
disable SSH host-key verification.

If bootstrap reports a short firmware response (for example, one byte instead
of 60), Podbay's `docs/SENSOR_PROVIDER.md` prescribes physical USB power removal
before another attempt. Stop there; repeated handshakes or a software `usbreset`
are not the documented recovery.

Once bootstrap succeeds, prepare or validate Podbay's external kernel tree and
deploy its service using the README's guarded workflow:

```sh
podbay_kernel_build="$(tools/prepare_infinity6c_kernel.sh)"
python3 tools/deploy.py --key ~/.ssh/id_ed25519 deploy \
  --kernel-build "$podbay_kernel_build" --accept-camera-changes
```

The supported deployment is RAM-only. Leave persistent boot-hook installation
out of recovery. The tools validate device identity and use external build
storage; no firmware or generated camera artifacts belong in Buttercup.

If UPC is installed, restore its publisher/descriptor for the current attachment
before starting Buttercup. An existing stale or unavailable UPC runtime refuses
startup. An absent runtime permits the existing standalone path, whose watchdog
still aborts if UPC appears later.

Restart the viewer after the service is ready. `--camera-cooperation-check`
checks ownership only; it does not connect to the camera or certify frames.
Confirm fresh eye images before pressing **M** for calibration. A restored saved
calibration is not evidence that a new nine-point run has passed.
