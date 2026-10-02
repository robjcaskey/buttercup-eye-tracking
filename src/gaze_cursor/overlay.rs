//! wlr-layer-shell OVERLAY surface for the gaze ring. Sway draws fullscreen
//! windows above every floating window, so a regular toplevel ring vanished
//! in fullscreen; the overlay layer stays above them. The surface takes no
//! keyboard focus and has an empty input region, so it is fully click-through.
//! It runs on its own Wayland connection and thread; the viewer only sends the
//! latest normalized target (or `None` to hide) over a channel.
use super::{render_ring, SIZE};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, Region},
    delegate_compositor, delegate_layer, delegate_output, delegate_registry, delegate_shm,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        wlr_layer::{Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler,
            LayerSurface, LayerSurfaceConfigure},
        WaylandSurface,
    },
    shm::{slot::{Buffer, SlotPool}, Shm, ShmHandler},
};
use std::os::fd::AsRawFd;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_output, wl_shm, wl_surface},
    Connection, QueueHandle,
};

const NAMESPACE: &str = "buttercup-gaze-cursor";
/// Integer buffer scale: crisp at Sway's fractional scales once downsampled.
const BUFFER_SCALE: i32 = 2;
/// Wake at least this often to pick up new targets.
const POLL: Duration = Duration::from_millis(10);

/// Handle owned by the viewer. Dropping it closes the overlay.
pub(crate) struct CursorOverlay {
    targets: mpsc::Sender<Option<(f64, f64)>>,
    error: Arc<Mutex<Option<String>>>,
}

impl CursorOverlay {
    /// `output_name` selects the Sway output whose logical rectangle the
    /// normalized target refers to; `None` lets the compositor choose.
    pub(crate) fn spawn(output_name: Option<String>) -> Self {
        let (targets, receiver) = mpsc::channel();
        let error = Arc::new(Mutex::new(None));
        let report = Arc::clone(&error);
        let spawned = std::thread::Builder::new().name("gaze-cursor-overlay".into()).spawn(move || {
            if let Err(e) = run(output_name, receiver) {
                eprintln!("gaze cursor overlay: {e}");
                if let Ok(mut slot) = report.lock() { *slot = Some(e); }
            }
        });
        if let Err(e) = spawned {
            *error.lock().unwrap() = Some(format!("gaze cursor overlay thread: {e}"));
        }
        Self { targets, error }
    }

    pub(crate) fn show(&self, target: Option<(f64, f64)>) {
        let _ = self.targets.send(target);
    }

    pub(crate) fn error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|e| e.clone())
    }
}

struct State {
    registry: RegistryState,
    outputs: OutputState,
    shm: Shm,
    configured: bool,
    closed: bool,
}

fn run(output_name: Option<String>, targets: mpsc::Receiver<Option<(f64, f64)>>) -> Result<(), String> {
    let conn = Connection::connect_to_env().map_err(|e| format!("wayland connect: {e}"))?;
    let (globals, mut queue) = registry_queue_init(&conn).map_err(|e| format!("registry: {e}"))?;
    let qh = queue.handle();
    let compositor = CompositorState::bind(&globals, &qh).map_err(|e| format!("wl_compositor: {e}"))?;
    let layer_shell = LayerShell::bind(&globals, &qh).map_err(|e| format!("layer shell: {e}"))?;
    let shm = Shm::bind(&globals, &qh).map_err(|e| format!("wl_shm: {e}"))?;
    let mut state = State {
        registry: RegistryState::new(&globals),
        outputs: OutputState::new(&globals, &qh),
        shm,
        configured: false,
        closed: false,
    };
    // Learn output names and logical sizes before choosing one.
    queue.roundtrip(&mut state).map_err(|e| format!("roundtrip: {e}"))?;
    queue.roundtrip(&mut state).map_err(|e| format!("roundtrip: {e}"))?;
    let output = state.outputs.outputs().find(|o| {
        output_name.is_some() && state.outputs.info(o).and_then(|i| i.name) == output_name
    }).or_else(|| state.outputs.outputs().next());
    let logical = output.as_ref().and_then(|o| state.outputs.info(o)).and_then(|i| i.logical_size)
        .ok_or("no output logical size")?;

    let surface = compositor.create_surface(&qh);
    surface.set_buffer_scale(BUFFER_SCALE);
    let empty = Region::new(&compositor).map_err(|e| format!("wl_region: {e}"))?;
    surface.set_input_region(Some(empty.wl_region()));
    let layer = layer_shell.create_layer_surface(&qh, surface, Layer::Overlay, Some(NAMESPACE), output.as_ref());
    layer.set_anchor(Anchor::TOP | Anchor::LEFT);
    // Margins are measured from the output edge, ignoring panels.
    layer.set_exclusive_zone(-1);
    layer.set_keyboard_interactivity(KeyboardInteractivity::None);
    layer.set_size(SIZE as u32, SIZE as u32);
    layer.commit();

    // Content never changes, so both buffers are drawn once and reattached.
    let side = SIZE as i32 * BUFFER_SCALE;
    let mut pool = SlotPool::new((side * side * 4 * 2) as usize, &state.shm).map_err(|e| format!("shm pool: {e}"))?;
    let mut buffer = |ring: bool| -> Result<Buffer, String> {
        let (buffer, canvas) = pool.create_buffer(side, side, side * 4, wl_shm::Format::Argb8888)
            .map_err(|e| format!("shm buffer: {e}"))?;
        let mut pixels = vec![0u32; (side * side) as usize];
        if ring { render_ring(&mut pixels, side as usize, side as usize); }
        for (chunk, pixel) in canvas.chunks_exact_mut(4).zip(pixels) {
            chunk.copy_from_slice(&pixel.to_le_bytes());
        }
        Ok(buffer)
    };
    let (ring, blank) = (buffer(true)?, buffer(false)?);

    let mut wanted: Option<Option<(f64, f64)>> = None;
    let mut shown: Option<Option<(i32, i32)>> = None;
    loop {
        conn.flush().map_err(|e| format!("flush: {e}"))?;
        if let Some(guard) = queue.prepare_read() {
            let mut fd = libc::pollfd { fd: guard.connection_fd().as_raw_fd(), events: libc::POLLIN, revents: 0 };
            // SAFETY: one valid pollfd for the duration of the call.
            if unsafe { libc::poll(&mut fd, 1, POLL.as_millis() as i32) } > 0 {
                guard.read().map_err(|e| format!("read: {e}"))?;
            }
        }
        queue.dispatch_pending(&mut state).map_err(|e| format!("dispatch: {e}"))?;
        if state.closed { return Err("overlay closed by compositor".into()); }
        loop {
            match targets.try_recv() {
                Ok(target) => wanted = Some(target),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            }
        }
        let Some(target) = wanted.filter(|_| state.configured) else { continue };
        let place = target.map(|(x, y)| (
            (x.clamp(0.0, 1.0) * f64::from(logical.0) - SIZE / 2.0).round() as i32,
            (y.clamp(0.0, 1.0) * f64::from(logical.1) - SIZE / 2.0).round() as i32,
        ));
        let update = match (shown, place) {
            (None, _) => true,
            (Some(None), None) => false,
            (Some(Some(a)), Some(b)) => f64::from((a.0 - b.0).abs().max((a.1 - b.1).abs())) >= super::MOVE_THRESHOLD,
            _ => true,
        };
        if !update { continue; }
        let wl = layer.wl_surface();
        let showing = matches!(shown, Some(Some(_)));
        if let Some((px, py)) = place { layer.set_margin(py, 0, 0, px); }
        if place.is_some() != showing || shown.is_none() {
            wl.attach(Some(if place.is_some() { ring.wl_buffer() } else { blank.wl_buffer() }), 0, 0);
            wl.damage_buffer(0, 0, side, side);
        }
        layer.commit();
        shown = Some(place);
    }
}

impl CompositorHandler for State {
    fn scale_factor_changed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: i32) {}
    fn transform_changed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: wl_output::Transform) {}
    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {}
    fn surface_enter(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: &wl_output::WlOutput) {}
    fn surface_leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: &wl_output::WlOutput) {}
}

impl OutputHandler for State {
    fn output_state(&mut self) -> &mut OutputState { &mut self.outputs }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl LayerShellHandler for State {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) { self.closed = true; }
    fn configure(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface, _: LayerSurfaceConfigure, _: u32) {
        self.configured = true;
    }
}

impl ShmHandler for State {
    fn shm_state(&mut self) -> &mut Shm { &mut self.shm }
}

impl ProvidesRegistryState for State {
    fn registry(&mut self) -> &mut RegistryState { &mut self.registry }
    registry_handlers![OutputState];
}

delegate_compositor!(State);
delegate_output!(State);
delegate_shm!(State);
delegate_layer!(State);
delegate_registry!(State);
