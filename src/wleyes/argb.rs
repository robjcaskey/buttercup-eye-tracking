//! Tiny ARGB presenter for the existing winit window. Softbuffer's Wayland
//! XRGB format discards alpha, so this overlay needs its own shared-memory pair.
use std::{fs::File,os::{fd::{AsFd,FromRawFd},unix::fs::FileExt},sync::{Arc,atomic::{AtomicBool,Ordering}}};
use wayland_client::{backend::{Backend,ObjectId},globals::{registry_queue_init,GlobalListContents},
    protocol::{wl_buffer,wl_registry,wl_shm,wl_surface},Connection,Dispatch,EventQueue,Proxy,QueueHandle};
use winit::{raw_window_handle::{HasDisplayHandle,HasWindowHandle,RawDisplayHandle,RawWindowHandle},window::Window};

pub(crate) struct ArgbWindow {
    surface: wl_surface::WlSurface,
    buffers: Vec<Buffer>,
    queue: EventQueue<State>,
    shm: wl_shm::WlShm,
    conn: Connection,
    next_redraw: std::time::Instant,
    // Foreign display/surface handles are borrowed from this owner. Drop last.
    pub window: Arc<Window>,
}
struct Buffer { proxy: wl_buffer::WlBuffer, file: File, size:(u32,u32), released:Arc<AtomicBool> }
impl Drop for Buffer {fn drop(&mut self){self.proxy.destroy();}}
impl ArgbWindow {
    pub fn new(window:Arc<Window>)->Result<Self,String>{
        let RawDisplayHandle::Wayland(display)=window.display_handle().map_err(|e|e.to_string())?.as_raw() else{return Err("wleyes needs Wayland".into());};
        let RawWindowHandle::Wayland(handle)=window.window_handle().map_err(|e|e.to_string())?.as_raw() else{return Err("wleyes needs a Wayland surface".into());};
        // The retained winit window and owning App outlive every borrowed proxy.
        let conn=Connection::from_backend(unsafe{Backend::from_foreign_display(display.display.as_ptr().cast())});
        let (globals,queue)=registry_queue_init::<State>(&conn).map_err(|e|e.to_string())?;
        let shm=globals.bind(&queue.handle(),1..=1,()).map_err(|e|e.to_string())?;
        let id=unsafe{ObjectId::from_ptr(wl_surface::WlSurface::interface(),handle.surface.as_ptr().cast())}.map_err(|e|e.to_string())?;
        let surface=wl_surface::WlSurface::from_id(&conn,id).map_err(|e|e.to_string())?;
        surface.set_opaque_region(None);
        Ok(Self{surface,buffers:Vec::new(),queue,shm,conn,next_redraw:std::time::Instant::now(),window})
    }
    pub fn request_redraw(&mut self){let now=std::time::Instant::now();if now>=self.next_redraw {
        self.next_redraw=now+std::time::Duration::from_millis(33);self.window.request_redraw();
    }}
    pub fn present(&mut self,pixels:&[u32],width:u32,height:u32)->Result<(),String>{
        if width==0 || height==0 || width>4096 || height>4096 || pixels.len()!=width as usize*height as usize {return Err("invalid wleyes dimensions".into());}
        self.queue.dispatch_pending(&mut State).map_err(|e|e.to_string())?;
        let index=self.buffers.iter().position(|b|b.released.load(Ordering::Acquire));
        let index=match index {
            Some(i)=>i,
            None if self.buffers.len()<2=>{
                self.buffers.push(Buffer::new(&self.shm,&self.queue.handle(),width,height)?);self.buffers.len()-1
            },
            // Never stall camera/UI work on a compositor buffer release.
            None=>return Ok(()),
        };
        if self.buffers[index].size!=(width,height) {self.buffers[index]=Buffer::new(&self.shm,&self.queue.handle(),width,height)?;}
        let buffer=&self.buffers[index];
        let bytes:Vec<u8>=pixels.iter().flat_map(|p|p.to_ne_bytes()).collect();
        buffer.file.write_all_at(&bytes,0).map_err(|e|e.to_string())?;
        buffer.released.store(false,Ordering::Release);
        self.surface.attach(Some(&buffer.proxy),0,0);
        if self.surface.version()>=4 {self.surface.damage_buffer(0,0,width as i32,height as i32);}
        else {self.surface.damage(0,0,i32::MAX,i32::MAX);}
        self.window.pre_present_notify();self.surface.commit();
        self.conn.flush().map_err(|e|e.to_string())
    }
}
impl Buffer {
    fn new(shm:&wl_shm::WlShm,qh:&QueueHandle<State>,w:u32,h:u32)->Result<Self,String>{
        let fd=unsafe{libc::memfd_create(c"buttercup-wleyes".as_ptr(),libc::MFD_CLOEXEC)};
        if fd<0{return Err(std::io::Error::last_os_error().to_string());}
        let file=unsafe{File::from_raw_fd(fd)};
        let bytes=w as u64*h as u64*4;file.set_len(bytes).map_err(|e|e.to_string())?;
        let pool=shm.create_pool(file.as_fd(),bytes as i32,qh,());
        let released=Arc::new(AtomicBool::new(true));
        let proxy=pool.create_buffer(0,w as i32,h as i32,w as i32*4,wl_shm::Format::Argb8888,qh,Arc::clone(&released));
        pool.destroy();
        Ok(Self{proxy,file,size:(w,h),released})
    }
}
struct State;
wayland_client::delegate_noop!(State: ignore wl_shm::WlShm);
wayland_client::delegate_noop!(State: ignore wayland_client::protocol::wl_shm_pool::WlShmPool);
impl Dispatch<wl_registry::WlRegistry,GlobalListContents> for State {
    fn event(_: &mut Self,_:&wl_registry::WlRegistry,_:wl_registry::Event,_:&GlobalListContents,_:&Connection,_:&QueueHandle<Self>){}
}
impl Dispatch<wl_buffer::WlBuffer,Arc<AtomicBool>> for State {
    fn event(_: &mut Self,_:&wl_buffer::WlBuffer,event:wl_buffer::Event,released:&Arc<AtomicBool>,_:&Connection,_:&QueueHandle<Self>){
        if let wl_buffer::Event::Release=event {released.store(true,Ordering::Release);}
    }
}
