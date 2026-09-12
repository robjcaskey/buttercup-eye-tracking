//! CUDA teacher export, student training, and matched evaluation. No camera IO.
#![allow(dead_code)]
#![recursion_limit = "256"]
#[path = "../geometry.rs"]
mod geometry;
#[path = "../raw10.rs"]
mod raw10;
#[path = "../roi_continuity.rs"]
mod roi_continuity;
#[path = "../roi_visibility.rs"]
mod roi_visibility;
#[path = "../sam31_outer.rs"]
mod sam31_outer;
#[path = "../"]
mod native {
    pub(crate) mod conic_solver;
    pub(crate) mod outline_conic_segments;
    pub(crate) mod roi_evidence;
}
use native::{conic_solver, outline_conic_segments, roi_evidence};
fn main() {
    #[cfg(feature = "sam31")]
    if let Err(error) = sam31_outer::student::run_cli(std::env::args().skip(1)) {
        eprintln!("eye student: {error}");
        std::process::exit(1);
    }
    #[cfg(not(feature = "sam31"))]
    {
        eprintln!("eye student requires --features sam31 (CUDA/LibTorch)");
        std::process::exit(1);
    }
}
