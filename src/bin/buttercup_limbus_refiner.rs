//! Supervised native-RAW local limbus model; no camera IO.
#![allow(dead_code)]
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
#[path = "../limbus_refiner.rs"]
mod limbus_refiner;
fn main() {
    #[cfg(feature = "sam31")]
    if let Err(error) = limbus_refiner::training::run_cli(std::env::args().skip(1)) {
        eprintln!("limbus refiner: {error}");
        std::process::exit(1);
    }
    #[cfg(not(feature = "sam31"))]
    {
        eprintln!("training requires --features sam31 for CUDA/LibTorch");
        std::process::exit(1);
    }
}
