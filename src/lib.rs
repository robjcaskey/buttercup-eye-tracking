//! Host-side native eye inference. Camera acquisition is an external service;
//! callers supply source-identified native samples to the bounded workers.
#![allow(dead_code)]
mod conic_solver;
pub mod geometry;
pub mod focus_region;
pub mod limbus_refiner;
mod outline_conic_segments;
pub mod raw10;
pub mod raw_motion_octrees;
pub mod parallel_work;
pub mod raw_sclera_vein_graph;
pub mod raw_sclera_red_canny;
pub mod roi_continuity;
mod roi_evidence;
pub mod roi_visibility;
pub mod sam31_outer;
pub use sam31_outer::student::raw as raw_student_input;
mod eye_scene_model;
// The offline calibration experiment uses the same non-learned RAW boundary
// detector as the viewer; it must not inherit archived custom-model fits.
pub mod raw_iris_focus;

pub mod recorded_bundle;
pub mod calibration_sign_model;
pub mod screen_reflection_code;
pub mod screen_reflection_raw;
