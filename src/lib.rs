//! Host-side native eye inference. Camera acquisition is an external service;
//! callers supply source-identified native samples to the bounded workers.
#![allow(dead_code)]
mod conic_solver;
pub mod geometry;
pub mod limbus_refiner;
mod outline_conic_segments;
pub mod raw10;
pub mod roi_continuity;
mod roi_evidence;
pub mod roi_visibility;
pub mod sam31_outer;
pub use sam31_outer::student::raw as raw_student_input;
// The shared inference module's opt-in corpus tests also exercise gaze state.
#[cfg(test)]
mod eye_scene_model;
#[cfg(test)]
mod raw_iris_focus;
