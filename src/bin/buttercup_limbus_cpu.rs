//! Cold CPU training from canonical band annotations; no custom teacher or camera.
#[path = "../bootstrapability.rs"]
mod bootstrapability;
#[path = "../limbus_refiner_cpu.rs"]
mod limbus_refiner_cpu;
#[path = "../training_refiner_data.rs"]
mod training_refiner_data;
fn main() {
    if let Err(error) = limbus_refiner_cpu::run(std::env::args().skip(1).collect()) {
        eprintln!("limbus CPU: {error}");
        std::process::exit(1);
    }
}
