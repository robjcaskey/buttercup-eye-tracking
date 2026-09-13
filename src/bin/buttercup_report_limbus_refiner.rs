#[path = "../training_refiner_data.rs"]
mod training_refiner_data;
fn main() {
    if let Err(error) = training_refiner_data::report_cli(std::env::args().skip(1).collect()) {
        eprintln!("report limbus refiner: {error}");
        std::process::exit(1);
    }
}
