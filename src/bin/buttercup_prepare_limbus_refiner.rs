#[path = "../training_refiner_data.rs"]
mod training_refiner_data;
fn main() {
    if let Err(error) = training_refiner_data::prepare_cli(std::env::args().skip(1).collect()) {
        eprintln!("prepare limbus refiner: {error}");
        std::process::exit(1);
    }
}
