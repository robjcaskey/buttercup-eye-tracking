// Separate illustration variant, sharing RAW preparation and provenance.
#[allow(dead_code)]
mod shared {
    include!("buttercup_pink_waterfall_sheet.rs");
    pub fn meridian() {
        prepare(true);
    }
}
fn main() {
    shared::meridian();
}
