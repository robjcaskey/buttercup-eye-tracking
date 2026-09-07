//! Offline export using the same RAW previews and annotation drawing as live.
use super::*;

pub(super) fn render(
    directory: &Path, index: usize, frame: &sam31_outer::RawFrame,
    proposal: Option<&sam31_outer::ProposalMasks>, contact: Option<SurfaceGazeSample>,
) -> Result<Value, String> {
    const SCALE: usize = 2;
    let width = frame.width * SCALE;
    let height = frame.height * SCALE;
    if proposal.is_some_and(|p| p.source_sequence != frame.sequence
        || p.source_timestamp_ns != frame.timestamp_ns
        || p.source_sensor_origin != (frame.sensor_x, frame.sensor_y)
        || p.source_width != frame.width || p.source_height != frame.height) {
        return Err("showcase proposal does not match source RAW frame".into());
    }
    let color = color_preview(&frame.pixels,frame.width,frame.height,
        frame.sensor_x,frame.sensor_y,100,None);
    let raw_luma = raw10_luma_preview(&frame.pixels,100);
    let mut result = serde_json::Map::new();
    for mode in ["color","blue","luma","flat-tire","contact"] {
        let source = if mode == "luma" { &raw_luma } else { &color };
        let mut pixels = vec![0; width * height];
        for y in 0..height { for x in 0..width {
            let value = source[y/SCALE*frame.width+x/SCALE];
            pixels[y*width+x] = if mode == "blue" { ViewMode::BlueFilter.filter_pixel(value) } else { value };
        } }
        match mode {
            "flat-tire" => { draw_sam31_outer_iris_fit(&mut pixels,width,height,0,0,SCALE,frame.sequence,proposal); }
            "contact" => { draw_sam31_virtual_contact_source(&mut pixels,width,height,0,0,SCALE,
                frame.sequence,proposal,contact,None); }
            _ => {}
        }
        let name = format!("{mode}/{index:06}.ppm");
        let path = directory.join(&name);
        fs::create_dir_all(path.parent().unwrap()).map_err(|e|e.to_string())?;
        export_eye_ppm(&path,&pixels,width,height)?;
        result.insert(mode.into(), json!(name));
    }
    Ok(Value::Object(result))
}
