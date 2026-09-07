#[path = "../raw10.rs"]
mod raw10;

use serde_json::Value;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum PreviewMode {
    Color,
    Blue,
}

fn usage() -> String {
    "usage: buttercup-raw10-preview CAPTURE_DIR SEQUENCE EYE_LABEL OUTPUT.ppm [color|blue] [contrast]\n"
        .to_string()
}

fn number(record: &Value, name: &str) -> Result<u64, String> {
    record
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("frame record lacks {name}"))
}

fn percentile(mut values: Vec<f32>, fraction: f32) -> f32 {
    let index =
        ((values.len().saturating_sub(1)) as f32 * fraction.clamp(0.0, 1.0)).round() as usize;
    let (_, value, _) = values.select_nth_unstable_by(index, f32::total_cmp);
    *value
}

fn quad_cell(raw: &[u16], width: usize, x: usize, y: usize) -> [f32; 3] {
    let average = |offset_x: usize, offset_y: usize| {
        let i = (y + offset_y) * width + x + offset_x;
        (f32::from(raw[i])
            + f32::from(raw[i + 1])
            + f32::from(raw[i + width])
            + f32::from(raw[i + width + 1]))
            * 0.25
    };
    [
        average(0, 0),
        (average(2, 0) + average(0, 2)) * 0.5,
        average(2, 2),
    ]
}

fn decode_quad_bayer(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    mode: PreviewMode,
    contrast: f32,
) -> Vec<u8> {
    // Align the physical 4x4 RG/GB cell to absolute sensor coordinates.
    let start_x = (4 - sensor_x as usize % 4) % 4;
    let start_y = (4 - sensor_y as usize % 4) % 4;
    let grid_width = width.saturating_sub(start_x) / 4;
    let grid_height = height.saturating_sub(start_y) / 4;
    let mut cells = Vec::with_capacity(grid_width * grid_height);
    for cell_y in 0..grid_height {
        for cell_x in 0..grid_width {
            cells.push(quad_cell(
                raw,
                width,
                start_x + cell_x * 4,
                start_y + cell_y * 4,
            ));
        }
    }
    let channel = |index: usize| cells.iter().map(|cell| cell[index]).collect::<Vec<_>>();
    let ranges = std::array::from_fn::<_, 3, _>(|index| {
        let values = channel(index);
        let low = percentile(values.clone(), 0.02);
        let high = percentile(values, 0.995).max(low + 1.0);
        (low, high)
    });
    let map = |value: f32, range: (f32, f32)| {
        let normalized = ((value - range.0) / (range.1 - range.0)).clamp(0.0, 1.0);
        let expanded = ((normalized - 0.5) * contrast + 0.5).clamp(0.0, 1.0);
        // Display gamma is presentation-only; all detection stays in linear
        // RAW values. Gamma makes faint vessel contrast visible on screen.
        (expanded.powf(1.0 / 2.2) * 255.0).round() as u8
    };
    let mut output = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            // Bilinear reconstruction between physical Quad-Bayer cell
            // centers. This does not invent extra sensor samples; it avoids
            // presenting each 2x2 blue block as a misleading 4x4 staircase.
            let grid_x = ((x as f32 - start_x as f32 - 1.5) / 4.0)
                .clamp(0.0, grid_width.saturating_sub(1) as f32);
            let grid_y = ((y as f32 - start_y as f32 - 1.5) / 4.0)
                .clamp(0.0, grid_height.saturating_sub(1) as f32);
            let x0 = grid_x.floor() as usize;
            let y0 = grid_y.floor() as usize;
            let x1 = (x0 + 1).min(grid_width - 1);
            let y1 = (y0 + 1).min(grid_height - 1);
            let tx = grid_x - x0 as f32;
            let ty = grid_y - y0 as f32;
            let rgb = std::array::from_fn::<_, 3, _>(|channel| {
                let top = cells[y0 * grid_width + x0][channel] * (1.0 - tx)
                    + cells[y0 * grid_width + x1][channel] * tx;
                let bottom = cells[y1 * grid_width + x0][channel] * (1.0 - tx)
                    + cells[y1 * grid_width + x1][channel] * tx;
                top * (1.0 - ty) + bottom * ty
            });
            let pixel = match mode {
                PreviewMode::Color => [
                    map(rgb[0], ranges[0]),
                    map(rgb[1], ranges[1]),
                    map(rgb[2], ranges[2]),
                ],
                PreviewMode::Blue => {
                    let blue = map(rgb[2], ranges[2]);
                    [blue, blue, blue]
                }
            };
            output[(y * width + x) * 3..(y * width + x + 1) * 3].copy_from_slice(&pixel);
        }
    }
    output
}

fn write_bmp(path: &Path, width: usize, height: usize, rgb: &[u8]) -> Result<(), String> {
    let row_bytes = width * 3;
    let padded_row_bytes = (row_bytes + 3) & !3;
    let pixel_bytes = padded_row_bytes * height;
    let file_bytes = 54usize + pixel_bytes;
    let mut destination =
        File::create(path).map_err(|error| format!("create {}: {error}", path.display()))?;
    let mut header = [0u8; 54];
    header[0..2].copy_from_slice(b"BM");
    header[2..6].copy_from_slice(&(file_bytes as u32).to_le_bytes());
    header[10..14].copy_from_slice(&54u32.to_le_bytes());
    header[14..18].copy_from_slice(&40u32.to_le_bytes());
    header[18..22].copy_from_slice(&(width as i32).to_le_bytes());
    header[22..26].copy_from_slice(&(height as i32).to_le_bytes());
    header[26..28].copy_from_slice(&1u16.to_le_bytes());
    header[28..30].copy_from_slice(&24u16.to_le_bytes());
    header[34..38].copy_from_slice(&(pixel_bytes as u32).to_le_bytes());
    destination
        .write_all(&header)
        .map_err(|error| format!("write BMP header: {error}"))?;
    let padding = vec![0u8; padded_row_bytes - row_bytes];
    for y in (0..height).rev() {
        for x in 0..width {
            let i = (y * width + x) * 3;
            destination
                .write_all(&[rgb[i + 2], rgb[i + 1], rgb[i]])
                .map_err(|error| format!("write BMP pixels: {error}"))?;
        }
        destination
            .write_all(&padding)
            .map_err(|error| format!("write BMP padding: {error}"))?;
    }
    Ok(())
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320u32 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

fn append_png_chunk(output: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
    output.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    let mut checked = Vec::with_capacity(4 + payload.len());
    checked.extend_from_slice(kind);
    checked.extend_from_slice(payload);
    output.extend_from_slice(&crc32(&checked).to_be_bytes());
}

fn write_png(path: &Path, width: usize, height: usize, rgb: &[u8]) -> Result<(), String> {
    // PNG scanlines use filter 0. The zlib payload contains standards-compliant
    // uncompressed DEFLATE blocks, implemented here so corpus color decoding
    // and file generation have no external image-tool dependency.
    let mut scanlines = Vec::with_capacity((width * 3 + 1) * height);
    for row in rgb.chunks_exact(width * 3) {
        scanlines.push(0);
        scanlines.extend_from_slice(row);
    }
    let mut zlib = vec![0x78, 0x01];
    let mut cursor = 0usize;
    while cursor < scanlines.len() {
        let length = (scanlines.len() - cursor).min(u16::MAX as usize);
        let final_block = cursor + length == scanlines.len();
        zlib.push(u8::from(final_block));
        zlib.extend_from_slice(&(length as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(length as u16)).to_le_bytes());
        zlib.extend_from_slice(&scanlines[cursor..cursor + length]);
        cursor += length;
    }
    let mut first = 1u32;
    let mut second = 0u32;
    for byte in &scanlines {
        first = (first + u32::from(*byte)) % 65_521;
        second = (second + first) % 65_521;
    }
    zlib.extend_from_slice(&((second << 16) | first).to_be_bytes());
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    append_png_chunk(&mut png, b"IHDR", &ihdr);
    append_png_chunk(&mut png, b"IDAT", &zlib);
    append_png_chunk(&mut png, b"IEND", &[]);
    fs::write(path, png).map_err(|error| format!("write {}: {error}", path.display()))
}

/// Read an inventory's exact native source receipt, including tar member
/// offsets, without extracting/copying a recording or using an image utility.
fn indexed_preview(arguments:&[String])->Result<(),String> {
    if arguments.len()<4 || arguments.len()>5 {return Err("usage: --source-index INPUT.jsonl INDEX OUTPUT.png [EVALUATION.jsonl]".into());}
    let index=arguments[2].parse::<u64>().map_err(|e|e.to_string())?;
    let source=File::open(&arguments[1]).map_err(|e|e.to_string())?;
    let mut record=None;
    for line in BufReader::new(source).lines() {
        let row:Value=serde_json::from_str(&line.map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        if row["index"].as_u64()==Some(index) {record=Some(row);break;}
    }
    let record=record.ok_or("source index absent")?;
    let meta=&record["frame"];
    let width=number(meta,"width")? as usize;let height=number(meta,"height")? as usize;
    let mut file=File::open(record["raw_file"].as_str().ok_or("missing raw_file")?).map_err(|e|e.to_string())?;
    file.seek(SeekFrom::Start(number(&record,"raw_offset")?)).map_err(|e|e.to_string())?;
    let mut packed=vec![0;number(&record,"raw_length")? as usize];file.read_exact(&mut packed).map_err(|e|e.to_string())?;
    let raw=raw10::try_unpack_raw10(&packed,width,height,number(meta,"stride")? as usize)?;
    let rgb=decode_quad_bayer(&raw,width,height,number(meta,"sensor_x")? as u32,number(meta,"sensor_y")? as u32,PreviewMode::Color,1.0);
    let mut evaluated=None;
    if let Some(path)=arguments.get(4) {
        for line in BufReader::new(File::open(path).map_err(|e|e.to_string())?).lines() {
            let row:Value=serde_json::from_str(&line.map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            if row["inputs"].as_array().is_some_and(|inputs|inputs.iter().any(|v|v["index"].as_u64()==Some(index))) {evaluated=Some(row);break;}
        }
    }
    let panels=if evaluated.is_some() {4} else {1};
    let mut comparison=vec![0;width*panels*height*3];
    for y in 0..height {for panel in 0..panels {
        let begin=(y*width*panels+panel*width)*3;
        comparison[begin..begin+width*3].copy_from_slice(&rgb[y*width*3..(y+1)*width*3]);
    }}
    if let Some(row)=evaluated {
        let eye=number(meta,"eye_id")? as usize-1;
        let mono=if eye==0 {"monocular_right"} else {"monocular_left"};
        for (panel,ellipse,color) in [(1,&row["baseline_sam_outer"][eye],[255,80,220]),
            (2,&row["joint"]["outer_ellipses"][eye],[40,255,100]),(3,&row[mono]["outer_ellipses"][eye],[255,210,30])] {
            let Some(x)=ellipse["center"][0].as_f64() else {continue;};
            let y=ellipse["center"][1].as_f64().ok_or("missing center y")?;
            let a=ellipse["major_radius"].as_f64().ok_or("missing major")?;
            let b=ellipse["minor_radius"].as_f64().ok_or("missing minor")?;
            let (s,c)=ellipse["angle"].as_f64().ok_or("missing angle")?.sin_cos();
            for i in 0..1024 {
                let phase=std::f64::consts::TAU*i as f64/1024.0;
                let px=(x+a*phase.cos()*c-b*phase.sin()*s).round() as isize;
                let py=(y+a*phase.cos()*s+b*phase.sin()*c).round() as isize;
                if px>=0 && py>=0 && px<width as isize && py<height as isize {
                    let at=(py as usize*width*panels+panel*width+px as usize)*3;
                    comparison[at..at+3].copy_from_slice(&color);
                }
            }
        }
        // Optional evaluator diagnostics: actual training points, never a
        // synthetic completed perimeter. Panel 1 shows all alternatives;
        // panel 2 shows selected/used points in cyan, rejected ones in gray.
        if let Some(arcs)=row["sparse_evidence"][eye]["arcs"].as_array() {
            for (index,arc) in arcs.iter().enumerate() {
                let used=row["joint"]["support"].as_array().is_some_and(|support|support.iter().any(|s|
                    s["roi"].as_u64()==Some(eye as u64+1)&&s["arc"].as_u64()==Some(index as u64)&&s["used"]==true));
                let Some(points)=arc["points"].as_array() else {continue;};
                for (sample_index,point) in points.iter().enumerate() {
                    let (Some(x),Some(y))=(point[0].as_f64(),point[1].as_f64()) else {continue;};
                    for (panel,color) in [(1,[255,80,220]),(2,if used {[50,230,255]} else {[130,130,130]})] {
                        for dy in -1..=1 {for dx in -1..=1 {
                            let px=x.round() as isize+dx;let py=y.round() as isize+dy;
                            if px>=0&&py>=0&&px<width as isize&&py<height as isize {
                                let at=(py as usize*width*panels+panel*width+px as usize)*3;
                                comparison[at..at+3].copy_from_slice(&color);
                            }
                        }}
                    }
                    // Sparse outward IMAGE-boundary direction cues, only
                    // when actually exported for this measured point. These
                    // are not reconstructed 3D gaze/surface-normal vectors.
                    if sample_index%4==0 {
                        let normal=&arc["outward_normals"][sample_index]["unit_outward_roi"];
                        if let (Some(nx),Some(ny))=(normal[0].as_f64(),normal[1].as_f64()) {
                            if nx.is_finite()&&ny.is_finite()&&(nx.hypot(ny)-1.0).abs()<1.0e-6 {
                                for step in 0..=10 {
                                    let px=(x+nx*step as f64).round() as isize;
                                    let py=(y+ny*step as f64).round() as isize;
                                    if px>=0&&py>=0&&px<width as isize&&py<height as isize {
                                        let at=(py as usize*width*panels+2*width+px as usize)*3;
                                        comparison[at..at+3].copy_from_slice(&if used {[50,230,255]} else {[130,130,130]});
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let output=Path::new(&arguments[3]);
    if let Some(parent)=output.parent() {fs::create_dir_all(parent).map_err(|e|e.to_string())?;}
    write_png(output,width*panels,height,&comparison)?;
    eprintln!("native RAW source {index}: left-to-right raw / magenta SAM baseline / green joint / yellow monocular -> {}",output.display());
    Ok(())
}

fn main() -> Result<(), String> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments.first().map(String::as_str)==Some("--source-index") {return indexed_preview(&arguments);}
    if arguments.len() < 4 || arguments.len() > 6 {
        return Err(usage());
    }
    let capture = PathBuf::from(&arguments[0]);
    let sequence = arguments[1]
        .parse::<u64>()
        .map_err(|error| format!("sequence: {error}"))?;
    let eye = &arguments[2];
    let output = PathBuf::from(&arguments[3]);
    let mode = match arguments.get(4).map(String::as_str).unwrap_or("blue") {
        "color" => PreviewMode::Color,
        "blue" => PreviewMode::Blue,
        other => return Err(format!("unknown preview mode {other:?}\n{}", usage())),
    };
    let contrast = arguments
        .get(5)
        .map(|value| value.parse::<f32>())
        .transpose()
        .map_err(|error| format!("contrast: {error}"))?
        .unwrap_or(2.8)
        .clamp(0.1, 12.0);
    let index = fs::read_to_string(capture.join("frames.jsonl"))
        .map_err(|error| format!("read frames.jsonl: {error}"))?;
    let record = index
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|record| {
            record.get("sequence").and_then(Value::as_u64) == Some(sequence)
                && record.get("label").and_then(Value::as_str) == Some(eye)
        })
        .ok_or_else(|| format!("no {eye} sequence {sequence} in {}", capture.display()))?;
    if record.get("pixel_format").and_then(Value::as_str) != Some("RAW10_LE40_1X1") {
        return Err("frame is not RAW10_LE40_1X1".to_string());
    }
    let width = number(&record, "width")? as usize;
    let height = number(&record, "height")? as usize;
    let stride = number(&record, "stride")? as usize;
    let offset = number(&record, "offset")?;
    let length = number(&record, "length")? as usize;
    let sensor_x = number(&record, "sensor_x")? as u32;
    let sensor_y = number(&record, "sensor_y")? as u32;
    let stream = record
        .get("stream")
        .and_then(Value::as_str)
        .ok_or_else(|| "frame record lacks stream".to_string())?;
    let mut source =
        File::open(capture.join(stream)).map_err(|error| format!("open {stream}: {error}"))?;
    source
        .seek(SeekFrom::Start(offset))
        .map_err(|error| format!("seek {stream}: {error}"))?;
    let mut packed = vec![0u8; length];
    source
        .read_exact(&mut packed)
        .map_err(|error| format!("read {stream}: {error}"))?;
    let raw = raw10::try_unpack_raw10(&packed, width, height, stride)?;
    let rgb = decode_quad_bayer(&raw, width, height, sensor_x, sensor_y, mode, contrast);
    if let Some(parent) = Path::new(&output).parent() {
        fs::create_dir_all(parent).map_err(|error| format!("create output directory: {error}"))?;
    }
    match output.extension().and_then(|extension| extension.to_str()) {
        Some("bmp") => write_bmp(&output, width, height, &rgb)?,
        Some("png") => write_png(&output, width, height, &rgb)?,
        _ => {
            let mut destination = File::create(&output)
                .map_err(|error| format!("create {}: {error}", output.display()))?;
            write!(destination, "P6\n{width} {height}\n255\n")
                .map_err(|error| format!("write PPM header: {error}"))?;
            destination
                .write_all(&rgb)
                .map_err(|error| format!("write PPM pixels: {error}"))?;
        }
    }
    eprintln!(
        "decoded {} sequence={} eye={} sensor={},{} {}x{} mode={} contrast={:.2} -> {}",
        capture.display(),
        sequence,
        eye,
        sensor_x,
        sensor_y,
        width,
        height,
        match mode {
            PreviewMode::Color => "color",
            PreviewMode::Blue => "blue",
        },
        contrast,
        output.display(),
    );
    Ok(())
}
