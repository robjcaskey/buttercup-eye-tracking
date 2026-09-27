#[allow(dead_code)]
mod existing {
    include!("buttercup_raw10_preview.rs");
    pub fn prepare(bytes: &[u8], w: usize, h: usize, stride: usize, sx: u32, sy: u32) -> Vec<u8> {
        let raw = raw10::try_unpack_raw10(bytes, w, h, stride).unwrap();
        decode_quad_bayer(&raw, w, h, sx, sy, PreviewMode::Color, 1.0)
    }
    pub fn save(path: &std::path::Path, w: usize, h: usize, rgb: &[u8]) {
        write_png(path, w, h, rgb).unwrap();
    }
}
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn prepare(meridian: bool) {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(),4,"usage: buttercup-pink-waterfall-sheet SELECTION.json RAW_FILENAMES.txt NEW_OUTPUT_DIR (12, 16, 24, 48 or 96 exposures)");
    let out = Path::new(&args[3]);
    assert!(!out.exists(), "output must be new");
    let parent = fs::canonicalize(out.parent().unwrap()).unwrap();
    assert!(
        parent.starts_with(fs::canonicalize("outputs").unwrap()),
        "output must be beneath checked outputs link"
    );
    let selection_bytes = fs::read(&args[1]).unwrap();
    let selection: Value = serde_json::from_slice(&selection_bytes).unwrap();
    let filenames = fs::read_to_string(&args[2]).unwrap();
    let filenames: Vec<_> = filenames.lines().collect();
    let frames = selection["frames"].as_array().unwrap();
    assert!(
        matches!(frames.len(), 12 | 16 | 24 | 48 | 96),
        "12, 16, 24, 48 or 96 exposures required"
    );
    assert_eq!(filenames.len(), frames.len());
    let cols = if frames.len() <= 24 { 4 } else { 8 };
    let rows = frames.len() / cols;
    // Twelve-frame diagnostic sheets preserve the actual native ROI size.
    // Historical larger generation batches keep their verified 384x256 layout.
    let (native_w, native_h) = if frames.len() == 12 {
        (frames[0]["frame"]["width"].as_u64().unwrap() as usize,
         frames[0]["frame"]["height"].as_u64().unwrap() as usize)
    } else { (384, 256) };
    assert!(native_w >= 4 && native_h >= 4 && native_w <= 1920 && native_h <= 1080);
    let (cw, ch, ox, oy) = if frames.len() <= 24 {
        (native_w, native_h, 0, 0)
    } else {
        (390, 264, 3, 4)
    };
    fs::create_dir(out).unwrap();
    fs::write(out.join("selection.json"), &selection_bytes).unwrap();
    fs::copy(&args[2], out.join("raw-filenames.txt")).unwrap();
    let (sw, sh) = (cols * cw, rows * ch);
    let mut sheet = vec![18u8; sw * sh * 3];
    let mut receipts = vec![];
    for (i, row) in frames.iter().enumerate() {
        assert_eq!(row["tile"].as_u64().unwrap() as usize, i + 1);
        let f = &row["frame"];
        let n = |k: &str| f[k].as_u64().unwrap();
        let capture = Path::new(row["capture"].as_str().unwrap());
        let stream = capture.join(f["stream"].as_str().unwrap());
        assert_eq!(
            fs::canonicalize(&stream).unwrap(),
            fs::canonicalize(filenames[i]).unwrap()
        );
        let index = fs::read_to_string(capture.join("frames.jsonl")).unwrap();
        let matches: Vec<Value> = index
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .filter(|v| v == f)
            .collect();
        assert_eq!(matches.len(), 1);
        let manifest_bytes = fs::read(capture.join("manifest.json")).unwrap();
        let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
        assert_eq!(manifest["pixel_format"], "RAW10_LE40_1X1");
        assert_eq!(f["pixel_format"], "RAW10_LE40_1X1");
        let (w, h) = (n("width") as usize, n("height") as usize);
        assert_eq!((w, h), (native_w, native_h), "mixed native ROI sizes");
        let mut file = fs::File::open(&stream).unwrap();
        assert!(n("offset") + n("length") <= file.metadata().unwrap().len());
        file.seek(SeekFrom::Start(n("offset"))).unwrap();
        let mut bytes = vec![0; n("length") as usize];
        file.read_exact(&mut bytes).unwrap();
        let rgb = existing::prepare(
            &bytes,
            w,
            h,
            n("stride") as usize,
            n("sensor_x") as u32,
            n("sensor_y") as u32,
        );
        let preview = format!("frame-{:02}.png", i + 1);
        existing::save(&out.join(&preview), w, h, &rgb);
        let (x, y) = (ox + (i % cols) * cw, oy + (i / cols) * ch);
        for yy in 0..h {
            let target = ((y + yy) * sw + x) * 3;
            sheet[target..target + w * 3].copy_from_slice(&rgb[yy * w * 3..(yy + 1) * w * 3]);
        }
        receipts.push(json!({"tile":i+1,"source":row,"raw_file":stream,"raw_sha256":hash(&bytes),"manifest_sha256":hash(&manifest_bytes),"preview":preview,"preview_sha256":hash(&fs::read(out.join(&preview)).unwrap()),"tile_xywh":[x,y,w,h]}));
    }
    existing::save(&out.join("contact-sheet-original.png"), sw, sh, &sheet);
    let provenance = json!({"selection_sha256":hash(&selection_bytes),"processing":{"implementation":"src/bin/buttercup_raw10_preview.rs decode_quad_bayer / write_png","source_sha256":hash(&fs::read("src/bin/buttercup_raw10_preview.rs").unwrap()),"raw10_decoder_sha256":hash(&fs::read("src/raw10.rs").unwrap()),"mode":"color","contrast":1.0,"cfa":"IMX582 Quad Bayer RGGB physical 2x2 groups, anchored to absolute sensor origin per existing decoder; CFA convention supplied by current source, capture manifests do not explicitly encode phase","cell":"4x4 physical cell channel means; bilinear reconstruction at native image dimensions","display_range":"per-channel 2nd and 99.5th percentile","gamma":2.2,"no_training":true},"sheet":{"width":sw,"height":sh,"tile_order":"row major manifest order","columns":cols,"rows":rows,"resized":false,"gutter_rgb":[18,18,18],"sha256":hash(&fs::read(out.join("contact-sheet-original.png")).unwrap())},"frames":receipts});
    fs::write(
        out.join("preparation-provenance.json"),
        serde_json::to_vec_pretty(&provenance).unwrap(),
    )
    .unwrap();
    let guide = if meridian {
        include_str!("../../docs/globe-meridian-contact-sheets.md")
    } else {
        include_str!("../../docs/pink-waterfall-contact-sheets.md")
    };
    let prompt=guide.lines().filter_map(|line|line.strip_prefix("> ")).collect::<Vec<_>>().join("\n")
 .replace("4-column 3-row contact sheet of 12", &format!("{cols}-column {rows}-row contact sheet of {}", frames.len()))
 .replace("exact 4x3 layout", &format!("exact {cols}x{rows} layout"))
 .replace("approximately 1.97:1 landscape aspect ratio", &format!("{sw}×{sh} pixel dimensions; each source eye occupies exactly {native_w}×{native_h} pixels, with no resizing, stretching or cropping"));
    fs::write(
        out.join(if meridian {
            "meridian-prompt.txt"
        } else {
            "pink-drop-prompt.txt"
        }),
        prompt,
    )
    .unwrap();
    println!("Validated and prepared {} exact exposures", frames.len());
}
fn main() {
    prepare(false);
}
