//! RAW-native Student v1: sixteen sensor-anchored, individually resampled CFA
//! phase fields. No demosaic, white balance, tone curve or eight-bit image.
use serde_json::{json, Value};

pub const CONTRACT: &str = "quad-rggb16-linear-code-f32-v1";
pub const WIDTH: usize = 192;
pub const HEIGHT: usize = 128;
pub const CHANNELS: usize = 16;
pub const VALUES: usize = CHANNELS * WIDTH * HEIGHT;

pub fn contract() -> Value {
    json!({"version":CONTRACT,"shape":[CHANNELS,HEIGHT,WIDTH],"dtype":"float32",
        "readout":"RAW10_LE40_1X1","cfa":"RRGG/RRGG/GGBB/GGBB",
        "channels":"4*(sensor_y mod 4)+(sensor_x mod 4)",
        "normalization":"sample/1023; nominal digital code range, NOT measured black/white calibration",
        "black_level":null,"white_level":null,"exposure_us":null,"analog_gain":null,
        "missing_metadata":"no invented radiometric calibration or exposure/gain correction",
        "sampling":"bilinear within each same-photosite-phase lattice only; edge replication within that phase",
        "registration":"source x=(x+0.5)*source_width/192-0.5; y=(y+0.5)*source_height/128-0.5",
        "resolution":"16 fields at half output width/height; 4-native-pixel sampling per phase; interpolation is not new sensor detail",
        "output_mapping":"unchanged six 384x256 masks in the full source crop"})
}

/// Explicit metadata prevents ordinary Bayer or binned RAW from being silently
/// interpreted as the camera's native 4x4 Quad Bayer lattice.
#[derive(Clone, Copy)]
pub struct Source<'a> {
    pub samples: &'a [u16],
    pub width: usize,
    pub height: usize,
    pub sensor_x: u32,
    pub sensor_y: u32,
    pub pixel_format: &'a str,
}

pub fn from_packed(
    payload: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    sensor_x: u32,
    sensor_y: u32,
    pixel_format: &str,
) -> Result<Vec<f32>, String> {
    let samples = crate::raw10::try_unpack_raw10(payload, width, height, stride)?;
    prepare(Source {
        samples: &samples,
        width,
        height,
        sensor_x,
        sensor_y,
        pixel_format,
    })
}

#[derive(Clone, Copy)]
struct Tap {
    a: usize,
    b: usize,
    fraction: f32,
}

fn taps(size: usize, origin: u32, phase: usize, out: usize) -> Vec<Tap> {
    let first = (phase + 4 - origin as usize % 4) % 4;
    let count = (size - 1 - first) / 4 + 1;
    (0..out)
        .map(|i| {
            // Common continuous full-crop coordinates, matching align_corners=false.
            let source = (i as f64 + 0.5) * size as f64 / out as f64 - 0.5;
            let p = ((source - first as f64) / 4.).clamp(0., (count - 1) as f64);
            let lo = p.floor() as usize;
            Tap {
                a: first + 4 * lo,
                b: first + 4 * (lo + 1).min(count - 1),
                fraction: (p - lo as f64) as f32,
            }
        })
        .collect()
}

pub fn prepare(source: Source<'_>) -> Result<Vec<f32>, String> {
    let Source {
        samples,
        width,
        height,
        sensor_x,
        sensor_y,
        pixel_format,
    } = source;
    if pixel_format != "RAW10_LE40_1X1" {
        return Err("RAW student requires declared RAW10_LE40_1X1 Quad Bayer readout".into());
    }
    if width < 8
        || height < 8
        || width.checked_mul(height) != Some(samples.len())
        || samples.len() > 16_000_000
        || samples.iter().any(|v| *v > 1023)
    {
        return Err("invalid RAW student dimensions or 10-bit samples".into());
    }
    let xt: [Vec<Tap>; 4] = std::array::from_fn(|p| taps(width, sensor_x, p, WIDTH));
    let yt: [Vec<Tap>; 4] = std::array::from_fn(|p| taps(height, sensor_y, p, HEIGHT));
    let mut output = vec![0.; VALUES];
    for py in 0..4 {
        for px in 0..4 {
            let plane =
                &mut output[(py * 4 + px) * WIDTH * HEIGHT..(py * 4 + px + 1) * WIDTH * HEIGHT];
            for (y, ty) in yt[py].iter().enumerate() {
                for (x, tx) in xt[px].iter().enumerate() {
                    let a = samples[ty.a * width + tx.a] as f32;
                    let b = samples[ty.a * width + tx.b] as f32;
                    let c = samples[ty.b * width + tx.a] as f32;
                    let d = samples[ty.b * width + tx.b] as f32;
                    let top = a + (b - a) * tx.fraction;
                    let bottom = c + (d - c) * tx.fraction;
                    plane[y * WIDTH + x] = (top + (bottom - top) * ty.fraction) / 1023.;
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source(samples: &[u16], width: usize, height: usize, x: u32, y: u32) -> Source<'_> {
        Source {
            samples,
            width,
            height,
            sensor_x: x,
            sensor_y: y,
            pixel_format: "RAW10_LE40_1X1",
        }
    }
    #[test]
    fn all_sixteen_origin_phases_preserve_photosite_identity_and_ten_bit_steps() {
        for oy in 0..4 {
            for ox in 0..4 {
                let pixels: Vec<_> = (0..16 * 12)
                    .map(|i| (4 * ((i / 16 + oy) % 4) + (i % 16 + ox) % 4) as u16 + 500)
                    .collect();
                let out = prepare(source(&pixels, 16, 12, ox as u32, oy as u32)).unwrap();
                for c in 0..16 {
                    for v in &out[c * WIDTH * HEIGHT..(c + 1) * WIDTH * HEIGHT] {
                        assert!((*v - (500 + c) as f32 / 1023.).abs() < 1e-7);
                    }
                }
                assert_ne!(out[0], out[WIDTH * HEIGHT]);
            }
        }
    }
    #[test]
    fn odd_crop_reframe_preserves_sensor_registered_ramp_not_trimmed_resize() {
        for (w, h) in [(33, 27), (36, 29)] {
            for (ox, oy) in [(0, 0), (1, 3), (2, 1), (3, 2)] {
                let pixels: Vec<_> = (0..w * h)
                    .map(|i| (100 + i % w + ox + 2 * (i / w + oy)) as u16)
                    .collect();
                let out = prepare(source(&pixels, w, h, ox as u32, oy as u32)).unwrap();
                let (x, y) = (80, 60);
                let sx = (x as f64 + 0.5) * w as f64 / WIDTH as f64 - 0.5 + ox as f64;
                let sy = (y as f64 + 0.5) * h as f64 / HEIGHT as f64 - 0.5 + oy as f64;
                for c in 0..16 {
                    assert!(
                        (out[c * WIDTH * HEIGHT + y * WIDTH + x] as f64 * 1023.
                            - (100. + sx + 2. * sy))
                            .abs()
                            < 0.0001
                    );
                }
            }
        }
    }
    #[test]
    fn packed_and_reused_unpacked_sources_match_exactly_and_reject_bad_metadata() {
        let pixels: Vec<_> = (0..16 * 12).map(|i| (i * 13 % 1024) as u16).collect();
        let bytes: Vec<_> = pixels
            .chunks_exact(4)
            .flat_map(|p| {
                let v = p[0] as u64
                    | ((p[1] as u64) << 10)
                    | ((p[2] as u64) << 20)
                    | ((p[3] as u64) << 30);
                v.to_le_bytes()[..5].to_vec()
            })
            .collect();
        assert_eq!(
            from_packed(&bytes, 16, 12, 20, 3, 1, "RAW10_LE40_1X1").unwrap(),
            prepare(source(&pixels, 16, 12, 3, 1)).unwrap()
        );
        assert!(from_packed(&bytes, 16, 12, 21, 0, 0, "RAW10_LE40_1X1").is_err());
        assert!(from_packed(&bytes, 16, 12, 20, 0, 0, "BAYER_RGGB").is_err());
        assert!(prepare(source(&[1024; 192], 16, 12, 0, 0)).is_err());
        assert!(prepare(source(&pixels, 4, 48, 0, 0)).is_err());
    }
}
