//! Rust-owned MediaPipe Tasks face-landmarker adapter.
//!
//! This module uses the stable native C ABI exported by `libmediapipe.so`.
//! It neither starts nor embeds a language runtime. Sensor samples remain in
//! Rust-owned memory until the C API copies them into an `MpImage`.

use libloading::Library;
use std::env;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::OnceLock;

const IMAGE_FORMAT_SRGB: c_int = 1;
const RUNNING_MODE_IMAGE: c_int = 1;
const REQUIRED_FACE_LANDMARKS: usize = 478;

const RIGHT_IRIS_CENTER: usize = 468;
const RIGHT_IRIS_RING: [usize; 4] = [469, 470, 471, 472];
const LEFT_IRIS_CENTER: usize = 473;
const LEFT_IRIS_RING: [usize; 4] = [474, 475, 476, 477];
const RIGHT_EYE_CORNERS: [usize; 2] = [33, 133];
const LEFT_EYE_CORNERS: [usize; 2] = [362, 263];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimePaths {
    pub library: PathBuf,
    pub model: PathBuf,
}

impl RuntimePaths {
    pub fn discover() -> Result<Self, String> {
        let library = discover_asset(
            "BUTTERCUP_MEDIAPIPE_LIBRARY",
            &[
                "data/runtime/mediapipe/libmediapipe.so",
                "/usr/local/lib/libmediapipe.so",
                "/usr/lib/libmediapipe.so",
                "/usr/lib/x86_64-linux-gnu/libmediapipe.so",
            ],
            "native MediaPipe Tasks shared library",
        )?;
        let model = discover_asset(
            "BUTTERCUP_MEDIAPIPE_MODEL",
            &[
                "data/models/mediapipe/face_landmarker.task",
                "/usr/local/share/mediapipe/face_landmarker.task",
                "/usr/share/mediapipe/face_landmarker.task",
            ],
            "MediaPipe face-landmarker task model",
        )?;
        Ok(Self { library, model })
    }
}

fn discover_asset(variable: &str, defaults: &[&str], label: &str) -> Result<PathBuf, String> {
    if let Some(value) = env::var_os(variable).filter(|value| !value.is_empty()) {
        let path = PathBuf::from(value);
        return path.is_file().then_some(path.clone()).ok_or_else(|| {
            format!(
                "{label} named by {variable} is not a file: {}",
                path.display()
            )
        });
    }
    defaults
        .iter()
        .map(PathBuf::from)
        .find(|path| path.is_file())
        .ok_or_else(|| {
            format!(
                "{label} is unavailable; install it under {} or set {variable}",
                defaults[0]
            )
        })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeCenter {
    /// Horizontal coordinate in the full inference image, normalized to 0..1.
    pub x: f32,
    /// Vertical coordinate in the full inference image, normalized to 0..1.
    pub y: f32,
    /// Approximate projected iris radius in normalized image-width units.
    pub iris_radius: f32,
    /// Projected canthus-to-canthus span in normalized image-width units.
    pub eye_width: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// Image-left followed by image-right, independent of anatomical naming.
    pub eyes: [EyeCenter; 2],
    pub landmark_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BaseOptionsC {
    model_asset_buffer: *const c_char,
    model_asset_buffer_count: u32,
    model_asset_path: *const c_char,
    delegate: c_int,
}

type ResultCallback =
    Option<unsafe extern "C" fn(c_int, *const FaceLandmarkerResultC, *mut c_void, i64)>;

#[repr(C)]
#[derive(Clone, Copy)]
struct FaceLandmarkerOptionsC {
    base_options: BaseOptionsC,
    running_mode: c_int,
    num_faces: c_int,
    min_face_detection_confidence: f32,
    min_face_presence_confidence: f32,
    min_tracking_confidence: f32,
    output_face_blendshapes: bool,
    output_facial_transformation_matrixes: bool,
    result_callback: ResultCallback,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormalizedLandmarkC {
    x: f32,
    y: f32,
    z: f32,
    has_visibility: bool,
    visibility: f32,
    has_presence: bool,
    presence: f32,
    name: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormalizedLandmarksC {
    landmarks: *const NormalizedLandmarkC,
    landmarks_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CategoriesC {
    categories: *const c_void,
    categories_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatrixC {
    rows: u32,
    cols: u32,
    data: *const f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FaceLandmarkerResultC {
    face_landmarks: *const NormalizedLandmarksC,
    face_landmarks_count: u32,
    face_blendshapes: *const CategoriesC,
    face_blendshapes_count: u32,
    facial_transformation_matrixes: *const MatrixC,
    facial_transformation_matrixes_count: u32,
}

impl Default for FaceLandmarkerResultC {
    fn default() -> Self {
        Self {
            face_landmarks: ptr::null(),
            face_landmarks_count: 0,
            face_blendshapes: ptr::null(),
            face_blendshapes_count: 0,
            facial_transformation_matrixes: ptr::null(),
            facial_transformation_matrixes_count: 0,
        }
    }
}

type CreateFn = unsafe extern "C" fn(
    *const FaceLandmarkerOptionsC,
    *mut *mut c_void,
    *mut *mut c_char,
) -> c_int;
type DetectImageFn = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const c_void,
    *mut FaceLandmarkerResultC,
    *mut *mut c_char,
) -> c_int;
type CloseResultFn = unsafe extern "C" fn(*mut FaceLandmarkerResultC);
type CloseFn = unsafe extern "C" fn(*mut c_void, *mut *mut c_char) -> c_int;
type ImageCreateU8Fn = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    *const u8,
    c_int,
    *mut *mut c_void,
    *mut *mut c_char,
) -> c_int;
type ImageFreeFn = unsafe extern "C" fn(*mut c_void);
type ErrorFreeFn = unsafe extern "C" fn(*mut c_void);

struct NativeApi {
    _library: Library,
    create: CreateFn,
    detect_image: DetectImageFn,
    close_result: CloseResultFn,
    close: CloseFn,
    image_create_u8: ImageCreateU8Fn,
    image_free: ImageFreeFn,
    error_free: ErrorFreeFn,
}

static NATIVE_API: OnceLock<Result<NativeApi, String>> = OnceLock::new();

fn native_api(path: &Path) -> Result<&'static NativeApi, String> {
    NATIVE_API
        .get_or_init(|| NativeApi::load(path))
        .as_ref()
        .map_err(Clone::clone)
}

impl NativeApi {
    fn load(path: &Path) -> Result<Self, String> {
        // SAFETY: The library remains owned by `NativeApi` for at least as
        // long as every copied function pointer below.
        let library = unsafe { Library::new(path) }.map_err(|error| {
            format!("load native MediaPipe runtime {}: {error}", path.display())
        })?;
        unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
            // SAFETY: Callers provide the exact signatures documented by the
            // MediaPipe Tasks C ABI. A missing or version-skewed symbol fails
            // here before any task handle is created.
            unsafe { library.get::<T>(name) }
                .map(|symbol| *symbol)
                .map_err(|error| {
                    format!(
                        "load native MediaPipe symbol {}: {error}",
                        String::from_utf8_lossy(&name[..name.len().saturating_sub(1)])
                    )
                })
        }
        // SAFETY: Each type alias exactly mirrors the installed C ABI.
        let create = unsafe { load_symbol(&library, b"MpFaceLandmarkerCreate\0")? };
        let detect_image = unsafe { load_symbol(&library, b"MpFaceLandmarkerDetectImage\0")? };
        let close_result = unsafe { load_symbol(&library, b"MpFaceLandmarkerCloseResult\0")? };
        let close = unsafe { load_symbol(&library, b"MpFaceLandmarkerClose\0")? };
        let image_create_u8 = unsafe { load_symbol(&library, b"MpImageCreateFromUint8Data\0")? };
        let image_free = unsafe { load_symbol(&library, b"MpImageFree\0")? };
        let error_free = unsafe { load_symbol(&library, b"MpErrorFree\0")? };
        Ok(Self {
            _library: library,
            create,
            detect_image,
            close_result,
            close,
            image_create_u8,
            image_free,
            error_free,
        })
    }

    fn status(&self, code: c_int, error: *mut c_char, operation: &str) -> Result<(), String> {
        let message = if error.is_null() {
            None
        } else {
            // SAFETY: MediaPipe returns a NUL-terminated error string whose
            // ownership is transferred to the caller and released below.
            let text = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: `MpErrorFree` is the matching allocator boundary.
            unsafe { (self.error_free)(error.cast()) };
            Some(text)
        };
        if code == 0 {
            Ok(())
        } else {
            Err(format!(
                "{operation} failed with MediaPipe status {code}{}",
                message.map_or_else(String::new, |message| format!(": {message}"))
            ))
        }
    }
}

struct FaceLandmarker {
    api: &'static NativeApi,
    handle: *mut c_void,
}

impl FaceLandmarker {
    fn create(paths: &RuntimePaths) -> Result<Self, String> {
        // The MediaPipe runtime starts internal XNNPACK workers. Keep the
        // shared object loaded for process lifetime so no worker can return
        // through unmapped code during a concurrent `dlclose` at task teardown.
        let api = native_api(&paths.library)?;
        let model_path = CString::new(paths.model.as_os_str().as_encoded_bytes())
            .map_err(|_| "native MediaPipe model path contains a NUL byte".to_string())?;
        let options = FaceLandmarkerOptionsC {
            base_options: BaseOptionsC {
                model_asset_buffer: ptr::null(),
                model_asset_buffer_count: 0,
                model_asset_path: model_path.as_ptr(),
                delegate: 0,
            },
            running_mode: RUNNING_MODE_IMAGE,
            num_faces: 1,
            min_face_detection_confidence: 0.35,
            min_face_presence_confidence: 0.35,
            min_tracking_confidence: 0.35,
            output_face_blendshapes: false,
            output_facial_transformation_matrixes: false,
            result_callback: None,
        };
        let mut handle = ptr::null_mut();
        let mut error = ptr::null_mut();
        // SAFETY: `options`, its model path, and both output pointers remain
        // valid for the complete synchronous create call.
        let code = unsafe { (api.create)(&options, &mut handle, &mut error) };
        api.status(code, error, "create native face landmarker")?;
        if handle.is_null() {
            return Err("native MediaPipe returned a null face-landmarker handle".to_string());
        }
        Ok(Self { api, handle })
    }

    fn detect_gray10(
        &mut self,
        samples: &[u16],
        width: usize,
        height: usize,
    ) -> Result<Detection, String> {
        let srgb = gray10_to_srgb(samples, width, height)?;
        let width_c = c_int::try_from(width).map_err(|_| "inference width exceeds c_int")?;
        let height_c = c_int::try_from(height).map_err(|_| "inference height exceeds c_int")?;
        let count_c = c_int::try_from(srgb.len()).map_err(|_| "inference image exceeds c_int")?;
        let mut image = ptr::null_mut();
        let mut error = ptr::null_mut();
        // SAFETY: The packed RGB allocation and output pointers remain valid
        // for this synchronous call. MediaPipe copies the pixel allocation.
        let code = unsafe {
            (self.api.image_create_u8)(
                IMAGE_FORMAT_SRGB,
                width_c,
                height_c,
                srgb.as_ptr(),
                count_c,
                &mut image,
                &mut error,
            )
        };
        self.api
            .status(code, error, "create native MediaPipe image")?;
        if image.is_null() {
            return Err("native MediaPipe returned a null image handle".to_string());
        }

        let mut result = FaceLandmarkerResultC::default();
        let mut detect_error = ptr::null_mut();
        // SAFETY: Handles were created by the same loaded ABI. `result` is a
        // writable C-layout output and processing options are intentionally
        // null because this is an uncropped, unrotated sensor overview.
        let detect_code = unsafe {
            (self.api.detect_image)(
                self.handle,
                image,
                ptr::null(),
                &mut result,
                &mut detect_error,
            )
        };
        let detected = self
            .api
            .status(detect_code, detect_error, "run native face landmarker")
            .and_then(|()| copy_detection(&result));
        // SAFETY: Both objects are released with the matching C API after all
        // result landmarks have been copied to Rust-owned values.
        unsafe {
            (self.api.close_result)(&mut result);
            (self.api.image_free)(image);
        }
        detected
    }
}

impl Drop for FaceLandmarker {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        let mut error = ptr::null_mut();
        // SAFETY: `handle` belongs to this object and is closed once here.
        let code = unsafe { (self.api.close)(self.handle, &mut error) };
        if let Err(message) = self.api.status(code, error, "close native face landmarker") {
            eprintln!("{message}");
        }
        self.handle = ptr::null_mut();
    }
}

pub fn detect_eye_centers_gray10(
    samples: &[u16],
    width: usize,
    height: usize,
) -> Result<(RuntimePaths, Detection), String> {
    let paths = RuntimePaths::discover()?;
    let mut landmarker = FaceLandmarker::create(&paths)?;
    let detection = landmarker.detect_gray10(samples, width, height)?;
    Ok((paths, detection))
}

fn copy_detection(result: &FaceLandmarkerResultC) -> Result<Detection, String> {
    if result.face_landmarks_count == 0 || result.face_landmarks.is_null() {
        return Err("native MediaPipe found no face in the sensor overview".to_string());
    }
    // `num_faces` is one, so the first list is the only admissible identity.
    // SAFETY: MediaPipe owns this array until `CloseResult`, and the caller
    // invokes us before closing it.
    let face = unsafe { &*result.face_landmarks };
    let count = face.landmarks_count as usize;
    if count < REQUIRED_FACE_LANDMARKS || face.landmarks.is_null() {
        return Err(format!(
            "native MediaPipe returned {count} face landmarks; refined iris landmarks require at least {REQUIRED_FACE_LANDMARKS}"
        ));
    }
    // SAFETY: `landmarks_count` describes the live result allocation.
    let landmarks = unsafe { std::slice::from_raw_parts(face.landmarks, count) };
    let points = landmarks
        .iter()
        .map(|point| (point.x, point.y, point.z))
        .collect::<Vec<_>>();
    Ok(Detection {
        eyes: extract_eye_centers(&points)?,
        landmark_count: count,
    })
}

fn extract_eye_centers(points: &[(f32, f32, f32)]) -> Result<[EyeCenter; 2], String> {
    if points.len() < REQUIRED_FACE_LANDMARKS {
        return Err(format!(
            "face has {} landmarks; {REQUIRED_FACE_LANDMARKS} are required",
            points.len()
        ));
    }
    let right = extract_eye_center(
        points,
        RIGHT_IRIS_CENTER,
        RIGHT_IRIS_RING,
        RIGHT_EYE_CORNERS,
    )?;
    let left = extract_eye_center(points, LEFT_IRIS_CENTER, LEFT_IRIS_RING, LEFT_EYE_CORNERS)?;
    let mut eyes = [right, left];
    eyes.sort_by(|a, b| a.x.total_cmp(&b.x));
    let separation = eyes[1].x - eyes[0].x;
    if !(0.04..=0.70).contains(&separation) || (eyes[1].y - eyes[0].y).abs() > 0.28 {
        return Err(format!(
            "native MediaPipe eye-pair geometry is implausible: separation={separation:.4} vertical_delta={:.4}",
            (eyes[1].y - eyes[0].y).abs()
        ));
    }
    Ok(eyes)
}

fn extract_eye_center(
    points: &[(f32, f32, f32)],
    center_index: usize,
    ring: [usize; 4],
    corners: [usize; 2],
) -> Result<EyeCenter, String> {
    let point = |index: usize| -> Result<(f32, f32), String> {
        let (x, y, _) = points[index];
        (x.is_finite() && y.is_finite())
            .then_some((x, y))
            .ok_or_else(|| format!("face landmark {index} is not finite"))
    };
    let direct = point(center_index)?;
    let ring_points = ring.map(point);
    let mut ring_xy = [(0.0f32, 0.0f32); 4];
    for (destination, source) in ring_xy.iter_mut().zip(ring_points) {
        *destination = source?;
    }
    let ring_center = ring_xy.iter().fold((0.0, 0.0), |sum, value| {
        (sum.0 + value.0 * 0.25, sum.1 + value.1 * 0.25)
    });
    // The explicit center is the strongest point; the ring centroid damps a
    // single refined-landmark outlier without regressing to the eyelid box.
    let center = (
        direct.0 * 0.75 + ring_center.0 * 0.25,
        direct.1 * 0.75 + ring_center.1 * 0.25,
    );
    if !(-0.02..=1.02).contains(&center.0) || !(-0.02..=1.02).contains(&center.1) {
        return Err(format!(
            "native MediaPipe iris center is outside the overview: ({:.4},{:.4})",
            center.0, center.1
        ));
    }
    let iris_radius = ring_xy
        .iter()
        .map(|value| (value.0 - center.0).hypot(value.1 - center.1))
        .sum::<f32>()
        * 0.25;
    let first_corner = point(corners[0])?;
    let second_corner = point(corners[1])?;
    let eye_width = (first_corner.0 - second_corner.0).hypot(first_corner.1 - second_corner.1);
    let eye_mid = (
        (first_corner.0 + second_corner.0) * 0.5,
        (first_corner.1 + second_corner.1) * 0.5,
    );
    let center_offset = (center.0 - eye_mid.0).hypot(center.1 - eye_mid.1);
    if !(0.005..=0.25).contains(&eye_width)
        || !(0.0003..=eye_width * 0.45).contains(&iris_radius)
        || center_offset > (eye_width * 0.80).max(0.02)
    {
        return Err(format!(
            "native MediaPipe refined-eye geometry is implausible: width={eye_width:.4} iris_radius={iris_radius:.4} center_offset={center_offset:.4}"
        ));
    }
    Ok(EyeCenter {
        x: center.0,
        y: center.1,
        iris_radius,
        eye_width,
    })
}

fn gray10_to_srgb(samples: &[u16], width: usize, height: usize) -> Result<Vec<u8>, String> {
    if width == 0
        || height == 0
        || samples.len() != width.saturating_mul(height)
        || samples.iter().any(|sample| *sample > 1023)
    {
        return Err("native MediaPipe input violates the linear 10-bit image contract".to_string());
    }
    let mut histogram = [0usize; 1024];
    for &sample in samples {
        histogram[sample as usize] += 1;
    }
    let percentile = |numerator: usize, denominator: usize| {
        let target = (samples.len() * numerator / denominator).max(1);
        let mut cumulative = 0usize;
        histogram
            .iter()
            .position(|count| {
                cumulative += *count;
                cumulative >= target
            })
            .unwrap_or(1023) as u16
    };
    let low = percentile(1, 200); // 0.5th percentile
    let high = percentile(199, 200);
    if high.saturating_sub(low) < 4 {
        return Err(format!(
            "native MediaPipe overview has insufficient linear contrast: {low}..{high}"
        ));
    }
    let span = f32::from(high - low);
    let mut output = Vec::with_capacity(samples.len() * 3);
    for &sample in samples {
        let linear = ((f32::from(sample) - f32::from(low)) / span).clamp(0.0, 1.0);
        let encoded = if linear <= 0.003_130_8 {
            linear * 12.92
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        let value = (encoded * 255.0).round() as u8;
        output.extend_from_slice(&[value, value, value]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_points() -> Vec<(f32, f32, f32)> {
        let mut points = vec![(0.5, 0.5, 0.0); REQUIRED_FACE_LANDMARKS];
        points[RIGHT_EYE_CORNERS[0]] = (0.22, 0.44, 0.0);
        points[RIGHT_EYE_CORNERS[1]] = (0.38, 0.44, 0.0);
        points[LEFT_EYE_CORNERS[0]] = (0.62, 0.45, 0.0);
        points[LEFT_EYE_CORNERS[1]] = (0.78, 0.45, 0.0);
        points[RIGHT_IRIS_CENTER] = (0.31, 0.44, 0.0);
        points[LEFT_IRIS_CENTER] = (0.69, 0.45, 0.0);
        for (indices, center) in [
            (RIGHT_IRIS_RING, (0.31, 0.44)),
            (LEFT_IRIS_RING, (0.69, 0.45)),
        ] {
            for (index, offset) in
                indices
                    .into_iter()
                    .zip([(-0.012, 0.0), (0.0, -0.012), (0.012, 0.0), (0.0, 0.012)])
            {
                points[index] = (center.0 + offset.0, center.1 + offset.1, 0.0);
            }
        }
        points
    }

    #[test]
    fn refined_iris_centers_are_sorted_in_image_order() {
        let eyes = extract_eye_centers(&synthetic_points()).unwrap();
        assert!((eyes[0].x - 0.31).abs() < 1.0e-5);
        assert!((eyes[1].x - 0.69).abs() < 1.0e-5);
        assert!((eyes[0].iris_radius - 0.012).abs() < 1.0e-5);
    }

    #[test]
    fn linear_gray_is_encoded_as_packed_srgb_without_downsampling() {
        let samples = (0..64).map(|value| value * 16).collect::<Vec<_>>();
        let rgb = gray10_to_srgb(&samples, 8, 8).unwrap();
        assert_eq!(rgb.len(), samples.len() * 3);
        assert!(rgb
            .chunks_exact(3)
            .all(|pixel| pixel[0] == pixel[1] && pixel[1] == pixel[2]));
        assert!(rgb[0] < rgb[rgb.len() - 3]);
    }

    #[test]
    fn impossible_eye_pair_is_rejected() {
        let mut points = synthetic_points();
        for index in [LEFT_IRIS_CENTER]
            .into_iter()
            .chain(LEFT_IRIS_RING)
            .chain(LEFT_EYE_CORNERS)
        {
            points[index].0 -= 0.37;
        }
        assert!(extract_eye_centers(&points).is_err());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn c_abi_layout_matches_mediapipe_tasks() {
        assert_eq!(std::mem::size_of::<BaseOptionsC>(), 32);
        assert_eq!(std::mem::size_of::<FaceLandmarkerOptionsC>(), 64);
        assert_eq!(std::mem::size_of::<NormalizedLandmarkC>(), 40);
        assert_eq!(std::mem::size_of::<FaceLandmarkerResultC>(), 48);
    }

    #[test]
    #[ignore = "requires an explicitly supplied lossless sensor overview and runtime assets"]
    fn native_runtime_smoke_from_gray16le() {
        let path = env::var_os("BUTTERCUP_MEDIAPIPE_SMOKE_GRAY16")
            .map(PathBuf::from)
            .expect("set BUTTERCUP_MEDIAPIPE_SMOKE_GRAY16");
        let bytes = std::fs::read(path).unwrap();
        let samples = bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        assert_eq!(samples.len(), 500 * 375);
        let (_, first) = detect_eye_centers_gray10(&samples, 500, 375).unwrap();
        let (_, second) = detect_eye_centers_gray10(&samples, 500, 375).unwrap();
        eprintln!("native MediaPipe smoke detection: {second:?}");
        assert_eq!(second.landmark_count, REQUIRED_FACE_LANDMARKS);
        for (first, second) in first.eyes.iter().zip(second.eyes.iter()) {
            assert!((first.x - second.x).abs() < 1.0e-6);
            assert!((first.y - second.y).abs() < 1.0e-6);
        }
    }
}
