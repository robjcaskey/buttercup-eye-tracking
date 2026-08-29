//! Low-distraction spatial frame identity for screen-to-eye reflection timing.
//!
//! The logical code is an 8 x 4 balanced lattice. The current stimulus tiles
//! that entire lattice twice horizontally and twice vertically, providing
//! four spatially separated observations of every logical cell without
//! changing the codeword or its mean level. Sixteen logical symbols each
//! occupy a complementary diagonal inside one of eight 2 x 2 logical blocks.
//! Every block therefore contains exactly two positive and two negative cells,
//! making both local and whole-screen mean level invariant. Pair differences
//! reject exposure and illumination drift. Legacy recordings use eleven Gray
//! counter bits plus a session-seeded CRC-4. New stimuli apply an invertible
//! temporal permutation to the fifteen session+counter bits so causal captures
//! distinguish absolute phase much sooner. The final symbol is a fixed
//! orientation/polarity pilot in both schemes.

// This source is path-included by two binaries which intentionally use
// different halves of the shared encoder/decoder API.
#![allow(dead_code)]

pub const GRID_COLUMNS: usize = 8;
pub const GRID_ROWS: usize = 4;
pub const PHYSICAL_CELL_COUNT: usize = GRID_COLUMNS * GRID_ROWS;
pub const SPATIAL_REPEAT_COLUMNS: usize = 2;
pub const SPATIAL_REPEAT_ROWS: usize = 2;
pub const DISPLAY_GRID_COLUMNS: usize = GRID_COLUMNS * SPATIAL_REPEAT_COLUMNS;
pub const DISPLAY_GRID_ROWS: usize = GRID_ROWS * SPATIAL_REPEAT_ROWS;
pub const DISPLAY_CELL_COUNT: usize = DISPLAY_GRID_COLUMNS * DISPLAY_GRID_ROWS;
pub const LOGICAL_BIT_COUNT: usize = 16;
pub const COUNTER_BITS: usize = 11;
pub const COUNTER_MODULUS: u16 = 1 << COUNTER_BITS;
pub const COUNTER_MASK: u16 = COUNTER_MODULUS - 1;
pub const CRC_BITS: usize = 4;
pub const CHECKED_COUNTER_BITS: usize = 5;
pub const CHECKED_COUNTER_MODULUS: u16 = 1 << CHECKED_COUNTER_BITS;
pub const CHECKED_COUNTER_MASK: u16 = CHECKED_COUNTER_MODULUS - 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OpticalCodeScheme {
    #[default]
    GrayCrcV1,
    PermutedCounterV2,
    ReedMullerV3,
}

impl OpticalCodeScheme {
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::GrayCrcV1 => "gray-crc-v1",
            Self::PermutedCounterV2 => "permuted-counter-session-v2",
            Self::ReedMullerV3 => "reed-muller-session-v3",
        }
    }

    pub fn from_wire_name(value: &str) -> Option<Self> {
        match value {
            "gray-crc-v1" => Some(Self::GrayCrcV1),
            "permuted-counter-session-v2" => Some(Self::PermutedCounterV2),
            "reed-muller-session-v3" => Some(Self::ReedMullerV3),
            _ => None,
        }
    }

    pub const fn counter_modulus(self) -> u16 {
        match self {
            Self::GrayCrcV1 | Self::PermutedCounterV2 => COUNTER_MODULUS,
            Self::ReedMullerV3 => CHECKED_COUNTER_MODULUS,
        }
    }

    pub const fn correctable_logical_bit_errors(self) -> usize {
        match self {
            Self::ReedMullerV3 => 3,
            Self::GrayCrcV1 | Self::PermutedCounterV2 => 0,
        }
    }
}

/// A bijection over all fifteen session+counter payload bits. Consecutive
/// counters therefore retain an exact inverse identity while their displayed
/// symbols have broad temporal Hamming distance. Each logical symbol still
/// occupies a complementary pair, so local and global screen means remain
/// invariant.
fn permute_payload15(mut value: u16) -> u16 {
    const MASK: u16 = 0x7fff;
    value &= MASK;
    value ^= value >> 7;
    value = value.wrapping_mul(0x4d2d) & MASK;
    value ^= (value << 5) & MASK;
    value ^= value >> 3;
    value = value.wrapping_mul(0x2c2b) & MASK;
    value ^= value >> 7;
    value & MASK
}

/// A first-order Reed-Muller RM(1,4) code has 16 symbols, 5 payload bits and
/// minimum Hamming distance 8. It therefore corrects any three corrupt
/// logical symbols in one camera frame. The four screen copies and the
/// complementary physical-cell pairs sit below this code and reduce the odds
/// that a physical reflection defect becomes a logical-symbol error at all.
fn reed_muller_16_5(payload: u8) -> u16 {
    let linear = payload & 0x0f;
    let constant = (payload >> 4) & 1;
    let mut word = 0u16;
    for coordinate in 0..16u8 {
        let bit = constant ^ ((linear & coordinate).count_ones() as u8 & 1);
        word |= u16::from(bit) << coordinate;
    }
    word
}

/// Keep adjacent optical counters far apart without changing the 32-word
/// Reed-Muller codebook. Multiplication by an odd value is a permutation
/// modulo 32.
fn permute_checked_counter(counter: u16) -> u8 {
    (((counter & CHECKED_COUNTER_MASK) * 13 + 7) & CHECKED_COUNTER_MASK) as u8
}

fn quadratic_truth_table(first: u8, second: u8) -> u16 {
    let mut word = 0u16;
    for coordinate in 0..16u8 {
        let left = (coordinate >> first) & 1;
        let right = (coordinate >> second) & 1;
        word |= u16::from(left & right) << coordinate;
    }
    word
}

/// A nonlinear coset mask makes screen orientation and session identity part
/// of every independently decodable word. The payload code remains RM(1,4),
/// so all 32 counters within one session retain distance 8.
fn checked_session_mask(session_tag: u8) -> u16 {
    const ORIENTATION_MASK: u16 = 0x2d6b;
    let bases = [
        quadratic_truth_table(0, 1),
        quadratic_truth_table(0, 2),
        quadratic_truth_table(0, 3),
        quadratic_truth_table(1, 2),
    ];
    bases
        .into_iter()
        .enumerate()
        .filter(|(bit, _)| session_tag & (1 << bit) != 0)
        .fold(ORIENTATION_MASK, |mask, (_, basis)| mask ^ basis)
}

fn checked_optical_word(counter: u16, session_tag: u8) -> u16 {
    reed_muller_16_5(permute_checked_counter(counter)) ^ checked_session_mask(session_tag & 0x0f)
}

/// Encode two complementary diagonals in every 2 x 2 block. This suppresses
/// gross left/right or top/bottom pulses and makes the visual code resemble a
/// very low-contrast, locally balanced texture rather than a flashing panel.
/// The adjacent diagonal pairing also keeps both members under the same broad
/// corneal shading field while remaining separable in an 8 x 4 reflection.
pub const PAIR_POSITIVE_CELLS: [usize; LOGICAL_BIT_COUNT] =
    [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23];
pub const PAIR_NEGATIVE_CELLS: [usize; LOGICAL_BIT_COUNT] =
    [9, 8, 11, 10, 13, 12, 15, 14, 25, 24, 27, 26, 29, 28, 31, 30];

/// How many complete copies of the canonical 8 x 4 code lattice occupy the
/// displayed screen. A repeat is a full code tile rather than a subdivision
/// of one macrocell, so each logical bit is observed at separated corneal
/// locations. Legacy manifests use `LEGACY`; new stimuli use `CURRENT`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpatialCodeLayout {
    pub repeat_columns: usize,
    pub repeat_rows: usize,
}

impl SpatialCodeLayout {
    pub const LEGACY: Self = Self {
        repeat_columns: 1,
        repeat_rows: 1,
    };
    pub const CURRENT: Self = Self {
        repeat_columns: SPATIAL_REPEAT_COLUMNS,
        repeat_rows: SPATIAL_REPEAT_ROWS,
    };

    pub fn new(repeat_columns: usize, repeat_rows: usize) -> Option<Self> {
        if repeat_columns == 0 || repeat_rows == 0 || repeat_columns > 4 || repeat_rows > 4 {
            return None;
        }
        Some(Self {
            repeat_columns,
            repeat_rows,
        })
    }

    pub fn display_columns(self) -> usize {
        GRID_COLUMNS * self.repeat_columns
    }

    pub fn display_rows(self) -> usize {
        GRID_ROWS * self.repeat_rows
    }

    /// Map a cell from the tiled display lattice back to the canonical 8 x 4
    /// decoder lattice. Returns `None` for coordinates outside this layout.
    pub fn canonical_cell(self, display_column: usize, display_row: usize) -> Option<usize> {
        if display_column >= self.display_columns() || display_row >= self.display_rows() {
            return None;
        }
        Some((display_row % GRID_ROWS) * GRID_COLUMNS + display_column % GRID_COLUMNS)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameCode {
    pub counter_mod: u16,
    pub gray: u16,
    pub crc4: u8,
    pub session_tag: u8,
    /// Bits 0..10 are Gray payload, 11..14 are CRC, and bit 15 is the fixed
    /// orientation/polarity symbol (always one).
    pub logical_word: u16,
}

impl FrameCode {
    pub fn new(presentation_index: u64, session_tag: u8) -> Self {
        Self::from_counter_mod((presentation_index as u16) & COUNTER_MASK, session_tag)
    }

    pub fn from_counter_mod(counter_mod: u16, session_tag: u8) -> Self {
        let counter_mod = counter_mod & COUNTER_MASK;
        let gray = binary_to_gray(counter_mod);
        let crc4 = crc4_gray(gray, session_tag);
        let logical_word = gray | ((crc4 as u16) << COUNTER_BITS) | (1 << 15);
        Self {
            counter_mod,
            gray,
            crc4,
            session_tag: session_tag & 0x0f,
            logical_word,
        }
    }

    pub fn logical_bit(self, index: usize) -> bool {
        debug_assert!(index < LOGICAL_BIT_COUNT);
        self.logical_word & (1 << index) != 0
    }

    pub fn optical_word(self, scheme: OpticalCodeScheme) -> u16 {
        match scheme {
            OpticalCodeScheme::GrayCrcV1 => self.logical_word,
            OpticalCodeScheme::PermutedCounterV2 => {
                let payload =
                    (u16::from(self.session_tag & 0x0f) << COUNTER_BITS) | self.counter_mod;
                permute_payload15(payload) | (1 << 15)
            }
            OpticalCodeScheme::ReedMullerV3 => {
                checked_optical_word(self.counter_mod, self.session_tag)
            }
        }
    }

    pub fn optical_bit(self, index: usize, scheme: OpticalCodeScheme) -> bool {
        debug_assert!(index < LOGICAL_BIT_COUNT);
        self.optical_word(scheme) & (1 << index) != 0
    }

    /// One sign for every physical macrocell. Positive and negative counts
    /// are exactly equal for every codeword.
    pub fn physical_signs(self) -> [i8; PHYSICAL_CELL_COUNT] {
        self.physical_signs_for(OpticalCodeScheme::GrayCrcV1)
    }

    pub fn physical_signs_for(self, scheme: OpticalCodeScheme) -> [i8; PHYSICAL_CELL_COUNT] {
        let mut signs = [0i8; PHYSICAL_CELL_COUNT];
        for logical in 0..LOGICAL_BIT_COUNT {
            let sign = if self.optical_bit(logical, scheme) {
                1
            } else {
                -1
            };
            signs[PAIR_POSITIVE_CELLS[logical]] = sign;
            signs[PAIR_NEGATIVE_CELLS[logical]] = -sign;
        }
        signs
    }

    /// Signs in the current 16 x 8 display lattice. This is intentionally a
    /// 2 x 2 tiling of complete codewords, not four adjacent samples of the
    /// same macrocell.
    pub fn display_signs(self) -> [i8; DISPLAY_CELL_COUNT] {
        self.display_signs_for(OpticalCodeScheme::GrayCrcV1)
    }

    pub fn display_signs_for(self, scheme: OpticalCodeScheme) -> [i8; DISPLAY_CELL_COUNT] {
        let canonical = self.physical_signs_for(scheme);
        std::array::from_fn(|cell| {
            let column = cell % DISPLAY_GRID_COLUMNS;
            let row = cell / DISPLAY_GRID_COLUMNS;
            canonical[SpatialCodeLayout::CURRENT
                .canonical_cell(column, row)
                .expect("current display coordinate is in bounds")]
        })
    }

    pub fn hard_decode(logical_word: u16, session_tag: u8) -> Option<Self> {
        if logical_word & (1 << 15) == 0 {
            return None;
        }
        let gray = logical_word & COUNTER_MASK;
        let crc4 = ((logical_word >> COUNTER_BITS) & 0x0f) as u8;
        if crc4 != crc4_gray(gray, session_tag) {
            return None;
        }
        let counter_mod = gray_to_binary(gray) & COUNTER_MASK;
        Some(Self::from_counter_mod(counter_mod, session_tag))
    }

    pub fn changed_logical_bits(self, next: Self) -> u32 {
        (self.logical_word ^ next.logical_word).count_ones()
    }

    pub fn changed_physical_cells(self, next: Self) -> u32 {
        self.changed_logical_bits(next) * 2
    }
}

pub fn binary_to_gray(value: u16) -> u16 {
    (value ^ (value >> 1)) & COUNTER_MASK
}

pub fn gray_to_binary(mut gray: u16) -> u16 {
    let mut value = gray;
    while gray != 0 {
        gray >>= 1;
        value ^= gray;
    }
    value & COUNTER_MASK
}

/// CRC polynomial x^4 + x + 1 (0b1_0011), initialized from a four-bit
/// per-session tag.  The tag prevents a valid codeword from a different run
/// being silently accepted against the wrong presentation manifest.
pub fn crc4_gray(gray: u16, session_tag: u8) -> u8 {
    let mut register = session_tag & 0x0f;
    for index in (0..COUNTER_BITS).rev() {
        let incoming = ((gray >> index) & 1) as u8;
        let feedback = ((register >> 3) & 1) ^ incoming;
        register = (register << 1) & 0x0f;
        if feedback != 0 {
            register ^= 0x03;
        }
    }
    register & 0x0f
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridTransform {
    Identity,
    MirrorHorizontal,
    MirrorVertical,
    Rotate180,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeGeometry {
    pub transform: GridTransform,
    pub polarity: i8,
}

impl GridTransform {
    pub const ALL: [Self; 4] = [
        Self::Identity,
        Self::MirrorHorizontal,
        Self::MirrorVertical,
        Self::Rotate180,
    ];

    pub fn observed_cell(self, canonical_cell: usize) -> usize {
        let x = canonical_cell % GRID_COLUMNS;
        let y = canonical_cell / GRID_COLUMNS;
        let (x, y) = match self {
            Self::Identity => (x, y),
            Self::MirrorHorizontal => (GRID_COLUMNS - 1 - x, y),
            Self::MirrorVertical => (x, GRID_ROWS - 1 - y),
            Self::Rotate180 => (GRID_COLUMNS - 1 - x, GRID_ROWS - 1 - y),
        };
        y * GRID_COLUMNS + x
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SoftDecode {
    pub counter_mod: u16,
    pub logical_word: u16,
    pub transform: GridTransform,
    pub polarity: i8,
    /// Normalized signed agreement in approximately -1..1.
    pub score: f64,
    pub runner_up_score: f64,
    pub confidence_margin: f64,
    /// Hamming distance after reducing every complementary cell pair to one
    /// hard logical symbol. For RM(1,4), a distance of at most three is a
    /// uniquely correctable per-frame read.
    pub hard_bit_distance: usize,
    pub hard_bit_errors: usize,
}

fn logical_pair_differences(
    cells: &[f64; PHYSICAL_CELL_COUNT],
    transform: GridTransform,
    polarity: i8,
) -> [f64; LOGICAL_BIT_COUNT] {
    let mut differences = [0.0; LOGICAL_BIT_COUNT];
    for logical in 0..LOGICAL_BIT_COUNT {
        let positive = transform.observed_cell(PAIR_POSITIVE_CELLS[logical]);
        let negative = transform.observed_cell(PAIR_NEGATIVE_CELLS[logical]);
        differences[logical] = (cells[positive] - cells[negative]) * f64::from(polarity);
    }
    differences
}

fn robust_soft_bits(differences: [f64; LOGICAL_BIT_COUNT]) -> [f64; LOGICAL_BIT_COUNT] {
    let mut magnitudes = differences.map(f64::abs);
    magnitudes.sort_by(f64::total_cmp);
    let scale = ((magnitudes[7] + magnitudes[8]) * 0.5).max(1.0e-9);
    differences.map(|difference| (difference / (0.70 * scale)).tanh())
}

fn wrapped_candidates(expected: Option<u16>, radius: u16, modulus: u16) -> Vec<u16> {
    let Some(expected) = expected else {
        return (0..modulus).collect();
    };
    let radius = radius.min(modulus.saturating_sub(1));
    let mut candidates = Vec::with_capacity(radius as usize * 2 + 1);
    for offset in -(radius as i32)..=(radius as i32) {
        let candidate = (i32::from(expected % modulus) + offset).rem_euclid(i32::from(modulus));
        let candidate = candidate as u16;
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    candidates
}

/// Decode sampled macrocell values.  The caller may pass an expected modulo
/// counter and a temporal search radius; without one, all 2048 valid CRC
/// codewords are searched.  Mirroring and opponent-axis polarity are inferred
/// jointly from the fixed pilot and CRC-valid codebook.
pub fn decode_soft_cells(
    cells: &[f64; PHYSICAL_CELL_COUNT],
    session_tag: u8,
    expected: Option<u16>,
    temporal_radius: u16,
) -> Option<SoftDecode> {
    decode_soft_cells_with_scheme(
        cells,
        session_tag,
        expected,
        temporal_radius,
        OpticalCodeScheme::GrayCrcV1,
    )
}

pub fn decode_soft_cells_with_scheme(
    cells: &[f64; PHYSICAL_CELL_COUNT],
    session_tag: u8,
    expected: Option<u16>,
    temporal_radius: u16,
    scheme: OpticalCodeScheme,
) -> Option<SoftDecode> {
    decode_soft_cells_impl(
        cells,
        None,
        session_tag,
        expected,
        temporal_radius,
        None,
        scheme,
    )
}

/// Decode while retaining the orientation and chromatic polarity established
/// by a multi-frame projective locator. A rigid corneal patch cannot mirror or
/// invert between adjacent global-shutter frames; allowing those states to
/// float gives static iris or glasses texture eight chances to impersonate a
/// different counter value.
pub fn decode_soft_cells_constrained(
    cells: &[f64; PHYSICAL_CELL_COUNT],
    session_tag: u8,
    expected: Option<u16>,
    temporal_radius: u16,
    geometry: DecodeGeometry,
) -> Option<SoftDecode> {
    decode_soft_cells_constrained_with_scheme(
        cells,
        session_tag,
        expected,
        temporal_radius,
        geometry,
        OpticalCodeScheme::GrayCrcV1,
    )
}

pub fn decode_soft_cells_constrained_with_scheme(
    cells: &[f64; PHYSICAL_CELL_COUNT],
    session_tag: u8,
    expected: Option<u16>,
    temporal_radius: u16,
    geometry: DecodeGeometry,
    scheme: OpticalCodeScheme,
) -> Option<SoftDecode> {
    decode_soft_cells_impl(
        cells,
        None,
        session_tag,
        expected,
        temporal_radius,
        Some(geometry),
        scheme,
    )
}

/// Decode one grid with the previous sampled grid and decoded counter as an
/// additional raw-delta constraint. Static corneal shading and exposure
/// offsets cancel in the pair differences; the transition term asks whether
/// the few cells that actually changed agree with the candidate Gray/CRC
/// transition. Absolute evidence remains dominant so an LCD transition caught
/// halfway cannot drag a strong current-frame code backward.
pub fn decode_soft_cells_temporal(
    cells: &[f64; PHYSICAL_CELL_COUNT],
    previous_cells: &[f64; PHYSICAL_CELL_COUNT],
    previous_counter: u16,
    session_tag: u8,
    expected: Option<u16>,
    temporal_radius: u16,
) -> Option<SoftDecode> {
    decode_soft_cells_temporal_with_scheme(
        cells,
        previous_cells,
        previous_counter,
        session_tag,
        expected,
        temporal_radius,
        OpticalCodeScheme::GrayCrcV1,
    )
}

pub fn decode_soft_cells_temporal_with_scheme(
    cells: &[f64; PHYSICAL_CELL_COUNT],
    previous_cells: &[f64; PHYSICAL_CELL_COUNT],
    previous_counter: u16,
    session_tag: u8,
    expected: Option<u16>,
    temporal_radius: u16,
    scheme: OpticalCodeScheme,
) -> Option<SoftDecode> {
    decode_soft_cells_impl(
        cells,
        Some((previous_cells, previous_counter)),
        session_tag,
        expected,
        temporal_radius,
        None,
        scheme,
    )
}

pub fn decode_soft_cells_temporal_constrained(
    cells: &[f64; PHYSICAL_CELL_COUNT],
    previous_cells: &[f64; PHYSICAL_CELL_COUNT],
    previous_counter: u16,
    session_tag: u8,
    expected: Option<u16>,
    temporal_radius: u16,
    geometry: DecodeGeometry,
) -> Option<SoftDecode> {
    decode_soft_cells_temporal_constrained_with_scheme(
        cells,
        previous_cells,
        previous_counter,
        session_tag,
        expected,
        temporal_radius,
        geometry,
        OpticalCodeScheme::GrayCrcV1,
    )
}

pub fn decode_soft_cells_temporal_constrained_with_scheme(
    cells: &[f64; PHYSICAL_CELL_COUNT],
    previous_cells: &[f64; PHYSICAL_CELL_COUNT],
    previous_counter: u16,
    session_tag: u8,
    expected: Option<u16>,
    temporal_radius: u16,
    geometry: DecodeGeometry,
    scheme: OpticalCodeScheme,
) -> Option<SoftDecode> {
    decode_soft_cells_impl(
        cells,
        Some((previous_cells, previous_counter)),
        session_tag,
        expected,
        temporal_radius,
        Some(geometry),
        scheme,
    )
}

fn decode_soft_cells_impl(
    cells: &[f64; PHYSICAL_CELL_COUNT],
    temporal: Option<(&[f64; PHYSICAL_CELL_COUNT], u16)>,
    session_tag: u8,
    expected: Option<u16>,
    temporal_radius: u16,
    geometry: Option<DecodeGeometry>,
    scheme: OpticalCodeScheme,
) -> Option<SoftDecode> {
    let modulus = scheme.counter_modulus();
    let candidates = wrapped_candidates(expected, temporal_radius, modulus);
    let mut ranked =
        Vec::<SoftDecode>::with_capacity(candidates.len() * GridTransform::ALL.len() * 2);
    for transform in GridTransform::ALL {
        if geometry.is_some_and(|fixed| transform != fixed.transform) {
            continue;
        }
        for polarity in [-1i8, 1i8] {
            // Packed RAW and the stimulus opponent axis have a known sign.
            // RM(1,4) contains the all-ones word, so permitting an arbitrary
            // polarity inversion would turn every counter into a second,
            // equally perfect counter. Legacy modes retain their historical
            // polarity search; the checked per-frame mode deliberately does
            // not introduce that ambiguity.
            if scheme == OpticalCodeScheme::ReedMullerV3 && polarity < 0 {
                continue;
            }
            if geometry.is_some_and(|fixed| polarity != fixed.polarity) {
                continue;
            }
            let differences = logical_pair_differences(cells, transform, polarity);
            let soft = robust_soft_bits(differences);
            let temporal_observation = temporal.map(|(previous_cells, previous_counter)| {
                let previous_differences =
                    logical_pair_differences(previous_cells, transform, polarity);
                let mut magnitudes = differences.map(f64::abs);
                magnitudes.sort_by(f64::total_cmp);
                let scale = ((magnitudes[7] + magnitudes[8]) * 0.5).max(1.0e-9);
                (
                    previous_differences,
                    FrameCode::from_counter_mod(previous_counter, session_tag),
                    scale,
                )
            });
            for counter_mod in candidates.iter().copied() {
                let code = FrameCode::from_counter_mod(counter_mod, session_tag);
                let mut absolute_score = 0.0;
                let mut hard_bit_errors = 0usize;
                let mut hard_bit_distance = 0usize;
                for (logical, observed) in soft.into_iter().enumerate() {
                    let expected_sign = if code.optical_bit(logical, scheme) {
                        1.0
                    } else {
                        -1.0
                    };
                    absolute_score += expected_sign * observed;
                    hard_bit_errors +=
                        usize::from(observed.abs() >= 0.18 && observed.signum() != expected_sign);
                    hard_bit_distance += usize::from(observed.signum() != expected_sign);
                }
                absolute_score /= LOGICAL_BIT_COUNT as f64;
                let score = temporal_observation.map_or(
                    absolute_score,
                    |(previous_differences, previous_code, scale)| {
                        let mut transition_error = 0.0;
                        for logical in 0..LOGICAL_BIT_COUNT {
                            let current_sign = if code.optical_bit(logical, scheme) {
                                1.0
                            } else {
                                -1.0
                            };
                            let previous_sign = if previous_code.optical_bit(logical, scheme) {
                                1.0
                            } else {
                                -1.0
                            };
                            let expected_delta = (current_sign - previous_sign) * 0.5;
                            let observed_delta = ((differences[logical]
                                - previous_differences[logical])
                                / (2.0 * scale))
                                .clamp(-1.5, 1.5);
                            transition_error += (observed_delta - expected_delta).abs().min(2.0);
                        }
                        let transition_agreement =
                            1.0 - transition_error / (2.0 * LOGICAL_BIT_COUNT as f64);
                        0.76 * absolute_score + 0.24 * (2.0 * transition_agreement - 1.0)
                    },
                );
                ranked.push(SoftDecode {
                    counter_mod,
                    logical_word: code.optical_word(scheme),
                    transform,
                    polarity,
                    score,
                    runner_up_score: f64::NEG_INFINITY,
                    confidence_margin: 0.0,
                    hard_bit_distance,
                    hard_bit_errors,
                });
            }
        }
    }
    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut best = *ranked.first()?;
    // Do not let the same code under an equivalent orientation/polarity count
    // as its own runner-up.  The useful margin is to the next frame identity.
    let runner_up = ranked
        .iter()
        .find(|candidate| candidate.counter_mod != best.counter_mod)
        .map_or(-1.0, |candidate| candidate.score);
    best.runner_up_score = runner_up;
    best.confidence_margin = (best.score - runner_up).max(0.0);
    Some(best)
}

/// Convert a modulo decode into the nearest unwrapped presentation index.
pub fn unwrap_counter_near(counter_mod: u16, expected_presentation: u64) -> u64 {
    unwrap_counter_near_with_scheme(
        counter_mod,
        expected_presentation,
        OpticalCodeScheme::GrayCrcV1,
    )
}

pub fn unwrap_counter_near_with_scheme(
    counter_mod: u16,
    expected_presentation: u64,
    scheme: OpticalCodeScheme,
) -> u64 {
    let modulus = u64::from(scheme.counter_modulus());
    let base = expected_presentation / modulus * modulus + u64::from(counter_mod);
    [
        base.saturating_sub(modulus),
        base,
        base.saturating_add(modulus),
    ]
    .into_iter()
    .min_by_key(|candidate| candidate.abs_diff(expected_presentation))
    .unwrap_or(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_cells(
        code: FrameCode,
        transform: GridTransform,
        polarity: f64,
    ) -> [f64; PHYSICAL_CELL_COUNT] {
        synthetic_cells_for(code, transform, polarity, OpticalCodeScheme::GrayCrcV1)
    }

    fn synthetic_cells_for(
        code: FrameCode,
        transform: GridTransform,
        polarity: f64,
        scheme: OpticalCodeScheme,
    ) -> [f64; PHYSICAL_CELL_COUNT] {
        let canonical = code
            .physical_signs_for(scheme)
            .map(|sign| 0.62 + 0.035 * f64::from(sign) * polarity);
        let mut observed = [0.0; PHYSICAL_CELL_COUNT];
        for canonical_cell in 0..PHYSICAL_CELL_COUNT {
            let observed_cell = transform.observed_cell(canonical_cell);
            let x = observed_cell % GRID_COLUMNS;
            let y = observed_cell / GRID_COLUMNS;
            // Strong global gain/offset plus a smooth spatial shading field.
            observed[observed_cell] =
                0.12 + 1.55 * canonical[canonical_cell] + 0.0012 * x as f64 - 0.0009 * y as f64;
        }
        observed
    }

    #[test]
    fn gray_and_crc_round_trip_every_counter() {
        for session_tag in [0, 1, 7, 15] {
            for counter in 0..COUNTER_MODULUS {
                let code = FrameCode::from_counter_mod(counter, session_tag);
                assert_eq!(gray_to_binary(code.gray), counter);
                assert_eq!(
                    FrameCode::hard_decode(code.logical_word, session_tag),
                    Some(code)
                );
                assert_eq!(
                    code.physical_signs()
                        .iter()
                        .filter(|sign| **sign > 0)
                        .count(),
                    16
                );
                assert_eq!(
                    code.physical_signs()
                        .iter()
                        .filter(|sign| **sign < 0)
                        .count(),
                    16
                );
                for block_y in 0..2 {
                    for block_x in 0..4 {
                        let origin_x = block_x * 2;
                        let origin_y = block_y * 2;
                        let block = [
                            origin_y * GRID_COLUMNS + origin_x,
                            origin_y * GRID_COLUMNS + origin_x + 1,
                            (origin_y + 1) * GRID_COLUMNS + origin_x,
                            (origin_y + 1) * GRID_COLUMNS + origin_x + 1,
                        ];
                        let positive = block
                            .into_iter()
                            .filter(|cell| code.physical_signs()[*cell] > 0)
                            .count();
                        assert_eq!(positive, 2, "unbalanced 2x2 block at {block_x},{block_y}");
                    }
                }
            }
        }
    }

    #[test]
    fn permuted_counter_symbols_are_unique_balanced_and_temporally_dense() {
        let mut words = std::collections::HashSet::with_capacity(1 << 15);
        let mut changed_bits = 0u64;
        let mut transitions = 0u64;
        let mut sparse_transitions = 0u64;
        for session_tag in 0..16u8 {
            for counter in 0..COUNTER_MODULUS {
                let code = FrameCode::from_counter_mod(counter, session_tag);
                let word = code.optical_word(OpticalCodeScheme::PermutedCounterV2);
                assert!(words.insert(word), "duplicate optical word {word:04x}");
                let signs = code.physical_signs_for(OpticalCodeScheme::PermutedCounterV2);
                assert_eq!(signs.iter().filter(|sign| **sign > 0).count(), 16);
                assert_eq!(signs.iter().filter(|sign| **sign < 0).count(), 16);
                if counter + 1 < COUNTER_MODULUS {
                    let next = FrameCode::from_counter_mod(counter + 1, session_tag)
                        .optical_word(OpticalCodeScheme::PermutedCounterV2);
                    let changed = (word ^ next).count_ones();
                    changed_bits += u64::from(changed);
                    transitions += 1;
                    sparse_transitions += u64::from(changed < 4);
                }
            }
        }
        assert_eq!(words.len(), 1 << 15);
        let mean = changed_bits as f64 / transitions as f64;
        assert!(mean >= 6.0, "mean temporal Hamming distance was {mean:.3}");
        assert!(
            sparse_transitions * 10 < transitions,
            "too many sparse temporal transitions: {sparse_transitions}/{transitions}"
        );
    }

    #[test]
    fn current_display_tiles_four_complete_balanced_code_lattices() {
        let code = FrameCode::from_counter_mod(917, 11);
        let canonical = code.physical_signs();
        let displayed = code.display_signs();
        assert_eq!(displayed.len(), DISPLAY_CELL_COUNT);
        for tile_y in 0..SPATIAL_REPEAT_ROWS {
            for tile_x in 0..SPATIAL_REPEAT_COLUMNS {
                let mut positive = 0;
                for row in 0..GRID_ROWS {
                    for column in 0..GRID_COLUMNS {
                        let display_column = tile_x * GRID_COLUMNS + column;
                        let display_row = tile_y * GRID_ROWS + row;
                        let display_cell = display_row * DISPLAY_GRID_COLUMNS + display_column;
                        let canonical_cell = row * GRID_COLUMNS + column;
                        assert_eq!(displayed[display_cell], canonical[canonical_cell]);
                        positive += usize::from(displayed[display_cell] > 0);
                    }
                }
                assert_eq!(positive, PHYSICAL_CELL_COUNT / 2);
            }
        }
    }

    #[test]
    fn spatial_layout_rejects_invalid_density_and_maps_every_repeat() {
        assert!(SpatialCodeLayout::new(0, 1).is_none());
        assert!(SpatialCodeLayout::new(1, 0).is_none());
        assert!(SpatialCodeLayout::new(5, 1).is_none());
        let layout = SpatialCodeLayout::new(2, 2).unwrap();
        for row in 0..layout.display_rows() {
            for column in 0..layout.display_columns() {
                assert_eq!(
                    layout.canonical_cell(column, row),
                    Some((row % GRID_ROWS) * GRID_COLUMNS + column % GRID_COLUMNS)
                );
            }
        }
        assert_eq!(layout.canonical_cell(layout.display_columns(), 0), None);
        assert_eq!(layout.canonical_cell(0, layout.display_rows()), None);
    }

    #[test]
    fn crc_rejects_every_single_bit_error() {
        let code = FrameCode::from_counter_mod(917, 11);
        for bit in 0..LOGICAL_BIT_COUNT {
            assert!(FrameCode::hard_decode(code.logical_word ^ (1 << bit), 11).is_none());
        }
        assert!(FrameCode::hard_decode(code.logical_word, 10).is_none());
    }

    #[test]
    fn consecutive_frames_change_only_a_small_screen_fraction() {
        let mut maximum = 0;
        let mut total = 0u64;
        for counter in 0..COUNTER_MODULUS {
            let current = FrameCode::from_counter_mod(counter, 6);
            let next = FrameCode::from_counter_mod((counter + 1) & COUNTER_MASK, 6);
            let changed = current.changed_physical_cells(next);
            maximum = maximum.max(changed);
            total += u64::from(changed);
        }
        assert!(
            maximum <= 10,
            "maximum changed physical cells was {maximum}"
        );
        assert!(total as f64 / COUNTER_MODULUS as f64 <= 6.5);
    }

    #[test]
    fn soft_decoder_recovers_mirroring_polarity_and_exposure() {
        let code = FrameCode::from_counter_mod(1337, 9);
        for transform in GridTransform::ALL {
            for polarity in [-1.0, 1.0] {
                let cells = synthetic_cells(code, transform, polarity);
                let decoded = decode_soft_cells(&cells, 9, Some(1335), 8).unwrap();
                assert_eq!(decoded.counter_mod, code.counter_mod);
                assert!(decoded.score > 0.85, "{decoded:?}");
                assert!(decoded.confidence_margin > 0.08, "{decoded:?}");
            }
        }
    }

    #[test]
    fn constrained_decoder_retains_the_multiframe_locator_geometry() {
        let code = FrameCode::from_counter_mod(741, 5);
        let cells = synthetic_cells(code, GridTransform::MirrorHorizontal, -1.0);
        let decoded = decode_soft_cells_constrained(
            &cells,
            5,
            Some(740),
            4,
            DecodeGeometry {
                transform: GridTransform::MirrorHorizontal,
                polarity: -1,
            },
        )
        .unwrap();
        assert_eq!(decoded.counter_mod, 741);
        assert_eq!(decoded.transform, GridTransform::MirrorHorizontal);
        assert_eq!(decoded.polarity, -1);
        assert!(decoded.confidence_margin > 0.08, "{decoded:?}");
    }

    #[test]
    fn temporal_codebook_corrects_a_corrupted_pair() {
        let code = FrameCode::from_counter_mod(502, 3);
        let mut cells = synthetic_cells(code, GridTransform::Identity, 1.0);
        // Reverse one complete logical pair and erase half of another.
        let first = 5;
        cells.swap(PAIR_POSITIVE_CELLS[first], PAIR_NEGATIVE_CELLS[first]);
        let second = 12;
        let midpoint =
            (cells[PAIR_POSITIVE_CELLS[second]] + cells[PAIR_NEGATIVE_CELLS[second]]) * 0.5;
        cells[PAIR_POSITIVE_CELLS[second]] = midpoint;
        let decoded = decode_soft_cells(&cells, 3, Some(500), 6).unwrap();
        assert_eq!(decoded.counter_mod, 502);
        assert_eq!(decoded.hard_bit_errors, 1);
    }

    #[test]
    fn raw_delta_term_survives_gain_drift_and_two_erased_pairs() {
        let previous_code = FrameCode::from_counter_mod(880, 12);
        let current_code = FrameCode::from_counter_mod(881, 12);
        let previous = synthetic_cells(previous_code, GridTransform::MirrorHorizontal, 1.0);
        let mut current = synthetic_cells(current_code, GridTransform::MirrorHorizontal, 1.0)
            .map(|value| 0.08 + 1.18 * value);
        for logical in [2usize, 9usize] {
            let first = GridTransform::MirrorHorizontal.observed_cell(PAIR_POSITIVE_CELLS[logical]);
            let second =
                GridTransform::MirrorHorizontal.observed_cell(PAIR_NEGATIVE_CELLS[logical]);
            let midpoint = (current[first] + current[second]) * 0.5;
            current[first] = midpoint;
            current[second] = midpoint;
        }
        let decoded = decode_soft_cells_temporal(
            &current,
            &previous,
            previous_code.counter_mod,
            12,
            Some(881),
            4,
        )
        .unwrap();
        assert_eq!(decoded.counter_mod, 881, "{decoded:?}");
        assert!(decoded.confidence_margin > 0.05, "{decoded:?}");
    }

    #[test]
    fn unwraps_across_modulo_boundary() {
        assert_eq!(unwrap_counter_near(2, 2049), 2050);
        assert_eq!(unwrap_counter_near(2046, 2049), 2046);
        assert_eq!(unwrap_counter_near(7, 8198), 8199);
    }

    #[test]
    fn reed_muller_words_are_unique_and_distance_eight_within_each_session() {
        for session_tag in 0..16u8 {
            let words = (0..CHECKED_COUNTER_MODULUS)
                .map(|counter| checked_optical_word(counter, session_tag))
                .collect::<Vec<_>>();
            let mut unique = words.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), CHECKED_COUNTER_MODULUS as usize);
            for right in 0..words.len() {
                for left in 0..right {
                    assert!((words[left] ^ words[right]).count_ones() >= 8);
                }
            }
        }
    }

    #[test]
    fn reed_muller_session_cosets_do_not_collide() {
        let mut words = Vec::new();
        for session_tag in 0..16u8 {
            for counter in 0..CHECKED_COUNTER_MODULUS {
                words.push(checked_optical_word(counter, session_tag));
            }
        }
        words.sort_unstable();
        words.dedup();
        assert_eq!(words.len(), 16 * CHECKED_COUNTER_MODULUS as usize);
    }

    #[test]
    fn reed_muller_single_frame_decoder_corrects_three_logical_cells() {
        let scheme = OpticalCodeScheme::ReedMullerV3;
        let code = FrameCode::from_counter_mod(21, 9);
        for transform in GridTransform::ALL {
            let mut cells = synthetic_cells_for(code, transform, 1.0, scheme);
            for logical in [1usize, 6, 14] {
                let positive = transform.observed_cell(PAIR_POSITIVE_CELLS[logical]);
                let negative = transform.observed_cell(PAIR_NEGATIVE_CELLS[logical]);
                cells.swap(positive, negative);
            }
            let decoded = decode_soft_cells_constrained_with_scheme(
                &cells,
                9,
                Some(20),
                15,
                DecodeGeometry {
                    transform,
                    polarity: 1,
                },
                scheme,
            )
            .unwrap();
            assert_eq!(decoded.counter_mod, 21, "{decoded:?}");
            assert_eq!(decoded.hard_bit_distance, 3, "{decoded:?}");
            assert!(decoded.confidence_margin > 0.08, "{decoded:?}");
        }
    }

    #[test]
    fn reed_muller_four_error_midpoint_is_rejected_by_distance_and_margin() {
        let scheme = OpticalCodeScheme::ReedMullerV3;
        let code = FrameCode::from_counter_mod(12, 4);
        let other = FrameCode::from_counter_mod(13, 4);
        let differing = (code.optical_word(scheme) ^ other.optical_word(scheme)).count_ones();
        assert_eq!(differing, 8);
        let mut cells = synthetic_cells_for(code, GridTransform::Identity, 1.0, scheme);
        let mut flipped = 0;
        for logical in 0..LOGICAL_BIT_COUNT {
            if code.optical_bit(logical, scheme) != other.optical_bit(logical, scheme)
                && flipped < 4
            {
                cells.swap(PAIR_POSITIVE_CELLS[logical], PAIR_NEGATIVE_CELLS[logical]);
                flipped += 1;
            }
        }
        let decoded = decode_soft_cells_constrained_with_scheme(
            &cells,
            4,
            Some(12),
            15,
            DecodeGeometry {
                transform: GridTransform::Identity,
                polarity: 1,
            },
            scheme,
        )
        .unwrap();
        assert!(
            decoded.hard_bit_distance > scheme.correctable_logical_bit_errors()
                || decoded.confidence_margin < 0.02,
            "{decoded:?}"
        );
    }

    #[test]
    fn unwraps_checked_counter_near_the_host_epoch() {
        let scheme = OpticalCodeScheme::ReedMullerV3;
        assert_eq!(unwrap_counter_near_with_scheme(2, 33, scheme), 34);
        assert_eq!(unwrap_counter_near_with_scheme(30, 33, scheme), 30);
        assert_eq!(unwrap_counter_near_with_scheme(7, 70, scheme), 71);
    }
}
