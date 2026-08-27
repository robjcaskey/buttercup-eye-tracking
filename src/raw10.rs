pub fn validate_raw10_geometry(
    payload_len: usize,
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), String> {
    if width == 0 || height == 0 || width % 4 != 0 {
        return Err(format!("invalid RAW10 dimensions {width}x{height}"));
    }
    let expected_stride = width / 4 * 5;
    if stride != expected_stride || payload_len != stride.saturating_mul(height) {
        return Err(format!(
            "RAW10 geometry mismatch {width}x{height} stride={stride} payload={payload_len}"
        ));
    }
    Ok(())
}

pub fn try_unpack_raw10(
    payload: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Vec<u16>, String> {
    validate_raw10_geometry(payload.len(), width, height, stride)?;
    Ok(unpack_raw10(payload, width, height, stride))
}

pub fn unpack_raw10(payload: &[u8], width: usize, height: usize, stride: usize) -> Vec<u16> {
    debug_assert!(validate_raw10_geometry(payload.len(), width, height, stride).is_ok());
    let mut pixels = vec![0u16; width * height];
    for y in 0..height {
        let row = &payload[y * stride..(y + 1) * stride];
        for group in 0..width / 4 {
            let offset = group * 5;
            let word = (row[offset] as u64)
                | ((row[offset + 1] as u64) << 8)
                | ((row[offset + 2] as u64) << 16)
                | ((row[offset + 3] as u64) << 24)
                | ((row[offset + 4] as u64) << 32);
            let pixel = y * width + group * 4;
            pixels[pixel] = (word & 0x3ff) as u16;
            pixels[pixel + 1] = ((word >> 10) & 0x3ff) as u16;
            pixels[pixel + 2] = ((word >> 20) & 0x3ff) as u16;
            pixels[pixel + 3] = ((word >> 30) & 0x3ff) as u16;
        }
    }
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(values: &[u16]) -> Vec<u8> {
        let mut output = Vec::new();
        for group in values.chunks_exact(4) {
            let word = group[0] as u64
                | ((group[1] as u64) << 10)
                | ((group[2] as u64) << 20)
                | ((group[3] as u64) << 30);
            output.extend_from_slice(&word.to_le_bytes()[..5]);
        }
        output
    }

    #[test]
    fn packed_little_endian_groups_round_trip() {
        let expected = vec![0, 1, 511, 1023, 17, 333, 777, 999];
        let payload = pack(&expected);
        assert_eq!(try_unpack_raw10(&payload, 8, 1, 10).unwrap(), expected);
    }

    #[test]
    fn malformed_geometry_is_rejected_before_indexing() {
        assert!(try_unpack_raw10(&[0; 5], 5, 1, 5).is_err());
        assert!(try_unpack_raw10(&[0; 4], 4, 1, 5).is_err());
    }
}
