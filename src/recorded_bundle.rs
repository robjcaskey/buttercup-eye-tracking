//! Shared bounded reader for native recording directories and uncompressed tar bundles.
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug)]
pub struct TarEntry {
    data_offset: u64,
    size: u64,
}

#[derive(Debug)]
pub enum BundleSource {
    Directory(PathBuf),
    Tar {
        path: PathBuf,
        entries: HashMap<String, TarEntry>,
    },
}

fn tar_text(field: &[u8]) -> String {
    let end = field
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(field.len());
    String::from_utf8_lossy(&field[..end]).trim().to_string()
}

fn tar_octal(field: &[u8]) -> Result<u64, String> {
    let text = String::from_utf8_lossy(field)
        .trim_matches(|character: char| character == '\0' || character.is_ascii_whitespace())
        .to_string();
    if text.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(&text, 8).map_err(|error| format!("invalid tar size {text:?}: {error}"))
}

impl BundleSource {
    pub fn open(path: &Path) -> Result<Self, String> {
        if path.is_dir() {
            return Ok(Self::Directory(path.to_path_buf()));
        }
        let mut file = File::open(path)
            .map_err(|error| format!("open RAW bundle {}: {error}", path.display()))?;
        let file_size = file
            .metadata()
            .map_err(|error| format!("stat {}: {error}", path.display()))?
            .len();
        let mut entries = HashMap::new();
        let mut header_offset = 0u64;
        while header_offset + 512 <= file_size {
            file.seek(SeekFrom::Start(header_offset))
                .map_err(|error| format!("seek tar: {error}"))?;
            let mut header = [0u8; 512];
            file.read_exact(&mut header)
                .map_err(|error| format!("read tar header: {error}"))?;
            if header.iter().all(|byte| *byte == 0) {
                break;
            }
            let name = tar_text(&header[..100]);
            let prefix = tar_text(&header[345..500]);
            let name = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            let size = tar_octal(&header[124..136])?;
            let data_offset = header_offset + 512;
            if data_offset.saturating_add(size) > file_size {
                return Err(format!("tar entry {name:?} extends beyond bundle"));
            }
            entries.insert(name, TarEntry { data_offset, size });
            header_offset = data_offset + size.div_ceil(512) * 512;
        }
        if entries.is_empty() {
            return Err(format!(
                "{} is not a populated POSIX tar bundle",
                path.display()
            ));
        }
        Ok(Self::Tar {
            path: path.to_path_buf(),
            entries,
        })
    }

    pub fn read_range(&self, name: &str, offset: u64, length: usize) -> Result<Vec<u8>, String> {
        let length_u64 = length as u64;
        let (path, absolute_offset, entry_size) = match self {
            Self::Directory(root) => {
                let path = root.join(name);
                let size = fs::metadata(&path)
                    .map_err(|error| format!("stat {}: {error}", path.display()))?
                    .len();
                (path, offset, size)
            }
            Self::Tar { path, entries } => {
                let entry = entries
                    .get(name)
                    .ok_or_else(|| format!("tar bundle lacks entry {name:?}"))?;
                (path.clone(), entry.data_offset + offset, entry.size)
            }
        };
        if offset.saturating_add(length_u64) > entry_size {
            return Err(format!(
                "read {name:?} range {offset}+{length} exceeds {entry_size} bytes"
            ));
        }
        let mut file =
            File::open(&path).map_err(|error| format!("open {}: {error}", path.display()))?;
        file.seek(SeekFrom::Start(absolute_offset))
            .map_err(|error| format!("seek {}: {error}", path.display()))?;
        let mut bytes = vec![0u8; length];
        file.read_exact(&mut bytes)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        Ok(bytes)
    }

    /// File path and absolute byte range of an entry, so an index can address
    /// native RAW bytes inside an uncompressed bundle without copying it.
    pub fn entry_location(&self, name: &str) -> Result<(PathBuf, u64, u64), String> {
        match self {
            Self::Directory(root) => {
                let path = root.join(name);
                let size = fs::metadata(&path)
                    .map_err(|error| format!("stat {}: {error}", path.display()))?
                    .len();
                Ok((path, 0, size))
            }
            Self::Tar { path, entries } => entries
                .get(name)
                .map(|entry| (path.clone(), entry.data_offset, entry.size))
                .ok_or_else(|| format!("tar bundle lacks entry {name:?}")),
        }
    }

    pub fn read_entry(&self, name: &str) -> Result<Vec<u8>, String> {
        let size = match self {
            Self::Directory(root) => fs::metadata(root.join(name))
                .map_err(|error| format!("stat bundle entry {name:?}: {error}"))?
                .len(),
            Self::Tar { entries, .. } => {
                entries
                    .get(name)
                    .ok_or_else(|| format!("tar bundle lacks entry {name:?}"))?
                    .size
            }
        };
        let length = usize::try_from(size)
            .map_err(|_| format!("bundle entry {name:?} is too large for this host"))?;
        self.read_range(name, 0, length)
    }
}

/// Strict, bounded OIM1 reader shared by recording diagnostics. An incomplete
/// record is an error; it cannot silently erase a later target/removal event.
pub struct MetadataRecords<R: Read> {
    reader: R,
    finished: bool,
}
pub fn metadata_records<R: Read>(reader: R) -> MetadataRecords<R> {
    MetadataRecords {
        reader,
        finished: false,
    }
}
impl<R: Read> Iterator for MetadataRecords<R> {
    type Item = Result<serde_json::Value, String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let mut h = [0u8; 24];
        match self.reader.read(&mut h[..1]) {
            Ok(0) => {
                self.finished = true;
                return None;
            }
            Ok(_) => {}
            Err(e) => {
                self.finished = true;
                return Some(Err(e.to_string()));
            }
        }
        let result = (|| {
            self.reader
                .read_exact(&mut h[1..])
                .map_err(|e| format!("truncated OIM1 header: {e}"))?;
            let n = u32::from_le_bytes(h[8..12].try_into().unwrap()) as usize;
            if &h[..4] != b"OIM1"
                || h[4..6] != 1u16.to_le_bytes()
                || h[6..8] != 24u16.to_le_bytes()
                || h[12..].iter().any(|b| *b != 0)
                || !(1..=4_000_000).contains(&n)
            {
                return Err("invalid or oversized OIM1 header".into());
            }
            let mut payload = vec![0; n];
            self.reader
                .read_exact(&mut payload)
                .map_err(|e| format!("truncated OIM1 payload: {e}"))?;
            serde_json::from_slice(&payload).map_err(|e| e.to_string())
        })();
        if result.is_err() {
            self.finished = true;
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record(value: &serde_json::Value) -> Vec<u8> {
        let p = serde_json::to_vec(value).unwrap();
        let mut h = vec![0; 24];
        h[..4].copy_from_slice(b"OIM1");
        h[4..6].copy_from_slice(&1u16.to_le_bytes());
        h[6..8].copy_from_slice(&24u16.to_le_bytes());
        h[8..12].copy_from_slice(&(p.len() as u32).to_le_bytes());
        h.extend(p);
        h
    }
    #[test]
    fn incomplete_removal_is_not_silent_end_of_evidence() {
        let start = record(&serde_json::json!({"target":"visible"}));
        let stop = record(&serde_json::json!({"target":null}));
        let mut bytes = start.clone();
        bytes.extend_from_slice(&stop[..stop.len() - 1]);
        let mut rows = metadata_records(bytes.as_slice());
        assert_eq!(rows.next().unwrap().unwrap()["target"], "visible");
        assert!(rows.next().unwrap().is_err());
        assert!(rows.next().is_none());
        let mut complete = start;
        complete.extend(stop);
        assert_eq!(
            metadata_records(complete.as_slice())
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
                .len(),
            2
        );
    }
    #[test]
    fn unknown_metadata_version_and_unbounded_payload_are_refused() {
        let mut bytes = record(&serde_json::json!({"a":1}));
        bytes[4] = 2;
        assert!(metadata_records(bytes.as_slice()).next().unwrap().is_err());
        bytes[4] = 1;
        bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(metadata_records(bytes.as_slice()).next().unwrap().is_err());
    }
}
