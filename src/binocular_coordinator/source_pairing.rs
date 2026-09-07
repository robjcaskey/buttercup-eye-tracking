//! Bounded pairing of actual detector source exposures, never display frames.
//! A caller explicitly advances the stream lineage. A delayed result cannot
//! reset that lineage, pair to a nearest timestamp, or supply a second vote.

use crate::roi_evidence::{ExposureKey, SourceClock};
use std::collections::VecDeque;
use std::sync::Arc;

const MAX_SOURCES_PER_ROI: usize = 32;
const MAX_SOURCE_SPAN_NS: u64 = 1_500_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PairingUnavailable {
    WrongClock,
    InvalidRoi,
    DuplicateSource,
    ConflictingSource,
    OutsideSourceWindow,
}

pub(crate) struct SourcePairer<T> {
    clock: Option<SourceClock>,
    newest_timestamp_ns: u64,
    rows: [VecDeque<(ExposureKey, Arc<T>)>; 2],
}

impl<T> Default for SourcePairer<T> {
    fn default() -> Self {
        Self {clock: None, newest_timestamp_ns: 0, rows: std::array::from_fn(|_| VecDeque::new())}
    }
}

impl<T> SourcePairer<T> {
    pub(crate) fn begin(&mut self, clock: SourceClock) {
        self.clock = Some(clock);
        self.newest_timestamp_ns = 0;
        for rows in &mut self.rows {rows.clear();}
    }

    pub(crate) fn insert(&mut self, key: ExposureKey, value: T)
        -> Result<[Option<Arc<T>>; 2], PairingUnavailable>
    {
        if self.clock != Some(key.clock) {return Err(PairingUnavailable::WrongClock);}
        let eye = key.roi.0.checked_sub(1).filter(|i| *i < 2)
            .ok_or(PairingUnavailable::InvalidRoi)? as usize;
        if self.newest_timestamp_ns.saturating_sub(key.timestamp_ns) > MAX_SOURCE_SPAN_NS {
            return Err(PairingUnavailable::OutsideSourceWindow);
        }
        if let Some((previous, _)) = self.rows[eye].iter().find(|(k, _)| k.timestamp_ns == key.timestamp_ns) {
            return Err(if *previous == key {PairingUnavailable::DuplicateSource} else {PairingUnavailable::ConflictingSource});
        }
        self.newest_timestamp_ns = self.newest_timestamp_ns.max(key.timestamp_ns);
        self.rows[eye].push_back((key, Arc::new(value)));
        for rows in &mut self.rows {
            rows.retain(|(k, _)| self.newest_timestamp_ns.saturating_sub(k.timestamp_ns) <= MAX_SOURCE_SPAN_NS);
            while rows.len() > MAX_SOURCES_PER_ROI {rows.pop_front();}
        }
        Ok(std::array::from_fn(|eye| self.rows[eye].iter()
            .find(|(k, _)| k.timestamp_ns == key.timestamp_ns).map(|(_, v)| Arc::clone(v))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roi_evidence::RoiId;
    fn key(eye:u32,sequence:u64,time:u64)->ExposureKey {
        ExposureKey {roi:RoiId(eye),sequence,timestamp_ns:time,clock:SourceClock {domain:1,epoch:7}}
    }
    #[test]
    fn source_timestamp_pairs_independently_of_roi_sequence_and_completion_order() {
        let mut pairer=SourcePairer::default();pairer.begin(key(1,0,0).clock);
        assert!(pairer.insert(key(1,100,50),"right-old").unwrap()[1].is_none());
        assert!(pairer.insert(key(1,101,60),"right-new").unwrap()[1].is_none());
        let pair=pairer.insert(key(2,4000,50),"left-old").unwrap();
        assert_eq!(pair.map(|v|v.map(|v|*v)),[Some("right-old"),Some("left-old")]);
        assert!(pairer.insert(key(2,4001,59),"left-unpaired").unwrap()[0].is_none());
        assert_eq!(pairer.insert(key(2,4000,50),"repeat"),Err(PairingUnavailable::DuplicateSource));
        assert_eq!(pairer.insert(key(2,4002,50),"conflict"),Err(PairingUnavailable::ConflictingSource));
    }
    #[test]
    fn reconnects_missing_eyes_and_stale_results_never_forge_synchrony() {
        let mut pairer=SourcePairer::default();pairer.begin(key(1,0,0).clock);
        pairer.insert(key(1,0,50),1).unwrap();
        let mut fresh=key(2,0,50);fresh.clock.epoch+=1;pairer.begin(fresh.clock);
        assert_eq!(pairer.insert(key(1,0,50),0),Err(PairingUnavailable::WrongClock));
        assert!(pairer.insert(fresh,2).unwrap()[0].is_none());
        fresh.timestamp_ns+=MAX_SOURCE_SPAN_NS+1;fresh.sequence+=1;pairer.insert(fresh,3).unwrap();
        let mut old=fresh;old.timestamp_ns=50;
        assert_eq!(pairer.insert(old,0),Err(PairingUnavailable::OutsideSourceWindow));
        for i in 0..100 {fresh.sequence+=1;fresh.timestamp_ns+=1;pairer.insert(fresh,i).unwrap();}
        assert_eq!(pairer.rows[1].len(),MAX_SOURCES_PER_ROI);
        assert!(pairer.rows[0].is_empty());
    }
}
