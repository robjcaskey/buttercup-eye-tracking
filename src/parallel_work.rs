//! Order-preserving data parallelism for pure per-item work on the live RAW
//! loop. Results are returned in input order, so callers that fold them
//! sequentially produce exactly the serial result. Worker count is bounded:
//! this runs on a shared workstation beside inference and other jobs.

/// Upper bound on scoped worker threads for one call.
pub const MAXIMUM_WORKERS: usize = 6;

/// Map `f` over `items` on up to [`MAXIMUM_WORKERS`] scoped threads.
/// `minimum_chunk` keeps tiny inputs serial where thread start-up dominates.
pub fn ordered_map<T: Sync, R: Send>(
    items: &[T],
    minimum_chunk: usize,
    f: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(MAXIMUM_WORKERS)
        .min(items.len().div_ceil(minimum_chunk.max(1)))
        .max(1);
    if workers == 1 {
        return items.iter().map(&f).collect();
    }
    let chunk = items.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let f = &f;
        let handles = items
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(f).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("parallel worker panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn ordered_map_matches_serial_order_for_every_size() {
        for n in [0usize, 1, 5, 63, 64, 65, 1000, 4097] {
            let items = (0..n).collect::<Vec<_>>();
            let serial = items.iter().map(|v| v * 3 + 1).collect::<Vec<_>>();
            assert_eq!(super::ordered_map(&items, 16, |v| v * 3 + 1), serial);
        }
    }
}
