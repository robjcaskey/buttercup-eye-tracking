//! A single replaceable waiting item, not a FIFO. In-progress work is owned
//! by the consumer and cannot be replaced. Closing either endpoint wakes it.
use std::sync::{Arc, Condvar, Mutex};
use std::sync::mpsc::{RecvError, TrySendError};

struct State<T> { pending: Option<T>, closed: bool }
struct Shared<T> { state: Mutex<State<T>>, ready: Condvar }
pub(super) struct Sender<T>(Arc<Shared<T>>);
pub(super) struct Receiver<T>(Arc<Shared<T>>);

pub(super) fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let shared = Arc::new(Shared {
        state: Mutex::new(State { pending: None, closed: false }), ready: Condvar::new(),
    });
    (Sender(shared.clone()), Receiver(shared))
}

impl<T> Sender<T> {
    /// True means a waiting item was superseded, before expensive processing.
    pub(super) fn try_send(&self, value: T, replace: impl FnOnce(&T, &T) -> bool)
        -> Result<bool, TrySendError<T>> {
        let Ok(mut state) = self.0.state.lock() else {
            return Err(TrySendError::Disconnected(value));
        };
        if state.closed { return Err(TrySendError::Disconnected(value)); }
        if state.pending.as_ref().is_some_and(|old| !replace(&value, old)) {
            return Err(TrySendError::Full(value));
        }
        let old = state.pending.replace(value);
        let replaced = old.is_some();
        self.0.ready.notify_one();
        drop(state);
        drop(old);
        Ok(replaced)
    }
}
impl<T> Receiver<T> {
    pub(super) fn recv(&self) -> Result<T, RecvError> {
        let mut state = self.0.state.lock().map_err(|_| RecvError)?;
        loop {
            if state.closed { return Err(RecvError); }
            if let Some(value) = state.pending.take() { return Ok(value); }
            state = self.0.ready.wait(state).map_err(|_| RecvError)?;
        }
    }
}
impl<T> Shared<T> {
    fn close(&self) {
        let old = if let Ok(mut state) = self.state.lock() {
            state.closed = true;
            state.pending.take()
        } else { None };
        self.ready.notify_all();
        drop(old);
    }
}
impl<T> Drop for Sender<T> { fn drop(&mut self) { self.0.close(); } }
impl<T> Drop for Receiver<T> { fn drop(&mut self) { self.0.close(); } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overload_keeps_only_latest_and_rejects_stale() {
        let (tx, rx) = channel();
        assert_eq!(tx.try_send(0, |new, old| new > old), Ok(false));
        for n in 1..10000 { assert_eq!(tx.try_send(n, |new, old| new > old), Ok(true)); }
        assert!(matches!(tx.try_send(5, |new, old| new > old), Err(TrySendError::Full(5))));
        assert_eq!(rx.recv(), Ok(9999));
        assert_eq!(tx.try_send(10000, |new, old| new > old), Ok(false));
        assert_eq!(rx.recv(), Ok(10000));
    }
    #[test]
    fn closing_wakes_waiter_and_disposes_pending_work() {
        let (tx, rx) = channel::<usize>();
        let waiter = std::thread::spawn(move || rx.recv());
        drop(tx);
        assert!(waiter.join().unwrap().is_err());
        let (tx, rx) = channel();
        tx.try_send(1, |_, _| true).unwrap();
        drop(tx);
        assert!(rx.recv().is_err());
        let (tx, rx) = channel::<usize>();
        drop(rx);
        assert!(matches!(tx.try_send(1, |_, _| true), Err(TrySendError::Disconnected(1))));
    }
    #[test]
    fn slow_consumer_does_not_let_new_input_replace_in_progress_work() {
        let (tx, rx) = channel();
        tx.try_send(1, |new, old| new > old).unwrap();
        let active = rx.recv().unwrap();
        for n in 2..100 { tx.try_send(n, |new, old| new > old).unwrap(); }
        assert_eq!(active, 1);
        assert_eq!(rx.recv(), Ok(99));
    }
}
