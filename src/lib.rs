pub mod engine;
pub mod nn;

use std::sync::atomic::{AtomicUsize, Ordering};

///
/// NOTE: Refer safe singleton globals in Rust: https://stackoverflow.com/a/27826181/6196679
///
fn get_id() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}
