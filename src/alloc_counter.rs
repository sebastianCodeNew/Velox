//! Velox Heap Allocation Counter
//!
//! A process-wide [`GlobalAlloc`] wrapper around the system allocator that counts every
//! heap allocation request (`alloc`, `alloc_zeroed`, `realloc`) made by the *current thread*.
//!
//! This turns the Velox Native Runtime's "zero dynamic heap allocations during inference"
//! invariant from an asserted claim into an empirically measured quantity: callers wrap a
//! region of code in [`count_allocations`] and receive the exact number of allocator calls
//! that the region issued on the calling thread.
//!
//! Counting is per-thread (via a `const`-initialised, destructor-free thread local), so
//! concurrently running tests or worker threads never contaminate each other's measurements.
//! Accessing such a thread local performs no allocation itself, which makes it safe to use
//! from inside the global allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// Global allocator that forwards to [`System`] while counting allocation requests per thread.
pub struct CountingAllocator;

thread_local! {
    static THREAD_ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

#[inline]
fn record_allocation() {
    // `try_with` avoids panicking if the allocator is invoked during thread-local teardown.
    let _ = THREAD_ALLOCATIONS.try_with(|count| count.set(count.get().wrapping_add(1)));
}

// SAFETY: Every method forwards directly to the system allocator with the caller's arguments
// unchanged; the only additional work is incrementing a non-allocating thread-local counter.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: Caller upholds `GlobalAlloc::alloc` contract; forwarded verbatim.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: Caller upholds `GlobalAlloc::alloc_zeroed` contract; forwarded verbatim.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        // SAFETY: Caller upholds `GlobalAlloc::realloc` contract; forwarded verbatim.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Caller upholds `GlobalAlloc::dealloc` contract; forwarded verbatim.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[cfg(test)]
#[global_allocator]
static GLOBAL_COUNTING_ALLOCATOR: CountingAllocator = CountingAllocator;

/// Returns the cumulative number of heap allocation requests issued by the current thread.
#[must_use]
pub fn thread_allocation_count() -> u64 {
    THREAD_ALLOCATIONS.try_with(Cell::get).unwrap_or(0)
}

/// Executes `f` and returns its result together with the exact number of heap allocation
/// requests (`alloc` + `alloc_zeroed` + `realloc`) it issued on the current thread.
pub fn count_allocations<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let before = thread_allocation_count();
    let result = f();
    let after = thread_allocation_count();
    (result, after.wrapping_sub(before))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counts_vec_allocation_and_growth() {
        let (_, allocs) = count_allocations(|| {
            let mut v: Vec<u64> = Vec::with_capacity(1);
            v.push(1);
            v.push(2); // forces a realloc
            std::hint::black_box(&v);
        });
        assert!(allocs >= 2, "expected alloc + realloc to be counted, got {allocs}");
    }

    #[test]
    fn test_pure_arithmetic_region_counts_zero() {
        let mut buf = [0.0f32; 64];
        let (sum, allocs) = count_allocations(|| {
            for (i, x) in buf.iter_mut().enumerate() {
                *x = i as f32 * 0.5;
            }
            buf.iter().sum::<f32>()
        });
        assert_eq!(allocs, 0);
        assert!((sum - 1008.0).abs() < 1e-3);
    }

    #[test]
    fn test_counts_are_thread_isolated() {
        let handle = std::thread::spawn(|| {
            for _ in 0..1000 {
                std::hint::black_box(vec![0u8; 32]);
            }
        });
        let (_, allocs) = count_allocations(|| {
            let mut acc = 0u64;
            for i in 0..100_000u64 {
                acc = acc.wrapping_add(i);
            }
            std::hint::black_box(acc);
        });
        handle.join().expect("allocating thread panicked");
        assert_eq!(allocs, 0, "allocations from another thread leaked into this thread's count");
    }
}
