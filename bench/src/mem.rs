//! Resident-set probing and a memory-budget watchdog.
//!
//! The hard constraint on this work is "never run the machine out of memory."
//! Every benchmark process polls its own RSS and aborts loudly the moment it
//! crosses the budget, rather than letting the OS swap or the kernel OOM-kill.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// `mach_task_basic_info` (from `<mach/task_info.h>`). mach2 0.4 doesn't export
/// this struct, so we mirror its layout and ask `task_info` to fill it.
#[repr(C)]
#[derive(Default)]
struct MachTaskBasicInfo {
    virtual_size: u64,
    resident_size: u64,
    resident_size_max: u64,
    user_time: [i32; 2],
    system_time: [i32; 2],
    policy: i32,
    suspend_count: i32,
}

/// Current resident set size of this process, in bytes, via Mach `task_info`.
pub fn rss_bytes() -> u64 {
    use mach2::kern_return::KERN_SUCCESS;
    use mach2::message::mach_msg_type_number_t;
    use mach2::task::task_info;
    use mach2::task_info::MACH_TASK_BASIC_INFO;
    use mach2::traps::mach_task_self;

    unsafe {
        let mut info = MachTaskBasicInfo::default();
        let mut count: mach_msg_type_number_t =
            (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<i32>())
                as mach_msg_type_number_t;
        let kr = task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as *mut i32,
            &mut count,
        );
        if kr == KERN_SUCCESS {
            info.resident_size
        } else {
            0
        }
    }
}

/// Tracks peak RSS and aborts if a budget is exceeded.
pub struct Watchdog {
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Watchdog {
    /// Spawn a watchdog polling every `interval`. If `budget_bytes` is `Some`
    /// and RSS crosses it, print a diagnostic and `abort()` the process.
    pub fn spawn(budget_bytes: Option<u64>, interval: Duration) -> Self {
        let peak = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let peak2 = peak.clone();
        let stop2 = stop.clone();
        let handle = std::thread::spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                let rss = rss_bytes();
                peak2.fetch_max(rss, Ordering::Relaxed);
                if let Some(budget) = budget_bytes {
                    if rss > budget {
                        eprintln!(
                            "\n[watchdog] RSS {:.2} GB exceeded budget {:.2} GB — aborting to protect the machine.",
                            rss as f64 / 1e9,
                            budget as f64 / 1e9,
                        );
                        std::process::abort();
                    }
                }
                std::thread::sleep(interval);
            }
        });
        Watchdog {
            peak,
            stop,
            handle: Some(handle),
        }
    }

    pub fn peak_bytes(&self) -> u64 {
        self.peak.load(Ordering::Relaxed).max(rss_bytes())
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
