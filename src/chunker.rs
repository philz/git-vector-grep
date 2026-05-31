//! Line-window chunker.
//!
//! Simple, language-agnostic. Splits text into overlapping line windows.
//! Skips binary files and oversized files.

use memchr::memchr;

pub const TARGET_LINES: usize = 40;
pub const OVERLAP_LINES: usize = 8;
pub const MAX_CHARS: usize = 4000;
pub const MIN_FILE_BYTES: usize = 16;
pub const MAX_FILE_BYTES: usize = 2_000_000;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    /// 1-based, inclusive.
    pub start_line: u32,
    /// 1-based, inclusive.
    pub end_line: u32,
}

/// Quick & cheap binary sniff: any NUL in the first 8KB.
pub fn looks_binary(data: &[u8]) -> bool {
    let head = &data[..data.len().min(8192)];
    memchr(0, head).is_some()
}

pub fn chunk_bytes(data: &[u8]) -> Vec<Chunk> {
    if data.len() < MIN_FILE_BYTES || data.len() > MAX_FILE_BYTES {
        return Vec::new();
    }
    if looks_binary(data) {
        return Vec::new();
    }
    let Ok(text) = std::str::from_utf8(data) else {
        return Vec::new();
    };
    chunk_text(text)
}

pub fn chunk_text(text: &str) -> Vec<Chunk> {
    // Collect lines (without the terminator).
    let lines: Vec<&str> = text.split('\n').collect();
    // If the file ends with '\n', the split produces a trailing empty element
    // we don't want to count as a line.
    let n = if lines.last().map(|s| s.is_empty()).unwrap_or(false) && lines.len() > 1 {
        lines.len() - 1
    } else {
        lines.len()
    };
    if n == 0 {
        return Vec::new();
    }

    if n <= TARGET_LINES {
        let body = join_lines(&lines[..n]);
        let body = truncate_chars(&body, MAX_CHARS);
        return vec![Chunk {
            text: body,
            start_line: 1,
            end_line: n as u32,
        }];
    }

    let mut out = Vec::with_capacity((n / (TARGET_LINES - OVERLAP_LINES)) + 1);
    let step = TARGET_LINES - OVERLAP_LINES;
    let mut i: usize = 0;
    loop {
        let end = (i + TARGET_LINES).min(n);
        let body = join_lines(&lines[i..end]);
        let body = truncate_chars(&body, MAX_CHARS);
        out.push(Chunk {
            text: body,
            start_line: (i + 1) as u32,
            end_line: end as u32,
        });
        if end == n {
            break;
        }
        i += step;
    }
    out
}

fn join_lines(lines: &[&str]) -> String {
    let cap: usize = lines.iter().map(|l| l.len() + 1).sum();
    let mut s = String::with_capacity(cap);
    for (i, l) in lines.iter().enumerate() {
        if i > 0 {
            s.push('\n');
        }
        s.push_str(l);
    }
    s
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_owned();
    }
    // Cut at the last char boundary <= max bytes.
    let mut i = max;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    s[..i].to_owned()
}
