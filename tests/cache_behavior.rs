use anyhow::{bail, Result};
use git_vector_grep::embed::Embed;
use git_vector_grep::indexer::index_repo;
use git_vector_grep::search::Index;
use git_vector_grep::store::Store;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex, MutexGuard};
use tempfile::TempDir;

const PROMPT: &str = "is git-vector-grep re-indexing every time? it seems like it should mostly be cached, but maybe it's just slow and things are changing alot? look into it.";
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct ChunkGroupEnv {
    _lock: MutexGuard<'static, ()>,
}

impl ChunkGroupEnv {
    fn set(value: &str) -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        std::env::set_var("GVG_CHUNK_GROUP", value);
        Self { _lock: lock }
    }
}

impl Drop for ChunkGroupEnv {
    fn drop(&mut self) {
        std::env::remove_var("GVG_CHUNK_GROUP");
    }
}

struct FakeEmbed {
    calls: AtomicUsize,
    fail_after: Option<usize>,
}

impl FakeEmbed {
    fn succeeds() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            fail_after: None,
        }
    }

    fn fail_after(successful_calls: usize) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            fail_after: Some(successful_calls),
        }
    }
}

impl Embed for FakeEmbed {
    fn embed_flat(&self, texts: Vec<String>, _batch_size: usize) -> Result<Vec<f32>> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_after.is_some_and(|limit| call >= limit) {
            bail!("injected embedding failure");
        }
        Ok(texts.into_iter().flat_map(|_| [1.0, 0.0]).collect())
    }

    fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![1.0, 0.0])
    }

    fn dim(&self) -> usize {
        2
    }

    fn short_id(&self) -> &str {
        "test"
    }
}

fn git(repo: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}

fn repo() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    git(dir.path(), &["init", "-q"]);
    git(dir.path(), &["config", "user.name", "Test"]);
    git(dir.path(), &["config", "user.email", "test@example.com"]);
    git(dir.path(), &["config", "core.hooksPath", "/dev/null"]);
    dir
}

fn commit(repo: &Path, message: &str, paths: &[&str]) -> String {
    let mut add = vec!["add"];
    add.extend(paths);
    git(repo, &add);
    let body = format!("{message}\n\nPrompt: {PROMPT}");
    git(repo, &["commit", "-qm", &body]);
    git(repo, &["rev-parse", "HEAD"])
}

fn write(repo: &Path, path: &str, text: &str) {
    fs::write(repo.join(path), text).unwrap();
}

fn replace_note(repo: &Path, ref_name: &str, source_sha: &str, bytes: &[u8]) {
    let parent = git(repo, &["rev-parse", ref_name]);
    let mut child = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["fast-import", "--quiet", "--force", "--done"])
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    let stdin = child.stdin.as_mut().unwrap();
    writeln!(stdin, "blob").unwrap();
    writeln!(stdin, "mark :1").unwrap();
    writeln!(stdin, "data {}", bytes.len()).unwrap();
    stdin.write_all(bytes).unwrap();
    writeln!(stdin, "commit {ref_name}").unwrap();
    writeln!(stdin, "committer Test <test@example.com> 1 +0000").unwrap();
    writeln!(stdin, "data 7").unwrap();
    writeln!(stdin, "corrupt").unwrap();
    writeln!(stdin, "from {parent}").unwrap();
    writeln!(
        stdin,
        "M 100644 :1 {}/{}",
        &source_sha[..2],
        &source_sha[2..]
    )
    .unwrap();
    writeln!(stdin, "done").unwrap();
    drop(child.stdin.take());
    assert!(child.wait().unwrap().success());
}

#[test]
fn branch_switch_reuses_cached_blob_versions() {
    let _env = ChunkGroupEnv::set("512");
    let repo = repo();
    write(repo.path(), "a.txt", "alpha authentication token\n");
    write(repo.path(), "b.txt", "beta virtual machine provisioning\n");
    let base = commit(repo.path(), "initial", &["a.txt", "b.txt"]);

    let embed = FakeEmbed::succeeds();
    let mut store = Store::open(repo.path(), "test", 2).unwrap();
    let stats = index_repo(repo.path(), &mut store, &embed, 16, false, true).unwrap();
    assert_eq!(stats.blobs_embedded, 2);
    store.commit().unwrap();

    write(
        repo.path(),
        "a.txt",
        "rotated browser authentication session\n",
    );
    commit(repo.path(), "change a", &["a.txt"]);
    let mut store = Store::open(repo.path(), "test", 2).unwrap();
    let stats = index_repo(repo.path(), &mut store, &embed, 16, false, true).unwrap();
    assert_eq!(stats.blobs_embedded, 1);
    store.commit().unwrap();
    assert_eq!(store.known_blob_shas().unwrap().len(), 3);

    git(repo.path(), &["checkout", "-q", &base]);
    let mut store = Store::open(repo.path(), "test", 2).unwrap();
    let stats = index_repo(repo.path(), &mut store, &embed, 16, false, true).unwrap();
    assert_eq!(stats.blobs_embedded, 0);
    store.commit().unwrap();
    assert_eq!(store.known_blob_shas().unwrap().len(), 3);

    let index = Index::load(repo.path(), &mut store).unwrap();
    assert_eq!(index.len(), 2, "only current blobs should be loaded");
}

#[test]
fn interrupted_index_keeps_completed_groups() {
    let _env = ChunkGroupEnv::set("1");

    let repo = repo();
    write(repo.path(), "a.txt", "alpha authentication token\n");
    write(repo.path(), "b.txt", "beta virtual machine provisioning\n");
    commit(repo.path(), "initial", &["a.txt", "b.txt"]);

    let embed = FakeEmbed::fail_after(1);
    let mut store = Store::open(repo.path(), "test", 2).unwrap();
    assert!(index_repo(repo.path(), &mut store, &embed, 16, false, true).is_err());

    let store = Store::open(repo.path(), "test", 2).unwrap();
    assert_eq!(
        store.known_blob_shas().unwrap().len(),
        1,
        "the successfully embedded group should be checkpointed"
    );
}

#[test]
fn empty_files_are_cached_as_zero_chunk_sentinels() {
    let _env = ChunkGroupEnv::set("512");
    let repo = repo();
    write(repo.path(), "empty.txt", "");
    write(repo.path(), "code.txt", "virtual machine provisioning\n");
    commit(repo.path(), "initial", &["empty.txt", "code.txt"]);

    let embed = FakeEmbed::succeeds();
    let mut store = Store::open(repo.path(), "test", 2).unwrap();
    let first = index_repo(repo.path(), &mut store, &embed, 16, false, true).unwrap();
    assert_eq!(first.blobs_embedded, 2);
    assert_eq!(first.blobs_skipped, 0);

    let mut store = Store::open(repo.path(), "test", 2).unwrap();
    let second = index_repo(repo.path(), &mut store, &embed, 16, false, true).unwrap();
    assert_eq!(second.blobs_already_cached, 2);
    assert_eq!(second.blobs_embedded, 0);
}

#[test]
fn clean_filtered_files_use_the_index_sha() {
    let _env = ChunkGroupEnv::set("512");
    let repo = repo();
    write(repo.path(), ".gitattributes", "*.txt text\n");
    fs::write(repo.path().join("filtered.txt"), b"first\r\nsecond\r\n").unwrap();
    commit(
        repo.path(),
        "initial",
        &[".gitattributes", "filtered.txt"],
    );
    assert!(git(repo.path(), &["diff", "--name-only"]).is_empty());
    let index_sha = git(repo.path(), &["rev-parse", "HEAD:filtered.txt"]);
    let raw_sha = git(repo.path(), &["hash-object", "--no-filters", "filtered.txt"]);
    assert_ne!(index_sha, raw_sha);

    let embed = FakeEmbed::succeeds();
    let mut store = Store::open(repo.path(), "test", 2).unwrap();
    let first = index_repo(repo.path(), &mut store, &embed, 16, false, true).unwrap();
    assert_eq!(first.blobs_embedded, 1);
    assert_eq!(first.blobs_skipped, 0);

    let mut store = Store::open(repo.path(), "test", 2).unwrap();
    let second = index_repo(repo.path(), &mut store, &embed, 16, false, true).unwrap();
    assert_eq!(second.blobs_already_cached, 1);
    assert_eq!(second.blobs_embedded, 0);
}

#[test]
fn concurrent_commits_preserve_both_payloads() {
    let repo = repo();
    let barrier = std::sync::Arc::new(Barrier::new(3));
    let handles: Vec<_> = [
        "1111111111111111111111111111111111111111",
        "2222222222222222222222222222222222222222",
    ]
    .into_iter()
    .map(|sha| {
        let root = repo.path().to_path_buf();
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            let mut store = Store::open(&root, "test", 2).unwrap();
            store.insert_chunks(sha, &[(1, 1)], &[1.0, 0.0], 2);
            barrier.wait();
            store.commit().unwrap();
        })
    })
    .collect();
    barrier.wait();
    for handle in handles {
        handle.join().unwrap();
    }

    let store = Store::open(repo.path(), "test", 2).unwrap();
    assert_eq!(store.known_blob_shas().unwrap().len(), 2);
}

#[test]
fn corrupt_payload_is_removed_and_reembedded() {
    let _env = ChunkGroupEnv::set("512");
    let repo = repo();
    write(repo.path(), "a.txt", "alpha authentication token\n");
    write(repo.path(), "b.txt", "beta virtual machine provisioning\n");
    commit(repo.path(), "initial", &["a.txt", "b.txt"]);

    let embed = FakeEmbed::succeeds();
    let mut store = Store::open(repo.path(), "test", 2).unwrap();
    index_repo(repo.path(), &mut store, &embed, 16, false, true).unwrap();
    let a_sha = git(repo.path(), &["rev-parse", "HEAD:a.txt"]);
    replace_note(
        repo.path(),
        "refs/notes/vector-grep/test",
        &a_sha,
        b"not a vector payload",
    );

    let mut store = Store::open(repo.path(), "test", 2).unwrap();
    let index = Index::load(repo.path(), &mut store).unwrap();
    assert_eq!(index.len(), 1);
    assert_eq!(store.known_blob_shas().unwrap().len(), 1);

    let stats = index_repo(repo.path(), &mut store, &embed, 16, false, true).unwrap();
    assert_eq!(stats.blobs_embedded, 1);
    assert_eq!(store.known_blob_shas().unwrap().len(), 2);
}
