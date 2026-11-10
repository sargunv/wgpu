//! The pipeline cache: the shader compiler's output, kept between runs.
//!
//! D3D12 drivers keep their own caches of what they make of DXIL, but
//! nothing keeps the compiler's work: each process compiles every
//! pipeline's HLSL with DXC (or FXC) again, which can cost more than the
//! rest of a cold start. A cache holds bytecode by a SHA-256 of what made
//! it (the compiler, its arguments and the HLSL), and its data is those
//! entries, for the application to keep as it keeps Vulkan's.

use alloc::{sync::Arc, vec::Vec};

use hashbrown::HashMap;
use parking_lot::Mutex;
use windows::Win32::Security::Cryptography::{BCryptHash, BCRYPT_SHA256_ALG_HANDLE};

/// A SHA-256 of what made an entry's bytecode.
pub(super) type Key = [u8; 32];

/// What the data starts with, and its layout's version.
const MAGIC: [u8; 4] = *b"WDXC";
const VERSION: u32 = 1;

/// The most bytecode the data keeps: entries this run used come first,
/// so a cache that outgrows it drops entries nothing has used since it
/// was loaded (an older build's shaders).
const MAX_BYTES: usize = 64 << 20;

#[derive(Debug, Default)]
pub struct PipelineCache {
    entries: Mutex<HashMap<Key, Entry>>,
}

impl crate::DynPipelineCache for PipelineCache {}

#[derive(Debug)]
struct Entry {
    bytecode: Arc<[u8]>,
    /// Looked up or added since the cache was made.
    used: bool,
}

impl PipelineCache {
    /// A cache holding `data`'s entries, or none if it doesn't parse.
    pub(super) fn new(data: Option<&[u8]>) -> Self {
        Self {
            entries: Mutex::new(data.and_then(parse).unwrap_or_default()),
        }
    }

    pub(super) fn get(&self, key: &Key) -> Option<Arc<[u8]>> {
        let mut entries = self.entries.lock();
        let entry = entries.get_mut(key)?;
        entry.used = true;
        Some(entry.bytecode.clone())
    }

    /// Keeps `bytecode` if its container is signed: DXIL that DXC's
    /// validator didn't sign loads only on a device in developer mode,
    /// where compiling it again costs nothing that matters.
    pub(super) fn insert(&self, key: Key, bytecode: &[u8]) {
        if !signed(bytecode) {
            return;
        }
        self.entries.lock().insert(
            key,
            Entry {
                bytecode: bytecode.into(),
                used: true,
            },
        );
    }

    /// The entries, in key order: the same bytes for the same entries,
    /// whichever a run used, so an application can tell nothing changed.
    pub(super) fn data(&self) -> Vec<u8> {
        let entries = self.entries.lock();
        let mut kept: Vec<(&Key, &Entry)> = entries.iter().collect();
        kept.sort_by_key(|&(key, entry)| (!entry.used, *key));
        let mut bytes = 0;
        let fit = kept
            .iter()
            .take_while(|(_, entry)| {
                bytes += entry.bytecode.len();
                bytes <= MAX_BYTES
            })
            .count();
        kept.truncate(fit);
        kept.sort_by_key(|&(key, _)| *key);
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(kept.len() as u32).to_le_bytes());
        for (key, entry) in kept {
            out.extend_from_slice(key);
            out.extend_from_slice(&(entry.bytecode.len() as u32).to_le_bytes());
            out.extend_from_slice(&entry.bytecode);
        }
        out
    }
}

/// The SHA-256 of `parts` (a compile's compiler, arguments and source),
/// each after its length, so no two lists of them hash alike by running
/// together.
pub(super) fn key<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> Option<Key> {
    let mut input = Vec::new();
    for part in parts {
        input.extend_from_slice(&(part.len() as u64).to_le_bytes());
        input.extend_from_slice(part);
    }
    let mut out = [0; 32];
    // SAFETY: SHA-256's pseudo-handle, which needs no opening or closing
    // (Windows 10 and later); it reads `input` and writes all of `out`.
    let status = unsafe { BCryptHash(BCRYPT_SHA256_ALG_HANDLE, None, &input, &mut out) };
    status.is_ok().then_some(out)
}

/// Whether `bytecode` is a DXIL or DXBC container with a digest: both
/// start with `DXBC` and the 16-byte hash the validator (or FXC) signs.
fn signed(bytecode: &[u8]) -> bool {
    bytecode.starts_with(b"DXBC")
        && bytecode
            .get(4..20)
            .is_some_and(|d| d.iter().any(|&b| b != 0))
}

fn parse(data: &[u8]) -> Option<HashMap<Key, Entry>> {
    fn take<'d>(r: &mut &'d [u8], n: usize) -> Option<&'d [u8]> {
        let (head, rest) = r.split_at_checked(n)?;
        *r = rest;
        Some(head)
    }
    fn u32_le(r: &mut &[u8]) -> Option<u32> {
        Some(u32::from_le_bytes(take(r, 4)?.try_into().ok()?))
    }
    let mut r = data;
    if take(&mut r, 4)? != MAGIC || u32_le(&mut r)? != VERSION {
        return None;
    }
    let count = u32_le(&mut r)?;
    let mut entries = HashMap::new();
    for _ in 0..count {
        let key: Key = take(&mut r, 32)?.try_into().ok()?;
        let len = u32_le(&mut r)? as usize;
        let bytecode = take(&mut r, len)?;
        if !signed(bytecode) {
            return None;
        }
        entries.insert(
            key,
            Entry {
                bytecode: bytecode.into(),
                used: false,
            },
        );
    }
    r.is_empty().then_some(entries)
}
