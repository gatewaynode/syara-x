use std::collections::HashMap;
use sha2::{Digest, Sha256};

/// Session-scoped cache for cleaned text, keyed by sha256(cleaner_name + ":" + text).
/// Prevents redundant cleaning when multiple rules share a cleaner.
/// Cleared after each scan call.
pub struct TextCache {
    store: HashMap<String, String>,
}

impl TextCache {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    fn cache_key(cleaner_name: &str, text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(cleaner_name.as_bytes());
        hasher.update(b":");
        hasher.update(text.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Returns cached cleaned text if present.
    pub fn get(&self, text: &str, cleaner_name: &str) -> Option<&str> {
        let key = Self::cache_key(cleaner_name, text);
        self.store.get(&key).map(|s| s.as_str())
    }

    /// Store a cleaned result.
    pub fn insert(&mut self, text: &str, cleaner_name: &str, cleaned: String) {
        let key = Self::cache_key(cleaner_name, text);
        self.store.insert(key, cleaned);
    }

    /// Returns the number of cached entries.
    pub fn size(&self) -> usize {
        self.store.len()
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.store.clear();
    }
}

impl Default for TextCache {
    fn default() -> Self {
        Self::new()
    }
}
