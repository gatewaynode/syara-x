use std::collections::HashMap;

/// Session-scoped cache for cleaned text, keyed by `(cleaner_name, text)` tuple.
/// Prevents redundant cleaning when multiple rules share a cleaner.
/// Cleared after each scan call.
///
/// BUG-016: replaced SHA256+hex key with direct tuple key — the HashMap
/// already hashes internally, so the extra crypto hash was pure overhead.
pub struct TextCache {
    store: HashMap<(String, String), String>,
}

impl TextCache {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    /// Returns cached cleaned text if present.
    pub fn get(&self, text: &str, cleaner_name: &str) -> Option<&str> {
        let key = (cleaner_name.to_owned(), text.to_owned());
        self.store.get(&key).map(|s| s.as_str())
    }

    /// Store a cleaned result.
    pub fn insert(&mut self, text: &str, cleaner_name: &str, cleaned: String) {
        let key = (cleaner_name.to_owned(), text.to_owned());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_tuple_key_insert_and_get() {
        // BUG-016: verify tuple-keyed cache works correctly
        let mut cache = TextCache::new();
        cache.insert("hello world", "default_cleaning", "hello world".into());
        assert_eq!(cache.get("hello world", "default_cleaning"), Some("hello world"));
        assert_eq!(cache.get("hello world", "aggressive_cleaning"), None);
        assert_eq!(cache.get("other text", "default_cleaning"), None);
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn cache_clear_empties_store() {
        let mut cache = TextCache::new();
        cache.insert("a", "c1", "cleaned_a".into());
        cache.insert("b", "c2", "cleaned_b".into());
        assert_eq!(cache.size(), 2);
        cache.clear();
        assert_eq!(cache.size(), 0);
        assert_eq!(cache.get("a", "c1"), None);
    }
}
