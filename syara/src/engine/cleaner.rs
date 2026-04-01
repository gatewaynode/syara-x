use unicode_normalization::UnicodeNormalization;

/// Trait for text cleaning before matching.
pub trait TextCleaner: Send + Sync {
    fn clean(&self, text: &str) -> String;
    fn name(&self) -> &str;
}

/// NFKC normalization + lowercase + whitespace collapsing.
/// Mirrors Python's `DefaultCleaner` exactly.
pub struct DefaultCleaner;

impl TextCleaner for DefaultCleaner {
    fn clean(&self, text: &str) -> String {
        let normalized: String = text.nfkc().collect();
        normalized
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn name(&self) -> &str {
        "default_cleaning"
    }
}

/// Returns text unchanged.
pub struct NoOpCleaner;

impl TextCleaner for NoOpCleaner {
    fn clean(&self, text: &str) -> String {
        text.to_owned()
    }

    fn name(&self) -> &str {
        "no_op"
    }
}

/// NFKC + lowercase + strip digits + strip punctuation + collapse whitespace.
/// Mirrors Python's `AggressiveCleaner`.
pub struct AggressiveCleaner;

impl TextCleaner for AggressiveCleaner {
    fn clean(&self, text: &str) -> String {
        let normalized: String = text.nfkc().collect();
        let filtered: String = normalized
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c.is_whitespace() {
                    c
                } else {
                    ' '
                }
            })
            .collect();
        // Remove digit sequences (matches Python `re.sub(r'\d+', '', ...)`)
        let no_digits: String = filtered
            .chars()
            .map(|c| if c.is_ascii_digit() { ' ' } else { c })
            .collect();
        no_digits
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn name(&self) -> &str {
        "aggressive_cleaning"
    }
}
