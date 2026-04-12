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
        // BUG-019: use `is_numeric()` to match Python `re.sub(r'\d+', '', ...)`
        // which strips Unicode digits (Arabic-Indic, Devanagari, etc.), not just ASCII.
        let no_digits: String = filtered
            .chars()
            .map(|c| if c.is_numeric() { ' ' } else { c })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggressive_cleaner_strips_unicode_digits() {
        // BUG-019: Arabic-Indic numerals (٠-٩) must be stripped like ASCII digits.
        let cleaner = AggressiveCleaner;
        let input = "hello ١٢٣ world";
        let cleaned = cleaner.clean(input);
        assert_eq!(cleaned, "hello world");
    }

    #[test]
    fn aggressive_cleaner_strips_ascii_digits() {
        let cleaner = AggressiveCleaner;
        let input = "test 123 value";
        let cleaned = cleaner.clean(input);
        assert_eq!(cleaned, "test value");
    }

    #[test]
    fn aggressive_cleaner_strips_devanagari_digits() {
        // BUG-019: Devanagari digits (०-९) are Unicode numeric.
        let cleaner = AggressiveCleaner;
        let input = "data ०१२ end";
        let cleaned = cleaner.clean(input);
        assert_eq!(cleaned, "data end");
    }
}
