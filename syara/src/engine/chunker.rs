/// Trait for splitting text into chunks for matching.
pub trait Chunker: Send + Sync {
    fn chunk(&self, text: &str) -> Vec<String>;
    fn name(&self) -> &str;
}

/// Returns the entire text as one chunk.
pub struct NoChunker;

impl Chunker for NoChunker {
    fn chunk(&self, text: &str) -> Vec<String> {
        vec![text.to_owned()]
    }

    fn name(&self) -> &str {
        "no_chunking"
    }
}

/// Splits on sentence-ending punctuation (. ! ?).
pub struct SentenceChunker;

impl Chunker for SentenceChunker {
    fn chunk(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);
            if matches!(ch, '.' | '!' | '?') {
                let trimmed = current.trim().to_owned();
                if !trimmed.is_empty() {
                    chunks.push(trimmed);
                }
                current.clear();
            }
        }

        let remainder = current.trim().to_owned();
        if !remainder.is_empty() {
            chunks.push(remainder);
        }

        if chunks.is_empty() {
            chunks.push(text.to_owned());
        }

        chunks
    }

    fn name(&self) -> &str {
        "sentence_chunking"
    }
}

/// Splits into fixed-size word windows with optional overlap.
pub struct FixedSizeChunker {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl FixedSizeChunker {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self { chunk_size, overlap }
    }
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return vec![text.to_owned()];
        }

        let step = if self.chunk_size > self.overlap {
            self.chunk_size - self.overlap
        } else {
            1
        };

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < words.len() {
            let end = (start + self.chunk_size).min(words.len());
            chunks.push(words[start..end].join(" "));
            start += step;
        }

        chunks
    }

    fn name(&self) -> &str {
        "fixed_size_chunking"
    }
}

/// Splits on blank lines (paragraph boundaries).
pub struct ParagraphChunker;

impl Chunker for ParagraphChunker {
    fn chunk(&self, text: &str) -> Vec<String> {
        let chunks: Vec<String> = text
            .split("\n\n")
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();

        if chunks.is_empty() {
            vec![text.to_owned()]
        } else {
            chunks
        }
    }

    fn name(&self) -> &str {
        "paragraph_chunking"
    }
}

/// Splits into fixed-size word windows (no overlap).
pub struct WordChunker {
    pub chunk_size: usize,
}

impl WordChunker {
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }
}

impl Chunker for WordChunker {
    fn chunk(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return vec![text.to_owned()];
        }

        words
            .chunks(self.chunk_size)
            .map(|w| w.join(" "))
            .collect()
    }

    fn name(&self) -> &str {
        "word_chunking"
    }
}
