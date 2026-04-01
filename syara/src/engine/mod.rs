pub mod chunker;
pub mod cleaner;
pub mod string_matcher;

#[cfg(feature = "sbert")]
pub mod semantic_matcher;

#[cfg(feature = "classifier")]
pub mod classifier;

#[cfg(feature = "llm")]
pub mod llm_evaluator;

#[cfg(feature = "phash")]
pub mod phash_matcher;
