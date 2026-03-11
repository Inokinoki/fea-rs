//! Utility modules for FEA analysis.
//!
//! This module provides:
//! - Sparse matrix data structures
//! - Post-processing utilities
//! - Visualization export

pub mod sparse;
pub mod postprocessing;
pub mod visualization;

pub use sparse::{CsrMatrix, SparseConjugateGradient};
pub use postprocessing::*;
pub use visualization::*;
