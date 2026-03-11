//! Algorithms for finite element analysis.
//!
//! This module provides:
//! - Solver traits and implementations
//! - Analysis type traits and implementations
//! - Nonlinear solution methods

pub mod solvers;
pub mod analysis;
pub mod nonlinear;

pub use solvers::{Solver, SolverResult};
pub use analysis::{Analysis, AnalysisResult};
