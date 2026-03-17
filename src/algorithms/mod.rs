//! Algorithms for finite element analysis.
#![allow(unused_variables)]
//!
//! This module provides:
//! - Solver traits and implementations
//! - Analysis type traits and implementations
//! - Nonlinear solution methods
//! - Thermal stress analysis

pub mod solvers;
pub mod analysis;
pub mod nonlinear;
pub mod thermal;
pub mod contact;
pub mod contact_advanced;
pub mod explicit_dynamics;
pub mod amr;
pub mod contact_enhanced;
pub mod dynamics_accel;
pub mod coupling;

pub use solvers::{Solver, SolverResult};
pub use analysis::{Analysis, AnalysisResult};
pub use thermal::{ThermalProperties, TemperatureField, ThermalAnalysis, ThermalStressResult};
pub use explicit_dynamics::{ExplicitDynamicAnalyzer, ExplicitConfig, ExplicitMethod, ExplicitDynamicResult};
