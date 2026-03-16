//! Multiphysics Coupling Algorithms.
//!
//! This module provides coupled multiphysics capabilities:
//! - Thermal-stress coupling
//! - Thermo-mechanical analysis
//! - Temperature-dependent materials

pub mod thermal_stress;

pub use thermal_stress::{
    ThermalExpansionTensor, TemperatureDependentProperties,
    ThermalStressResult, ThermalStressSolver, ThermalContactModifier,
};
