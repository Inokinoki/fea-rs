//! Thermal stress analysis module.
#![allow(unused_variables)]
//!
//! This module provides:
//! - Thermal load computation from temperature fields
//! - Thermal stress analysis for truss elements
//! - Temperature-dependent material properties

use crate::core::{Dof, Load, Model};

/// Thermal expansion coefficient data.
#[derive(Debug, Clone, Copy)]
pub struct ThermalProperties {
    /// Coefficient of thermal expansion (1/K).
    pub alpha: f64,
    /// Reference temperature (K).
    pub reference_temp: f64,
}

impl Default for ThermalProperties {
    fn default() -> Self {
        Self {
            alpha: 12e-6, // Steel
            reference_temp: 293.0, // 20°C
        }
    }
}

impl ThermalProperties {
    /// Creates new thermal properties.
    pub fn new(alpha: f64, reference_temp: f64) -> Self {
        Self { alpha, reference_temp }
    }

    /// Steel thermal properties.
    pub fn steel() -> Self {
        Self {
            alpha: 12e-6,
            reference_temp: 293.0,
        }
    }

    /// Aluminum thermal properties.
    pub fn aluminum() -> Self {
        Self {
            alpha: 23e-6,
            reference_temp: 293.0,
        }
    }
}

/// Temperature field for thermal analysis.
#[derive(Debug, Clone)]
pub struct TemperatureField {
    /// Temperature at each node.
    pub node_temps: Vec<f64>,
}

impl TemperatureField {
    /// Creates a uniform temperature field.
    pub fn uniform(temp: f64, num_nodes: usize) -> Self {
        Self {
            node_temps: vec![temp; num_nodes],
        }
    }

    /// Creates a temperature field from nodal values.
    pub fn from_nodes(temps: Vec<f64>) -> Self {
        Self { node_temps: temps }
    }

    /// Creates a linear temperature gradient.
    pub fn linear_gradient(temp_start: f64, temp_end: f64, num_nodes: usize) -> Self {
        let mut temps = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let t = if num_nodes > 1 {
                temp_start + (temp_end - temp_start) * (i as f64 / (num_nodes - 1) as f64)
            } else {
                temp_start
            };
            temps.push(t);
        }
        Self { node_temps: temps }
    }
}

/// Thermal load computation for truss elements.
pub struct ThermalAnalysis;

impl ThermalAnalysis {
    /// Computes equivalent thermal loads for a truss model.
    ///
    /// For a truss element with temperature change ΔT:
    /// - Thermal strain: ε_th = α * ΔT
    /// - Equivalent nodal forces: F = EA * α * ΔT
    pub fn compute_thermal_loads<E>(
        model: &Model<E>,
        temp_field: &TemperatureField,
        thermal_props: ThermalProperties,
    ) -> Vec<Load>
    where
        E: crate::elements::Element,
    {
        let mut loads = Vec::new();

        use crate::core::{Material, Section};
        let default_mat = Material::new("default", 210e9, 0.3, 7850.0, 250e6);
        let default_sec = Section::new("default", 1e-4, 1e-8, 1e-8, 2e-8);

        for (elem_idx, elem) in model.elements.iter().enumerate() {
            let node_ids = elem.node_ids();
            if node_ids.len() != 2 {
                continue;
            }

            let n1 = node_ids[0];
            let n2 = node_ids[1];

            // Get temperatures at nodes
            let t1 = temp_field.node_temps.get(n1).copied().unwrap_or(thermal_props.reference_temp);
            let t2 = temp_field.node_temps.get(n2).copied().unwrap_or(thermal_props.reference_temp);

            // Average temperature change
            let dt_avg = ((t1 + t2) / 2.0) - thermal_props.reference_temp;

            // Get element properties
            let ctx = crate::elements::ElementContext::new(
                &model.nodes,
                model.materials.first().unwrap_or(&default_mat),
                model.sections.first().unwrap_or(&default_sec),
            );

            let e = ctx.material.young_modulus;
            let alpha = thermal_props.alpha;
            let area = ctx.section.area;

            // Thermal force: F_thermal = EA * alpha * ΔT
            let thermal_force = e * area * alpha * dt_avg;

            // Get element direction for projecting thermal force to global DOFs
            let n1_pos = model.nodes[n1].as_array();
            let n2_pos = model.nodes[n2].as_array();
            let dx = n2_pos[0] - n1_pos[0];
            let dy = n2_pos[1] - n1_pos[1];
            let dz = n2_pos[2] - n1_pos[2];
            let length = (dx * dx + dy * dy + dz * dz).sqrt();

            if length < 1e-15 {
                continue;
            }

            // Direction cosines
            let cx = dx / length;
            let cy = dy / length;
            let cz = dz / length;

            // Equivalent nodal forces (equal and opposite at the two nodes)
            // Node 1: force in -x direction (compressive when heated)
            // Node 2: force in +x direction
            let fx = thermal_force * cx;
            let fy = thermal_force * cy;
            let fz = thermal_force * cz;

            // Add equivalent nodal forces
            // At node 1: -F_thermal in axial direction
            loads.push(Load::new(n1, Dof::Ux, -fx));
            loads.push(Load::new(n1, Dof::Uy, -fy));
            loads.push(Load::new(n1, Dof::Uz, -fz));

            // At node 2: +F_thermal in axial direction
            loads.push(Load::new(n2, Dof::Ux, fx));
            loads.push(Load::new(n2, Dof::Uy, fy));
            loads.push(Load::new(n2, Dof::Uz, fz));

            let _ = elem_idx; // Keep for API compatibility
        }

        loads
    }

    /// Applies thermal loads to a model.
    pub fn apply_thermal_loads<E>(
        model: &mut Model<E>,
        temp_field: &TemperatureField,
        thermal_props: ThermalProperties,
    ) where
        E: crate::elements::Element,
    {
        let loads = Self::compute_thermal_loads(model, temp_field, thermal_props);
        for load in loads {
            model.add_load(load);
        }
    }

    /// Computes thermal stress in a truss element.
    ///
    /// σ_th = E * α * ΔT
    pub fn compute_thermal_stress(
        e: f64,
        thermal_props: ThermalProperties,
        temperature: f64,
    ) -> f64 {
        let dt = temperature - thermal_props.reference_temp;
        e * thermal_props.alpha * dt
    }

    /// Computes total stress (mechanical + thermal).
    pub fn compute_total_stress(
        mechanical_stress: f64,
        thermal_stress: f64,
    ) -> f64 {
        mechanical_stress + thermal_stress
    }
}

/// Result of thermal stress analysis.
#[derive(Debug, Clone)]
pub struct ThermalStressResult {
    /// Thermal stress at each element.
    pub thermal_stresses: Vec<f64>,
    /// Total stress (mechanical + thermal) at each element.
    pub total_stresses: Vec<f64>,
    /// Temperature at each node.
    pub temperatures: Vec<f64>,
    /// Maximum thermal stress.
    pub max_thermal_stress: f64,
    /// Maximum total stress.
    pub max_total_stress: f64,
}

impl ThermalStressResult {
    /// Creates a new thermal stress result.
    pub fn new(
        thermal_stresses: Vec<f64>,
        total_stresses: Vec<f64>,
        temperatures: Vec<f64>,
    ) -> Self {
        let max_thermal_stress = thermal_stresses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let max_total_stress = total_stresses.iter().map(|s| s.abs()).fold(f64::NEG_INFINITY, f64::max);

        Self {
            thermal_stresses,
            total_stresses,
            temperatures,
            max_thermal_stress,
            max_total_stress,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_properties_steel() {
        let props = ThermalProperties::steel();
        assert!((props.alpha - 12e-6).abs() < 1e-10);
        assert!((props.reference_temp - 293.0).abs() < 1e-10);
    }

    #[test]
    fn test_thermal_properties_aluminum() {
        let props = ThermalProperties::aluminum();
        assert!((props.alpha - 23e-6).abs() < 1e-10);
    }

    #[test]
    fn test_temperature_field_uniform() {
        let field = TemperatureField::uniform(100.0, 5);
        assert_eq!(field.node_temps.len(), 5);
        for &t in &field.node_temps {
            assert!((t - 100.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_temperature_field_gradient() {
        let field = TemperatureField::linear_gradient(20.0, 100.0, 5);
        assert_eq!(field.node_temps.len(), 5);
        assert!((field.node_temps[0] - 20.0).abs() < 1e-10);
        assert!((field.node_temps[4] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_thermal_stress_computation() {
        let e = 210e9;
        let props = ThermalProperties::steel();
        let temp = 100.0 + 293.0; // 100°C above reference

        let stress = ThermalAnalysis::compute_thermal_stress(e, props, temp);

        // σ = E * α * ΔT = 210e9 * 12e-6 * 100 = 252e6 Pa
        let expected = e * props.alpha * 100.0;
        assert!((stress - expected).abs() < 1e6);
    }

    #[test]
    fn test_total_stress() {
        let mech = 100e6;
        let thermal = 50e6;

        let total = ThermalAnalysis::compute_total_stress(mech, thermal);
        assert!((total - 150e6).abs() < 1e6);
    }

    #[test]
    fn test_thermal_stress_result() {
        let thermal = vec![10e6, 20e6, 30e6];
        let total = vec![15e6, 25e6, 35e6];
        let temps = vec![20.0, 50.0, 80.0];

        let result = ThermalStressResult::new(thermal, total, temps);

        assert!((result.max_thermal_stress - 30e6).abs() < 1e6);
        assert!((result.max_total_stress - 35e6).abs() < 1e6);
        assert_eq!(result.temperatures.len(), 3);
    }
}
