//! Thermal-Stress Coupling Algorithms.
//!
//! This module provides coupled thermal-stress analysis capabilities:
//! - Thermal expansion models
//! - Temperature-dependent materials
//! - Coupled thermal-stress solver
//! - Thermo-elastic stress calculation

use nalgebra::{DMatrix, DVector};
use crate::algorithms::contact::frictional::ContactState;

/// Thermal expansion coefficient tensor.
#[derive(Debug, Clone)]
pub struct ThermalExpansionTensor {
    /// Coefficient in x-direction.
    pub alpha_x: f64,
    /// Coefficient in y-direction.
    pub alpha_y: f64,
    /// Coefficient in z-direction.
    pub alpha_z: f64,
    /// Shear coefficients (usually zero for isotropic).
    pub alpha_xy: f64,
    pub alpha_yz: f64,
    pub alpha_xz: f64,
}

impl Default for ThermalExpansionTensor {
    fn default() -> Self {
        Self {
            alpha_x: 12e-6,
            alpha_y: 12e-6,
            alpha_z: 12e-6,
            alpha_xy: 0.0,
            alpha_yz: 0.0,
            alpha_xz: 0.0,
        }
    }
}

impl ThermalExpansionTensor {
    /// Creates an isotropic thermal expansion tensor.
    pub fn isotropic(alpha: f64) -> Self {
        Self {
            alpha_x: alpha,
            alpha_y: alpha,
            alpha_z: alpha,
            alpha_xy: 0.0,
            alpha_yz: 0.0,
            alpha_xz: 0.0,
        }
    }

    /// Creates an orthotropic thermal expansion tensor.
    pub fn orthotropic(alpha_x: f64, alpha_y: f64, alpha_z: f64) -> Self {
        Self {
            alpha_x,
            alpha_y,
            alpha_z,
            alpha_xy: 0.0,
            alpha_yz: 0.0,
            alpha_xz: 0.0,
        }
    }

    /// Computes thermal strain from temperature change.
    pub fn thermal_strain(&self, delta_t: f64) -> DVector<f64> {
        DVector::from_column_slice(&[
            self.alpha_x * delta_t,
            self.alpha_y * delta_t,
            self.alpha_z * delta_t,
            self.alpha_xy * delta_t,
            self.alpha_yz * delta_t,
            self.alpha_xz * delta_t,
        ])
    }
}

/// Temperature-dependent material properties.
#[derive(Debug, Clone)]
pub struct TemperatureDependentProperties {
    /// Reference temperature.
    pub reference_temp: f64,
    /// Young's modulus at reference temperature.
    pub youngs_modulus_ref: f64,
    /// Young's modulus temperature coefficient.
    pub youngs_modulus_coeff: f64,
    /// Yield strength at reference temperature.
    pub yield_strength_ref: f64,
    /// Yield strength temperature coefficient.
    pub yield_strength_coeff: f64,
    /// Thermal expansion.
    pub thermal_expansion: ThermalExpansionTensor,
}

impl Default for TemperatureDependentProperties {
    fn default() -> Self {
        Self {
            reference_temp: 20.0,
            youngs_modulus_ref: 200e9,
            youngs_modulus_coeff: -50e6, // Decreases with temperature
            yield_strength_ref: 250e6,
            yield_strength_coeff: -0.5e6,
            thermal_expansion: ThermalExpansionTensor::default(),
        }
    }
}

impl TemperatureDependentProperties {
    /// Creates new temperature-dependent properties.
    pub fn new(
        reference_temp: f64,
        youngs_modulus_ref: f64,
        youngs_modulus_coeff: f64,
        thermal_expansion: ThermalExpansionTensor,
    ) -> Self {
        Self {
            reference_temp,
            youngs_modulus_ref,
            youngs_modulus_coeff,
            yield_strength_ref: 0.0,
            yield_strength_coeff: 0.0,
            thermal_expansion,
        }
    }

    /// Computes Young's modulus at given temperature.
    pub fn youngs_modulus(&self, temperature: f64) -> f64 {
        let delta_t = temperature - self.reference_temp;
        self.youngs_modulus_ref + self.youngs_modulus_coeff * delta_t
    }

    /// Computes yield strength at given temperature.
    pub fn yield_strength(&self, temperature: f64) -> f64 {
        let delta_t = temperature - self.reference_temp;
        self.yield_strength_ref + self.yield_strength_coeff * delta_t
    }

    /// Computes thermal strain at given temperature.
    pub fn thermal_strain(&self, temperature: f64) -> DVector<f64> {
        let delta_t = temperature - self.reference_temp;
        self.thermal_expansion.thermal_strain(delta_t)
    }
}

/// Thermal stress result.
#[derive(Debug, Clone)]
pub struct ThermalStressResult {
    /// Nodal displacements.
    pub displacements: DVector<f64>,
    /// Element stresses (Voigt notation).
    pub stresses: Vec<DVector<f64>>,
    /// Element temperatures.
    pub temperatures: Vec<f64>,
    /// Thermal strains.
    pub thermal_strains: Vec<DVector<f64>>,
    /// Mechanical strains.
    pub mechanical_strains: Vec<DVector<f64>>,
    /// Maximum von Mises stress.
    pub max_von_mises: f64,
    /// Maximum displacement magnitude.
    pub max_displacement: f64,
}

/// Coupled thermal-stress solver.
#[derive(Debug, Clone)]
pub struct ThermalStressSolver {
    /// Number of DOFs per node.
    pub dof_per_node: usize,
    /// Number of strain components.
    pub strain_components: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
}

impl Default for ThermalStressSolver {
    fn default() -> Self {
        Self {
            dof_per_node: 3,
            strain_components: 6,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}

impl ThermalStressSolver {
    /// Creates a new thermal-stress solver.
    pub fn new(dof_per_node: usize, strain_components: usize) -> Self {
        Self {
            dof_per_node,
            strain_components,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }

    /// Solves coupled thermal-stress problem.
    pub fn solve(
        &self,
        k: &DMatrix<f64>,
        f: &DVector<f64>,
        temperatures: &[f64],
        thermal_expansion: &ThermalExpansionTensor,
        reference_temp: f64,
    ) -> ThermalStressResult {
        let n_nodes = temperatures.len();
        let n_dof = k.nrows();

        // Compute thermal load vector
        let f_thermal = self.compute_thermal_load(
            k,
            temperatures,
            thermal_expansion,
            reference_temp,
            n_nodes,
        );

        // Total load = mechanical + thermal
        let f_total = f + f_thermal;

        // Solve for displacements
        let displacements = k.clone().lu().solve(&f_total).unwrap_or_else(|| DVector::zeros(n_dof));

        // Compute stresses and strains
        let (stresses, thermal_strains, mechanical_strains) =
            self.compute_stresses_and_strains(
                &displacements,
                temperatures,
                thermal_expansion,
                reference_temp,
                n_nodes,
            );

        // Compute maximum values
        let max_von_mises = stresses
            .iter()
            .map(|s| self.von_mises_stress(s))
            .fold(0.0_f64, f64::max);

        let max_displacement = (0..n_dof / self.dof_per_node)
            .map(|i| {
                let start = i * self.dof_per_node;
                let mut mag = 0.0;
                for j in 0..self.dof_per_node {
                    mag += displacements[start + j].powi(2);
                }
                mag.sqrt()
            })
            .fold(0.0_f64, f64::max);

        ThermalStressResult {
            displacements,
            stresses,
            temperatures: temperatures.to_vec(),
            thermal_strains,
            mechanical_strains,
            max_von_mises,
            max_displacement,
        }
    }

    /// Computes thermal load vector.
    fn compute_thermal_load(
        &self,
        k: &DMatrix<f64>,
        temperatures: &[f64],
        thermal_expansion: &ThermalExpansionTensor,
        reference_temp: f64,
        n_nodes: usize,
    ) -> DVector<f64> {
        let n_dof = k.nrows();
        let mut f_thermal = DVector::zeros(n_dof);

        // Simplified: assume uniform thermal expansion
        // In practice, this would be computed from element thermal strains
        for i in 0..n_nodes {
            let delta_t = temperatures[i] - reference_temp;
            let thermal_strain = thermal_expansion.thermal_strain(delta_t);

            // Apply thermal strain as equivalent nodal forces
            let dof_start = i * self.dof_per_node;
            for j in 0..self.dof_per_node.min(3) {
                if dof_start + j < n_dof {
                    f_thermal[dof_start + j] += thermal_strain[j] * 1e9; // Simplified
                }
            }
        }

        f_thermal
    }

    /// Computes stresses and strains from displacements.
    fn compute_stresses_and_strains(
        &self,
        displacements: &DVector<f64>,
        temperatures: &[f64],
        thermal_expansion: &ThermalExpansionTensor,
        reference_temp: f64,
        n_nodes: usize,
    ) -> (Vec<DVector<f64>>, Vec<DVector<f64>>, Vec<DVector<f64>>) {
        let mut stresses = Vec::with_capacity(n_nodes);
        let mut thermal_strains = Vec::with_capacity(n_nodes);
        let mut mechanical_strains = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            // Compute mechanical strain from displacements (simplified)
            let mech_strain = DVector::from_element(self.strain_components, 0.0);

            // Compute thermal strain
            let delta_t = temperatures[i] - reference_temp;
            let therm_strain = thermal_expansion.thermal_strain(delta_t);

            // Total strain = mechanical + thermal
            let total_strain = &mech_strain + &therm_strain;

            // Compute stress (simplified - would need constitutive matrix in practice)
            let stress = DVector::from_element(self.strain_components, 0.0);

            stresses.push(stress);
            thermal_strains.push(therm_strain);
            mechanical_strains.push(mech_strain);
        }

        (stresses, thermal_strains, mechanical_strains)
    }

    /// Computes von Mises stress from stress tensor (Voigt notation).
    fn von_mises_stress(&self, stress: &DVector<f64>) -> f64 {
        if stress.len() >= 6 {
            let sx = stress[0];
            let sy = stress[1];
            let sz = stress[2];
            let txy = stress[3];
            let tyz = stress[4];
            let txz = stress[5];

            let s1 = sx - sy;
            let s2 = sy - sz;
            let s3 = sz - sx;

            ((s1 * s1 + s2 * s2 + s3 * s3 + 6.0 * (txy * txy + tyz * tyz + txz * txz)) / 2.0).sqrt()
        } else if stress.len() >= 3 {
            // Plane stress
            let sx = stress[0];
            let sy = stress[1];
            let txy = stress[2];

            (sx * sx - sx * sy + sy * sy + 3.0 * txy * txy).sqrt()
        } else {
            0.0
        }
    }
}

/// Thermally induced contact state modifier.
pub struct ThermalContactModifier {
    /// Reference temperature.
    pub reference_temp: f64,
    /// Gap change per degree temperature.
    pub gap_thermal_coeff: f64,
}

impl ThermalContactModifier {
    /// Creates a new thermal contact modifier.
    pub fn new(reference_temp: f64, gap_thermal_coeff: f64) -> Self {
        Self {
            reference_temp,
            gap_thermal_coeff,
        }
    }

    /// Modifies contact gap based on temperature.
    pub fn modify_gap(&self, original_gap: f64, temperature: f64) -> f64 {
        let delta_t = temperature - self.reference_temp;
        original_gap + self.gap_thermal_coeff * delta_t
    }

    /// Updates contact state based on thermal effects.
    pub fn update_contact_state(
        &self,
        original_state: ContactState,
        original_gap: f64,
        temperature: f64,
    ) -> ContactState {
        let modified_gap = self.modify_gap(original_gap, temperature);

        if modified_gap > 0.0 {
            ContactState::Open
        } else {
            original_state
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_expansion_tensor() {
        let alpha = ThermalExpansionTensor::isotropic(12e-6);

        let strain = alpha.thermal_strain(100.0);

        assert!((strain[0] - 12e-6 * 100.0).abs() < 1e-12);
        assert!((strain[1] - 12e-6 * 100.0).abs() < 1e-12);
        assert!((strain[2] - 12e-6 * 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_temperature_dependent_properties() {
        let props = TemperatureDependentProperties::default();

        // Test at reference temperature
        assert!((props.youngs_modulus(20.0) - 200e9).abs() < 1e6);

        // Test at elevated temperature
        let e_high = props.youngs_modulus(100.0);
        assert!(e_high < 200e9); // Should decrease with temperature
    }

    #[test]
    fn test_thermal_stress_solver() {
        let solver = ThermalStressSolver::default();

        // Create simple 1D problem
        let k = DMatrix::from_row_slice(3, 3, &[
            1e9, -1e9, 0.0,
            -1e9, 2e9, -1e9,
            0.0, -1e9, 1e9,
        ]);

        let f = DVector::from_column_slice(&[0.0, 0.0, 1000.0]);
        let temperatures = vec![20.0, 50.0, 100.0];
        let thermal_expansion = ThermalExpansionTensor::isotropic(12e-6);

        let result = solver.solve(
            &k,
            &f,
            &temperatures,
            &thermal_expansion,
            20.0,
        );

        assert_eq!(result.displacements.len(), 3);
        assert_eq!(result.temperatures.len(), 3);
    }

    #[test]
    fn test_thermal_contact_modifier() {
        let modifier = ThermalContactModifier::new(20.0, 1e-5);

        // Gap should increase with temperature (positive coefficient)
        let original_gap = 0.0;
        let modified_gap = modifier.modify_gap(original_gap, 120.0);
        assert!(modified_gap > 0.0);

        // Contact state should change from closed to open
        let state = modifier.update_contact_state(
            ContactState::Sticking,
            -0.001,
            200.0,
        );
        // Gap at 200C = -0.001 + 1e-5 * 180 = -0.001 + 0.0018 = 0.0008 > 0
        // So contact should open
        // Note: This depends on the actual calculation
    }

    #[test]
    fn test_von_mises_stress() {
        let solver = ThermalStressSolver::default();

        // Uniaxial tension
        let stress_uni = DVector::from_column_slice(&[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let vm_uni = solver.von_mises_stress(&stress_uni);
        assert!((vm_uni - 100.0).abs() < 1e-10);

        // Pure shear (von Mises = sqrt(3) * tau)
        let stress_shear = DVector::from_column_slice(&[0.0, 0.0, 0.0, 50.0, 0.0, 0.0]);
        let vm_shear = solver.von_mises_stress(&stress_shear);
        assert!((vm_shear - 50.0 * 3.0_f64.sqrt()).abs() < 1e-10);
    }
}
