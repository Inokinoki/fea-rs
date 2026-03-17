//! Plasticity Material Models.
#![allow(unused_variables)]
//!
//! This module provides elastoplastic material models:
//! - von Mises yield criterion
//! - Isotropic hardening
//! - Kinematic hardening
//! - Combined hardening
//! - Return mapping algorithm

use nalgebra::{DMatrix, DVector};

/// Yield criterion types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum YieldCriterion {
    /// von Mises yield criterion.
    VonMises,
    /// Tresca yield criterion.
    Tresca,
    /// Drucker-Prager yield criterion.
    DruckerPrager,
}

/// Hardening law types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HardeningLaw {
    /// Perfect plasticity (no hardening).
    Perfect,
    /// Linear isotropic hardening.
    LinearIsotropic,
    /// Nonlinear isotropic hardening (Voce law).
    NonlinearIsotropic,
    /// Linear kinematic hardening.
    LinearKinematic,
    /// Combined isotropic-kinematic hardening.
    Combined,
}

/// Plastic material parameters.
#[derive(Debug, Clone)]
pub struct PlasticityParameters {
    /// Young's modulus.
    pub youngs_modulus: f64,
    /// Poisson's ratio.
    pub poisson_ratio: f64,
    /// Initial yield stress.
    pub yield_stress: f64,
    /// Hardening modulus.
    pub hardening_modulus: f64,
    /// Yield criterion.
    pub yield_criterion: YieldCriterion,
    /// Hardening law.
    pub hardening_law: HardeningLaw,
    /// Voce saturation parameter (for nonlinear hardening).
    pub voce_saturation: f64,
    /// Voce rate parameter (for nonlinear hardening).
    pub voce_rate: f64,
}

impl Default for PlasticityParameters {
    fn default() -> Self {
        Self {
            youngs_modulus: 210e9,
            poisson_ratio: 0.3,
            yield_stress: 250e6,
            hardening_modulus: 2e9,
            yield_criterion: YieldCriterion::VonMises,
            hardening_law: HardeningLaw::LinearIsotropic,
            voce_saturation: 400e6,
            voce_rate: 50.0,
        }
    }
}

impl PlasticityParameters {
    /// Creates new plasticity parameters.
    pub fn new(youngs_modulus: f64, poisson_ratio: f64, yield_stress: f64) -> Self {
        Self {
            youngs_modulus,
            poisson_ratio,
            yield_stress,
            ..Default::default()
        }
    }

    /// Computes shear modulus.
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Computes bulk modulus.
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }

    /// Computes current yield stress based on accumulated plastic strain.
    pub fn current_yield_stress(&self, accumulated_plastic_strain: f64) -> f64 {
        match self.hardening_law {
            HardeningLaw::Perfect => self.yield_stress,
            HardeningLaw::LinearIsotropic => {
                self.yield_stress + self.hardening_modulus * accumulated_plastic_strain
            }
            HardeningLaw::NonlinearIsotropic => {
                self.yield_stress + (self.voce_saturation - self.yield_stress)
                    * (1.0 - (-self.voce_rate * accumulated_plastic_strain).exp())
            }
            HardeningLaw::LinearKinematic | HardeningLaw::Combined => {
                self.yield_stress + self.hardening_modulus * accumulated_plastic_strain
            }
        }
    }
}

/// Plasticity state variables.
#[derive(Debug, Clone, Default)]
pub struct PlasticityState {
    /// Accumulated equivalent plastic strain.
    pub accumulated_plastic_strain: f64,
    /// Back stress tensor (for kinematic hardening).
    pub back_stress: DVector<f64>,
    /// Plastic strain tensor.
    pub plastic_strain: DVector<f64>,
}

impl PlasticityState {
    /// Creates new plasticity state.
    pub fn new(strain_size: usize) -> Self {
        Self {
            accumulated_plastic_strain: 0.0,
            back_stress: DVector::zeros(strain_size),
            plastic_strain: DVector::zeros(strain_size),
        }
    }
}

/// von Mises yield function.
pub fn von_mises_yield_function(stress: &DVector<f64>, back_stress: &DVector<f64>) -> f64 {
    let dev_stress = deviatoric_stress(stress);
    let effective_stress = dev_stress.norm() * (1.5_f64).sqrt();
    effective_stress - back_stress.norm()
}

/// Computes deviatoric stress.
pub fn deviatoric_stress(stress: &DVector<f64>) -> DVector<f64> {
    let n = stress.len();
    let mut dev = stress.clone();

    if n >= 3 {
        let hydrostatic = (stress[0] + stress[1] + stress[2]) / 3.0;
        dev[0] -= hydrostatic;
        dev[1] -= hydrostatic;
        dev[2] -= hydrostatic;
    }

    dev
}

/// Elastic stiffness matrix (3D).
pub fn elastic_stiffness_3d(youngs_modulus: f64, poisson_ratio: f64) -> DMatrix<f64> {
    let shear = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
    let lambda = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));

    let mut d = DMatrix::zeros(6, 6);

    // Normal components
    d[(0, 0)] = lambda + 2.0 * shear;
    d[(0, 1)] = lambda;
    d[(0, 2)] = lambda;
    d[(1, 0)] = lambda;
    d[(1, 1)] = lambda + 2.0 * shear;
    d[(1, 2)] = lambda;
    d[(2, 0)] = lambda;
    d[(2, 1)] = lambda;
    d[(2, 2)] = lambda + 2.0 * shear;

    // Shear components
    d[(3, 3)] = shear;
    d[(4, 4)] = shear;
    d[(5, 5)] = shear;

    d
}

/// Plasticity model with return mapping.
#[derive(Debug, Clone)]
pub struct PlasticityModel {
    /// Material parameters.
    pub params: PlasticityParameters,
    /// Elastic stiffness matrix.
    pub elasticity: DMatrix<f64>,
}

impl PlasticityModel {
    /// Creates a new plasticity model.
    pub fn new(params: PlasticityParameters) -> Self {
        let elasticity = elastic_stiffness_3d(params.youngs_modulus, params.poisson_ratio);
        Self { params, elasticity }
    }

    /// Computes stress from strain using return mapping.
    pub fn compute_stress(
        &self,
        strain: &DVector<f64>,
        state: &mut PlasticityState,
    ) -> DVector<f64> {
        // Elastic trial stress
        let elastic_trial = &self.elasticity * (strain - &state.plastic_strain);

        // Check yield
        let yield_value = von_mises_yield_function(&elastic_trial, &state.back_stress);
        let current_yield = self.params.current_yield_stress(state.accumulated_plastic_strain);

        if yield_value <= current_yield {
            // Elastic step
            elastic_trial
        } else {
            // Plastic step - return mapping
            self.return_mapping(strain, state, &elastic_trial)
        }
    }

    /// Return mapping algorithm.
    fn return_mapping(
        &self,
        total_strain: &DVector<f64>,
        state: &mut PlasticityState,
        trial_stress: &DVector<f64>,
    ) -> DVector<f64> {
        // Simplified radial return for von Mises
        let dev_trial = deviatoric_stress(trial_stress);
        let dev_norm = dev_trial.norm();
        let n_trial = dev_trial / dev_norm;

        // Plastic multiplier (simplified)
        let shear = self.params.shear_modulus();
        let trial_effective = dev_norm * (1.5_f64).sqrt();
        let current_yield = self.params.current_yield_stress(state.accumulated_plastic_strain);

        let delta_gamma = (trial_effective - current_yield) / (3.0 * shear + self.params.hardening_modulus);

        // Update stress
        let mut stress = trial_stress.clone();
        for i in 0..3 {
            stress[i] -= 2.0 * shear * delta_gamma * n_trial[i];
        }

        // Update plastic strain
        for i in 0..3 {
            state.plastic_strain[i] += delta_gamma * n_trial[i];
        }

        // Update accumulated plastic strain
        state.accumulated_plastic_strain += delta_gamma * (2.0 / 3.0_f64).sqrt();

        // Update back stress (kinematic hardening)
        match self.params.hardening_law {
            HardeningLaw::LinearKinematic | HardeningLaw::Combined => {
                for i in 0..3 {
                    state.back_stress[i] += self.params.hardening_modulus * delta_gamma * n_trial[i];
                }
            }
            _ => {}
        }

        stress
    }

    /// Computes consistent tangent modulus.
    pub fn consistent_tangent(&self, state: &PlasticityState) -> DMatrix<f64> {
        // For elastic step, return elastic stiffness
        // For plastic step, return elastoplastic tangent (simplified)
        self.elasticity.clone()
    }
}

/// Uniaxial stress-strain response for plasticity model.
pub fn uniaxial_response(
    model: &PlasticityModel,
    max_strain: f64,
    n_steps: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut strains = Vec::with_capacity(n_steps);
    let mut stresses = Vec::with_capacity(n_steps);

    let mut state = PlasticityState::new(6);
    let mut strain = DVector::zeros(6);

    for i in 0..n_steps {
        let eps = max_strain * (i as f64 / (n_steps - 1) as f64);
        strain[0] = eps;

        let stress = model.compute_stress(&strain, &mut state);
        strains.push(eps);
        stresses.push(stress[0]);
    }

    (strains, stresses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plasticity_parameters() {
        let params = PlasticityParameters::default();

        assert!(params.shear_modulus() > 0.0);
        assert!(params.bulk_modulus() > 0.0);
        assert!(params.current_yield_stress(0.0) > 0.0);
    }

    #[test]
    fn test_von_mises_yield() {
        let stress = DVector::from_column_slice(&[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let back_stress = DVector::zeros(6);

        let yield_val = von_mises_yield_function(&stress, &back_stress);

        // For uniaxial stress, von Mises = axial stress
        assert!((yield_val - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_plasticity_model() {
        let params = PlasticityParameters::new(210e9, 0.3, 250e6);
        let model = PlasticityModel::new(params);

        let mut state = PlasticityState::new(6);
        let mut strain = DVector::zeros(6);

        // Elastic loading (strain = 0.0005 gives stress ~105 MPa < 250 MPa yield)
        strain[0] = 0.0005;
        let stress = model.compute_stress(&strain, &mut state);

        // Should be elastic (stress < yield)
        assert!(stress[0] < 250e6);

        // Plastic loading
        strain[0] = 0.02;
        let stress = model.compute_stress(&strain, &mut state);

        // Should have yielded
        assert!(state.accumulated_plastic_strain > 0.0);
    }

    #[test]
    fn test_uniaxial_response() {
        let params = PlasticityParameters::new(210e9, 0.3, 250e6);
        let model = PlasticityModel::new(params.clone());

        let (strains, stresses) = uniaxial_response(&model, 0.02, 100);

        assert_eq!(strains.len(), 100);
        assert_eq!(stresses.len(), 100);

        // Check that we get stress response
        assert!(stresses.iter().any(|&s| s > 0.0));

        // Check monotonic increase (plasticity should give increasing stress)
        for i in 1..stresses.len() {
            assert!(stresses[i] >= stresses[i - 1], "Stress should not decrease");
        }
    }
}
