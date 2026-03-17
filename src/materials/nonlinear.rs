//! Material models for nonlinear FEA.
#![allow(unused_variables)]
//!
//! This module provides:
//! - Linear elastic material
//! - Neo-Hookean hyperelastic material
//! - Mooney-Rivlin hyperelastic material
//! - Von Mises plasticity with isotropic hardening
//! - Johnson-Cook viscoplasticity
//! - Damage mechanics

use nalgebra::Matrix6;

/// Stress-strain state.
#[derive(Debug, Clone, Default)]
pub struct StressState {
    /// Cauchy stress tensor (Voigt notation: σxx, σyy, σzz, σxy, σyz, σxz).
    pub stress: [f64; 6],
    /// Green-Lagrange strain tensor (Voigt notation).
    pub strain: [f64; 6],
    /// Equivalent plastic strain.
    pub plastic_strain: f64,
    /// Damage variable (0 = undamaged, 1 = failed).
    pub damage: f64,
}

/// Material tangent stiffness.
#[derive(Debug, Clone)]
pub struct MaterialTangent {
    /// Consistent tangent matrix (6x6).
    pub tangent: Matrix6<f64>,
    /// Algorithmic tangent for Newton iteration.
    pub algorithmic: Option<Matrix6<f64>>,
}

impl Default for MaterialTangent {
    fn default() -> Self {
        Self {
            tangent: Matrix6::zeros(),
            algorithmic: None,
        }
    }
}

/// Linear elastic isotropic material.
#[derive(Debug, Clone)]
pub struct LinearElastic {
    /// Young's modulus.
    pub e: f64,
    /// Poisson's ratio.
    pub nu: f64,
    /// Density.
    pub rho: f64,
    /// Thermal expansion coefficient.
    pub alpha: f64,
}

impl LinearElastic {
    /// Creates a new linear elastic material.
    pub fn new(e: f64, nu: f64, rho: f64) -> Self {
        Self { e, nu, rho, alpha: 0.0 }
    }

    /// Creates with thermal expansion.
    pub fn with_thermal(e: f64, nu: f64, rho: f64, alpha: f64) -> Self {
        Self { e, nu, rho, alpha }
    }

    /// Computes stress from strain.
    pub fn stress(&self, strain: &[f64; 6], delta_t: f64) -> [f64; 6] {
        let lambda = self.e * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu));
        let mu = self.e / (2.0 * (1.0 + self.nu));

        let trace = strain[0] + strain[1] + strain[2];
        let thermal = 3.0 * lambda * self.alpha * delta_t;

        let mut stress = [0.0; 6];
        for i in 0..3 {
            stress[i] = lambda * trace + 2.0 * mu * strain[i] - thermal;
        }
        stress[3] = 2.0 * mu * strain[3];
        stress[4] = 2.0 * mu * strain[4];
        stress[5] = 2.0 * mu * strain[5];

        stress
    }

    /// Computes consistent tangent.
    pub fn tangent(&self) -> Matrix6<f64> {
        let lambda = self.e * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu));
        let mu = self.e / (2.0 * (1.0 + self.nu));

        let mut c = Matrix6::zeros();
        for i in 0..3 {
            for j in 0..3 {
                c[(i, j)] = lambda + 2.0 * mu * (if i == j { 1.0 } else { 0.0 });
            }
        }
        c[(3, 3)] = mu;
        c[(4, 4)] = mu;
        c[(5, 5)] = mu;

        c
    }

    /// Computes bulk modulus.
    pub fn bulk_modulus(&self) -> f64 {
        self.e / (3.0 * (1.0 - 2.0 * self.nu))
    }

    /// Computes shear modulus.
    pub fn shear_modulus(&self) -> f64 {
        self.e / (2.0 * (1.0 + self.nu))
    }

    /// Computes wave speed.
    pub fn wave_speed(&self) -> f64 {
        let e_eff = self.e * (1.0 - self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu));
        (e_eff / self.rho).sqrt()
    }
}

/// Neo-Hookean hyperelastic material.
#[derive(Debug, Clone)]
pub struct NeoHookean {
    /// Shear modulus.
    pub mu: f64,
    /// Bulk modulus.
    pub kappa: f64,
}

impl NeoHookean {
    /// Creates a new Neo-Hookean material.
    pub fn new(mu: f64, kappa: f64) -> Self {
        Self { mu, kappa }
    }

    /// Creates from Young's modulus and Poisson's ratio.
    pub fn from_en(e: f64, nu: f64) -> Self {
        let mu = e / (2.0 * (1.0 + nu));
        let kappa = e / (3.0 * (1.0 - 2.0 * nu));
        Self { mu, kappa }
    }

    /// Computes strain energy density.
    pub fn strain_energy(&self, f: &Matrix6<f64>) -> f64 {
        // F is deformation gradient
        let j = f[(0, 0)] * f[(1, 1)] * f[(2, 2)]; // Simplified det(F)

        let i1 = f[(0, 0)].powi(2) + f[(1, 1)].powi(2) + f[(2, 2)].powi(2)
            + 2.0 * (f[(0, 1)].powi(2) + f[(0, 2)].powi(2) + f[(1, 2)].powi(2));

        self.mu / 2.0 * (i1 - 3.0) + self.kappa / 2.0 * (j - 1.0).powi(2)
    }

    /// Computes second Piola-Kirchhoff stress.
    pub fn pk2_stress(&self, c: &Matrix6<f64>) -> [f64; 6] {
        // C is right Cauchy-Green tensor
        let i1 = c[(0, 0)] + c[(1, 1)] + c[(2, 2)];
        let j = (c[(0, 0)] * c[(1, 1)] * c[(2, 2)]).sqrt();

        let mut s = [0.0; 6];
        s[0] = self.mu * (1.0 - 1.0 / c[(0, 0)]) + self.kappa * (j - 1.0) * j / c[(0, 0)];
        s[1] = self.mu * (1.0 - 1.0 / c[(1, 1)]) + self.kappa * (j - 1.0) * j / c[(1, 1)];
        s[2] = self.mu * (1.0 - 1.0 / c[(2, 2)]) + self.kappa * (j - 1.0) * j / c[(2, 2)];

        s
    }
}

/// Mooney-Rivlin hyperelastic material.
#[derive(Debug, Clone)]
pub struct MooneyRivlin {
    /// Material parameter C1.
    pub c1: f64,
    /// Material parameter C2.
    pub c2: f64,
    /// Bulk modulus.
    pub kappa: f64,
}

impl MooneyRivlin {
    /// Creates a new Mooney-Rivlin material.
    pub fn new(c1: f64, c2: f64, kappa: f64) -> Self {
        Self { c1, c2, kappa }
    }

    /// Computes strain energy density.
    pub fn strain_energy(&self, i1: f64, i2: f64, j: f64) -> f64 {
        self.c1 * (i1 - 3.0) + self.c2 * (i2 - 3.0) + self.kappa / 2.0 * (j - 1.0).powi(2)
    }
}

/// Von Mises plasticity with isotropic hardening.
#[derive(Debug, Clone)]
pub struct VonMisesPlasticity {
    /// Young's modulus.
    pub e: f64,
    /// Poisson's ratio.
    pub nu: f64,
    /// Initial yield stress.
    pub yield_stress: f64,
    /// Hardening modulus.
    pub hardening: f64,
}

impl VonMisesPlasticity {
    /// Creates a new von Mises plasticity material.
    pub fn new(e: f64, nu: f64, yield_stress: f64, hardening: f64) -> Self {
        Self { e, nu, yield_stress, hardening }
    }

    /// Computes updated stress and internal variables.
    pub fn update(&self, strain: &[f64; 6], state: &StressState) -> (StressState, MaterialTangent) {
        let mut new_state = state.clone();
        let mut tangent = MaterialTangent::default();

        let mu = self.e / (2.0 * (1.0 + self.nu));
        let lambda = self.e * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu));

        // Elastic trial stress
        let trace = strain[0] + strain[1] + strain[2];
        let mut trial_stress = [0.0; 6];

        for i in 0..3 {
            trial_stress[i] = lambda * trace + 2.0 * mu * strain[i];
        }
        trial_stress[3] = 2.0 * mu * strain[3];
        trial_stress[4] = 2.0 * mu * strain[4];
        trial_stress[5] = 2.0 * mu * strain[5];

        // Add existing stress
        for i in 0..6 {
            trial_stress[i] += state.stress[i];
        }

        // Compute von Mises equivalent stress
        let s = self.deviatoric(&trial_stress);
        let seq = self.mises_equivalent(&s);

        // Current yield stress with hardening
        let yield_current = self.yield_stress + self.hardening * state.plastic_strain;

        if seq > yield_current {
            // Plastic loading - radial return
            let gamma = (seq - yield_current) / (3.0 * mu + self.hardening);

            // Update stress
            for i in 0..6 {
                let n = if i < 3 {
                    s[i] / seq.max(1e-15)
                } else {
                    s[i] / seq.max(1e-15)
                };
                new_state.stress[i] = trial_stress[i] - 3.0 * mu * gamma * n;
            }

            // Update plastic strain
            new_state.plastic_strain += gamma * seq / yield_current.max(1e-15);

            // Consistent tangent (simplified)
            tangent = self.plastic_tangent(mu, lambda, yield_current, seq);
        } else {
            // Elastic
            new_state.stress = trial_stress;
            tangent.tangent = self.elastic_tangent(lambda, mu);
        }

        (new_state, tangent)
    }

    /// Computes deviatoric stress.
    fn deviatoric(&self, stress: &[f64; 6]) -> [f64; 6] {
        let p = (stress[0] + stress[1] + stress[2]) / 3.0;
        [
            stress[0] - p,
            stress[1] - p,
            stress[2] - p,
            stress[3],
            stress[4],
            stress[5],
        ]
    }

    /// Computes von Mises equivalent stress.
    fn mises_equivalent(&self, s: &[f64; 6]) -> f64 {
        let j2 = 0.5 * (s[0].powi(2) + s[1].powi(2) + s[2].powi(2))
            + s[3].powi(2) + s[4].powi(2) + s[5].powi(2);
        (3.0 * j2).sqrt()
    }

    /// Elastic tangent.
    fn elastic_tangent(&self, lambda: f64, mu: f64) -> Matrix6<f64> {
        let mut c = Matrix6::zeros();
        for i in 0..3 {
            for j in 0..3 {
                c[(i, j)] = lambda + 2.0 * mu * (if i == j { 1.0 } else { 0.0 });
            }
        }
        c[(3, 3)] = mu;
        c[(4, 4)] = mu;
        c[(5, 5)] = mu;
        c
    }

    /// Plastic tangent (simplified).
    fn plastic_tangent(&self, mu: f64, lambda: f64, yield_stress: f64, seq: f64) -> MaterialTangent {
        MaterialTangent {
            tangent: self.elastic_tangent(lambda, mu),
            algorithmic: None,
        }
    }
}

/// Johnson-Cook viscoplasticity material.
#[derive(Debug, Clone)]
pub struct JohnsonCook {
    /// Young's modulus.
    pub e: f64,
    /// Poisson's ratio.
    pub nu: f64,
    /// Reference yield stress.
    pub a: f64,
    /// Hardening modulus.
    pub b: f64,
    /// Strain rate sensitivity.
    pub c: f64,
    /// Strain hardening exponent.
    pub n: f64,
    /// Thermal softening exponent.
    pub m: f64,
    /// Reference strain rate.
    pub ref_strain_rate: f64,
    /// Melting temperature.
    pub melt_temp: f64,
    /// Reference temperature.
    pub ref_temp: f64,
}

impl JohnsonCook {
    /// Creates a new Johnson-Cook material.
    pub fn new(
        e: f64, nu: f64, a: f64, b: f64, c: f64, n: f64, m: f64,
        ref_strain_rate: f64, melt_temp: f64, ref_temp: f64,
    ) -> Self {
        Self {
            e, nu, a, b, c, n, m, ref_strain_rate, melt_temp, ref_temp,
        }
    }

    /// Computes flow stress.
    pub fn flow_stress(&self, plastic_strain: f64, strain_rate: f64, temperature: f64) -> f64 {
        let hardening = self.a + self.b * plastic_strain.powf(self.n);
        let rate_factor = 1.0 + self.c * (strain_rate / self.ref_strain_rate).ln().max(0.0);

        let t_star = ((temperature - self.ref_temp) / (self.melt_temp - self.ref_temp))
            .clamp(0.0, 1.0);
        let thermal_factor = 1.0 - t_star.powf(self.m);

        hardening * rate_factor * thermal_factor
    }
}

/// Continuum damage mechanics material.
#[derive(Debug, Clone)]
pub struct DamageModel {
    /// Young's modulus.
    pub e: f64,
    /// Poisson's ratio.
    pub nu: f64,
    /// Damage initiation strain.
    pub init_strain: f64,
    /// Failure strain.
    pub failure_strain: f64,
}

impl DamageModel {
    /// Creates a new damage model.
    pub fn new(e: f64, nu: f64, init_strain: f64, failure_strain: f64) -> Self {
        Self { e, nu, init_strain, failure_strain }
    }

    /// Computes damage variable.
    pub fn compute_damage(&self, equivalent_strain: f64) -> f64 {
        if equivalent_strain < self.init_strain {
            0.0
        } else if equivalent_strain >= self.failure_strain {
            1.0
        } else {
            // Linear damage evolution
            (equivalent_strain - self.init_strain)
                / (self.failure_strain - self.init_strain)
        }
    }

    /// Computes damaged stress.
    pub fn damaged_stress(&self, stress: &[f64; 6], damage: f64) -> [f64; 6] {
        let factor = 1.0 - damage;
        [
            stress[0] * factor,
            stress[1] * factor,
            stress[2] * factor,
            stress[3] * factor,
            stress[4] * factor,
            stress[5] * factor,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_elastic() {
        let mat = LinearElastic::new(210e9, 0.3, 7850.0);

        let strain = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0];
        let stress = mat.stress(&strain, 0.0);

        assert!(stress[0] > 0.0);
        assert!(stress[0] < 300e6);
    }

    #[test]
    fn test_neo_hookean() {
        let mat = NeoHookean::new(1e6, 2e6);
        assert!(mat.mu > 0.0);
        assert!(mat.kappa > 0.0);
    }

    #[test]
    fn test_mooney_rivlin() {
        let mat = MooneyRivlin::new(1e6, 0.5e6, 2e6);
        let energy = mat.strain_energy(3.5, 3.2, 1.0);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_von_mises_plasticity() {
        let mat = VonMisesPlasticity::new(210e9, 0.3, 250e6, 1e9);
        let state = StressState::default();
        let strain = [0.002, 0.0, 0.0, 0.0, 0.0, 0.0];

        let (new_state, _tangent) = mat.update(&strain, &state);
        assert!(new_state.stress.iter().any(|&s| s != 0.0));
    }

    #[test]
    fn test_johnson_cook() {
        let mat = JohnsonCook::new(
            210e9, 0.3, 500e6, 500e6, 0.02, 0.3, 1.0,
            1.0, 1800.0, 293.0,
        );

        let flow = mat.flow_stress(0.1, 100.0, 500.0);
        assert!(flow > 0.0);
    }

    #[test]
    fn test_damage_model() {
        let mat = DamageModel::new(210e9, 0.3, 0.001, 0.01);

        let d1 = mat.compute_damage(0.0005);
        assert!((d1 - 0.0).abs() < 1e-10);

        let d2 = mat.compute_damage(0.0055); // Halfway point
        assert!((d2 - 0.5).abs() < 0.01);

        let d3 = mat.compute_damage(0.02);
        assert!((d3 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wave_speed() {
        let mat = LinearElastic::new(210e9, 0.3, 7850.0);
        let c = mat.wave_speed();
        assert!(c > 4000.0); // Steel wave speed ~5000 m/s
        assert!(c < 7000.0);
    }
}
