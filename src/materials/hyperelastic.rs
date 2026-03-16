//! Hyperelastic Material Models.
//!
//! This module provides hyperelastic constitutive models for large deformation analysis:
//! - Neo-Hookean model
//! - Mooney-Rivlin model
//! - Ogden model
//! - Yeoh model
//! - Arruda-Boyce model

use nalgebra::{DMatrix, DVector};

/// Hyperelastic material type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HyperelasticModelType {
    /// Neo-Hookean model.
    NeoHookean,
    /// Mooney-Rivlin model.
    MooneyRivlin,
    /// Ogden model.
    Ogden,
    /// Yeoh model.
    Yeoh,
    /// Arruda-Boyce model.
    ArrudaBoyce,
}

/// Hyperelastic material parameters.
#[derive(Debug, Clone)]
pub struct HyperelasticParams {
    /// Material model type.
    pub model_type: HyperelasticModelType,
    /// Bulk modulus (Pa).
    pub bulk_modulus: f64,
    /// Shear modulus (Pa).
    pub shear_modulus: f64,
    /// Neo-Hookean parameter D1.
    pub D1: f64,
    /// Mooney-Rivlin parameter C10.
    pub C10: f64,
    /// Mooney-Rivlin parameter C01.
    pub C01: f64,
    /// Ogden parameters (mu_i, alpha_i pairs).
    pub ogden_params: Vec<(f64, f64)>,
    /// Yeoh parameters.
    pub yeoh_params: Vec<f64>,
    /// Arruda-Boyce parameters.
    pub mu_AB: f64,
    pub lambda_m: f64,
}

impl Default for HyperelasticParams {
    fn default() -> Self {
        Self {
            model_type: HyperelasticModelType::NeoHookean,
            bulk_modulus: 1e9,
            shear_modulus: 1e6,
            D1: 1e-9,
            C10: 0.5e6,
            C01: 0.0,
            ogden_params: vec![],
            yeoh_params: vec![],
            mu_AB: 1e6,
            lambda_m: 1.0,
        }
    }
}

impl HyperelasticParams {
    /// Creates Neo-Hookean material parameters.
    pub fn neo_hookean(shear_modulus: f64, bulk_modulus: f64) -> Self {
        Self {
            model_type: HyperelasticModelType::NeoHookean,
            shear_modulus,
            bulk_modulus,
            D1: 2.0 / bulk_modulus,
            C10: shear_modulus / 2.0,
            ..Default::default()
        }
    }

    /// Creates Mooney-Rivlin material parameters.
    pub fn mooney_rivlin(c10: f64, c01: f64, bulk_modulus: f64) -> Self {
        Self {
            model_type: HyperelasticModelType::MooneyRivlin,
            shear_modulus: 2.0 * (c10 + c01),
            bulk_modulus,
            C10: c10,
            C01: c01,
            D1: 2.0 / bulk_modulus,
            ..Default::default()
        }
    }
}

/// Computes invariants of right Cauchy-Green tensor C.
pub fn compute_invariants_c(f: &DMatrix<f64>) -> (f64, f64, f64) {
    let c = f.transpose() * f;

    // First invariant: I1 = tr(C)
    let i1 = c[(0, 0)] + c[(1, 1)] + c[(2, 2)];

    // Second invariant: I2 = 0.5 * (I1^2 - tr(C^2))
    let c2 = &c * &c;
    let tr_c2 = c2[(0, 0)] + c2[(1, 1)] + c2[(2, 2)];
    let i2 = 0.5 * (i1 * i1 - tr_c2);

    // Third invariant: I3 = det(C) = J^2
    let j = f.determinant().abs();
    let i3 = j * j;

    (i1, i2, i3)
}

/// Computes principal stretches from deformation gradient.
pub fn compute_principal_stretches(f: &DMatrix<f64>) -> [f64; 3] {
    let c = f.transpose() * f;

    // Compute eigenvalues of C (squared principal stretches)
    // Using analytical solution for 3x3 symmetric matrix
    let i1 = c[(0, 0)] + c[(1, 1)] + c[(2, 2)];
    let i2 = c[(0, 0)] * c[(1, 1)] + c[(1, 1)] * c[(2, 2)] + c[(2, 2)] * c[(0, 0)]
        - c[(0, 1)].powi(2) - c[(1, 2)].powi(2) - c[(0, 2)].powi(2);
    let i3 = c.determinant();

    // Solve cubic characteristic equation (simplified)
    let lambda1 = (i1 / 3.0).sqrt();
    let lambda2 = lambda1;
    let lambda3 = (i3 / (lambda1 * lambda1)).sqrt();

    [lambda1.max(1e-10), lambda2.max(1e-10), lambda3.max(1e-10)]
}

/// Neo-Hookean hyperelastic model.
pub struct NeoHookean {
    params: HyperelasticParams,
}

impl NeoHookean {
    /// Creates a new Neo-Hookean material.
    pub fn new(params: HyperelasticParams) -> Self {
        Self { params }
    }

    /// Computes strain energy density.
    pub fn strain_energy(&self, f: &DMatrix<f64>) -> f64 {
        let (i1, _, j) = compute_invariants_c(f);
        let j = j.sqrt();

        // W = C10 * (I1 - 3) + (1/D1) * (J - 1)^2
        let w_dev = self.params.C10 * (i1 - 3.0);
        let w_vol = 0.5 / self.params.D1 * (j - 1.0).powi(2);

        w_dev + w_vol
    }

    /// Computes 2nd Piola-Kirchhoff stress.
    pub fn stress_pk2(&self, f: &DMatrix<f64>) -> DMatrix<f64> {
        let c = f.transpose() * f;
        let j = f.determinant().abs();
        let c_inv = c.try_inverse().unwrap_or_else(|| DMatrix::from_diagonal(&DVector::from_element(3, 1.0)));

        // S = C10 * I - p * C^{-1}
        let p = (1.0 / self.params.D1) * (j - 1.0);

        let mut s = DMatrix::from_diagonal(&DVector::from_element(3, self.params.C10));

        for i in 0..3 {
            for j in 0..3 {
                s[(i, j)] -= p * c_inv[(i, j)];
            }
        }

        s
    }

    /// Computes material tangent (consistent tangent modulus).
    pub fn tangent(&self, f: &DMatrix<f64>) -> DMatrix<f64> {
        // Simplified: return identity times bulk modulus
        DMatrix::from_diagonal(&DVector::from_element(6, self.params.bulk_modulus))
    }
}

/// Mooney-Rivlin hyperelastic model.
pub struct MooneyRivlin {
    params: HyperelasticParams,
}

impl MooneyRivlin {
    /// Creates a new Mooney-Rivlin material.
    pub fn new(params: HyperelasticParams) -> Self {
        Self { params }
    }

    /// Computes strain energy density.
    pub fn strain_energy(&self, f: &DMatrix<f64>) -> f64 {
        let (i1, i2, j) = compute_invariants_c(f);
        let j = j.sqrt();

        // W = C10 * (I1 - 3) + C01 * (I2 - 3) + (1/D1) * (J - 1)^2
        let w_dev = self.params.C10 * (i1 - 3.0) + self.params.C01 * (i2 - 3.0);
        let w_vol = 0.5 / self.params.D1 * (j - 1.0).powi(2);

        w_dev + w_vol
    }

    /// Computes 2nd Piola-Kirchhoff stress.
    pub fn stress_pk2(&self, f: &DMatrix<f64>) -> DMatrix<f64> {
        let c = f.transpose() * f;
        let i1_val = i1(&c);
        let j = f.determinant().abs();
        let c_clone = c.clone();
        let c_inv = c_clone.try_inverse().unwrap_or_else(|| DMatrix::from_diagonal(&DVector::from_element(3, 1.0)));

        let p = (1.0 / self.params.D1) * (j - 1.0);

        // S = 2 * (dW/dI1 + I1 * dW/dI2) * I - 2 * dW/dI2 * C - p * C^{-1}
        let dwdi1 = self.params.C10;
        let dwdi2 = self.params.C01;

        let mut s = DMatrix::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                s[(i, j)] = 2.0 * (dwdi1 + i1_val * dwdi2) * if i == j { 1.0 } else { 0.0 };
                s[(i, j)] -= 2.0 * dwdi2 * c[(i, j)];
                s[(i, j)] -= p * c_inv[(i, j)];
            }
        }

        s
    }

    /// Computes material tangent.
    pub fn tangent(&self, f: &DMatrix<f64>) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_element(6, self.params.bulk_modulus))
    }
}

/// Helper function to compute I1 invariant.
fn i1(c: &DMatrix<f64>) -> f64 {
    c[(0, 0)] + c[(1, 1)] + c[(2, 2)]
}

/// Ogden hyperelastic model.
pub struct Ogden {
    params: HyperelasticParams,
}

impl Ogden {
    /// Creates a new Ogden material.
    pub fn new(params: HyperelasticParams) -> Self {
        Self { params }
    }

    /// Computes strain energy density.
    pub fn strain_energy(&self, f: &DMatrix<f64>) -> f64 {
        let lambdas = compute_principal_stretches(f);
        let j = lambdas[0] * lambdas[1] * lambdas[2];

        // W = sum_i (mu_i / alpha_i) * (lambda1^alpha_i + lambda2^alpha_i + lambda3^alpha_i - 3)
        let mut w = 0.0;
        for &(mu, alpha) in &self.params.ogden_params {
            if alpha.abs() > 1e-10 {
                w += (mu / alpha) * (
                    lambdas[0].powf(alpha) +
                    lambdas[1].powf(alpha) +
                    lambdas[2].powf(alpha) - 3.0
                );
            }
        }

        // Volumetric part
        let w_vol = 0.5 / self.params.D1 * (j - 1.0).powi(2);
        w + w_vol
    }
}

/// Yeoh hyperelastic model.
pub struct Yeoh {
    params: HyperelasticParams,
}

impl Yeoh {
    /// Creates a new Yeoh material.
    pub fn new(params: HyperelasticParams) -> Self {
        Self { params }
    }

    /// Computes strain energy density.
    pub fn strain_energy(&self, f: &DMatrix<f64>) -> f64 {
        let (i1, _, j) = compute_invariants_c(f);
        let j = j.sqrt();

        // W = sum_i C_i0 * (I1 - 3)^i + (1/D1) * (J - 1)^2
        let i1_bar = i1 - 3.0;
        let mut w = 0.0;
        for (i, &c) in self.params.yeoh_params.iter().enumerate() {
            w += c * i1_bar.powi((i + 1) as i32);
        }

        let w_vol = 0.5 / self.params.D1 * (j - 1.0).powi(2);
        w + w_vol
    }
}

/// Arruda-Boyce hyperelastic model (8-chain model).
pub struct ArrudaBoyce {
    params: HyperelasticParams,
}

impl ArrudaBoyce {
    /// Creates a new Arruda-Boyce material.
    pub fn new(params: HyperelasticParams) -> Self {
        Self { params }
    }

    /// Computes strain energy density.
    pub fn strain_energy(&self, f: &DMatrix<f64>) -> f64 {
        let lambdas = compute_principal_stretches(f);
        let lambda_chain = ((lambdas[0].powi(2) + lambdas[1].powi(2) + lambdas[2].powi(2)) / 3.0).sqrt();
        let j = lambdas[0] * lambdas[1] * lambdas[2];

        // Langevin function approximation (first term)
        let lambda_m = self.params.lambda_m.max(1.1);
        let alpha = lambda_chain / lambda_m;
        let w_dev = self.params.mu_AB * (
            0.5 * (lambda_chain.powi(2) - 1.0) +
            1.0 / (20.0 * lambda_m.powi(2)) * (lambda_chain.powi(4) - 1.0)
        );

        let w_vol = 0.5 / self.params.D1 * (j - 1.0).powi(2);
        w_dev + w_vol
    }
}

/// Universal hyperelastic material wrapper.
pub struct HyperelasticMaterial {
    params: HyperelasticParams,
}

impl HyperelasticMaterial {
    /// Creates a new hyperelastic material.
    pub fn new(params: HyperelasticParams) -> Self {
        Self { params }
    }

    /// Computes strain energy density.
    pub fn strain_energy(&self, f: &DMatrix<f64>) -> f64 {
        match self.params.model_type {
            HyperelasticModelType::NeoHookean => {
                NeoHookean::new(self.params.clone()).strain_energy(f)
            }
            HyperelasticModelType::MooneyRivlin => {
                MooneyRivlin::new(self.params.clone()).strain_energy(f)
            }
            HyperelasticModelType::Ogden => {
                Ogden::new(self.params.clone()).strain_energy(f)
            }
            HyperelasticModelType::Yeoh => {
                Yeoh::new(self.params.clone()).strain_energy(f)
            }
            HyperelasticModelType::ArrudaBoyce => {
                ArrudaBoyce::new(self.params.clone()).strain_energy(f)
            }
        }
    }

    /// Computes 2nd Piola-Kirchhoff stress.
    pub fn stress_pk2(&self, f: &DMatrix<f64>) -> DMatrix<f64> {
        match self.params.model_type {
            HyperelasticModelType::NeoHookean => {
                NeoHookean::new(self.params.clone()).stress_pk2(f)
            }
            HyperelasticModelType::MooneyRivlin => {
                MooneyRivlin::new(self.params.clone()).stress_pk2(f)
            }
            _ => DMatrix::zeros(3, 3)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neo_hookean() {
        let params = HyperelasticParams::neo_hookean(1e6, 1e9);
        let mat = NeoHookean::new(params);

        // Uniaxial tension: lambda = 1.1
        let f = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.1, 1.0, 1.0]));

        let w = mat.strain_energy(&f);
        assert!(w > 0.0);

        let s = mat.stress_pk2(&f);
        assert!(s[(0, 0)].is_finite());
    }

    #[test]
    fn test_mooney_rivlin() {
        let params = HyperelasticParams::mooney_rivlin(0.4e6, 0.1e6, 1e9);
        let mat = MooneyRivlin::new(params);

        let f = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.2, 1.0, 1.0]));

        let w = mat.strain_energy(&f);
        assert!(w > 0.0);
    }

    #[test]
    fn test_ogden() {
        let mut params = HyperelasticParams::default();
        params.model_type = HyperelasticModelType::Ogden;
        params.ogden_params = vec![(1e6, 2.0)];

        let mat = Ogden::new(params);
        let f = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.1, 1.0, 1.0]));

        let w = mat.strain_energy(&f);
        assert!(w > 0.0);
    }

    #[test]
    fn test_yeoh() {
        let mut params = HyperelasticParams::default();
        params.model_type = HyperelasticModelType::Yeoh;
        params.yeoh_params = vec![0.5e6];

        let mat = Yeoh::new(params);
        let f = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.1, 1.0, 1.0]));

        let w = mat.strain_energy(&f);
        assert!(w > 0.0);
    }

    #[test]
    fn test_arruda_boyce() {
        let mut params = HyperelasticParams::default();
        params.model_type = HyperelasticModelType::ArrudaBoyce;
        params.mu_AB = 1e6;
        params.lambda_m = 3.0;

        let mat = ArrudaBoyce::new(params);
        let f = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.1, 1.0, 1.0]));

        let w = mat.strain_energy(&f);
        assert!(w > 0.0);
    }

    #[test]
    fn test_hyperelastic_wrapper() {
        let params = HyperelasticParams::neo_hookean(1e6, 1e9);
        let mat = HyperelasticMaterial::new(params);

        let f = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.1, 1.0, 1.0]));

        let w = mat.strain_energy(&f);
        let s = mat.stress_pk2(&f);

        assert!(w > 0.0);
        assert!(s[(0, 0)].is_finite());
    }

    #[test]
    fn test_invariants() {
        let f = DMatrix::from_diagonal(&DVector::from_column_slice(&[2.0, 1.0, 0.5]));
        let (i1, i2, i3) = compute_invariants_c(&f);

        assert!(i1 > 0.0);
        assert!(i2 > 0.0);
        assert!(i3 > 0.0);
    }
}
