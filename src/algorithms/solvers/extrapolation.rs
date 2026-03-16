//! Vector extrapolation methods for accelerating convergence.
//!
//! This module provides:
//! - Minimal Polynomial Extrapolation (MPE)
//! - Reduced Rank Extrapolation (RRE)
//! - Minimal Residual Extrapolation (MRE)

use nalgebra::{DMatrix, DVector};

/// Minimal Polynomial Extrapolation (MPE).
///
/// Accelerates convergence of fixed-point iterations by extrapolating
/// to the limit using a minimal polynomial.
pub struct MPE {
    /// Maximum number of stored vectors.
    max_vectors: usize,
    /// Stored iteration vectors.
    vectors: Vec<DVector<f64>>,
}

impl MPE {
    /// Creates a new MPE accelerator.
    pub fn new(max_vectors: usize) -> Self {
        Self {
            max_vectors,
            vectors: Vec::new(),
        }
    }

    /// Updates MPE with new iterate and returns accelerated solution if available.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        self.vectors.push(x.clone());

        // Need at least 3 vectors for MPE
        if self.vectors.len() < 3 {
            return None;
        }

        // Limit storage
        if self.vectors.len() > self.max_vectors {
            self.vectors.remove(0);
        }

        // Compute differences
        let k = self.vectors.len() - 1;
        let mut u: Vec<DVector<f64>> = Vec::with_capacity(k);
        for i in 0..k {
            u.push(&self.vectors[i + 1] - &self.vectors[i]);
        }

        // Form U matrix
        let n = x.len();
        let mut U = DMatrix::zeros(n, k);
        for j in 0..k {
            for i in 0..n {
                U[(i, j)] = u[j][i];
            }
        }

        // Solve least squares: U^T * U * c = -U^T * u_k
        let UtU = U.transpose() * &U;
        let Utu = U.transpose() * &u[k - 1];

        if let Some(c) = UtU.lu().solve(&Utu) {
            // Compute accelerated solution
            let mut sum = DVector::zeros(n);
            for j in 0..k {
                sum = sum + &self.vectors[j].scale(c[j]);
            }
            // Normalize
            let c_sum: f64 = c.iter().sum();
            if c_sum.abs() > 1e-15 {
                Some(sum / c_sum)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Resets MPE state.
    pub fn reset(&mut self) {
        self.vectors.clear();
    }
}

/// Reduced Rank Extrapolation (RRE).
///
/// Similar to MPE but uses different formulation for better stability.
pub struct RRE {
    max_vectors: usize,
    vectors: Vec<DVector<f64>>,
}

impl RRE {
    /// Creates a new RRE accelerator.
    pub fn new(max_vectors: usize) -> Self {
        Self {
            max_vectors,
            vectors: Vec::new(),
        }
    }

    /// Updates RRE with new iterate.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        self.vectors.push(x.clone());

        if self.vectors.len() < 3 {
            return None;
        }

        if self.vectors.len() > self.max_vectors {
            self.vectors.remove(0);
        }

        let k = self.vectors.len() - 1;

        // Compute first and second differences
        let mut u: Vec<DVector<f64>> = Vec::with_capacity(k);
        let mut du: Vec<DVector<f64>> = Vec::with_capacity(k - 1);

        for i in 0..k {
            u.push(&self.vectors[i + 1] - &self.vectors[i]);
        }
        for i in 0..k - 1 {
            du.push(&u[i + 1] - &u[i]);
        }

        let n = x.len();
        let mut D = DMatrix::zeros(n, k - 1);
        for j in 0..k - 1 {
            for i in 0..n {
                D[(i, j)] = du[j][i];
            }
        }

        // Solve least squares for coefficients
        let DtD = D.transpose() * &D;
        let Dtu = D.transpose() * &u[k - 1];

        if let Some(gamma) = DtD.lu().solve(&Dtu) {
            let mut sum = DVector::zeros(n);
            let mut c_sum: f64 = 1.0;

            for j in 0..k - 1 {
                sum = sum + self.vectors[j + 1].scale(gamma[j]);
                c_sum -= gamma[j];
            }

            if c_sum.abs() > 1e-15 {
                Some((self.vectors[0].scale(c_sum) + sum) / c_sum)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Resets RRE state.
    pub fn reset(&mut self) {
        self.vectors.clear();
    }
}

/// Minimal Residual Extrapolation (MRE).
///
/// Accelerates convergence by minimizing the residual norm.
pub struct MRE {
    max_vectors: usize,
    vectors: Vec<DVector<f64>>,
    residuals: Vec<DVector<f64>>,
}

impl MRE {
    /// Creates a new MRE accelerator.
    pub fn new(max_vectors: usize) -> Self {
        Self {
            max_vectors,
            vectors: Vec::new(),
            residuals: Vec::new(),
        }
    }

    /// Updates MRE with new iterate and residual.
    pub fn update(&mut self, x: &DVector<f64>, r: &DVector<f64>) -> Option<DVector<f64>> {
        self.vectors.push(x.clone());
        self.residuals.push(r.clone());

        if self.vectors.len() < 3 {
            return None;
        }

        if self.vectors.len() > self.max_vectors {
            self.vectors.remove(0);
            self.residuals.remove(0);
        }

        let k = self.residuals.len();

        // Form residual matrix
        let n = r.len();
        let mut R = DMatrix::zeros(n, k);
        for j in 0..k {
            for i in 0..n {
                R[(i, j)] = self.residuals[j][i];
            }
        }

        // Minimize ||R * c|| subject to sum(c) = 1
        // Use Lagrange multipliers
        let RtR = R.transpose() * &R;
        let ones = DVector::from_element(k, 1.0);

        // Solve augmented system using block structure
        let mut A = DMatrix::zeros(k + 1, k + 1);
        for i in 0..k {
            for j in 0..k {
                A[(i, j)] = RtR[(i, j)];
            }
            A[(i, k)] = 1.0;
            A[(k, i)] = 1.0;
        }

        let mut b = DVector::zeros(k + 1);
        b[k] = 1.0;

        if let Some(sol) = A.lu().solve(&b) {
            let c = sol.rows(0, k);
            let mut sum = DVector::zeros(n);
            for j in 0..k {
                sum = sum + self.vectors[j].scale(c[j]);
            }
            Some(sum)
        } else {
            None
        }
    }

    /// Resets MRE state.
    pub fn reset(&mut self) {
        self.vectors.clear();
        self.residuals.clear();
    }
}

/// Steffensen acceleration with MPE/RRE.
pub struct SteffensenExtrapolation {
    mpe: MPE,
    rre: RRE,
}

impl SteffensenExtrapolation {
    /// Creates a new Steffensen extrapolation accelerator.
    pub fn new(max_vectors: usize) -> Self {
        Self {
            mpe: MPE::new(max_vectors),
            rre: RRE::new(max_vectors),
        }
    }

    /// Attempts MPE acceleration.
    pub fn try_mpe(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        self.mpe.update(x)
    }

    /// Attempts RRE acceleration.
    pub fn try_rre(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        self.rre.update(x)
    }

    /// Resets all accelerators.
    pub fn reset(&mut self) {
        self.mpe.reset();
        self.rre.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpe_convergence() {
        // Test with converging sequence x_{k+1} = 0.5 * x_k
        let mut mpe = MPE::new(10);
        let mut x = DVector::from_element(5, 1.0);

        for _ in 0..5 {
            x = x.scale(0.5);
            let _ = mpe.update(&x);
        }

        // MPE should give result close to zero
        if let Some(x_accel) = mpe.update(&x.scale(0.5)) {
            assert!(x_accel.norm() < x.norm());
        }
    }

    #[test]
    fn test_rre_convergence() {
        let mut rre = RRE::new(10);
        let mut x = DVector::from_element(5, 1.0);

        for _ in 0..5 {
            x = x.scale(0.5);
            let _ = rre.update(&x);
        }

        if let Some(x_accel) = rre.update(&x.scale(0.5)) {
            assert!(x_accel.norm() < x.norm());
        }
    }

    #[test]
    fn test_steveffensen_extrapolation() {
        let mut accel = SteffensenExtrapolation::new(10);
        let x = DVector::from_element(5, 0.1);

        for i in 0..10 {
            let x_new = x.scale(0.9f64.powi(i as i32));
            let _ = accel.try_mpe(&x_new);
            let _ = accel.try_rre(&x_new);
        }
    }
}
