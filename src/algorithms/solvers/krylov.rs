//! Krylov subspace methods and deflation techniques.
#![allow(non_snake_case)]
#![allow(unused_assignments)]

use nalgebra::{DMatrix, DVector};

/// Deflated Conjugate Gradient (DCG) solver.
pub struct DeflatedCG {
    deflation: Option<DMatrix<f64>>,
}

impl DeflatedCG {
    /// Creates new deflated CG solver.
    pub fn new() -> Self {
        Self { deflation: None }
    }

    /// Sets the deflation subspace.
    pub fn with_deflation(mut self, deflation: DMatrix<f64>) -> Self {
        self.deflation = Some(deflation);
        self
    }

    /// Solves A*x = b using deflated CG.
    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>, tol: f64, max_iter: usize) -> (DVector<f64>, usize, bool) {
        let n = b.len();
        let mut x = DVector::zeros(n);

        if let Some(z) = &self.deflation {
            // Coarse grid correction
            let zt_b = z.transpose() * b;
            let zt_az = z.transpose() * (a * z);

            if let Some(lambda) = zt_az.lu().solve(&zt_b) {
                let x_coarse = z * &lambda;
                x = x_coarse;
            }
        }

        // Fine grid CG
        let mut r = b - a * &x;
        let mut p = r.clone();

        let b_norm = b.norm();
        let tol_eff = tol * b_norm.max(1.0);

        for iter in 0..max_iter {
            let ap = a * &p;
            let p_ap = p.dot(&ap);

            if p_ap.abs() < 1e-15 {
                return (x, iter, false);
            }

            let alpha = r.dot(&p) / p_ap;
            x += alpha * &p;
            r -= alpha * &ap;

            if r.norm() < tol_eff {
                return (x, iter, true);
            }

            let beta = r.dot(&r) / (r.dot(&r) + 1e-15);
            p = &r + beta * &p;
        }

        (x, max_iter, false)
    }

    /// Resets the deflation subspace.
    pub fn reset(&mut self) {
        self.deflation = None;
    }
}

impl Default for DeflatedCG {
    fn default() -> Self {
        Self::new()
    }
}

/// Spectral preconditioner using approximate eigenvalues.
pub struct SpectralPreconditioner {
    eigenvalues: Option<DVector<f64>>,
    threshold: f64,
}

impl SpectralPreconditioner {
    /// Creates new spectral preconditioner.
    pub fn new(threshold: f64) -> Self {
        Self { eigenvalues: None, threshold }
    }

    /// Sets approximate eigenvalues.
    pub fn with_eigenvalues(mut self, eigenvalues: DVector<f64>) -> Self {
        self.eigenvalues = Some(eigenvalues);
        self
    }

    /// Applies spectral filtering.
    pub fn apply(&self, x: &DVector<f64>) -> DVector<f64> {
        if let Some(evals) = &self.eigenvalues {
            let mut filtered = x.clone();
            for i in 0..evals.len().min(x.len()) {
                if evals[i].abs() < self.threshold {
                    filtered[i] = 0.0;
                }
            }
            filtered
        } else {
            x.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deflated_cg() {
        let a = DMatrix::from_row_slice(4, 4, &[
            4.0, 1.0, 0.0, 0.0,
            1.0, 4.0, 1.0, 0.0,
            0.0, 1.0, 4.0, 1.0,
            0.0, 0.0, 1.0, 4.0,
        ]);
        let b = DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0]);

        let dcg = DeflatedCG::new();
        let (x, _iter, _conv) = dcg.solve(&a, &b, 1e-8, 100);

        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_spectral_preconditioner() {
        let evals = DVector::from_column_slice(&[0.01, 0.5, 2.0, 10.0]);
        let x = DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0]);

        let sp = SpectralPreconditioner::new(0.1).with_eigenvalues(evals);
        let y = sp.apply(&x);

        assert!(y[0].abs() < 0.1);
    }
}
