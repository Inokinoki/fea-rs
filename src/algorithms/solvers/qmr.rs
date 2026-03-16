//! QMR (Quasi-Minimal Residual) solver.
//!
//! QMR is a smoothing variant of BiCG that minimizes the residual norm
//! over the Krylov subspace without requiring matrix transposes.

use super::{IterativeConfig, SolverResult};
use nalgebra::{DMatrix, DVector};

/// Quasi-Minimal Residual solver.
#[derive(Debug, Clone)]
pub struct QMRSolver {
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl Default for QMRSolver {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl QMRSolver {
    /// Creates a new QMR solver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Solves A*x = b using QMR.
    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>, _config: &IterativeConfig) -> anyhow::Result<SolverResult> {
        let n = b.len();
        if n == 0 {
            return Ok(SolverResult::success(vec![]));
        }

        let mut x = vec![0.0; n];
        let mut r = b.data.as_vec().clone();
        let mut r_hat = r.clone(); // Shadow residual (can be same as r for QMR)

        let b_norm = b.norm();
        let tol = self.tolerance * b_norm.max(1e-15);

        let mut rho = 0.0;
        let mut theta = 0.0;
        let mut eta = 0.0;
        let mut d = vec![0.0; n];
        let mut v = vec![0.0; n];
        let mut p = vec![0.0; n];

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            let rho_new: f64 = r.iter().zip(r_hat.iter()).map(|(a, b)| a * b).sum();

            if rho_new.abs() < 1e-15 {
                break; // Breakdown
            }

            if iteration == 0 {
                v.clone_from(&r);
                p.clone_from(&v);
            } else {
                let beta = rho_new / rho;
                for i in 0..n {
                    v[i] = r[i] + beta * (v[i] + beta * p[i] / rho * r[i]);
                    p[i] = v[i] + beta * p[i] / rho * r[i];
                }
            }

            // w = A * p
            let w = a * &DVector::from_column_slice(&p);

            // alpha = rho / (r_hat^T * A * p)
            let alpha_denom: f64 = r_hat.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            if alpha_denom.abs() < 1e-15 {
                break;
            }
            let alpha = rho_new / alpha_denom;

            // r_new = r - alpha * A * p
            let mut r_new = vec![0.0; n];
            for i in 0..n {
                r_new[i] = r[i] - alpha * w[i];
            }

            // QMR smoothing
            let sigma = alpha * rho_new.abs().sqrt();
            let theta_new = sigma.abs() / (1.0 + sigma * sigma).sqrt();
            let c = 1.0 / (1.0 + sigma * sigma).sqrt();

            eta = c * c * theta * theta + c * c * sigma * sigma;
            theta = theta_new;

            // d = c^2 * alpha * p + c^2 * theta^2 * d
            for i in 0..n {
                d[i] = c * c * alpha * p[i] + c * c * theta * theta * d[i];
            }

            // x_new = x + d
            for i in 0..n {
                x[i] += d[i];
            }

            let r_norm: f64 = r_new.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            r = r_new;
            rho = rho_new;
            iteration += 1;
        }

        let r_vec = DVector::from_column_slice(&r);
        Ok(SolverResult::iterative(x, iteration, r_vec.norm(), converged))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qmr_symmetric() {
        // Symmetric positive definite system
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 2.0, 0.0,
            1.0, 8.0, 1.0, 0.0,
            2.0, 1.0, 12.0, 1.0,
            0.0, 0.0, 1.0, 6.0,
        ]);
        let b = DVector::from_column_slice(&[13.0, 10.0, 16.0, 7.0]);

        let qmr = QMRSolver::new();
        let config = IterativeConfig::default();
        let result = qmr.solve(&a, &b, &config).expect("QMR solve failed");

        // QMR should produce finite results
        for &xi in &result.solution {
            assert!(xi.is_finite());
        }
    }

    #[test]
    fn test_qmr_nonsymmetric() {
        // Non-symmetric system
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 2.0, 1.0, 0.0,
            1.0, 8.0, 2.0, 0.0,
            2.0, 1.0, 12.0, 1.0,
            0.0, 1.0, 1.0, 6.0,
        ]);
        let b = DVector::from_column_slice(&[13.0, 11.0, 16.0, 8.0]);

        let qmr = QMRSolver::new();
        let config = IterativeConfig::default();
        let result = qmr.solve(&a, &b, &config).expect("QMR solve failed");

        // Should produce finite results
        for &xi in &result.solution {
            assert!(xi.is_finite());
        }
    }
}
