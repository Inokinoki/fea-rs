//! Generalized Conjugate Residual (GCR) solver.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! GCR is effective for non-symmetric positive definite systems.
//! It minimizes the residual norm over the Krylov subspace.

use super::{IterativeConfig, Preconditioner, SolverResult};
use nalgebra::{DMatrix, DVector};

/// Generalized Conjugate Residual solver.
#[derive(Debug, Clone)]
pub struct GCRSolver {
    /// Maximum dimension of Krylov subspace before restart.
    pub restart: usize,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl Default for GCRSolver {
    fn default() -> Self {
        Self {
            restart: 30,
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl GCRSolver {
    /// Creates a new GCR solver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a GCR solver with custom restart parameter.
    pub fn with_restart(restart: usize) -> Self {
        Self { restart, ..Default::default() }
    }

    /// Solves A*x = b using GCR.
    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>, config: &IterativeConfig) -> anyhow::Result<SolverResult> {
        let n = b.len();
        if n == 0 {
            return Ok(SolverResult::success(vec![]));
        }

        let mut x = vec![0.0; n];
        let mut r = b.data.as_vec().clone();
        let b_norm = b.norm();
        let tol = config.tolerance * b_norm.max(1e-15);

        let mut total_iterations = 0;
        let mut converged = false;

        // Store Krylov basis vectors and matrix-vector products
        let mut v: Vec<DVector<f64>> = Vec::with_capacity(self.restart);
        let mut w: Vec<DVector<f64>> = Vec::with_capacity(self.restart);
        let mut h: Vec<Vec<f64>> = Vec::with_capacity(self.restart);

        while total_iterations < self.max_iterations {
            // Compute initial residual
            let r_vec = DVector::from_column_slice(&r);
            let r_norm = r_vec.norm();

            if r_norm <= tol {
                converged = true;
                break;
            }

            // Clear storage for new cycle
            v.clear();
            w.clear();
            h.clear();

            // Normalize initial residual
            let mut v0 = r_vec / r_norm;

            // Apply preconditioner if enabled
            if config.preconditioner == Preconditioner::Jacobi {
                for i in 0..n {
                    let a_ii = a[(i, i)];
                    if a_ii.abs() > 1e-15 {
                        v0[i] /= a_ii;
                    }
                }
                let norm = v0.norm();
                if norm > 1e-15 {
                    v0 /= norm;
                }
            }

            v.push(v0.clone());

            let mut cs: Vec<f64> = Vec::with_capacity(self.restart); // Cosines for Givens rotations
            let mut sn: Vec<f64> = Vec::with_capacity(self.restart); // Sines for Givens rotations
            let mut g = vec![0.0; self.restart + 1];
            g[0] = r_norm;

            let mut j = 0;
            while j < self.restart && total_iterations < self.max_iterations {
                // w_j = A * v_j
                let w_j = a * &v[j];
                w.push(w_j.clone());

                // Modified Gram-Schmidt orthogonalization
                let mut h_j = vec![0.0; j + 1];
                for i in 0..=j {
                    h_j[i] = v[i].dot(&w_j);
                }

                // Orthogonalize
                let mut w_j_ortho = w_j.clone();
                for i in 0..=j {
                    for k in 0..n {
                        w_j_ortho[k] -= h_j[i] * v[i][k];
                    }
                }

                let h_next = w_j_ortho.norm();
                h_j.push(h_next);
                h.push(h_j);

                if h_next > 1e-15 {
                    v.push(w_j_ortho / h_next);
                }

                // Apply previous Givens rotations
                for i in 0..j {
                    let temp = cs[i] * h[j][i] + sn[i] * h[j][i + 1];
                    h[j][i + 1] = -sn[i] * h[j][i] + cs[i] * h[j][i + 1];
                    h[j][i] = temp;
                }

                // Compute new Givens rotation
                let (c, s) = givens(h[j][j], h[j][j + 1]);
                cs.push(c);
                sn.push(s);

                // Apply to H and g
                h[j][j] = c * h[j][j] + s * h[j][j + 1];
                h[j][j + 1] = 0.0;
                g[j] = c * g[j];
                g[j + 1] = -s * g[j + 1];

                // Check convergence
                if g[j + 1].abs() <= tol {
                    break;
                }

                j += 1;
                total_iterations += 1;
            }

            // Solve upper triangular system
            let k = (j + 1).min(self.restart);
            let mut y = vec![0.0; k];
            for i in (0..k).rev() {
                y[i] = g[i];
                for l in (i + 1)..k {
                    y[i] -= h[l][i] * y[l];
                }
                if h[i][i].abs() > 1e-15 {
                    y[i] /= h[i][i];
                }
            }

            // Update solution
            for i in 0..k {
                for l in 0..n {
                    x[l] += y[i] * v[i][l];
                }
            }

            if g[j].abs() <= tol {
                converged = true;
                break;
            }

            // Update residual
            let x_vec = DVector::from_column_slice(&x);
            r = (b - a * &x_vec).data.as_vec().clone();
        }

        let r_vec = DVector::from_column_slice(&r);
        Ok(SolverResult::iterative(x, total_iterations, r_vec.norm(), converged))
    }
}

/// Givens rotation coefficients.
fn givens(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (1.0, 0.0)
    } else if b.abs() > a.abs() {
        let t = -a / b;
        let s = 1.0_f64 / (1.0 + t * t).sqrt();
        (s * t, s)
    } else {
        let t = -b / a;
        let c = 1.0_f64 / (1.0 + t * t).sqrt();
        (c, c * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcr_symmetric() {
        // Symmetric positive definite system
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 2.0, 0.0,
            1.0, 8.0, 1.0, 0.0,
            2.0, 1.0, 12.0, 1.0,
            0.0, 0.0, 1.0, 6.0,
        ]);
        let b = DVector::from_column_slice(&[13.0, 10.0, 16.0, 7.0]);

        let gcr = GCRSolver::new();
        let config = IterativeConfig::default();
        let result = gcr.solve(&a, &b, &config).expect("GCR solve failed");

        assert!(result.converged, "GCR should converge for SPD matrix");
        assert!(result.iterations.unwrap() < 50);

        // Verify solution
        let x = DVector::from_column_slice(&result.solution);
        let residual = &b - &a * &x;
        assert!(residual.norm() < 1e-6);
    }

    #[test]
    fn test_gcr_nonsymmetric() {
        // Non-symmetric system
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 2.0, 1.0, 0.0,
            1.0, 8.0, 2.0, 0.0,
            2.0, 1.0, 12.0, 1.0,
            0.0, 1.0, 1.0, 6.0,
        ]);
        let b = DVector::from_column_slice(&[13.0, 11.0, 16.0, 8.0]);

        let gcr = GCRSolver::new();
        let config = IterativeConfig::default();
        let result = gcr.solve(&a, &b, &config).expect("GCR solve failed");

        assert!(result.converged, "GCR should converge for non-symmetric matrix");

        // Verify solution produces finite results
        for &xi in &result.solution {
            assert!(xi.is_finite());
        }
    }

    #[test]
    fn test_gcr_restart() {
        // Larger system requiring restart
        let n = 50;
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 4.0;
            if i > 0 {
                a[(i, i - 1)] = -1.0;
            }
            if i < n - 1 {
                a[(i, i + 1)] = -1.0;
            }
        }

        let b = DVector::from_element(n, 3.0);

        let gcr = GCRSolver::with_restart(10); // Small restart to force multiple cycles
        let config = IterativeConfig::default();
        let result = gcr.solve(&a, &b, &config).expect("GCR solve failed");

        assert!(result.converged, "GCR should converge with restarts");

        // Verify solution
        let x = DVector::from_column_slice(&result.solution);
        let residual = &b - &a * &x;
        assert!(residual.norm() < 1e-6);
    }

    #[test]
    fn test_givens_rotation() {
        let (c, s) = givens(3.0, 4.0);

        // Should satisfy c^2 + s^2 = 1
        assert!((c * c + s * s - 1.0).abs() < 1e-10);

        // Should be bounded
        assert!(c.abs() <= 1.0);
        assert!(s.abs() <= 1.0);

        // Test with b = 0
        let (c2, s2) = givens(5.0, 0.0);
        assert!((c2 - 1.0).abs() < 1e-10);
        assert!(s2.abs() < 1e-10);
    }
}
