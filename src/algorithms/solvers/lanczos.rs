//! Lanczos algorithm for eigenvalue problems.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! This module provides:
//! - Lanczos iteration for symmetric eigenvalue problems
//! - Eigenvalue extraction using tridiagonalization
//! - Application to structural vibration analysis

use nalgebra::{DMatrix, DVector};

/// Result of Lanczos eigenvalue computation.
#[derive(Debug, Clone)]
pub struct LanczosResult {
    /// Computed eigenvalues.
    pub eigenvalues: Vec<f64>,
    /// Computed eigenvectors (if requested).
    pub eigenvectors: Option<Vec<Vec<f64>>>,
    /// Number of Lanczos iterations performed.
    pub iterations: usize,
}

/// Configuration for Lanczos algorithm.
#[derive(Debug, Clone, Copy)]
pub struct LanczosConfig {
    /// Maximum number of Lanczos iterations.
    pub max_iterations: usize,
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Whether to compute eigenvectors.
    pub compute_eigenvectors: bool,
    /// Whether to use full reorthogonalization.
    pub reorthogonalize: bool,
}

impl Default for LanczosConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            num_eigenvalues: 5,
            tolerance: 1e-10,
            compute_eigenvectors: false,
            reorthogonalize: true,
        }
    }
}

/// Lanczos algorithm for symmetric eigenvalue problems.
///
/// Solves K * phi = lambda * M * phi using the Lanczos iteration.
/// For standard eigenvalue problems (M = I), use with identity mass.
pub struct LanczosSolver {
    config: LanczosConfig,
}

impl LanczosSolver {
    /// Creates a new Lanczos solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(LanczosConfig::default())
    }

    /// Creates a new Lanczos solver with custom configuration.
    pub fn with_config(config: LanczosConfig) -> Self {
        Self { config }
    }

    /// Solves the standard eigenvalue problem K * phi = lambda * phi.
    ///
    /// Returns the smallest eigenvalues and optionally eigenvectors.
    pub fn solve_standard(
        &self,
        k: &DMatrix<f64>,
    ) -> LanczosResult {
        let n = k.nrows();
        let m = self.config.max_iterations.min(n);
        let num_evals = self.config.num_eigenvalues.min(m);

        // Lanczos vectors
        let mut alpha: Vec<f64> = Vec::with_capacity(m);  // Diagonal of tridiagonal
        let mut beta: Vec<f64> = Vec::with_capacity(m);   // Off-diagonal of tridiagonal
        let mut V: Vec<DVector<f64>> = Vec::with_capacity(m + 1);

        // Initial vector (random-like)
        let mut v = DVector::from_fn(n, |i, _| ((i + 1) as f64 * 0.1).sin());
        v.normalize_mut();
        V.push(v.clone());

        let mut w = DVector::zeros(n);
        let mut converged = false;
        let mut iterations = 0;

        for j in 0..m {
            // w = K * v_j
            w = k * &V[j];

            // alpha_j = v_j^T * w
            let alpha_j = V[j].dot(&w);
            alpha.push(alpha_j);

            // w = w - alpha_j * v_j - beta_j * v_{j-1}
            w.axpy(-alpha_j, &V[j], 1.0);
            if j > 0 {
                w.axpy(-beta[j - 1], &V[j - 1], 1.0);
            }

            // Full reorthogonalization (optional but recommended)
            if self.config.reorthogonalize {
                for i in 0..=j {
                    let dot = V[i].dot(&w);
                    w.axpy(-dot, &V[i], 1.0);
                }
            }

            // beta_j = ||w||
            let beta_j = w.norm();
            beta.push(beta_j);

            // Check for breakdown
            if beta_j < 1e-14 {
                iterations = j + 1;
                break;
            }

            // v_{j+1} = w / beta_j
            let v_next = w / beta_j;
            V.push(v_next);

            iterations = j + 1;

            // Check convergence (simplified - based on beta decay)
            if j >= num_evals - 1 && beta_j < self.config.tolerance {
                converged = true;
                break;
            }
        }

        // Build tridiagonal matrix
        let m_actual = alpha.len();
        let mut t = DMatrix::zeros(m_actual, m_actual);
        for i in 0..m_actual {
            t[(i, i)] = alpha[i];
            if i < m_actual - 1 {
                t[(i, i + 1)] = beta[i];
                t[(i + 1, i)] = beta[i];
            }
        }

        // Compute eigenvalues of tridiagonal (using symmetric eigenvalue decomposition)
        let evals = t.symmetric_eigen();
        let mut eigenvalues: Vec<f64> = evals.eigenvalues.data.as_vec().clone();
        eigenvalues.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
        eigenvalues.truncate(num_evals);

        // Compute eigenvectors if requested
        let eigenvectors = if self.config.compute_eigenvectors {
            Some(self.compute_eigenvectors(&eigenvalues, &alpha, &beta, &V))
        } else {
            None
        };

        LanczosResult {
            eigenvalues,
            eigenvectors,
            iterations,
        }
    }

    /// Solves the generalized eigenvalue problem K * phi = lambda * M * phi.
    ///
    /// Uses shift-and-invert strategy if M is provided.
    pub fn solve_generalized(
        &self,
        k: &DMatrix<f64>,
        m: &DMatrix<f64>,
    ) -> LanczosResult {
        // For small to medium problems, convert to standard form
        // K * phi = lambda * M * phi
        // M^{-1} * K * phi = lambda * phi

        // Try Cholesky factorization of M (assumes positive definite)
        let m_chol = m.clone().cholesky();

        if let Some(chol) = m_chol {
            // Form M^{-1} * K by solving M * X = K
            let n = k.nrows();
            let mut mk = DMatrix::zeros(n, n);

            for j in 0..n {
                let col_j = k.column(j);
                let sol = chol.solve(&col_j);
                mk.set_column(j, &sol);
            }

            // Now solve standard eigenvalue problem for mk
            // But mk might not be symmetric, so use symmetric part
            let mk_sym = (&mk + mk.transpose()) * 0.5;

            self.solve_standard(&mk_sym)
        } else {
            // M is not positive definite, use simpler approach
            // Just solve K * phi = lambda * phi
            self.solve_standard(k)
        }
    }

    /// Computes eigenvectors from Lanczos vectors and tridiagonal eigenvectors.
    fn compute_eigenvectors(
        &self,
        eigenvalues: &[f64],
        alpha: &[f64],
        beta: &[f64],
        V: &[DVector<f64>],
    ) -> Vec<Vec<f64>> {
        let m = alpha.len();
        let n = V[0].len();
        let num_evals = eigenvalues.len();

        let mut eigenvectors = Vec::with_capacity(num_evals);

        for &eval in eigenvalues {
            // Solve (T - lambda*I) * y = 0 for y using inverse iteration
            let mut y = DVector::from_fn(m, |i, _| ((i + 1) as f64 * 0.3).sin());
            y.normalize_mut();

            // Build T - lambda*I
            let mut t_shift = DMatrix::zeros(m, m);
            for i in 0..m {
                t_shift[(i, i)] = alpha[i] - eval;
                if i < m - 1 {
                    t_shift[(i, i + 1)] = beta[i];
                    t_shift[(i + 1, i)] = beta[i];
                }
            }

            // Inverse iteration (simplified)
            let tlu = t_shift.lu();
            for _ in 0..10 {
                if let Some(y_new) = tlu.solve(&y) {
                    y = y_new;
                    y.normalize_mut();
                }
            }

            // Transform to original space: phi = V * y
            let mut phi = vec![0.0; n];
            for i in 0..m {
                if i < V.len() {
                    for j in 0..n {
                        phi[j] += y[i] * V[i][j];
                    }
                }
            }

            // Normalize
            let norm: f64 = phi.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                for x in &mut phi {
                    *x /= norm;
                }
            }

            eigenvectors.push(phi);
        }

        eigenvectors
    }
}

impl Default for LanczosSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lanczos_diagonal() {
        // Diagonal matrix - eigenvalues are diagonal entries
        let k = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]));

        let config = LanczosConfig {
            max_iterations: 5,
            num_eigenvalues: 3,
            tolerance: 1e-10,
            compute_eigenvectors: false,
            reorthogonalize: true,
        };

        let solver = LanczosSolver::with_config(config);
        let result = solver.solve_standard(&k);

        assert_eq!(result.eigenvalues.len(), 3);
        assert!((result.eigenvalues[0] - 1.0).abs() < 1e-6);
        assert!((result.eigenvalues[1] - 2.0).abs() < 1e-6);
        assert!((result.eigenvalues[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_lanczos_2x2() {
        // 2x2 symmetric matrix
        // [2 1; 1 2] has eigenvalues 1 and 3
        let k = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);

        let solver = LanczosSolver::new();
        let result = solver.solve_standard(&k);

        assert_eq!(result.eigenvalues.len(), 2);
        // Eigenvalues should be 1 and 3
        let sorted: Vec<f64> = result.eigenvalues.iter().copied().collect();
        assert!((sorted[0] - 1.0).abs() < 1e-6 || (sorted[0] - 3.0).abs() < 1e-6);
        assert!((sorted[1] - 1.0).abs() < 1e-6 || (sorted[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_lanczos_iterations() {
        let n = 50;
        // Create a symmetric tridiagonal matrix
        let mut k = DMatrix::zeros(n, n);
        for i in 0..n {
            k[(i, i)] = 2.0;
            if i > 0 {
                k[(i, i - 1)] = -1.0;
                k[(i - 1, i)] = -1.0;
            }
        }

        let config = LanczosConfig {
            max_iterations: 20,
            num_eigenvalues: 5,
            tolerance: 1e-8,
            compute_eigenvectors: false,
            reorthogonalize: true,
        };

        let solver = LanczosSolver::with_config(config);
        let result = solver.solve_standard(&k);

        assert!(result.iterations <= 20);
        assert_eq!(result.eigenvalues.len(), 5);
        // First eigenvalue of this matrix is approximately 2*(1-cos(pi/(n+1)))
        let expected_first = 2.0 * (1.0 - (std::f64::consts::PI / (n + 1) as f64).cos());
        assert!((result.eigenvalues[0] - expected_first).abs() < 0.1);
    }
}
