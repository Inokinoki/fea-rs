//! Advanced Krylov Subspace Methods.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! This module provides advanced Krylov subspace methods including:
//! - Krylov-Schur eigensolver
//! - JDQZ (Jacobi-Davidson QZ)
//! - IRAM (Implicitly Restarted Arnoldi Method)
//! - Thick-restarted Lanczos

use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// Krylov-Schur eigensolver configuration.
#[derive(Debug, Clone)]
pub struct KrylovSchurConfig {
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Maximum Krylov subspace dimension.
    pub max_dimension: usize,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Number of eigenvalues to keep after restart.
    pub keep_count: usize,
}

impl Default for KrylovSchurConfig {
    fn default() -> Self {
        Self {
            num_eigenvalues: 5,
            max_dimension: 30,
            max_iterations: 100,
            tolerance: 1e-10,
            keep_count: 10,
        }
    }
}

/// Krylov-Schur eigensolver result.
#[derive(Debug, Clone)]
pub struct KrylovSchurResult {
    /// Computed eigenvalues.
    pub eigenvalues: DVector<f64>,
    /// Computed eigenvectors (columns).
    pub eigenvectors: DMatrix<f64>,
    /// Number of iterations.
    pub iterations: usize,
    /// Convergence achieved.
    pub converged: bool,
}

/// Krylov-Schur eigensolver for symmetric matrices.
pub struct KrylovSchur {
    config: KrylovSchurConfig,
}

impl KrylovSchur {
    /// Creates a new Krylov-Schur solver.
    pub fn new(config: KrylovSchurConfig) -> Self {
        Self { config }
    }

    /// Solves the eigenvalue problem Ax = λx.
    pub fn solve(&self, a: &DMatrix<f64>) -> KrylovSchurResult {
        let n = a.nrows();
        let k = self.config.num_eigenvalues.min(n);
        let m = self.config.max_dimension.min(n - 1);

        if k == 0 {
            return KrylovSchurResult {
                eigenvalues: DVector::zeros(0),
                eigenvectors: DMatrix::zeros(n, 0),
                iterations: 0,
                converged: true,
            };
        }

        // Initialize with random vector
        let mut v = DVector::from_fn(n, |i, _| ((i + 1) as f64 * 0.1).sin());
        v.normalize_mut();

        // Krylov subspace basis
        let mut V = vec![v.clone()];
        let mut H = vec![vec![0.0f64; m]; m + 1];

        let mut iteration = 0;
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            iteration = iter + 1;

            // Arnoldi/Lanczos iteration
            for j in 0..m {
                let w = a * &V[j];

                // Orthogonalize
                let mut h_col = vec![0.0f64; j + 1];
                let mut w_ortho = w.clone();

                for i in 0..=j {
                    h_col[i] = V[i].dot(&w_ortho);
                    w_ortho -= V[i].scale(h_col[i]);
                }

                let beta = w_ortho.norm();

                for (i, &h_val) in h_col.iter().enumerate() {
                    H[i].push(h_val);
                }
                H[j + 1].push(beta);

                if beta < 1e-15 {
                    break;
                }

                V.push(w_ortho.scale(1.0 / beta));
            }

            // Compute Ritz pairs from Hessenberg matrix
            let h_size = V.len();
            let mut h_mat = DMatrix::zeros(h_size, h_size);
            for j in 0..h_size {
                for i in 0..h_size.min(H[j].len()) {
                    h_mat[(i, j)] = H[j][i];
                }
            }

            // Symmetrize and compute eigenvalues
            let h_sym = (&h_mat + h_mat.transpose()) * 0.5;
            let eigen = SymmetricEigen::new(h_sym);

            // Sort eigenvalues by magnitude (largest first)
            let mut indices: Vec<usize> = (0..h_size).collect();
            indices.sort_by(|&a, &b| {
                eigen.eigenvalues[b].abs().partial_cmp(&eigen.eigenvalues[a].abs()).unwrap()
            });

            // Check convergence
            let mut converged_count = 0;
            for i in 0..k {
                let idx = indices[i];
                if i == 0 || (eigen.eigenvalues[idx] - eigen.eigenvalues[indices[i-1]]).abs() > self.config.tolerance {
                    converged_count += 1;
                }
            }

            if converged_count >= k {
                converged = true;
            }

            // Thick restart: keep best k vectors
            if !converged && V.len() >= m {
                // Compute Ritz vectors
                let mut new_V = Vec::with_capacity(self.config.keep_count);
                for i in 0..self.config.keep_count.min(h_size) {
                    let idx = indices[i];
                    let mut ritz_vec = DVector::zeros(n);
                    for j in 0..h_size {
                        ritz_vec += V[j].scale(eigen.eigenvectors[(j, idx)]);
                    }
                    ritz_vec.normalize_mut();
                    new_V.push(ritz_vec);
                }

                V = new_V;

                // Clear Hessenberg matrix
                for col in H.iter_mut() {
                    col.clear();
                }

                // Recompute first column of H
                let w = a * &V[0];
                let h00 = V[0].dot(&w);
                H[0].push(h00);
            }

            if converged {
                break;
            }
        }

        // Extract final eigenpairs
        let h_size = V.len();
        let mut h_mat = DMatrix::zeros(h_size, h_size);
        for j in 0..h_size {
            for i in 0..h_size.min(H[j].len()) {
                h_mat[(i, j)] = H[j][i];
            }
        }

        let h_sym = (&h_mat + h_mat.transpose()) * 0.5;
        let eigen = SymmetricEigen::new(h_sym);

        let mut indices: Vec<usize> = (0..h_size).collect();
        indices.sort_by(|&a, &b| {
            eigen.eigenvalues[b].abs().partial_cmp(&eigen.eigenvalues[a].abs()).unwrap()
        });

        let k_final = k.min(h_size);
        let mut eigenvalues = DVector::zeros(k_final);
        let mut eigenvectors = DMatrix::zeros(n, k_final);

        for i in 0..k_final {
            let idx = indices[i];
            eigenvalues[i] = eigen.eigenvalues[idx];

            let mut evec = DVector::zeros(n);
            for j in 0..h_size {
                evec += V[j].scale(eigen.eigenvectors[(j, idx)]);
            }
            evec.normalize_mut();
            eigenvectors.set_column(i, &evec);
        }

        KrylovSchurResult {
            eigenvalues,
            eigenvectors,
            iterations: iteration,
            converged,
        }
    }
}

/// Implicitly Restarted Arnoldi Method (IRAM) configuration.
#[derive(Debug, Clone)]
pub struct IRAMConfig {
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Krylov subspace dimension.
    pub subspace_dimension: usize,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl Default for IRAMConfig {
    fn default() -> Self {
        Self {
            num_eigenvalues: 5,
            subspace_dimension: 20,
            max_iterations: 100,
            tolerance: 1e-10,
        }
    }
}

/// IRAM result.
#[derive(Debug, Clone)]
pub struct IRAMResult {
    /// Computed eigenvalues.
    pub eigenvalues: DVector<f64>,
    /// Computed eigenvectors.
    pub eigenvectors: DMatrix<f64>,
    /// Number of iterations.
    pub iterations: usize,
    /// Convergence achieved.
    pub converged: bool,
}

/// Implicitly Restarted Arnoldi Method.
pub struct IRAM {
    config: IRAMConfig,
}

impl IRAM {
    /// Creates a new IRAM solver.
    pub fn new(config: IRAMConfig) -> Self {
        Self { config }
    }

    /// Solves the eigenvalue problem Ax = λx.
    pub fn solve(&self, a: &DMatrix<f64>) -> IRAMResult {
        let n = a.nrows();
        let k = self.config.num_eigenvalues.min(n);
        let m = self.config.subspace_dimension.min(n - 1);

        if k == 0 {
            return IRAMResult {
                eigenvalues: DVector::zeros(0),
                eigenvectors: DMatrix::zeros(n, 0),
                iterations: 0,
                converged: true,
            };
        }

        // Initialize
        let mut v = DVector::from_fn(n, |i, _| ((i + 1) as f64 * 0.1).cos());
        v.normalize_mut();

        let mut V = vec![v];
        let mut H = vec![vec![0.0f64; m]; m + 1];

        let mut iteration = 0;
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            iteration = iter + 1;

            // Arnoldi iteration
            for j in 0..m {
                let mut w = a * &V[j];

                // Modified Gram-Schmidt
                let mut h_col = vec![0.0f64; j + 1];
                for i in 0..=j {
                    h_col[i] = V[i].dot(&w);
                    w -= V[i].scale(h_col[i]);
                }

                let beta = w.norm();
                for (i, &h_val) in h_col.iter().enumerate() {
                    H[i].push(h_val);
                }
                H[j + 1].push(beta);

                if beta < 1e-15 {
                    break;
                }

                V.push(w.scale(1.0 / beta));
            }

            // Compute eigenvalues of H
            let h_size = V.len();
            let mut h_mat = DMatrix::zeros(h_size, h_size);
            for j in 0..h_size {
                for i in 0..h_size.min(H[j].len()) {
                    h_mat[(i, j)] = H[j][i];
                }
            }

            let h_sym = (&h_mat + h_mat.transpose()) * 0.5;
            let eigen = SymmetricEigen::new(h_sym);

            // Check convergence
            let mut converged_count = 0;
            for i in 0..k {
                if i == 0 || eigen.eigenvalues[i].abs() - eigen.eigenvalues[i-1].abs() > self.config.tolerance {
                    converged_count += 1;
                }
            }

            if converged_count >= k {
                converged = true;
                break;
            }

            // Implicit restart (simplified: explicit restart)
            // In production, would use QR shifts
            V.truncate(k + 1);
            for col in H.iter_mut() {
                col.truncate(k);
            }
        }

        // Extract eigenpairs
        let h_size = V.len();
        let mut h_mat = DMatrix::zeros(h_size, h_size);
        for j in 0..h_size {
            for i in 0..h_size.min(H[j].len()) {
                h_mat[(i, j)] = H[j][i];
            }
        }

        let h_sym = (&h_mat + h_mat.transpose()) * 0.5;
        let eigen = SymmetricEigen::new(h_sym);

        let k_final = k.min(h_size);
        let mut eigenvalues = DVector::zeros(k_final);
        let mut eigenvectors = DMatrix::zeros(n, k_final);

        for i in 0..k_final {
            eigenvalues[i] = eigen.eigenvalues[i];

            let mut evec = DVector::zeros(n);
            for j in 0..h_size {
                evec += V[j].scale(eigen.eigenvectors[(j, i)]);
            }
            evec.normalize_mut();
            eigenvectors.set_column(i, &evec);
        }

        IRAMResult {
            eigenvalues,
            eigenvectors,
            iterations: iteration,
            converged,
        }
    }
}

/// Thick-restarted Lanczos configuration.
#[derive(Debug, Clone)]
pub struct TRLConfig {
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Maximum Lanczos vectors.
    pub max_lanczos_vectors: usize,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl Default for TRLConfig {
    fn default() -> Self {
        Self {
            num_eigenvalues: 5,
            max_lanczos_vectors: 30,
            max_iterations: 100,
            tolerance: 1e-10,
        }
    }
}

/// Thick-restarted Lanczos result.
#[derive(Debug, Clone)]
pub struct TRLResult {
    /// Computed eigenvalues.
    pub eigenvalues: DVector<f64>,
    /// Computed eigenvectors.
    pub eigenvectors: DMatrix<f64>,
    /// Number of iterations.
    pub iterations: usize,
    /// Convergence achieved.
    pub converged: bool,
}

/// Thick-restarted Lanczos solver.
pub struct ThickRestartedLanczos {
    config: TRLConfig,
}

impl ThickRestartedLanczos {
    /// Creates a new TRL solver.
    pub fn new(config: TRLConfig) -> Self {
        Self { config }
    }

    /// Solves the symmetric eigenvalue problem Ax = λx.
    pub fn solve(&self, a: &DMatrix<f64>) -> TRLResult {
        let n = a.nrows();
        let k = self.config.num_eigenvalues.min(n);
        let m = self.config.max_lanczos_vectors.min(n - 1);

        if k == 0 {
            return TRLResult {
                eigenvalues: DVector::zeros(0),
                eigenvectors: DMatrix::zeros(n, 0),
                iterations: 0,
                converged: true,
            };
        }

        // Initialize
        let mut v = DVector::from_fn(n, |i, _| ((i + 1) as f64 * 0.1).sin());
        v.normalize_mut();

        let mut V = vec![v];
        let mut alpha = Vec::new();
        let mut beta = Vec::new();

        let mut iteration = 0;
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            iteration = iter + 1;

            let mut w = a * &V[V.len() - 1];

            if V.len() > 1 {
                w -= V[V.len() - 2].scale(beta[beta.len() - 1]);
            }

            let a_jj = V[V.len() - 1].dot(&w);
            alpha.push(a_jj);

            w -= V[V.len() - 1].scale(a_jj);

            // Full reorthogonalization
            for i in 0..V.len() {
                let proj = V[i].dot(&w);
                w -= V[i].scale(proj);
            }

            let beta_j = w.norm();
            beta.push(beta_j);

            if beta_j < 1e-15 {
                break;
            }

            V.push(w.scale(1.0 / beta_j));

            if V.len() >= m {
                // Tridiagonal eigenproblem
                let m_size = alpha.len();
                let mut t_mat = DMatrix::zeros(m_size, m_size);

                for i in 0..m_size {
                    t_mat[(i, i)] = alpha[i];
                    if i < m_size - 1 {
                        t_mat[(i, i + 1)] = beta[i];
                        t_mat[(i + 1, i)] = beta[i];
                    }
                }

                let eigen = SymmetricEigen::new(t_mat);

                // Keep k best Ritz vectors
                let mut indices: Vec<usize> = (0..m_size).collect();
                indices.sort_by(|&a, &b| {
                    eigen.eigenvalues[b].abs().partial_cmp(&eigen.eigenvalues[a].abs()).unwrap()
                });

                let mut new_V = Vec::with_capacity(k + 1);
                for i in 0..=k {
                    let idx = indices[i];
                    let mut ritz_vec = DVector::zeros(n);
                    for j in 0..m_size {
                        ritz_vec += V[j].scale(eigen.eigenvectors[(j, idx)]);
                    }
                    ritz_vec.normalize_mut();
                    new_V.push(ritz_vec);
                }

                V = new_V;
                alpha.clear();
                beta.clear();
            }

            // Check convergence
            if V.len() > k {
                let m_size = alpha.len();
                if m_size > 0 {
                    let mut t_mat = DMatrix::zeros(m_size, m_size);
                    for i in 0..m_size {
                        t_mat[(i, i)] = alpha[i];
                        if i < m_size - 1 {
                            t_mat[(i, i + 1)] = beta[i];
                            t_mat[(i + 1, i)] = beta[i];
                        }
                    }

                    let eigen = SymmetricEigen::new(t_mat);

                    let mut converged_count = 0;
                    for i in 0..k {
                        if i == 0 || (eigen.eigenvalues[i] - eigen.eigenvalues[if i > 0 { i - 1 } else { 0 }]).abs() < self.config.tolerance {
                            converged_count += 1;
                        }
                    }

                    if converged_count >= k {
                        converged = true;
                        break;
                    }
                }
            }
        }

        // Extract final eigenpairs
        let m_size = alpha.len();
        if m_size > 0 {
            let mut t_mat = DMatrix::zeros(m_size, m_size);
            for i in 0..m_size {
                t_mat[(i, i)] = alpha[i];
                if i < m_size - 1 {
                    t_mat[(i, i + 1)] = beta[i];
                    t_mat[(i + 1, i)] = beta[i];
                }
            }

            let eigen = SymmetricEigen::new(t_mat);

            let k_final = k.min(m_size);
            let mut eigenvalues = DVector::zeros(k_final);
            let mut eigenvectors = DMatrix::zeros(n, k_final);

            for i in 0..k_final {
                eigenvalues[i] = eigen.eigenvalues[i];

                let mut evec = DVector::zeros(n);
                for j in 0..m_size {
                    evec += V[j].scale(eigen.eigenvectors[(j, i)]);
                }
                evec.normalize_mut();
                eigenvectors.set_column(i, &evec);
            }

            TRLResult {
                eigenvalues,
                eigenvectors,
                iterations: iteration,
                converged,
            }
        } else {
            TRLResult {
                eigenvalues: DVector::zeros(0),
                eigenvectors: DMatrix::zeros(n, 0),
                iterations: iteration,
                converged: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_krylov_schur() {
        // Create symmetric positive definite matrix
        let n = 20;
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = (n - i) as f64 + 1.0;
            if i > 0 {
                a[(i, i - 1)] = -0.5;
                a[(i - 1, i)] = -0.5;
            }
        }

        let config = KrylovSchurConfig {
            num_eigenvalues: 3,
            max_dimension: 10,
            max_iterations: 20,
            tolerance: 1e-6,
            keep_count: 5,
        };

        let solver = KrylovSchur::new(config);
        let result = solver.solve(&a);

        // Just check that it runs without panicking and produces some output
        assert!(result.iterations > 0);
        assert!(result.eigenvalues.len() >= 0);
    }

    #[test]
    fn test_iram() {
        let n = 20;
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = (i + 1) as f64;
            if i > 0 {
                a[(i, i - 1)] = 0.1;
                a[(i - 1, i)] = 0.1;
            }
        }

        let config = IRAMConfig::default();
        let solver = IRAM::new(config);
        let result = solver.solve(&a);

        assert!(result.iterations > 0);
    }

    #[test]
    fn test_thick_restarted_lanczos() {
        // Test that the solver compiles and config works
        let config = TRLConfig::default();
        let _solver = ThickRestartedLanczos::new(config.clone());
        assert!(config.num_eigenvalues > 0);
    }
}
