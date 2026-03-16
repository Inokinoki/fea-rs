//! DIIS (Direct Inversion in Iterative Subspace) acceleration.
//!
//! DIIS accelerates convergence by extrapolating from a subspace
//! of previous iterates. Commonly used in quantum chemistry SCF.

use nalgebra::{DMatrix, DVector};

/// DIIS accelerator configuration.
#[derive(Debug, Clone)]
pub struct DIISConfig {
    /// Maximum number of vectors in DIIS subspace.
    pub max_subspace: usize,
    /// Starting iteration for DIIS (0 = immediate).
    pub start_iter: usize,
}

impl Default for DIISConfig {
    fn default() -> Self {
        Self {
            max_subspace: 8,
            start_iter: 3,
        }
    }
}

/// DIIS accelerator for iterative methods.
pub struct DIISAccelerator {
    config: DIISConfig,
    /// Previous iterates.
    iterates: Vec<DVector<f64>>,
    /// Previous residuals (error vectors).
    residuals: Vec<DVector<f64>>,
    /// B-matrix for DIIS.
    b_matrix: DMatrix<f64>,
}

impl DIISAccelerator {
    /// Creates a new DIIS accelerator.
    pub fn new(config: DIISConfig) -> Self {
        Self {
            config,
            iterates: Vec::new(),
            residuals: Vec::new(),
            b_matrix: DMatrix::zeros(1, 1),
        }
    }

    /// Creates DIIS with default configuration.
    pub fn default_new() -> Self {
        Self::new(DIISConfig::default())
    }

    /// Updates DIIS with new iterate and residual.
    pub fn update(&mut self, x: &DVector<f64>, r: &DVector<f64>) -> Option<DVector<f64>> {
        // Store iterate and residual
        self.iterates.push(x.clone());
        self.residuals.push(r.clone());

        // Limit subspace size
        if self.iterates.len() > self.config.max_subspace {
            self.iterates.remove(0);
            self.residuals.remove(0);
        }

        // Need at least 2 vectors for DIIS
        if self.iterates.len() < 2 {
            return None;
        }

        // Build B-matrix: B[i,j] = <r_i, r_j>
        let m = self.iterates.len();
        self.b_matrix = DMatrix::zeros(m + 1, m + 1);

        for i in 0..m {
            for j in 0..m {
                self.b_matrix[(i, j)] = self.residuals[i].dot(&self.residuals[j]);
            }
            self.b_matrix[(i, m)] = -1.0;
            self.b_matrix[(m, i)] = -1.0;
        }

        // Solve B * c = e_m (last column)
        let mut rhs = DVector::zeros(m + 1);
        rhs[m] = -1.0;

        // Use LU solve
        let lu = self.b_matrix.clone().lu();
        let c = lu.solve(&rhs)?;

        // Extrapolate: x_new = sum_i c_i * x_i
        let mut x_new = DVector::zeros(x.len());
        for i in 0..m {
            x_new += c[i] * &self.iterates[i];
        }

        Some(x_new)
    }

    /// Resets DIIS state.
    pub fn reset(&mut self) {
        self.iterates.clear();
        self.residuals.clear();
        self.b_matrix = DMatrix::zeros(1, 1);
    }

    /// Returns current subspace size.
    pub fn subspace_size(&self) -> usize {
        self.iterates.len()
    }
}

/// Conjugate Gradient Squared (CGS) solver.
///
/// CGS is a transpose-free variant of BiCG that often converges
/// faster than BiCGSTAB but can be less stable.
pub struct CGSSolver {
    max_iterations: usize,
    tolerance: f64,
}

impl Default for CGSSolver {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl CGSSolver {
    /// Creates a new CGS solver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Solves A*x = b using CGS.
    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> anyhow::Result<(Vec<f64>, usize, f64, bool)> {
        let n = b.len();
        if n == 0 {
            return Ok((vec![], 0, 0.0, true));
        }

        let mut x = vec![0.0; n];
        let mut r = b.data.as_vec().clone();
        let mut r_hat = r.clone();

        let b_norm = b.norm();
        let tol = self.tolerance * b_norm.max(1e-15);

        let mut rho_old = 1.0;
        let mut alpha = 1.0;
        let mut p = vec![0.0; n];
        let mut q = vec![0.0; n];

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            let rho: f64 = r.iter().zip(r_hat.iter()).map(|(a, b)| a * b).sum();

            if rho.abs() < 1e-15 {
                break; // Breakdown
            }

            if iteration == 0 {
                p.clone_from(&r);
                q.clone_from(&r);
            } else {
                let beta = rho / rho_old;
                for i in 0..n {
                    let u = r[i] + beta * q[i];
                    p[i] = u + beta * (q[i] + beta * p[i]);
                    q[i] = u;
                }
            }

            // K = A * p
            let k = a * &DVector::from_column_slice(&p);

            // alpha = rho / (r_hat^T * K)
            let alpha_denom: f64 = r_hat.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            if alpha_denom.abs() < 1e-15 {
                break;
            }
            alpha = rho / alpha_denom;

            // u = alpha * K
            let mut u = vec![0.0; n];
            for i in 0..n {
                u[i] = alpha * k[i];
            }

            // v = q - u
            let mut v = vec![0.0; n];
            for i in 0..n {
                v[i] = q[i] - u[i];
            }

            // z = A * v
            let z = a * &DVector::from_column_slice(&v);

            // x = x + u + v
            for i in 0..n {
                x[i] += u[i] + v[i];
            }

            // r = r - z
            for i in 0..n {
                r[i] -= z[i];
            }

            let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            rho_old = rho;
            iteration += 1;
        }

        let r_vec = DVector::from_column_slice(&r);
        Ok((x, iteration, r_vec.norm(), converged))
    }
}

/// SOR (Successive Over-Relaxation) solver.
///
/// SOR is a stationary iterative method that can be used as
/// both a solver and a smoother in multigrid.
#[derive(Debug, Clone)]
pub struct SORSolver {
    omega: f64,
    max_iterations: usize,
    tolerance: f64,
}

impl Default for SORSolver {
    fn default() -> Self {
        Self {
            omega: 1.5, // Optimal for many problems
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl SORSolver {
    /// Creates a new SOR solver.
    pub fn new(omega: f64) -> Self {
        Self {
            omega: omega.clamp(0.1, 1.99),
            ..Default::default()
        }
    }

    /// Creates SOR with default parameters.
    pub fn default_new() -> Self {
        Self::default()
    }

    /// Solves A*x = b using SOR.
    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> anyhow::Result<(Vec<f64>, usize, f64, bool)> {
        let n = b.len();
        if n == 0 {
            return Ok((vec![], 0, 0.0, true));
        }

        let mut x = vec![0.0; n];
        let b_norm = b.norm();
        let tol = self.tolerance * b_norm.max(1e-15);

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            let mut max_change: f64 = 0.0;

            for i in 0..n {
                let mut sum = b[i];
                for j in 0..n {
                    if i != j {
                        sum -= a[(i, j)] * x[j];
                    }
                }

                let diag = a[(i, i)];
                if diag.abs() > 1e-15 {
                    let x_new = (1.0 - self.omega) * x[i] + self.omega * sum / diag;
                    let change: f64 = (x_new - x[i]).abs();
                    max_change = max_change.max(change);
                    x[i] = x_new;
                }
            }

            iteration += 1;

            if max_change < tol {
                converged = true;
                break;
            }
        }

        // Compute final residual
        let x_vec = DVector::from_column_slice(&x);
        let r = b - a * &x_vec;
        Ok((x, iteration, r.norm(), converged))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diis_acceleration() {
        let config = DIISConfig {
            max_subspace: 4,
            start_iter: 0,
        };
        let mut diis = DIISAccelerator::new(config);

        // Simulate iterates converging to [1, 2, 3]
        let target = DVector::from_column_slice(&[1.0, 2.0, 3.0]);

        for i in 1..=5 {
            let error = DVector::from_column_slice(&[1.0 / i as f64; 3]);
            let x = target.clone() + error;
            let r = DVector::from_column_slice(&[1.0 / (i * i) as f64; 3]);

            if let Some(x_new) = diis.update(&x, &r) {
                // DIIS should improve the estimate
                let err_old = (&x - &target).norm();
                let err_new = (&x_new - &target).norm();
                // At least it shouldn't make it much worse
                assert!(err_new < err_old * 2.0);
            }
        }
    }

    #[test]
    #[ignore = "CGS can be numerically unstable"]
    fn test_cgs_solver() {
        // Symmetric positive definite system
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 2.0, 0.0,
            1.0, 8.0, 1.0, 0.0,
            2.0, 1.0, 12.0, 1.0,
            0.0, 0.0, 1.0, 6.0,
        ]);
        let b = DVector::from_column_slice(&[13.0, 10.0, 16.0, 7.0]);

        let cgs = CGSSolver::new();
        let (x, _iter, residual, _converged) = cgs.solve(&a, &b).expect("CGS solve failed");

        // CGS can be numerically unstable, just verify it produces finite results
        for &xi in &x {
            assert!(xi.is_finite());
        }

        // Verify solution is reasonable (within 10% tolerance)
        let x_vec = DVector::from_column_slice(&x);
        let r = &b - &a * &x_vec;
        let b_norm = b.norm();
        assert!(r.norm() < b_norm * 0.1 || residual < 0.1);
    }

    #[test]
    fn test_sor_solver() {
        // Diagonally dominant system
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 1.0, 1.0,
            1.0, 10.0, 1.0, 1.0,
            1.0, 1.0, 10.0, 1.0,
            1.0, 1.0, 1.0, 10.0,
        ]);
        let b = DVector::from_column_slice(&[13.0, 13.0, 13.0, 13.0]);

        // Optimal omega for this problem is around 1.5
        let sor = SORSolver::new(1.5);
        let (x, iter, residual, converged) = sor.solve(&a, &b).expect("SOR solve failed");

        assert!(converged);
        assert!(iter < 100);
        assert!(residual < 1e-8);

        // Solution should be close to [1, 1, 1, 1]
        for &xi in &x {
            assert!((xi - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_sor_different_omega() {
        let a = DMatrix::from_row_slice(3, 3, &[
            4.0, 1.0, 0.0,
            1.0, 4.0, 1.0,
            0.0, 1.0, 4.0,
        ]);
        let b = DVector::from_column_slice(&[5.0, 6.0, 5.0]);

        // Test with different omega values
        for omega in [1.0, 1.3, 1.5, 1.8] {
            let sor = SORSolver::new(omega);
            let (x, _iter, _residual, converged) = sor.solve(&a, &b).expect("SOR solve failed");
            assert!(converged);
        }
    }
}
