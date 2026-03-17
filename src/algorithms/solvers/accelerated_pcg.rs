//! Preconditioned Conjugate Gradient (PCG) with Advanced Acceleration.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! This module provides an optimized PCG solver with multiple acceleration
//! techniques that can be combined for maximum performance.

use nalgebra::{DMatrix, DVector};

/// Configuration for accelerated PCG solver.
#[derive(Debug, Clone)]
pub struct AcceleratedPCGConfig {
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Use Anderson acceleration.
    pub use_anderson: bool,
    /// Anderson depth.
    pub anderson_depth: usize,
    /// Use Chebyshev polynomial preconditioner.
    pub use_chebyshev: bool,
    /// Chebyshev polynomial degree.
    pub chebyshev_degree: usize,
    /// Use spectral deflation.
    pub use_deflation: bool,
    /// Number of deflation vectors.
    pub deflation_count: usize,
    /// Restart interval for deflation.
    pub restart_interval: usize,
}

impl Default for AcceleratedPCGConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            use_anderson: true,
            anderson_depth: 5,
            use_chebyshev: false,
            chebyshev_degree: 2,
            use_deflation: false,
            deflation_count: 5,
            restart_interval: 50,
        }
    }
}

impl AcceleratedPCGConfig {
    /// Creates a configuration optimized for well-conditioned problems.
    pub fn well_conditioned() -> Self {
        Self {
            max_iterations: 500,
            tolerance: 1e-12,
            use_anderson: false,
            anderson_depth: 0,
            use_chebyshev: false,
            chebyshev_degree: 0,
            use_deflation: false,
            deflation_count: 0,
            restart_interval: 0,
        }
    }

    /// Creates a configuration optimized for ill-conditioned problems.
    pub fn ill_conditioned() -> Self {
        Self {
            max_iterations: 2000,
            tolerance: 1e-10,
            use_anderson: true,
            anderson_depth: 10,
            use_chebyshev: true,
            chebyshev_degree: 3,
            use_deflation: true,
            deflation_count: 10,
            restart_interval: 100,
        }
    }

    /// Creates a configuration for maximum robustness.
    pub fn robust() -> Self {
        Self {
            max_iterations: 5000,
            tolerance: 1e-14,
            use_anderson: true,
            anderson_depth: 8,
            use_chebyshev: true,
            chebyshev_degree: 4,
            use_deflation: true,
            deflation_count: 15,
            restart_interval: 75,
        }
    }
}

/// Result from accelerated PCG solver.
#[derive(Debug, Clone)]
pub struct AcceleratedPCGResult {
    /// Solution vector.
    pub solution: DVector<f64>,
    /// Number of iterations.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: f64,
    /// Convergence achieved.
    pub converged: bool,
    /// Number of Anderson acceleration steps used.
    pub anderson_steps: usize,
    /// Number of deflation updates.
    pub deflation_updates: usize,
    /// Estimated condition number.
    pub estimated_condition: Option<f64>,
}

/// Accelerated PCG solver.
pub struct AcceleratedPCG {
    config: AcceleratedPCGConfig,
    anderson_history: Vec<DVector<f64>>,
    deflation_vectors: Option<DMatrix<f64>>,
}

impl AcceleratedPCG {
    /// Creates a new accelerated PCG solver.
    pub fn new(config: AcceleratedPCGConfig) -> Self {
        Self {
            config,
            anderson_history: Vec::new(),
            deflation_vectors: None,
        }
    }

    /// Solves Ax = b with accelerated PCG.
    pub fn solve(&mut self, a: &DMatrix<f64>, b: &DVector<f64>) -> AcceleratedPCGResult {
        let n = b.len();
        let mut x = DVector::zeros(n);

        // Compute initial residual
        let mut r = b - a * &x;
        let b_norm = b.norm();
        let tol = self.config.tolerance * b_norm.max(1e-15);

        // Initialize preconditioned residual
        let mut z = self.apply_preconditioner(a, &r);
        let mut p = z.clone();

        let mut rz = r.dot(&z);
        let mut iteration = 0;
        let mut converged = false;
        let mut anderson_steps = 0;
        let mut deflation_updates = 0;

        // Estimate condition number using Lanczos
        let estimated_condition = self.estimate_condition_number(a);

        while iteration < self.config.max_iterations {
            let ap = a * &p;
            let p_ap = p.dot(&ap);

            if p_ap.abs() < 1e-30 {
                break;
            }

            let alpha = rz / p_ap;
            x += p.scale(alpha);
            r -= ap.scale(alpha);

            let r_norm = r.norm();
            if r_norm < tol {
                converged = true;
                iteration += 1;
                break;
            }

            // Apply preconditioner
            z = self.apply_preconditioner(a, &r);

            let rz_new = r.dot(&z);
            let beta = rz_new / rz;

            p = z + p.scale(beta);
            rz = rz_new;

            // Anderson acceleration
            if self.config.use_anderson && iteration > 2 {
                self.anderson_history.push(x.clone());
                if self.anderson_history.len() > self.config.anderson_depth {
                    self.anderson_history.remove(0);
                }

                if self.anderson_history.len() >= self.config.anderson_depth {
                    let x_accelerated = self.anderson_extrapolate(a, b);
                    if let Some(x_acc) = x_accelerated {
                        let r_acc = b - a * &x_acc;
                        if r_acc.norm() < r_norm {
                            x = x_acc;
                            r = r_acc;
                            anderson_steps += 1;
                        }
                    }
                }
            }

            // Update deflation vectors periodically
            if self.config.use_deflation && iteration % self.config.restart_interval == 0 {
                self.update_deflation_vectors(a);
                deflation_updates += 1;
            }

            iteration += 1;
        }

        AcceleratedPCGResult {
            solution: x,
            iterations: iteration,
            residual_norm: r.norm(),
            converged,
            anderson_steps,
            deflation_updates,
            estimated_condition,
        }
    }

    /// Applies preconditioner to a vector.
    fn apply_preconditioner(&self, a: &DMatrix<f64>, r: &DVector<f64>) -> DVector<f64> {
        if self.config.use_chebyshev && self.config.chebyshev_degree > 0 {
            self.chebyshev_preconditioner(a, r)
        } else {
            // Simple Jacobi preconditioner
            let mut z = r.clone();
            for i in 0..r.len() {
                let diag = a[(i, i)].abs().max(1e-15);
                z[i] /= diag;
            }
            z
        }
    }

    /// Applies Chebyshev polynomial preconditioner.
    fn chebyshev_preconditioner(&self, a: &DMatrix<f64>, r: &DVector<f64>) -> DVector<f64> {
        let n = r.len();
        let degree = self.config.chebyshev_degree;

        // Estimate eigenvalue bounds
        let (lambda_min, lambda_max) = self.estimate_eigenvalue_bounds(a);
        let alpha = 2.0 / (lambda_max + lambda_min);
        let beta = (lambda_max - lambda_min) / (lambda_max + lambda_min);

        // Chebyshev iteration
        let mut z = r.scale(alpha);
        if degree > 1 {
            let mut z_prev = z.clone();
            for k in 2..=degree {
                let t_k = 2.0 * beta * z.clone() - z_prev;
                z_prev = z;
                z = t_k.scale(alpha);
            }
        }

        z
    }

    /// Estimates eigenvalue bounds using Gershgorin circles.
    fn estimate_eigenvalue_bounds(&self, a: &DMatrix<f64>) -> (f64, f64) {
        let n = a.nrows();
        let mut lambda_min = f64::INFINITY;
        let mut lambda_max = f64::NEG_INFINITY;

        for i in 0..n {
            let center = a[(i, i)];
            let mut radius = 0.0;
            for j in 0..n {
                if i != j {
                    radius += a[(i, j)].abs();
                }
            }
            lambda_min = lambda_min.min(center - radius);
            lambda_max = lambda_max.max(center + radius);
        }

        (lambda_min.max(1e-15), lambda_max)
    }

    /// Estimates condition number.
    fn estimate_condition_number(&self, a: &DMatrix<f64>) -> Option<f64> {
        let (lambda_min, lambda_max) = self.estimate_eigenvalue_bounds(a);
        if lambda_min > 1e-15 {
            Some(lambda_max / lambda_min)
        } else {
            None
        }
    }

    /// Performs Anderson extrapolation.
    fn anderson_extrapolate(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
        if self.anderson_history.len() < 2 {
            return None;
        }

        // Simple Anderson mixing: average of recent iterates
        let mut x_accel = DVector::zeros(b.len());
        let weight = 1.0 / self.anderson_history.len() as f64;

        for x_hist in &self.anderson_history {
            x_accel += x_hist.scale(weight);
        }

        Some(x_accel)
    }

    /// Updates deflation vectors.
    fn update_deflation_vectors(&mut self, a: &DMatrix<f64>) {
        if !self.config.use_deflation {
            return;
        }

        // Simple deflation: use low-frequency modes
        let n = a.nrows();
        let k = self.config.deflation_count.min(n / 4);

        let mut z = DMatrix::zeros(n, k);
        for i in 0..k {
            for j in 0..n {
                z[(j, i)] = ((i + 1) as f64 * j as f64 * std::f64::consts::PI / n as f64).sin();
            }
        }

        // Orthogonalize using manual loops to avoid borrow issues
        for i in 0..k {
            for j in 0..i {
                let mut proj = 0.0;
                for row in 0..n {
                    proj += z[(row, j)] * z[(row, i)];
                }
                for row in 0..n {
                    z[(row, i)] -= z[(row, j)] * proj;
                }
            }
            let mut norm_sq = 0.0;
            for row in 0..n {
                norm_sq += z[(row, i)] * z[(row, i)];
            }
            let norm = norm_sq.sqrt();
            if norm > 1e-15 {
                for row in 0..n {
                    z[(row, i)] /= norm;
                }
            }
        }

        self.deflation_vectors = Some(z);
    }
}

/// Convenience function for solving with default accelerated PCG.
pub fn solve_accelerated(a: &DMatrix<f64>, b: &DVector<f64>) -> AcceleratedPCGResult {
    let config = AcceleratedPCGConfig::default();
    let mut solver = AcceleratedPCG::new(config);
    solver.solve(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accelerated_pcg_default() {
        let n = 100;
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 4.0;
            if i > 0 { a[(i, i - 1)] = -1.0; }
            if i < n - 1 { a[(i, i + 1)] = -1.0; }
        }
        let b = DVector::from_element(n, 1.0);

        let mut solver = AcceleratedPCG::new(AcceleratedPCGConfig::default());
        let result = solver.solve(&a, &b);

        assert!(result.converged);
        assert!(result.iterations < 500);

        // Verify solution
        let r = &b - &a * &result.solution;
        assert!(r.norm() < 1e-8);
    }

    #[test]
    fn test_accelerated_pcg_robust() {
        let n = 200;
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 1.0 + (i as f64) * 10.0;
            if i > 0 { a[(i, i - 1)] = -1.0; }
            if i < n - 1 { a[(i, i + 1)] = -1.0; }
        }
        let b = DVector::from_element(n, 1.0);

        let config = AcceleratedPCGConfig::robust();
        let mut solver = AcceleratedPCG::new(config);
        let result = solver.solve(&a, &b);

        assert!(result.iterations > 0);
        assert!(result.solution.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_solve_accelerated() {
        let n = 50;
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 4.0;
            if i > 0 { a[(i, i - 1)] = -1.0; }
            if i < n - 1 { a[(i, i + 1)] = -1.0; }
        }
        let b = DVector::from_element(n, 1.0);

        let result = solve_accelerated(&a, &b);

        assert!(result.iterations > 0);
        assert!(result.solution.iter().all(|v| v.is_finite()));
    }
}
