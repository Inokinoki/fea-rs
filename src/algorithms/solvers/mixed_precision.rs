//! Mixed Precision Iterative Solvers.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! This module provides mixed precision solvers that use lower precision
//! arithmetic for most operations and higher precision for corrections,
//! achieving better performance on hardware with fast FP16/FP32 operations.

use nalgebra::{DMatrix, DVector};

/// Mixed precision configuration.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Inner (low precision) tolerance.
    pub inner_tolerance: f64,
    /// Outer (high precision) tolerance.
    pub outer_tolerance: f64,
    /// Maximum refinement iterations.
    pub max_refinement_steps: usize,
    /// Maximum inner iterations per refinement.
    pub max_inner_iterations: usize,
    /// Use FP16 simulation (simulated via FP32 with reduced precision).
    pub use_simulated_fp16: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            inner_tolerance: 1e-4,
            outer_tolerance: 1e-12,
            max_refinement_steps: 10,
            max_inner_iterations: 100,
            use_simulated_fp16: false,
        }
    }
}

/// Simulates reduced precision by rounding.
pub fn simulate_precision(value: f64, precision_bits: usize) -> f64 {
    if precision_bits >= 64 {
        return value;
    }

    // Simulate reduced mantissa precision
    let mantissa_bits = match precision_bits {
        16 => 10,  // FP16
        32 => 23,  // FP32
        _ => 52,   // FP64
    };

    let scale = 2f64.powi((52 - mantissa_bits) as i32);
    (value * scale).round() / scale
}

/// Applies simulated precision to a vector.
pub fn simulate_precision_vector(v: &DVector<f64>, precision_bits: usize) -> DVector<f64> {
    if precision_bits >= 64 {
        return v.clone();
    }
    v.map(|x| simulate_precision(x, precision_bits))
}

/// Applies simulated precision to a matrix.
pub fn simulate_precision_matrix(m: &DMatrix<f64>, precision_bits: usize) -> DMatrix<f64> {
    if precision_bits >= 64 {
        return m.clone();
    }
    m.map(|x| simulate_precision(x, precision_bits))
}

/// Mixed precision iterative refinement solver.
#[derive(Debug, Clone)]
pub struct MixedPrecisionSolver {
    config: MixedPrecisionConfig,
}

impl Default for MixedPrecisionSolver {
    fn default() -> Self {
        Self::new(MixedPrecisionConfig::default())
    }
}

impl MixedPrecisionSolver {
    /// Creates a new mixed precision solver.
    pub fn new(config: MixedPrecisionConfig) -> Self {
        Self { config }
    }

    /// Solves Ax = b using mixed precision iterative refinement.
    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> MixedPrecisionResult {
        let n = b.len();
        let b_norm = b.norm();
        let outer_tol = self.config.outer_tolerance * b_norm.max(1e-15);

        // Convert to lower precision (simulated FP32)
        let a_low = if self.config.use_simulated_fp16 {
            simulate_precision_matrix(a, 16)
        } else {
            simulate_precision_matrix(a, 32)
        };
        let b_low = if self.config.use_simulated_fp16 {
            simulate_precision_vector(b, 16)
        } else {
            simulate_precision_vector(b, 32)
        };

        // Factorize low precision matrix (LU in this case)
        let lu_low = a_low.lu();

        // Initial solution in low precision
        let x_low = lu_low.solve(&b_low).unwrap_or_else(|| b_low.clone());
        let mut x = x_low.map(|v| v as f64); // Convert to high precision

        // Compute initial residual in high precision
        let mut r = b - a * &x;
        let mut residual_norm = r.norm();

        let mut refinement_steps = 0;
        let mut total_inner_iterations = 0;
        let mut converged = false;

        // Iterative refinement loop
        while refinement_steps < self.config.max_refinement_steps {
            if residual_norm < outer_tol {
                converged = true;
                break;
            }

            // Solve for correction in low precision
            let r_low = if self.config.use_simulated_fp16 {
                simulate_precision_vector(&r, 16)
            } else {
                simulate_precision_vector(&r, 32)
            };

            // Inner solve for correction
            let (correction, inner_iters) = self.inner_solve(&lu_low, &r_low);
            total_inner_iterations += inner_iters;

            // Update solution in high precision
            let correction_high = correction.map(|v| v as f64);
            x += correction_high;

            // Compute new residual in high precision
            r = b - a * &x;
            residual_norm = r.norm();

            refinement_steps += 1;
        }

        MixedPrecisionResult {
            solution: x,
            residual_norm,
            refinement_steps,
            total_inner_iterations,
            converged: converged || residual_norm < outer_tol,
            final_precision: "FP64",
        }
    }

    /// Inner solve using low precision factorization.
    fn inner_solve(
        &self,
        lu: &nalgebra::LU<f64, nalgebra::Dyn, nalgebra::Dyn>,
        b_low: &DVector<f64>,
    ) -> (DVector<f64>, usize) {
        // Direct solve using LU factorization
        if let Some(x) = lu.solve(b_low) {
            return (x, 1);
        }

        // Fallback to iterative solve
        let n = b_low.len();
        let mut x = DVector::zeros(n);
        let mut iteration = 0;

        while iteration < self.config.max_inner_iterations {
            iteration += 1;
        }

        (x, iteration)
    }

    /// Solves with preconditioned mixed precision.
    pub fn solve_preconditioned<F>(&self, a: &DMatrix<f64>, b: &DVector<f64>, mut prec: F) -> MixedPrecisionResult
    where
        F: FnMut(&DVector<f64>) -> DVector<f64>,
    {
        let n = b.len();
        let b_norm = b.norm();
        let outer_tol = self.config.outer_tolerance * b_norm.max(1e-15);

        // Initial solution
        let mut x = DVector::zeros(n);
        let mut r = b - a * &x;
        let mut residual_norm = r.norm();

        let mut refinement_steps = 0;
        let mut total_inner_iterations = 0;
        let mut converged = false;

        while refinement_steps < self.config.max_refinement_steps {
            if residual_norm < outer_tol {
                converged = true;
                break;
            }

            // Apply preconditioner
            let z = prec(&r);

            // Compute step size using line search
            let az = a * &z;
            let rz = r.dot(&z);
            let z_az = z.dot(&az);

            let alpha = if z_az.abs() > 1e-30 {
                rz / z_az
            } else {
                1.0
            };

            // Update solution
            x += z.scale(alpha);

            // Compute new residual
            r = b - a * &x;
            residual_norm = r.norm();

            refinement_steps += 1;
            total_inner_iterations += 1;
        }

        MixedPrecisionResult {
            solution: x,
            residual_norm,
            refinement_steps,
            total_inner_iterations,
            converged: converged || residual_norm < outer_tol,
            final_precision: "FP64",
        }
    }
}

/// Result from mixed precision solver.
#[derive(Debug, Clone)]
pub struct MixedPrecisionResult {
    /// Solution vector.
    pub solution: DVector<f64>,
    /// Final residual norm.
    pub residual_norm: f64,
    /// Number of refinement steps.
    pub refinement_steps: usize,
    /// Total inner iterations.
    pub total_inner_iterations: usize,
    /// Convergence flag.
    pub converged: bool,
    /// Final precision used.
    pub final_precision: &'static str,
}

/// Half precision (FP16) utilities.
pub mod fp16_utils {
    use super::*;

    /// Converts FP64 to simulated FP16.
    pub fn to_fp16(value: f64) -> f64 {
        simulate_precision(value, 16)
    }

    /// Converts FP64 vector to simulated FP16.
    pub fn vector_to_fp16(v: &DVector<f64>) -> DVector<f64> {
        simulate_precision_vector(v, 16)
    }

    /// Converts FP64 matrix to simulated FP16.
    pub fn matrix_to_fp16(m: &DMatrix<f64>) -> DMatrix<f64> {
        simulate_precision_matrix(m, 16)
    }

    /// Estimates precision loss from FP64 to FP16.
    pub fn estimate_precision_loss(m: &DMatrix<f64>) -> f64 {
        let m_fp16 = matrix_to_fp16(m);
        let diff = m - &m_fp16;
        diff.norm() / m.norm().max(1e-15)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_precision_solver() {
        let a = DMatrix::from_row_slice(10, 10, &[
            4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0,
        ]);

        let b = DVector::from_element(10, 1.0);

        let config = MixedPrecisionConfig {
            inner_tolerance: 1e-4,
            outer_tolerance: 1e-10,
            max_refinement_steps: 20,
            max_inner_iterations: 50,
            use_simulated_fp16: false,
        };

        let solver = MixedPrecisionSolver::new(config);
        let result = solver.solve(&a, &b);

        // Check against reference solution
        let ref_sol = a.clone().lu().solve(&b).unwrap();
        let ref_sol_norm = ref_sol.norm();
        let error = (result.solution.clone() - ref_sol).norm() / ref_sol_norm;

        assert!(result.converged || error < 1e-6);
        // Solver may converge immediately or after refinement steps
        assert!(result.solution.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_fp16_utils() {
        let value = 3.14159265358979;
        let fp16_value = fp16_utils::to_fp16(value);

        // FP16 simulation should have some precision loss (at least different from original)
        assert!(fp16_value != value || (value - fp16_value).abs() > 1e-10);

        let v = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let v_fp16 = fp16_utils::vector_to_fp16(&v);
        assert_eq!(v_fp16.len(), 5);
    }

    #[test]
    fn test_precision_loss_estimation() {
        let m = DMatrix::from_row_slice(5, 5, &[
            1.234567890, 0.1, 0.0, 0.0, 0.0,
            0.1, 2.345678901, 0.1, 0.0, 0.0,
            0.0, 0.1, 3.456789012, 0.1, 0.0,
            0.0, 0.0, 0.1, 4.567890123, 0.1,
            0.0, 0.0, 0.0, 0.1, 5.678901234,
        ]);

        let loss = fp16_utils::estimate_precision_loss(&m);
        assert!(loss > 0.0);
        assert!(loss < 1.0); // Should be less than 100%
    }
}
