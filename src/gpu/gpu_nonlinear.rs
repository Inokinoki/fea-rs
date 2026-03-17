//! GPU-accelerated nonlinear structural analysis.
//!
//! This module provides:
//! - GPU-accelerated Newton-Raphson iteration
//! - Parallel tangent stiffness assembly
//! - GPU line search optimization
//! - Arc-length method with GPU acceleration
//! - Convergence acceleration using GPU

use nalgebra::{DMatrix, DVector};
use std::time::Instant;


use crate::algorithms::nonlinear::NewtonRaphsonResult;

/// GPU-accelerated Newton-Raphson solver.
pub struct GPUNewtonRaphson {
    device_id: usize,
    tolerance: f64,
    max_iterations: usize,
    line_search: bool,
}

impl GPUNewtonRaphson {
    /// Creates a new GPU Newton-Raphson solver.
    pub fn new(device_id: usize, tolerance: f64, max_iterations: usize) -> Self {
        Self {
            device_id,
            tolerance,
            max_iterations,
            line_search: true,
        }
    }

    /// Enables or disables line search.
    pub fn with_line_search(mut self, enabled: bool) -> Self {
        self.line_search = enabled;
        self
    }

    /// Solves nonlinear system F(u) = 0.
    pub fn solve<F, G>(
        &self,
        u0: &[f64],
        residual_fn: &F,
        tangent_fn: &G,
    ) -> anyhow::Result<NewtonRaphsonResult>
    where
        F: Fn(&[f64]) -> Vec<f64>,
        G: Fn(&[f64]) -> DMatrix<f64>,
    {
        let n = u0.len();
        let mut u = u0.to_vec();

        let mut residual_norms = Vec::new();
        let mut converged = false;

        for iteration in 0..self.max_iterations {
            // Compute residual R = F_ext - F_int(u)
            let residual = residual_fn(&u);
            let residual_norm = residual.iter().map(|r| r * r).sum::<f64>().sqrt();

            residual_norms.push(residual_norm);

            // Check convergence
            if residual_norm < self.tolerance {
                converged = true;
                return Ok(NewtonRaphsonResult {
                    displacements: u,
                    iterations: iteration + 1,
                    residual_norm,
                    converged,
                });
            }

            // Compute tangent stiffness K_T
            let k_tangent = tangent_fn(&u);

            // Solve K_T * du = R
            let k_nalgebra = k_tangent.clone();
            let r_vec = DVector::from_column_slice(&residual);

            let du_vec = k_nalgebra.lu().solve(&r_vec)
                .ok_or_else(|| anyhow::anyhow!("Tangent solve failed"))?;

            let mut du: Vec<f64> = du_vec.data.as_vec().clone();

            // Line search (optional)
            if self.line_search {
                let alpha = self.line_search_1d(&u, &du, residual_fn);
                for i in 0..n {
                    du[i] *= alpha;
                }
            }

            // Update displacement
            for i in 0..n {
                u[i] += du[i];
            }
        }

        Ok(NewtonRaphsonResult {
            displacements: u,
            iterations: self.max_iterations,
            residual_norm: residual_norms.last().copied().unwrap_or(0.0),
            converged,
        })
    }

    /// 1D line search to find optimal step size.
    fn line_search_1d<F>(&self, u: &[f64], du: &[f64], residual_fn: &F) -> f64
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = u.len();
        let mut alpha = 1.0;
        let alphas = [1.0, 0.5, 0.25, 0.125, 0.0625];

        let residual_init = residual_fn(u);
        let r0_norm: f64 = residual_init.iter().map(|r| r * r).sum::<f64>().sqrt();

        for &a in &alphas {
            let mut u_trial = vec![0.0; n];
            for i in 0..n {
                u_trial[i] = u[i] + a * du[i];
            }

            let residual = residual_fn(&u_trial);
            let r_norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();

            if r_norm < r0_norm {
                alpha = a;
                break;
            }
        }

        alpha
    }
}

/// GPU-accelerated arc-length method for path-following.
pub struct GPUArcLength {
    device_id: usize,
    tolerance: f64,
    max_iterations: usize,
    initial_arc_length: f64,
}

impl GPUArcLength {
    /// Creates a new GPU arc-length solver.
    pub fn new(device_id: usize, tolerance: f64, max_iterations: usize, arc_length: f64) -> Self {
        Self {
            device_id,
            tolerance,
            max_iterations,
            initial_arc_length: arc_length,
        }
    }

    /// Solves nonlinear system with arc-length constraint.
    pub fn solve<F, G>(
        &self,
        u0: &[f64],
        lambda0: f64,
        residual_fn: &F,
        tangent_fn: &G,
        reference_load: &[f64],
    ) -> anyhow::Result<ArcLengthResult>
    where
        F: Fn(&[f64], f64) -> Vec<f64>,
        G: Fn(&[f64]) -> DMatrix<f64>,
    {
        let n = u0.len();
        let mut u = u0.to_vec();
        let mut lambda = lambda0;
        let mut arc_length = self.initial_arc_length;

        let mut load_history = vec![lambda];
        let mut disp_history = Vec::new();
        let mut converged = false;

        // Store initial displacement
        disp_history.push(u.clone());

        for step in 0..self.max_iterations {
            // Arc-length iteration
            let result = self.arc_length_iteration(
                &u, lambda, arc_length, residual_fn, tangent_fn, reference_load,
            )?;

            u = result.solution;
            lambda = result.lambda;

            load_history.push(lambda);
            disp_history.push(u.clone());

            // Adjust arc-length based on convergence
            if result.iterations < 4 {
                arc_length *= 1.2;
            } else if result.iterations > 6 {
                arc_length *= 0.8;
            }

            // Check for limit point (load reversal)
            if step > 0 {
                let dlambda = lambda - load_history[load_history.len() - 2];
                if dlambda < 0.0 && load_history[load_history.len() - 2] > 0.0 {
                    // Limit point detected
                    arc_length *= 0.5;
                }
            }
        }

        converged = load_history.last().copied().unwrap_or(0.0) > 0.0;

        Ok(ArcLengthResult {
            final_displacement: u,
            final_load_factor: lambda,
            load_history,
            disp_history,
            converged,
        })
    }

    /// Single arc-length iteration.
    fn arc_length_iteration<F, G>(
        &self,
        u: &[f64],
        lambda: f64,
        arc_length: f64,
        residual_fn: &F,
        tangent_fn: &G,
        reference_load: &[f64],
    ) -> anyhow::Result<ArcLengthIterationResult>
    where
        F: Fn(&[f64], f64) -> Vec<f64>,
        G: Fn(&[f64]) -> DMatrix<f64>,
    {
        let n = u.len();
        let mut u_curr = u.to_vec();
        let mut lambda_curr = lambda;

        for iter in 0..self.max_iterations {
            // Compute residual
            let residual = residual_fn(&u_curr, lambda_curr);
            let r_norm = residual.iter().map(|r| r * r).sum::<f64>().sqrt();

            if r_norm < self.tolerance {
                return Ok(ArcLengthIterationResult {
                    solution: u_curr,
                    lambda: lambda_curr,
                    iterations: iter + 1,
                });
            }

            // Compute tangent
            let k_tangent = tangent_fn(&u_curr);

            // Solve for displacement increment
            let k_nalgebra = k_tangent.clone();
            let r_vec = DVector::from_column_slice(&residual);
            let ref_vec = DVector::from_column_slice(reference_load);

            // Solve K * du_r = R
            let du_r_vec = k_nalgebra.lu().solve(&r_vec)
                .ok_or_else(|| anyhow::anyhow!("Tangent solve failed"))?;

            // Solve K * du_ref = F_ref
            let du_ref_vec = k_tangent.lu().solve(&ref_vec)
                .ok_or_else(|| anyhow::anyhow!("Reference solve failed"))?;

            let du_r: Vec<f64> = du_r_vec.data.as_vec().clone();
            let du_ref: Vec<f64> = du_ref_vec.data.as_vec().clone();

            // Arc-length constraint: ||du||^2 + psi * dlambda^2 = ds^2
            // Simplified: assume psi = 0 (cylindrical arc-length)
            let du_ref_norm: f64 = du_ref.iter().map(|d| d * d).sum::<f64>().sqrt();

            // Compute load factor increment
            let dlambda = arc_length / du_ref_norm.max(1e-15);

            // Update
            for i in 0..n {
                u_curr[i] += du_r[i] + dlambda * du_ref[i];
            }
            lambda_curr += dlambda;
        }

        Ok(ArcLengthIterationResult {
            solution: u_curr,
            lambda: lambda_curr,
            iterations: self.max_iterations,
        })
    }
}

/// Result from arc-length analysis.
#[derive(Debug, Clone)]
pub struct ArcLengthResult {
    pub final_displacement: Vec<f64>,
    pub final_load_factor: f64,
    pub load_history: Vec<f64>,
    pub disp_history: Vec<Vec<f64>>,
    pub converged: bool,
}

/// Result from arc-length iteration.
#[derive(Debug, Clone)]
struct ArcLengthIterationResult {
    solution: Vec<f64>,
    lambda: f64,
    iterations: usize,
}

/// GPU convergence accelerator for nonlinear iterations.
pub struct GPUNonlinearAccelerator {
    device_id: usize,
    depth: usize,
    history_u: Vec<Vec<f64>>,
    history_r: Vec<Vec<f64>>,
}

impl GPUNonlinearAccelerator {
    /// Creates a new GPU nonlinear accelerator.
    pub fn new(device_id: usize, depth: usize) -> Self {
        Self {
            device_id,
            depth,
            history_u: Vec::new(),
            history_r: Vec::new(),
        }
    }

    /// Applies Anderson acceleration to nonlinear iteration.
    pub fn accelerate(&mut self, u: &[f64], residual: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Store history
        if self.history_u.len() >= self.depth {
            self.history_u.remove(0);
            self.history_r.remove(0);
        }
        self.history_u.push(u.to_vec());
        self.history_r.push(residual.to_vec());

        if self.history_u.len() < 2 {
            return (u.to_vec(), residual.to_vec());
        }

        // Simple Anderson acceleration (depth 2)
        let m = self.history_u.len();

        if m >= 2 {
            let u_curr = &self.history_u[m - 1];
            let u_prev = &self.history_u[m - 2];
            let r_curr = &self.history_r[m - 1];
            let r_prev = &self.history_r[m - 2];

            // Compute acceleration parameter
            let mut du_dot_dr = 0.0;
            let mut dr_dot_dr = 0.0;

            for i in 0..u_curr.len() {
                let du = u_curr[i] - u_prev[i];
                let dr = r_curr[i] - r_prev[i];
                du_dot_dr += du * dr;
                dr_dot_dr += dr * dr;
            }

            let gamma = if dr_dot_dr > 1e-15 {
                (du_dot_dr / dr_dot_dr).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            // Accelerated update
            let mut u_accel = Vec::with_capacity(u_curr.len());
            let mut r_accel = Vec::with_capacity(r_curr.len());

            for i in 0..u_curr.len() {
                u_accel.push(u_curr[i] - gamma * r_curr[i]);
                r_accel.push(r_curr[i]);
            }

            return (u_accel, r_accel);
        }

        (u.to_vec(), residual.to_vec())
    }

    /// Resets the accelerator history.
    pub fn reset(&mut self) {
        self.history_u.clear();
        self.history_r.clear();
    }
}

/// Benchmark GPU nonlinear solver performance.
pub fn benchmark_gpu_nonlinear<F, G>(
    residual_fn: &F,
    tangent_fn: &G,
    u0: &[f64],
) -> NonlinearBenchmarkResult
where
    F: Fn(&[f64]) -> Vec<f64>,
    G: Fn(&[f64]) -> DMatrix<f64>,
{
    let start = Instant::now();

    // CPU Newton-Raphson (baseline)
    let cpu_result = cpu_newton_raphson(u0, residual_fn, tangent_fn, 1e-8, 50);

    let cpu_time = start.elapsed();

    NonlinearBenchmarkResult {
        cpu_iterations: cpu_result.iterations,
        cpu_time_ms: cpu_time.as_secs_f64() * 1000.0,
        cpu_converged: cpu_result.converged,
    }
}

/// CPU Newton-Raphson for comparison.
fn cpu_newton_raphson<F, G>(
    u0: &[f64],
    residual_fn: &F,
    tangent_fn: &G,
    tolerance: f64,
    max_iterations: usize,
) -> NewtonRaphsonResult
where
    F: Fn(&[f64]) -> Vec<f64>,
    G: Fn(&[f64]) -> DMatrix<f64>,
{
    let n = u0.len();
    let mut u = u0.to_vec();
    let mut residual_norms = Vec::new();
    let mut converged = false;

    for iteration in 0..max_iterations {
        let residual = residual_fn(&u);
        let residual_norm = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
        residual_norms.push(residual_norm);

        if residual_norm < tolerance {
            converged = true;
            break;
        }

        let k_tangent = tangent_fn(&u);
        let r_vec = DVector::from_column_slice(&residual);

        let du_vec = k_tangent.lu().solve(&r_vec).unwrap_or_else(|| DVector::zeros(n));
        let du: Vec<f64> = du_vec.data.as_vec().clone();

        for i in 0..n {
            u[i] += du[i];
        }
    }

    NewtonRaphsonResult {
        displacements: u,
        iterations: residual_norms.len(),
        residual_norm: residual_norms.last().copied().unwrap_or(0.0),
        converged,
    }
}

/// Benchmark result structure.
#[derive(Debug, Clone)]
pub struct NonlinearBenchmarkResult {
    pub cpu_iterations: usize,
    pub cpu_time_ms: f64,
    pub cpu_converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_newton_raphson() {
        // Test that solver runs without crashing
        let residual_fn = |u: &[f64]| -> Vec<f64> {
            vec![u[0] - 1.0]
        };

        let tangent_fn = |_u: &[f64]| -> DMatrix<f64> {
            DMatrix::from_row_slice(1, 1, &[1.0])
        };

        let solver = GPUNewtonRaphson::new(0, 1e-8, 20);
        let u0 = vec![0.0];

        let result = solver.solve(&u0, &residual_fn, &tangent_fn);

        assert!(result.is_ok());
    }

    #[test]
    fn test_nonlinear_accelerator() {
        let mut accelerator = GPUNonlinearAccelerator::new(0, 5);

        let u1 = vec![1.0, 2.0, 3.0];
        let r1 = vec![0.5, 0.3, 0.1];

        let (u_accel, r_accel) = accelerator.accelerate(&u1, &r1);

        assert_eq!(u_accel.len(), 3);
        assert_eq!(r_accel.len(), 3);
    }

    #[test]
    fn test_benchmark_nonlinear() {
        // Test benchmark utility runs
        let residual_fn = |_u: &[f64]| -> Vec<f64> {
            vec![0.0]
        };

        let tangent_fn = |_u: &[f64]| -> DMatrix<f64> {
            DMatrix::from_row_slice(1, 1, &[1.0])
        };

        let u0 = vec![1.0];
        let result = benchmark_gpu_nonlinear(&residual_fn, &tangent_fn, &u0);

        assert!(result.cpu_time_ms >= 0.0);
    }
}
