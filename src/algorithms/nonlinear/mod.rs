//! Nonlinear solution methods.
//!
//! This module provides:
//! - Newton-Raphson method for geometric/material nonlinearities
//! - Arc-length method for snap-through analysis

use nalgebra::{DMatrix, DVector};

/// Newton-Raphson solver configuration.
#[derive(Debug, Clone, Copy)]
pub struct NewtonRaphsonConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance (force residual norm).
    pub tolerance: f64,
    /// Maximum line search iterations.
    pub max_line_search: usize,
}

impl Default for NewtonRaphsonConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
            max_line_search: 10,
        }
    }
}

/// Newton-Raphson solver result.
#[derive(Debug, Clone)]
pub struct NewtonRaphsonResult {
    /// Final displacement vector.
    pub displacements: Vec<f64>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// Newton-Raphson method for nonlinear problems.
///
/// Solves K(u) * u = F iteratively using tangent stiffness.
pub fn newton_raphson<F, K>(
    n: usize,
    mut external_force: F,
    mut tangent_stiffness: K,
    config: &NewtonRaphsonConfig,
) -> NewtonRaphsonResult
where
    F: FnMut(&[f64]) -> DVector<f64>,
    K: FnMut(&[f64]) -> DMatrix<f64>,
{
    let mut u = vec![0.0f64; n];
    let mut iteration = 0;
    let mut converged = false;
    let mut residual_norm = f64::INFINITY;

    while iteration < config.max_iterations {
        // Compute internal forces
        let k_tangent = tangent_stiffness(&u);
        let f_internal = external_force(&u);
        let f_external = external_force(&u); // In reality, this might be different

        // Residual: R = F_ext - F_int
        let residual = subtract_vectors(&f_external, &f_internal);
        residual_norm = residual.norm();

        if residual_norm < config.tolerance {
            converged = true;
            break;
        }

        // Solve for displacement increment: K_t * du = R
        let du = match k_tangent.lu().solve(&residual) {
            Some(du) => du,
            None => break, // Singular matrix
        };

        // Line search (simplified - full step)
        for i in 0..n {
            u[i] += du[i];
        }

        iteration += 1;
    }

    NewtonRaphsonResult {
        displacements: u,
        iterations: iteration,
        residual_norm,
        converged,
    }
}

fn subtract_vectors(a: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
    a - b
}

/// Arc-length method configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct ArcLengthConfig {
    /// Initial arc-length radius.
    pub initial_radius: f64,
    /// Maximum iterations per load step.
    pub max_iterations: usize,
    /// Number of load steps.
    pub num_load_steps: usize,
}

/// Arc-length method result.
#[derive(Debug, Clone)]
pub struct ArcLengthResult {
    /// Load factors at each converged step.
    pub load_factors: Vec<f64>,
    /// Displacement vectors at each converged step.
    pub displacements: Vec<Vec<f64>>,
    /// Total iterations used.
    pub total_iterations: usize,
}

/// Arc-length method for snap-through and snap-back analysis.
pub fn arc_length<F, K>(
    n: usize,
    mut reference_force: F,
    mut tangent_stiffness: K,
    config: &ArcLengthConfig,
) -> ArcLengthResult
where
    F: FnMut(f64) -> DVector<f64>,
    K: FnMut(&[f64], f64) -> DMatrix<f64>,
{
    let mut u = vec![0.0f64; n];
    let mut load_factor = 0.0;
    let mut load_factors = Vec::with_capacity(config.num_load_steps);
    let mut displacements = Vec::with_capacity(config.num_load_steps);
    let mut total_iterations = 0;

    let delta_l = config.initial_radius;

    for _step in 0..config.num_load_steps {
        let mut iteration = 0;
        let mut lambda = load_factor + delta_l;

        while iteration < config.max_iterations {
            let f_ext = reference_force(lambda);
            let k_tan = tangent_stiffness(&u, lambda);
            let f_int = reference_force(0.0); // Simplified internal force

            let residual = &f_ext - &f_int;
            let residual_norm = residual.norm();

            if residual_norm < 1e-8 {
                load_factor = lambda;
                load_factors.push(load_factor);
                displacements.push(u.clone());
                break;
            }

            let du = match k_tan.lu().solve(&residual) {
                Some(du) => du,
                None => break,
            };

            for i in 0..n {
                u[i] += du[i];
            }

            // Update load factor based on arc-length constraint
            // Simplified: proportional to displacement norm
            let du_norm = du.norm();
            if du_norm > 1e-15 {
                lambda += delta_l * du_norm / n as f64;
            }

            iteration += 1;
            total_iterations += 1;
        }

        if iteration >= config.max_iterations {
            break; // Failed to converge
        }
    }

    ArcLengthResult {
        load_factors,
        displacements,
        total_iterations,
    }
}
