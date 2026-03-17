//! Full Multigrid (FMG) solver.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! This module provides a full multigrid solver that can be used
//! as a standalone solver or as a preconditioner.

use nalgebra::{DMatrix, DVector};

/// Full Multigrid solver configuration.
#[derive(Debug, Clone)]
pub struct FullMultigridConfig {
    /// Number of grid levels.
    pub num_levels: usize,
    /// Number of pre-smoothing iterations.
    pub nu_pre: usize,
    /// Number of post-smoothing iterations.
    pub nu_post: usize,
    /// Number of coarse grid iterations.
    pub nu_coarse: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
}

impl Default for FullMultigridConfig {
    fn default() -> Self {
        Self {
            num_levels: 4,
            nu_pre: 2,
            nu_post: 2,
            nu_coarse: 10,
            tolerance: 1e-8,
            max_iterations: 100,
        }
    }
}

/// Full Multigrid solver using V-cycles.
pub struct FullMultigridSolver {
    config: FullMultigridConfig,
    /// Grid operators (restriction and prolongation).
    grids: Vec<GridOperator>,
}

/// Grid transfer operators for one level.
#[derive(Debug, Clone)]
struct GridOperator {
    /// Restriction operator (fine to coarse).
    restriction: DMatrix<f64>,
    /// Prolongation operator (coarse to fine).
    prolongation: DMatrix<f64>,
}

impl FullMultigridSolver {
    /// Creates a new FMG solver.
    pub fn new(config: FullMultigridConfig) -> Self {
        Self {
            config,
            grids: Vec::new(),
        }
    }

    /// Sets up the multigrid hierarchy from a fine-grid operator.
    pub fn setup(&mut self, a_fine: &DMatrix<f64>) {
        let n = a_fine.nrows();
        self.grids.clear();

        let mut current_n = n;
        while current_n > 4 {
            let next_n = (current_n + 1) / 2;

            // Build restriction (full weighting)
            let mut r = DMatrix::zeros(next_n, current_n);
            for i in 0..next_n {
                let fine_i = 2 * i;
                if fine_i < current_n {
                    r[(i, fine_i)] = 0.5;
                }
                if fine_i + 1 < current_n {
                    r[(i, fine_i + 1)] = 0.25;
                }
                if fine_i > 0 {
                    r[(i, fine_i - 1)] = 0.25;
                }
            }

            // Prolongation is transpose of restriction (scaled)
            let p = r.transpose() * 2.0;

            self.grids.push(GridOperator {
                restriction: r,
                prolongation: p,
            });

            current_n = next_n;
        }
    }

    /// Solves A*x = b using full multigrid.
    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>, x_init: &DVector<f64>) -> (DVector<f64>, usize, bool) {
        let mut x = x_init.clone();
        let mut residual = b - a * &x;
        let b_norm = b.norm();
        let tol = self.config.tolerance * b_norm.max(1e-15);

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.config.max_iterations {
            // V-cycle
            x = self.v_cycle(a, &x, b, 0);

            // Compute new residual
            residual = b - a * &x;
            let r_norm = residual.norm();

            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            iteration += 1;
        }

        (x, iteration, converged)
    }

    /// Performs one V-cycle.
    fn v_cycle(&self, a: &DMatrix<f64>, x: &DVector<f64>, b: &DVector<f64>, level: usize) -> DVector<f64> {
        let mut x_out = x.clone();

        // Pre-smoothing (Gauss-Seidel)
        for _ in 0..self.config.nu_pre {
            x_out = self.gauss_seidel(a, &x_out, b);
        }

        // Compute residual
        let r = b - a * &x_out;

        if level < self.grids.len() {
            // Restrict residual to coarse grid
            let r_coarse = &self.grids[level].restriction * &r;

            // Coarse grid correction (recursive or direct solve)
            let mut e_coarse = DVector::zeros(r_coarse.len());
            if level == self.grids.len() - 1 {
                // Coarsest level: solve directly
                // Build coarse operator
                let a_coarse = self.build_coarse_operator(a, level);
                for _ in 0..self.config.nu_coarse {
                    e_coarse = self.gauss_seidel(&a_coarse, &e_coarse, &r_coarse);
                }
            } else {
                // Recursive V-cycle
                let a_coarse = self.build_coarse_operator(a, level);
                e_coarse = self.v_cycle(&a_coarse, &e_coarse, &r_coarse, level + 1);
            }

            // Prolongate correction
            let e_fine = &self.grids[level].prolongation * &e_coarse;
            x_out = &x_out + &e_fine;
        }

        // Post-smoothing
        for _ in 0..self.config.nu_post {
            x_out = self.gauss_seidel(a, &x_out, b);
        }

        x_out
    }

    /// Builds coarse grid operator using Galerkin projection.
    fn build_coarse_operator(&self, a: &DMatrix<f64>, level: usize) -> DMatrix<f64> {
        let r = &self.grids[level].restriction;
        let p = &self.grids[level].prolongation;
        r * a * p
    }

    /// Gauss-Seidel smoother.
    fn gauss_seidel(&self, a: &DMatrix<f64>, x: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
        let n = b.len();
        let mut x_new = x.clone();

        for i in 0..n {
            let mut sum = b[i];
            for j in 0..n {
                if i != j {
                    sum -= a[(i, j)] * x_new[j];
                }
            }
            if a[(i, i)].abs() > 1e-15 {
                x_new[i] = sum / a[(i, i)];
            }
        }

        x_new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmg_solver() {
        // Create a 1D Poisson matrix
        let n = 64;
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 2.0;
            if i > 0 {
                a[(i, i - 1)] = -1.0;
            }
            if i < n - 1 {
                a[(i, i + 1)] = -1.0;
            }
        }

        let b = DVector::from_element(n, 1.0);
        let x_init = DVector::zeros(n);

        let config = FullMultigridConfig {
            num_levels: 4,
            nu_pre: 2,
            nu_post: 2,
            tolerance: 1e-6,
            max_iterations: 50,
            ..Default::default()
        };

        let mut fmg = FullMultigridSolver::new(config);
        fmg.setup(&a);

        let (x, iterations, converged) = fmg.solve(&a, &b, &x_init);

        assert!(converged, "FMG should converge for Poisson problem, took {} iterations", iterations);
        assert!(iterations < 20, "Should converge quickly with FMG");

        // Verify solution
        let residual = &b - &a * &x;
        assert!(residual.norm() < 1e-4, "Residual should be small");
    }

    #[test]
    fn test_v_cycle() {
        let n = 16;
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 2.0;
            if i > 0 {
                a[(i, i - 1)] = -1.0;
            }
            if i < n - 1 {
                a[(i, i + 1)] = -1.0;
            }
        }

        let b = DVector::from_element(n, 1.0);
        let x = DVector::zeros(n);

        let config = FullMultigridConfig::default();
        let mut fmg = FullMultigridSolver::new(config);
        fmg.setup(&a);

        // Single V-cycle should reduce residual
        let x_new = fmg.v_cycle(&a, &x, &b, 0);
        let r_old = b.norm();
        let r_new = (&b - &a * &x_new).norm();

        assert!(r_new < r_old, "V-cycle should reduce residual");
    }

    #[test]
    fn test_coarse_operator() {
        let n = 8;
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 2.0;
            if i > 0 {
                a[(i, i - 1)] = -1.0;
            }
            if i < n - 1 {
                a[(i, i + 1)] = -1.0;
            }
        }

        let config = FullMultigridConfig::default();
        let mut fmg = FullMultigridSolver::new(config);
        fmg.setup(&a);

        // Check that coarse operator is symmetric positive definite
        let a_coarse = fmg.build_coarse_operator(&a, 0);

        // Symmetry
        for i in 0..a_coarse.nrows() {
            for j in (i + 1)..a_coarse.ncols() {
                assert!((a_coarse[(i, j)] - a_coarse[(j, i)]).abs() < 1e-10);
            }
        }

        // Positive diagonal
        for i in 0..a_coarse.nrows() {
            assert!(a_coarse[(i, i)] > 0.0);
        }
    }
}
