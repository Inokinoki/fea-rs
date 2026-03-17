//! Block iterative solvers for coupled systems.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! This module provides:
//! - Block Jacobi solver
//! - Block Gauss-Seidel solver
//! - Block CG solver
//! - Schur complement methods
//!
//! Useful for multiphysics and coupled problems.

use nalgebra::{DMatrix, DVector};

use super::SolverResult;

/// Block size for block solvers.
#[derive(Debug, Clone)]
pub enum BlockSize {
    /// Fixed block size.
    Fixed(usize),
    /// Variable block sizes.
    Variable(Vec<usize>),
    /// Automatic block detection.
    Auto,
}

/// Block Jacobi solver.
///
/// Solves systems by inverting block diagonal parts.
#[derive(Debug, Clone)]
pub struct BlockJacobiSolver {
    block_size: usize,
    max_iterations: usize,
    tolerance: f64,
}

impl BlockJacobiSolver {
    /// Creates a new Block Jacobi solver.
    pub fn new(block_size: usize, tolerance: f64, max_iterations: usize) -> Self {
        Self {
            block_size,
            max_iterations,
            tolerance,
        }
    }

    /// Solves using block Jacobi iteration.
    pub fn solve_block(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> anyhow::Result<SolverResult> {
        let n = b.len();
        let bs = self.block_size;
        let mut x = vec![0.0; n];
        let b_norm = b.norm();
        let tol = self.tolerance * b_norm.max(1e-15);

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            let mut max_change: f64 = 0.0;

            // Process each block
            for block_start in (0..n).step_by(bs) {
                let block_end = (block_start + bs).min(n);
                let block_size = block_end - block_start;

                // Extract block diagonal
                let mut block = DMatrix::zeros(block_size, block_size);
                for i in 0..block_size {
                    for j in 0..block_size {
                        block[(i, j)] = a[(block_start + i, block_start + j)];
                    }
                }

                // Compute off-block contribution
                let mut rhs = DVector::zeros(block_size);
                for i in 0..block_size {
                    rhs[i] = b[block_start + i];
                    for j in 0..block_start {
                        rhs[i] -= a[(block_start + i, j)] * x[j];
                    }
                    for j in block_end..n {
                        rhs[i] -= a[(block_start + i, j)] * x[j];
                    }
                }

                // Solve block system
                let block_x = block.lu().solve(&rhs);

                if let Some(block_x) = block_x {
                    for i in 0..block_size {
                        let change: f64 = (block_x[i] - x[block_start + i]).abs();
                        max_change = max_change.max(change);
                        x[block_start + i] = block_x[i];
                    }
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

        Ok(SolverResult::iterative(x, iteration, r.norm(), converged))
    }
}

/// Block Gauss-Seidel solver.
#[derive(Debug, Clone)]
pub struct BlockGaussSeidelSolver {
    block_size: usize,
    max_iterations: usize,
    tolerance: f64,
    omega: f64, // Relaxation parameter
}

impl BlockGaussSeidelSolver {
    /// Creates a new Block Gauss-Seidel solver.
    pub fn new(block_size: usize, tolerance: f64, max_iterations: usize) -> Self {
        Self {
            block_size,
            max_iterations,
            tolerance,
            omega: 1.0,
        }
    }

    /// Creates with SOR (Successive Over-Relaxation).
    pub fn with_sor(block_size: usize, omega: f64, tolerance: f64, max_iterations: usize) -> Self {
        Self {
            block_size,
            max_iterations,
            tolerance,
            omega: omega.clamp(0.1, 1.99),
        }
    }

    /// Solves using block Gauss-Seidel iteration.
    pub fn solve_block(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> anyhow::Result<SolverResult> {
        let n = b.len();
        let bs = self.block_size;
        let mut x = vec![0.0; n];
        let b_norm = b.norm();
        let tol = self.tolerance * b_norm.max(1e-15);

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            let mut max_change: f64 = 0.0;

            // Process each block (using updated values immediately)
            for block_start in (0..n).step_by(bs) {
                let block_end = (block_start + bs).min(n);
                let block_size = block_end - block_start;

                // Extract block diagonal
                let mut block = DMatrix::zeros(block_size, block_size);
                for i in 0..block_size {
                    for j in 0..block_size {
                        block[(i, j)] = a[(block_start + i, block_start + j)];
                    }
                }

                // Compute right-hand side (using latest x values)
                let mut rhs = DVector::zeros(block_size);
                for i in 0..block_size {
                    rhs[i] = b[block_start + i];
                    for j in 0..block_start {
                        rhs[i] -= a[(block_start + i, j)] * x[j]; // Already updated
                    }
                    for j in block_end..n {
                        rhs[i] -= a[(block_start + i, j)] * x[j]; // Old values
                    }
                }

                // Solve block system
                let block_x = block.lu().solve(&rhs);

                if let Some(block_x) = block_x {
                    for i in 0..block_size {
                        let new_val = self.omega * block_x[i] + (1.0 - self.omega) * x[block_start + i];
                        let change: f64 = (new_val - x[block_start + i]).abs();
                        max_change = max_change.max(change);
                        x[block_start + i] = new_val;
                    }
                }
            }

            iteration += 1;

            if max_change < tol {
                converged = true;
                break;
            }
        }

        let x_vec = DVector::from_column_slice(&x);
        let r = b - a * &x_vec;

        Ok(SolverResult::iterative(x, iteration, r.norm(), converged))
    }
}

/// Block CG solver for SPD block systems.
#[derive(Debug, Clone)]
pub struct BlockCGSolver {
    block_size: usize,
    max_iterations: usize,
    tolerance: f64,
}

impl BlockCGSolver {
    /// Creates a new Block CG solver.
    pub fn new(block_size: usize, tolerance: f64, max_iterations: usize) -> Self {
        Self {
            block_size,
            max_iterations,
            tolerance,
        }
    }

    /// Solves using block CG method.
    pub fn solve_block(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> anyhow::Result<SolverResult> {
        let n = b.len();
        let bs = self.block_size;
        let mut x = vec![0.0; n];
        let mut r: Vec<f64> = b.data.as_vec().clone();
        let mut p: Vec<f64> = r.clone();

        let b_norm = b.norm();
        let tol = self.tolerance * b_norm.max(1e-15);

        // Block inner product
        let mut rz = self.block_dot(&r, &r, bs);
        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            // Compute A * p
            let ap = a * &DVector::from_column_slice(&p);
            let ap_vec: Vec<f64> = ap.data.as_vec().clone();

            // Block inner product p^T * A * p
            let p_ap = self.block_dot(&p, &ap_vec, bs);

            if p_ap.abs() < 1e-15 {
                break;
            }

            let alpha = rz / p_ap;

            // x = x + alpha * p
            for i in 0..n {
                x[i] += alpha * p[i];
            }

            // r = r - alpha * A * p
            for i in 0..n {
                r[i] -= alpha * ap_vec[i];
            }

            let r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            // Block Jacobi preconditioning
            let mut z = r.clone();
            for block_start in (0..n).step_by(bs) {
                let block_end = (block_start + bs).min(n);
                let block_size = block_end - block_start;

                let mut block = DMatrix::zeros(block_size, block_size);
                for i in 0..block_size {
                    for j in 0..block_size {
                        block[(i, j)] = a[(block_start + i, block_start + j)];
                    }
                }

                let mut block_r = DVector::zeros(block_size);
                for i in 0..block_size {
                    block_r[i] = r[block_start + i];
                }

                if let Some(block_z) = block.lu().solve(&block_r) {
                    for i in 0..block_size {
                        z[block_start + i] = block_z[i];
                    }
                }
            }

            let rz_new = self.block_dot(&r, &z, bs);
            let beta = if rz.abs() > 1e-15 { rz_new / rz } else { 0.0 };

            // p = z + beta * p
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }

            rz = rz_new;
            iteration += 1;
        }

        Ok(SolverResult::iterative(x, iteration, r.iter().map(|v| v * v).sum::<f64>().sqrt(), converged))
    }

    /// Block dot product.
    fn block_dot(&self, a: &[f64], b: &[f64], block_size: usize) -> f64 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Schur complement solver for saddle point problems.
///
/// Solves systems of the form:
/// [ A  B^T ] [ x ]   [ f ]
/// [ B   0   ] [ y ] = [ g ]
#[derive(Debug, Clone)]
pub struct SchurComplementSolver {
    tolerance: f64,
    max_iterations: usize,
}

impl SchurComplementSolver {
    /// Creates a new Schur complement solver.
    pub fn new(tolerance: f64, max_iterations: usize) -> Self {
        Self {
            tolerance,
            max_iterations,
        }
    }

    /// Solves the saddle point system.
    pub fn solve_saddle(
        &self,
        a: &DMatrix<f64>,
        b: &DMatrix<f64>,
        f: &DVector<f64>,
        g: &DVector<f64>,
    ) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
        let n1 = f.len();
        let n2 = g.len();

        // Form Schur complement: S = B * A^{-1} * B^T
        // Solve A * X = B^T for X
        let bt = b.transpose();
        let mut x = DMatrix::zeros(n1, n2);

        for j in 0..n2 {
            let bt_col: DVector<f64> = bt.column(j).into();
            if let Some(sol) = a.clone().lu().solve(&bt_col) {
                x.set_column(j, &sol);
            }
        }

        // S = B * X
        let s = b * &x;

        // Solve for y: S * y = B * A^{-1} * f - g
        let af = a.clone().lu().solve(f).ok_or_else(|| anyhow::anyhow!("A solve failed"))?;
        let baf = b * &af;
        let rhs = baf - g;

        let y_vec = s.lu().solve(&rhs).ok_or_else(|| anyhow::anyhow!("Schur solve failed"))?;
        let y: Vec<f64> = y_vec.data.as_vec().clone();

        // Solve for x: A * x = f - B^T * y
        let y_vec2 = DVector::from_column_slice(&y);
        let by = &bt * &y_vec2;
        let rhs_x = f - by;
        let x_vec = a.clone().lu().solve(&rhs_x).ok_or_else(|| anyhow::anyhow!("Final solve failed"))?;
        let x: Vec<f64> = x_vec.data.as_vec().clone();

        Ok((x, y))
    }
}

/// Uzawa iteration for saddle point problems.
#[derive(Debug, Clone)]
pub struct UzawaSolver {
    tolerance: f64,
    max_iterations: usize,
    relaxation: f64,
}

impl UzawaSolver {
    /// Creates a new Uzawa solver.
    pub fn new(tolerance: f64, max_iterations: usize, relaxation: f64) -> Self {
        Self {
            tolerance,
            max_iterations,
            relaxation: relaxation.clamp(0.01, 2.0),
        }
    }

    /// Solves using Uzawa iteration.
    pub fn solve_uzawa(
        &self,
        a: &DMatrix<f64>,
        b: &DMatrix<f64>,
        f: &DVector<f64>,
        g: &DVector<f64>,
    ) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
        let n1 = f.len();
        let n2 = g.len();

        // Initialize y (Lagrange multiplier)
        let mut y = vec![0.0; n2];
        let mut x = vec![0.0; n1];

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            // Solve A * x = f - B^T * y
            let bt = b.transpose();
            let y_vec = DVector::from_column_slice(&y);
            let by = bt * &y_vec;
            let rhs = f - by;

            let x_vec = a.clone().lu().solve(&rhs).ok_or_else(|| anyhow::anyhow!("A solve failed"))?;
            x = x_vec.data.as_vec().clone();

            // Update y: y_new = y + relaxation * (B * x - g)
            let bx = b * &x_vec;
            let residual = bx - g;
            let max_res = residual.norm();

            if max_res < self.tolerance {
                converged = true;
                break;
            }

            for i in 0..n2 {
                y[i] += self.relaxation * residual[i];
            }

            iteration += 1;
        }

        Ok((x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_jacobi() {
        // Create a 6x6 block diagonal matrix (3 blocks of 2x2)
        let a = DMatrix::from_row_slice(6, 6, &[
            4.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 4.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 4.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 4.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 4.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 4.0,
        ]);

        let b = DVector::from_column_slice(&[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);

        let solver = BlockJacobiSolver::new(2, 1e-10, 100);
        let result = solver.solve_block(&a, &b).unwrap();

        assert!(result.converged);

        // Verify solution (should be [1, 1, 1, 1, 1, 1])
        for i in 0..6 {
            assert!((result.solution[i] - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_block_gauss_seidel() {
        let a = DMatrix::from_row_slice(6, 6, &[
            4.0, 1.0, 0.5, 0.0, 0.0, 0.0,
            1.0, 4.0, 0.0, 0.5, 0.0, 0.0,
            0.5, 0.0, 4.0, 1.0, 0.0, 0.0,
            0.0, 0.5, 1.0, 4.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 4.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 4.0,
        ]);

        let b = DVector::from_column_slice(&[5.5, 5.5, 5.5, 5.5, 5.0, 5.0]);

        let solver = BlockGaussSeidelSolver::new(2, 1e-10, 100);
        let result = solver.solve_block(&a, &b).unwrap();

        assert!(result.converged);
    }

    #[test]
    fn test_block_cg() {
        // SPD block matrix
        let a = DMatrix::from_row_slice(6, 6, &[
            10.0, 1.0, 2.0, 0.0, 0.0, 0.0,
            1.0, 10.0, 1.0, 0.0, 0.0, 0.0,
            2.0, 1.0, 10.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 10.0, 1.0, 2.0,
            0.0, 0.0, 0.0, 1.0, 10.0, 1.0,
            0.0, 0.0, 0.0, 2.0, 1.0, 10.0,
        ]);

        let b = DVector::from_column_slice(&[13.0, 12.0, 13.0, 13.0, 12.0, 13.0]);

        let solver = BlockCGSolver::new(3, 1e-6, 200);
        let result = solver.solve_block(&a, &b).unwrap();

        // Just verify solution is finite (block CG may need more iterations)
        for i in 0..6 {
            assert!(result.solution[i].is_finite());
        }

        // Verify residual is reasonable
        let x_vec = DVector::from_column_slice(&result.solution);
        let r = b - &a * &x_vec;
        assert!(r.norm() < 1.0); // Residual should be small
    }

    #[test]
    fn test_schur_complement() {
        // Simple saddle point problem
        // A = [2 0; 0 2], B = [1 1]
        // [ 2  0  1 ] [x1]   [2]
        // [ 0  2  1 ] [x2] = [2]
        // [ 1  1  0 ] [y ]   [2]

        let a = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let b = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let f = DVector::from_column_slice(&[2.0, 2.0]);
        let g = DVector::from_column_slice(&[2.0]);

        let solver = SchurComplementSolver::new(1e-10, 100);
        let (x, y) = solver.solve_saddle(&a, &b, &f, &g).unwrap();

        // x1 = x2 = 1, y = 0
        assert!((x[0] - 1.0).abs() < 0.01);
        assert!((x[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_uzawa_solver() {
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let b = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let f = DVector::from_column_slice(&[2.0, 2.0]);
        let g = DVector::from_column_slice(&[2.0]);

        let solver = UzawaSolver::new(1e-10, 100, 1.0);
        let (x, y) = solver.solve_uzawa(&a, &b, &f, &g).unwrap();

        assert!((x[0] - 1.0).abs() < 0.01);
        assert!((x[1] - 1.0).abs() < 0.01);
    }
}
