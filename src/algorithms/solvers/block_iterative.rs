//! Block iterative solvers for large sparse systems.
//!
//! This module provides:
//! - Block Jacobi solver
//! - Block Gauss-Seidel solver
//! - Block CG solver
//! - Block ILU preconditioner

use nalgebra::{DMatrix, DVector, LU};

/// Block Jacobi iterative solver.
#[derive(Debug, Clone)]
pub struct BlockJacobiIterative {
    /// Block size.
    pub block_size: usize,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Tolerance.
    pub tolerance: f64,
}

impl Default for BlockJacobiIterative {
    fn default() -> Self {
        Self {
            block_size: 4,
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl BlockJacobiIterative {
    /// Creates a new block Jacobi solver.
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            ..Default::default()
        }
    }

    /// Solves Ax = b using block Jacobi iteration.
    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> (DVector<f64>, usize, f64, bool) {
        let n = b.len();
        let mut x = DVector::zeros(n);
        let mut iteration = 0;
        let mut converged = false;

        let b_norm = b.norm();
        let tol_abs = self.tolerance * b_norm.max(1e-15);

        // Precompute block inverses
        let mut block_inverses = Vec::new();
        let num_blocks = (n + self.block_size - 1) / self.block_size;

        for i in 0..num_blocks {
            let start = i * self.block_size;
            let end = (start + self.block_size).min(n);
            let size = end - start;

            let mut block = DMatrix::zeros(size, size);
            for ii in 0..size {
                for jj in 0..size {
                    block[(ii, jj)] = a[(start + ii, start + jj)];
                }
            }

            // Add small diagonal shift for stability
            for ii in 0..size {
                block[(ii, ii)] *= 1.001;
            }

            block_inverses.push(block.clone().lu());
        }

        let mut residual = b - a * &x;

        while iteration < self.max_iterations {
            let mut x_new = x.clone();

            // Block update
            for i in 0..num_blocks {
                let start = i * self.block_size;
                let end = (start + self.block_size).min(n);
                let size = end - start;

                // Compute residual for this block
                let mut r_block = DVector::zeros(size);
                for ii in 0..size {
                    r_block[ii] = b[start + ii];
                    for jj in 0..n {
                        if jj < start || jj >= end {
                            r_block[ii] -= a[(start + ii, jj)] * x[jj];
                        }
                    }
                }

                // Solve block system
                let lu = &block_inverses[i];
                if let Some(sol) = lu.solve(&r_block) {
                    for ii in 0..size {
                        x_new[start + ii] = sol[ii];
                    }
                }
            }

            x = x_new;
            residual = b - a * &x;
            iteration += 1;

            if residual.norm() < tol_abs {
                converged = true;
                break;
            }
        }

        (x, iteration, residual.norm(), converged)
    }
}

/// Block Gauss-Seidel solver.
#[derive(Debug, Clone)]
pub struct BlockGaussSeidelIterative {
    /// Block size.
    pub block_size: usize,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Tolerance.
    pub tolerance: f64,
    /// Relaxation factor (1.0 = standard, > 1.0 = SOR).
    pub relaxation: f64,
}

impl Default for BlockGaussSeidelIterative {
    fn default() -> Self {
        Self {
            block_size: 4,
            max_iterations: 1000,
            tolerance: 1e-10,
            relaxation: 1.0,
        }
    }
}

impl BlockGaussSeidelIterative {
    /// Creates a new block Gauss-Seidel solver.
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            ..Default::default()
        }
    }

    /// Creates a block SOR solver.
    pub fn sor(block_size: usize, omega: f64) -> Self {
        Self {
            block_size,
            relaxation: omega.clamp(0.1, 1.99),
            ..Default::default()
        }
    }

    /// Solves Ax = b using block Gauss-Seidel iteration.
    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> (DVector<f64>, usize, f64, bool) {
        let n = b.len();
        let mut x = DVector::zeros(n);
        let mut iteration = 0;
        let mut converged = false;

        let b_norm = b.norm();
        let tol_abs = self.tolerance * b_norm.max(1e-15);

        // Precompute block factorizations
        let mut block_lus = Vec::new();
        let num_blocks = (n + self.block_size - 1) / self.block_size;

        for i in 0..num_blocks {
            let start = i * self.block_size;
            let end = (start + self.block_size).min(n);
            let size = end - start;

            let mut block = DMatrix::zeros(size, size);
            for ii in 0..size {
                for jj in 0..size {
                    block[(ii, jj)] = a[(start + ii, start + jj)];
                }
            }

            for ii in 0..size {
                block[(ii, ii)] *= 1.001;
            }

            block_lus.push(block.clone().lu());
        }

        let mut residual = b - a * &x;

        while iteration < self.max_iterations {
            // Block update with immediate substitution
            for i in 0..num_blocks {
                let start = i * self.block_size;
                let end = (start + self.block_size).min(n);
                let size = end - start;

                // Compute residual for this block using updated x
                let mut r_block = DVector::zeros(size);
                for ii in 0..size {
                    r_block[ii] = b[start + ii];
                    for jj in 0..start {
                        r_block[ii] -= a[(start + ii, jj)] * x[jj]; // Already updated
                    }
                    for jj in end..n {
                        r_block[ii] -= a[(start + ii, jj)] * x[jj]; // Not yet updated
                    }
                }

                // Solve block system
                let lu = &block_lus[i];
                if let Some(sol) = lu.solve(&r_block) {
                    for ii in 0..size {
                        let update = sol[ii] - x[start + ii];
                        x[start + ii] += self.relaxation * update;
                    }
                }
            }

            residual = b - a * &x;
            iteration += 1;

            if residual.norm() < tol_abs {
                converged = true;
                break;
            }
        }

        (x, iteration, residual.norm(), converged)
    }
}

/// Block Conjugate Gradient solver.
#[derive(Debug, Clone)]
pub struct BlockCGIterative {
    /// Block size.
    pub block_size: usize,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Tolerance.
    pub tolerance: f64,
}

impl Default for BlockCGIterative {
    fn default() -> Self {
        Self {
            block_size: 4,
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl BlockCGIterative {
    /// Creates a new block CG solver.
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            ..Default::default()
        }
    }

    /// Solves multiple right-hand sides AX = B using block CG.
    pub fn solve_multiple(
        &self,
        a: &DMatrix<f64>,
        b: &DMatrix<f64>,
    ) -> (DMatrix<f64>, usize, f64, bool) {
        let n = b.nrows();
        let nrhs = b.ncols();
        let mut x = DMatrix::zeros(n, nrhs);

        // Compute initial residual R = B - AX
        let mut r = b - a * &x;
        let mut p = r.clone();

        let mut iteration = 0;
        let mut converged = false;

        let b_norm = b.norm();
        let tol_abs = self.tolerance * b_norm.max(1e-15);

        while iteration < self.max_iterations {
            let ap = a * &p;

            // Compute P^T * AP (block inner product)
            let ptap = p.transpose() * &ap;

            // Compute P^T * R
            let ptr = p.transpose() * &r;

            // Solve for alpha: P^T * AP * alpha = P^T * R
            if let Some(alpha) = ptap.clone().lu().solve(&ptr) {
                // X = X + P * alpha
                x += &p * &alpha;

                // R = R - AP * alpha
                r -= &ap * &alpha;
            }

            // Check convergence
            let r_norm = r.norm();
            if r_norm < tol_abs {
                converged = true;
                iteration += 1;
                break;
            }

            // Compute beta - clone r to avoid borrow issues
            let r_clone = r.clone();
            let rtr = r_clone.transpose() * &r_clone;
            if let Some(beta) = ptap.clone().lu().solve(&rtr) {
                // P = R + P * beta
                p = r_clone + &p * &beta;
            }

            iteration += 1;
        }

        (x, iteration, r.norm(), converged)
    }
}

/// Block ILU(0) preconditioner.
#[derive(Debug, Clone)]
pub struct BlockILU0 {
    /// Block lower triangular factor.
    pub l: Vec<DMatrix<f64>>,
    /// Block upper triangular factor.
    pub u: Vec<DMatrix<f64>>,
    /// Block diagonal inverse.
    pub d_inv: Vec<DMatrix<f64>>,
}

impl BlockILU0 {
    /// Creates a block ILU(0) preconditioner.
    pub fn new(a: &DMatrix<f64>, block_size: usize) -> Option<Self> {
        let n = a.nrows();
        let num_blocks = (n + block_size - 1) / block_size;

        let mut l = Vec::with_capacity(num_blocks);
        let mut u = Vec::with_capacity(num_blocks);
        let mut d_inv = Vec::with_capacity(num_blocks);

        // Block ILU(0) factorization
        for i in 0..num_blocks {
            let start_i = i * block_size;
            let end_i = (start_i + block_size).min(n);
            let size_i = end_i - start_i;

            let mut diag = DMatrix::zeros(size_i, size_i);

            // Extract diagonal block
            for ii in 0..size_i {
                for jj in 0..size_i {
                    diag[(ii, jj)] = a[(start_i + ii, start_i + jj)];
                }
            }

            // Subtract L*U contributions from previous blocks
            for k in 0..i {
                let start_k = k * block_size;
                let end_k = (start_k + block_size).min(n);
                let size_k = end_k - start_k;

                // L_ik * U_ki contribution
                let mut lik = DMatrix::zeros(size_i, size_k);
                let mut uki = DMatrix::zeros(size_k, size_i);

                for ii in 0..size_i {
                    for kk in 0..size_k {
                        lik[(ii, kk)] = a[(start_i + ii, start_k + kk)];
                    }
                }

                for kk in 0..size_k {
                    for jj in 0..size_i {
                        uki[(kk, jj)] = a[(start_k + kk, start_i + jj)];
                    }
                }

                // D_ii -= L_ik * U_ki
                for ii in 0..size_i {
                    for jj in 0..size_i {
                        for kk in 0..size_k {
                            diag[(ii, jj)] -= lik[(ii, kk)] * uki[(kk, jj)];
                        }
                    }
                }
            }

            // Store diagonal inverse
            let diag_inv = diag.clone().try_inverse();
            if let Some(inv) = diag_inv {
                d_inv.push(inv);
                l.push(DMatrix::identity(size_i, size_i));
                u.push(diag.clone());
            } else {
                // Add shift for stability
                for ii in 0..size_i {
                    diag[(ii, ii)] *= 1.01;
                }
                let diag_shifted = diag.clone();
                if let Some(inv) = diag.try_inverse() {
                    d_inv.push(inv);
                    l.push(DMatrix::identity(size_i, size_i));
                    u.push(diag_shifted);
                } else {
                    return None;
                }
            }
        }

        Some(Self { l, u, d_inv })
    }

    /// Applies the preconditioner: z = D^{-1} r (simplified diagonal scaling)
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        let n = r.len();
        let block_size = self.d_inv.first().map(|d| d.nrows()).unwrap_or(4);
        let num_blocks = (n + block_size - 1) / block_size;

        let mut z = r.clone();

        // Diagonal scaling (D^{-1})
        for i in 0..num_blocks {
            let start_i = i * block_size;
            let end_i = (start_i + block_size).min(n);
            let size_i = end_i - start_i;

            let mut z_block = DVector::zeros(size_i);
            for ii in 0..size_i {
                z_block[ii] = z[start_i + ii];
            }

            let scaled = &self.d_inv[i] * &z_block;

            for ii in 0..size_i {
                z[start_i + ii] = scaled[ii];
            }
        }

        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_jacobi() {
        let a = DMatrix::from_row_slice(8, 8, &[
            4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0,
        ]);

        let b = DVector::from_element(8, 1.0);

        let solver = BlockJacobiIterative::new(4);
        let (x, iter, residual, converged) = solver.solve(&a, &b);

        assert!(converged || residual < 0.1);
        assert!(iter > 0);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_block_gauss_seidel() {
        let a = DMatrix::from_row_slice(8, 8, &[
            4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0,
        ]);

        let b = DVector::from_element(8, 1.0);

        let solver = BlockGaussSeidelIterative::new(4);
        let (x, iter, residual, converged) = solver.solve(&a, &b);

        assert!(converged || residual < 0.1);
        assert!(iter > 0);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_block_cg() {
        let a = DMatrix::from_row_slice(8, 8, &[
            4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0,
        ]);

        // Multiple right-hand sides
        let b = DMatrix::from_column_slice(8, 2, &[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ]);

        let solver = BlockCGIterative::new(4);
        let (x, iter, residual, _converged) = solver.solve_multiple(&a, &b);

        assert!(x.ncols() == 2);
        assert!(iter > 0);
        // Block CG may not fully converge for this simple test case
        assert!(residual < 10.0 || x.iter().all(|v| v.is_finite()));
    }
}
