//! Sparse matrix data structures and solvers for large-scale FEA.
//!
//! This module provides:
//! - Compressed Sparse Row (CSR) matrix format
//! - Sparse matrix-vector multiplication
//! - Iterative solvers optimized for sparse systems

use std::fmt::Debug;

/// Compressed Sparse Row (CSR) matrix format.
///
/// Storage format:
/// - `values`: Non-zero values in row-major order
/// - `col_indices`: Column index for each non-zero value
/// - `row_ptr`: Index into values where each row starts
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub nnz: usize,
    /// Non-zero values (length = nnz)
    pub values: Vec<f64>,
    /// Column indices for each non-zero value (length = nnz)
    pub col_indices: Vec<usize>,
    /// Row pointers (length = n_rows + 1)
    pub row_ptr: Vec<usize>,
}

impl CsrMatrix {
    /// Creates a new empty CSR matrix.
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            n_rows,
            n_cols,
            nnz: 0,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptr: vec![0; n_rows + 1],
        }
    }

    /// Creates a CSR matrix from dense matrix.
    pub fn from_dense(matrix: &nalgebra::DMatrix<f64>, tolerance: f64) -> Self {
        let (n_rows, n_cols) = matrix.shape();
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0; n_rows + 1];

        let mut nnz = 0;
        for i in 0..n_rows {
            row_ptr[i] = nnz;
            for j in 0..n_cols {
                let v = matrix[(i, j)];
                if v.abs() > tolerance {
                    values.push(v);
                    col_indices.push(j);
                    nnz += 1;
                }
            }
        }
        row_ptr[n_rows] = nnz;

        Self {
            n_rows,
            n_cols,
            nnz,
            values,
            col_indices,
            row_ptr,
        }
    }

    /// Computes y = alpha * A * x + beta * y
    pub fn gemv(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) {
        assert_eq!(x.len(), self.n_cols);
        assert_eq!(y.len(), self.n_rows);

        // y = beta * y
        if beta == 0.0 {
            y.fill(0.0);
        } else if beta != 1.0 {
            for yi in y.iter_mut() {
                *yi *= beta;
            }
        }

        // y += alpha * A * x
        for i in 0..self.n_rows {
            let row_start = self.row_ptr[i];
            let row_end = self.row_ptr[i + 1];
            let mut sum = 0.0;
            for k in row_start..row_end {
                let j = self.col_indices[k];
                sum += self.values[k] * x[j];
            }
            y[i] += alpha * sum;
        }
    }

    /// Computes A * x (simple case)
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n_rows];
        self.gemv(1.0, x, 0.0, &mut y);
        y
    }

    /// Returns the diagonal of the matrix.
    pub fn diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0.0; self.n_rows.min(self.n_cols)];
        for i in 0..diag.len() {
            let row_start = self.row_ptr[i];
            let row_end = self.row_ptr[i + 1];
            for k in row_start..row_end {
                if self.col_indices[k] == i {
                    diag[i] = self.values[k];
                    break;
                }
            }
        }
        diag
    }

    /// Returns number of non-zeros.
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Returns sparsity (fraction of non-zero elements).
    pub fn sparsity(&self) -> f64 {
        let total = self.n_rows * self.n_cols;
        if total == 0 {
            return 0.0;
        }
        self.nnz as f64 / total as f64
    }

    /// Creates identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut values = Vec::with_capacity(n);
        let mut col_indices = Vec::with_capacity(n);
        let mut row_ptr = Vec::with_capacity(n + 1);

        for i in 0..n {
            row_ptr.push(i);
            values.push(1.0);
            col_indices.push(i);
        }
        row_ptr.push(n);

        Self {
            n_rows: n,
            n_cols: n,
            nnz: n,
            values,
            col_indices,
            row_ptr,
        }
    }
}

/// Builds a CSR matrix from COO (coordinate) format.
pub fn coo_to_csr(
    n_rows: usize,
    n_cols: usize,
    rows: &[usize],
    cols: &[usize],
    values: &[f64],
) -> CsrMatrix {
    assert_eq!(rows.len(), cols.len());
    assert_eq!(rows.len(), values.len());

    let nnz = values.len();
    let mut col_indices = vec![0; nnz];
    let mut sorted_values = vec![0.0; nnz];
    let mut row_ptr = vec![0; n_rows + 1];

    // Count entries per row
    for &r in rows {
        row_ptr[r + 1] += 1;
    }

    // Cumulative sum
    for i in 0..n_rows {
        row_ptr[i + 1] += row_ptr[i];
    }

    // Place values in correct positions
    let mut row_pos = row_ptr.clone();
    for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
        let pos = row_pos[r];
        col_indices[pos] = c;
        sorted_values[pos] = v;
        row_pos[r] += 1;
    }

    CsrMatrix {
        n_rows,
        n_cols,
        nnz,
        values: sorted_values,
        col_indices,
        row_ptr,
    }
}

/// Conjugate Gradient solver for sparse symmetric positive definite systems.
pub struct SparseConjugateGradient {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub use_preconditioner: bool,
}

impl Default for SparseConjugateGradient {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            use_preconditioner: true,
        }
    }
}

impl SparseConjugateGradient {
    pub fn new() -> Self {
        Self::default()
    }

    /// Solves A * x = b using Conjugate Gradient.
    ///
    /// Returns (solution, iterations, converged)
    pub fn solve(&self, a: &CsrMatrix, b: &[f64]) -> (Vec<f64>, usize, bool) {
        let n = b.len();
        if n == 0 {
            return (vec![], 0, true);
        }

        // Initial guess: x = 0
        let mut x = vec![0.0f64; n];
        let mut r = b.to_vec(); // r = b - A*x = b (since x=0)
        let mut p = r.clone();

        // Preconditioner (Jacobi)
        let z = if self.use_preconditioner {
            let diag = a.diagonal();
            let mut z_vec = vec![0.0; n];
            for i in 0..n {
                z_vec[i] = if diag[i].abs() > 1e-15 {
                    r[i] / diag[i]
                } else {
                    r[i]
                };
            }
            z_vec
        } else {
            r.clone()
        };

        let mut rz = dot(&r, &z);
        let b_norm = norm(b);
        let tol = self.tolerance * b_norm.max(1.0);

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            let ap = a.matvec(&p);
            let p_ap = dot(&p, &ap);

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
                r[i] -= alpha * ap[i];
            }

            let r_norm = norm(&r);
            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            // Update preconditioner
            let z_new = if self.use_preconditioner {
                let diag = a.diagonal();
                let mut z_vec = vec![0.0; n];
                for i in 0..n {
                    z_vec[i] = if diag[i].abs() > 1e-15 {
                        r[i] / diag[i]
                    } else {
                        r[i]
                    };
                }
                z_vec
            } else {
                r.clone()
            };

            let rz_new = dot(&r, &z_new);
            let beta = if rz.abs() > 1e-15 {
                rz_new / rz
            } else {
                0.0
            };

            // p = z_new + beta * p
            for i in 0..n {
                p[i] = z_new[i] + beta * p[i];
            }

            rz = rz_new;
            iteration += 1;
        }

        (x, iteration, converged)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_csr_from_dense() {
        let dense = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.0, 2.0,
            0.0, 3.0, 0.0,
            4.0, 0.0, 5.0,
        ]);

        let csr = CsrMatrix::from_dense(&dense, 1e-15);

        assert_eq!(csr.n_rows, 3);
        assert_eq!(csr.n_cols, 3);
        assert_eq!(csr.nnz, 5); // Only non-zeros
    }

    #[test]
    fn test_csr_matvec() {
        let dense = DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 0.0,
            3.0, 4.0, 5.0,
            0.0, 6.0, 7.0,
        ]);

        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let x = vec![1.0, 2.0, 3.0];
        let y = csr.matvec(&x);

        // Expected: [1*1 + 2*2 + 0*3, 3*1 + 4*2 + 5*3, 0*1 + 6*2 + 7*3]
        assert!((y[0] - 5.0).abs() < 1e-12);
        assert!((y[1] - 26.0).abs() < 1e-12);
        assert!((y[2] - 33.0).abs() < 1e-12);
    }

    #[test]
    fn test_csr_diagonal() {
        let dense = DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);

        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let diag = csr.diagonal();

        assert!((diag[0] - 1.0).abs() < 1e-12);
        assert!((diag[1] - 5.0).abs() < 1e-12);
        assert!((diag[2] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_csr_identity() {
        let identity = CsrMatrix::identity(4);
        assert_eq!(identity.n_rows, 4);
        assert_eq!(identity.n_cols, 4);
        assert_eq!(identity.nnz, 4);

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = identity.matvec(&x);
        assert_eq!(x, y);
    }

    #[test]
    fn test_sparse_cg_solver() {
        // Create a symmetric positive definite sparse matrix
        let dense = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 2.0, 0.0,
            1.0, 8.0, 1.0, 0.0,
            2.0, 1.0, 12.0, 1.0,
            0.0, 0.0, 1.0, 6.0,
        ]);

        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let b = vec![13.0, 10.0, 16.0, 7.0];

        // Use solver without preconditioner for this test
        let solver = SparseConjugateGradient {
            max_iterations: 500,
            tolerance: 1e-8,
            use_preconditioner: false,
        };
        let (x, _iterations, _converged) = solver.solve(&csr, &b);

        // Verify solution by checking residual
        let ax = csr.matvec(&x);
        let residual: f64 = ax.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();

        assert!(residual < 1e-4, "Residual should be small, got {}", residual);

        // Verify solution: x should be close to [1, 1, 1, 1]
        for &xi in &x {
            assert!((xi - 1.0).abs() < 0.1, "Solution should be close to 1.0, got {}", xi);
        }
    }

    #[test]
    fn test_sparse_cg_zero_initial_guess() {
        let dense = DMatrix::from_row_slice(3, 3, &[
            4.0, 1.0, 1.0,
            1.0, 4.0, 1.0,
            1.0, 1.0, 4.0,
        ]);

        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let b = vec![6.0, 6.0, 6.0];

        let solver = SparseConjugateGradient {
            max_iterations: 500,
            tolerance: 1e-8,
            use_preconditioner: false,
        };
        let (x, _, _converged) = solver.solve(&csr, &b);

        // Verify solution by checking residual
        let ax = csr.matvec(&x);
        let residual: f64 = ax.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();

        assert!(residual < 1e-4, "Residual should be small, got {}", residual);

        // Solution should be close to [1, 1, 1]
        for &xi in &x {
            assert!((xi - 1.0).abs() < 0.1, "Solution should be close to 1.0, got {}", xi);
        }
    }

    #[test]
    fn test_csr_sparsity() {
        // Dense 3x3 matrix
        let dense = DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let csr_dense = CsrMatrix::from_dense(&dense, 1e-15);
        assert!((csr_dense.sparsity() - 1.0).abs() < 1e-12);

        // Sparse 3x3 matrix (only diagonal)
        let sparse = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0,
        ]);
        let csr_sparse = CsrMatrix::from_dense(&sparse, 1e-15);
        assert!((csr_sparse.sparsity() - 1.0/3.0).abs() < 1e-12);
    }

    #[test]
    fn test_csr_gemv() {
        let dense = DMatrix::from_row_slice(2, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);

        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![1.0, 1.0]; // Initial y

        // y = 2.0 * A * x + 0.5 * y
        csr.gemv(2.0, &x, 0.5, &mut y);

        // A*x = [14, 32]
        // y = 2.0 * [14, 32] + 0.5 * [1, 1] = [28.5, 64.5]
        assert!((y[0] - 28.5).abs() < 1e-12);
        assert!((y[1] - 64.5).abs() < 1e-12);
    }
}
