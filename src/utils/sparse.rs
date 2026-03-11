//! Sparse matrix data structures and solvers.
//!
//! Re-exported from the original sparse module.

use std::fmt::Debug;

/// Compressed Sparse Row (CSR) matrix format.
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
    pub fn solve(&self, a: &CsrMatrix, b: &[f64]) -> (Vec<f64>, usize, bool) {
        let n = b.len();
        if n == 0 {
            return (vec![], 0, true);
        }

        let mut x = vec![0.0f64; n];
        let mut r = b.to_vec();
        let mut p = r.clone();

        let diag = a.diagonal();
        let z = if self.use_preconditioner {
            r.iter()
                .enumerate()
                .map(|(i, &ri)| {
                    if diag[i].abs() > 1e-15 {
                        ri / diag[i]
                    } else {
                        ri
                    }
                })
                .collect()
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

            for i in 0..n {
                x[i] += alpha * p[i];
            }

            for i in 0..n {
                r[i] -= alpha * ap[i];
            }

            let r_norm = norm(&r);
            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            let z_new = if self.use_preconditioner {
                let diag = a.diagonal();
                r.iter()
                    .enumerate()
                    .map(|(i, &ri)| {
                        if diag[i].abs() > 1e-15 {
                            ri / diag[i]
                        } else {
                            ri
                        }
                    })
                    .collect()
            } else {
                r.clone()
            };

            let rz_new = dot(&r, &z_new);
            let beta = if rz.abs() > 1e-15 { rz_new / rz } else { 0.0 };

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
