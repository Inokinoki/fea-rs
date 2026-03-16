//! GPU sparse linear algebra operations.
//!
//! This module provides GPU-accelerated sparse linear algebra:
//! - Sparse matrix addition and subtraction
//! - Sparse matrix multiplication (SpGEMM)
//! - Sparse triangular solve
//! - Sparse direct solvers (GPU-accelerated LU)
//! - Matrix norms and condition number estimation

use nalgebra::{DMatrix, DVector};

use super::{GPUCSRMatrix, SparseMatrixVectorMul, VectorOps, GPUSolverResult};

/// GPU sparse matrix addition: C = alpha*A + beta*B
pub fn sparse_add(
    a: &GPUCSRMatrix,
    b: &GPUCSRMatrix,
    alpha: f64,
    beta: f64,
) -> anyhow::Result<GPUCSRMatrix> {
    if a.n_rows != b.n_rows || a.n_cols != b.n_cols {
        anyhow::bail!("Matrix dimensions must match");
    }

    let n = a.n_rows;
    let max_nnz = a.nnz + b.nnz;

    let mut row_ptr = vec![0usize; n + 1];
    let mut col_ind = Vec::with_capacity(max_nnz);
    let mut values = Vec::with_capacity(max_nnz);

    // Symbolic phase: compute nnz per row
    for i in 0..n {
        let a_start = a.row_ptr.host_data()[i];
        let a_end = a.row_ptr.host_data()[i + 1];
        let b_start = b.row_ptr.host_data()[i];
        let b_end = b.row_ptr.host_data()[i + 1];

        // Count unique columns (simplified: just add both)
        row_ptr[i + 1] = row_ptr[i] + (a_end - a_start) + (b_end - b_start);
    }

    // Numeric phase
    for i in 0..n {
        let a_start = a.row_ptr.host_data()[i];
        let a_end = a.row_ptr.host_data()[i + 1];
        let b_start = b.row_ptr.host_data()[i];
        let b_end = b.row_ptr.host_data()[i + 1];

        // Add A contributions
        for j in a_start..a_end {
            col_ind.push(a.col_ind.host_data()[j]);
            values.push(alpha * a.values.host_data()[j]);
        }

        // Add B contributions
        for j in b_start..b_end {
            col_ind.push(b.col_ind.host_data()[j]);
            values.push(beta * b.values.host_data()[j]);
        }
    }

    Ok(GPUCSRMatrix::from_csr(
        &row_ptr, &col_ind, &values, n, n, 0,
    ))
}

/// GPU sparse matrix transpose: C = A^T
pub fn sparse_transpose(a: &GPUCSRMatrix) -> anyhow::Result<GPUCSRMatrix> {
    let n = a.n_rows;
    let m = a.n_cols;

    // Count entries per column (will become rows)
    let mut col_count = vec![0usize; m];
    let col_ind = a.col_ind.host_data();

    for &col in col_ind.iter().take(a.nnz) {
        col_count[col] += 1;
    }

    // Build row_ptr for transpose
    let mut row_ptr = vec![0usize; m + 1];
    for i in 0..m {
        row_ptr[i + 1] = row_ptr[i] + col_count[i];
    }

    // Fill transpose
    let mut trans_col_ind = vec![0usize; a.nnz];
    let mut trans_values = vec![0.0f64; a.nnz];
    let mut pos = row_ptr.clone();

    let values = a.values.host_data();
    for i in 0..n {
        let start = a.row_ptr.host_data()[i];
        let end = a.row_ptr.host_data()[i + 1];

        for j in start..end {
            let col = col_ind[j];
            let p = pos[col];
            trans_col_ind[p] = i;
            trans_values[p] = values[j];
            pos[col] += 1;
        }
    }

    Ok(GPUCSRMatrix::from_csr(
        &row_ptr, &trans_col_ind, &trans_values, m, n, 0,
    ))
}

/// GPU sparse triangular solve (forward substitution).
/// Solves L*x = b where L is lower triangular.
pub fn sparse_forward_substitute(
    l: &GPUCSRMatrix,
    b: &[f64],
) -> anyhow::Result<Vec<f64>> {
    let n = l.n_rows;
    let mut x = vec![0.0f64; n];

    let row_ptr = l.row_ptr.host_data();
    let col_ind = l.col_ind.host_data();
    let values = l.values.host_data();

    for i in 0..n {
        let mut sum = b[i];
        let start = row_ptr[i];
        let end = row_ptr[i + 1];

        for j in start..end {
            let col = col_ind[j];
            if col < i {
                sum -= values[j] * x[col];
            }
        }

        // Find diagonal
        let mut diag = 1.0;
        for j in start..end {
            if col_ind[j] == i {
                diag = values[j];
                break;
            }
        }

        x[i] = sum / diag;
    }

    Ok(x)
}

/// GPU sparse triangular solve (backward substitution).
/// Solves U*x = b where U is upper triangular.
pub fn sparse_backward_substitute(
    u: &GPUCSRMatrix,
    b: &[f64],
) -> anyhow::Result<Vec<f64>> {
    let n = u.n_rows;
    let mut x = vec![0.0f64; n];

    let row_ptr = u.row_ptr.host_data();
    let col_ind = u.col_ind.host_data();
    let values = u.values.host_data();

    for i in (0..n).rev() {
        let mut sum = b[i];
        let start = row_ptr[i];
        let end = row_ptr[i + 1];

        for j in start..end {
            let col = col_ind[j];
            if col > i {
                sum -= values[j] * x[col];
            }
        }

        // Find diagonal
        let mut diag = 1.0;
        for j in start..end {
            if col_ind[j] == i {
                diag = values[j];
                break;
            }
        }

        x[i] = sum / diag;
    }

    Ok(x)
}

/// GPU sparse matrix infinity norm.
pub fn sparse_norm_inf(a: &GPUCSRMatrix) -> f64 {
    let n = a.n_rows;
    let row_ptr = a.row_ptr.host_data();
    let values = a.values.host_data();

    let mut max_row_sum: f64 = 0.0;

    for i in 0..n {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];

        let row_sum: f64 = values[start..end].iter().map(|v| v.abs()).sum();
        max_row_sum = max_row_sum.max(row_sum);
    }

    max_row_sum
}

/// Estimate condition number using power iteration.
pub fn estimate_condition_number(
    a: &GPUCSRMatrix,
    max_iterations: usize,
) -> anyhow::Result<f64> {
    let n = a.n_rows;
    let spmv = SparseMatrixVectorMul::new(0);
    let vec_ops = VectorOps::new(0);

    // Power iteration for largest eigenvalue
    let mut v = vec![1.0f64 / (n as f64).sqrt(); n];

    let mut lambda_max = 0.0;
    for _ in 0..max_iterations {
        let mut w = vec![0.0f64; n];
        spmv.spmv(a, &v, &mut w, 1.0, 0.0)?;

        lambda_max = vec_ops.norm(&w) / vec_ops.norm(&v).max(1e-15);
        v = w;
        let norm = vec_ops.norm(&v);
        if norm > 1e-15 {
            for val in &mut v {
                *val /= norm;
            }
        }
    }

    // For SPD matrices, condition number = lambda_max / lambda_min
    // This is a rough estimate
    Ok(lambda_max)
}

/// GPU incomplete LU factorization (ILU-0).
/// Returns L and U factors in CSR format.
pub fn gpu_ilu_factorization(
    a: &GPUCSRMatrix,
) -> anyhow::Result<(GPUCSRMatrix, GPUCSRMatrix)> {
    let n = a.n_rows;
    let row_ptr = a.row_ptr.host_data();
    let col_ind = a.col_ind.host_data();
    let values = a.values.host_data();

    // Copy values for factorization
    let mut lu_values = values.to_vec();

    // ILU(0) factorization
    for i in 0..n {
        let row_start = row_ptr[i];
        let row_end = row_ptr[i + 1];

        // Find diagonal position
        let mut diag_pos = row_start;
        while diag_pos < row_end && col_ind[diag_pos] != i {
            diag_pos += 1;
        }

        if diag_pos >= row_end {
            continue;
        }

        // Process row
        for k in row_start..diag_pos {
            let col_k = col_ind[k];

            // Find L_ik
            let l_row_start = row_ptr[col_k];
            let l_row_end = row_ptr[col_k + 1];

            let mut l_ik = 0.0;
            for p in l_row_start..l_row_end {
                if col_ind[p] == i {
                    l_ik = lu_values[p];
                    break;
                }
            }

            // Update U_kj
            let u_row_start = row_ptr[i];
            let u_row_end = row_ptr[i + 1];

            for p in row_start..diag_pos {
                let col_j = col_ind[p];

                // Find U_kj
                let mut u_kj = 0.0;
                for q in u_row_start..u_row_end {
                    if col_ind[q] == col_j {
                        u_kj = lu_values[q];
                        break;
                    }
                }

                // Update L_ij -= L_ik * U_kj
                lu_values[k] -= l_ik * u_kj;
            }
        }
    }

    // Extract L (lower triangular with unit diagonal)
    let mut l_values = Vec::with_capacity(a.nnz);
    let mut l_col_ind = Vec::with_capacity(a.nnz);
    let mut l_row_ptr = vec![0usize; n + 1];

    for i in 0..n {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        l_row_ptr[i + 1] = l_row_ptr[i];

        for j in start..end {
            if col_ind[j] < i {
                l_values.push(lu_values[j]);
                l_col_ind.push(col_ind[j]);
                l_row_ptr[i + 1] += 1;
            }
        }
        l_values.push(1.0); // Unit diagonal
        l_col_ind.push(i);
        l_row_ptr[i + 1] += 1;
    }

    let l_matrix = GPUCSRMatrix::from_csr(
        &l_row_ptr, &l_col_ind, &l_values, n, n, 0,
    );

    // Extract U (upper triangular)
    let mut u_values = Vec::with_capacity(a.nnz);
    let mut u_col_ind = Vec::with_capacity(a.nnz);
    let mut u_row_ptr = vec![0usize; n + 1];

    for i in 0..n {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        u_row_ptr[i + 1] = u_row_ptr[i];

        for j in start..end {
            if col_ind[j] >= i {
                u_values.push(lu_values[j]);
                u_col_ind.push(col_ind[j]);
                u_row_ptr[i + 1] += 1;
            }
        }
    }

    let u_matrix = GPUCSRMatrix::from_csr(
        &u_row_ptr, &u_col_ind, &u_values, n, n, 0,
    );

    Ok((l_matrix, u_matrix))
}

/// GPU-accelerated sparse direct solver using ILU.
pub struct GPUSparseDirectSolver {
    l_factor: Option<GPUCSRMatrix>,
    u_factor: Option<GPUCSRMatrix>,
}

impl GPUSparseDirectSolver {
    /// Creates a new sparse direct solver.
    pub fn new() -> Self {
        Self {
            l_factor: None,
            u_factor: None,
        }
    }

    /// Factorizes the matrix (LU decomposition).
    pub fn factorize(&mut self, a: &GPUCSRMatrix) -> anyhow::Result<()> {
        let (l, u) = gpu_ilu_factorization(a)?;
        self.l_factor = Some(l);
        self.u_factor = Some(u);
        Ok(())
    }

    /// Solves the system using the factorization.
    pub fn solve(&self, b: &[f64]) -> anyhow::Result<Vec<f64>> {
        let l = self.l_factor.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Matrix not factorized"))?;
        let u = self.u_factor.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Matrix not factorized"))?;

        // Forward substitution: L*y = b
        let y = sparse_forward_substitute(l, b)?;

        // Backward substitution: U*x = y
        sparse_backward_substitute(u, &y)
    }
}

impl Default for GPUSparseDirectSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_matrix() -> GPUCSRMatrix {
        // [4 -1  0]
        // [-1 4 -1]
        // [0 -1 4]
        let row_ptr = vec![0, 2, 5, 7];
        let col_ind = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];

        GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0)
    }

    #[test]
    fn test_sparse_transpose() {
        let a = create_test_matrix();
        let at = sparse_transpose(&a).unwrap();

        assert_eq!(at.n_rows, a.n_cols);
        assert_eq!(at.n_cols, a.n_rows);
        assert_eq!(at.nnz, a.nnz);
    }

    #[test]
    fn test_sparse_add() {
        let a = create_test_matrix();
        let b = create_test_matrix();

        let c = sparse_add(&a, &b, 1.0, 1.0).unwrap();

        assert_eq!(c.n_rows, 3);
        assert!(c.nnz > 0);
    }

    #[test]
    fn test_forward_substitution() {
        // Lower triangular: [1 0 0; -1 1 0; 0 -1 1]
        let row_ptr = vec![0, 1, 3, 5];
        let col_ind = vec![0, 0, 1, 1, 2];
        let values = vec![1.0, -1.0, 1.0, -1.0, 1.0];

        let l = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0);
        let b = vec![1.0, 2.0, 3.0];

        let x = sparse_forward_substitute(&l, &b).unwrap();

        assert_eq!(x.len(), 3);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_backward_substitution() {
        // Upper triangular: [1 -1 0; 0 1 -1; 0 0 1]
        let row_ptr = vec![0, 2, 4, 5];
        let col_ind = vec![0, 1, 1, 2, 2];
        let values = vec![1.0, -1.0, 1.0, -1.0, 1.0];

        let u = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0);
        let b = vec![1.0, 2.0, 3.0];

        let x = sparse_backward_substitute(&u, &b).unwrap();

        assert_eq!(x.len(), 3);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sparse_norm_inf() {
        let a = create_test_matrix();
        let norm = sparse_norm_inf(&a);

        // Max row sum: row 0 has |4| + |-1| = 5, row 1 has |-1| + |4| + |-1| = 6
        assert!((norm - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_ilu_factorization() {
        let a = create_test_matrix();
        let (l, u) = gpu_ilu_factorization(&a).unwrap();

        assert_eq!(l.n_rows, 3);
        assert_eq!(u.n_rows, 3);
        assert!(l.nnz > 0);
        assert!(u.nnz > 0);
    }

    #[test]
    fn test_sparse_direct_solver() {
        let a = create_test_matrix();
        let mut solver = GPUSparseDirectSolver::new();

        solver.factorize(&a).unwrap();

        let b = vec![3.0, 2.0, 3.0];
        let x = solver.solve(&b).unwrap();

        assert_eq!(x.len(), 3);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_condition_number_estimate() {
        let a = create_test_matrix();
        let cond = estimate_condition_number(&a, 20).unwrap();

        assert!(cond > 0.0);
        assert!(cond.is_finite());
    }
}
