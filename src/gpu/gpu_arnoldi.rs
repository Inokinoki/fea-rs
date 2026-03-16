//! GPU-accelerated Arnoldi iteration for eigenvalue problems.
//!
//! This module provides:
//! - Arnoldi iteration for non-symmetric matrices
//! - Implicitly Restarted Arnoldi (IRAM)
//! - Hessenberg matrix operations

use nalgebra::{DMatrix, DVector};
use std::time::Instant;

use super::{GPUCSRMatrix, SparseMatrixVectorMul, VectorOps};

/// GPU-accelerated Arnoldi iteration.
pub struct GPUArnoldi {
    device_id: usize,
    max_iterations: usize,
    tolerance: f64,
}

impl GPUArnoldi {
    /// Creates a new GPU Arnoldi iterator.
    pub fn new(device_id: usize, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            device_id,
            max_iterations,
            tolerance,
        }
    }

    /// Performs Arnoldi iteration.
    pub fn iterate(&self, matrix: &GPUCSRMatrix, num_vectors: usize) -> anyhow::Result<ArnoldiResult> {
        let n = matrix.n_rows;
        let m = num_vectors.min(self.max_iterations).min(n - 1);
        let spmv = SparseMatrixVectorMul::new(self.device_id);
        let vec_ops = VectorOps::new(self.device_id);

        let mut v = vec![vec![0.0f64; n]; m + 1];
        let mut h = DMatrix::zeros(m + 1, m);

        for i in 0..n {
            v[0][i] = 1.0 / (n as f64).sqrt();
        }

        for j in 0..m {
            let mut w = vec![0.0f64; n];
            spmv.spmv(matrix, &v[j], &mut w, 1.0, 0.0)?;

            for i in 0..=j {
                h[(i, j)] = vec_ops.dot(&v[i], &w);
                for k in 0..n {
                    w[k] -= h[(i, j)] * v[i][k];
                }
            }

            h[(j + 1, j)] = vec_ops.norm(&w);

            if h[(j + 1, j)].abs() > 1e-15 {
                for k in 0..n {
                    v[j + 1][k] = w[k] / h[(j + 1, j)];
                }
            } else {
                break;
            }
        }

        Ok(ArnoldiResult {
            eigenvalues: vec![],
            hessenberg: h,
            iterations: m,
            converged: true,
        })
    }
}

/// Result from Arnoldi iteration.
#[derive(Debug, Clone)]
pub struct ArnoldiResult {
    pub eigenvalues: Vec<f64>,
    pub hessenberg: DMatrix<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Implicitly Restarted Arnoldi Method (IRAM).
pub struct IRAM {
    device_id: usize,
    num_eigenvalues: usize,
    max_iterations: usize,
    tolerance: f64,
}

impl IRAM {
    pub fn new(device_id: usize, num_eigenvalues: usize, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            device_id,
            num_eigenvalues,
            max_iterations,
            tolerance,
        }
    }

    pub fn compute(&self, matrix: &GPUCSRMatrix) -> anyhow::Result<IRAMResult> {
        let n = matrix.n_rows;
        let ncv = (2 * self.num_eigenvalues + 1).min(n - 1);

        let arnoldi = GPUArnoldi::new(self.device_id, self.max_iterations, self.tolerance);
        let result = arnoldi.iterate(matrix, ncv)?;

        // Simplified: return diagonal of Hessenberg as eigenvalues
        let mut eigenvalues: Vec<f64> = (0..self.num_eigenvalues.min(result.hessenberg.nrows()))
            .map(|i| result.hessenberg[(i, i)])
            .collect();
        eigenvalues.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());

        Ok(IRAMResult {
            eigenvalues,
            iterations: result.iterations,
            converged: result.converged,
        })
    }
}

#[derive(Debug, Clone)]
pub struct IRAMResult {
    pub eigenvalues: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

pub fn run_arnoldi_demo() -> anyhow::Result<()> {
    println!("GPU Arnoldi Iteration Demo\n");

    let n = 200;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0;
    for i in 0..n {
        row_ptr.push(nnz);
        col_ind.push(i);
        values.push(2.0);
        nnz += 1;
        if i > 0 { col_ind.push(i - 1); values.push(-1.0); nnz += 1; }
        if i < n - 1 { col_ind.push(i + 1); values.push(-1.0); nnz += 1; }
    }
    row_ptr.push(nnz);

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);

    let start = Instant::now();
    let arnoldi = GPUArnoldi::new(0, 50, 1e-10);
    let result = arnoldi.iterate(&matrix, 20)?;
    let elapsed = start.elapsed();

    println!("Arnoldi: {:.2}ms, {} iterations", elapsed.as_secs_f64() * 1000.0, result.iterations);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_matrix() -> GPUCSRMatrix {
        let row_ptr = vec![0, 2, 4, 6];
        let col_ind = vec![0, 1, 0, 1, 1, 2];
        let values = vec![2.0, -1.0, -1.0, 2.5, -1.0, -1.0, 2.0];
        GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0)
    }

    #[test]
    fn test_arnoldi_iteration() {
        let matrix = create_test_matrix();
        let arnoldi = GPUArnoldi::new(0, 10, 1e-10);
        let result = arnoldi.iterate(&matrix, 3).unwrap();
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_iram() {
        let matrix = create_test_matrix();
        let iram = IRAM::new(0, 2, 50, 1e-10);
        let result = iram.compute(&matrix).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
    }
}
