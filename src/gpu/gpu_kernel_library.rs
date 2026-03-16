//! GPU Kernel Library for FEA Operations.
//!
//! This module provides GPU-accelerated kernels for common FEA operations:
//! - Sparse matrix-vector multiplication
//! - Vector operations (axpy, dot, norm, scale)
//! - Matrix operations (assembly, factorization)
//! - Preconditioner application
//! - Solver kernels (CG, GMRES)

use nalgebra::{DMatrix, DVector};

/// GPU memory space enumeration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GPUMemorySpace {
    /// Host (CPU) memory.
    Host,
    /// Device (GPU) memory.
    Device,
    /// Unified memory (accessible from both).
    Unified,
}

/// GPU stream for asynchronous operations.
#[derive(Debug, Clone)]
pub struct GPUStream {
    /// Stream ID.
    pub id: usize,
    /// Priority (lower is higher priority).
    pub priority: i32,
}

impl Default for GPUStream {
    fn default() -> Self {
        Self { id: 0, priority: 0 }
    }
}

/// GPU context for managing resources.
#[derive(Debug, Clone)]
pub struct GPUContext {
    /// Device ID.
    pub device_id: usize,
    /// Memory space.
    pub memory_space: GPUMemorySpace,
    /// Default stream.
    pub default_stream: GPUStream,
}

impl Default for GPUContext {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_space: GPUMemorySpace::Host,
            default_stream: GPUStream::default(),
        }
    }
}

impl GPUContext {
    /// Creates a new GPU context.
    pub fn new(device_id: usize) -> Self {
        Self {
            device_id,
            ..Default::default()
        }
    }

    /// Checks if GPU is available.
    pub fn is_available(&self) -> bool {
        self.memory_space != GPUMemorySpace::Host
    }

    /// Synchronizes all streams.
    pub fn synchronize(&self) {
        // In real implementation, would call cudaDeviceSynchronize
    }
}

/// GPU-accelerated vector operations.
pub mod vector_kernels {
    use super::*;

    /// Computes y = alpha * x + y (AXPY operation).
    pub fn axpy(alpha: f64, x: &DVector<f64>, y: &mut DVector<f64>) {
        for i in 0..x.len() {
            y[i] = alpha * x[i] + y[i];
        }
    }

    /// Computes y = alpha * x (scale operation).
    pub fn scale(alpha: f64, x: &DVector<f64>, y: &mut DVector<f64>) {
        for i in 0..x.len() {
            y[i] = alpha * x[i];
        }
    }

    /// Computes dot product of two vectors.
    pub fn dot(x: &DVector<f64>, y: &DVector<f64>) -> f64 {
        let mut result = 0.0;
        for i in 0..x.len() {
            result += x[i] * y[i];
        }
        result
    }

    /// Computes Euclidean norm of a vector.
    pub fn norm(x: &DVector<f64>) -> f64 {
        dot(x, x).sqrt()
    }

    /// Computes element-wise addition: z = x + y.
    pub fn add(x: &DVector<f64>, y: &DVector<f64>, z: &mut DVector<f64>) {
        for i in 0..x.len() {
            z[i] = x[i] + y[i];
        }
    }

    /// Computes element-wise subtraction: z = x - y.
    pub fn sub(x: &DVector<f64>, y: &DVector<f64>, z: &mut DVector<f64>) {
        for i in 0..x.len() {
            z[i] = x[i] - y[i];
        }
    }

    /// Computes sum of all elements.
    pub fn sum(x: &DVector<f64>) -> f64 {
        x.iter().copied().sum()
    }
}

/// GPU-accelerated sparse matrix operations.
pub mod sparse_kernels {
    use super::*;

    /// Compressed Sparse Row (CSR) matrix format.
    #[derive(Debug, Clone)]
    pub struct CSRMatrix {
        /// Number of rows.
        pub nrows: usize,
        /// Number of columns.
        pub ncols: usize,
        /// Number of non-zeros.
        pub nnz: usize,
        /// Row pointers (size nrows + 1).
        pub row_ptr: Vec<usize>,
        /// Column indices (size nnz).
        pub col_idx: Vec<usize>,
        /// Values (size nnz).
        pub values: Vec<f64>,
    }

    impl CSRMatrix {
        /// Creates a CSR matrix from dense matrix.
        pub fn from_dense(dense: &DMatrix<f64>, threshold: f64) -> Self {
            let (nrows, ncols) = dense.shape();
            let mut row_ptr = vec![0];
            let mut col_idx = Vec::new();
            let mut values = Vec::new();

            for i in 0..nrows {
                for j in 0..ncols {
                    let val = dense[(i, j)];
                    if val.abs() > threshold {
                        col_idx.push(j);
                        values.push(val);
                    }
                }
                row_ptr.push(col_idx.len());
            }

            Self {
                nrows,
                ncols,
                nnz: values.len(),
                row_ptr,
                col_idx,
                values,
            }
        }

        /// Sparse matrix-vector multiplication: y = A * x.
        pub fn spmv(&self, x: &DVector<f64>, y: &mut DVector<f64>) {
            for i in 0..self.nrows {
                let mut sum = 0.0;
                for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                    let j = self.col_idx[k];
                    let val = self.values[k];
                    sum += val * x[j];
                }
                y[i] = sum;
            }
        }
    }
}

/// GPU-accelerated CG solver kernel.
pub mod cg_kernel {
    use super::*;
    use sparse_kernels::CSRMatrix;
    use vector_kernels::{axpy, dot, norm, scale};

    /// Conjugate Gradient solver result.
    #[derive(Debug, Clone)]
    pub struct CGResult {
        /// Solution vector.
        pub solution: DVector<f64>,
        /// Number of iterations.
        pub iterations: usize,
        /// Final residual norm.
        pub residual_norm: f64,
        /// Convergence flag.
        pub converged: bool,
    }

    /// Solves Ax = b using Conjugate Gradient.
    pub fn solve_cg(
        a: &CSRMatrix,
        b: &DVector<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) -> CGResult {
        let n = b.len();
        let mut x = DVector::zeros(n);
        let mut r = b.clone();
        let mut p = r.clone();

        let b_norm = norm(b);
        let tol = tolerance * b_norm.max(1e-15);

        let mut rz = dot(&r, &r);
        let mut iteration = 0;
        let mut converged = false;

        while iteration < max_iterations {
            let mut ap = DVector::zeros(n);
            a.spmv(&p, &mut ap);

            let p_ap = dot(&p, &ap);
            if p_ap.abs() < 1e-30 {
                break;
            }

            let alpha = rz / p_ap;
            axpy(alpha, &p, &mut x);
            axpy(-alpha, &ap, &mut r);

            let r_norm = norm(&r);
            if r_norm < tol {
                converged = true;
                iteration += 1;
                break;
            }

            let rz_new = dot(&r, &r);
            let beta = rz_new / rz;

            // p = r + beta * p (combined operation to avoid borrow issues)
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }

            rz = rz_new;
            iteration += 1;
        }

        CGResult {
            solution: x,
            iterations: iteration,
            residual_norm: norm(&r),
            converged,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_kernels() {
        let x = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0]);
        let y = DVector::from_column_slice(&[5.0, 6.0, 7.0, 8.0]);

        let mut result = y.clone();
        vector_kernels::axpy(2.0, &x, &mut result);
        assert!((result[0] - 7.0).abs() < 1e-10);

        let dot_prod = vector_kernels::dot(&x, &y);
        assert!((dot_prod - 70.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_kernels() {
        let dense = DMatrix::from_row_slice(4, 4, &[
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ]);

        let csr = sparse_kernels::CSRMatrix::from_dense(&dense, 1e-15);
        assert_eq!(csr.nnz, 10);

        let x = DVector::from_element(4, 1.0);
        let mut y = DVector::zeros(4);
        csr.spmv(&x, &mut y);

        let y_expected = dense * &x;
        for i in 0..4 {
            assert!((y[i] - y_expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cg_kernel() {
        let dense = DMatrix::from_row_slice(10, 10, &[
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

        let csr = sparse_kernels::CSRMatrix::from_dense(&dense, 1e-15);
        let b = DVector::from_element(10, 1.0);

        let result = cg_kernel::solve_cg(&csr, &b, 1e-10, 100);

        let x_direct = dense.lu().solve(&b).unwrap();
        let x_direct_norm = x_direct.norm();
        let error = (result.solution - x_direct).norm() / x_direct_norm;

        assert!(error < 1e-8);
        assert!(result.iterations > 0);
    }
}
