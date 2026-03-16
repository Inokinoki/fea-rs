//! GPU acceleration module for FEA computations.
//!
//! This module provides:
//! - GPU memory management for CUDA and OpenCL backends
//! - Sparse matrix operations on GPU
//! - Parallel assembly of stiffness matrices
//! - GPU-accelerated iterative solvers
//!
//! # Feature Flags
//!
//! - `gpu-cuda` - CUDA backend support
//! - `gpu-opencl` - OpenCL backend support
//! - `gpu` - Enable all GPU backends

pub mod cuda;
pub mod opencl;
pub mod kernels;
pub mod opencl_kernels;
pub mod gpu_solvers;
pub mod multi_gpu;
pub mod gpu_precond;
pub mod gpu_eigen;
pub mod gpu_fft;
pub mod opencl_utils;
pub mod gpu_sparse_la;
pub mod gpu_nonlinear;
pub mod gpu_advanced;
pub mod gpu_nonlinear_dynamics;
pub mod gpu_kernel_optimizer;
pub mod gpu_cg_advanced;
pub mod gpu_amg;
pub mod gpu_sparse_direct;
pub mod gpu_rom;
pub mod gpu_eigen_enhanced;
pub mod gpu_kernels_extra;
pub mod gpu_arnoldi;
pub mod gpu_advanced_solvers;
pub mod gpu_multigrid;
pub mod gpu_kernels_complete;
pub mod gpu_sparse_kernels;
pub mod gpu_postprocessing_kernels;
pub mod gpu_preprocessing_kernels;
pub mod gpu_kernels_lib;
pub mod gpu_kernel_library;
pub mod gpu_thermal;

use std::fmt;

/// GPU device information.
#[derive(Debug, Clone)]
pub struct GPUDevice {
    /// Device name.
    pub name: String,
    /// Device type (CUDA or OpenCL).
    pub device_type: DeviceType,
    /// Compute capability (major, minor) for CUDA, or (version, 0) for OpenCL.
    pub compute_capability: (u32, u32),
    /// Global memory in GB.
    pub global_memory_gb: f64,
    /// Number of multiprocessors (CUDA) or compute units (OpenCL).
    pub num_multiprocessors: u32,
    /// Maximum threads per block.
    pub max_threads_per_block: usize,
    /// Device ID.
    pub device_id: usize,
}

/// Type of GPU device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// NVIDIA CUDA device.
    CUDA,
    /// OpenCL device.
    OpenCL,
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::CUDA => write!(f, "CUDA"),
            DeviceType::OpenCL => write!(f, "OpenCL"),
        }
    }
}

impl GPUDevice {
    /// Returns list of available GPU devices.
    pub fn available_devices() -> Vec<Self> {
        // In real implementation, would query CUDA/OpenCL
        // Return empty for now (system may not have GPUs)
        vec![]
    }
}

/// GPU memory buffer for device-side data.
#[derive(Debug, Clone)]
pub struct GPUMemory<T> {
    /// Host-side data buffer.
    host_data: Vec<T>,
    /// Size of the buffer.
    size: usize,
    /// Device ID this memory is allocated on.
    device_id: usize,
    /// Whether data has been synced to device.
    synced_to_device: bool,
    /// Whether data has been synced from device.
    synced_from_device: bool,
}

impl<T: Clone + Copy + Zeroable> GPUMemory<T> {
    /// Creates a new GPU memory buffer.
    pub fn new(size: usize, device_id: usize) -> Self {
        Self {
            host_data: vec![T::new(); size],
            size,
            device_id,
            synced_to_device: false,
            synced_from_device: false,
        }
    }

    /// Creates a new GPU memory buffer from existing data.
    pub fn from_data(data: &[T], device_id: usize) -> Self {
        Self {
            host_data: data.to_vec(),
            size: data.len(),
            device_id,
            synced_to_device: false,
            synced_from_device: false,
        }
    }

    /// Returns the size of the buffer.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns a reference to the host data.
    pub fn host_data(&self) -> &[T] {
        &self.host_data
    }

    /// Returns a mutable reference to the host data.
    pub fn host_data_mut(&mut self) -> &mut [T] {
        &mut self.host_data
    }

    /// Marks data as synced to device (called by async transfer).
    pub fn mark_synced_to_device(&mut self) {
        self.synced_to_device = true;
    }

    /// Marks data as synced from device (called by async transfer).
    pub fn mark_synced_from_device(&mut self) {
        self.synced_from_device = true;
    }

    /// Checks if data is synced to device.
    pub fn is_synced_to_device(&self) -> bool {
        self.synced_to_device
    }

    /// Checks if data is synced from device.
    pub fn is_synced_from_device(&self) -> bool {
        self.synced_from_device
    }
}

impl<T: Clone + Copy + Default> GPUMemory<T> {
    /// Creates a new GPU memory buffer with default values.
    pub fn zeros(size: usize, device_id: usize) -> Self {
        Self {
            host_data: vec![T::default(); size],
            size,
            device_id,
            synced_to_device: false,
            synced_from_device: false,
        }
    }
}

// Helper trait for types that can have a "zero" value.
pub trait Zeroable {
    fn new() -> Self;
}

impl Zeroable for f64 {
    fn new() -> Self {
        0.0
    }
}

impl Zeroable for f32 {
    fn new() -> Self {
        0.0
    }
}

impl Zeroable for i32 {
    fn new() -> Self {
        0
    }
}

impl Zeroable for i64 {
    fn new() -> Self {
        0
    }
}

impl Zeroable for usize {
    fn new() -> Self {
        0
    }
}

/// GPU stream for asynchronous operations.
#[derive(Debug, Clone)]
pub struct GPUStream {
    /// Stream ID.
    stream_id: usize,
    /// Device ID.
    device_id: usize,
    /// Whether the stream is synchronized.
    synchronized: bool,
}

impl GPUStream {
    /// Creates a new GPU stream.
    pub fn new(device_id: usize) -> Self {
        Self {
            stream_id: 0,
            device_id,
            synchronized: true,
        }
    }

    /// Creates a new async GPU stream.
    pub fn async_stream(device_id: usize, stream_id: usize) -> Self {
        Self {
            stream_id,
            device_id,
            synchronized: false,
        }
    }

    /// Waits for the stream to complete.
    pub fn synchronize(&self) {
        // In a real implementation, this would call cudaStreamSynchronize
        // or clFinish depending on the backend
    }

    /// Returns the device ID.
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Returns whether the stream is synchronized.
    pub fn is_synchronized(&self) -> bool {
        self.synchronized
    }
}

/// CSR (Compressed Sparse Row) matrix format for GPU.
#[derive(Debug, Clone)]
pub struct GPUCSRMatrix {
    /// Row pointers (size: n_rows + 1).
    pub row_ptr: GPUMemory<usize>,
    /// Column indices (size: nnz).
    pub col_ind: GPUMemory<usize>,
    /// Non-zero values (size: nnz).
    pub values: GPUMemory<f64>,
    /// Number of rows.
    pub n_rows: usize,
    /// Number of columns.
    pub n_cols: usize,
    /// Number of non-zero elements.
    pub nnz: usize,
}

impl GPUCSRMatrix {
    /// Creates a new GPU CSR matrix from CSR data.
    pub fn from_csr(
        row_ptr: &[usize],
        col_ind: &[usize],
        values: &[f64],
        n_rows: usize,
        n_cols: usize,
        device_id: usize,
    ) -> Self {
        let nnz = values.len();
        Self {
            row_ptr: GPUMemory::from_data(row_ptr, device_id),
            col_ind: GPUMemory::from_data(col_ind, device_id),
            values: GPUMemory::from_data(values, device_id),
            n_rows,
            n_cols,
            nnz,
        }
    }

    /// Returns the number of rows.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Returns the number of columns.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Returns the number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.nnz
    }
}

/// GPU-accelerated sparse matrix-vector multiplication (SpMV).
pub struct SparseMatrixVectorMul {
    /// Device ID.
    device_id: usize,
}

impl SparseMatrixVectorMul {
    /// Creates a new SpMV operator.
    pub fn new(device_id: usize) -> Self {
        Self { device_id }
    }

    /// Performs sparse matrix-vector multiplication: y = alpha * A * x + beta * y
    ///
    /// This is a CPU implementation that would be replaced with CUDA/OpenCL kernels.
    pub fn spmv(
        &self,
        matrix: &GPUCSRMatrix,
        x: &[f64],
        y: &mut [f64],
        alpha: f64,
        beta: f64,
    ) -> anyhow::Result<()> {
        // CPU fallback implementation
        // In real GPU implementation, this would be a kernel launch

        // First scale y by beta
        if beta != 1.0 {
            for val in y.iter_mut() {
                *val *= beta;
            }
        }

        // Perform SpMV: y += alpha * A * x
        for i in 0..matrix.n_rows {
            let row_start = matrix.row_ptr.host_data()[i];
            let row_end = matrix.row_ptr.host_data()[i + 1];

            let mut sum = 0.0;
            for j in row_start..row_end {
                let col = matrix.col_ind.host_data()[j];
                let val = matrix.values.host_data()[j];
                sum += val * x[col];
            }

            y[i] += alpha * sum;
        }

        Ok(())
    }

    /// Performs SpMV with y = A * x (simplified).
    pub fn spmv_simple(&self, matrix: &GPUCSRMatrix, x: &[f64]) -> anyhow::Result<Vec<f64>> {
        let mut y = vec![0.0; matrix.n_rows];
        self.spmv(matrix, x, &mut y, 1.0, 0.0)?;
        Ok(y)
    }
}

/// GPU-accelerated vector operations.
pub struct VectorOps {
    device_id: usize,
}

impl VectorOps {
    /// Creates new vector operations.
    pub fn new(device_id: usize) -> Self {
        Self { device_id }
    }

    /// Computes dot product of two vectors.
    pub fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Computes the norm of a vector.
    pub fn norm(&self, v: &[f64]) -> f64 {
        self.dot(v, v).sqrt()
    }

    /// Computes y = alpha * x + y (AXPY operation).
    pub fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        for i in 0..x.len() {
            y[i] += alpha * x[i];
        }
    }

    /// Computes y = alpha * x (scaling operation).
    pub fn scale(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        for i in 0..x.len() {
            y[i] = alpha * x[i];
        }
    }

    /// Copies one vector to another.
    pub fn copy(&self, src: &[f64], dst: &mut [f64]) {
        dst.copy_from_slice(src);
    }

    /// Sets all elements to a value.
    pub fn set(&self, value: f64, dst: &mut [f64]) {
        for val in dst.iter_mut() {
            *val = value;
        }
    }
}

/// GPU context for managing device resources.
#[derive(Debug, Clone)]
pub struct GPUContext {
    /// Device ID.
    device_id: usize,
    /// Device type.
    device_type: DeviceType,
    /// Whether the context is initialized.
    initialized: bool,
}

impl GPUContext {
    /// Creates a new GPU context.
    pub fn new(device_id: usize, device_type: DeviceType) -> Self {
        Self {
            device_id,
            device_type,
            initialized: false,
        }
    }

    /// Initializes the GPU context.
    pub fn initialize(&mut self) -> anyhow::Result<()> {
        // In a real implementation, this would:
        // - For CUDA: call cudaSetDevice and initialize cuBLAS, cuSPARSE
        // - For OpenCL: create context, command queue, etc.
        self.initialized = true;
        Ok(())
    }

    /// Returns whether the context is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Returns the device ID.
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Returns the device type.
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }
}

/// GPU-accelerated CG solver.
pub struct GPUCGSolver {
    /// Device ID.
    device_id: usize,
    /// Maximum iterations.
    max_iterations: usize,
    /// Tolerance.
    tolerance: f64,
    /// Vector operations.
    vector_ops: VectorOps,
    /// SpMV operator.
    spmv: SparseMatrixVectorMul,
}

impl GPUCGSolver {
    /// Creates a new GPU CG solver.
    pub fn new(device_id: usize, tolerance: f64, max_iterations: usize) -> Self {
        Self {
            device_id,
            max_iterations,
            tolerance,
            vector_ops: VectorOps::new(device_id),
            spmv: SparseMatrixVectorMul::new(device_id),
        }
    }

    /// Solves the system Ax = b using CG method.
    pub fn solve(
        &self,
        matrix: &GPUCSRMatrix,
        b: &[f64],
        x: &mut [f64],
    ) -> anyhow::Result<GPUSolverResult> {
        let n = b.len();
        let mut r = vec![0.0; n];
        let mut p = vec![0.0; n];
        let mut z = vec![0.0; n];

        // Compute initial residual r = b - Ax
        self.spmv.spmv(matrix, x, &mut r, -1.0, 1.0)?;
        self.vector_ops.axpy(1.0, b, &mut r);

        // Initial search direction p = r
        self.vector_ops.copy(&r, &mut p);

        let mut rho = self.vector_ops.dot(&r, &r);
        let b_norm = self.vector_ops.norm(b);
        let tol = self.tolerance * b_norm.max(1e-15);

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            // Compute Ap
            let mut ap = vec![0.0; n];
            self.spmv.spmv(matrix, &p, &mut ap, 1.0, 0.0)?;

            let p_ap = self.vector_ops.dot(&p, &ap);
            if p_ap.abs() < 1e-15 {
                break;
            }

            let alpha = rho / p_ap;

            // x = x + alpha * p
            self.vector_ops.axpy(alpha, &p, x);

            // r = r - alpha * Ap
            self.vector_ops.axpy(-alpha, &ap, &mut r);

            let r_norm = self.vector_ops.norm(&r);
            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            // Jacobi preconditioning (simplified)
            for i in 0..n {
                z[i] = r[i] / matrix.values.host_data()[i].max(1e-15);
            }

            let rho_new = self.vector_ops.dot(&r, &z);
            let beta = if rho.abs() > 1e-15 {
                rho_new / rho
            } else {
                0.0
            };

            // p = z + beta * p
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }

            rho = rho_new;
            iteration += 1;
        }

        Ok(GPUSolverResult {
            iterations: iteration,
            residual_norm: self.vector_ops.norm(&r),
            converged,
        })
    }
}

/// Result of a GPU solver operation.
#[derive(Debug, Clone)]
pub struct GPUSolverResult {
    /// Number of iterations.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// List available GPU devices.
pub fn list_gpu_devices() -> Vec<GPUDevice> {
    // In a real implementation, this would query CUDA/OpenCL devices
    // For now, return empty vector as placeholder
    vec![]
}

/// Check if GPU acceleration is available.
pub fn gpu_available() -> bool {
    // In a real implementation, this would check for CUDA/OpenCL runtime
    !list_gpu_devices().is_empty()
}

/// Create a GPU context for the specified device.
pub fn create_gpu_context(device_id: usize, device_type: DeviceType) -> anyhow::Result<GPUContext> {
    let mut ctx = GPUContext::new(device_id, device_type);
    ctx.initialize()?;
    Ok(ctx)
}

/// GPU-accelerated stiffness matrix assembly.
pub mod assembly {
    use super::*;

    /// Assembles element stiffness matrices into global matrix on GPU.
    ///
    /// This is a high-level interface that would dispatch to CUDA/OpenCL kernels.
    pub fn assemble_stiffness_matrix(
        element_matrices: &[Vec<f64>],
        element_connectivity: &[Vec<usize>],
        n_dofs: usize,
        device_id: usize,
    ) -> anyhow::Result<GPUCSRMatrix> {
        // CPU implementation for assembly - would be GPU in real implementation
        let mut csr_builder = CSRBuilder::new(n_dofs);

        for (elem_idx, conn) in element_connectivity.iter().enumerate() {
            let ke = &element_matrices[elem_idx];
            let n_nodes = conn.len();

            for i in 0..n_nodes {
                for j in 0..n_nodes {
                    // Assuming 2 DOFs per node for 2D problems
                    for dof_i in 0..2 {
                        for dof_j in 0..2 {
                            let global_i = conn[i] * 2 + dof_i;
                            let global_j = conn[j] * 2 + dof_j;
                            let local_idx = (i * 2 + dof_i) * (n_nodes * 2) + (j * 2 + dof_j);
                            csr_builder.add(global_i, global_j, ke[local_idx]);
                        }
                    }
                }
            }
        }

        let (row_ptr, col_ind, values) = csr_builder.build_csr();
        Ok(GPUCSRMatrix::from_csr(
            &row_ptr,
            &col_ind,
            &values,
            n_dofs,
            n_dofs,
            device_id,
        ))
    }

    /// CSR matrix builder for assembly.
    struct CSRBuilder {
        n_rows: usize,
        entries: Vec<(usize, usize, f64)>,
    }

    impl CSRBuilder {
        fn new(n_rows: usize) -> Self {
            Self {
                n_rows,
                entries: Vec::new(),
            }
        }

        fn add(&mut self, row: usize, col: usize, val: f64) {
            self.entries.push((row, col, val));
        }

        fn build_csr(self) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
            let mut entries = self.entries;
            entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

            let mut row_ptr = vec![0usize; self.n_rows + 1];
            let mut col_ind = Vec::with_capacity(entries.len());
            let mut values = Vec::with_capacity(entries.len());

            let mut current_row = 0;
            for (i, (row, col, val)) in entries.into_iter().enumerate() {
                while current_row < row {
                    row_ptr[current_row + 1] = col_ind.len();
                    current_row += 1;
                }
                col_ind.push(col);
                values.push(val);
            }

            while current_row < self.n_rows {
                row_ptr[current_row + 1] = col_ind.len();
                current_row += 1;
            }

            (row_ptr, col_ind, values)
        }
    }
}

/// GPU-accelerated preconditioners.
pub mod preconditioners {
    use super::*;

    /// GPU-accelerated Jacobi preconditioner.
    pub struct JacobiPreconditioner {
        /// Diagonal elements (inverse).
        pub inv_diag: GPUMemory<f64>,
    }

    impl JacobiPreconditioner {
        /// Creates a Jacobi preconditioner from a matrix.
        pub fn from_matrix(matrix: &GPUCSRMatrix) -> Self {
            let n = matrix.n_rows;
            let mut inv_diag = vec![1.0; n];

            for i in 0..n {
                let row_start = matrix.row_ptr.host_data()[i];
                let row_end = matrix.row_ptr.host_data()[i + 1];

                // Find diagonal element
                for j in row_start..row_end {
                    if matrix.col_ind.host_data()[j] == i {
                        let diag = matrix.values.host_data()[j];
                        inv_diag[i] = if diag.abs() > 1e-15 { 1.0 / diag } else { 1.0 };
                        break;
                    }
                }
            }

            Self {
                inv_diag: GPUMemory::from_data(&inv_diag, matrix.row_ptr.device_id),
            }
        }

        /// Applies the preconditioner: z = M^{-1} * r
        pub fn apply(&self, r: &[f64], z: &mut [f64]) {
            let inv_diag = self.inv_diag.host_data();
            for i in 0..r.len() {
                z[i] = r[i] * inv_diag[i];
            }
        }
    }

    /// GPU-accelerated ILU(0) preconditioner.
    pub struct ILUPreconditioner {
        /// Lower triangular part.
        l_values: Vec<f64>,
        /// Upper triangular part.
        u_values: Vec<f64>,
        /// Row pointers.
        row_ptr: Vec<usize>,
        /// Column indices.
        col_ind: Vec<usize>,
        /// Inverse of L diagonal.
        inv_l_diag: Vec<f64>,
    }

    impl ILUPreconditioner {
        /// Creates ILU(0) preconditioner from matrix.
        pub fn from_matrix(matrix: &GPUCSRMatrix) -> Self {
            // CPU-based ILU(0) factorization
            // In real GPU implementation, this would use cuSPARSE or clSPARSE

            let n = matrix.n_rows;
            let mut ilu_values = matrix.values.host_data().to_vec();
            let row_ptr = matrix.row_ptr.host_data().to_vec();
            let col_ind = matrix.col_ind.host_data().to_vec();

            // ILU(0) factorization
            for i in 0..n {
                let row_start = row_ptr[i];
                let row_end = row_ptr[i + 1];

                // Find diagonal position
                let mut diag_pos = row_start;
                while diag_pos < row_end && col_ind[diag_pos] != i {
                    diag_pos += 1;
                }

                if diag_pos < row_end {
                    let mut sum = 0.0;
                    for k in row_start..diag_pos {
                        let col_k = col_ind[k];
                        // Find corresponding entries in L and U
                        let l_row_start = row_ptr[col_k];
                        let l_row_end = row_ptr[col_k + 1];
                        let mut l_ik = 0.0;
                        let mut u_kj = 0.0;

                        for p in l_row_start..l_row_end {
                            if col_ind[p] == i {
                                l_ik = ilu_values[p];
                                break;
                            }
                        }

                        let u_row_start = row_ptr[i];
                        let u_row_end = row_ptr[i + 1];
                        for p in u_row_start..u_row_end {
                            if col_ind[p] == col_k {
                                u_kj = ilu_values[p];
                                break;
                            }
                        }

                        sum += l_ik * u_kj;
                    }

                    ilu_values[diag_pos] -= sum;
                }
            }

            // Compute inverse of L diagonal
            let mut inv_l_diag = vec![1.0; n];
            for i in 0..n {
                let row_start = row_ptr[i];
                let row_end = row_ptr[i + 1];
                for j in row_start..row_end {
                    if col_ind[j] == i {
                        inv_l_diag[i] = 1.0 / ilu_values[j].max(1e-15);
                        break;
                    }
                }
            }

            Self {
                l_values: ilu_values.clone(),
                u_values: ilu_values,
                row_ptr,
                col_ind,
                inv_l_diag,
            }
        }

        /// Applies ILU preconditioner: solves LUz = r
        pub fn apply(&self, r: &[f64], z: &mut [f64]) {
            let n = r.len();

            // Forward substitution: Lz1 = r
            let mut z1 = vec![0.0; n];
            for i in 0..n {
                let row_start = self.row_ptr[i];
                let row_end = self.row_ptr[i + 1];
                let mut sum = r[i];

                for j in row_start..row_end {
                    let col = self.col_ind[j];
                    if col < i {
                        sum -= self.l_values[j] * z1[col];
                    }
                }
                z1[i] = sum * self.inv_l_diag[i];
            }

            // Backward substitution: Uz = z1
            for i in (0..n).rev() {
                let row_start = self.row_ptr[i];
                let row_end = self.row_ptr[i + 1];
                let mut sum = z1[i];

                for j in row_start..row_end {
                    let col = self.col_ind[j];
                    if col > i {
                        sum -= self.u_values[j] * z[col];
                    }
                }

                let diag_val = if let Some(pos) = (row_start..row_end).find(|&j| self.col_ind[j] == i)
                {
                    self.u_values[pos]
                } else {
                    1.0
                };

                z[i] = sum / diag_val.max(1e-15);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_memory_creation() {
        let mem = GPUMemory::<f64>::zeros(100, 0);
        assert_eq!(mem.size(), 100);
        assert_eq!(mem.host_data().len(), 100);
    }

    #[test]
    fn test_gpu_stream_creation() {
        let stream = GPUStream::new(0);
        assert_eq!(stream.device_id(), 0);
        assert!(stream.is_synchronized());

        let async_stream = GPUStream::async_stream(1, 5);
        assert_eq!(async_stream.device_id(), 1);
        assert!(!async_stream.is_synchronized());
    }

    #[test]
    fn test_vector_ops() {
        let ops = VectorOps::new(0);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        assert!((ops.dot(&a, &b) - 32.0).abs() < 1e-10);
        assert!((ops.norm(&a) - 14.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_matrix_vector_mul() {
        // Create a simple 3x3 sparse matrix
        // [2 -1  0]
        // [-1 2 -1]
        // [0 -1  2]
        let row_ptr = vec![0, 2, 5, 7];
        let col_ind = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0);
        let spmv = SparseMatrixVectorMul::new(0);

        let x = vec![1.0, 2.0, 3.0];
        let y = spmv.spmv_simple(&matrix, &x).unwrap();

        // Expected: [2*1 + (-1)*2, (-1)*1 + 2*2 + (-1)*3, (-1)*2 + 2*3]
        //         = [0, 0, 4]
        assert!((y[0] - 0.0).abs() < 1e-10);
        assert!((y[1] - 0.0).abs() < 1e-10);
        assert!((y[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_cg_solver() {
        // Create a simple SPD matrix (3x3)
        // [4 -1 -1]
        // [-1 4 -1]
        // [-1 -1 4]
        // CSR format: row_ptr has n+1 elements, col_ind and values have nnz elements
        let row_ptr = vec![0, 3, 6, 9];
        let col_ind = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let values = vec![4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0];

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0);
        let solver = GPUCGSolver::new(0, 1e-8, 100);

        let b = vec![2.0, 2.0, 2.0];
        let mut x = vec![0.0; 3];

        let result = solver.solve(&matrix, &b, &mut x).unwrap();

        // Solution should be [1, 1, 1] since [4-1-1, -1+4-1, -1-1+4] = [2, 2, 2]
        assert!((x[0] - 1.0).abs() < 0.1);
        assert!((x[1] - 1.0).abs() < 0.1);
        assert!((x[2] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_jacobi_preconditioner() {
        let row_ptr = vec![0, 1, 3, 4];
        let col_ind = vec![0, 0, 1, 2];
        let values = vec![2.0, 1.0, 4.0, 3.0];

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0);
        let prec = preconditioners::JacobiPreconditioner::from_matrix(&matrix);

        let r = vec![2.0, 8.0, 9.0];
        let mut z = vec![0.0; 3];
        prec.apply(&r, &mut z);

        // Expected: z = [2/2, 8/4, 9/3] = [1, 2, 3]
        assert!((z[0] - 1.0).abs() < 1e-10);
        assert!((z[1] - 2.0).abs() < 1e-10);
        assert!((z[2] - 3.0).abs() < 1e-10);
    }
}
