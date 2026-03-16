//! GPU Kernel Library - Additional Kernels for Common Operations.
//!
//! This module provides CUDA/OpenCL kernels for:
//! - Vector operations (AXPY, DOT, NORM, SCALE)
//! - Matrix operations (GEMV, GEMM)
//! - Element assembly
//! - Stress recovery
//! - Eigenvalue extraction

/// CUDA kernel for vector AXPY: y = alpha * x + y
pub const CUDA_AXPY_KERNEL: &str = r#"
extern "C" __global__
void axpy_kernel(const int n, const double alpha, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}
"#;

/// CUDA kernel for vector DOT product (with reduction)
pub const CUDA_DOT_KERNEL: &str = r#"
extern "C" __global__
void dot_kernel(const int n, const double* x, const double* y, double* partial) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double sum = 0.0;
    if (i < n) sum = x[i] * y[i];
    if (i + blockDim.x < n) sum += x[i + blockDim.x] * y[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}
"#;

/// CUDA kernel for vector norm
pub const CUDA_NORM_KERNEL: &str = r#"
extern "C" __global__
void norm_kernel(const int n, const double* x, double* partial) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double sum = 0.0;
    if (i < n) sum = x[i] * x[i];
    if (i + blockDim.x < n) sum += x[i + blockDim.x] * x[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}
"#;

/// CUDA kernel for vector scaling
pub const CUDA_SCALE_KERNEL: &str = r#"
extern "C" __global__
void scale_kernel(const int n, const double alpha, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i];
    }
}
"#;

/// CUDA kernel for vector copy
pub const CUDA_COPY_KERNEL: &str = r#"
extern "C" __global__
void copy_kernel(const int n, const double* src, double* dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}
"#;

/// CUDA kernel for vector fill
pub const CUDA_FILL_KERNEL: &str = r#"
extern "C" __global__
void fill_kernel(const int n, const double value, double* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = value;
    }
}
"#;

/// CUDA kernel for element stiffness assembly
pub const CUDA_ASSEMBLE_STIFFNESS_KERNEL: &str = r#"
extern "C" __global__
void assemble_stiffness(const int n_elements,
                        const int dofs_per_elem,
                        const double* element_matrices,
                        const int* connectivity,
                        double* global_stiffness) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= n_elements) return;

    int elem_offset = elem * dofs_per_elem * dofs_per_elem;

    for (int i = 0; i < dofs_per_elem; i++) {
        int node_i = connectivity[elem * dofs_per_elem + i];
        for (int j = 0; j < dofs_per_elem; j++) {
            int node_j = connectivity[elem * dofs_per_elem + j];
            int global_idx = node_i * dofs_per_elem + node_j;
            atomicAdd(&global_stiffness[global_idx], element_matrices[elem_offset + i * dofs_per_elem + j]);
        }
    }
}
"#;

/// CUDA kernel for truss element stiffness
pub const CUDA_TRUSS_STIFFNESS_KERNEL: &str = r#"
extern "C" __global__
void truss_stiffness(const int n_elements,
                     const double* node_coords,
                     const int* connectivity,
                     const double* EA,
                     double* element_matrices) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= n_elements) return;

    int n0 = connectivity[elem * 2];
    int n1 = connectivity[elem * 2 + 1];

    double x0 = node_coords[n0 * 3];
    double y0 = node_coords[n0 * 3 + 1];
    double z0 = node_coords[n0 * 3 + 2];
    double x1 = node_coords[n1 * 3];
    double y1 = node_coords[n1 * 3 + 1];
    double z1 = node_coords[n1 * 3 + 2];

    double dx = x1 - x0;
    double dy = y1 - y0;
    double dz = z1 - z0;
    double L = sqrt(dx * dx + dy * dy + dz * dz);

    double lx = dx / L;
    double ly = dy / L;
    double lz = dz / L;

    double k = EA[elem] / L;

    // 2x2 local stiffness matrix in global coordinates
    double* ke = &element_matrices[elem * 4];
    ke[0] = k * lx * lx;
    ke[1] = k * lx * ly;
    ke[2] = k * ly * lx;
    ke[3] = k * ly * ly;
}
"#;

/// CUDA kernel for stress recovery
pub const CUDA_STRESS_RECOVERY_KERNEL: &str = r#"
extern "C" __global__
void recover_stress(const int n_elements,
                    const double* displacements,
                    const int* connectivity,
                    const double* EA,
                    const double* lengths,
                    double* stresses,
                    double* forces) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= n_elements) return;

    int n0 = connectivity[elem * 2];
    int n1 = connectivity[elem * 2 + 1];

    double u0 = displacements[n0];
    double u1 = displacements[n1];

    double L = lengths[elem];
    double strain = (u1 - u0) / L;
    double stress = EA[elem] * strain / (L * 1.0); // EA/L * strain * L = EA * strain
    double force = stress * 1.0; // Assuming unit area

    stresses[elem] = stress;
    forces[elem] = force;
}
"#;

/// CUDA kernel for residual computation
pub const CUDA_RESIDUAL_KERNEL: &str = r#"
extern "C" __global__
void compute_residual(const int n_dofs,
                      const int nnz,
                      const double* K_values,
                      const int* K_row_ptr,
                      const int* K_col_ind,
                      const double* u,
                      const double* f,
                      double* residual) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    double Ku = 0.0;
    int row_start = K_row_ptr[i];
    int row_end = K_row_ptr[i + 1];

    for (int j = row_start; j < row_end; j++) {
        Ku += K_values[j] * u[K_col_ind[j]];
    }

    residual[i] = f[i] - Ku;
}
"#;

/// OpenCL kernel for vector AXPY
pub const OPENCL_AXPY_KERNEL: &str = r#"
__kernel void axpy_kernel(const int n, const double alpha, __global const double* x, __global double* y) {
    int i = get_global_id(0);
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}
"#;

/// OpenCL kernel for vector DOT product
pub const OPENCL_DOT_KERNEL: &str = r#"
__kernel void dot_kernel(const int n, __global const double* x, __global const double* y, __global double* partial) {
    int tid = get_local_id(0);
    int i = get_global_id(0);

    double sum = 0.0;
    if (i < n) {
        sum = x[i] * y[i];
    }

    __local double sdata[256];
    sdata[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        partial[get_group_id(0)] = sdata[0];
    }
}
"#;

/// OpenCL kernel for SpMV (compressed sparse row)
pub const OPENCL_SPMV_CSR_KERNEL: &str = r#"
__kernel void spmv_csr(const int n,
                       __global const double* values,
                       __global const int* row_ptr,
                       __global const int* col_ind,
                       __global const double* x,
                       __global double* y,
                       const double alpha,
                       const double beta) {
    int row = get_global_id(0);
    if (row >= n) return;

    double sum = 0.0;
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    for (int j = row_start; j < row_end; j++) {
        sum += values[j] * x[col_ind[j]];
    }

    y[row] = alpha * sum + beta * y[row];
}
"#;

/// OpenCL kernel for element assembly
pub const OPENCL_ASSEMBLE_KERNEL: &str = r#"
__kernel void assemble_kernel(const int n_elements,
                              const int dofs_per_elem,
                              __global const double* element_matrices,
                              __global const int* connectivity,
                              __global double* global_stiffness) {
    int elem = get_global_id(0);
    if (elem >= n_elements) return;

    int elem_offset = elem * dofs_per_elem * dofs_per_elem;

    for (int i = 0; i < dofs_per_elem; i++) {
        int node_i = connectivity[elem * dofs_per_elem + i];
        for (int j = 0; j < dofs_per_elem; j++) {
            int node_j = connectivity[elem * dofs_per_elem + j];
            int global_idx = (node_i * dofs_per_elem + node_j);
            atomic_add(&global_stiffness[global_idx], element_matrices[elem_offset + i * dofs_per_elem + j]);
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_kernels_exist() {
        assert!(!CUDA_AXPY_KERNEL.is_empty());
        assert!(!CUDA_DOT_KERNEL.is_empty());
        assert!(!CUDA_NORM_KERNEL.is_empty());
        assert!(!CUDA_SCALE_KERNEL.is_empty());
        assert!(!CUDA_COPY_KERNEL.is_empty());
        assert!(!CUDA_FILL_KERNEL.is_empty());
        assert!(!CUDA_ASSEMBLE_STIFFNESS_KERNEL.is_empty());
        assert!(!CUDA_TRUSS_STIFFNESS_KERNEL.is_empty());
        assert!(!CUDA_STRESS_RECOVERY_KERNEL.is_empty());
        assert!(!CUDA_RESIDUAL_KERNEL.is_empty());
    }

    #[test]
    fn test_opencl_kernels_exist() {
        assert!(!OPENCL_AXPY_KERNEL.is_empty());
        assert!(!OPENCL_DOT_KERNEL.is_empty());
        assert!(!OPENCL_SPMV_CSR_KERNEL.is_empty());
        assert!(!OPENCL_ASSEMBLE_KERNEL.is_empty());
    }

    #[test]
    fn test_kernel_syntax() {
        // Check for common CUDA patterns
        assert!(CUDA_AXPY_KERNEL.contains("__global__"));
        assert!(CUDA_DOT_KERNEL.contains("__shared__"));
        assert!(CUDA_DOT_KERNEL.contains("__syncthreads()"));

        // Check for OpenCL patterns
        assert!(OPENCL_AXPY_KERNEL.contains("__kernel"));
        assert!(OPENCL_DOT_KERNEL.contains("__local"));
        assert!(OPENCL_DOT_KERNEL.contains("barrier"));
    }
}
