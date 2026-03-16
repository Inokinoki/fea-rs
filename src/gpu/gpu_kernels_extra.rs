//! GPU Kernel Library - Additional Optimized Kernels.
//!
//! This module provides additional CUDA/OpenCL kernels:
//! - Sparse matrix-matrix multiplication (SpGEMM)
//! - Sparse triangular solve (SpTRSV)
//! - Vector normalization
//! - Parallel reduction
//! - Atomic operations for assembly

/// CUDA kernel for sparse matrix-matrix multiplication (SpGEMM).
pub const CUDA_SPGEMM_KERNEL: &str = r#"
extern "C" __global__
void spgemm_csr(const int n,
                const int nnz_a,
                const int nnz_c,
                __global const double* a_values,
                __global const int* a_row_ptr,
                __global const int* a_col_ind,
                __global const double* b_values,
                __global const int* b_row_ptr,
                __global const int* b_col_ind,
                __global double* c_values,
                __global int* c_row_ptr,
                __global int* c_col_ind) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int c_row_start = c_row_ptr[row];
    int c_row_end = c_row_ptr[row + 1];

    for (int j = c_row_start; j < c_row_end; j++) {
        double sum = 0.0;
        int col = c_col_ind[j];

        // C[i,j] = sum_k(A[i,k] * B[k,j])
        int a_row_start = a_row_ptr[row];
        int a_row_end = a_row_ptr[row + 1];

        for (int k = a_row_start; k < a_row_end; k++) {
            int a_col = a_col_ind[k];
            double a_val = a_values[k];

            int b_row_start = b_row_ptr[a_col];
            int b_row_end = b_row_ptr[a_col + 1];

            for (int m = b_row_start; m < b_row_end; m++) {
                if (b_col_ind[m] == col) {
                    sum += a_val * b_values[m];
                    break;
                }
            }
        }

        c_values[j] = sum;
    }
}
"#;

/// CUDA kernel for sparse triangular solve (SpTRSV).
pub const CUDA_SPTRSV_KERNEL: &str = r#"
extern "C" __global__
void sptrsv_lower(const int n,
                  __global const double* l_values,
                  __global const int* l_row_ptr,
                  __global const int* l_col_ind,
                  __global const double* b,
                  __global double* x) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    double sum = b[row];
    int row_start = l_row_ptr[row];
    int row_end = l_row_ptr[row + 1];

    for (int j = row_start; j < row_end; j++) {
        int col = l_col_ind[j];
        if (col < row) {
            sum -= l_values[j] * x[col];
        } else if (col == row) {
            sum /= l_values[j];
        }
    }

    x[row] = sum;
}

extern "C" __global__
void sptrsv_upper(const int n,
                  __global const double* u_values,
                  __global const int* u_row_ptr,
                  __global const int* u_col_ind,
                  __global const double* b,
                  __global double* x) {
    int row = n - 1 - (blockIdx.x * blockDim.x + threadIdx.x);
    if (row < 0) return;

    double sum = b[row];
    int row_start = u_row_ptr[row];
    int row_end = u_row_ptr[row + 1];

    for (int j = row_start; j < row_end; j++) {
        int col = u_col_ind[j];
        if (col > row) {
            sum -= u_values[j] * x[col];
        } else if (col == row) {
            sum /= u_values[j];
        }
    }

    x[row] = sum;
}
"#;

/// CUDA kernel for vector normalization.
pub const CUDA_NORMALIZE_KERNEL: &str = r#"
extern "C" __global__
void normalize_vector(const int n,
                      __global double* v,
                      __global double* norm) {
    // Compute norm using parallel reduction
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;
    if (i < n) {
        sum = v[i] * v[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *norm = sqrt(sdata[0]);
    }

    __syncthreads();

    // Normalize
    if (i < n) {
        v[i] /= (*norm);
    }
}
"#;

/// CUDA kernel for parallel reduction (sum).
pub const CUDA_REDUCE_SUM_KERNEL: &str = r#"
extern "C" __global__
void reduce_sum(const int n,
                __global const double* input,
                __global double* output) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double sum = 0.0;
    if (i < n) {
        sum = input[i];
    }
    if (i + blockDim.x < n) {
        sum += input[i + blockDim.x];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
"#;

/// CUDA kernel for parallel reduction (max).
pub const CUDA_REDUCE_MAX_KERNEL: &str = r#"
extern "C" __global__
void reduce_max(const int n,
                __global const double* input,
                __global double* output) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double max_val = -1e30;
    if (i < n) {
        max_val = input[i];
    }
    if (i + blockDim.x < n) {
        max_val = fmax(max_val, input[i + blockDim.x]);
    }

    sdata[tid] = max_val;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
"#;

/// CUDA kernel for atomic assembly (used in FEA assembly).
pub const CUDA_ATOMIC_ASSEMBLY_KERNEL: &str = r#"
extern "C" __global__
void atomic_assemble(const int n_elements,
                     const int dofs_per_elem,
                     __global const double* element_matrices,
                     __global const int* connectivity,
                     __global double* global_matrix) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= n_elements) return;

    int elem_offset = elem * dofs_per_elem * dofs_per_elem;

    for (int i = 0; i < dofs_per_elem; i++) {
        int global_i = connectivity[elem * dofs_per_elem + i];

        for (int j = 0; j < dofs_per_elem; j++) {
            int global_j = connectivity[elem * dofs_per_elem + j];
            double val = element_matrices[elem_offset + i * dofs_per_elem + j];

            int global_idx = global_i * dofs_per_elem + global_j;
            atomicAdd(&global_matrix[global_idx], val);
        }
    }
}
"#;

/// OpenCL kernel for sparse matrix-vector multiplication.
pub const OPENCL_SPMV_KERNEL: &str = r#"
__kernel void spmv_csr(const int n,
                       const int nnz,
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

/// OpenCL kernel for SpMV with shared memory optimization.
pub const OPENCL_SPMV_SHARED_KERNEL: &str = r#"
__kernel void spmv_csr_shared(const int n,
                              const int nnz,
                              __global const double* values,
                              __global const int* row_ptr,
                              __global const int* col_ind,
                              __global const double* x,
                              __global double* y,
                              const double alpha,
                              const double beta,
                              __local double* shared_x,
                              __local int* shared_col) {
    int row = get_global_id(0);
    int tid = get_local_id(0);
    int local_size = get_local_size(0);

    if (row >= n) return;

    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    int nnz_row = row_end - row_start;

    // Load x values to shared memory
    for (int i = tid; i < nnz_row; i += local_size) {
        int col = col_ind[row_start + i];
        shared_x[i] = x[col];
        shared_col[i] = col;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute dot product using shared memory
    double sum = 0.0;
    for (int j = 0; j < nnz_row; j++) {
        sum += values[row_start + j] * shared_x[j];
    }

    y[row] = alpha * sum + beta * y[row];
}
"#;

/// OpenCL kernel for parallel reduction.
pub const OPENCL_REDUCE_KERNEL: &str = r#"
__kernel void reduce_sum(__global const double* input,
                         __global double* output,
                         const int n) {
    int tid = get_local_id(0);
    int gid = get_global_id(0);

    extern __local double sdata[];

    double sum = 0.0;
    if (gid < n) {
        sum = input[gid];
    }

    sdata[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction in local memory
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_kernels_exist() {
        assert!(!CUDA_SPGEMM_KERNEL.is_empty());
        assert!(!CUDA_SPTRSV_KERNEL.is_empty());
        assert!(!CUDA_NORMALIZE_KERNEL.is_empty());
        assert!(!CUDA_REDUCE_SUM_KERNEL.is_empty());
        assert!(!CUDA_REDUCE_MAX_KERNEL.is_empty());
        assert!(!CUDA_ATOMIC_ASSEMBLY_KERNEL.is_empty());
    }

    #[test]
    fn test_opencl_kernels_exist() {
        assert!(!OPENCL_SPMV_KERNEL.is_empty());
        assert!(!OPENCL_SPMV_SHARED_KERNEL.is_empty());
        assert!(!OPENCL_REDUCE_KERNEL.is_empty());
    }

    #[test]
    fn test_kernel_syntax() {
        // Check for common CUDA/OpenCL patterns
        assert!(CUDA_NORMALIZE_KERNEL.contains("__shared__"));
        assert!(CUDA_REDUCE_SUM_KERNEL.contains("__syncthreads()"));
        assert!(OPENCL_SPMV_SHARED_KERNEL.contains("__local"));
        assert!(OPENCL_REDUCE_KERNEL.contains("barrier"));
    }
}
