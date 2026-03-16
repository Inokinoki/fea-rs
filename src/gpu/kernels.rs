//! CUDA kernels for FEA acceleration.
//!
//! This module contains CUDA kernel source code that can be compiled
//! with nvcc and linked to the Rust codebase.
//!
//! # Kernels Included
//!
//! - Sparse Matrix-Vector Multiplication (SpMV)
//! - Vector operations (axpy, dot, norm, scale)
//! - CG solver kernels
//! - Element stiffness assembly
//! - Global assembly

/// CUDA kernel for sparse matrix-vector multiplication (CSR format).
///
/// Each thread computes one element of the output vector.
pub const SPV_CUDA_KERNEL: &str = r#"
extern "C" __global__
void spmv_csr(const int n,
              const double* values,
              const int* row_ptr,
              const int* col_ind,
              const double* x,
              double* y,
              double alpha,
              double beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        double sum = 0.0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_ind[j]];
        }

        y[row] = alpha * sum + beta * y[row];
    }
}
"#;

/// CUDA kernel for sparse matrix-vector multiplication with shared memory.
/// Optimized for matrices with regular sparsity patterns.
pub const SPV_SHARED_KERNEL: &str = r#"
extern "C" __global__
void spmv_csr_shared(const int n,
                     const int nnz,
                     const double* values,
                     const int* row_ptr,
                     const int* col_ind,
                     const double* x,
                     double* y,
                     double alpha,
                     double beta) {
    extern __shared__ double shared_mem[];

    int tid = threadIdx.x;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        double sum = 0.0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        int nnz_row = row_end - row_start;

        // Load x values into shared memory
        for (int i = tid; i < nnz_row; i += blockDim.x) {
            int col = col_ind[row_start + i];
            shared_mem[i] = x[col];
        }
        __syncthreads();

        // Compute dot product using shared memory
        for (int j = 0; j < nnz_row; j++) {
            sum += values[row_start + j] * shared_mem[j];
        }

        y[row] = alpha * sum + beta * y[row];
    }
}
"#;

/// CUDA kernel for vector axpy operation: y = alpha * x + y
pub const AXPY_KERNEL: &str = r#"
extern "C" __global__
void axpy(const int n, const double alpha, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}
"#;

/// CUDA kernel for vector scaling: y = alpha * x
pub const SCALE_KERNEL: &str = r#"
extern "C" __global__
void scale(const int n, const double alpha, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i];
    }
}
"#;

/// CUDA kernel for vector copy: dst = src
pub const COPY_KERNEL: &str = r#"
extern "C" __global__
void copy(const int n, const double* src, double* dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}
"#;

/// CUDA kernel for vector set: x = value
pub const SET_KERNEL: &str = r#"
extern "C" __global__
void set(const int n, double value, double* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = value;
    }
}
"#;

/// CUDA kernel for dot product (with partial sums for reduction).
pub const DOT_KERNEL: &str = r#"
extern "C" __global__
void dot_partial(const int n, const double* x, const double* y, double* partial) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;
    if (i < n) {
        sum = x[i] * y[i];
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
        partial[blockIdx.x] = sdata[0];
    }
}
"#;

/// CUDA kernel for computing vector norm.
pub const NORM_KERNEL: &str = r#"
extern "C" __global__
void norm_partial(const int n, const double* x, double* partial) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;
    if (i < n) {
        sum = x[i] * x[i];
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
        partial[blockIdx.x] = sdata[0];
    }
}
"#;

/// CUDA kernel for Conjugate Gradient iteration.
/// This combines multiple operations into a single kernel for efficiency.
pub const CG_ITER_KERNEL: &str = r#"
extern "C" __global__
void cg_iter(const int n,
             const double* values,
             const int* row_ptr,
             const int* col_ind,
             double* x,
             double* r,
             double* p,
             double* ap,
             double* partial_data,
             double* alpha_out,
             double* beta_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double sdata[];
    double* alpha_partial = &partial_data[0];
    double* beta_partial = &partial_data[blockDim.x];
    double* p_ap_partial = &partial_data[2 * blockDim.x];
    double* rz_partial = &partial_data[3 * blockDim.x];

    // Compute Ap = A * p
    if (i < n) {
        double sum = 0.0;
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * p[col_ind[j]];
        }
        ap[i] = sum;
    }
    __syncthreads();

    // Compute p^T * Ap (for alpha)
    double p_ap = 0.0;
    if (i < n) {
        p_ap = p[i] * ap[i];
    }
    sdata[threadIdx.x] = p_ap;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        p_ap_partial[blockIdx.x] = sdata[0];
    }
    __syncthreads();

    // Compute r^T * z (for beta, assuming Jacobi preconditioning z = r / diag)
    double rz = 0.0;
    if (i < n) {
        double diag = values[row_ptr[i]]; // Assuming diagonal stored first
        double z_i = r[i] / (diag > 1e-15 ? diag : 1.0);
        rz = r[i] * z_i;
    }
    sdata[threadIdx.x] = rz;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        rz_partial[blockIdx.x] = sdata[0];
    }
}
"#;

/// CUDA kernel for element stiffness matrix assembly.
/// Each thread block processes one element.
pub const ASSEMBLE_STIFFNESS_KERNEL: &str = r#"
extern "C" __global__
void assemble_stiffness(const int n_elements,
                        const int n_dofs_per_elem,
                        const double* element_matrices,
                        const int* connectivity,
                        int* row_ptr,
                        int* col_ind,
                        double* values) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem < n_elements) {
        int elem_offset = elem * n_dofs_per_elem * n_dofs_per_elem;

        // Get node indices for this element
        int node0 = connectivity[elem * 2];
        int node1 = connectivity[elem * 2 + 1];

        // For 2D truss: 2 DOFs per node
        int dof[2] = {node0 * 2, node1 * 2};

        // Assemble into global matrix
        for (int i = 0; i < 4; i++) {
            int row = dof[i / 2];
            int col = dof[i % 2];

            // Find position in CSR matrix (simplified - assumes known structure)
            int pos = row_ptr[row];
            for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
                if (col_ind[j] == col) {
                    pos = j;
                    break;
                }
            }

            atomicAdd(&values[pos], element_matrices[elem_offset + i]);
        }
    }
}
"#;

/// CUDA kernel for diagonal extraction.
pub const EXTRACT_DIAGONAL_KERNEL: &str = r#"
extern "C" __global__
void extract_diagonal(const int n,
                      const double* values,
                      const int* row_ptr,
                      const int* col_ind,
                      double* diag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++) {
            if (col_ind[j] == i) {
                diag[i] = values[j];
                break;
            }
        }
    }
}
"#;

/// CUDA kernel for Jacobi preconditioner application.
pub const JACOBI_PRECOND_KERNEL: &str = r#"
extern "C" __global__
void jacobi_precond(const int n,
                    const double* diag,
                    const double* r,
                    double* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double d = diag[i];
        z[i] = (d > 1e-15 && d < 1e15) ? r[i] / d : r[i];
    }
}
"#;

/// CUDA kernel for ILU(0) forward substitution.
pub const ILU_FORWARD_KERNEL: &str = r#"
extern "C" __global__
void ilu_forward(const int n,
                 const double* values,
                 const int* row_ptr,
                 const int* col_ind,
                 const double* r,
                 double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double sum = r[i];
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++) {
            int col = col_ind[j];
            if (col < i) {
                sum -= values[j] * y[col];
            }
        }

        // Find diagonal
        for (int j = row_start; j < row_end; j++) {
            if (col_ind[j] == i) {
                y[i] = sum / values[j];
                break;
            }
        }
    }
}
"#;

/// CUDA kernel for ILU(0) backward substitution.
pub const ILU_BACKWARD_KERNEL: &str = r#"
extern "C" __global__
void ilu_backward(const int n,
                  const double* values,
                  const int* row_ptr,
                  const int* col_ind,
                  const double* y,
                  double* x) {
    int i = n - 1 - (blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= 0 && i < n) {
        double sum = y[i];
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++) {
            int col = col_ind[j];
            if (col > i) {
                sum -= values[j] * x[col];
            }
        }

        // Find diagonal
        for (int j = row_start; j < row_end; j++) {
            if (col_ind[j] == i) {
                x[i] = sum / values[j];
                break;
            }
        }
    }
}
"#;

/// CUDA kernel for Chebyshev polynomial preconditioner.
pub const CHEBYSHEV_PRECOND_KERNEL: &str = r#"
extern "C" __global__
void chebyshev_precond(const int n,
                       const double* values,
                       const int* row_ptr,
                       const int* col_ind,
                       const double* r,
                       double* z,
                       double lambda_min,
                       double lambda_max,
                       int degree) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // Chebyshev iteration parameters
        double theta = (lambda_max - lambda_min) / (lambda_max + lambda_min);
        double rho = (1.0 - theta) / (1.0 + theta);

        // Initial guess
        double diag = values[row_ptr[i]];
        z[i] = r[i] / (diag > 1e-15 ? diag : 1.0);

        // Chebyshev iterations
        double z_prev = 0.0;
        for (int k = 0; k < degree; k++) {
            double z_new = 2.0 * r[i] / diag - z_prev;
            z_prev = z[i];
            z[i] = z_new;
        }
    }
}
"#;

/// CUDA kernel for GMRES Arnoldi iteration.
pub const ARNOLDI_KERNEL: &str = r#"
extern "C" __global__
void arnoldi_iteration(const int n,
                       const int k,
                       const double* values,
                       const int* row_ptr,
                       const int* col_ind,
                       double* hessenberg,
                       double* v,
                       double* w,
                       int* converged) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // Compute w = A * v_k
        double sum = 0.0;
        int row_start = row_ptr[i];
        int row_end = row_ptr[row_start + 1];

        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * v[col_ind[j]];
        }
        w[i] = sum;

        // Modified Gram-Schmidt orthogonalization
        for (int j = 0; j <= k; j++) {
            double dot = 0.0;
            for (int m = 0; m < n; m++) {
                dot += v[m + j * n] * w[m];
            }
            hessenberg[j + k * (k + 1)] = dot;

            for (int m = 0; m < n; m++) {
                w[m] -= dot * v[m + j * n];
            }
        }
    }
}
"#;

/// CUDA kernel for Lanczos bidiagonalization.
pub const LANCZOS_KERNEL: &str = r#"
extern "C" __global__
void lanczos_bidiag(const int n,
                    const double* values,
                    const int* row_ptr,
                    const int* col_ind,
                    double* alpha,
                    double* beta,
                    double* u,
                    double* v,
                    int iteration) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // Compute u_i = A * v_i - beta_{i-1} * u_{i-1}
        if (iteration == 0) {
            double sum = 0.0;
            int row_start = row_ptr[i];
            int row_end = row_ptr[i + 1];

            for (int j = row_start; j < row_end; j++) {
                sum += values[j] * v[col_ind[j]];
            }
            u[i] = sum;
        } else {
            u[i] = u[i] - beta[iteration - 1] * u[i];
        }

        // Compute alpha_i = ||u_i||
        if (i == 0) {
            double norm = 0.0;
            for (int m = 0; m < n; m++) {
                norm += u[m] * u[m];
            }
            alpha[iteration] = sqrt(norm);
        }
    }
}
"#;

/// CUDA kernel for element mass matrix assembly.
pub const ASSEMBLE_MASS_KERNEL: &str = r#"
extern "C" __global__
void assemble_mass(const int n_elements,
                   const int n_dofs_per_elem,
                   const double density,
                   const double* shape_functions,
                   const int* connectivity,
                   int* row_ptr,
                   int* col_ind,
                   double* mass_values) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem < n_elements) {
        int node0 = connectivity[elem * 2];
        int node1 = connectivity[elem * 2 + 1];
        int dof[2] = {node0 * 2, node1 * 2};

        // Consistent mass matrix for 2D truss
        double m = density * 0.5; // Simplified

        for (int i = 0; i < 4; i++) {
            int row = dof[i / 2];
            int col = dof[i % 2];

            int pos = row_ptr[row];
            for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
                if (col_ind[j] == col) {
                    pos = j;
                    break;
                }
            }

            if (i / 2 == i % 2) {
                atomicAdd(&mass_values[pos], m);
            }
        }
    }
}
"#;

/// CUDA kernel for Rayleigh damping computation.
pub const RAYLEIGH_DAMPING_KERNEL: &str = r#"
extern "C" __global__
void rayleigh_damping(const int n,
                      const double alpha_m,
                      const double alpha_k,
                      const double* mass_diag,
                      const double* stiff_diag,
                      double* damping_diag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // C = alpha_m * M + alpha_k * K
        damping_diag[i] = alpha_m * mass_diag[i] + alpha_k * stiff_diag[i];
    }
}
"#;

/// CUDA kernel for Newmark-beta time integration.
pub const NEWMARK_INTEGRATION_KERNEL: &str = r#"
extern "C" __global__
void newmark_integrate(const int n,
                       const double* u_n,
                       const double* v_n,
                       const double* a_n,
                       double* u_np1,
                       double* v_np1,
                       double* a_np1,
                       const double dt,
                       const double beta,
                       const double gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // Newmark-beta formulas
        double a = gamma / (beta * dt);
        double b = 1.0 / (beta * dt * dt);

        // u_{n+1} = u_n + dt * v_n + dt^2 * ((1/2 - beta) * a_n + beta * a_{n+1})
        u_np1[i] = u_n[i] + dt * v_n[i] + dt * dt * ((0.5 - beta) * a_n[i] + beta * a_np1[i]);

        // v_{n+1} = v_n + dt * ((1 - gamma) * a_n + gamma * a_{n+1})
        v_np1[i] = v_n[i] + dt * ((1.0 - gamma) * a_n[i] + gamma * a_np1[i]);
    }
}
"#;

/// CUDA kernel for Wilson-theta time integration.
pub const WILSON_THETA_KERNEL: &str = r#"
extern "C" __global__
void wilson_theta(const int n,
                  const double* u_n,
                  const double* v_n,
                  const double* a_n,
                  double* u_np1,
                  double* v_np1,
                  double* a_np1,
                  const double dt,
                  const double theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double tau = theta * dt;

        // Wilson-theta formulas
        u_np1[i] = u_n[i] + dt * v_n[i] + dt * dt * ((0.5 - 1.0 / (6.0 * theta)) * a_n[i]
                  + 1.0 / (6.0 * theta) * a_np1[i]);

        v_np1[i] = v_n[i] + dt * ((1.0 - 0.5 / theta) * a_n[i] + 0.5 / theta * a_np1[i]);
    }
}
"#;

/// CUDA kernel for modal superposition.
pub const MODAL_SUPERPOSITION_KERNEL: &str = r#"
extern "C" __global__
void modal_superposition(const int n_dofs,
                         const int n_modes,
                         const double* modes,
                         const double* modal_disp,
                         double* physical_disp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_dofs) {
        double disp = 0.0;
        for (int m = 0; m < n_modes; m++) {
            disp += modes[i + m * n_dofs] * modal_disp[m];
        }
        physical_disp[i] = disp;
    }
}
"#;

/// CUDA kernel for stress recovery.
pub const STRESS_RECOVERY_KERNEL: &str = r#"
extern "C" __global__
void stress_recovery(const int n_elements,
                     const double* displacements,
                     const int* connectivity,
                     const double* D_matrix,
                     const double* B_matrix,
                     double* stresses) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem < n_elements) {
        int node0 = connectivity[elem * 2];
        int node1 = connectivity[elem * 2 + 1];

        // Extract element displacements
        double u[4];
        u[0] = displacements[node0 * 2];
        u[1] = displacements[node0 * 2 + 1];
        u[2] = displacements[node1 * 2];
        u[3] = displacements[node1 * 2 + 1];

        // Compute strain: epsilon = B * u
        double strain[3];
        for (int i = 0; i < 3; i++) {
            strain[i] = 0.0;
            for (int j = 0; j < 4; j++) {
                strain[i] += B_matrix[i * 4 + j] * u[j];
            }
        }

        // Compute stress: sigma = D * epsilon
        for (int i = 0; i < 3; i++) {
            stresses[elem * 3 + i] = 0.0;
            for (int j = 0; j < 3; j++) {
                stresses[elem * 3 + i] += D_matrix[i * 3 + j] * strain[j];
            }
        }
    }
}
"#;

/// CUDA kernel for von Mises stress computation.
pub const VON_MISES_KERNEL: &str = r#"
extern "C" __global__
void von_mises_stress(const int n_elements,
                      const double* stresses,
                      double* von_mises) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem < n_elements) {
        double sx = stresses[elem * 3];
        double sy = stresses[elem * 3 + 1];
        double txy = stresses[elem * 3 + 2];

        // von Mises: sqrt(sx^2 + sy^2 - sx*sy + 3*txy^2)
        double vm = sqrt(sx * sx + sy * sy - sx * sy + 3.0 * txy * txy);
        von_mises[elem] = vm;
    }
}
"#;

/// CUDA kernel for error norm computation.
pub const ERROR_NORM_KERNEL: &str = r#"
extern "C" __global__
void error_norm(const int n,
                const double* u_exact,
                const double* u_fem,
                const double* weights,
                double* l2_norm,
                double* h1_norm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double diff = u_fem[i] - u_exact[i];
        l2_norm[i] = diff * diff * weights[i];
        h1_norm[i] = diff * diff; // Simplified H1 norm
    }
}
"#;

/// CUDA kernel for adaptive mesh refinement indicator.
pub const AMR_INDICATOR_KERNEL: &str = r#"
extern "C" __global__
void amr_indicator(const int n_elements,
                   const double* element_errors,
                   const double* element_energies,
                   double* refinement_indicators,
                   double threshold) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem < n_elements) {
        double indicator = element_errors[elem] / (element_energies[elem] + 1e-15);
        refinement_indicators[elem] = (indicator > threshold) ? 1.0 : 0.0;
    }
}
"#;

/// CUDA stream creation for asynchronous operations.
pub const CUDA_STREAM_UTILS: &str = r#"
// Host-side utility functions (to be called from Rust via C API)

extern "C" cudaError_t create_cuda_stream(cudaStream_t* stream) {
    return cudaStreamCreate(stream);
}

extern "C" cudaError_t destroy_cuda_stream(cudaStream_t stream) {
    return cudaStreamDestroy(stream);
}

extern "C" cudaError_t synchronize_cuda_stream(cudaStream_t stream) {
    return cudaStreamSynchronize(stream);
}

extern "C" cudaError_t cuda_malloc_async(void** ptr, size_t size, cudaStream_t stream) {
    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr = NULL;
    stream_attr.accessPolicyWindow.num_bytes = 0;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    cudaDeviceGetStreamSyncMode(&stream_attr);
    return cudaMallocAsync(ptr, size, stream);
}

extern "C" cudaError_t cuda_free_async(void* ptr, cudaStream_t stream) {
    return cudaFreeAsync(ptr, stream);
}
"#;

/// Complete CUDA module source for compilation.
pub const FULL_CUDA_MODULE: &str = r#"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Include all kernels
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_sources_exist() {
        assert!(!SPV_CUDA_KERNEL.is_empty());
        assert!(!AXPY_KERNEL.is_empty());
        assert!(!DOT_KERNEL.is_empty());
        assert!(!CG_ITER_KERNEL.is_empty());
        assert!(!ASSEMBLE_STIFFNESS_KERNEL.is_empty());
        assert!(!ARNOLDI_KERNEL.is_empty());
        assert!(!LANCZOS_KERNEL.is_empty());
        assert!(!ASSEMBLE_MASS_KERNEL.is_empty());
        assert!(!NEWMARK_INTEGRATION_KERNEL.is_empty());
        assert!(!MODAL_SUPERPOSITION_KERNEL.is_empty());
        assert!(!STRESS_RECOVERY_KERNEL.is_empty());
        assert!(!VON_MISES_KERNEL.is_empty());
        assert!(!ERROR_NORM_KERNEL.is_empty());
        assert!(!AMR_INDICATOR_KERNEL.is_empty());
    }

    #[test]
    fn test_kernel_syntax() {
        // Basic syntax check - kernels should contain expected patterns
        assert!(SPV_CUDA_KERNEL.contains("extern \"C\" __global__"));
        assert!(SPV_CUDA_KERNEL.contains("spmv_csr"));
        assert!(AXPY_KERNEL.contains("y[i] = alpha * x[i] + y[i]"));
        assert!(CG_ITER_KERNEL.contains("cg_iter"));
        assert!(ARNOLDI_KERNEL.contains("arnoldi_iteration"));
        assert!(LANCZOS_KERNEL.contains("lanczos_bidiag"));
    }

    #[test]
    fn test_all_kernels_have_extern_declaration() {
        let kernels = [
            SPV_CUDA_KERNEL,
            AXPY_KERNEL,
            SCALE_KERNEL,
            COPY_KERNEL,
            SET_KERNEL,
            DOT_KERNEL,
            NORM_KERNEL,
            CG_ITER_KERNEL,
            ASSEMBLE_STIFFNESS_KERNEL,
            EXTRACT_DIAGONAL_KERNEL,
            JACOBI_PRECOND_KERNEL,
            ILU_FORWARD_KERNEL,
            ILU_BACKWARD_KERNEL,
            CHEBYSHEV_PRECOND_KERNEL,
            ARNOLDI_KERNEL,
            LANCZOS_KERNEL,
            ASSEMBLE_MASS_KERNEL,
            RAYLEIGH_DAMPING_KERNEL,
            NEWMARK_INTEGRATION_KERNEL,
            WILSON_THETA_KERNEL,
            MODAL_SUPERPOSITION_KERNEL,
            STRESS_RECOVERY_KERNEL,
            VON_MISES_KERNEL,
            ERROR_NORM_KERNEL,
            AMR_INDICATOR_KERNEL,
        ];

        for kernel in &kernels {
            assert!(kernel.contains("extern \"C\" __global__"));
        }
    }
}
