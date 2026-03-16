//! OpenCL kernel source code for FEA acceleration.
//!
//! This module contains OpenCL kernel source that can be compiled
//! at runtime for cross-platform GPU acceleration.
//!
//! # Supported Operations
//!
//! - Sparse matrix operations (SpMV, SpMM)
//! - Vector operations
//! - Iterative solver kernels
//! - Assembly kernels
//! - Preconditioner kernels

/// OpenCL program header with common utilities.
pub const OCL_HEADER: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Common constants
#define WARP_SIZE 32
#define MAX_LOCAL_SIZE 256

// Atomic max for doubles (emulated)
inline double atomic_max_double(volatile __global double* p, double val) {
    union {
        double d;
        ulong u;
    } old, assumed;

    old.u = as_ulong(vload_half(0, (const half*)p));
    do {
        assumed.u = old.u;
        if (assumed.d >= val) break;
        old.u = atom_cmpxchg((volatile __global ulong*)p, assumed.u,
                             (val > assumed.d) ? as_ulong(val) : assumed.u);
    } while (old.u != assumed.u);

    return (old.d > val) ? old.d : val;
}
"#;

/// OpenCL kernel for sparse matrix-vector multiplication (CSR format).
/// Each work-item computes one element of the output vector.
pub const OCL_SPMV_KERNEL: &str = r#"
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

/// Optimized SpMV kernel using shared memory (local memory in OpenCL).
pub const OCL_SPMV_SHARED_KERNEL: &str = r#"
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

    // Load frequently accessed x values into local memory
    for (int i = tid; i < nnz_row; i += local_size) {
        int col = col_ind[row_start + i];
        shared_x[i] = x[col];
        shared_col[i] = col;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute dot product using local memory
    double sum = 0.0;
    for (int j = 0; j < nnz_row; j++) {
        sum += values[row_start + j] * shared_x[j];
    }

    y[row] = alpha * sum + beta * y[row];
}
"#;

/// OpenCL kernel for vector AXPY: y = alpha * x + y
pub const OCL_AXPY_KERNEL: &str = r#"
__kernel void axpy(const int n,
                   const double alpha,
                   __global const double* x,
                   __global double* y) {
    int i = get_global_id(0);
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}
"#;

/// OpenCL kernel for vector scaling: y = alpha * x
pub const OCL_SCALE_KERNEL: &str = r#"
__kernel void scale(const int n,
                    const double alpha,
                    __global const double* x,
                    __global double* y) {
    int i = get_global_id(0);
    if (i < n) {
        y[i] = alpha * x[i];
    }
}
"#;

/// OpenCL kernel for vector copy: dst = src
pub const OCL_COPY_KERNEL: &str = r#"
__kernel void copy(const int n,
                   __global const double* src,
                   __global double* dst) {
    int i = get_global_id(0);
    if (i < n) {
        dst[i] = src[i];
    }
}
"#;

/// OpenCL kernel for vector set: x = value
pub const OCL_SET_KERNEL: &str = r#"
__kernel void set(const int n,
                  const double value,
                  __global double* x) {
    int i = get_global_id(0);
    if (i < n) {
        x[i] = value;
    }
}
"#;

/// OpenCL kernel for dot product with partial sum reduction.
pub const OCL_DOT_KERNEL: &str = r#"
__kernel void dot_partial(const int n,
                          __global const double* x,
                          __global const double* y,
                          __global double* partial) {
    int tid = get_local_id(0);
    int gid = get_global_id(0);
    int local_size = get_local_size(0);

    extern __local double sdata[];

    double sum = 0.0;
    if (gid < n) {
        sum = x[gid] * y[gid];
    }

    sdata[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction in local memory
    for (unsigned int s = local_size / 2; s > 0; s >>= 1) {
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

/// OpenCL kernel for computing vector norm (squared).
pub const OCL_NORM_KERNEL: &str = r#"
__kernel void norm_squared_partial(const int n,
                                   __global const double* x,
                                   __global double* partial) {
    int tid = get_local_id(0);
    int gid = get_global_id(0);
    int local_size = get_local_size(0);

    extern __local double sdata[];

    double sum = 0.0;
    if (gid < n) {
        sum = x[gid] * x[gid];
    }

    sdata[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction
    for (unsigned int s = local_size / 2; s > 0; s >>= 1) {
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

/// OpenCL kernel for Conjugate Gradient iteration step.
/// Combines multiple operations for efficiency.
pub const OCL_CG_ITER_KERNEL: &str = r#"
__kernel void cg_iter(const int n,
                      __global const double* values,
                      __global const int* row_ptr,
                      __global const int* col_ind,
                      __global double* x,
                      __global double* r,
                      __global double* p,
                      __global double* ap,
                      __global double* partial,
                      const int use_jacobi,
                      __global const double* diag_inv) {
    int i = get_global_id(0);
    int tid = get_local_id(0);
    int local_size = get_local_size(0);

    extern __local double sdata_p_ap[];
    extern __local double sdata_r_z[];

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
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Compute p^T * Ap
    double p_ap = 0.0;
    if (i < n) {
        p_ap = p[i] * ap[i];
    }
    sdata_p_ap[tid] = p_ap;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int s = local_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_p_ap[tid] += sdata_p_ap[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        partial[0] = sdata_p_ap[0];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Compute r^T * z (Jacobi preconditioning)
    double rz = 0.0;
    if (i < n && use_jacobi) {
        double z_i = r[i] * diag_inv[i];
        rz = r[i] * z_i;
    } else if (i < n) {
        rz = r[i] * r[i];
    }
    sdata_r_z[tid] = rz;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int s = local_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_r_z[tid] += sdata_r_z[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        partial[1] = sdata_r_z[0];
    }
}
"#;

/// OpenCL kernel for element stiffness matrix assembly.
pub const OCL_ASSEMBLE_KERNEL: &str = r#"
__kernel void assemble_stiffness(const int n_elements,
                                 const int n_dofs_per_elem,
                                 __global const double* element_matrices,
                                 __global const int* connectivity,
                                 __global int* row_ptr,
                                 __global int* col_ind,
                                 __global double* values) {
    int elem = get_global_id(0);

    if (elem < n_elements) {
        int elem_offset = elem * n_dofs_per_elem * n_dofs_per_elem;

        // Get node indices (assuming 2 nodes per element for truss)
        int node0 = connectivity[elem * 2];
        int node1 = connectivity[elem * 2 + 1];

        // For 2D truss: 2 DOFs per node
        int dof0 = node0 * 2;
        int dof1 = node1 * 2;
        int dofs[2] = {dof0, dof1};

        // Assemble into global matrix
        for (int i = 0; i < n_dofs_per_elem; i++) {
            for (int j = 0; j < n_dofs_per_elem; j++) {
                int row = dofs[i];
                int col = dofs[j];
                double val = element_matrices[elem_offset + i * n_dofs_per_elem + j];

                // Find position in CSR (simplified - binary search would be better)
                for (int k = row_ptr[row]; k < row_ptr[row + 1]; k++) {
                    if (col_ind[k] == col) {
                        atomic_add(&values[k], val);
                        break;
                    }
                }
            }
        }
    }
}
"#;

/// OpenCL kernel for Jacobi preconditioner.
pub const OCL_JACOBI_KERNEL: &str = r#"
__kernel void jacobi_precond(const int n,
                             __global const double* diag,
                             __global const double* r,
                             __global double* z) {
    int i = get_global_id(0);
    if (i < n) {
        double d = diag[i];
        z[i] = (d > 1e-15 && d < 1e15) ? r[i] / d : r[i];
    }
}
"#;

/// OpenCL kernel for SSOR preconditioner (forward sweep).
pub const OCL_SSOR_FORWARD_KERNEL: &str = r#"
__kernel void ssor_forward(const int n,
                           __global const double* values,
                           __global const int* row_ptr,
                           __global const int* col_ind,
                           __global const double* r,
                           __global double* y,
                           const double omega) {
    int i = get_global_id(0);
    if (i >= n) return;

    double sum = r[i];
    int row_start = row_ptr[i];
    int row_end = row_ptr[i + 1];

    // Forward sweep - only use already computed values
    for (int j = row_start; j < row_end; j++) {
        int col = col_ind[j];
        if (col < i) {
            sum -= values[j] * y[col];
        }
    }

    double diag = 1.0;
    for (int j = row_start; j < row_end; j++) {
        if (col_ind[j] == i) {
            diag = values[j];
            break;
        }
    }

    y[i] = (1.0 - omega) * r[i] + omega * sum / diag;
}
"#;

/// OpenCL kernel for SSOR preconditioner (backward sweep).
pub const OCL_SSOR_BACKWARD_KERNEL: &str = r#"
__kernel void ssor_backward(const int n,
                            __global const double* values,
                            __global const int* row_ptr,
                            __global const int* col_ind,
                            __global const double* y,
                            __global double* x,
                            const double omega) {
    int i = n - 1 - get_global_id(0);
    if (i < 0 || i >= n) return;

    double sum = y[i];
    int row_start = row_ptr[i];
    int row_end = row_ptr[i + 1];

    // Backward sweep - only use already computed values
    for (int j = row_start; j < row_end; j++) {
        int col = col_ind[j];
        if (col > i) {
            sum -= values[j] * x[col];
        }
    }

    double diag = 1.0;
    for (int j = row_start; j < row_end; j++) {
        if (col_ind[j] == i) {
            diag = values[j];
            break;
        }
    }

    x[i] = (1.0 - omega) * y[i] + omega * sum / diag;
}
"#;

/// OpenCL kernel for Chebyshev polynomial preconditioner.
pub const OCL_CHEBYSHEV_KERNEL: &str = r#"
__kernel void chebyshev_precond(const int n,
                                __global const double* values,
                                __global const int* row_ptr,
                                __global const int* col_ind,
                                __global const double* r,
                                __global double* z,
                                const double lambda_min,
                                const double lambda_max,
                                const int degree) {
    int i = get_global_id(0);
    if (i >= n) return;

    // Get diagonal for initial estimate
    double diag = 1.0;
    int row_start = row_ptr[i];
    for (int j = row_start; j < row_ptr[i + 1]; j++) {
        if (col_ind[j] == i) {
            diag = values[j];
            break;
        }
    }

    // Chebyshev parameters
    double theta = (lambda_max - lambda_min) / (lambda_max + lambda_min);
    double rho = (1.0 - theta) / (1.0 + theta);

    // Initial guess
    double z_curr = r[i] / (diag > 1e-15 ? diag : 1.0);
    double z_prev = 0.0;

    // Chebyshev iterations
    for (int k = 0; k < degree; k++) {
        double z_next = 2.0 * r[i] / diag - z_prev;
        z_prev = z_curr;
        z_curr = z_next;
    }

    z[i] = z_curr;
}
"#;

/// OpenCL kernel for ILU(0) forward substitution.
pub const OCL_ILU_FORWARD_KERNEL: &str = r#"
__kernel void ilu_forward(const int n,
                          __global const double* values,
                          __global const int* row_ptr,
                          __global const int* col_ind,
                          __global const double* r,
                          __global double* y) {
    int i = get_global_id(0);
    if (i >= n) return;

    double sum = r[i];
    int row_start = row_ptr[i];
    int row_end = row_ptr[i + 1];

    // Forward substitution
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
"#;

/// OpenCL kernel for ILU(0) backward substitution.
pub const OCL_ILU_BACKWARD_KERNEL: &str = r#"
__kernel void ilu_backward(const int n,
                           __global const double* values,
                           __global const int* row_ptr,
                           __global const int* col_ind,
                           __global const double* y,
                           __global double* x) {
    int i = n - 1 - get_global_id(0);
    if (i < 0 || i >= n) return;

    double sum = y[i];
    int row_start = row_ptr[i];
    int row_end = row_ptr[i + 1];

    // Backward substitution
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
"#;

/// OpenCL kernel for parallel reduction (sum).
pub const OCL_REDUCE_SUM_KERNEL: &str = r#"
__kernel void reduce_sum(__global const double* input,
                         __global double* output,
                         const int n) {
    extern __local double sdata[];

    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0);

    double sum = 0.0;
    if (gid < n) {
        sum = input[gid];
    }

    sdata[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction
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

/// OpenCL kernel for parallel reduction (max).
pub const OCL_REDUCE_MAX_KERNEL: &str = r#"
__kernel void reduce_max(__global const double* input,
                         __global double* output,
                         const int n) {
    extern __local double sdata[];

    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0);

    double max_val = -INFINITY;
    if (gid < n) {
        max_val = input[gid];
    }

    sdata[tid] = max_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction for max
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
    }
}
"#;

/// OpenCL kernel for GMRES Arnoldi iteration.
pub const OCL_ARNOLDI_KERNEL: &str = r#"
__kernel void arnoldi_iteration(const int n,
                                const int k,
                                __global const double* values,
                                __global const int* row_ptr,
                                __global const int* col_ind,
                                __global double* hessenberg,
                                __global double* v,
                                __global double* w) {
    int i = get_global_id(0);
    if (i >= n) return;

    // Compute w = A * v_k
    double sum = 0.0;
    int row_start = row_ptr[i];
    int row_end = row_ptr[i + 1];

    for (int j = row_start; j < row_end; j++) {
        sum += values[j] * v[col_ind[j]];
    }
    w[i] = sum;

    // Modified Gram-Schmidt (simplified for OpenCL)
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
"#;

/// OpenCL kernel for Lanczos bidiagonalization.
pub const OCL_LANCZOS_KERNEL: &str = r#"
__kernel void lanczos_bidiag(const int n,
                             __global const double* values,
                             __global const int* row_ptr,
                             __global const int* col_ind,
                             __global double* alpha,
                             __global double* beta,
                             __global double* u,
                             __global double* v,
                             const int iteration) {
    int i = get_global_id(0);
    if (i >= n) return;

    if (iteration == 0) {
        // First iteration: u = A * v
        double sum = 0.0;
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * v[col_ind[j]];
        }
        u[i] = sum;
    } else {
        // Subsequent iterations
        u[i] = u[i] - beta[iteration - 1] * v[i];
    }

    // Compute norm for alpha (requires reduction - simplified here)
    if (i == 0) {
        double norm = 0.0;
        for (int m = 0; m < n; m++) {
            norm += u[m] * u[m];
        }
        alpha[iteration] = sqrt(norm);
    }
}
"#;

/// OpenCL kernel for element mass matrix assembly.
pub const OCL_ASSEMBLE_MASS_KERNEL: &str = r#"
__kernel void assemble_mass(const int n_elements,
                            const int n_dofs_per_elem,
                            const double density,
                            __global const int* connectivity,
                            __global int* row_ptr,
                            __global int* col_ind,
                            __global double* mass_values) {
    int elem = get_global_id(0);
    if (elem >= n_elements) return;

    int node0 = connectivity[elem * 2];
    int node1 = connectivity[elem * 2 + 1];
    int dof0 = node0 * 2;
    int dof1 = node1 * 2;

    // Consistent mass matrix for 2D truss
    double m = density * 0.5;

    // Assemble diagonal terms
    int dofs[2] = {dof0, dof1};
    for (int i = 0; i < 2; i++) {
        int row = dofs[i];
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            if (col_ind[j] == row) {
                atomic_add(&mass_values[j], m);
                break;
            }
        }
    }
}
"#;

/// OpenCL kernel for Rayleigh damping computation.
pub const OCL_RAYLEIGH_DAMPING_KERNEL: &str = r#"
__kernel void rayleigh_damping(const int n,
                               const double alpha_m,
                               const double alpha_k,
                               __global const double* mass_diag,
                               __global const double* stiff_diag,
                               __global double* damping_diag) {
    int i = get_global_id(0);
    if (i >= n) return;

    // C = alpha_m * M + alpha_k * K
    damping_diag[i] = alpha_m * mass_diag[i] + alpha_k * stiff_diag[i];
}
"#;

/// OpenCL kernel for Newmark-beta time integration.
pub const OCL_NEWMARK_KERNEL: &str = r#"
__kernel void newmark_integrate(const int n,
                                __global const double* u_n,
                                __global const double* v_n,
                                __global const double* a_n,
                                __global double* u_np1,
                                __global double* v_np1,
                                __global double* a_np1,
                                const double dt,
                                const double beta,
                                const double gamma) {
    int i = get_global_id(0);
    if (i >= n) return;

    // Newmark-beta formulas
    double a = gamma / (beta * dt);
    double b = 1.0 / (beta * dt * dt);

    // u_{n+1} = u_n + dt * v_n + dt^2 * ((1/2 - beta) * a_n + beta * a_{n+1})
    u_np1[i] = u_n[i] + dt * v_n[i] + dt * dt * ((0.5 - beta) * a_n[i] + beta * a_np1[i]);

    // v_{n+1} = v_n + dt * ((1 - gamma) * a_n + gamma * a_{n+1})
    v_np1[i] = v_n[i] + dt * ((1.0 - gamma) * a_n[i] + gamma * a_np1[i]);
}
"#;

/// OpenCL kernel for Wilson-theta time integration.
pub const OCL_WILSON_THETA_KERNEL: &str = r#"
__kernel void wilson_theta(const int n,
                           __global const double* u_n,
                           __global const double* v_n,
                           __global const double* a_n,
                           __global double* u_np1,
                           __global double* v_np1,
                           __global double* a_np1,
                           const double dt,
                           const double theta) {
    int i = get_global_id(0);
    if (i >= n) return;

    // Wilson-theta formulas
    u_np1[i] = u_n[i] + dt * v_n[i] + dt * dt * ((0.5 - 1.0 / (6.0 * theta)) * a_n[i]
                  + 1.0 / (6.0 * theta) * a_np1[i]);

    v_np1[i] = v_n[i] + dt * ((1.0 - 0.5 / theta) * a_n[i] + 0.5 / theta * a_np1[i]);
}
"#;

/// OpenCL kernel for modal superposition.
pub const OCL_MODAL_SUPERPOSITION_KERNEL: &str = r#"
__kernel void modal_superposition(const int n_dofs,
                                  const int n_modes,
                                  __global const double* modes,
                                  __global const double* modal_disp,
                                  __global double* physical_disp) {
    int i = get_global_id(0);
    if (i >= n_dofs) return;

    double disp = 0.0;
    for (int m = 0; m < n_modes; m++) {
        disp += modes[i + m * n_dofs] * modal_disp[m];
    }
    physical_disp[i] = disp;
}
"#;

/// OpenCL kernel for stress recovery.
pub const OCL_STRESS_RECOVERY_KERNEL: &str = r#"
__kernel void stress_recovery(const int n_elements,
                              __global const double* displacements,
                              __global const int* connectivity,
                              __global const double* D_matrix,
                              __global const double* B_matrix,
                              __global double* stresses) {
    int elem = get_global_id(0);
    if (elem >= n_elements) return;

    int node0 = connectivity[elem * 2];
    int node1 = connectivity[elem * 2 + 1];

    // Extract element displacements
    double u0_x = displacements[node0 * 2];
    double u0_y = displacements[node0 * 2 + 1];
    double u1_x = displacements[node1 * 2];
    double u1_y = displacements[node1 * 2 + 1];

    // Compute strain: epsilon = B * u (simplified 2D truss)
    double strain = (u1_x - u0_x); // Axial strain

    // Compute stress: sigma = E * strain
    stresses[elem] = D_matrix[0] * strain;
}
"#;

/// OpenCL kernel for von Mises stress computation.
pub const OCL_VON_MISES_KERNEL: &str = r#"
__kernel void von_mises_stress(const int n_elements,
                               __global const double* stresses,
                               __global double* von_mises) {
    int elem = get_global_id(0);
    if (elem >= n_elements) return;

    double sx = stresses[elem * 3];
    double sy = stresses[elem * 3 + 1];
    double txy = stresses[elem * 3 + 2];

    // von Mises: sqrt(sx^2 + sy^2 - sx*sy + 3*txy^2)
    double vm = sqrt(sx * sx + sy * sy - sx * sy + 3.0 * txy * txy);
    von_mises[elem] = vm;
}
"#;

/// OpenCL kernel for error norm computation.
pub const OCL_ERROR_NORM_KERNEL: &str = r#"
__kernel void error_norm(const int n,
                         __global const double* u_exact,
                         __global const double* u_fem,
                         __global const double* weights,
                         __global double* l2_norm,
                         __global double* h1_norm) {
    int i = get_global_id(0);
    if (i >= n) return;

    double diff = u_fem[i] - u_exact[i];
    l2_norm[i] = diff * diff * weights[i];
    h1_norm[i] = diff * diff;
}
"#;

/// OpenCL kernel for adaptive mesh refinement indicator.
pub const OCL_AMR_INDICATOR_KERNEL: &str = r#"
__kernel void amr_indicator(const int n_elements,
                            __global const double* element_errors,
                            __global const double* element_energies,
                            __global double* refinement_indicators,
                            const double threshold) {
    int elem = get_global_id(0);
    if (elem >= n_elements) return;

    double indicator = element_errors[elem] / (element_energies[elem] + 1e-15);
    refinement_indicators[elem] = (indicator > threshold) ? 1.0 : 0.0;
}
"#;

/// OpenCL kernel for mesh quality metric computation.
pub const OCL_MESH_QUALITY_KERNEL: &str = r#"
__kernel void mesh_quality(const int n_elements,
                           __global const double* node_coords,
                           __global const int* connectivity,
                           __global double* aspect_ratios,
                           __global double* skew_angles) {
    int elem = get_global_id(0);
    if (elem >= n_elements) return;

    // Get node coordinates for quad element
    int n0 = connectivity[elem * 4];
    int n1 = connectivity[elem * 4 + 1];
    int n2 = connectivity[elem * 4 + 2];
    int n3 = connectivity[elem * 4 + 3];

    // Compute edge lengths
    double x0 = node_coords[n0 * 2], y0 = node_coords[n0 * 2 + 1];
    double x1 = node_coords[n1 * 2], y1 = node_coords[n1 * 2 + 1];
    double x2 = node_coords[n2 * 2], y2 = node_coords[n2 * 2 + 1];
    double x3 = node_coords[n3 * 2], y3 = node_coords[n3 * 2 + 1];

    double l01 = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
    double l12 = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
    double l23 = sqrt((x3-x2)*(x3-x2) + (y3-y2)*(y3-y2));
    double l30 = sqrt((x0-x3)*(x0-x3) + (y0-y3)*(y0-y3));

    // Aspect ratio (max/min edge length)
    double max_l = fmax(fmax(l01, l12), fmax(l23, l30));
    double min_l = fmin(fmin(l01, l12), fmin(l23, l30));
    aspect_ratios[elem] = max_l / (min_l + 1e-15);

    // Skew angle (simplified)
    double dot1 = (x1-x0)*(x3-x0) + (y1-y0)*(y3-y0);
    double dot2 = (x2-x1)*(x0-x1) + (y2-y1)*(y0-y1);
    skew_angles[elem] = acos(fmin(1.0, fmax(-1.0, dot1 / (l01 * l30 + 1e-15)))) * 180.0 / 3.14159265359;
}
"#;

/// Complete OpenCL program containing all kernels.
pub const OCL_FULL_PROGRAM: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// All kernel definitions would be concatenated here
// This is a placeholder for the complete program
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencl_kernels_exist() {
        assert!(!OCL_SPMV_KERNEL.is_empty());
        assert!(!OCL_AXPY_KERNEL.is_empty());
        assert!(!OCL_DOT_KERNEL.is_empty());
        assert!(!OCL_CG_ITER_KERNEL.is_empty());
        assert!(!OCL_JACOBI_KERNEL.is_empty());
        assert!(!OCL_ILU_FORWARD_KERNEL.is_empty());
        assert!(!OCL_ILU_BACKWARD_KERNEL.is_empty());
        assert!(!OCL_ARNOLDI_KERNEL.is_empty());
        assert!(!OCL_LANCZOS_KERNEL.is_empty());
        assert!(!OCL_ASSEMBLE_MASS_KERNEL.is_empty());
        assert!(!OCL_RAYLEIGH_DAMPING_KERNEL.is_empty());
        assert!(!OCL_NEWMARK_KERNEL.is_empty());
        assert!(!OCL_MODAL_SUPERPOSITION_KERNEL.is_empty());
        assert!(!OCL_STRESS_RECOVERY_KERNEL.is_empty());
        assert!(!OCL_VON_MISES_KERNEL.is_empty());
        assert!(!OCL_ERROR_NORM_KERNEL.is_empty());
        assert!(!OCL_AMR_INDICATOR_KERNEL.is_empty());
        assert!(!OCL_MESH_QUALITY_KERNEL.is_empty());
    }

    #[test]
    fn test_opencl_kernel_syntax() {
        // Basic syntax check - kernels should contain expected OpenCL patterns
        assert!(OCL_SPMV_KERNEL.contains("__kernel void spmv_csr"));
        assert!(OCL_AXPY_KERNEL.contains("__kernel void axpy"));
        assert!(OCL_DOT_KERNEL.contains("__local double sdata"));
        assert!(OCL_CG_ITER_KERNEL.contains("barrier"));
        assert!(OCL_ARNOLDI_KERNEL.contains("__kernel void arnoldi_iteration"));
        assert!(OCL_LANCZOS_KERNEL.contains("__kernel void lanczos_bidiag"));
    }

    #[test]
    fn test_all_kernels_have_kernel_declaration() {
        let kernels = [
            OCL_SPMV_KERNEL,
            OCL_AXPY_KERNEL,
            OCL_SCALE_KERNEL,
            OCL_COPY_KERNEL,
            OCL_SET_KERNEL,
            OCL_DOT_KERNEL,
            OCL_NORM_KERNEL,
            OCL_CG_ITER_KERNEL,
            OCL_ASSEMBLE_KERNEL,
            OCL_JACOBI_KERNEL,
            OCL_SSOR_FORWARD_KERNEL,
            OCL_SSOR_BACKWARD_KERNEL,
            OCL_CHEBYSHEV_KERNEL,
            OCL_ILU_FORWARD_KERNEL,
            OCL_ILU_BACKWARD_KERNEL,
            OCL_REDUCE_SUM_KERNEL,
            OCL_REDUCE_MAX_KERNEL,
            OCL_ARNOLDI_KERNEL,
            OCL_LANCZOS_KERNEL,
            OCL_ASSEMBLE_MASS_KERNEL,
            OCL_RAYLEIGH_DAMPING_KERNEL,
            OCL_NEWMARK_KERNEL,
            OCL_WILSON_THETA_KERNEL,
            OCL_MODAL_SUPERPOSITION_KERNEL,
            OCL_STRESS_RECOVERY_KERNEL,
            OCL_VON_MISES_KERNEL,
            OCL_ERROR_NORM_KERNEL,
            OCL_AMR_INDICATOR_KERNEL,
            OCL_MESH_QUALITY_KERNEL,
        ];

        for kernel in &kernels {
            assert!(kernel.contains("__kernel void"));
        }
    }

    #[test]
    fn test_opencl_kernels_have_barrier_synchronization() {
        // Kernels using local memory should have barrier synchronization
        assert!(OCL_DOT_KERNEL.contains("barrier"));
        assert!(OCL_NORM_KERNEL.contains("barrier"));
        assert!(OCL_CG_ITER_KERNEL.contains("barrier"));
        assert!(OCL_REDUCE_SUM_KERNEL.contains("barrier"));
        assert!(OCL_REDUCE_MAX_KERNEL.contains("barrier"));
    }
}
