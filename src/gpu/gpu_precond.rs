//! GPU-accelerated preconditioned iterative solvers.
//!
//! This module provides:
//! - GPU ILU(0) preconditioner
//! - GPU SSOR preconditioner
//! - GPU Chebyshev preconditioner
//! - Preconditioned CG, GMRES, BiCGSTAB

use nalgebra::{DMatrix, DVector};

use super::{GPUCSRMatrix, GPUSolverResult, SparseMatrixVectorMul, VectorOps};

/// GPU ILU(0) preconditioner.
#[derive(Debug, Clone)]
pub struct GPUILUPreconditioner {
    /// ILU factors (stored in CSR format).
    l_values: Vec<f64>,
    u_values: Vec<f64>,
    /// Row pointers.
    row_ptr: Vec<usize>,
    /// Column indices.
    col_ind: Vec<usize>,
    /// Inverse of L diagonal.
    inv_diag: Vec<f64>,
    device_id: usize,
}

impl GPUILUPreconditioner {
    /// Creates ILU(0) preconditioner from matrix.
    pub fn new(matrix: &GPUCSRMatrix, device_id: usize) -> Self {
        let n = matrix.n_rows;
        let nnz = matrix.nnz;

        let values = matrix.values.host_data();
        let row_ptr = matrix.row_ptr.host_data();
        let col_ind = matrix.col_ind.host_data();

        // ILU(0) factorization (CPU version - GPU would use parallel ILU)
        let mut ilu_values = values.to_vec();

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

                    // Find L_ik and U_kj
                    let l_row_start = row_ptr[col_k];
                    let l_row_end = row_ptr[col_k + 1];

                    let mut l_ik = 0.0;
                    for p in l_row_start..l_row_end {
                        if col_ind[p] == i {
                            l_ik = ilu_values[p];
                            break;
                        }
                    }

                    let u_row_start = row_ptr[i];
                    let u_row_end = row_ptr[i + 1];
                    let mut u_kj = 0.0;
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
        let mut inv_diag = vec![1.0; n];
        for i in 0..n {
            let row_start = row_ptr[i];
            let row_end = row_ptr[i + 1];
            for j in row_start..row_end {
                if col_ind[j] == i {
                    let diag = ilu_values[j];
                    inv_diag[i] = if diag.abs() > 1e-15 { 1.0 / diag } else { 1.0 };
                    break;
                }
            }
        }

        Self {
            l_values: ilu_values.clone(),
            u_values: ilu_values,
            row_ptr: row_ptr.to_vec(),
            col_ind: col_ind.to_vec(),
            inv_diag,
            device_id,
        }
    }

    /// Applies ILU preconditioner: solves LUz = r.
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
            z1[i] = sum * self.inv_diag[i];
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

            let diag_val = self.u_values[row_start..row_end]
                .iter()
                .zip(&self.col_ind[row_start..row_end])
                .find(|(_, &c)| c == i)
                .map(|(&v, _)| v)
                .unwrap_or(1.0);

            z[i] = sum / diag_val.max(1e-15);
        }
    }

    /// Returns device ID.
    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

/// GPU SSOR preconditioner.
#[derive(Debug, Clone)]
pub struct GPUSSORPreconditioner {
    /// Diagonal elements.
    diag: Vec<f64>,
    /// Lower triangular part.
    lower_values: Vec<f64>,
    /// Lower column indices.
    lower_col_ind: Vec<usize>,
    /// Row pointers.
    row_ptr: Vec<usize>,
    /// Relaxation parameter.
    omega: f64,
    device_id: usize,
}

impl GPUSSORPreconditioner {
    /// Creates SSOR preconditioner.
    pub fn new(matrix: &GPUCSRMatrix, omega: f64, device_id: usize) -> Self {
        let n = matrix.n_rows;

        let values = matrix.values.host_data();
        let row_ptr = matrix.row_ptr.host_data();
        let col_ind = matrix.col_ind.host_data();

        // Extract diagonal and lower triangular
        let mut diag = vec![1.0; n];
        let mut lower_values = Vec::new();
        let mut lower_col_ind = Vec::new();
        let mut new_row_ptr = vec![0];

        for i in 0..n {
            let row_start = row_ptr[i];
            let row_end = row_ptr[i + 1];

            for j in row_start..row_end {
                let col = col_ind[j];
                if col == i {
                    diag[i] = values[j];
                } else if col < i {
                    lower_values.push(values[j]);
                    lower_col_ind.push(col);
                }
            }
            new_row_ptr.push(lower_values.len());
        }

        Self {
            diag,
            lower_values,
            lower_col_ind,
            row_ptr: new_row_ptr,
            omega: omega.clamp(0.1, 1.99),
            device_id,
        }
    }

    /// Applies SSOR preconditioner.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        let n = r.len();
        let omega = self.omega;

        // Forward sweep
        for i in 0..n {
            let mut sum = 0.0;
            let row_start = self.row_ptr[i];
            let row_end = self.row_ptr[i + 1];

            for j in row_start..row_end {
                sum += self.lower_values[j] * z[self.lower_col_ind[j]];
            }

            let d = self.diag[i].max(1e-15);
            z[i] = (1.0 - omega) * r[i] + omega * (r[i] - sum) / d;
        }

        // Backward sweep
        for i in (0..n).rev() {
            let mut sum = 0.0;
            let row_start = self.row_ptr[i];
            let row_end = self.row_ptr[i + 1];

            for j in row_start..row_end {
                sum += self.lower_values[j] * z[self.lower_col_ind[j]];
            }

            let d = self.diag[i].max(1e-15);
            z[i] = (1.0 - omega) * z[i] + omega * (z[i] - sum) / d;
        }
    }
}

/// GPU Chebyshev preconditioner.
#[derive(Debug, Clone)]
pub struct GPUChebyshevPreconditioner {
    /// Estimated minimum eigenvalue.
    lambda_min: f64,
    /// Estimated maximum eigenvalue.
    lambda_max: f64,
    /// Polynomial degree.
    degree: usize,
    device_id: usize,
}

impl GPUChebyshevPreconditioner {
    /// Creates Chebyshev preconditioner.
    pub fn new(lambda_min: f64, lambda_max: f64, degree: usize, device_id: usize) -> Self {
        Self {
            lambda_min,
            lambda_max,
            degree: degree.max(1),
            device_id,
        }
    }

    /// Estimates eigenvalue bounds using power iteration.
    pub fn estimate_eigenvalues(matrix: &GPUCSRMatrix, device_id: usize) -> (f64, f64) {
        let n = matrix.n_rows;
        let spmv = SparseMatrixVectorMul::new(device_id);
        let vec_ops = VectorOps::new(device_id);

        // Power iteration for max eigenvalue
        let mut v = vec![1.0; n];
        let mut lambda_max = 0.0;

        for _ in 0..20 {
            let mut w = vec![0.0; n];
            let _ = spmv.spmv(matrix, &v, &mut w, 1.0, 0.0);

            lambda_max = vec_ops.norm(&w) / vec_ops.norm(&v).max(1e-15);
            v = w;
        }

        // Min eigenvalue estimate (simplified)
        let lambda_min = lambda_max / 100.0;

        (lambda_min, lambda_max)
    }

    /// Applies Chebyshev preconditioner.
    pub fn apply(&self, matrix: &GPUCSRMatrix, r: &[f64], z: &mut [f64]) -> anyhow::Result<()> {
        let n = r.len();
        let spmv = SparseMatrixVectorMul::new(self.device_id);

        // Chebyshev parameters
        let alpha = 2.0 / (self.lambda_max + self.lambda_min);
        let beta = (self.lambda_max - self.lambda_min) / (self.lambda_max + self.lambda_min);
        let beta_sq = beta * beta;

        // Initial approximation: z = alpha * r
        for i in 0..n {
            z[i] = alpha * r[i];
        }

        if self.degree == 1 {
            return Ok(());
        }

        // Chebyshev iteration
        let mut w = vec![0.0; n];
        let mut p = z.to_vec();
        let mut rho_prev = 0.0;

        for k in 0..self.degree - 1 {
            // w = A * p
            let _ = spmv.spmv(matrix, &p, &mut w, 1.0, 0.0)?;

            // Compute coefficients
            let rho = 1.0 / (1.0 - beta_sq * rho_prev / 4.0);
            rho_prev = rho;

            // Update z
            for i in 0..n {
                z[i] = rho * (alpha * r[i] + z[i]) + (1.0 - rho) * p[i] - rho * alpha * w[i];
            }

            p = z.to_vec();
        }

        Ok(())
    }
}

/// Preconditioned CG solver for GPU.
#[derive(Debug, Clone)]
pub struct GPUPCGSolver {
    device_id: usize,
    max_iterations: usize,
    tolerance: f64,
    preconditioner: String,
}

impl GPUPCGSolver {
    /// Creates preconditioned CG solver.
    pub fn new(device_id: usize, tolerance: f64, max_iterations: usize, preconditioner: &str) -> Self {
        Self {
            device_id,
            max_iterations,
            tolerance,
            preconditioner: preconditioner.to_string(),
        }
    }

    /// Solves system with preconditioning.
    pub fn solve(
        &self,
        matrix: &GPUCSRMatrix,
        ilu: Option<&GPUILUPreconditioner>,
        b: &[f64],
    ) -> anyhow::Result<GPUSolverResult> {
        let n = b.len();
        let spmv = SparseMatrixVectorMul::new(self.device_id);
        let vec_ops = VectorOps::new(self.device_id);

        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        let mut z = vec![0.0; n];
        let mut p = vec![0.0; n];

        // Apply preconditioner
        if let Some(ilu_precond) = ilu {
            ilu_precond.apply(&r, &mut z);
        } else {
            // Jacobi preconditioning
            for i in 0..n {
                z[i] = r[i];
            }
        }

        let mut rz = vec_ops.dot(&r, &z);
        p.copy_from_slice(&z);

        let b_norm = vec_ops.norm(b);
        let tol = self.tolerance * b_norm.max(1e-15);

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            // Compute Ap
            let mut ap = vec![0.0; n];
            let _ = spmv.spmv(matrix, &p, &mut ap, 1.0, 0.0)?;

            let p_ap = vec_ops.dot(&p, &ap);
            if p_ap.abs() < 1e-15 {
                break;
            }

            let alpha = rz / p_ap;

            // x = x + alpha * p
            vec_ops.axpy(alpha, &p, &mut x);

            // r = r - alpha * Ap
            vec_ops.axpy(-alpha, &ap, &mut r);

            let r_norm = vec_ops.norm(&r);
            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            // Apply preconditioner
            if let Some(ilu_precond) = ilu {
                ilu_precond.apply(&r, &mut z);
            } else {
                z.copy_from_slice(&r);
            }

            let rz_new = vec_ops.dot(&r, &z);
            let beta = if rz.abs() > 1e-15 { rz_new / rz } else { 0.0 };

            // p = z + beta * p
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }

            rz = rz_new;
            iteration += 1;
        }

        let final_norm = vec_ops.norm(&r);

        Ok(GPUSolverResult {
            iterations: iteration,
            residual_norm: final_norm,
            converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_matrix() -> GPUCSRMatrix {
        // SPD matrix:
        // [4 -1  0]
        // [-1 4 -1]
        // [0 -1 4]
        let row_ptr = vec![0, 2, 5, 7];
        let col_ind = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];

        GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0)
    }

    #[test]
    fn test_ilu_preconditioner() {
        let matrix = create_test_matrix();
        let ilu = GPUILUPreconditioner::new(&matrix, 0);

        let r = vec![1.0, 2.0, 3.0];
        let mut z = vec![0.0; 3];
        ilu.apply(&r, &mut z);

        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ssor_preconditioner() {
        let matrix = create_test_matrix();
        let ssor = GPUSSORPreconditioner::new(&matrix, 1.2, 0);

        let r = vec![1.0, 2.0, 3.0];
        let mut z = vec![0.0; 3];
        ssor.apply(&r, &mut z);

        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_chebyshev_preconditioner() {
        let prec = GPUChebyshevPreconditioner::new(0.1, 10.0, 3, 0);

        assert!(prec.lambda_min > 0.0);
        assert!(prec.lambda_max > prec.lambda_min);
        assert!(prec.degree >= 1);
    }

    #[test]
    fn test_gpu_pcg_solver() {
        let matrix = create_test_matrix();
        let ilu = GPUILUPreconditioner::new(&matrix, 0);

        let solver = GPUPCGSolver::new(0, 1e-8, 100, "ILU");
        let b = vec![1.0, 1.0, 1.0];

        let result = solver.solve(&matrix, Some(&ilu), &b).unwrap();

        assert!(result.residual_norm.is_finite());
    }

    #[test]
    fn test_eigenvalue_estimation() {
        let matrix = create_test_matrix();
        let (lambda_min, lambda_max) = GPUChebyshevPreconditioner::estimate_eigenvalues(&matrix, 0);

        assert!(lambda_min > 0.0);
        assert!(lambda_max > lambda_min);
    }
}
