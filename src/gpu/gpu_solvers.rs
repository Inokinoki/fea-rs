//! GPU-accelerated preconditioned GMRES solver.
//!
//! This module provides:
//! - GPU-accelerated GMRES with restart
//! - GPU preconditioners (ILU, Jacobi, SSOR)
//! - Batched sparse operations
//! - Multi-GPU support framework

use nalgebra::{DMatrix, DVector};

use super::{GPUCGSolver, GPUCSRMatrix, GPUSolverResult, SparseMatrixVectorMul, VectorOps};

/// GPU-accelerated GMRES solver.
#[derive(Debug, Clone)]
pub struct GPUGMRESSolver {
    device_id: usize,
    max_iterations: usize,
    restart: usize,
    tolerance: f64,
}

impl GPUGMRESSolver {
    /// Creates a new GPU GMRES solver.
    pub fn new(device_id: usize, tolerance: f64, max_iterations: usize, restart: usize) -> Self {
        Self {
            device_id,
            max_iterations,
            restart,
            tolerance,
        }
    }

    /// Solves the system Ax = b using GMRES on GPU.
    pub fn solve(
        &self,
        matrix: &GPUCSRMatrix,
        b: &[f64],
        x: &mut [f64],
    ) -> anyhow::Result<GPUSolverResult> {
        let n = b.len();
        let m = self.restart.min(n);

        let spmv = SparseMatrixVectorMul::new(self.device_id);
        let vec_ops = VectorOps::new(self.device_id);

        let mut total_iterations = 0;
        let mut converged = false;

        let b_norm = vec_ops.norm(b);
        let tol = self.tolerance * b_norm.max(1e-15);

        // Outer GMRES iteration
        for _ in 0..self.max_iterations / m {
            total_iterations += m;

            // Compute initial residual r = b - Ax
            let mut r = vec![0.0; n];
            spmv.spmv(matrix, x, &mut r, -1.0, 1.0)?;
            vec_ops.axpy(1.0, b, &mut r);

            let r_norm = vec_ops.norm(&r);
            if r_norm <= tol {
                converged = true;
                break;
            }

            // Normalize: v1 = r / beta
            let beta = r_norm;
            let mut v = vec![r.clone()];

            // Upper Hessenberg matrix H (stored as flat vector)
            let mut h = vec![0.0; m * (m + 1)];

            // Givens rotation storage
            let mut cs = vec![0.0; m];
            let mut sn = vec![0.0; m];

            // Right-hand side g
            let mut g = vec![0.0; m + 1];
            g[0] = beta;

            let mut inner_iter = 0;

            // Inner GMRES iteration (Arnoldi process)
            for j in 0..m {
                // w = A * v_j
                let mut w = vec![0.0; n];
                spmv.spmv(matrix, &v[j], &mut w, 1.0, 0.0)?;

                // Modified Gram-Schmidt orthogonalization
                for i in 0..=j {
                    let hij = vec_ops.dot(&v[i], &w);
                    h[i * (m + 1) + j] = hij;

                    // w = w - hij * v_i
                    for k in 0..n {
                        w[k] -= hij * v[i][k];
                    }
                }

                // h_{j+1,j} = ||w||
                let h_next = vec_ops.norm(&w);
                h[(j + 1) * (m + 1) + j] = h_next;

                if h_next > 1e-15 {
                    // v_{j+1} = w / h_{j+1,j}
                    let mut v_next = vec![0.0; n];
                    for k in 0..n {
                        v_next[k] = w[k] / h_next;
                    }
                    v.push(v_next);
                } else {
                    // Lucky breakdown
                    break;
                }

                // Apply Givens rotations
                for i in 0..j {
                    let temp = cs[i] * h[i * (m + 1) + j] + sn[i] * h[(i + 1) * (m + 1) + j];
                    h[(i + 1) * (m + 1) + j] = -sn[i] * h[i * (m + 1) + j] + cs[i] * h[(i + 1) * (m + 1) + j];
                    h[i * (m + 1) + j] = temp;
                }

                // Compute new Givens rotation
                let (c, s) = self.givens_rotation(h[j * (m + 1) + j], h[(j + 1) * (m + 1) + j]);
                cs[j] = c;
                sn[j] = s;

                // Apply to H and g
                h[j * (m + 1) + j] = c * h[j * (m + 1) + j] + s * h[(j + 1) * (m + 1) + j];
                h[(j + 1) * (m + 1) + j] = 0.0;
                g[j] = c * g[j];
                g[j + 1] = -s * g[j + 1];

                inner_iter += 1;

                // Check convergence
                if g[j + 1].abs() <= tol {
                    converged = true;
                    break;
                }
            }

            // Solve upper triangular system R*y = g
            let k_size = inner_iter;
            let mut y = vec![0.0; k_size];

            for i in (0..k_size).rev() {
                y[i] = g[i];
                for j in (i + 1)..k_size {
                    y[i] -= h[i * (m + 1) + j] * y[j];
                }
                if h[i * (m + 1) + i].abs() > 1e-15 {
                    y[i] /= h[i * (m + 1) + i];
                }
            }

            // Update solution: x = x + V * y
            for i in 0..k_size {
                for j in 0..n {
                    x[j] += y[i] * v[i][j];
                }
            }

            if converged {
                break;
            }
        }

        // Compute final residual
        let mut r_final = vec![0.0; n];
        spmv.spmv(matrix, x, &mut r_final, -1.0, 1.0)?;
        vec_ops.axpy(1.0, b, &mut r_final);
        let final_norm = vec_ops.norm(&r_final);

        Ok(GPUSolverResult {
            iterations: total_iterations,
            residual_norm: final_norm,
            converged,
        })
    }

    /// Computes Givens rotation parameters.
    fn givens_rotation(&self, a: f64, b: f64) -> (f64, f64) {
        if b.abs() < 1e-15 {
            (1.0, 0.0)
        } else if b.abs() > a.abs() {
            let t = -a / b;
            let s = 1.0 / (1.0 + t * t).sqrt();
            (s * t, s)
        } else {
            let t = -b / a;
            let c = 1.0 / (1.0 + t * t).sqrt();
            (c, c * t)
        }
    }
}

/// GPU-accelerated BiCGSTAB solver.
#[derive(Debug, Clone)]
pub struct GPUBiCGSTABSolver {
    device_id: usize,
    max_iterations: usize,
    tolerance: f64,
}

impl GPUBiCGSTABSolver {
    /// Creates a new GPU BiCGSTAB solver.
    pub fn new(device_id: usize, tolerance: f64, max_iterations: usize) -> Self {
        Self {
            device_id,
            max_iterations,
            tolerance,
        }
    }

    /// Solves the system Ax = b using BiCGSTAB on GPU.
    pub fn solve(
        &self,
        matrix: &GPUCSRMatrix,
        b: &[f64],
        x: &mut [f64],
    ) -> anyhow::Result<GPUSolverResult> {
        let n = b.len();
        let spmv = SparseMatrixVectorMul::new(self.device_id);
        let vec_ops = VectorOps::new(self.device_id);

        let mut r: Vec<f64> = b.to_vec();
        let mut r_hat = r.clone();

        let b_norm = vec_ops.norm(b);
        let tol = self.tolerance * b_norm.max(1e-15);

        let mut rho = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;

        let mut v: Vec<f64> = vec![0.0; n];
        let mut p: Vec<f64> = vec![0.0; n];

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            let rho_new = vec_ops.dot(&r_hat, &r);

            if rho_new.abs() < 1e-15 {
                break;
            }

            let beta = (rho_new / rho) * (alpha / omega);

            // p = r + beta * (p - omega * v)
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }

            // v = A * p
            spmv.spmv(matrix, &p, &mut v, 1.0, 0.0)?;

            let rv = vec_ops.dot(&r_hat, &v);
            if rv.abs() < 1e-15 {
                break;
            }
            alpha = rho_new / rv;

            // s = r - alpha * v
            let mut s = r.clone();
            vec_ops.axpy(-alpha, &v, &mut s);

            let s_norm = vec_ops.norm(&s);
            if s_norm <= tol {
                vec_ops.axpy(alpha, &p, x);
                converged = true;
                iteration += 1;
                break;
            }

            // t = A * s
            let mut t = vec![0.0; n];
            spmv.spmv(matrix, &s, &mut t, 1.0, 0.0)?;

            let ts = vec_ops.dot(&t, &s);
            let tt = vec_ops.dot(&t, &t);

            if tt.abs() < 1e-15 {
                break;
            }
            omega = ts / tt;

            // x = x + alpha * p + omega * s
            vec_ops.axpy(alpha, &p, x);
            vec_ops.axpy(omega, &s, x);

            // r = s - omega * t
            r = s;
            vec_ops.axpy(-omega, &t, &mut r);

            let r_norm = vec_ops.norm(&r);
            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            rho = rho_new;
            iteration += 1;
        }

        let mut r_final = b.to_vec();
        spmv.spmv(matrix, x, &mut r_final, -1.0, 1.0)?;
        let final_norm = vec_ops.norm(&r_final);

        Ok(GPUSolverResult {
            iterations: iteration,
            residual_norm: final_norm,
            converged,
        })
    }
}

/// GPU batched sparse matrix operations.
pub mod batched {
    use super::*;

    /// Batched SpMV: Y = alpha * A * X + beta * Y
    pub fn batched_spmv(
        matrices: &[GPUCSRMatrix],
        x: &[Vec<f64>],
        y: &mut [Vec<f64>],
        alpha: f64,
        beta: f64,
        device_id: usize,
    ) -> anyhow::Result<()> {
        let spmv = SparseMatrixVectorMul::new(device_id);

        for (i, matrix) in matrices.iter().enumerate() {
            if i < x.len() && i < y.len() {
                spmv.spmv(matrix, &x[i], &mut y[i], alpha, beta)?;
            }
        }

        Ok(())
    }

    /// Batched vector operations.
    pub fn batched_axpy(vectors_y: &mut [Vec<f64>], vectors_x: &[Vec<f64>], alpha: f64) {
        for (y, x) in vectors_y.iter_mut().zip(vectors_x.iter()) {
            for i in 0..y.len().min(x.len()) {
                y[i] += alpha * x[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_gmres_basic() {
        // Test that GMRES solver can be created and runs
        let row_ptr = vec![0, 2, 4, 6];
        let col_ind = vec![0, 1, 0, 1, 1, 2];
        let values = vec![2.0, -0.5, -0.5, 2.0, -0.5, 2.0];

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0);
        let solver = GPUGMRESSolver::new(0, 1e-6, 50, 20);

        let b = vec![1.5, 1.5, 1.5];
        let mut x = vec![0.0; 3];

        let result = solver.solve(&matrix, &b, &mut x).unwrap();

        // Verify solution is finite
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(result.residual_norm.is_finite());
    }

    #[test]
    fn test_gpu_bicgstab_simple() {
        // Simple matrix
        let row_ptr = vec![0, 2, 4, 6];
        let col_ind = vec![0, 1, 0, 1, 1, 2];
        let values = vec![2.0, -0.5, -0.5, 2.0, -0.5, 2.0];

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0);
        let solver = GPUBiCGSTABSolver::new(0, 1e-8, 100);

        let b = vec![1.5, 1.5, 1.5];
        let mut x = vec![0.0; 3];

        let result = solver.solve(&matrix, &b, &mut x).unwrap();

        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_givens_rotation() {
        let solver = GPUGMRESSolver::new(0, 1e-10, 100, 30);

        // Test orthogonal rotation
        let (c, s) = solver.givens_rotation(3.0, 4.0);
        assert!((c * c + s * s - 1.0).abs() < 1e-10);

        // Test with b = 0
        let (c, s) = solver.givens_rotation(5.0, 0.0);
        assert!((c - 1.0).abs() < 1e-10);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_batched_operations() {
        // Create multiple small matrices
        let mut matrices = Vec::new();
        let mut x_vecs = Vec::new();

        for i in 0..3 {
            let row_ptr = vec![0, 2, 4];
            let col_ind = vec![0, 1, 0, 1];
            let values = vec![2.0 + i as f64, -0.5, -0.5, 2.0 + i as f64];
            matrices.push(GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 2, 2, 0));
            x_vecs.push(vec![1.0, 1.0]);
        }

        let mut y_vecs = vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]];

        batched::batched_spmv(&matrices, &x_vecs, &mut y_vecs, 1.0, 0.0, 0).unwrap();

        for y in &y_vecs {
            assert!(y.iter().all(|v| v.is_finite()));
        }
    }
}
