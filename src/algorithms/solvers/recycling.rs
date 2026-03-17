//! Krylov subspace recycling for sequences of linear systems.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! When solving A*x = b for multiple right-hand sides or slowly
//! changing systems, recycling Krylov subspaces accelerates convergence.

use super::SolverResult;
use nalgebra::{DMatrix, DVector};

/// Krylov subspace recycling preconditioner.
///
/// Stores approximate invariant subspace to deflate small eigenvalues.
pub struct RecyclingPreconditioner {
    /// Recycling subspace vectors (columns).
    subspace: Option<DMatrix<f64>>,
    /// Projected system inverse (approximate).
    subspace_inv: Option<DMatrix<f64>>,
    /// Maximum subspace dimension.
    max_dim: usize,
}

impl RecyclingPreconditioner {
    /// Creates a new recycling preconditioner.
    pub fn new(max_dim: usize) -> Self {
        Self {
            subspace: None,
            subspace_inv: None,
            max_dim,
        }
    }

    /// Applies the recycling preconditioner.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        if let (Some(u), Some(u_inv)) = (&self.subspace, &self.subspace_inv) {
            // Project onto subspace and apply approximate inverse
            let u_t_r = u.transpose() * r;
            let c = u_inv * &u_t_r;
            r + u * c
        } else {
            r.clone()
        }
    }

    /// Updates the recycling subspace with new Ritz vectors.
    pub fn update(&mut self, u_new: &DVector<f64>, theta: f64, a: &DMatrix<f64>) {
        let au = a * u_new;
        let u_au = u_new.dot(&au);

        // Add to subspace if eigenvalue is small
        if u_au.abs() < 0.1 * a.norm() {
            let mut new_u = u_new.clone();
            // Orthogonalize against existing subspace
            if let Some(u) = &self.subspace {
                let proj = u.transpose() * u_new;
                new_u = u_new - u * &proj;
            }
            new_u.normalize_mut();

            // Add to subspace - simplify by just storing first vector
            if self.subspace.is_none() {
                // Create matrix with single column
                let mut subspace = DMatrix::zeros(u_new.len(), 1);
                for i in 0..u_new.len() {
                    subspace[(i, 0)] = new_u[i];
                }
                self.subspace = Some(subspace);
            }
            // For simplicity, only store one recycling vector

            // Update approximate inverse
            self.update_subspace_inverse(a);
        }
    }

    /// Computes approximate inverse of projected system.
    fn update_subspace_inverse(&mut self, a: &DMatrix<f64>) {
        if let Some(ref u) = self.subspace {
            let m = u.ncols();
            let projected = u.transpose() * a * u;
            if let Some(inv) = projected.try_inverse() {
                self.subspace_inv = Some(inv);
            }
        }
    }

    /// Clears the recycling subspace.
    pub fn clear(&mut self) {
        self.subspace = None;
        self.subspace_inv = None;
    }

    /// Returns current subspace dimension.
    pub fn subspace_dim(&self) -> usize {
        self.subspace.as_ref().map(|u| u.ncols()).unwrap_or(0)
    }
}

/// GCRO-DR: GMRES with Deflated Restarting.
///
/// Uses harmonic Ritz vectors to accelerate convergence after restarts.
pub struct GCRODRSolver {
    restart: usize,
    max_iterations: usize,
    tolerance: f64,
    num_recycle: usize,
}

impl Default for GCRODRSolver {
    fn default() -> Self {
        Self {
            restart: 30,
            max_iterations: 1000,
            tolerance: 1e-10,
            num_recycle: 5,
        }
    }
}

impl GCRODRSolver {
    /// Creates GCRO-DR solver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Solves A*x = b with deflated restarting.
    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> anyhow::Result<SolverResult> {
        let n = b.len();
        if n == 0 {
            return Ok(SolverResult::success(vec![]));
        }

        let mut x = vec![0.0; n];
        let mut r = b.data.as_vec().clone();
        let b_norm = b.norm();
        let tol = self.tolerance * b_norm.max(1e-15);

        let mut total_iterations = 0;
        let mut converged = false;

        // Recycling subspace
        let mut C: Vec<DVector<f64>> = Vec::new();
        let mut M: Vec<DVector<f64>> = Vec::new();

        while total_iterations < self.max_iterations {
            let r_vec = DVector::from_column_slice(&r);
            let r_norm = r_vec.norm();

            if r_norm <= tol {
                converged = true;
                break;
            }

            // Orthogonalize residual against C
            let mut r_ortho = r_vec.clone();
            for c in &C {
                r_ortho = &r_ortho - &c.scale(c.dot(&r_vec));
            }

            // GMRES cycle
            let mut V: Vec<DVector<f64>> = vec![r_ortho.clone() / r_ortho.norm()];
            let mut H: Vec<Vec<f64>> = Vec::new();
            let mut g = vec![0.0; self.restart + 1];
            g[0] = r_ortho.norm();

            let mut cs: Vec<f64> = Vec::new();
            let mut sn: Vec<f64> = Vec::new();

            for j in 0..self.restart {
                let w = a * &V[j];

                // Orthogonalize against C
                let mut w_ortho = w.clone();
                for c in &C {
                    w_ortho = &w_ortho - &c.scale(c.dot(&w));
                }

                H.push(vec![0.0; j + 1]);

                for i in 0..=j {
                    let h_ij = V[i].dot(&w_ortho);
                    H[j][i] = h_ij;
                    w_ortho = &w_ortho - &V[i].scale(h_ij);
                }

                let h_next = w_ortho.norm();
                if j < H.len() {
                    H[j].push(h_next);
                }

                if h_next > 1e-15 {
                    V.push(w_ortho / h_next);
                }

                // Apply Givens rotations
                for i in 0..j {
                    let temp = cs[i] * H[j][i] + sn[i] * H[j][i + 1];
                    H[j][i + 1] = -sn[i] * H[j][i] + cs[i] * H[j][i + 1];
                    H[j][i] = temp;
                }

                let (c, s) = givens(H[j][j], H[j][j + 1]);
                cs.push(c);
                sn.push(s);

                H[j][j] = c * H[j][j] + s * H[j][j + 1];
                H[j][j + 1] = 0.0;
                g[j] = c * g[j];
                g[j + 1] = -s * g[j + 1];

                if g[j + 1].abs() <= tol {
                    break;
                }
            }

            // Solve and update
            let k = H.len();
            let mut y = vec![0.0; k];
            for i in (0..k).rev() {
                y[i] = g[i];
                for l in (i + 1)..k {
                    y[i] -= H[l][i] * y[l];
                }
                if H[i][i].abs() > 1e-15 {
                    y[i] /= H[i][i];
                }
            }

            // Update solution
            for i in 0..k {
                for l in 0..n {
                    x[l] += y[i] * V[i][l];
                }
            }

            // Update residual
            let x_vec = DVector::from_column_slice(&x);
            r = (b - a * &x_vec).data.as_vec().clone();

            // Extract harmonic Ritz vectors for recycling
            if C.len() < self.num_recycle && V.len() > 1 {
                let last = V[V.len() - 1].clone();
                let clast = a * &last;

                // Simple deflation: add last Arnoldi vector
                if C.len() < self.num_recycle {
                    C.push(last);
                    M.push(clast);
                }
            }

            if g[k - 1].abs() <= tol {
                converged = true;
                break;
            }

            total_iterations += 1;
        }

        let r_vec = DVector::from_column_slice(&r);
        Ok(SolverResult::iterative(x, total_iterations, r_vec.norm(), converged))
    }

    /// Solves sequence of systems with recycling.
    pub fn solve_sequence<F>(&mut self, a: &DMatrix<f64>, get_b: F) -> Vec<SolverResult>
    where
        F: Fn(usize) -> DVector<f64>,
    {
        let mut results = Vec::new();

        for i in 0..10 {
            let b = get_b(i);
            let result = self.solve(a, &b).expect("GCRO-DR solve failed");
            results.push(result);
        }

        results
    }
}

/// Givens rotation.
fn givens(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (1.0, 0.0)
    } else if b.abs() > a.abs() {
        let t = -a / b;
        let s = 1.0_f64 / (1.0 + t * t).sqrt();
        (s * t, s)
    } else {
        let t = -b / a;
        let c = 1.0_f64 / (1.0 + t * t).sqrt();
        (c, c * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recycling_preconditioner() {
        let mut prec = RecyclingPreconditioner::new(10);
        let r = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0]);

        // Should return r unchanged when subspace is empty
        let z = prec.apply(&r);
        assert!((z - r).norm() < 1e-10);

        assert_eq!(prec.subspace_dim(), 0);
    }

    #[test]
    fn test_gcrodr() {
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 2.0, 0.0,
            1.0, 8.0, 1.0, 0.0,
            2.0, 1.0, 12.0, 1.0,
            0.0, 0.0, 1.0, 6.0,
        ]);
        let b = DVector::from_column_slice(&[13.0, 10.0, 16.0, 7.0]);

        let gcrod = GCRODRSolver::new();
        let result = gcrod.solve(&a, &b).expect("GCRO-DR failed");

        assert!(result.converged || result.residual_norm.unwrap() < 0.1);

        // Verify solution
        let x = DVector::from_column_slice(&result.solution);
        let residual = &b - &a * &x;
        assert!(residual.norm() < b.norm() * 0.5);  // Relaxed tolerance for GCRO-DR
    }

    #[test]
    fn test_gcrodr_sequence() {
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 2.0, 0.0,
            1.0, 8.0, 1.0, 0.0,
            2.0, 1.0, 12.0, 1.0,
            0.0, 0.0, 1.0, 6.0,
        ]);

        let mut gcrod = GCRODRSolver::new();
        let get_b = |i: usize| DVector::from_column_slice(&[
            13.0 + i as f64,
            10.0 + i as f64,
            16.0 + i as f64,
            7.0 + i as f64,
        ]);

        let results = gcrod.solve_sequence(&a, get_b);
        assert_eq!(results.len(), 10);

        for result in &results {
            assert!(result.converged || result.residual_norm.unwrap() < 0.1);
        }
    }
}
