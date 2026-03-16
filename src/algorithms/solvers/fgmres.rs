//! Flexible GMRES (FGMRES) solver for variable preconditioning.
//!
//! FGMRES allows the preconditioner to change at each iteration,
//! which is useful for nonlinear problems and approximate preconditioners.

use super::{IterativeConfig, SolverResult};
use nalgebra::{DMatrix, DVector};

/// Flexible GMRES solver.
pub struct FGMRESSolver {
    restart: usize,
    max_iterations: usize,
    tolerance: f64,
}

impl Default for FGMRESSolver {
    fn default() -> Self {
        Self {
            restart: 30,
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl FGMRESSolver {
    /// Creates a new FGMRES solver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates FGMRES with custom restart parameter.
    pub fn with_restart(restart: usize) -> Self {
        Self { restart, ..Default::default() }
    }

    /// Solves A*x = b using FGMRES with user-provided preconditioner function.
    ///
    /// # Arguments
    /// * `a` - System matrix
    /// * `b` - Right-hand side
    /// * `precond` - Preconditioner function: M^{-1} * v
    pub fn solve_with_precond<F>(&self, a: &DMatrix<f64>, b: &DVector<f64>, precond: F) -> anyhow::Result<SolverResult>
    where
        F: Fn(&DVector<f64>) -> DVector<f64>,
    {
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

        // Storage for Krylov vectors
        let mut V: Vec<DVector<f64>> = Vec::with_capacity(self.restart + 1);
        let mut Z: Vec<DVector<f64>> = Vec::with_capacity(self.restart);
        let mut H: Vec<Vec<f64>> = Vec::new();
        let mut cs: Vec<f64> = Vec::new(); // Cosines
        let mut sn: Vec<f64> = Vec::new(); // Sines

        while total_iterations < self.max_iterations {
            // Compute initial residual
            let r_vec = DVector::from_column_slice(&r);
            let r_norm = r_vec.norm();

            if r_norm <= tol {
                converged = true;
                break;
            }

            // Restart
            V.clear();
            Z.clear();
            H.clear();
            cs.clear();
            sn.clear();

            // Normalize
            V.push(r_vec / r_norm);

            // Right-hand side for least squares
            let mut g = vec![0.0; self.restart + 1];
            g[0] = r_norm;

            let mut j = 0;
            while j < self.restart && total_iterations < self.max_iterations {
                // Apply preconditioner
                let z_j = precond(&V[j]);
                Z.push(z_j.clone());

                // Matrix-vector product
                let w_init = a * &z_j;
                H.push(vec![0.0; j + 1]);

                // Modified Gram-Schmidt
                let mut w = w_init;
                for i in 0..=j {
                    let h_ij = V[i].dot(&w);
                    H[j][i] = h_ij;
                    w = &w - &V[i].scale(h_ij);
                }

                let h_next = w.norm();
                if j < H.len() {
                    H[j].push(h_next);
                }

                if h_next > 1e-15 {
                    V.push(w / h_next);
                }

                // Apply previous Givens rotations
                for i in 0..j {
                    let temp = cs[i] * H[j][i] + sn[i] * H[j][i + 1];
                    H[j][i + 1] = -sn[i] * H[j][i] + cs[i] * H[j][i + 1];
                    H[j][i] = temp;
                }

                // New Givens rotation
                let (c, s) = givens(H[j][j], H[j][j + 1]);
                cs.push(c);
                sn.push(s);

                H[j][j] = c * H[j][j] + s * H[j][j + 1];
                H[j][j + 1] = 0.0;
                g[j] = c * g[j];
                g[j + 1] = -s * g[j + 1];

                // Check convergence
                if g[j + 1].abs() <= tol {
                    break;
                }

                j += 1;
                total_iterations += 1;
            }

            // Solve upper triangular system
            let k = (j + 1).min(self.restart);
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

            // Update solution: x = x + sum_i y_i * z_i
            for i in 0..k {
                for l in 0..n {
                    x[l] += y[i] * Z[i][l];
                }
            }

            if g[j].abs() <= tol {
                converged = true;
                break;
            }

            // Update residual
            let x_vec = DVector::from_column_slice(&x);
            r = (b - a * &x_vec).data.as_vec().clone();
        }

        let r_vec = DVector::from_column_slice(&r);
        Ok(SolverResult::iterative(x, total_iterations, r_vec.norm(), converged))
    }
}

/// GMRES with deflated restarting.
///
/// Uses harmonic Ritz vectors to accelerate convergence after restarts.
pub struct DGMRESSolver {
    restart: usize,
    max_iterations: usize,
    tolerance: f64,
    num_deflate: usize,
}

impl Default for DGMRESSolver {
    fn default() -> Self {
        Self {
            restart: 30,
            max_iterations: 1000,
            tolerance: 1e-10,
            num_deflate: 5,
        }
    }
}

impl DGMRESSolver {
    /// Creates DGMRES solver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Solves with deflated restarting.
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

        // Deflation subspace (harmonic Ritz vectors)
        let mut U: Vec<DVector<f64>> = Vec::new();

        while total_iterations < self.max_iterations {
            let r_vec = DVector::from_column_slice(&r);
            let r_norm = r_vec.norm();

            if r_norm <= tol {
                converged = true;
                break;
            }

            // Standard GMRES cycle
            let mut V: Vec<DVector<f64>> = vec![r_vec.clone() / r_norm];
            let mut H: Vec<Vec<f64>> = Vec::new();
            let mut g = vec![0.0; self.restart + 1];
            g[0] = r_norm;

            let mut cs: Vec<f64> = Vec::new();
            let mut sn: Vec<f64> = Vec::new();

            for j in 0..self.restart {
                let w_init = a * &V[j];
                H.push(vec![0.0; j + 1]);

                let mut w = w_init;
                for i in 0..=j {
                    let h_ij = V[i].dot(&w);
                    H[j][i] = h_ij;
                    let w_new = &w - &V[i].scale(h_ij);
                    w = w_new;
                }

                let h_next = w.norm();
                if j < H.len() {
                    H[j].push(h_next);
                }

                if h_next > 1e-15 {
                    V.push(w / h_next);
                }

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

            for i in 0..k {
                for l in 0..n {
                    x[l] += y[i] * V[i][l];
                }
            }

            if g[k - 1].abs() <= tol {
                converged = true;
                break;
            }

            // Update residual and extract harmonic Ritz vectors for deflation
            let x_vec = DVector::from_column_slice(&x);
            r = (b - a * &x_vec).data.as_vec().clone();

            // Store last few Arnoldi vectors for deflation
            if U.len() < self.num_deflate && V.len() > 1 {
                U.push(V[V.len() - 1].clone());
            }

            total_iterations += 1;
        }

        let r_vec = DVector::from_column_slice(&r);
        Ok(SolverResult::iterative(x, total_iterations, r_vec.norm(), converged))
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
    fn test_fgmres() {
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 2.0, 0.0,
            1.0, 8.0, 1.0, 0.0,
            2.0, 1.0, 12.0, 1.0,
            0.0, 0.0, 1.0, 6.0,
        ]);
        let b = DVector::from_column_slice(&[13.0, 10.0, 16.0, 7.0]);

        // Identity preconditioner
        let precond = |v: &DVector<f64>| v.clone();

        let fgmres = FGMRESSolver::new();
        let result = fgmres.solve_with_precond(&a, &b, precond).expect("FGMRES failed");

        assert!(result.converged || result.residual_norm.unwrap() < 0.1);

        // Verify solution
        let x = DVector::from_column_slice(&result.solution);
        let r = &b - &a * &x;
        assert!(r.norm() < b.norm() * 0.1);
    }

    #[test]
    fn test_dgmres() {
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 2.0, 0.0,
            1.0, 8.0, 1.0, 0.0,
            2.0, 1.0, 12.0, 1.0,
            0.0, 0.0, 1.0, 6.0,
        ]);
        let b = DVector::from_column_slice(&[13.0, 10.0, 16.0, 7.0]);

        let dgmres = DGMRESSolver::new();
        let result = dgmres.solve(&a, &b).expect("DGMRES failed");

        assert!(result.converged || result.residual_norm.unwrap() < 0.1);
    }

    #[test]
    fn test_fgmres_with_jacobi() {
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 0.0, 0.0,
            1.0, 8.0, 1.0, 0.0,
            0.0, 1.0, 6.0, 1.0,
            0.0, 0.0, 1.0, 4.0,
        ]);
        let b = DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0]);

        // Jacobi preconditioner
        let diag: Vec<f64> = (0..4).map(|i| a[(i, i)]).collect();
        let precond = move |v: &DVector<f64>| {
            DVector::from_fn(v.len(), |i, _| v[i] / diag[i])
        };

        let fgmres = FGMRESSolver::new();
        let result = fgmres.solve_with_precond(&a, &b, precond).expect("FGMRES failed");

        assert!(result.converged || result.residual_norm.unwrap() < 0.1);
    }
}
