//! Solver traits and implementations.
//!
//! This module provides:
//! - The Solver trait for linear system solvers
//! - Direct solvers (LU, Cholesky)
//! - Iterative solvers (CG, PCG, GMRES)

use nalgebra::{DMatrix, DVector};

/// Result of a linear solver operation.
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Solution vector.
    pub solution: Vec<f64>,
    /// Number of iterations (for iterative solvers).
    pub iterations: Option<usize>,
    /// Final residual norm (for iterative solvers).
    pub residual_norm: Option<f64>,
    /// Whether convergence was achieved.
    pub converged: bool,
}

impl SolverResult {
    /// Creates a successful result from a direct solve.
    pub fn success(solution: Vec<f64>) -> Self {
        Self {
            solution,
            iterations: None,
            residual_norm: None,
            converged: true,
        }
    }

    /// Creates a result from an iterative solve.
    pub fn iterative(solution: Vec<f64>, iterations: usize, residual: f64, converged: bool) -> Self {
        Self {
            solution,
            iterations: Some(iterations),
            residual_norm: Some(residual),
            converged,
        }
    }
}

/// Configuration for direct solvers.
#[derive(Debug, Clone, Copy, Default)]
pub struct DirectConfig {
    /// Use Cholesky decomposition (assumes SPD matrix).
    /// If false, uses LU decomposition.
    pub use_cholesky: bool,
}

/// Configuration for iterative solvers.
#[derive(Debug, Clone)]
pub struct IterativeConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance (relative residual norm).
    pub tolerance: f64,
    /// Preconditioner to use.
    pub preconditioner: Preconditioner,
}

impl Default for IterativeConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            preconditioner: Preconditioner::Jacobi,
        }
    }
}

/// Preconditioner types for iterative solvers.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Preconditioner {
    /// No preconditioning.
    None,
    /// Jacobi (diagonal) preconditioner.
    #[default]
    Jacobi,
    /// Incomplete Cholesky preconditioner.
    IncompleteCholesky,
}

/// The Solver trait for linear system solvers.
///
/// Solves the linear system K * u = f.
pub trait Solver {
    /// Configuration type for this solver.
    type Config: Clone + std::fmt::Debug;
    /// Result type for this solver.

    /// Solves the linear system K * u = f.
    fn solve(&self, k: &DMatrix<f64>, f: &DVector<f64>, config: &Self::Config) -> anyhow::Result<SolverResult>;
}

/// Direct solver using LU or Cholesky decomposition.
#[derive(Debug, Clone, Default)]
pub struct DirectSolver;

impl DirectSolver {
    /// Creates a new direct solver.
    pub fn new() -> Self {
        Self
    }

    /// Creates a direct solver with Cholesky decomposition.
    pub fn cholesky() -> Self {
        Self
    }
}

impl Solver for DirectSolver {
    type Config = DirectConfig;

    fn solve(&self, k: &DMatrix<f64>, f: &DVector<f64>, config: &Self::Config) -> anyhow::Result<SolverResult> {
        let solution = if config.use_cholesky {
            // Try Cholesky (for SPD matrices)
            let chol = k.clone().cholesky()
                .ok_or_else(|| anyhow::anyhow!("Matrix is not SPD, Cholesky decomposition failed"))?;
            chol.solve(f)
        } else {
            // Use LU decomposition (general matrices)
            k.clone()
                .lu()
                .solve(f)
                .ok_or_else(|| anyhow::anyhow!("LU solve failed: matrix is singular"))?
        };

        Ok(SolverResult::success(solution.data.as_vec().clone()))
    }
}

/// Conjugate Gradient solver for SPD systems.
#[derive(Debug, Clone)]
pub struct CGSolver {
    config: IterativeConfig,
}

impl CGSolver {
    /// Creates a new CG solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(IterativeConfig::default())
    }

    /// Creates a new CG solver with custom configuration.
    pub fn with_config(config: IterativeConfig) -> Self {
        Self { config }
    }

    /// Creates a CG solver with specified tolerance.
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            config: IterativeConfig {
                tolerance,
                ..Default::default()
            },
        }
    }
}

impl Default for CGSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for CGSolver {
    type Config = IterativeConfig;

    fn solve(&self, k: &DMatrix<f64>, f: &DVector<f64>, config: &Self::Config) -> anyhow::Result<SolverResult> {
        let n = f.len();
        if n == 0 {
            return Ok(SolverResult::success(vec![]));
        }

        // Initial guess: zero
        let mut x = vec![0.0f64; n];
        let mut r = f.clone();
        let mut p = r.clone();

        // Apply Jacobi preconditioner if enabled
        let mut z = if config.preconditioner == Preconditioner::Jacobi {
            let mut z_vec = DVector::zeros(n);
            for i in 0..n {
                let k_ii = k[(i, i)];
                z_vec[i] = if k_ii.abs() > 1e-15 { r[i] / k_ii } else { r[i] };
            }
            z_vec
        } else {
            r.clone()
        };

        let mut rz = r.dot(&z);
        let b_norm = f.norm();
        let tol = config.tolerance * b_norm.max(1e-15);

        let mut iteration = 0;
        let mut converged = false;

        while iteration < config.max_iterations {
            let kp = k * &p;
            let p_kp = p.iter().zip(kp.iter()).map(|(pi, kpi)| pi * kpi).sum::<f64>();

            if p_kp.abs() < 1e-15 {
                break;
            }

            let alpha = rz / p_kp;

            // x = x + alpha * p
            for i in 0..n {
                x[i] += alpha * p[i];
            }

            // r = r - alpha * K * p
            for i in 0..n {
                r[i] -= alpha * kp[i];
            }

            let r_norm = r.norm();
            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            // Update preconditioner
            if config.preconditioner == Preconditioner::Jacobi {
                z = DVector::from_fn(n, |i, _| {
                    let k_ii = k[(i, i)];
                    if k_ii.abs() > 1e-15 { r[i] / k_ii } else { r[i] }
                });
            } else {
                z.copy_from_slice(r.as_slice());
            }

            let rz_new = r.dot(&z);
            let beta = if rz.abs() > 1e-15 { rz_new / rz } else { 0.0 };

            // p = z + beta * p
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }

            rz = rz_new;
            iteration += 1;
        }

        Ok(SolverResult::iterative(x, iteration, r.norm(), converged))
    }
}

/// Preconditioned Conjugate Gradient solver.
#[derive(Debug, Clone)]
pub struct PCGSolver {
    config: IterativeConfig,
}

impl PCGSolver {
    /// Creates a new PCG solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(IterativeConfig::default())
    }

    /// Creates a new PCG solver with custom configuration.
    pub fn with_config(config: IterativeConfig) -> Self {
        Self { config }
    }

    /// Creates a PCG solver with specified tolerance.
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            config: IterativeConfig {
                tolerance,
                preconditioner: Preconditioner::Jacobi,
                ..Default::default()
            },
        }
    }
}

impl Default for PCGSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for PCGSolver {
    type Config = IterativeConfig;

    fn solve(&self, k: &DMatrix<f64>, f: &DVector<f64>, config: &Self::Config) -> anyhow::Result<SolverResult> {
        // PCG is essentially CG with Jacobi preconditioning
        let cg = CGSolver::with_config(IterativeConfig {
            preconditioner: Preconditioner::Jacobi,
            ..config.clone()
        });
        cg.solve(k, f, config)
    }
}

/// GMRES solver for non-symmetric systems.
#[derive(Debug, Clone)]
pub struct GMRESSolver {
    max_iterations: usize,
    restart: usize,
    tolerance: f64,
}

impl GMRESSolver {
    /// Creates a new GMRES solver with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new GMRES solver with custom restart parameter.
    pub fn with_restart(restart: usize) -> Self {
        Self {
            restart,
            ..Default::default()
        }
    }
}

impl Default for GMRESSolver {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            restart: 30,
            tolerance: 1e-10,
        }
    }
}

impl Solver for GMRESSolver {
    type Config = IterativeConfig;

    fn solve(&self, k: &DMatrix<f64>, f: &DVector<f64>, config: &Self::Config) -> anyhow::Result<SolverResult> {
        let n = f.len();
        if n == 0 {
            return Ok(SolverResult::success(vec![]));
        }

        let mut x = vec![0.0; n];
        let b_norm = f.norm();
        let tol = config.tolerance * b_norm.max(1e-15);

        let mut total_iterations = 0;
        let mut converged = false;

        // Outer restart loop
        while total_iterations < config.max_iterations && !converged {
            // Compute initial residual
            let ax = k * &DVector::from_column_slice(&x);
            let r = f - ax;
            let r_norm = r.norm();

            if r_norm <= tol {
                converged = true;
                break;
            }

            // Normalize initial residual
            let beta = r_norm;
            let v1 = r / beta;

            // Arnoldi process: build Krylov subspace
            let m = self.restart.min(config.max_iterations - total_iterations);
            let mut V: Vec<DVector<f64>> = Vec::with_capacity(m + 1);
            V.push(v1);

            // Upper Hessenberg matrix H
            let mut H = vec![vec![0.0f64; m]; m + 1];

            // Givens rotation storage
            let mut cs = vec![0.0f64; m];
            let mut sn = vec![0.0f64; m];

            // Right-hand side of least squares problem
            let mut g = vec![0.0f64; m + 1];
            g[0] = beta;

            let mut inner_iter = 0;

            // Inner GMRES iteration
            for j in 0..m {
                // Arnoldi: w = A * v_j
                let w = k * &V[j];

                // Modified Gram-Schmidt orthogonalization
                let mut w_ortho = w.data.as_vec().clone();
                for i in 0..=j {
                    let h_ij = V[i].dot(&w);
                    H[i][j] = h_ij;
                    for k_idx in 0..n {
                        w_ortho[k_idx] -= h_ij * V[i][k_idx];
                    }
                }

                let h_next = w_ortho.iter().map(|v| v * v).sum::<f64>().sqrt();
                H[j + 1][j] = h_next;

                if h_next > 1e-15 {
                    let v_next = DVector::from_column_slice(
                        &w_ortho.iter().map(|v| v / h_next).collect::<Vec<_>>()
                    );
                    V.push(v_next);
                }

                // Apply Givens rotations
                for i in 0..j {
                    let temp = cs[i] * H[i][j] + sn[i] * H[i + 1][j];
                    H[i + 1][j] = -sn[i] * H[i][j] + cs[i] * H[i + 1][j];
                    H[i][j] = temp;
                }

                // Compute new Givens rotation
                let (c, s) = givens(H[j][j], H[j + 1][j]);
                cs[j] = c;
                sn[j] = s;

                // Apply to H and g
                H[j][j] = c * H[j][j] + s * H[j + 1][j];
                H[j + 1][j] = 0.0;
                g[j] = c * g[j];
                g[j + 1] = -s * g[j + 1];

                inner_iter += 1;
                total_iterations += 1;

                // Check convergence
                if g[j + 1].abs() <= tol {
                    converged = true;
                    break;
                }
            }

            // Solve upper triangular system
            let k_size = inner_iter;
            let mut y = vec![0.0f64; k_size];
            for i in (0..k_size).rev() {
                y[i] = g[i];
                for j in (i + 1)..k_size {
                    y[i] -= H[i][j] * y[j];
                }
                if H[i][i].abs() > 1e-15 {
                    y[i] /= H[i][i];
                }
            }

            // Update solution: x = x + V * y
            for i in 0..k_size {
                for j in 0..n {
                    x[j] += y[i] * V[i][j];
                }
            }
        }

        // Compute final residual
        let x_vec = DVector::from_column_slice(&x);
        let r_final = f - k * &x_vec;
        Ok(SolverResult::iterative(x, total_iterations, r_final.norm(), converged))
    }
}

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

/// Gauss-Seidel iterative solver (for comparison).
#[derive(Debug, Clone, Default)]
pub struct GaussSeidelSolver {
    max_iterations: usize,
    tolerance: f64,
    omega: f64, // Relaxation factor (1.0 = standard GS, > 1.0 = SOR)
}

impl GaussSeidelSolver {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            omega: 1.0,
        }
    }

    /// Creates a Gauss-Seidel solver with SOR (Successive Over-Relaxation).
    pub fn with_sor(omega: f64) -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            omega: omega.clamp(0.1, 1.99),
        }
    }
}

impl Solver for GaussSeidelSolver {
    type Config = IterativeConfig;

    fn solve(&self, k: &DMatrix<f64>, f: &DVector<f64>, config: &Self::Config) -> anyhow::Result<SolverResult> {
        let n = f.len();
        if n == 0 {
            return Ok(SolverResult::success(vec![]));
        }

        let mut u = vec![0.0f64; n];
        let b_norm = f.norm();
        let tol = config.tolerance * b_norm.max(1e-15);
        let omega = self.omega;

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            let mut max_change: f64 = 0.0;

            for i in 0..n {
                let mut sum = f[i];
                for j in 0..n {
                    if i != j {
                        sum -= k[(i, j)] * u[j];
                    }
                }

                let diag = k[(i, i)];
                if diag.abs() > 1e-15 {
                    let u_new = omega * (sum / diag) + (1.0 - omega) * u[i];
                    max_change = max_change.max((u_new - u[i]).abs());
                    u[i] = u_new;
                }
            }

            iteration += 1;

            if max_change < tol {
                converged = true;
                break;
            }
        }

        // Compute final residual
        let u_vec = DVector::from_column_slice(&u);
        let r = f - k * u_vec;
        Ok(SolverResult::iterative(u, iteration, r.norm(), converged))
    }
}
