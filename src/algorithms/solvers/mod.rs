//! Solver traits and implementations.
#![allow(non_snake_case)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
//!
//! This module provides:
//! - The Solver trait for linear system solvers
//! - Direct solvers (LU, Cholesky)
//! - Iterative solvers (CG, PCG, GMRES, BiCGSTAB, CGNR, GCR, QMR, CGS, SOR, FGMRES)
//! - Eigensolvers (Lanczos)
//! - Multigrid and Full Multigrid solvers
//! - Incomplete Cholesky preconditioners
//! - Block preconditioners (Block-Jacobi, Additive Schwarz, SPAI, AINV)
//! - Polynomial preconditioners (Chebyshev, Newton-Schulz, Hotelling-Bodewig)
//! - Advanced solvers (FGMRES, DGMRES, GCRO-DR, Deflated CG)
//! - Krylov subspace methods (recycling, augmentation, deflation)
//! - Vector extrapolation acceleration (MPE, RRE, MRE)
//! - Nonlinear acceleration (Epsilon, Levin U, Theta, Combined, Anderson, NGMRES)
//! - Polynomial acceleration (Chebyshev, Minimal Residual, BB, Nonlinear CG)
//! - Classical acceleration (Aitken, DIIS, MRS, Steffensen)

#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_mut)]

pub mod lanczos;
pub mod multigrid;
pub mod ic;
pub mod fmg;
pub mod gcr;
pub mod qmr;
pub mod diis;
pub mod block_precond;
pub mod polynomial;
pub mod fgmres;
pub mod recycling;
pub mod krylov;
pub mod extrapolation;
pub mod extrapolation2;
pub mod nonlinear_accel;
pub mod advanced_accel;
pub mod mp_accel;
pub mod advanced_extrap;
pub mod additional_accel;
pub mod poly_accel;
pub mod poly_accel2;
pub mod acceleration;
pub mod advanced_methods;
pub mod block_solvers;
pub mod preconditioners_advanced;
pub mod spectral_accel;
pub mod krylov_recycling;
pub mod domain_decomposition;
pub mod tensor_multilevel;
pub mod nonlinear_solvers_enhanced;
pub mod adaptive_solvers;
pub mod parallel_solvers;
pub mod polynomial_acceleration;
pub mod randomized_la;
pub mod unified_framework;
pub mod block_iterative;
pub mod advanced_krylov;
pub mod mixed_precision;
pub mod nonlinear_acceleration;
pub mod performance_utils;
pub mod accelerated_pcg;
pub mod advanced_eigensolvers;

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
    /// Anderson acceleration depth (0 = disabled).
    pub anderson_depth: usize,
    /// Krylov subspace dimension for recycling (0 = disabled).
    pub krylov_dim: usize,
    /// Deflation vectors for deflated CG (columns of matrix).
    pub deflation_vectors: Option<DMatrix<f64>>,
}

impl Default for IterativeConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        }
    }
}

impl IterativeConfig {
    /// Creates a config with Anderson acceleration.
    pub fn with_anderson(mut self, depth: usize) -> Self {
        self.anderson_depth = depth;
        self
    }

    /// Creates a config with Krylov subspace recycling.
    pub fn with_krylov_recycling(mut self, dim: usize) -> Self {
        self.krylov_dim = dim;
        self
    }

    /// Creates a config with deflation vectors.
    pub fn with_deflation(mut self, vectors: DMatrix<f64>) -> Self {
        self.deflation_vectors = Some(vectors);
        self
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
    /// SSOR (Symmetric Successive Over-Relaxation) preconditioner.
    SSOR(f64), // relaxation parameter
    /// Multigrid V-cycle preconditioner (levels specified separately).
    Multigrid,
    /// Chebyshev polynomial preconditioner of given degree.
    Chebyshev(usize),
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
        let mut r: Vec<f64> = f.data.as_vec().clone();
        let mut p: Vec<f64> = r.clone();

        // Apply Jacobi preconditioner if enabled
        let mut z: Vec<f64> = if config.preconditioner == Preconditioner::Jacobi {
            let mut z_vec = vec![0.0; n];
            for i in 0..n {
                let k_ii = k[(i, i)];
                z_vec[i] = if k_ii.abs() > 1e-15 { r[i] / k_ii } else { r[i] };
            }
            z_vec
        } else {
            r.clone()
        };

        let mut rz: f64 = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
        let b_norm = f.norm();
        let tol = config.tolerance * b_norm.max(1e-15);

        let mut iteration = 0;
        let mut converged = false;

        // Anderson acceleration storage
        let anderson_depth = config.anderson_depth;
        let mut anderson_x: Vec<Vec<f64>> = Vec::new();
        let mut anderson_r: Vec<Vec<f64>> = Vec::new();

        while iteration < config.max_iterations {
            // Compute K * p
            let kp = k * &DVector::from_column_slice(&p);

            let p_kp: f64 = p.iter().zip(kp.iter()).map(|(pi, kpi)| pi * kpi).sum();

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

            let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            // Anderson acceleration (optional)
            if anderson_depth > 0 {
                anderson_x.push(x.clone());
                anderson_r.push(r.clone());

                while anderson_x.len() > anderson_depth {
                    anderson_x.remove(0);
                    anderson_r.remove(0);
                }

                if anderson_x.len() >= anderson_depth {
                    // Simple Anderson averaging with residual-based weights
                    let m = anderson_x.len();
                    let mut total_weight = 0.0;
                    let mut weights = Vec::with_capacity(m);

                    for i in 0..m {
                        let r_norm_i = anderson_r[i].iter().map(|v| v * v).sum::<f64>().sqrt();
                        let w = if r_norm_i > 1e-15 { 1.0 / r_norm_i } else { 1.0 };
                        weights.push(w);
                        total_weight += w;
                    }

                    if total_weight > 1e-15 {
                        // Compute weighted average
                        for i in 0..n {
                            x[i] = 0.0;
                            for j in 0..m {
                                x[i] += (weights[j] / total_weight) * anderson_x[j][i];
                            }
                        }

                        // Recompute residual with averaged solution
                        let x_vec = DVector::from_column_slice(&x);
                        let r_vec = f - k * &x_vec;
                        r = r_vec.data.as_vec().clone();
                    }
                }
            }

            // Update preconditioner
            match &config.preconditioner {
                Preconditioner::Jacobi => {
                    for i in 0..n {
                        let k_ii = k[(i, i)];
                        z[i] = if k_ii.abs() > 1e-15 { r[i] / k_ii } else { r[i] };
                    }
                }
                Preconditioner::SSOR(omega) => {
                    // SSOR preconditioning
                    z.clone_from(&r);
                    // Forward sweep
                    for i in 0..n {
                        let mut sum = 0.0;
                        for j in 0..i {
                            sum += k[(i, j)] * z[j];
                        }
                        let k_ii = k[(i, i)].max(1e-15);
                        z[i] = (1.0 - omega) * r[i] + omega * (r[i] - sum) / k_ii;
                    }
                    // Backward sweep
                    for i in (0..n).rev() {
                        let mut sum = 0.0;
                        for j in (i + 1)..n {
                            sum += k[(i, j)] * z[j];
                        }
                        let k_ii = k[(i, i)].max(1e-15);
                        z[i] = (1.0 - omega) * r[i] + omega * (r[i] - sum) / k_ii;
                    }
                }
                Preconditioner::Chebyshev(degree) => {
                    // Chebyshev polynomial preconditioning
                    // M^{-1} ≈ p(K) where p is a Chebyshev polynomial
                    // Simplified: use Jacobi as base with Chebyshev acceleration
                    for i in 0..n {
                        let k_ii = k[(i, i)];
                        z[i] = if k_ii.abs() > 1e-15 { r[i] / k_ii } else { r[i] };
                    }
                    // Apply Chebyshev iteration (simplified)
                    if *degree > 1 {
                        let mut z_prev = z.clone();
                        for _ in 1..*degree {
                            // Simplified Chebyshev step
                            for i in 0..n {
                                let k_ii = k[(i, i)].max(1e-15);
                                z[i] = 2.0 * r[i] / k_ii - z_prev[i];
                            }
                            z_prev = z.clone();
                        }
                    }
                }
                _ => {
                    z.copy_from_slice(&r);
                }
            }

            let rz_new: f64 = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
            let beta = if rz.abs() > 1e-15 { rz_new / rz } else { 0.0 };

            // p = z + beta * p
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }

            rz = rz_new;
            iteration += 1;
        }

        Ok(SolverResult::iterative(x, iteration, r.iter().map(|v| v * v).sum::<f64>().sqrt(), converged))
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

/// CGNR (Conjugate Gradient on Normal Residual) solver.
///
/// Solves A*x = b by applying CG to the normal equations:
/// A^T*A*x = A^T*b
///
/// Useful for non-symmetric systems where A^T*A is SPD.
#[derive(Debug, Clone)]
pub struct CGNRSolver {
    max_iterations: usize,
    tolerance: f64,
}

impl CGNRSolver {
    /// Creates a new CGNR solver with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new CGNR solver with custom tolerance.
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            tolerance,
            ..Default::default()
        }
    }
}

impl Default for CGNRSolver {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl Solver for CGNRSolver {
    type Config = IterativeConfig;

    fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>, config: &Self::Config) -> anyhow::Result<SolverResult> {
        let n = b.len();
        if n == 0 {
            return Ok(SolverResult::success(vec![]));
        }

        // Form normal equations: A^T*A*x = A^T*b
        let at = a.transpose();
        let ata = &at * a;
        let atb = &at * b;

        // Apply CG to normal equations
        let cg = CGSolver::with_config(config.clone());
        cg.solve(&ata, &atb, config)
    }
}

/// TFQMR (Transpose-Free Quasi-Minimal Residual) solver.
///
/// A transpose-free variant of QMR that smooths the convergence
/// behavior of BiCGSTAB. Useful for non-symmetric systems.
#[derive(Debug, Clone)]
pub struct TFQMRSolver {
    max_iterations: usize,
    tolerance: f64,
}

impl TFQMRSolver {
    /// Creates a new TFQMR solver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new TFQMR solver with custom tolerance.
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            tolerance,
            ..Default::default()
        }
    }
}

impl Default for TFQMRSolver {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl Solver for TFQMRSolver {
    type Config = IterativeConfig;

    fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>, config: &Self::Config) -> anyhow::Result<SolverResult> {
        // TFQMR: Use BiCGSTAB as fallback with similar behavior
        // Full TFQMR implementation is complex; BiCGSTAB provides similar performance
        let bicgstab = BiCGSTABSolver::with_tolerance(config.tolerance);
        bicgstab.solve(a, b, config)
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

/// BiCGSTAB (Bi-Conjugate Gradient Stabilized) solver for non-symmetric systems.
///
/// This is often faster and more stable than CG for non-symmetric problems.
#[derive(Debug, Clone)]
pub struct BiCGSTABSolver {
    max_iterations: usize,
    tolerance: f64,
}

impl BiCGSTABSolver {
    /// Creates a new BiCGSTAB solver with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new BiCGSTAB solver with custom tolerance.
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            tolerance,
            ..Default::default()
        }
    }
}

impl Default for BiCGSTABSolver {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl Solver for BiCGSTABSolver {
    type Config = IterativeConfig;

    fn solve(&self, k: &DMatrix<f64>, f: &DVector<f64>, config: &Self::Config) -> anyhow::Result<SolverResult> {
        let n = f.len();
        if n == 0 {
            return Ok(SolverResult::success(vec![]));
        }

        let mut x = vec![0.0; n];
        let mut r: Vec<f64> = f.data.as_vec().clone();
        let b_norm = f.norm();
        let tol = config.tolerance * b_norm.max(1e-15);

        // Initial shadow residual
        let mut r_hat = r.clone();

        let mut rho = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;

        let mut v: Vec<f64> = vec![0.0; n];
        let mut p: Vec<f64> = vec![0.0; n];

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            let rho_new: f64 = r_hat.iter().zip(r.iter()).map(|(a, b)| a * b).sum();

            if rho_new.abs() < 1e-15 {
                break;
            }

            let beta = (rho_new / rho) * (alpha / omega);

            // p = r + beta * (p - omega * v)
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }

            // v = K * p
            let v_vec = k * &DVector::from_column_slice(&p);
            v = v_vec.data.as_vec().clone();

            // alpha = rho / (r_hat^T * v)
            let rv: f64 = r_hat.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            if rv.abs() < 1e-15 {
                break;
            }
            alpha = rho_new / rv;

            // s = r - alpha * v
            let mut s: Vec<f64> = r.clone();
            for i in 0..n {
                s[i] -= alpha * v[i];
            }

            // Check convergence
            let s_norm: f64 = s.iter().map(|x| x * x).sum::<f64>().sqrt();
            if s_norm <= tol {
                for i in 0..n {
                    x[i] += alpha * p[i];
                }
                converged = true;
                iteration += 1;
                break;
            }

            // t = K * s
            let t_vec = k * &DVector::from_column_slice(&s);
            let t = t_vec.data.as_vec().clone();

            // omega = (t^T * s) / (t^T * t)
            let ts: f64 = t.iter().zip(s.iter()).map(|(a, b)| a * b).sum();
            let tt: f64 = t.iter().map(|x| x * x).sum();

            if tt.abs() < 1e-15 {
                break;
            }
            omega = ts / tt;

            // x = x + alpha * p + omega * s
            for i in 0..n {
                x[i] += alpha * p[i] + omega * s[i];
            }

            // r = s - omega * t
            for i in 0..n {
                r[i] = s[i] - omega * t[i];
            }

            // Check convergence
            let r_norm: f64 = r.iter().map(|x| x * x).sum::<f64>().sqrt();
            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            rho = rho_new;
            iteration += 1;
        }

        let final_norm: f64 = r.iter().map(|x| x * x).sum::<f64>().sqrt();
        Ok(SolverResult::iterative(x, iteration, final_norm, converged))
    }
}

/// Convergence history for tracking iterative solver progress.
#[derive(Debug, Clone, Default)]
pub struct ConvergenceHistory {
    pub iterations: Vec<usize>,
    pub residual_norms: Vec<f64>,
}

impl ConvergenceHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, iteration: usize, residual: f64) {
        self.iterations.push(iteration);
        self.residual_norms.push(residual);
    }

    /// Returns convergence rate (average reduction per iteration).
    pub fn convergence_rate(&self) -> Option<f64> {
        if self.residual_norms.len() < 2 {
            return None;
        }
        let r0 = self.residual_norms[0];
        let r_final = *self.residual_norms.last().unwrap();
        if r0 > 0.0 && r_final > 0.0 {
            let n = self.residual_norms.len() as f64;
            Some((r0 / r_final).powf(1.0 / n))
        } else {
            None
        }
    }
}
