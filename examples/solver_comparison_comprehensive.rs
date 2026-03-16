//! Comprehensive solver comparison example.

use fea::prelude::*;
use fea::algorithms::solvers::{Preconditioner, IterativeConfig};
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("=== Comprehensive Solver Comparison ===\n");

    // Test problem: 2D Poisson equation
    let n = 30;
    let (a, b) = create_poisson_2d(n);

    println!("Problem: 2D Poisson equation ({}x{} grid)\n", n, n);

    // Test all solvers with various preconditioners
    test_solvers(&a, &b)?;

    println!("\n=== Complete ===");
    Ok(())
}

/// Test all solver configurations
fn test_solvers(a: &DMatrix<f64>, b: &DVector<f64>) -> anyhow::Result<()> {
    let tol = 1e-8;
    let max_iter = 500;

    println!("{:<30} | {:>8} | {:>10}", "Method", "Iterations", "Time (ms)");
    println!("-------------------------------|----------|------------");

    // Direct solver (reference)
    let direct = DirectSolver::new();
    let start = std::time::Instant::now();
    let result = direct.solve(a, b, &DirectConfig { use_cholesky: false })?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<30} | {:>8} | {:>10.1f}",
        "Direct (LU)", "N/A", time);

    // CG variants
    test_cg_variants(a, b, tol, max_iter)?;

    // GMRES variants
    test_gmres_variants(a, b, tol, max_iter)?;

    // Advanced solvers
    test_advanced_solvers(a, b, tol, max_iter)?;

    Ok(())
}

/// Test CG solver variants
fn test_cg_variants(a: &DMatrix<f64>, b: &DVector<f64>, tol: f64, max_iter: usize) -> anyhow::Result<()> {
    // Standard CG
    let cg = CGSolver::with_tolerance(tol);
    let start = std::time::Instant::now();
    let result = cg.solve(a, b, &IterativeConfig::default())?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<30} | {:>8} | {:>10.1f}",
        "CG", result.iterations.unwrap_or(0), time);

    // CG + Jacobi
    let config = IterativeConfig {
        preconditioner: Preconditioner::Jacobi,
        max_iterations: max_iter,
        tolerance: tol,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };
    let cg = CGSolver::with_tolerance(tol);
    let start = std::time::Instant::now();
    let result = cg.solve(a, b, &config)?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<30} | {:>8} | {:>10.1f}",
        "CG + Jacobi", result.iterations.unwrap_or(0), time);

    // CG + SSOR
    let config = IterativeConfig {
        preconditioner: Preconditioner::SSOR(1.5),
        max_iterations: max_iter,
        tolerance: tol,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };
    let cg = CGSolver::with_tolerance(tol);
    let start = std::time::Instant::now();
    let result = cg.solve(a, b, &config)?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<30} | {:>8} | {:>10.1f}",
        "CG + SSOR(1.5)", result.iterations.unwrap_or(0), time);

    // CG + Anderson
    let config = IterativeConfig::default().with_anderson(5);
    let cg = CGSolver::with_tolerance(tol);
    let start = std::time::Instant::now();
    let result = cg.solve(a, b, &config)?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<30} | {:>8} | {:>10.1f}",
        "CG + Anderson(5)", result.iterations.unwrap_or(0), time);

    // PCG
    let pcg = PCGSolver::with_tolerance(tol);
    let start = std::time::Instant::now();
    let result = pcg.solve(a, b, &IterativeConfig::default())?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<30} | {:>8} | {:>10.1f}",
        "PCG", result.iterations.unwrap_or(0), time);

    Ok(())
}

/// Test GMRES variants
fn test_gmres_variants(a: &DMatrix<f64>, b: &DVector<f64>, tol: f64, max_iter: usize) -> anyhow::Result<()> {
    // GMRES
    let gmres = GMRESSolver::with_restart(30);
    let start = std::time::Instant::now();
    let result = gmres.solve(a, b, &IterativeConfig::default())?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<30} | {:>8} | {:>10.1f}",
        "GMRES(30)", result.iterations.unwrap_or(0), time);

    // BiCGSTAB
    let bicgstab = BiCGSTABSolver::with_tolerance(tol);
    let start = std::time::Instant::now();
    let result = bicgstab.solve(a, b, &IterativeConfig::default())?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<30} | {:>8} | {:>10.1f}",
        "BiCGSTAB", result.iterations.unwrap_or(0), time);

    // QMR
    let qmr = QMRSolver::new();
    let start = std::time::Instant::now();
    let result = qmr.solve(a, b, &IterativeConfig::default())?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<30} | {:>8} | {:>10.1f}",
        "QMR", result.iterations.unwrap_or(0), time);

    Ok(())
}

/// Test advanced solvers
fn test_advanced_solvers(a: &DMatrix<f64>, b: &DVector<f64>, tol: f64, max_iter: usize) -> anyhow::Result<()> {
    // FGMRES + Chebyshev
    let cheb = ChebyshevPreconditioner::new(0.01, 20.0, 5);
    let cheb_precond = |v: &DVector<f64>| cheb.apply(a, v);
    let fgmres = FGMRESSolver::new();
    let start = std::time::Instant::now();
    let result = fgmres.solve_with_precond(a, b, cheb_precond)?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<30} | {:>8} | {:>10.1f}",
        "FGMRES + Chebyshev", result.iterations.unwrap_or(0), time);

    // Deflated CG
    let deflated_cg = DeflatedCG::new();
    let start = std::time::Instant::now();
    let (x, iter, _conv) = deflated_cg.solve(a, b, tol, max_iter);
    let time = start.elapsed().as_secs_f64() * 1000.0;
    let residual = (a * &x - b).norm();
    println!("{:<30} | {:>8} | {:>10.1f} (res: {:.2e})",
        "Deflated CG", iter, time, residual);

    // Spectral preconditioner
    let spectral = SpectralPreconditioner::new(0.1);
    let start = std::time::Instant::now();
    let x_prec = spectral.apply(b);
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<30} | {:>8} | {:>10.1f}",
        "Spectral Prec", "N/A", time);

    Ok(())
}

fn create_poisson_2d(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let size = n * n;
    let mut a = DMatrix::zeros(size, size);
    let h = 1.0 / (n + 1) as f64;
    let scale = 1.0 / (h * h);

    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            a[(idx, idx)] = 4.0 * scale;
            if i > 0 { a[(idx, idx - n)] = -scale; }
            if i < n - 1 { a[(idx, idx + n)] = -scale; }
            if j > 0 { a[(idx, idx - 1)] = -scale; }
            if j < n - 1 { a[(idx, idx + 1)] = -scale; }
        }
    }
    let b = DVector::from_element(size, 1.0);
    (a, b)
}
