//! Comprehensive benchmark suite for FEA solver acceleration techniques.

use fea::prelude::*;
use fea::algorithms::solvers::{Preconditioner, IterativeConfig};
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("=== FEA Acceleration Benchmark Suite ===\n");

    benchmark_acceleration_methods()?;
    benchmark_problem_scaling()?;

    println!("\n=== Benchmark Complete ===");
    Ok(())
}

fn benchmark_acceleration_methods() -> anyhow::Result<()> {
    println!("1. Acceleration Methods Comparison\n");

    let (a, b): (DMatrix<f64>, DVector<f64>) = create_poisson_problem(50);

    println!("{:<20} | {:>10} | {:>12}", "Method", "Iterations", "Time (ms)");
    println!("---------------------|------------|--------------");

    // Baseline: Standard CG
    let cg = CGSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = cg.solve(&a, &b, &IterativeConfig::default())?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<20} | {:>10} | {:>12.1}", "CG", result.iterations.unwrap_or(0), time);

    // PCG
    let pcg = PCGSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = pcg.solve(&a, &b, &IterativeConfig::default())?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<20} | {:>10} | {:>12.1}", "PCG", result.iterations.unwrap_or(0), time);

    // Anderson acceleration
    let config = IterativeConfig::default().with_anderson(5);
    let cg = CGSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = cg.solve(&a, &b, &config)?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<20} | {:>10} | {:>12.1}", "CG+Anderson(5)", result.iterations.unwrap_or(0), time);

    // SSOR preconditioner
    let config = IterativeConfig {
        preconditioner: Preconditioner::SSOR(1.5),
        ..Default::default()
    };
    let cg = CGSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = cg.solve(&a, &b, &config)?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<20} | {:>10} | {:>12.1}", "CG+SSOR(1.5)", result.iterations.unwrap_or(0), time);

    // FGMRES with Chebyshev preconditioner
    let cheb = ChebyshevPreconditioner::new(0.1, 10.0, 5);
    let cheb_precond = |v: &DVector<f64>| cheb.apply(&a, v);
    let fgmres = FGMRESSolver::new();
    let start = std::time::Instant::now();
    let result = fgmres.solve_with_precond(&a, &b, cheb_precond)?;
    let time = start.elapsed().as_secs_f64() * 1000.0;
    println!("{:<20} | {:>10} | {:>12.1}", "FGMRES+Chebyshev", result.iterations.unwrap_or(0), time);

    println!();
    Ok(())
}

fn benchmark_problem_scaling() -> anyhow::Result<()> {
    println!("2. Problem Size Scaling (CG+Jacobi)\n");

    println!("{:<12} | {:>10} | {:>12}", "Grid Size", "Iterations", "Time (ms)");
    println!("-------------|------------|--------------");

    for n in [20, 40, 60] {
        let (a, b) = create_poisson_problem(n);
        let config = IterativeConfig {
            preconditioner: Preconditioner::Jacobi,
            ..Default::default()
        };
        let cg = CGSolver::with_tolerance(1e-8);
        let start = std::time::Instant::now();
        let result = cg.solve(&a, &b, &config)?;
        let time = start.elapsed().as_secs_f64() * 1000.0;
        println!("{:<12} | {:>10} | {:>12.1}", format!("{}x{}", n, n), result.iterations.unwrap_or(0), time);
    }
    println!();
    Ok(())
}

fn create_poisson_problem(n: usize) -> (DMatrix<f64>, DVector<f64>) {
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
