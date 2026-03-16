//! Comprehensive Acceleration Comparison Example.
//!
//! This example provides detailed comparison of all acceleration methods
//! across various problem types and sizes.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

/// Test problem configuration.
#[derive(Debug, Clone)]
pub struct TestProblem {
    pub name: String,
    pub matrix: DMatrix<f64>,
    pub rhs: DVector<f64>,
    pub reference_solution: DVector<f64>,
}

/// Performance metrics for a solver run.
#[derive(Debug, Clone)]
pub struct SolverMetrics {
    pub solver_name: String,
    pub problem_name: String,
    pub iterations: usize,
    pub time_ms: f64,
    pub residual: f64,
    pub error: f64,
    pub converged: bool,
}

fn main() -> anyhow::Result<()> {
    println!("=== Comprehensive Acceleration Comparison ===\n");
    println!("Testing all acceleration methods across various problems.\n");

    // Generate test problems
    let problems = generate_test_problems();

    println!("Generated {} test problems\n", problems.len());

    // Run comprehensive comparison
    let mut all_metrics = Vec::new();

    for problem in &problems {
        println!("─".repeat(70));
        println!("Problem: {} (size: {} x {})", problem.name, problem.matrix.nrows(), problem.matrix.ncols());
        println!("─".repeat(70));

        let metrics = run_solver_comparison(problem)?;
        all_metrics.extend(metrics);

        println!();
    }

    // Generate summary report
    generate_summary_report(&all_metrics)?;

    println!("\n=== Comparison Complete ===");
    Ok(())
}

/// Generates a suite of test problems.
fn generate_test_problems() -> Vec<TestProblem> {
    let mut problems = Vec::new();

    // Problem 1: 1D Poisson (tridiagonal)
    let n1 = 200;
    let (k1, f1) = generate_1d_poisson(n1);
    let ref1 = k1.clone().lu().solve(&f1).unwrap();
    problems.push(TestProblem {
        name: "1D Poisson (n=200)".to_string(),
        matrix: k1,
        rhs: f1,
        reference_solution: ref1,
    });

    // Problem 2: 2D Poisson (pentadiagonal)
    let n2 = 100;
    let (k2, f2) = generate_2d_poisson(n2);
    let ref2 = k2.clone().lu().solve(&f2).unwrap();
    problems.push(TestProblem {
        name: "2D Poisson (n=100)".to_string(),
        matrix: k2,
        rhs: f2,
        reference_solution: ref2,
    });

    // Problem 3: High condition number
    let n3 = 100;
    let (k3, f3) = generate_ill_conditioned(n3);
    let ref3 = k3.clone().lu().solve(&f3).unwrap();
    problems.push(TestProblem {
        name: "Ill-conditioned (n=100)".to_string(),
        matrix: k3,
        rhs: f3,
        reference_solution: ref3,
    });

    // Problem 4: FEA-like stiffness matrix
    let n4 = 150;
    let (k4, f4) = generate_fea_stiffness(n4);
    let ref4 = k4.clone().lu().solve(&f4).unwrap();
    problems.push(TestProblem {
        name: "FEA Stiffness (n=150)".to_string(),
        matrix: k4,
        rhs: f4,
        reference_solution: ref4,
    });

    problems
}

/// Generates 1D Poisson equation matrix.
fn generate_1d_poisson(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut k = DMatrix::zeros(n, n);
    for i in 0..n {
        k[(i, i)] = 2.0;
        if i > 0 { k[(i, i - 1)] = -1.0; }
        if i < n - 1 { k[(i, i + 1)] = -1.0; }
    }
    let f = DVector::from_element(n, 1.0);
    (k, f)
}

/// Generates 2D Poisson equation matrix (5-point stencil).
fn generate_2d_poisson(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let size = n * n;
    let mut k = DMatrix::zeros(size, size);
    let h = 1.0 / (n + 1) as f64;

    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            k[(idx, idx)] = 4.0 / (h * h);
            if i > 0 { k[(idx, idx - n)] = -1.0 / (h * h); }
            if i < n - 1 { k[(idx, idx + n)] = -1.0 / (h * h); }
            if j > 0 { k[(idx, idx - 1)] = -1.0 / (h * h); }
            if j < n - 1 { k[(idx, idx + 1)] = -1.0 / (h * h); }
        }
    }
    let f = DVector::from_element(size, 1.0);
    (k, f)
}

/// Generates an ill-conditioned matrix.
fn generate_ill_conditioned(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut k = DMatrix::zeros(n, n);
    for i in 0..n {
        k[(i, i)] = 1.0 + (i as f64) * 100.0; // Increasing diagonal
        if i > 0 { k[(i, i - 1)] = -1.0; }
        if i < n - 1 { k[(i, i + 1)] = -1.0; }
    }
    let f = DVector::from_element(n, 1.0);
    (k, f)
}

/// Generates FEA-like stiffness matrix.
fn generate_fea_stiffness(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut k = DMatrix::zeros(n, n);
    for i in 0..n {
        k[(i, i)] = 4.0 + (i as f64) * 0.1;
        if i > 0 { k[(i, i - 1)] = -1.0 - (i as f64) * 0.05; }
        if i < n - 1 { k[(i, i + 1)] = -1.0 - (i as f64) * 0.05; }
        if i > 1 { k[(i, i - 2)] = -0.1; }
        if i < n - 2 { k[(i, i + 2)] = -0.1; }
    }
    let f = DVector::from_fn(n, |i, _| (i as f64 * 0.1).sin());
    (k, f)
}

/// Runs solver comparison on a single problem.
fn run_solver_comparison(problem: &TestProblem) -> anyhow::Result<Vec<SolverMetrics>> {
    let mut metrics = Vec::new();

    let solver_configs: Vec<(&str, Box<dyn Fn(&DMatrix<f64>, &DVector<f64>) -> (usize, f64, bool)>)> = vec![
        ("CG + Jacobi", Box::new(|k, f| {
            let cfg = IterativeConfig {
                max_iterations: 500,
                tolerance: 1e-10,
                preconditioner: Preconditioner::Jacobi,
                anderson_depth: 0,
                krylov_dim: 0,
                deflation_vectors: None,
            };
            let cg = CGSolver::with_config(cfg.clone());
            if let Ok(res) = cg.solve(k, f, &cfg) {
                (res.iterations.unwrap_or(0), res.residual_norm.unwrap_or(f64::INFINITY), res.converged)
            } else {
                (0, f64::INFINITY, false)
            }
        })),
        ("CG + Chebyshev(2)", Box::new(|k, f| {
            let cfg = IterativeConfig {
                max_iterations: 500,
                tolerance: 1e-10,
                preconditioner: Preconditioner::Chebyshev(2),
                anderson_depth: 0,
                krylov_dim: 0,
                deflation_vectors: None,
            };
            let cg = CGSolver::with_config(cfg.clone());
            if let Ok(res) = cg.solve(k, f, &cfg) {
                (res.iterations.unwrap_or(0), res.residual_norm.unwrap_or(f64::INFINITY), res.converged)
            } else {
                (0, f64::INFINITY, false)
            }
        })),
        ("CG + Chebyshev(3)", Box::new(|k, f| {
            let cfg = IterativeConfig {
                max_iterations: 500,
                tolerance: 1e-10,
                preconditioner: Preconditioner::Chebyshev(3),
                anderson_depth: 0,
                krylov_dim: 0,
                deflation_vectors: None,
            };
            let cg = CGSolver::with_config(cfg.clone());
            if let Ok(res) = cg.solve(k, f, &cfg) {
                (res.iterations.unwrap_or(0), res.residual_norm.unwrap_or(f64::INFINITY), res.converged)
            } else {
                (0, f64::INFINITY, false)
            }
        })),
        ("CG + Anderson(5)", Box::new(|k, f| {
            let cfg = IterativeConfig {
                max_iterations: 500,
                tolerance: 1e-10,
                preconditioner: Preconditioner::Jacobi,
                anderson_depth: 5,
                krylov_dim: 0,
                deflation_vectors: None,
            };
            let cg = CGSolver::with_config(cfg.clone());
            if let Ok(res) = cg.solve(k, f, &cfg) {
                (res.iterations.unwrap_or(0), res.residual_norm.unwrap_or(f64::INFINITY), res.converged)
            } else {
                (0, f64::INFINITY, false)
            }
        })),
        ("BiCGSTAB", Box::new(|k, f| {
            let solver = BiCGSTABSolver::with_tolerance(1e-10);
            if let Ok(res) = solver.solve(k, f, &IterativeConfig::default()) {
                (res.iterations.unwrap_or(0), res.residual_norm.unwrap_or(f64::INFINITY), res.converged)
            } else {
                (0, f64::INFINITY, false)
            }
        })),
        ("GMRES(30)", Box::new(|k, f| {
            let solver = GMRESSolver::with_tolerance(1e-10);
            if let Ok(res) = solver.solve(k, f, &IterativeConfig::default()) {
                (res.iterations.unwrap_or(0), res.residual_norm.unwrap_or(f64::INFINITY), res.converged)
            } else {
                (0, f64::INFINITY, false)
            }
        })),
    ];

    println!("\n{:<20} | {:>10} | {:>12} | {:>10} | {:>8}",
             "Solver", "Iterations", "Time (ms)", "Residual", "Conv");
    println!("{}", "-".repeat(70));

    for (name, solver_fn) in solver_configs {
        let start = Instant::now();
        let (iters, residual, converged) = solver_fn(&problem.matrix, &problem.rhs);
        let elapsed = start.elapsed();

        // Compute solution error
        let cfg = IterativeConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };
        let cg = CGSolver::with_config(cfg.clone());
        let error = if let Ok(res) = cg.solve(&problem.matrix, &problem.rhs, &cfg) {
            let x = DVector::from_column_slice(&res.solution);
            (x - &problem.reference_solution).norm() / problem.reference_solution.norm()
        } else {
            f64::INFINITY
        };

        println!("{:<20} | {:>10} | {:>12.4} | {:>10.2e} | {:>8}",
                 name, iters, elapsed.as_secs_f64() * 1000.0, residual,
                 if converged { "Yes" } else { "No" });

        metrics.push(SolverMetrics {
            solver_name: name.to_string(),
            problem_name: problem.name.clone(),
            iterations: iters,
            time_ms: elapsed.as_secs_f64() * 1000.0,
            residual,
            error,
            converged,
        });
    }

    Ok(metrics)
}

/// Generates a summary report of all results.
fn generate_summary_report(metrics: &[SolverMetrics]) -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY REPORT");
    println!("{}\n", "=".repeat(70));

    // Group by problem
    let problems: Vec<String> = metrics.iter()
        .map(|m| m.problem_name.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    for problem in &problems {
        println!("Problem: {}", problem);
        println!("{}", "-".repeat(50));

        let problem_metrics: Vec<_> = metrics.iter()
            .filter(|m| &m.problem_name == problem)
            .collect();

        // Find best by iterations
        if let Some(best) = problem_metrics.iter()
            .filter(|m| m.converged)
            .min_by(|a, b| a.iterations.cmp(&b.iterations))
        {
            println!("  Best by iterations: {} ({} iterations)", best.solver_name, best.iterations);
        }

        // Find best by time
        if let Some(best) = problem_metrics.iter()
            .filter(|m| m.converged)
            .min_by(|a, b| a.time_ms.partial_cmp(&b.time_ms).unwrap())
        {
            println!("  Best by time: {} ({:.4} ms)", best.solver_name, best.time_ms);
        }

        println!();
    }

    // Overall statistics
    println!("Overall Statistics:");
    println!("{}", "-".repeat(50));

    let total_solvers = metrics.len();
    let converged_count = metrics.iter().filter(|m| m.converged).count();
    let avg_iterations = metrics.iter()
        .filter(|m| m.converged)
        .map(|m| m.iterations)
        .sum::<usize>() as f64 / converged_count.max(1) as f64;
    let avg_time = metrics.iter()
        .filter(|m| m.converged)
        .map(|m| m.time_ms)
        .sum::<f64>() / converged_count.max(1) as f64;

    println!("  Total solver runs: {}", total_solvers);
    println!("  Converged: {} ({:.1}%)", converged_count, 100.0 * converged_count as f64 / total_solvers as f64);
    println!("  Average iterations: {:.1}", avg_iterations);
    println!("  Average time: {:.4} ms", avg_time);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_comparison() {
        let problems = generate_test_problems();
        assert!(!problems.is_empty());

        for problem in &problems {
            let metrics = run_solver_comparison(problem).unwrap();
            assert!(!metrics.is_empty());
        }
    }

    #[test]
    fn test_problem_generation() {
        let (k, f) = generate_1d_poisson(10);
        assert_eq!(k.nrows(), 10);
        assert_eq!(f.len(), 10);

        let (k, f) = generate_2d_poisson(5);
        assert_eq!(k.nrows(), 25);
        assert_eq!(f.len(), 25);
    }
}
