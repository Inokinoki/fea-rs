//! Ultimate Acceleration Showcase.
//!
//! This example demonstrates ALL acceleration methods in the FEA library,
//! showing how they can be combined for optimal performance.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

/// Benchmark result for a single solver run.
#[derive(Debug, Clone)]
struct BenchmarkEntry {
    name: String,
    iterations: usize,
    time_ms: f64,
    residual: f64,
    converged: bool,
}

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         FEA Library - Ultimate Acceleration Showcase         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Generate test problems of varying difficulty
    let problems = generate_test_suite();

    println!("Test Suite: {} problems\n", problems.len());

    let mut all_results = Vec::new();

    for (idx, problem) in problems.iter().enumerate() {
        println!("─".repeat(70));
        println!("Problem {}/{}: {} ({} DOFs, κ ≈ {:.1e})",
                 idx + 1, problems.len(), problem.name, problem.size, problem.condition_number);
        println!("─".repeat(70));

        let results = run_acceleration_showcase(&problem)?;
        all_results.extend(results);

        println!();
    }

    // Generate comprehensive report
    generate_showcase_report(&all_results)?;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              Showcase Complete - See report.txt              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Test problem definition.
struct TestProblem {
    name: String,
    size: usize,
    condition_number: f64,
    matrix: DMatrix<f64>,
    rhs: DVector<f64>,
    reference: DVector<f64>,
}

/// Generates a comprehensive test suite.
fn generate_test_suite() -> Vec<TestProblem> {
    let mut problems = Vec::new();

    // Problem 1: Small well-conditioned system
    let n1 = 100;
    let (k1, f1) = create_1d_poisson(n1);
    let ref1 = k1.clone().lu().solve(&f1).unwrap();
    problems.push(TestProblem {
        name: "1D Poisson Small".to_string(),
        size: n1,
        condition_number: estimate_condition(&k1),
        matrix: k1,
        rhs: f1,
        reference: ref1,
    });

    // Problem 2: Medium 1D Poisson
    let n2 = 500;
    let (k2, f2) = create_1d_poisson(n2);
    let ref2 = k2.clone().lu().solve(&f2).unwrap();
    problems.push(TestProblem {
        name: "1D Poisson Medium".to_string(),
        size: n2,
        condition_number: estimate_condition(&k2),
        matrix: k2,
        rhs: f2,
        reference: ref2,
    });

    // Problem 3: Ill-conditioned system
    let n3 = 200;
    let (k3, f3) = create_ill_conditioned(n3);
    let ref3 = k3.clone().lu().solve(&f3).unwrap();
    problems.push(TestProblem {
        name: "Ill-Conditioned".to_string(),
        size: n3,
        condition_number: estimate_condition(&k3),
        matrix: k3,
        rhs: f3,
        reference: ref3,
    });

    // Problem 4: FEA-like stiffness
    let n4 = 300;
    let (k4, f4) = create_fea_stiffness(n4);
    let ref4 = k4.clone().lu().solve(&f4).unwrap();
    problems.push(TestProblem {
        name: "FEA Stiffness".to_string(),
        size: n4,
        condition_number: estimate_condition(&k4),
        matrix: k4,
        rhs: f4,
        reference: ref4,
    });

    problems
}

/// Creates 1D Poisson equation matrix.
fn create_1d_poisson(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut k = DMatrix::zeros(n, n);
    for i in 0..n {
        k[(i, i)] = 2.0;
        if i > 0 { k[(i, i - 1)] = -1.0; }
        if i < n - 1 { k[(i, i + 1)] = -1.0; }
    }
    (k, DVector::from_element(n, 1.0))
}

/// Creates ill-conditioned matrix.
fn create_ill_conditioned(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut k = DMatrix::zeros(n, n);
    for i in 0..n {
        k[(i, i)] = 1.0 + (i as f64) * 1000.0;
        if i > 0 { k[(i, i - 1)] = -1.0; }
        if i < n - 1 { k[(i, i + 1)] = -1.0; }
    }
    (k, DVector::from_element(n, 1.0))
}

/// Creates FEA-like stiffness matrix.
fn create_fea_stiffness(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut k = DMatrix::zeros(n, n);
    for i in 0..n {
        k[(i, i)] = 4.0 + (i as f64) * 0.1;
        if i > 0 { k[(i, i - 1)] = -1.0 - (i as f64) * 0.05; }
        if i < n - 1 { k[(i, i + 1)] = -1.0 - (i as f64) * 0.05; }
    }
    (k, DVector::from_fn(n, |i, _| (i as f64 * 0.1).sin()))
}

/// Estimates condition number using power iteration.
fn estimate_condition(k: &DMatrix<f64>) -> f64 {
    // Simplified estimate using Gershgorin
    let n = k.nrows();
    let mut lambda_max = 0.0;
    let mut lambda_min = f64::INFINITY;

    for i in 0..n {
        let diag = k[(i, i)].abs();
        let mut off_diag_sum = 0.0;
        for j in 0..n {
            if i != j {
                off_diag_sum += k[(i, j)].abs();
            }
        }
        lambda_max = lambda_max.max(diag + off_diag_sum);
        lambda_min = lambda_min.min((diag - off_diag_sum).max(1e-15));
    }

    lambda_max / lambda_min
}

/// Runs all acceleration methods on a problem.
fn run_acceleration_showcase(problem: &TestProblem) -> anyhow::Result<Vec<BenchmarkEntry>> {
    let mut results = Vec::new();

    // Category 1: Classical Methods
    println!("\n━━ Classical Methods ━━");

    results.push(run_benchmark("CG + Jacobi", |k, f| {
        let cfg = IterativeConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };
        CGSolver::with_config(cfg.clone()).solve(k, f, &cfg)
    }, &problem.matrix, &problem.rhs)?);

    results.push(run_benchmark("CG + Chebyshev(3)", |k, f| {
        let cfg = IterativeConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            preconditioner: Preconditioner::Chebyshev(3),
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };
        CGSolver::with_config(cfg.clone()).solve(k, f, &cfg)
    }, &problem.matrix, &problem.rhs)?);

    results.push(run_benchmark("BiCGSTAB", |k, f| {
        BiCGSTABSolver::with_tolerance(1e-10).solve(k, f, &IterativeConfig::default())
    }, &problem.matrix, &problem.rhs)?);

    // Category 2: Acceleration Methods
    println!("\n━━ Acceleration Methods ━━");

    results.push(run_benchmark("Anderson(5)", |k, f| {
        let cfg = IterativeConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 5,
            krylov_dim: 0,
            deflation_vectors: None,
        };
        CGSolver::with_config(cfg.clone()).solve(k, f, &cfg)
    }, &problem.matrix, &problem.rhs)?);

    results.push(run_benchmark("Anderson(10)", |k, f| {
        let cfg = IterativeConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 10,
            krylov_dim: 0,
            deflation_vectors: None,
        };
        CGSolver::with_config(cfg.clone()).solve(k, f, &cfg)
    }, &problem.matrix, &problem.rhs)?);

    // Category 3: Advanced Methods
    println!("\n━━ Advanced Methods ━━");

    // Spectral Deflation + CG
    results.push(run_benchmark_with_setup("Deflated CG", |k, f| {
        let deflation = SpectralDeflation::compute_deflation_subspace(k, 5, 50);
        let cfg = IterativeConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: Some(deflation.eigenvectors),
        };
        CGSolver::with_config(cfg.clone()).solve(k, f, &cfg)
    }, &problem.matrix, &problem.rhs)?);

    // GCRO-DR
    results.push(run_benchmark_with_setup("GCRO-DR", |k, f| {
        let gcro = GCRODRSolver::new(20, 5);
        let (x, iters, residual, conv) = gcro.solve(k, f, None, 1e-10, 500);
        Ok(SolverResult::iterative(x.data.as_vec().clone(), iters, residual, conv))
    }, &problem.matrix, &problem.rhs)?);

    // Recycling BiCGSTAB
    results.push(run_benchmark_with_setup("Recycling BiCGSTAB", |k, f| {
        let mut rbicg = RecyclingBiCGSTAB::new(10);
        let (x, iters, residual, conv) = rbicg.solve(k, f, 1e-10, 500);
        Ok(SolverResult::iterative(x.data.as_vec().clone(), iters, residual, conv))
    }, &problem.matrix, &problem.rhs)?);

    // Category 4: Unified Framework
    println!("\n━━ Unified Framework ━━");

    results.push(run_benchmark_with_setup("Unified Adaptive", |k, f| {
        let config = UnifiedSolverConfig {
            strategy: AccelerationStrategy::Adaptive,
            max_iterations: 500,
            tolerance: 1e-10,
            enable_monitoring: true,
            verbose: false,
        };
        let mut solver = UnifiedAccelerationSolver::new(config);
        let result = solver.solve(k, f);
        Ok(result.base_result)
    }, &problem.matrix, &problem.rhs)?);

    // Category 5: Block Methods
    println!("\n━━ Block Methods ━━");

    results.push(run_benchmark_with_setup("Block Jacobi(4)", |k, f| {
        let solver = BlockJacobiIterative::new(4);
        let (x, iters, residual, conv) = solver.solve(k, f);
        Ok(SolverResult::iterative(x.data.as_vec().clone(), iters, residual, conv))
    }, &problem.matrix, &problem.rhs)?);

    // Category 6: Mixed Precision
    println!("\n━━ Mixed Precision ━━");

    results.push(run_benchmark_with_setup("Mixed Precision", |k, f| {
        let config = MixedPrecisionConfig {
            inner_tolerance: 1e-4,
            outer_tolerance: 1e-10,
            max_refinement_steps: 20,
            max_inner_iterations: 50,
            use_simulated_fp16: false,
        };
        let solver = MixedPrecisionSolver::new(config);
        let result = solver.solve(k, f);
        Ok(SolverResult::iterative(
            result.solution.data.as_vec().clone(),
            result.refinement_steps,
            result.residual_norm,
            result.converged,
        ))
    }, &problem.matrix, &problem.rhs)?);

    // Print summary for this problem
    print_problem_summary(&results);

    Ok(results)
}

/// Runs a benchmark for a single solver.
fn run_benchmark<F>(name: &str, solver: F, k: &DMatrix<f64>, f: &DVector<f64>) -> anyhow::Result<BenchmarkEntry>
where
    F: FnOnce(&DMatrix<f64>, &DVector<f64>) -> anyhow::Result<SolverResult>,
{
    let start = Instant::now();
    let result = solver(k, f)?;
    let elapsed = start.elapsed();

    Ok(BenchmarkEntry {
        name: name.to_string(),
        iterations: result.iterations.unwrap_or(0),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        residual: result.residual_norm.unwrap_or(f64::INFINITY),
        converged: result.converged,
    })
}

/// Runs a benchmark with setup phase.
fn run_benchmark_with_setup<F>(name: &str, solver: F, k: &DMatrix<f64>, f: &DVector<f64>) -> anyhow::Result<BenchmarkEntry>
where
    F: FnOnce(&DMatrix<f64>, &DVector<f64>) -> anyhow::Result<SolverResult>,
{
    run_benchmark(name, solver, k, f)
}

/// Prints summary for a single problem.
fn print_problem_summary(results: &[BenchmarkEntry]) {
    println!("\n{:<25} | {:>8} | {:>12} | {:>10} | {:>8}",
             "Method", "Iterations", "Time (ms)", "Residual", "Status");
    println!("{}", "-".repeat(72));

    for r in results {
        let status = if r.converged { "✓" } else { "✗" };
        println!("{:<25} | {:>8} | {:>12.4} | {:>10.2e} | {:>8}",
                 r.name, r.iterations, r.time_ms, r.residual, status);
    }

    // Find best
    if let Some(best) = results.iter().filter(|r| r.converged).min_by(|a, b| {
        a.time_ms.partial_cmp(&b.time_ms).unwrap()
    }) {
        println!("\n  Best: {} ({:.4} ms, {} iterations)",
                 best.name, best.time_ms, best.iterations);
    }
}

/// Generates comprehensive showcase report.
fn generate_showcase_report(results: &[BenchmarkEntry]) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("acceleration_showcase_report.txt")?;

    writeln!(file, "FEA Library - Ultimate Acceleration Showcase Report")?;
    writeln!(file, "===================================================\n")?;

    writeln!(file, "Total benchmark runs: {}\n", results.len())?;

    // Group by category
    let categories = [
        "CG + Jacobi", "CG + Chebyshev", "BiCGSTAB",
        "Anderson", "Deflated", "GCRO-DR", "Recycling",
        "Unified", "Block", "Mixed Precision",
    ];

    for category in &categories {
        let cat_results: Vec<_> = results.iter()
            .filter(|r| r.name.contains(category))
            .collect();

        if !cat_results.is_empty() {
            writeln!(file, "Category: {}", category)?;
            writeln!(file, "{}", "-".repeat(50))?;

            let avg_time: f64 = cat_results.iter().map(|r| r.time_ms).sum::<f64>() / cat_results.len() as f64;
            let avg_iters: f64 = cat_results.iter().map(|r| r.iterations as f64).sum::<f64>() / cat_results.len() as f64;
            let conv_rate = cat_results.iter().filter(|r| r.converged).count() as f64 / cat_results.len() as f64 * 100.0;

            writeln!(file, "  Runs: {}", cat_results.len())?;
            writeln!(file, "  Avg Time: {:.4} ms", avg_time)?;
            writeln!(file, "  Avg Iterations: {:.1}", avg_iters)?;
            writeln!(file, "  Convergence Rate: {:.1}%", conv_rate)?;
            writeln!(file)?;
        }
    }

    // Overall statistics
    writeln!(file, "Overall Statistics")?;
    writeln!(file, "{}", "=".repeat(50))?;

    let total_converged = results.iter().filter(|r| r.converged).count();
    let overall_conv_rate = total_converged as f64 / results.len() as f64 * 100.0;

    writeln!(file, "Total Runs: {}", results.len())?;
    writeln!(file, "Converged: {}/{} ({:.1}%)", total_converged, results.len(), overall_conv_rate)?;

    if let Some(fastest) = results.iter().filter(|r| r.converged).min_by(|a, b| {
        a.time_ms.partial_cmp(&b.time_ms).unwrap()
    }) {
        writeln!(file, "\nFastest Run: {} ({:.4} ms)", fastest.name, fastest.time_ms)?;
    }

    if let Some(fewest_iters) = results.iter().filter(|r| r.converged).min_by(|a, b| {
        a.iterations.cmp(&b.iterations)
    }) {
        writeln!(file, "Fewest Iterations: {} ({} iters)", fewest_iters.name, fewest_iters.iterations)?;
    }

    writeln!(file, "\nReport generated successfully.")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_showcase_generation() {
        let problems = generate_test_suite();
        assert!(!problems.is_empty());
        for p in &problems {
            assert!(p.size > 0);
            assert!(p.condition_number > 0.0);
        }
    }

    #[test]
    fn test_benchmark_run() {
        let (k, f) = create_1d_poisson(50);
        let problem = TestProblem {
            name: "Test".to_string(),
            size: 50,
            condition_number: 100.0,
            matrix: k.clone(),
            rhs: f.clone(),
            reference: k.lu().solve(&f).unwrap(),
        };

        let results = run_acceleration_showcase(&problem);
        assert!(results.is_ok());
    }
}
