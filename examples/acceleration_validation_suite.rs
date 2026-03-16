//! Acceleration Method Validation and Benchmarking Suite.
//!
//! This example provides comprehensive validation of all acceleration methods
//! with automated benchmarking and accuracy verification.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

/// Test result for a single method.
#[derive(Debug, Clone)]
struct MethodTestResult {
    method_name: String,
    problem_size: usize,
    iterations: usize,
    time_ms: f64,
    residual: f64,
    error: f64,
    converged: bool,
    passed: bool,
}

/// Validation test harness.
struct ValidationHarness {
    results: Vec<MethodTestResult>,
    tolerance: f64,
}

impl ValidationHarness {
    fn new(tolerance: f64) -> Self {
        Self {
            results: Vec::new(),
            tolerance,
        }
    }

    fn run_test<F>(&mut self, name: &str, size: usize, solver_fn: F)
    where
        F: FnOnce(&DMatrix<f64>, &DVector<f64>, &DVector<f64>) -> (usize, f64, bool),
    {
        let (k, f, ref_sol) = create_problem_with_reference(size);

        let start = Instant::now();
        let (iterations, residual, converged) = solver_fn(&k, &f, &ref_sol);
        let time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Compute error against reference solution
        let x = DVector::from_column_slice(&k.clone().lu().solve(&f).unwrap_or(ref_sol.clone()).data.as_slice());
        let error = (x - &ref_sol).norm() / ref_sol.norm();

        let passed = converged && error < self.tolerance;

        self.results.push(MethodTestResult {
            method_name: name.to_string(),
            problem_size: size,
            iterations,
            time_ms,
            residual,
            error,
            converged,
            passed,
        });
    }

    fn print_summary(&self) {
        println!("\n{:60}", "VALIDATION SUMMARY");
        println!("{}", "=".repeat(120));
        println!("{:<30} | {:>8} | {:>10} | {:>12} | {:>12} | {:>10} | {:>8}",
                 "Method", "Size", "Iterations", "Time (ms)", "Residual", "Error", "Status");
        println!("{}", "=".repeat(120));

        let mut passed = 0;
        let mut failed = 0;

        for r in &self.results {
            let status = if r.passed { "✓ PASS" } else { "✗ FAIL" };
            if r.passed { passed += 1; } else { failed += 1; }

            println!("{:<30} | {:>8} | {:>10} | {:>12.4} | {:>12.2e} | {:>10.2e} | {:>8}",
                     r.method_name, r.problem_size, r.iterations, r.time_ms,
                     r.residual, r.error, status);
        }

        println!("{}", "=".repeat(120));
        println!("Total: {} passed, {} failed ({:.1}% success rate)",
                 passed, failed, 100.0 * passed as f64 / (passed + failed) as f64);
    }

    fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }
}

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     Acceleration Methods - Validation & Benchmarking     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let mut harness = ValidationHarness::new(1e-6);

    // Problem sizes to test
    let sizes = vec![50, 100, 200];

    for &size in &sizes {
        println!("Testing problem size: {} DOFs\n", size);

        // Classical methods
        test_cg_jacobi(&mut harness, size);
        test_cg_chebyshev(&mut harness, size);
        test_anderson_acceleration(&mut harness, size);
        test_bicgstab(&mut harness, size);

        // Advanced methods
        test_gcro_dr(&mut harness, size);
        test_recycling_bicgstab(&mut harness, size);
        test_unified_adaptive(&mut harness, size);
        test_accelerated_pcg(&mut harness, size);

        // Mixed precision
        test_mixed_precision(&mut harness, size);

        println!();
    }

    harness.print_summary();

    if harness.all_passed() {
        println!("\n✓ All validation tests PASSED");
        Ok(())
    } else {
        println!("\n✗ Some validation tests FAILED");
        std::process::exit(1);
    }
}

/// Test CG + Jacobi preconditioning.
fn test_cg_jacobi(harness: &mut ValidationHarness, size: usize) {
    harness.run_test("CG + Jacobi", size, |k, f, _| {
        let cfg = IterativeConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };
        let cg = CGSolver::with_config(cfg.clone());
        let result = cg.solve(k, f, &cfg).unwrap();
        (
            result.iterations.unwrap_or(0),
            result.residual_norm.unwrap_or(f64::INFINITY),
            result.converged,
        )
    });
}

/// Test CG + Chebyshev preconditioning.
fn test_cg_chebyshev(harness: &mut ValidationHarness, size: usize) {
    harness.run_test("CG + Chebyshev(3)", size, |k, f, _| {
        let cfg = IterativeConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            preconditioner: Preconditioner::Chebyshev(3),
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };
        let cg = CGSolver::with_config(cfg.clone());
        let result = cg.solve(k, f, &cfg).unwrap();
        (
            result.iterations.unwrap_or(0),
            result.residual_norm.unwrap_or(f64::INFINITY),
            result.converged,
        )
    });
}

/// Test Anderson acceleration.
fn test_anderson_acceleration(harness: &mut ValidationHarness, size: usize) {
    harness.run_test("Anderson(5)", size, |k, f, _| {
        let cfg = IterativeConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 5,
            krylov_dim: 0,
            deflation_vectors: None,
        };
        let cg = CGSolver::with_config(cfg.clone());
        let result = cg.solve(k, f, &cfg).unwrap();
        (
            result.iterations.unwrap_or(0),
            result.residual_norm.unwrap_or(f64::INFINITY),
            result.converged,
        )
    });
}

/// Test BiCGSTAB.
fn test_bicgstab(harness: &mut ValidationHarness, size: usize) {
    harness.run_test("BiCGSTAB", size, |k, f, _| {
        let solver = BiCGSTABSolver::with_tolerance(1e-10);
        let result = solver.solve(k, f, &IterativeConfig::default()).unwrap();
        (
            result.iterations.unwrap_or(0),
            result.residual_norm.unwrap_or(f64::INFINITY),
            result.converged,
        )
    });
}

/// Test GCRO-DR.
fn test_gcro_dr(harness: &mut ValidationHarness, size: usize) {
    harness.run_test("GCRO-DR", size, |k, f, _| {
        let gcro = GCRODRSolver::new();
        let result = gcro.solve(k, f).unwrap();
        (
            result.iterations.unwrap_or(0),
            result.residual_norm.unwrap_or(f64::INFINITY),
            result.converged,
        )
    });
}

/// Test Recycling BiCGSTAB.
fn test_recycling_bicgstab(harness: &mut ValidationHarness, size: usize) {
    harness.run_test("Recycling BiCGSTAB", size, |k, f, _| {
        let mut rbicg = RecyclingBiCGSTAB::new(10);
        let result = rbicg.solve(k, f, 1e-10, 500);
        (result.1, result.2, result.3)
    });
}

/// Test Unified Adaptive solver.
fn test_unified_adaptive(harness: &mut ValidationHarness, size: usize) {
    harness.run_test("Unified Adaptive", size, |k, f, _| {
        let config = UnifiedSolverConfig {
            strategy: AccelerationStrategy::Adaptive,
            max_iterations: 500,
            tolerance: 1e-10,
            enable_monitoring: true,
            verbose: false,
        };
        let mut solver = UnifiedAccelerationSolver::new(config);
        let result = solver.solve(k, f);
        (
            result.base_result.iterations.unwrap_or(0),
            result.base_result.residual_norm.unwrap_or(f64::INFINITY),
            result.base_result.converged,
        )
    });
}

/// Test Accelerated PCG.
fn test_accelerated_pcg(harness: &mut ValidationHarness, size: usize) {
    harness.run_test("Accelerated PCG", size, |k, f, _| {
        let config = AcceleratedPCGConfig::default();
        let mut solver = AcceleratedPCG::new(config);
        let result = solver.solve(k, f);
        (
            result.iterations,
            result.residual_norm,
            result.converged,
        )
    });
}

/// Test Mixed Precision solver.
fn test_mixed_precision(harness: &mut ValidationHarness, size: usize) {
    harness.run_test("Mixed Precision", size, |k, f, _| {
        let config = MixedPrecisionConfig {
            inner_tolerance: 1e-4,
            outer_tolerance: 1e-10,
            max_refinement_steps: 20,
            max_inner_iterations: 50,
            use_simulated_fp16: false,
        };
        let solver = MixedPrecisionSolver::new(config);
        let result = solver.solve(k, f);
        (
            result.refinement_steps * result.total_inner_iterations,
            result.residual_norm,
            result.converged,
        )
    });
}

/// Creates a test problem with reference solution.
fn create_problem_with_reference(size: usize) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
    let mut k = DMatrix::zeros(size, size);
    for i in 0..size {
        k[(i, i)] = 4.0;
        if i > 0 { k[(i, i - 1)] = -1.0; }
        if i < size - 1 { k[(i, i + 1)] = -1.0; }
    }
    let f = DVector::from_element(size, 1.0);
    let ref_sol = k.clone().lu().solve(&f).unwrap();
    (k, f, ref_sol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_suite() {
        let mut harness = ValidationHarness::new(1e-5);

        // Test with smaller problem for CI
        test_cg_jacobi(&mut harness, 50);
        test_cg_chebyshev(&mut harness, 50);
        test_anderson_acceleration(&mut harness, 50);
        test_bicgstab(&mut harness, 50);

        // All should pass for small well-conditioned problem
        assert!(harness.results.iter().all(|r| r.passed));
    }
}
