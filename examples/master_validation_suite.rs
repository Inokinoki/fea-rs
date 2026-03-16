//! Master Validation Suite for FEA Acceleration Methods.
//!
//! This validation suite provides rigorous testing of all acceleration
//! methods with known analytical solutions and error bounds.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("=== Master Validation Suite ===\n");
    println!("Validating all acceleration methods with rigorous error bounds.\n");

    let mut results = ValidationResults::new();

    // Run all validation tests
    results.add(run_cg_validation()?);
    results.add(run_preconditioner_validation()?);
    results.add(run_spectral_validation()?);
    results.add(run_recycling_validation()?);
    results.add(run_adaptive_validation()?);
    results.add(run_block_validation()?);
    results.add(run_unified_validation()?);

    // Print summary
    results.print_summary();

    // Check if all passed
    if results.all_passed() {
        println!("\n✓ All validation tests PASSED");
        Ok(())
    } else {
        println!("\n✗ Some validation tests FAILED");
        std::process::exit(1);
    }
}

/// Validation result for a single test.
#[derive(Debug, Clone)]
pub struct ValidationTest {
    pub name: String,
    pub passed: bool,
    pub error: f64,
    pub tolerance: f64,
    pub message: String,
}

/// Collection of validation results.
#[derive(Debug, Clone)]
pub struct ValidationResults {
    pub tests: Vec<ValidationTest>,
}

impl ValidationResults {
    pub fn new() -> Self {
        Self { tests: Vec::new() }
    }

    pub fn add(&mut self, test: ValidationTest) {
        let status = if test.passed { "PASS" } else { "FAIL" };
        println!("  [{}] {} (error: {:.2e})", status, test.name, test.error);
        self.tests.push(test);
    }

    pub fn all_passed(&self) -> bool {
        self.tests.iter().all(|t| t.passed)
    }

    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(60));
        println!("VALIDATION SUMMARY");
        println!("{}\n", "=".repeat(60));

        let passed = self.tests.iter().filter(|t| t.passed).count();
        let total = self.tests.len();

        println!("Tests passed: {}/{} ({:.1}%)", passed, total, 100.0 * passed as f64 / total as f64);

        if !self.all_passed() {
            println!("\nFailed tests:");
            for test in &self.tests {
                if !test.passed {
                    println!("  - {} (error: {:.2e}, tolerance: {:.2e})",
                             test.name, test.error, test.tolerance);
                }
            }
        }
    }
}

/// Validates CG solver variants.
fn run_cg_validation() -> anyhow::Result<ValidationTest> {
    println!("─".repeat(60));
    println!("CG Solver Validation");
    println!("─".repeat(60));

    let n = 100;
    let mut k = DMatrix::zeros(n, n);
    for i in 0..n {
        k[(i, i)] = 4.0;
        if i > 0 { k[(i, i - 1)] = -1.0; }
        if i < n - 1 { k[(i, i + 1)] = -1.0; }
    }
    let f = DVector::from_element(n, 1.0);

    // Reference solution
    let ref_sol = k.clone().lu().solve(&f).unwrap();

    // Test CG with Jacobi
    let cfg = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-12,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };
    let cg = CGSolver::with_config(cfg.clone());
    let res = cg.solve(&k, &f, &cfg)?;

    let sol = DVector::from_column_slice(&res.solution);
    let error = (sol - &ref_sol).norm() / ref_sol.norm();

    Ok(ValidationTest {
        name: "CG + Jacobi".to_string(),
        passed: error < 1e-8,
        error,
        tolerance: 1e-8,
        message: format!("CG converged in {} iterations", res.iterations.unwrap_or(0)),
    })
}

/// Validates preconditioner methods.
fn run_preconditioner_validation() -> anyhow::Result<ValidationTest> {
    println!("\n─".repeat(60));
    println!("Preconditioner Validation");
    println!("─".repeat(60));

    let n = 100;
    let (k, f) = create_test_system(n);
    let ref_sol = k.clone().lu().solve(&f).unwrap();

    // Test Chebyshev preconditioner
    let cfg = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-12,
        preconditioner: Preconditioner::Chebyshev(3),
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };
    let cg = CGSolver::with_config(cfg.clone());
    let res = cg.solve(&k, &f, &cfg)?;

    let sol = DVector::from_column_slice(&res.solution);
    let error = (sol - &ref_sol).norm() / ref_sol.norm();

    // Compare with Jacobi
    let cfg_jacobi = IterativeConfig {
        preconditioner: Preconditioner::Jacobi,
        ..cfg.clone()
    };
    let cg_jacobi = CGSolver::with_config(cfg_jacobi.clone());
    let res_jacobi = cg_jacobi.solve(&k, &f, &cfg_jacobi)?;

    let speedup = res_jacobi.iterations.unwrap_or(1) as f64 / res.iterations.unwrap_or(1) as f64;

    println!("  Chebyshev(3) iterations: {}", res.iterations.unwrap_or(0));
    println!("  Jacobi iterations: {}", res_jacobi.iterations.unwrap_or(0));
    println!("  Speedup: {:.2}x", speedup);

    Ok(ValidationTest {
        name: "Chebyshev Preconditioner".to_string(),
        passed: error < 1e-8 && speedup > 1.0,
        error,
        tolerance: 1e-8,
        message: format!("Speedup: {:.2}x over Jacobi", speedup),
    })
}

/// Validates spectral acceleration methods.
fn run_spectral_validation() -> anyhow::Result<ValidationTest> {
    println!("\n─".repeat(60));
    println!("Spectral Acceleration Validation");
    println!("─".repeat(60));

    let n = 100;
    let (k, f) = create_test_system(n);
    let ref_sol = k.clone().lu().solve(&f).unwrap();

    // Estimate eigenvalues
    let (lam_min, lam_max) = ChebyshevSemiIterative::estimate_eigenvalues_gershgorin(&k);
    println!("  Estimated eigenvalue range: [{:.4}, {:.4}]", lam_min, lam_max);

    // Test Chebyshev semi-iterative
    let mut cheb = ChebyshevSemiIterative::new(lam_min, lam_max);
    let mut x = DVector::zeros(n);
    let mut max_iter = 200;

    for iter in 0..max_iter {
        cheb.iterate(&mut x, &k, &f);
        let residual = (f - &k * &x).norm();
        if residual < 1e-10 * f.norm() {
            max_iter = iter + 1;
            break;
        }
    }

    let error = (x - &ref_sol).norm() / ref_sol.norm();

    Ok(ValidationTest {
        name: "Chebyshev Semi-Iterative".to_string(),
        passed: error < 1e-6,
        error,
        tolerance: 1e-6,
        message: format!("Converged in {} iterations", max_iter),
    })
}

/// Validates Krylov recycling methods.
fn run_recycling_validation() -> anyhow::Result<ValidationTest> {
    println!("\n─".repeat(60));
    println!("Krylov Recycling Validation");
    println!("─".repeat(60));

    let n = 100;
    let (k, f) = create_test_system(n);
    let ref_sol = k.clone().lu().solve(&f).unwrap();

    // Test GCRO-DR
    let gcro = GCRODRSolver::new(20, 5);
    let (x, iters, residual, _conv) = gcro.solve(&k, &f, None, 1e-10, 300);

    let error = (x - &ref_sol).norm() / ref_sol.norm();

    println!("  GCRO-DR iterations: {}", iters);
    println!("  Final residual: {:.2e}", residual);

    Ok(ValidationTest {
        name: "GCRO-DR Solver".to_string(),
        passed: error < 1e-6,
        error,
        tolerance: 1e-6,
        message: format!("Residual: {:.2e}", residual),
    })
}

/// Validates adaptive solver methods.
fn run_adaptive_validation() -> anyhow::Result<ValidationTest> {
    println!("\n─".repeat(60));
    println!("Adaptive Solver Validation");
    println!("─".repeat(60));

    let n = 100;
    let (k, f) = create_test_system(n);
    let ref_sol = k.clone().lu().solve(&f).unwrap();

    // Test adaptive CG
    let mut solver = AdaptiveCGSolver::new(1e-10, 500);
    let result = solver.solve(&k, &f);

    let sol = result.solution.local_data;
    let x = DVector::from_column_slice(&sol);
    let error = (x - &ref_sol).norm() / ref_sol.norm();

    println!("  Adaptive CG iterations: {}", result.iterations);
    println!("  Strategy changes: {}", result.strategy_changes);

    Ok(ValidationTest {
        name: "Adaptive CG Solver".to_string(),
        passed: error < 1e-8,
        error,
        tolerance: 1e-8,
        message: format!("{} strategy changes", result.strategy_changes),
    })
}

/// Validates block iterative solvers.
fn run_block_validation() -> anyhow::Result<ValidationTest> {
    println!("\n─".repeat(60));
    println!("Block Solver Validation");
    println!("─".repeat(60));

    let n = 100;
    let (k, f) = create_test_system(n);
    let ref_sol = k.clone().lu().solve(&f).unwrap();

    // Test Block Jacobi
    let solver = BlockJacobiIterative::new(4);
    let (x, iters, residual, _conv) = solver.solve(&k, &f);

    let error = (x - &ref_sol).norm() / ref_sol.norm();

    println!("  Block Jacobi iterations: {}", iters);
    println!("  Final residual: {:.2e}", residual);

    Ok(ValidationTest {
        name: "Block Jacobi Solver".to_string(),
        passed: error < 1e-6,
        error,
        tolerance: 1e-6,
        message: format!("Residual: {:.2e}", residual),
    })
}

/// Validates unified framework.
fn run_unified_validation() -> anyhow::Result<ValidationTest> {
    println!("\n─".repeat(60));
    println!("Unified Framework Validation");
    println!("─".repeat(60));

    let n = 100;
    let (k, f) = create_test_system(n);
    let ref_sol = k.clone().lu().solve(&f).unwrap();

    // Test unified solver with adaptive strategy
    let config = UnifiedSolverConfig {
        strategy: AccelerationStrategy::Adaptive,
        max_iterations: 500,
        tolerance: 1e-10,
        enable_monitoring: true,
        verbose: false,
    };

    let mut solver = UnifiedAccelerationSolver::new(config);
    let result = solver.solve(&k, &f);

    let sol = DVector::from_column_slice(&result.base_result.solution);
    let error = (sol - &ref_sol).norm() / ref_sol.norm();

    println!("  Unified solver iterations: {}", result.base_result.iterations.unwrap_or(0));
    println!("  MatVec count: {}", result.matvec_count);

    Ok(ValidationTest {
        name: "Unified Adaptive Solver".to_string(),
        passed: error < 1e-8,
        error,
        tolerance: 1e-8,
        message: format!("MatVec: {}", result.matvec_count),
    })
}

/// Creates a test linear system.
fn create_test_system(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut k = DMatrix::zeros(n, n);
    for i in 0..n {
        k[(i, i)] = 4.0;
        if i > 0 { k[(i, i - 1)] = -1.0; }
        if i < n - 1 { k[(i, i + 1)] = -1.0; }
    }
    let f = DVector::from_element(n, 1.0);
    (k, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_suite() {
        let mut results = ValidationResults::new();

        results.add(run_cg_validation().unwrap());
        results.add(run_preconditioner_validation().unwrap());
        results.add(run_spectral_validation().unwrap());
        results.add(run_recycling_validation().unwrap());
        results.add(run_adaptive_validation().unwrap());
        results.add(run_block_validation().unwrap());
        results.add(run_unified_validation().unwrap());

        assert!(results.all_passed());
    }

    #[test]
    fn test_create_test_system() {
        let (k, f) = create_test_system(10);
        assert_eq!(k.nrows(), 10);
        assert_eq!(f.len(), 10);
    }
}
