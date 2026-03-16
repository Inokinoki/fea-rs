//! FEA Acceleration Quick Start Guide.
//!
//! This example provides a quick start guide for using acceleration methods
//! in the FEA library, with simple copy-paste examples.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         FEA Acceleration - Quick Start Guide            ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Create a simple test problem
    let (k, f) = create_test_problem(100);

    println!("Test problem: {} DOF structural system\n", k.nrows());

    // Quick Start 1: Basic CG with Jacobi preconditioning
    quick_start_basic(&k, &f)?;

    // Quick Start 2: Anderson acceleration
    quick_start_anderson(&k, &f)?;

    // Quick Start 3: Chebyshev preconditioning
    quick_start_chebyshev(&k, &f)?;

    // Quick Start 4: Unified adaptive solver
    quick_start_unified(&k, &f)?;

    // Quick Start 5: Accelerated PCG
    quick_start_accelerated_pcg(&k, &f)?;

    println!("\n=== Quick Start Complete ===");
    println!("\nFor more advanced examples, see:");
    println!("  - acceleration_reference_guide.rs");
    println!("  - practical_fea_acceleration.rs");
    println!("  - comprehensive_solver_comparison.rs");

    Ok(())
}

/// Quick Start 1: Basic CG with Jacobi preconditioning.
fn quick_start_basic(k: &DMatrix<f64>, f: &DVector<f64>) -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Quick Start 1: Basic CG + Jacobi");
    println!("─".repeat(60));

    // Create solver configuration
    let config = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-10,
        preconditioner: Preconditioner::Jacobi,  // Simple diagonal preconditioning
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };

    // Create and run solver
    let cg = CGSolver::with_config(config.clone());
    let result = cg.solve(k, f, &config)?;

    println!("Result:");
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations.unwrap_or(0));
    println!("  Residual: {:.2e}", result.residual_norm.unwrap_or(f64::INFINITY));
    println!();

    Ok(())
}

/// Quick Start 2: Anderson acceleration.
fn quick_start_anderson(k: &DMatrix<f64>, f: &DVector<f64>) -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Quick Start 2: Anderson Acceleration");
    println!("─".repeat(60));

    // Anderson acceleration is easy to enable - just set anderson_depth
    let config = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-10,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 5,  // Use 5 previous iterates for acceleration
        krylov_dim: 0,
        deflation_vectors: None,
    };

    let cg = CGSolver::with_config(config.clone());
    let result = cg.solve(k, f, &config)?;

    println!("Result (with Anderson depth=5):");
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations.unwrap_or(0));
    println!("  Residual: {:.2e}", result.residual_norm.unwrap_or(f64::INFINITY));
    println!();

    Ok(())
}

/// Quick Start 3: Chebyshev preconditioning.
fn quick_start_chebyshev(k: &DMatrix<f64>, f: &DVector<f64>) -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Quick Start 3: Chebyshev Preconditioning");
    println!("─".repeat(60));

    // Chebyshev preconditioning - specify polynomial degree
    let config = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-10,
        preconditioner: Preconditioner::Chebyshev(3),  // Degree 3 polynomial
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };

    let cg = CGSolver::with_config(config.clone());
    let result = cg.solve(k, f, &config)?;

    println!("Result (with Chebyshev degree=3):");
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations.unwrap_or(0));
    println!("  Residual: {:.2e}", result.residual_norm.unwrap_or(f64::INFINITY));
    println!();

    Ok(())
}

/// Quick Start 4: Unified adaptive solver.
fn quick_start_unified(k: &DMatrix<f64>, f: &DVector<f64>) -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Quick Start 4: Unified Adaptive Solver");
    println!("─".repeat(60));

    // The unified solver automatically selects the best acceleration strategy
    let config = UnifiedSolverConfig {
        strategy: AccelerationStrategy::Adaptive,  // Let the solver decide
        max_iterations: 500,
        tolerance: 1e-10,
        enable_monitoring: true,
        verbose: false,
    };

    let mut solver = UnifiedAccelerationSolver::new(config);
    let result = solver.solve(k, f);

    println!("Result (adaptive strategy):");
    println!("  Converged: {}", result.base_result.converged);
    println!("  Iterations: {}", result.base_result.iterations.unwrap_or(0));
    println!("  Residual: {:.2e}", result.base_result.residual_norm.unwrap_or(f64::INFINITY));

    // Get recommendation for future solves
    if let Some(rec) = solver.get_recommended_strategy() {
        println!("  Recommended: {:?}", rec);
    }
    println!();

    Ok(())
}

/// Quick Start 5: Accelerated PCG.
fn quick_start_accelerated_pcg(k: &DMatrix<f64>, f: &DVector<f64>) -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Quick Start 5: Accelerated PCG");
    println!("─".repeat(60));

    // Accelerated PCG combines multiple acceleration techniques
    let config = AcceleratedPCGConfig {
        max_iterations: 500,
        tolerance: 1e-10,
        use_anderson: true,       // Enable Anderson acceleration
        anderson_depth: 5,
        use_chebyshev: true,      // Enable Chebyshev preconditioning
        chebyshev_degree: 2,
        use_deflation: false,     // Deflation for very ill-conditioned problems
        deflation_count: 0,
        restart_interval: 0,
    };

    let mut solver = AcceleratedPCG::new(config);
    let result = solver.solve(k, f);

    println!("Result (accelerated PCG):");
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Residual: {:.2e}", result.residual_norm);
    println!("  Anderson steps: {}", result.anderson_steps);

    if let Some(cond) = result.estimated_condition {
        println!("  Estimated condition: {:.2e}", cond);
    }
    println!();

    Ok(())
}

/// Creates a simple test problem (1D Poisson equation).
fn create_test_problem(size: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut k = DMatrix::zeros(size, size);
    for i in 0..size {
        k[(i, i)] = 4.0;
        if i > 0 { k[(i, i - 1)] = -1.0; }
        if i < size - 1 { k[(i, i + 1)] = -1.0; }
    }
    let f = DVector::from_element(size, 1.0);
    (k, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_start() {
        let (k, f) = create_test_problem(50);
        quick_start_basic(&k, &f).unwrap();
        quick_start_anderson(&k, &f).unwrap();
        quick_start_chebyshev(&k, &f).unwrap();
    }
}
