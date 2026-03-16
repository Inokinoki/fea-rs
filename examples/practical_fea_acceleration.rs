//! Practical FEA Acceleration Examples.
//!
//! This example demonstrates how to use acceleration methods in practical
//! FEA applications, with real-world problem setups and best practices.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       Practical FEA Acceleration Examples                ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Example 1: Solving large structural system
    example_structural_analysis()?;

    // Example 2: Eigenvalue analysis for modal superposition
    example_modal_analysis()?;

    // Example 3: Ill-conditioned heat conduction
    example_heat_conduction()?;

    // Example 4: Dynamic response with reduced basis
    example_dynamic_reduction()?;

    // Example 5: Multi-load case with recycling
    example_multi_load()?;

    println!("\n=== All Examples Complete ===");
    Ok(())
}

/// Example 1: Large structural system with accelerated PCG.
fn example_structural_analysis() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 1: Large Structural Analysis");
    println!("{}\n", "=".repeat(60));

    // Simulate a 2D plane stress problem (1000 DOFs)
    let size = 1000;
    let k = create_structural_stiffness(size);
    let f = DVector::from_fn(size, |i, _| {
        if i < size / 10 { 100.0 } else { 0.0 }
    });

    println!("Problem: 2D plane stress structure");
    println!("  DOFs: {}", size);
    println!("  Non-zero entries: ~{}", size * 9);
    println!();

    // Method 1: Standard CG + Jacobi
    println!("Method 1: CG + Jacobi");
    let start = Instant::now();
    let cfg = IterativeConfig {
        max_iterations: 1000,
        tolerance: 1e-8,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };
    let cg = CGSolver::with_config(cfg.clone());
    let result = cg.solve(&k, &f, &cfg)?;
    let elapsed = start.elapsed();
    println!("  Iterations: {}", result.iterations.unwrap_or(0));
    println!("  Time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!();

    // Method 2: Accelerated PCG
    println!("Method 2: Accelerated PCG");
    let start = Instant::now();
    let config = AcceleratedPCGConfig {
        max_iterations: 1000,
        tolerance: 1e-8,
        use_anderson: true,
        anderson_depth: 5,
        use_chebyshev: true,
        chebyshev_degree: 2,
        use_deflation: false,
        deflation_count: 0,
        restart_interval: 0,
    };
    let mut solver = AcceleratedPCG::new(config);
    let result = solver.solve(&k, &f);
    let elapsed = start.elapsed();
    println!("  Iterations: {}", result.iterations);
    println!("  Time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Anderson steps: {}", result.anderson_steps);
    println!();

    // Method 3: Unified adaptive solver
    println!("Method 3: Unified Adaptive");
    let start = Instant::now();
    let config = UnifiedSolverConfig {
        strategy: AccelerationStrategy::Adaptive,
        max_iterations: 1000,
        tolerance: 1e-8,
        enable_monitoring: true,
        verbose: false,
    };
    let mut solver = UnifiedAccelerationSolver::new(config);
    let result = solver.solve(&k, &f);
    let elapsed = start.elapsed();
    println!("  Iterations: {}", result.base_result.iterations.unwrap_or(0));
    println!("  Time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!();

    Ok(())
}

/// Example 2: Modal analysis with Krylov-Schur.
fn example_modal_analysis() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 2: Modal Analysis with Krylov-Schur");
    println!("{}\n", "=".repeat(60));

    // Create mass and stiffness matrices for a beam
    let size = 200;
    let k = create_beam_stiffness(size);
    let m = create_beam_mass(size);

    println!("Problem: Cantilever beam modal analysis");
    println!("  DOFs: {}", size);
    println!();

    // Solve generalized eigenvalue problem using shift-invert
    println!("Computing first 5 natural frequencies...");

    // For demonstration, solve standard eigenvalue problem Kx = λx
    // In practice, would solve (K - σM)x = λMx
    let config = KrylovSchurConfig {
        num_eigenvalues: 5,
        max_dimension: 30,
        max_iterations: 50,
        tolerance: 1e-8,
        keep_count: 15,
    };

    let solver = KrylovSchur::new(config);
    let start = Instant::now();
    let result = solver.solve(&k);
    let elapsed = start.elapsed();

    println!("  Computation time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Eigenvalues found: {}", result.eigenvalues.len());

    // Convert eigenvalues to frequencies (Hz)
    println!("\n  Natural frequencies (Hz):");
    for (i, lambda) in result.eigenvalues.iter().take(5).enumerate() {
        if *lambda > 0.0 {
            let freq = lambda.sqrt() / (2.0 * std::f64::consts::PI);
            println!("    Mode {}: {:.4} Hz", i + 1, freq);
        }
    }
    println!();

    Ok(())
}

/// Example 3: Ill-conditioned heat conduction.
fn example_heat_conduction() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 3: Ill-Conditioned Heat Conduction");
    println!("{}\n", "=".repeat(60));

    // Simulate heat conduction with high conductivity ratio
    let size = 500;
    let k = create_ill_conditioned_laplacian(size, 1000.0);
    let f = DVector::from_fn(size, |i, _| {
        if i == size / 2 { 1000.0 } else { 0.0 }
    });

    println!("Problem: Heat conduction with high conductivity ratio");
    println!("  DOFs: {}", size);
    println!("  Condition number: ~1000");
    println!();

    // Method 1: CG without preconditioning (will be slow)
    println!("Method 1: CG (no preconditioning)");
    let cfg = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-8,
        preconditioner: Preconditioner::None,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };
    let cg = CGSolver::with_config(cfg.clone());
    let start = Instant::now();
    let result = cg.solve(&k, &f, &cfg);
    let elapsed = start.elapsed();
    println!("  Iterations: {}", result.as_ref().ok().and_then(|r| r.iterations).unwrap_or(0));
    println!("  Time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!();

    // Method 2: CG with Chebyshev preconditioning
    println!("Method 2: CG + Chebyshev(3)");
    let cfg = IterativeConfig {
        preconditioner: Preconditioner::Chebyshev(3),
        ..cfg.clone()
    };
    let cg = CGSolver::with_config(cfg.clone());
    let start = Instant::now();
    let result = cg.solve(&k, &f, &cfg)?;
    let elapsed = start.elapsed();
    println!("  Iterations: {}", result.iterations.unwrap_or(0));
    println!("  Time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!();

    // Method 3: Mixed precision with iterative refinement
    println!("Method 3: Mixed Precision");
    let config = MixedPrecisionConfig {
        inner_tolerance: 1e-4,
        outer_tolerance: 1e-8,
        max_refinement_steps: 20,
        max_inner_iterations: 50,
        use_simulated_fp16: false,
    };
    let solver = MixedPrecisionSolver::new(config);
    let start = Instant::now();
    let result = solver.solve(&k, &f);
    let elapsed = start.elapsed();
    println!("  Refinement steps: {}", result.refinement_steps);
    println!("  Time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!();

    Ok(())
}

/// Example 4: Dynamic response with model reduction.
fn example_dynamic_reduction() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 4: Dynamic Response with Model Reduction");
    println!("{}\n", "=".repeat(60));

    // Create structural system
    let size = 300;
    let k = create_structural_stiffness(size);
    let m = create_structural_mass(size);

    println!("Problem: Dynamic response analysis");
    println!("  Original DOFs: {}", size);
    println!();

    // Compute reduced basis using IRAM
    println!("Computing reduced basis (10 modes)...");
    let config = IRAMConfig {
        num_eigenvalues: 10,
        subspace_dimension: 25,
        max_iterations: 50,
        tolerance: 1e-8,
    };

    let solver = IRAM::new(config);
    let start = Instant::now();
    let result = solver.solve(&k);
    let elapsed = start.elapsed();

    println!("  Computation time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Modes computed: {}", result.eigenvalues.len());

    // Project system onto reduced basis
    let phi = result.eigenvectors;
    let k_reduced = phi.transpose() * &k * &phi;
    let m_reduced = phi.transpose() * &m * &phi;

    println!("\n  Reduced system:");
    println!("    K_reduced DOFs: {}", k_reduced.nrows());
    println!("    M_reduced DOFs: {}", m_reduced.nrows());
    println!();

    Ok(())
}

/// Example 5: Multi-load case with Krylov recycling.
fn example_multi_load() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 5: Multi-Load Case with Recycling");
    println!("{}\n", "=".repeat(60));

    let size = 400;
    let k = create_structural_stiffness(size);

    // Multiple load cases
    let load_cases = vec![
        DVector::from_fn(size, |i, _| if i < size / 20 { 100.0 } else { 0.0 }),
        DVector::from_fn(size, |i, _| if i >= size / 2 && i < size / 2 + size / 20 { 100.0 } else { 0.0 }),
        DVector::from_fn(size, |i, _| if i >= size - size / 20 { 100.0 } else { 0.0 }),
    ];

    println!("Problem: Multiple load cases");
    println!("  DOFs: {}", size);
    println!("  Load cases: {}", load_cases.len());
    println!();

    // Solve with recycling
    println!("Solving with GCRO-DR (with recycling)...");
    let mut gcro = GCRODRSolver::new(20, 10);

    let start = Instant::now();
    for (i, f) in load_cases.iter().enumerate() {
        let result = gcro.solve(&k, f, None, 1e-8, 300)?;
        println!("  Load case {}: {} iterations", i + 1, result.iterations.unwrap_or(0));
    }
    let elapsed = start.elapsed();
    println!("  Total time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!();

    // Solve without recycling for comparison
    println!("Solving without recycling (standard CG)...");
    let cfg = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-8,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };

    let start = Instant::now();
    for (i, f) in load_cases.iter().enumerate() {
        let cg = CGSolver::with_config(cfg.clone());
        let result = cg.solve(&k, f, &cfg)?;
        println!("  Load case {}: {} iterations", i + 1, result.iterations.unwrap_or(0));
    }
    let elapsed = start.elapsed();
    println!("  Total time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!();

    Ok(())
}

/// Creates a structural stiffness matrix (tridiagonal with off-diagonals).
fn create_structural_stiffness(size: usize) -> DMatrix<f64> {
    let mut k = DMatrix::zeros(size, size);
    for i in 0..size {
        k[(i, i)] = 4.0;
        if i > 0 { k[(i, i - 1)] = -1.0; }
        if i < size - 1 { k[(i, i + 1)] = -1.0; }
        if i > 1 { k[(i, i - 2)] = -0.1; }
        if i < size - 1 { k[(i, i + 2)] = -0.1; }
    }
    k
}

/// Creates a beam stiffness matrix.
fn create_beam_stiffness(size: usize) -> DMatrix<f64> {
    let mut k = DMatrix::zeros(size, size);
    for i in 0..size {
        k[(i, i)] = (i + 1) as f64 * 10.0;
        if i > 0 {
            k[(i, i - 1)] = -(i as f64) * 5.0;
            k[(i - 1, i)] = -(i as f64) * 5.0;
        }
    }
    k
}

/// Creates a beam mass matrix (diagonal).
fn create_beam_mass(size: usize) -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_fn(size, |i, _| 1.0 + (i as f64) * 0.1))
}

/// Creates an ill-conditioned Laplacian.
fn create_ill_conditioned_laplacian(size: usize, ratio: f64) -> DMatrix<f64> {
    let mut k = DMatrix::zeros(size, size);
    for i in 0..size {
        let coef = if i < size / 2 { 1.0 } else { ratio };
        k[(i, i)] = coef * 2.0;
        if i > 0 { k[(i, i - 1)] = -coef; }
        if i < size - 1 { k[(i, i + 1)] = -coef; }
    }
    k
}

/// Creates a structural mass matrix.
fn create_structural_mass(size: usize) -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_element(size, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_practical_examples() {
        // Run a subset of examples for CI
        example_structural_analysis().unwrap();
        example_heat_conduction().unwrap();
    }
}
