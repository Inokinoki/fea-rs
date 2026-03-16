//! Acceleration Methods Reference Guide with Examples.
//!
//! This example provides a comprehensive reference guide for all acceleration
//! methods available in the FEA library, with practical examples and usage patterns.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║      FEA Library - Acceleration Methods Reference        ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Part 1: Classical Acceleration Methods
    run_classical_methods()?;

    // Part 2: Advanced Preconditioning
    run_advanced_preconditioning()?;

    // Part 3: Spectral Methods
    run_spectral_methods()?;

    // Part 4: Recycling Methods
    run_recycling_methods()?;

    // Part 5: Adaptive Methods
    run_adaptive_methods()?;

    // Part 6: Mixed Precision
    run_mixed_precision()?;

    // Part 7: Performance Comparison
    run_performance_comparison()?;

    println!("\n=== Reference Guide Complete ===");
    println!("\nFor detailed API documentation, see the crate documentation.");
    println!("For benchmark results, see the benchmark examples.");

    Ok(())
}

/// Demonstrates classical acceleration methods.
fn run_classical_methods() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Part 1: Classical Acceleration Methods");
    println!("{}\n", "=".repeat(60));

    // Create test problem
    let (k, f) = create_test_problem(200);

    println!("Test problem: {} DOFs\n", k.nrows());

    // Method 1: CG + Jacobi (baseline)
    println!("1. CG + Jacobi Preconditioning");
    println!("   -----------------------------");
    let cfg = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-10,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };
    let (iters, time_ms, converged) = run_solver(&k, &f, cfg);
    println!("   Iterations: {}", iters);
    println!("   Time: {:.4} ms", time_ms);
    println!("   Converged: {}\n", converged);

    // Method 2: CG + Chebyshev
    println!("2. CG + Chebyshev Preconditioning");
    println!("   -------------------------------");
    let cfg = IterativeConfig {
        preconditioner: Preconditioner::Chebyshev(3),
        ..cfg.clone()
    };
    let (iters, time_ms, converged) = run_solver(&k, &f, cfg);
    println!("   Polynomial degree: 3");
    println!("   Iterations: {}", iters);
    println!("   Time: {:.4} ms", time_ms);
    println!("   Converged: {}\n", converged);

    // Method 3: Anderson Acceleration
    println!("3. Anderson Acceleration");
    println!("   ----------------------");
    for depth in [3, 5, 10] {
        let cfg = IterativeConfig {
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: depth,
            ..cfg.clone()
        };
        let (iters, time_ms, converged) = run_solver(&k, &f, cfg);
        println!("   Depth {}: {} iterations, {:.4} ms, converged: {}",
                 depth, iters, time_ms, converged);
    }
    println!();

    // Method 4: BiCGSTAB
    println!("4. BiCGSTAB Solver");
    println!("   ----------------");
    let solver = BiCGSTABSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = solver.solve(&k, &f, &IterativeConfig::default())?;
    let elapsed = start.elapsed();
    println!("   Iterations: {}", result.iterations.unwrap_or(0));
    println!("   Time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("   Converged: {}\n", result.converged);

    Ok(())
}

/// Demonstrates advanced preconditioning.
fn run_advanced_preconditioning() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Part 2: Advanced Preconditioning");
    println!("{}\n", "=".repeat(60));

    let (k, f) = create_test_problem(100);

    // FSAI Preconditioner
    println!("1. FSAI Preconditioner");
    println!("   --------------------");
    let pattern: Vec<Vec<usize>> = (0..k.nrows()).map(|i| {
        let mut p = vec![i];
        if i > 0 { p.push(i - 1); }
        if i < k.nrows() - 1 { p.push(i + 1); }
        p
    }).collect();

    if let Some(fsai) = FSAIPreconditioner::new(&k, pattern) {
        let start = std::time::Instant::now();
        let z = fsai.apply(&f);
        let elapsed = start.elapsed();
        println!("   Apply time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
        println!("   Result norm: {:.6e}", z.norm());
    }
    println!();

    // Elasticity Preconditioner
    println!("2. Elasticity Preconditioner");
    println!("   --------------------------");
    let prec = ElasticityPreconditioner::new(&k, 200e9);
    let start = std::time::Instant::now();
    let z = prec.apply(&f);
    let elapsed = start.elapsed();
    println!("   Apply time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("   Result norm: {:.6e}", z.norm());
    println!();

    // Additive Schwarz Multilevel
    println!("3. Additive Schwarz Multilevel");
    println!("   ----------------------------");
    let asm = AdditiveSchwarzMultilevel::new(&k, 4);
    let start = std::time::Instant::now();
    let corr = asm.apply_vcycle(&f);
    let elapsed = start.elapsed();
    println!("   Levels: {}", asm.levels);
    println!("   V-cycle time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("   Correction norm: {:.6e}", corr.norm());
    println!();

    Ok(())
}

/// Demonstrates spectral methods.
fn run_spectral_methods() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Part 3: Spectral Acceleration Methods");
    println!("{}\n", "=".repeat(60));

    let (k, f) = create_test_problem(100);

    // Eigenvalue estimation
    println!("1. Eigenvalue Estimation");
    println!("   ----------------------");
    let (lam_min, lam_max) = ChebyshevSemiIterative::estimate_eigenvalues_gershgorin(&k);
    println!("   Gershgorin estimate: [{:.4}, {:.4}]", lam_min, lam_max);

    let (lam_min_l, lam_max_l) = ChebyshevSemiIterative::estimate_eigenvalues_lanczos(&k, 30);
    println!("   Lanczos estimate: [{:.4}, {:.4}]", lam_min_l, lam_max_l);
    println!();

    // Spectral Deflation
    println!("2. Spectral Deflation");
    println!("   -------------------");
    let deflation = SpectralDeflation::compute_deflation_subspace(&k, 5, 100);
    println!("   Deflation vectors: {}", deflation.eigenvectors.ncols());
    println!("   Smallest eigenvalue: {:.6e}", deflation.eigenvalues.min());

    let r = DVector::from_element(100, 1.0);
    let r_deflated = deflation.apply(&r);
    println!("   Deflated residual norm: {:.6e}", r_deflated.norm());
    println!();

    // Rational Chebyshev Filter
    println!("3. Rational Chebyshev Filter");
    println!("   --------------------------");
    let filter = RationalChebyshevFilter::new(lam_min_l, (lam_max_l - lam_min_l) / 2.0, 4);
    let start = std::time::Instant::now();
    let filtered = filter.apply(&k, &f);
    let elapsed = start.elapsed();
    println!("   Filter order: 4");
    println!("   Apply time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("   Filtered norm: {:.6e}", filtered.norm());
    println!();

    Ok(())
}

/// Demonstrates recycling methods.
fn run_recycling_methods() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Part 4: Krylov Recycling Methods");
    println!("{}\n", "=".repeat(60));

    let (k, f) = create_test_problem(100);

    // GCRO-DR
    println!("1. GCRO-DR (GMRES with Deflated Restarting)");
    println!("   -----------------------------------------");
    let gcro = GCRODRSolver::new(20, 5);
    let result = gcro.solve(&k, &f, None, 1e-10, 300)?;
    println!("   Iterations: {}", result.iterations.unwrap_or(0));
    println!("   Residual: {:.6e}", result.residual_norm.unwrap_or(f64::INFINITY));
    println!("   Converged: {}", result.converged);
    println!();

    // Recycling BiCGSTAB
    println!("2. Recycling BiCGSTAB");
    println!("   -------------------");
    let mut rbicg = RecyclingBiCGSTAB::new(10);
    let result = rbicg.solve(&k, &f, 1e-10, 300);
    println!("   Iterations: {}", result.1);
    println!("   Residual: {:.6e}", result.2);
    println!("   Converged: {}", result.3);
    println!();

    // Deflated CG (Advanced Krylov)
    println!("3. Deflated CG (Advanced Krylov)");
    println!("   -------------------------------");
    let z = DeflatedCGAdvanced::generate_coarse_deflation(&k, 5);
    let dcg = DeflatedCGAdvanced::new(z);
    let result = dcg.solve(&k, &f);
    println!("   Iterations: {}", result.1);
    println!("   Residual: {:.6e}", result.2);
    println!();

    Ok(())
}

/// Demonstrates adaptive methods.
fn run_adaptive_methods() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Part 5: Adaptive Acceleration Methods");
    println!("{}\n", "=".repeat(60));

    let (k, f) = create_test_problem(100);

    // Unified Adaptive Solver
    println!("1. Unified Adaptive Solver");
    println!("   ------------------------");
    let config = UnifiedSolverConfig {
        strategy: AccelerationStrategy::Adaptive,
        max_iterations: 500,
        tolerance: 1e-10,
        enable_monitoring: true,
        verbose: false,
    };
    let mut solver = UnifiedAccelerationSolver::new(config);
    let result = solver.solve(&k, &f);
    println!("   Iterations: {}", result.base_result.iterations.unwrap_or(0));
    println!("   Residual: {:.6e}", result.base_result.residual_norm.unwrap_or(f64::INFINITY));
    println!("   Converged: {}", result.base_result.converged);

    if let Some(rec) = solver.get_recommended_strategy() {
        println!("   Recommended strategy: {:?}", rec);
    }
    println!();

    // Accelerated PCG
    println!("2. Accelerated PCG");
    println!("   ----------------");
    let config = AcceleratedPCGConfig::ill_conditioned();
    let mut solver = AcceleratedPCG::new(config);
    let result = solver.solve(&k, &f);
    println!("   Iterations: {}", result.iterations);
    println!("   Anderson steps: {}", result.anderson_steps);
    println!("   Deflation updates: {}", result.deflation_updates);
    println!("   Converged: {}", result.converged);
    if let Some(cond) = result.estimated_condition {
        println!("   Estimated condition: {:.2e}", cond);
    }
    println!();

    Ok(())
}

/// Demonstrates mixed precision methods.
fn run_mixed_precision() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Part 6: Mixed Precision Methods");
    println!("{}\n", "=".repeat(60));

    let (k, f) = create_test_problem(100);

    println!("1. Mixed Precision Iterative Refinement");
    println!("   -------------------------------------");
    let config = MixedPrecisionConfig {
        inner_tolerance: 1e-4,
        outer_tolerance: 1e-12,
        max_refinement_steps: 20,
        max_inner_iterations: 50,
        use_simulated_fp16: false,
    };
    let solver = MixedPrecisionSolver::new(config);
    let result = solver.solve(&k, &f);
    println!("   Refinement steps: {}", result.refinement_steps);
    println!("   Inner iterations: {}", result.total_inner_iterations);
    println!("   Final residual: {:.6e}", result.residual_norm);
    println!("   Converged: {}", result.converged);
    println!();

    // FP16 utilities
    println!("2. FP16 Utilities");
    println!("   ---------------");
    let value = 3.14159265358979;
    let fp16_value = fp16_utils::to_fp16(value);
    println!("   Original: {:.15}", value);
    println!("   FP16 simulated: {:.15}", fp16_value);
    println!("   Precision loss: {:.2e}", (value - fp16_value).abs());
    println!();

    Ok(())
}

/// Runs performance comparison.
fn run_performance_comparison() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Part 7: Performance Comparison Summary");
    println!("{}\n", "=".repeat(60));

    println!("{:<30} | {:>10} | {:>12} | {:>10}",
             "Method", "Iterations", "Time (ms)", "Status");
    println!("{}", "-".repeat(70));

    let test_sizes = [100, 200, 500];
    for &size in &test_sizes {
        let (k, f) = create_test_problem(size);
        let ref_sol = k.clone().lu().solve(&f).unwrap();

        // Direct solve
        let start = std::time::Instant::now();
        let _ = k.clone().lu().solve(&f);
        let direct_time = start.elapsed();

        // CG + Jacobi
        let cfg = IterativeConfig::default();
        let cg = CGSolver::with_config(cfg.clone());
        let start = std::time::Instant::now();
        let result = cg.solve(&k, &f, &cfg)?;
        let cg_time = start.elapsed();

        let error = (DVector::from_column_slice(&result.solution) - ref_sol).norm() / ref_sol.norm();

        println!("{:<30} | {:>10} | {:>12.4} | {:>10}",
                 format!("CG+Jacobi (n={})", size),
                 result.iterations.unwrap_or(0),
                 cg_time.as_secs_f64() * 1000.0,
                 format!("{:.0e}", error));
    }

    println!();

    Ok(())
}

/// Creates a test problem (SPD matrix).
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

/// Runs a solver and returns metrics.
fn run_solver(
    k: &DMatrix<f64>,
    f: &DVector<f64>,
    config: IterativeConfig,
) -> (usize, f64, bool) {
    let cg = CGSolver::with_config(config.clone());
    let start = std::time::Instant::now();
    let result = cg.solve(k, f, &config).unwrap_or_else(|_| SolverResult {
        solution: vec![0.0; f.len()],
        iterations: Some(0),
        residual_norm: Some(f64::INFINITY),
        converged: false,
    });
    let elapsed = start.elapsed();

    (
        result.iterations.unwrap_or(0),
        elapsed.as_secs_f64() * 1000.0,
        result.converged,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_guide() {
        // Run all parts of the reference guide
        run_classical_methods().unwrap();
        run_advanced_preconditioning().unwrap();
        run_spectral_methods().unwrap();
        run_recycling_methods().unwrap();
        run_adaptive_methods().unwrap();
        run_mixed_precision().unwrap();
        run_performance_comparison().unwrap();
    }
}
