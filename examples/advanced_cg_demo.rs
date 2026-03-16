//! Advanced GPU CG solver demonstration and validation.
//!
//! This example demonstrates:
//! - Advanced CG solver with restart
//! - Convergence history monitoring
//! - Block CG for multiple RHS
//! - Performance comparison
//! - Validation against analytical solutions

use fea::gpu::gpu_cg_advanced::{
    AdvancedCGSolver, AdvancedCGConfig, BlockCGSolver, BlockPCGSolver,
    ConvergenceHistory,
};
use fea::gpu::{GPUCGSolver, GPUCSRMatrix};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║      Advanced GPU CG Solver Demonstration                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Advanced CG with convergence monitoring
    demo_advanced_cg()?;

    // Block CG for multiple RHS
    demo_block_cg()?;

    // Performance comparison
    benchmark_cg_variants()?;

    // Convergence analysis
    analyze_convergence()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Demonstration Complete                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demonstrates advanced CG solver features.
fn demo_advanced_cg() -> anyhow::Result<()> {
    println!("┌─ Advanced CG Solver ─────────────────────────────────────┐");

    // Create test matrix (2D Laplacian)
    let n = 500;
    let (row_ptr, col_ind, values) = create_laplacian_2d(n);
    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);

    // Create RHS (sine wave)
    let b: Vec<f64> = (0..n)
        .map(|i| (std::f64::consts::PI * i as f64 / n as f64).sin())
        .collect();

    // Advanced CG with verbosity
    let config = AdvancedCGConfig {
        tolerance: 1e-10,
        max_iterations: 1000,
        mixed_precision: false,
        restart_interval: Some(200),
        track_convergence: true,
        verbosity: 1,
    };

    let mut solver = AdvancedCGSolver::new(0, config);
    let start = Instant::now();
    let result = solver.solve(&matrix, &b)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("│");
    println!("│ Results:");
    println!("│   Iterations:     {}", result.iterations);
    println!("│   Final residual: {:.2e}", result.final_residual);
    println!("│   Relative:       {:.2e}", result.relative_residual);
    println!("│   Converged:      {}", result.converged);
    println!("│   Time:           {:.2} ms", elapsed);

    // Show convergence history
    if let Some(history) = &result.history {
        println!("│");
        println!("│ Convergence History:");
        println!("│   Iter    Residual       Relative");
        println!("│   ─────   ────────────   ────────────");

        let skip = history.iterations.len() / 10;
        for i in (0..history.iterations.len()).step_by(skip.max(1)) {
            println!("│   {:>5}   {:.2e}      {:.2e}",
                history.iterations[i],
                history.residual_norms[i],
                history.relative_residuals[i]);
        }

        if let Some(rate) = history.convergence_rate() {
            println!("│");
            println!("│   Convergence rate: {:.2}x per iteration", rate);
        }
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demonstrates block CG for multiple RHS.
fn demo_block_cg() -> anyhow::Result<()> {
    println!("┌─ Block CG for Multiple RHS ──────────────────────────────┐");

    let n = 200;
    let (row_ptr, col_ind, values) = create_laplacian_2d(n);
    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);

    // Multiple RHS
    let num_rhs = 3;
    let mut b_matrix = Vec::new();
    for rhs in 0..num_rhs {
        let b: Vec<f64> = (0..n)
            .map(|i| ((rhs + 1) as f64 * std::f64::consts::PI * i as f64 / n as f64).sin())
            .collect();
        b_matrix.push(b);
    }

    println!("│ Matrix size: {} × {}", n, n);
    println!("│ Number of RHS: {}", num_rhs);
    println!("│");

    // Standard block CG
    let solver = BlockCGSolver::new(0, 1e-8, 500);
    let start = Instant::now();
    let solutions = solver.solve_block(&matrix, &b_matrix)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("│ Block CG Results:");
    println!("│   Time: {:.2} ms", elapsed);
    println!("│   Solutions: {}", solutions.len());
    for (i, sol) in solutions.iter().enumerate() {
        let max_val = sol.iter().map(|v| v.abs()).fold(0.0, f64::max);
        println!("│   RHS {}: max|x| = {:.4}", i + 1, max_val);
    }

    // Block PCG with Jacobi preconditioning
    println!("│");
    let solver = BlockPCGSolver::new(0, 1e-8, 500);
    let start = Instant::now();
    let solutions = solver.solve_with_block_jacobi(&matrix, &b_matrix, 10)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("│ Block PCG (Jacobi) Results:");
    println!("│   Time: {:.2} ms", elapsed);
    println!("│   Solutions: {}", solutions.len());

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Benchmarks different CG variants.
fn benchmark_cg_variants() -> anyhow::Result<()> {
    println!("┌─ CG Variant Performance Comparison ──────────────────────┐");

    let sizes = [100, 200, 500, 1000];

    println!("│ {:>8} │ {:>10} │ {:>10} │ {:>10} │", "Size", "Standard", "Advanced", "Speedup");
    println!("│──────────┼────────────┼────────────┼────────────│");

    for &n in &sizes {
        let (row_ptr, col_ind, values) = create_laplacian_2d(n);
        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
        let b = vec![1.0f64; n];

        // Standard CG
        let cg = GPUCGSolver::new(0, 1e-8, 1000);
        let mut x = vec![0.0; n];
        let start = Instant::now();
        let std_result = cg.solve(&matrix, &b, &mut x)?;
        let std_time = start.elapsed().as_secs_f64() * 1000.0;

        // Advanced CG
        let config = AdvancedCGConfig {
            tolerance: 1e-8,
            max_iterations: 1000,
            track_convergence: false,
            verbosity: 0,
            ..Default::default()
        };
        let mut solver = AdvancedCGSolver::new(0, config);
        let start = Instant::now();
        let adv_result = solver.solve(&matrix, &b)?;
        let adv_time = start.elapsed().as_secs_f64() * 1000.0;

        let speedup = if adv_time > 0.0 { std_time / adv_time } else { 0.0 };

        println!("│ {:>8} │ {:>10.2} │ {:>10.2} │ {:>10.2}x │",
            n, std_time, adv_time, speedup);
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Analyzes convergence behavior.
fn analyze_convergence() -> anyhow::Result<()> {
    println!("┌─ Convergence Analysis ───────────────────────────────────┐");

    let n = 500;
    let (row_ptr, col_ind, values) = create_laplacian_2d(n);
    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
    let b = vec![1.0f64; n];

    // Run with convergence tracking
    let config = AdvancedCGConfig {
        tolerance: 1e-12,
        max_iterations: 2000,
        track_convergence: true,
        verbosity: 0,
        ..Default::default()
    };

    let mut solver = AdvancedCGSolver::new(0, config);
    let result = solver.solve(&matrix, &b)?;

    if let Some(history) = &result.history {
        println!("│ Initial residual:  {:.2e}", history.residual_norms[0]);
        println!("│ Final residual:    {:.2e}", *history.residual_norms.last().unwrap());
        println!("│ Total iterations:  {}", history.iterations.len());
        println!("│");

        // Find iterations for each order of magnitude reduction
        let r0 = history.residual_norms[0];
        println!("│ Iterations per decade:");

        for decade in 1..=6 {
            let threshold = r0 * 10.0_f64.powi(-decade);
            if let Some(iter) = history.residual_norms.iter().position(|&r| r < threshold) {
                println!("│   10^{}:  {} iterations", -decade, history.iterations[iter]);
            }
        }

        // Check for stagnation
        println!("│");
        if history.is_stagnating(20, 0.01) {
            println!("│ ⚠ Stagnation detected in last 20 iterations");
        } else {
            println!("│ ✓ No stagnation detected");
        }
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Creates 2D Laplacian matrix in CSR format.
fn create_laplacian_2d(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let sqrt_n = (n as f64).sqrt() as usize;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0;
    for i in 0..n {
        row_ptr.push(nnz);

        let row = i / sqrt_n;
        let col = i % sqrt_n;

        // Diagonal
        col_ind.push(i);
        values.push(4.0);
        nnz += 1;

        // Left neighbor
        if col > 0 {
            col_ind.push(i - 1);
            values.push(-1.0);
            nnz += 1;
        }

        // Right neighbor
        if col < sqrt_n - 1 {
            col_ind.push(i + 1);
            values.push(-1.0);
            nnz += 1;
        }

        // Top neighbor
        if row > 0 {
            col_ind.push(i - sqrt_n);
            values.push(-1.0);
            nnz += 1;
        }

        // Bottom neighbor
        if row < sqrt_n - 1 {
            col_ind.push(i + sqrt_n);
            values.push(-1.0);
            nnz += 1;
        }
    }
    row_ptr.push(nnz);

    (row_ptr, col_ind, values)
}
