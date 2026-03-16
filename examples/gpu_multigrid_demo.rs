//! GPU Multigrid Solver demonstration.
//!
//! This example demonstrates:
//! - V-cycle, W-cycle, and F-cycle multigrid
//! - Full Multigrid (FMG) initialization
//! - Performance comparison with CG
//! - Scaling analysis

use fea::gpu::gpu_multigrid::{
    GPUMultigrid, MultigridCycle, MultigridResult,
    create_1d_poisson_hierarchy, run_multigrid_demo,
};
use fea::gpu::{GPUCGSolver, GPUCSRMatrix};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║          GPU Multigrid Solver Demonstration               ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Built-in demo
    run_multigrid_demo()?;

    // Multigrid vs CG comparison
    compare_multigrid_cg()?;

    // Scaling analysis
    scaling_analysis()?;

    // Cycle type comparison
    compare_cycle_types()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Demonstration Complete                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Compare multigrid with CG.
fn compare_multigrid_cg() -> anyhow::Result<()> {
    println!("┌─ Multigrid vs CG Comparison ─────────────────────────────┐");

    let sizes = [63, 127, 255, 511];

    println!("│ {:>8} │ {:>10} │ {:>12} │ {:>12} │ {:>8} │", "Size", "CG", "MG V-cyc", "Speedup", "Optimal");
    println!("│──────────┼────────────┼──────────────┼──────────────┼──────────│");

    for &n in &sizes {
        // Create 1D Poisson matrix
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_ind = Vec::new();
        let mut values = Vec::new();

        let mut nnz = 0;
        for i in 0..n {
            row_ptr.push(nnz);

            if i > 0 { col_ind.push(i - 1); values.push(-1.0); nnz += 1; }
            col_ind.push(i); values.push(2.0); nnz += 1;
            if i < n - 1 { col_ind.push(i + 1); values.push(-1.0); nnz += 1; }
        }
        row_ptr.push(nnz);

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
        let b = vec![1.0f64; n];

        // CG solve
        let cg = GPUCGSolver::new(0, 1e-10, 1000);
        let mut x = vec![0.0f64; n];
        let start = Instant::now();
        let cg_result = cg.solve(&matrix, &b, &mut x)?;
        let cg_time = start.elapsed();

        // Multigrid solve
        let levels = create_1d_poisson_hierarchy(n, 0);
        let mg = GPUMultigrid::new(0, levels)
            .with_cycle_type(MultigridCycle::VCycle)
            .with_smoothing(2, 2);

        let start = Instant::now();
        let mg_result = mg.solve(&b)?;
        let mg_time = start.elapsed();

        let speedup = cg_time.as_secs_f64() / mg_time.as_secs_f64();

        // Optimal would be O(N) - MG should achieve this
        let optimal = mg_time.as_secs_f64() < cg_time.as_secs_f64() * 0.5;

        println!("│ {:>8} │ {:>10} │ {:>12.2} │ {:>12.2}x │ {:>8} │",
            n, cg_result.iterations, mg_result.time_ms, speedup,
            if optimal { "Yes" } else { "No" });
    }

    println!("│");
    println!("│ Multigrid achieves O(N) complexity vs O(N^1.5) for CG");
    println!("│   Speedup increases with problem size");
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Scaling analysis.
fn scaling_analysis() -> anyhow::Result<()> {
    println!("┌─ Multigrid Scaling Analysis ─────────────────────────────┐");

    let sizes = [31, 63, 127, 255, 511];

    println!("│ {:>8} │ {:>12} │ {:>12} │ {:>12} │", "Size", "Time (ms)", "Time/N", "O(N)?");
    println!("│──────────┼──────────────┼──────────────┼──────────────│");

    let mut prev_time_per_n = 0.0;

    for &n in &sizes {
        let levels = create_1d_poisson_hierarchy(n, 0);
        let mg = GPUMultigrid::new(0, levels)
            .with_cycle_type(MultigridCycle::FCycle)
            .with_smoothing(1, 1);

        let b = vec![1.0f64; n];

        let start = Instant::now();
        let _ = mg.solve(&b)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let time_per_n = elapsed / n as f64;
        let ratio = if prev_time_per_n > 0.0 { time_per_n / prev_time_per_n } else { 1.0 };

        // O(N) means time_per_n should be constant (ratio ~ 1.0)
        let is_linear = (0.8..=1.2).contains(&ratio);

        println!("│ {:>8} │ {:>12.2} │ {:>12.2e} │ {:>12} │",
            n, elapsed, time_per_n,
            if prev_time_per_n > 0.0 { if is_linear { "Yes" } else { "No" } } else { "N/A" });

        prev_time_per_n = time_per_n;
    }

    println!("│");
    println!("│ F-cycle (FMG) achieves optimal O(N) complexity");
    println!("│   Time per unknown remains constant as N increases");
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Compare different cycle types.
fn compare_cycle_types() -> anyhow::Result<()> {
    println!("┌─ Cycle Type Comparison ──────────────────────────────────┐");

    let n = 511;
    let levels = create_1d_poisson_hierarchy(n, 0);
    let b = vec![1.0f64; n];

    println!("│ Problem size: {}", n);
    println!("│");

    let cycles = [
        (MultigridCycle::VCycle, "V-cycle"),
        (MultigridCycle::WCycle, "W-cycle"),
        (MultigridCycle::FCycle, "F-cycle (FMG)"),
    ];

    println!("│ {:>20} │ {:>12} │ {:>12} │", "Cycle Type", "Time (ms)", "Work Units");
    println!("│──────────────────────┼──────────────┼──────────────│");

    let mut best_time = f64::INFINITY;
    let mut best_name = "";

    for (cycle_type, name) in &cycles {
        let mg = GPUMultigrid::new(0, levels.clone())
            .with_cycle_type(*cycle_type)
            .with_smoothing(1, 1);

        let start = Instant::now();
        let result = mg.solve(&b)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        // Work units (approximate)
        let work_units = match cycle_type {
            MultigridCycle::VCycle => 1.0,
            MultigridCycle::WCycle => 2.0,
            MultigridCycle::FCycle => 1.5,
        };

        println!("│ {:>20} │ {:>12.2} │ {:>12.1} │", name, elapsed, work_units);

        if elapsed < best_time {
            best_time = elapsed;
            best_name = name;
        }
    }

    println!("│");
    println!("│ Best performer: {}", best_name);
    println!("│");
    println!("│ V-cycle:  Fastest per iteration, may need more iterations");
    println!("│ W-cycle:  More expensive per iteration, better convergence");
    println!("│ F-cycle:  Best overall for elliptic problems (FMG init)");
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_multigrid_cg() {
        assert!(compare_multigrid_cg().is_ok());
    }

    #[test]
    fn test_scaling_analysis() {
        assert!(scaling_analysis().is_ok());
    }

    #[test]
    fn test_compare_cycle_types() {
        assert!(compare_cycle_types().is_ok());
    }
}
