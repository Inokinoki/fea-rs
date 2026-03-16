//! GPU Advanced Solvers demonstration.
//!
//! This example demonstrates:
//! - Mixed precision iterative refinement
//! - Flexible GMRES with variable preconditioning
//! - Convergence monitoring and stagnation detection
//! - Performance comparison

use fea::gpu::gpu_advanced_solvers::{
    MixedPrecisionCG, MixedPrecisionConfig,
    FlexibleGMRES, ConvergenceMonitor,
    run_advanced_solver_demo,
};
use fea::gpu::{GPUCGSolver, GPUCSRMatrix};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║        GPU Advanced Solvers Demonstration                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Built-in demo
    run_advanced_solver_demo()?;

    // Mixed precision demonstration
    demo_mixed_precision()?;

    // Flexible GMRES demonstration
    demo_flexible_gmres()?;

    // Convergence monitoring demonstration
    demo_convergence_monitoring()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Demonstration Complete                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demonstrate mixed precision CG.
fn demo_mixed_precision() -> anyhow::Result<()> {
    println!("┌─ Mixed Precision CG Demonstration ───────────────────────┐");

    let n = 5000;
    let (row_ptr, col_ind, values) = create_test_matrix(n, 25000);
    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
    let b = vec![1.0f64; n];

    println!("│ Matrix: {} × {}", n, n);
    println!("│");

    // Standard CG
    println!("│ Standard CG (double precision):");
    let cg = GPUCGSolver::new(0, 1e-10, 500);
    let mut x = vec![0.0f64; n];
    let start = Instant::now();
    let result = cg.solve(&matrix, &b, &mut x)?;
    let cg_time = start.elapsed();

    println!("│   Time: {:.2}ms", cg_time.as_secs_f64() * 1000.0);
    println!("│   Iterations: {}", result.iterations);
    println!("│");

    // Mixed precision CG
    println!("│ Mixed Precision CG:");
    let config = MixedPrecisionConfig {
        tolerance: 1e-10,
        single_tolerance: 1e-4,
        max_iterations: 500,
        max_inner_iterations: 30,
        enable_refinement: true,
        max_refinement_steps: 5,
    };
    let mp_cg = MixedPrecisionCG::new(0, config);
    let start = Instant::now();
    let result = mp_cg.solve(&matrix, &b)?;
    let mp_time = start.elapsed();

    println!("│   Time: {:.2}ms", mp_time.as_secs_f64() * 1000.0);
    println!("│   Inner iterations: {}", result.iterations);
    println!("│   Refinement steps: {}", result.refinement_steps);
    println!("│   Converged: {}", result.converged);
    println!("│");

    let speedup = if mp_time.as_secs_f64() > 0.0 {
        cg_time.as_secs_f64() / mp_time.as_secs_f64()
    } else {
        0.0
    };

    println!("│ Mixed precision can provide 2-5x speedup on GPUs with");
    println!("│   strong single-precision performance (consumer GPUs)");
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Demonstrate Flexible GMRES.
fn demo_flexible_gmres() -> anyhow::Result<()> {
    println!("┌─ Flexible GMRES Demonstration ───────────────────────────┐");

    let n = 2000;
    let (row_ptr, col_ind, values) = create_test_matrix(n, 10000);
    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
    let b = vec![1.0f64; n];

    println!("│ Matrix: {} × {}", n, n);
    println!("│");

    // Standard GMRES
    println!("│ Standard GMRES(30):");
    let gmres = fea::gpu::GPUGMRESSolver::new(0, 1e-8, 300, 30);
    let mut x = vec![0.0f64; n];
    let start = Instant::now();
    let result = gmres.solve(&matrix, &b, &mut x)?;
    let gmres_time = start.elapsed();

    println!("│   Time: {:.2}ms", gmres_time.as_secs_f64() * 1000.0);
    println!("│   Iterations: {}", result.iterations);
    println!("│");

    // Flexible GMRES with Jacobi
    println!("│ Flexible GMRES(30) + Jacobi:");
    let fgmres = FlexibleGMRES::new(0, 30, 300, 1e-8);
    let fgmres = fgmres.with_preconditioner("jacobi");
    let start = Instant::now();
    let result = fgmres.solve(&matrix, &b)?;
    let fgmres_time = start.elapsed();

    println!("│   Time: {:.2}ms", fgmres_time.as_secs_f64() * 1000.0);
    println!("│   Iterations: {}", result.iterations);
    println!("│   Converged: {}", result.converged);
    println!("│");

    println!("│ Flexible GMRES allows variable preconditioning per iteration,");
    println!("│   useful for nonlinear problems and adaptive preconditioners");
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Demonstrate convergence monitoring.
fn demo_convergence_monitoring() -> anyhow::Result<()> {
    println!("┌─ Convergence Monitoring Demonstration ───────────────────┐");

    let n = 1000;
    let (row_ptr, col_ind, values) = create_test_matrix(n, 5000);
    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
    let b = vec![1.0f64; n];
    let b_norm = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Solve with monitoring
    let mut monitor = ConvergenceMonitor::new(b_norm);
    let cg = GPUCGSolver::new(0, 1e-10, 100);
    let mut x = vec![0.0f64; n];

    // Simplified monitoring (in practice would hook into solver)
    for i in 0..20 {
        let res = b_norm / (i + 1) as f64;
        monitor.record(i + 1, res);
    }

    println!("│ Convergence History (every 5 iterations):");
    monitor.print_history(5);

    if let Some(rate) = monitor.convergence_rate() {
        println!("│");
        println!("│ Convergence rate: {:.2}x residual reduction per iteration", rate);
    }

    if monitor.is_stagnating(5, 0.01) {
        println!("│");
        println!("│ WARNING: Convergence stagnation detected!");
        println!("│   Consider: changing preconditioner, adjusting tolerance,");
        println!("│   or using flexible/restarted methods");
    }

    println!("│");
    println!("│ Convergence monitoring enables:");
    println!("│   - Early detection of stagnation");
    println!("│   - Adaptive parameter tuning");
    println!("│   - Automatic solver switching");
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Creates test sparse matrix.
fn create_test_matrix(n: usize, target_nnz: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0;
    for i in 0..n {
        row_ptr.push(nnz);
        col_ind.push(i);
        values.push(4.0);
        nnz += 1;

        for offset in 1..=5 {
            if i >= offset {
                col_ind.push(i - offset);
                values.push(-1.0);
                nnz += 1;
            }
            if i + offset < n {
                col_ind.push(i + offset);
                values.push(-1.0);
                nnz += 1;
            }
        }
    }
    row_ptr.push(nnz);

    (row_ptr, col_ind, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_mixed_precision() {
        assert!(demo_mixed_precision().is_ok());
    }

    #[test]
    fn test_demo_flexible_gmres() {
        assert!(demo_flexible_gmres().is_ok());
    }

    #[test]
    fn test_demo_convergence_monitoring() {
        assert!(demo_convergence_monitoring().is_ok());
    }
}
