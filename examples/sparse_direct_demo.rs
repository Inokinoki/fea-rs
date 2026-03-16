//! GPU Sparse Direct Solver demonstration and validation.
//!
//! This example demonstrates:
//! - Sparse LU factorization
//! - Sparse Cholesky factorization
//! - Batched direct solves
//! - Validation against iterative methods
//! - Performance comparison

use fea::gpu::gpu_sparse_direct::{
    GPUSparseLU, GPUSparseCholesky, GPUBatchedLU,
    benchmark_sparse_direct_solvers,
};
use fea::gpu::{GPUCGSolver, GPUCSRMatrix};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║      GPU Sparse Direct Solver Demonstration               ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Sparse LU factorization demo
    demo_sparse_lu()?;

    // Sparse Cholesky factorization demo
    demo_sparse_cholesky()?;

    // Batched direct solves
    demo_batched_solves()?;

    // Comparison with iterative methods
    compare_direct_iterative()?;

    // Benchmark suite
    run_benchmarks();

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Demonstration Complete                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demonstrates sparse LU factorization.
fn demo_sparse_lu() -> anyhow::Result<()> {
    println!("┌─ Sparse LU Factorization ────────────────────────────────┐");

    // Create test matrix (2D Laplacian)
    let n = 50;
    let (row_ptr, col_ind, values) = create_2d_laplacian(n);
    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n * n, n * n, 0);

    println!("│ Matrix size: {} × {}", n * n, n * n);
    println!("│ Non-zeros: {}", matrix.nnz);
    println!("│ Sparsity: {:.2}%", 100.0 * (1.0 - matrix.nnz as f64 / ((n * n) * (n * n)) as f64));

    // Factorize
    let mut lu = GPUSparseLU::new(0);
    let start = Instant::now();
    let fact_ok = lu.factorize(&matrix).is_ok();
    let fact_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("│");
    println!("│ Factorization:");
    println!("│   Status: {}", if fact_ok { "Success" } else { "Failed" });
    println!("│   Time: {:.2} ms", fact_time);

    if fact_ok {
        println!("│   nnz(L): {}", lu.nnz_l());
        println!("│   nnz(U): {}", lu.nnz_u());

        // Solve
        let b: Vec<f64> = (0..n * n).map(|i| ((i % 10) as f64 * 0.1).sin()).collect();
        let start = Instant::now();
        let x = lu.solve(&b);
        let solve_time = start.elapsed().as_secs_f64() * 1000.0;

        match x {
            Ok(sol) => {
                let max_val = sol.iter().map(|v| v.abs()).fold(0.0, f64::max);
                println!("│");
                println!("│ Solve:");
                println!("│   Time: {:.2} ms", solve_time);
                println!("│   Max |x|: {:.4}", max_val);
                println!("│   Status: Success");
            }
            Err(e) => {
                println!("│   Solve failed: {}", e);
            }
        }
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demonstrates sparse Cholesky factorization.
fn demo_sparse_cholesky() -> anyhow::Result<()> {
    println!("┌─ Sparse Cholesky Factorization ──────────────────────────┐");

    // Create SPD matrix
    let n = 50;
    let (row_ptr, col_ind, values) = create_2d_laplacian(n);
    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n * n, n * n, 0);

    println!("│ Matrix size: {} × {}", n * n, n * n);
    println!("│ Matrix type: SPD (Symmetric Positive Definite)");

    // Factorize
    let mut chol = GPUSparseCholesky::new(0);
    let start = Instant::now();
    let fact_ok = chol.factorize(&matrix).is_ok();
    let fact_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("│");
    println!("│ Factorization:");
    println!("│   Status: {}", if fact_ok { "Success" } else { "Failed" });
    println!("│   Time: {:.2} ms", fact_time);

    if fact_ok {
        // Solve
        let b = vec![1.0f64; n * n];
        let start = Instant::now();
        let x = chol.solve(&b);
        let solve_time = start.elapsed().as_secs_f64() * 1000.0;

        match x {
            Ok(sol) => {
                let max_val = sol.iter().map(|v| v.abs()).fold(0.0, f64::max);
                println!("│");
                println!("│ Solve:");
                println!("│   Time: {:.2} ms", solve_time);
                println!("│   Max |x|: {:.4}", max_val);
                println!("│   Status: Success");
            }
            Err(e) => {
                println!("│   Solve failed: {}", e);
            }
        }
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demonstrates batched direct solves.
fn demo_batched_solves() -> anyhow::Result<()> {
    println!("┌─ Batched Direct Solves ──────────────────────────────────┐");

    // Create multiple matrices
    let n = 30;
    let num_batches = 5;
    let mut matrices = Vec::new();

    for _ in 0..num_batches {
        let (row_ptr, col_ind, values) = create_2d_laplacian(n);
        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n * n, n * n, 0);
        matrices.push(matrix);
    }

    println!("│ Number of matrices: {}", num_batches);
    println!("│ Matrix size: {} × {}", n * n, n * n);

    // Batched factorization
    let mut batched = GPUBatchedLU::new(0, num_batches);
    let start = Instant::now();
    let fact_ok = batched.factorize_batch(&matrices).is_ok();
    let fact_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("│");
    println!("│ Batch Factorization:");
    println!("│   Status: {}", if fact_ok { "Success" } else { "Failed" });
    println!("│   Time: {:.2} ms", fact_time);
    println!("│   Time per matrix: {:.2} ms", fact_time / num_batches as f64);

    if fact_ok {
        // Batched solve
        let b_vectors: Vec<Vec<f64>> = (0..num_batches)
            .map(|i| vec![(i + 1) as f64; n * n])
            .collect();

        let start = Instant::now();
        let solutions = batched.solve_batch(&b_vectors);
        let solve_time = start.elapsed().as_secs_f64() * 1000.0;

        match solutions {
            Ok(sols) => {
                println!("│");
                println!("│ Batch Solve:");
                println!("│   Status: Success");
                println!("│   Time: {:.2} ms", solve_time);
                println!("│   Time per solve: {:.2} ms", solve_time / num_batches as f64);
                println!("│   Solutions: {}", sols.len());
            }
            Err(e) => {
                println!("│   Solve failed: {}", e);
            }
        }
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Compares direct and iterative methods.
fn compare_direct_iterative() -> anyhow::Result<()> {
    println!("┌─ Direct vs Iterative Comparison ─────────────────────────┐");

    let n = 100;
    let (row_ptr, col_ind, values) = create_2d_laplacian(n);
    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n * n, n * n, 0);
    let b = vec![1.0f64; n * n];

    println!("│ Matrix size: {} × {}", n * n, n * n);
    println!("│");

    // Sparse Cholesky (direct)
    let mut chol = GPUSparseCholesky::new(0);
    let start = Instant::now();
    let chol_fact = chol.factorize(&matrix);
    let chol_fact_time = start.elapsed().as_secs_f64() * 1000.0;

    let mut chol_solve_time = 0.0;
    let mut chol_total = 0.0;
    if chol_fact.is_ok() {
        let start = Instant::now();
        let _ = chol.solve(&b);
        chol_solve_time = start.elapsed().as_secs_f64() * 1000.0;
        chol_total = chol_fact_time + chol_solve_time;
    }

    println!("│ Sparse Cholesky (Direct):");
    println!("│   Factorize: {:.2} ms", chol_fact_time);
    println!("│   Solve: {:.2} ms", chol_solve_time);
    println!("│   Total: {:.2} ms", chol_total);
    println!("│");

    // CG (iterative)
    let cg = GPUCGSolver::new(0, 1e-10, 1000);
    let mut x = vec![0.0; n * n];
    let start = Instant::now();
    let cg_result = cg.solve(&matrix, &b, &mut x);
    let cg_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("│ CG (Iterative):");
    println!("│   Solve: {:.2} ms", cg_time);
    match cg_result {
        Ok(r) => {
            println!("│   Iterations: {}", r.iterations);
            println!("│   Converged: {}", r.converged);
        }
        Err(e) => {
            println!("│   Failed: {}", e);
        }
    }
    println!("│");

    // Summary
    println!("│ Comparison:");
    if chol_total > 0.0 && cg_time > 0.0 {
        let speedup = cg_time / chol_total;
        println!("│   Direct is {:.2}x {} than iterative",
            if speedup > 1.0 { speedup } else { 1.0 / speedup },
            if speedup > 1.0 { "slower" } else { "faster" });
        println!("│");
        println!("│ Note: Direct methods better for multiple RHS");
        println!("│       Iterative methods better for single RHS");
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Runs benchmark suite.
fn run_benchmarks() {
    println!("┌─ Sparse Direct Solver Benchmarks ────────────────────────┐");

    let result = benchmark_sparse_direct_solvers();

    println!("│ Sparse LU:");
    println!("│   Factorize: {:.2} ms", result.lu_factorize_time_ms);
    println!("│   Solve: {:.2} ms", result.lu_solve_time_ms);
    println!("│   Success: {}", result.lu_solve_success);
    println!("│");
    println!("│ Sparse Cholesky:");
    println!("│   Factorize: {:.2} ms", result.cholesky_factorize_time_ms);
    println!("│   Solve: {:.2} ms", result.cholesky_solve_time_ms);
    println!("│   Success: {}", result.cholesky_solve_success);

    println!("└────────────────────────────────────────────────────────┘\n");
}

/// Creates 2D Laplacian matrix (5-point stencil).
fn create_2d_laplacian(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let size = n * n;
    let mut row_ptr = Vec::with_capacity(size + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0;
    for i in 0..n {
        for j in 0..n {
            row_ptr.push(nnz);

            let idx = i * n + j;

            // Diagonal
            col_ind.push(idx);
            values.push(4.0);
            nnz += 1;

            // Left neighbor
            if j > 0 {
                col_ind.push(idx - 1);
                values.push(-1.0);
                nnz += 1;
            }

            // Right neighbor
            if j < n - 1 {
                col_ind.push(idx + 1);
                values.push(-1.0);
                nnz += 1;
            }

            // Top neighbor
            if i > 0 {
                col_ind.push(idx - n);
                values.push(-1.0);
                nnz += 1;
            }

            // Bottom neighbor
            if i < n - 1 {
                col_ind.push(idx + n);
                values.push(-1.0);
                nnz += 1;
            }
        }
    }
    row_ptr.push(nnz);

    (row_ptr, col_ind, values)
}
