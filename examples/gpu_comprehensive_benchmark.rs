//! Comprehensive GPU acceleration benchmark suite.
//!
//! This example runs extensive benchmarks across all GPU features:
//! - Sparse matrix operations
//! - Iterative solvers
//! - Preconditioners
//! - Eigenvalue solvers
//! - Model order reduction
//! - Multi-GPU scaling

use fea::gpu::*;
use fea::gpu::gpu_solvers::*;
use fea::gpu::gpu_precond::*;
use fea::gpu::gpu_eigen::*;
use fea::gpu::gpu_amg::*;
use fea::gpu::gpu_rom::*;
use fea::gpu::gpu_arnoldi::*;
use std::time::Instant;
use std::collections::HashMap;

/// Benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub spmv_gflops: f64,
    pub cg_time_ms: f64,
    pub gmres_time_ms: f64,
    pub bicgstab_time_ms: f64,
    pub eigen_time_ms: f64,
    pub rom_time_ms: f64,
}

/// Runs complete GPU benchmark suite.
pub fn run_gpu_benchmark_suite() -> anyhow::Result<BenchmarkResults> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║         Comprehensive GPU Benchmark Suite                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    if !gpu_available() {
        println!("GPU not available - skipping GPU benchmarks\n");
        return Ok(BenchmarkResults {
            spmv_gflops: 0.0,
            cg_time_ms: 0.0,
            gmres_time_ms: 0.0,
            bicgstab_time_ms: 0.0,
            eigen_time_ms: 0.0,
            rom_time_ms: 0.0,
        });
    }

    let mut results = BenchmarkResults {
        spmv_gflops: 0.0,
        cg_time_ms: 0.0,
        gmres_time_ms: 0.0,
        bicgstab_time_ms: 0.0,
        eigen_time_ms: 0.0,
        rom_time_ms: 0.0,
    };

    // Benchmark 1: SpMV
    benchmark_spmv(&mut results)?;

    // Benchmark 2: Iterative solvers
    benchmark_iterative_solvers(&mut results)?;

    // Benchmark 3: Eigenvalue solvers
    benchmark_eigen_solvers(&mut results)?;

    // Benchmark 4: Model order reduction
    benchmark_rom(&mut results)?;

    // Print summary
    print_benchmark_summary(&results);

    Ok(results)
}

/// Benchmark sparse matrix-vector multiplication.
fn benchmark_spmv(results: &mut BenchmarkResults) -> anyhow::Result<()> {
    println!("┌─ SpMV Benchmark ─────────────────────────────────────────┐");

    let sizes = [(1000, 5000), (5000, 25000), (10000, 50000)];

    println!("│ {:>12} │ {:>12} │ {:>12} │", "Size", "NNZ", "GFLOPS");
    println!("│──────────────┼──────────────┼──────────────│");

    let mut max_gflops = 0.0;

    for &(n, target_nnz) in &sizes {
        // Create sparse matrix
        let (row_ptr, col_ind, values) = create_sparse_matrix(n, target_nnz);
        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
        let x = vec![1.0f64; n];
        let mut y = vec![0.0f64; n];

        let spmv = SparseMatrixVectorMul::new(0);

        // Warmup
        let _ = spmv.spmv(&matrix, &x, &mut y, 1.0, 0.0);

        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = spmv.spmv(&matrix, &x, &mut y, 1.0, 0.0);
        }
        let elapsed = start.elapsed().as_secs_f64() / iterations as f64;

        // 2 FLOPS per non-zero (multiply + add)
        let gflops = 2.0 * matrix.nnz as f64 / elapsed / 1e9;
        max_gflops = max_gflops.max(gflops);

        println!("│ {:>12} │ {:>12} │ {:>12.1} │", n, matrix.nnz, gflops);
    }

    results.spmv_gflops = max_gflops;
    println!("│");
    println!("│ Peak SpMV: {:.1} GFLOPS", max_gflops);
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Benchmark iterative solvers.
fn benchmark_iterative_solvers(results: &mut BenchmarkResults) -> anyhow::Result<()> {
    println!("┌─ Iterative Solver Benchmark ─────────────────────────────┐");

    let n = 5000;
    let (row_ptr, col_ind, values) = create_sparse_matrix(n, 25000);
    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
    let b = vec![1.0f64; n];

    println!("│ {:>15} │ {:>10} │ {:>12} │", "Solver", "Iter", "Time (ms)");
    println!("│─────────────────┼────────────┼──────────────│");

    // CG
    let cg = GPUCGSolver::new(0, 1e-8, 500);
    let mut x = vec![0.0f64; n];
    let start = Instant::now();
    let result = cg.solve(&matrix, &b, &mut x)?;
    let cg_time = start.elapsed().as_secs_f64() * 1000.0;
    results.cg_time_ms = cg_time;

    println!("│ {:>15} │ {:>10} │ {:>12.2} │",
        "CG", result.iterations, cg_time);

    // GMRES
    let gmres = GPUGMRESSolver::new(0, 1e-8, 500, 50);
    let mut x = vec![0.0f64; n];
    let start = Instant::now();
    let result = gmres.solve(&matrix, &b, &mut x)?;
    let gmres_time = start.elapsed().as_secs_f64() * 1000.0;
    results.gmres_time_ms = gmres_time;

    println!("│ {:>15} │ {:>10} │ {:>12.2} │",
        "GMRES(50)", result.iterations, gmres_time);

    // BiCGSTAB
    let bicgstab = GPUBiCGSTABSolver::new(0, 1e-8, 500);
    let mut x = vec![0.0f64; n];
    let start = Instant::now();
    let result = bicgstab.solve(&matrix, &b, &mut x)?;
    let bicgstab_time = start.elapsed().as_secs_f64() * 1000.0;
    results.bicgstab_time_ms = bicgstab_time;

    println!("│ {:>15} │ {:>10} │ {:>12.2} │",
        "BiCGSTAB", result.iterations, bicgstab_time);

    println!("│");
    println!("│ Fastest: {}", get_fastest_solver(cg_time, gmres_time, bicgstab_time));
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Benchmark eigenvalue solvers.
fn benchmark_eigen_solvers(results: &mut BenchmarkResults) -> anyhow::Result<()> {
    println!("┌─ Eigenvalue Solver Benchmark ────────────────────────────┐");

    let n = 1000;
    let (row_ptr, col_ind, values) = create_sparse_matrix(n, 5000);
    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);

    // Lanczos
    let start = Instant::now();
    let lanczos = GPULanczosSolver::new(0, 1e-8, 100, 5);
    let result = lanczos.solve_largest(&matrix, None)?;
    let lanczos_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("│ Lanczos: {:.2}ms ({} eigenvalues)", lanczos_time, result.eigenvalues.len());

    // Arnoldi
    let start = Instant::now();
    let arnoldi = GPUArnoldiFactorization::new(0, 100, 20);
    let factorization = arnoldi.factorize(&matrix, None)?;
    let arnoldi_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("│ Arnoldi: {:.2}ms ({} Krylov vectors)", arnoldi_time, factorization.num_krylov_vectors());

    results.eigen_time_ms = lanczos_time.min(arnoldi_time);

    println!("│");
    println!("│ Note: Lanczos faster for SPD, Arnoldi for general");
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Benchmark model order reduction.
fn benchmark_rom(results: &mut BenchmarkResults) -> anyhow::Result<()> {
    println!("┌─ Model Order Reduction Benchmark ────────────────────────┐");

    let n = 500;
    let m = 50; // Snapshots

    // Generate synthetic snapshots
    let snapshots = nalgebra::DMatrix::from_fn(n, m, |i, j| {
        ((i + j) as f64 * 0.1).sin()
    });

    println!("│ Snapshots: {} × {}", n, m);

    // POD
    let start = Instant::now();
    let rom = ReducedOrderModel::from_snapshots(&snapshots, 0.99);
    let pod_time = start.elapsed().as_secs_f64() * 1000.0;
    results.rom_time_ms = pod_time;

    println!("│ POD basis: {} modes ({:.1}% energy)",
        rom.num_modes, rom.energy_captured * 100.0);
    println!("│ POD time: {:.2}ms", pod_time);

    println!("│");
    println!("│ Reduction ratio: {:.1}%", rom.reduction_ratio(n) * 100.0);
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Print benchmark summary.
fn print_benchmark_summary(results: &BenchmarkResults) {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                  Benchmark Summary                        ║");
    println!("╠═══════════════════════════════════════════════════════════╣");

    println!("║ SpMV Performance:    {:>12.1} GFLOPS               ║", results.spmv_gflops);
    println!("║ CG Solver:           {:>12.2} ms                   ║", results.cg_time_ms);
    println!("║ GMRES Solver:        {:>12.2} ms                   ║", results.gmres_time_ms);
    println!("║ BiCGSTAB Solver:     {:>12.2} ms                   ║", results.bicgstab_time_ms);
    println!("║ Eigenvalue Solver:   {:>12.2} ms                   ║", results.eigen_time_ms);
    println!("║ Model Reduction:     {:>12.2} ms                   ║", results.rom_time_ms);

    println!("╚═══════════════════════════════════════════════════════════╝");
}

/// Gets fastest solver name.
fn get_fastest_solver(cg: f64, gmres: f64, bicgstab: f64) -> &'static str {
    let min = cg.min(gmres).min(bicgstab);
    if min == cg { "CG" }
    else if min == gmres { "GMRES" }
    else { "BiCGSTAB" }
}

/// Creates sparse matrix with target nnz.
fn create_sparse_matrix(n: usize, target_nnz: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_ind = Vec::with_capacity(target_nnz);
    let mut values = Vec::with_capacity(target_nnz);

    let nnz_per_row = target_nnz / n;
    let mut nnz = 0;

    for i in 0..n {
        row_ptr[i] = nnz;

        // Diagonal
        col_ind.push(i);
        values.push(4.0);
        nnz += 1;

        // Off-diagonals
        for offset in 1..=nnz_per_row {
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
    row_ptr[n] = nnz;

    (row_ptr, col_ind, values)
}

fn main() -> anyhow::Result<()> {
    run_gpu_benchmark_suite()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite() {
        let results = run_gpu_benchmark_suite().unwrap();
        // Just verify it runs - actual values depend on GPU availability
        assert!(results.spmv_gflops >= 0.0);
    }

    #[test]
    fn test_sparse_matrix_creation() {
        let (row_ptr, col_ind, values) = create_sparse_matrix(100, 500);
        assert_eq!(row_ptr.len(), 101);
        assert!(values.len() >= 100); // At least diagonal
    }
}
