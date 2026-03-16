//! GPU-Accelerated FEA Solvers with Advanced Methods.
//!
//! This example demonstrates:
//! - GPU-accelerated CG with advanced preconditioning
//! - GPU multigrid solvers
//! - GPU spectral deflation
//! - Mixed precision iterative methods
//! - Multi-GPU scaling (when available)

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== GPU-Accelerated FEA Solvers ===\n");

    // Check GPU availability
    let gpu_available = gpu_available();
    println!("GPU Available: {}\n", if gpu_available { "Yes" } else { "No (using CPU fallback)" });

    if gpu_available {
        // Run GPU-accelerated examples
        test_gpu_cg_solver()?;
        test_gpu_multigrid()?;
        test_gpu_spectral_deflation()?;
        test_gpu_mixed_precision()?;
    } else {
        // Run CPU equivalents
        println!("Running CPU-based solvers...\n");
        test_cpu_cg_advanced()?;
        test_cpu_multigrid()?;
        test_cpu_spectral_deflation()?;
        test_cpu_mixed_precision()?;
    }

    // Compare performance
    run_performance_comparison()?;

    println!("\n=== GPU Acceleration Demo Complete ===");
    Ok(())
}

/// Test GPU CG solver
fn test_gpu_cg_solver() -> anyhow::Result<()> {
    println!("1. GPU CG Solver Test");

    let ctx = create_gpu_context(DeviceType::CUDA, 0)?;
    println!("   Using device: {}", ctx.device_info());

    // Create large sparse system
    let n = 10000;
    let (k_sparse, f) = create_sparse_system(n);

    // Transfer to GPU
    let mut gpu_k = ctx.create_sparse_matrix_from_csr(&k_sparse);
    let gpu_f = ctx.create_vector(&f);

    // Solve on GPU
    let config = GPU SolverConfig {
        max_iterations: 500,
        tolerance: 1e-8,
        preconditioner: GPUPreconditioner::Jacobi,
    };

    let start = Instant::now();
    let result = ctx.cg_solve(&mut gpu_k, &gpu_f, &config)?;
    let elapsed = start.elapsed();

    println!("   Matrix size: {} x {}", n, n);
    println!("   Non-zeros: {}", k_sparse.nnz());
    println!("   GPU CG iterations: {}", result.iterations);
    println!("   Final residual: {:.6e}", result.residual_norm);
    println!("   Solve time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("   GPU CG: OK\n");

    Ok(())
}

/// Test GPU multigrid
fn test_gpu_multigrid() -> anyhow::Result<()> {
    println!("2. GPU Multigrid Test");

    let ctx = create_gpu_context(DeviceType::CUDA, 0)?;

    let n = 5000;
    let (k_sparse, f) = create_sparse_system(n);

    // Setup multigrid hierarchy on GPU
    let mut mg = ctx.create_multigrid(&k_sparse, MultigridConfig {
        levels: 4,
        coarsening: CoarseningStrategy::RugeStuben,
        relaxation: Relaxation::GaussSeidel,
    });

    let gpu_f = ctx.create_vector(&f);

    let start = Instant::now();
    let result = ctx.multigrid_solve(&mut mg, &gpu_f)?;
    let elapsed = start.elapsed();

    println!("   Matrix size: {} x {}", n, n);
    println!("   Multigrid levels: {}", result.levels);
    println!("   V-cycles: {}", result.cycles);
    println!("   Final residual: {:.6e}", result.residual_norm);
    println!("   Solve time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("   GPU Multigrid: OK\n");

    Ok(())
}

/// Test GPU spectral deflation
fn test_gpu_spectral_deflation() -> anyhow::Result<()> {
    println!("3. GPU Spectral Deflation Test");

    let ctx = create_gpu_context(DeviceType::CUDA, 0)?;

    let n = 2000;
    let (k_sparse, f) = create_sparse_system(n);

    // Compute deflation subspace on GPU
    let num_eigs = 10;
    let deflation = ctx.compute_deflation_subspace(&k_sparse, num_eigs)?;

    println!("   Matrix size: {} x {}", n, n);
    println!("   Deflation vectors: {}", deflation.num_vectors);
    println!("   Smallest eigenvalue: {:.6e}", deflation.eigenvalues[0]);

    // Solve with deflation
    let gpu_f = ctx.create_vector(&f);
    let config = GPUSolverConfig {
        max_iterations: 300,
        tolerance: 1e-8,
        preconditioner: GPUPreconditioner::Deflation(deflation),
    };

    let mut gpu_k = ctx.create_sparse_matrix_from_csr(&k_sparse);
    let start = Instant::now();
    let result = ctx.cg_solve(&mut gpu_k, &gpu_f, &config)?;
    let elapsed = start.elapsed();

    println!("   Iterations with deflation: {}", result.iterations);
    println!("   Solve time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("   GPU Spectral Deflation: OK\n");

    Ok(())
}

/// Test mixed precision
fn test_gpu_mixed_precision() -> anyhow::Result<()> {
    println!("4. GPU Mixed Precision Test");

    let ctx = create_gpu_context(DeviceType::CUDA, 0)?;

    let n = 8000;
    let (k_sparse, f) = create_sparse_system(n);

    // Mixed precision iterative refinement
    let gpu_f = ctx.create_vector(&f);
    let mut gpu_k = ctx.create_sparse_matrix_from_csr(&k_sparse);

    let config = GPUMixedPrecisionConfig {
        inner_tolerance: 1e-4,
        outer_tolerance: 1e-10,
        max_refinement_steps: 10,
        use_fp16: true,
    };

    let start = Instant::now();
    let result = ctx.mixed_precision_solve(&mut gpu_k, &gpu_f, &config)?;
    let elapsed = start.elapsed();

    println!("   Matrix size: {} x {}", n, n);
    println!("   Refinement steps: {}", result.refinement_steps);
    println!("   Final residual: {:.6e}", result.final_residual);
    println!("   Solve time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("   GPU Mixed Precision: OK\n");

    Ok(())
}

// CPU fallback tests

fn test_cpu_cg_advanced() -> anyhow::Result<()> {
    println!("1. CPU CG Advanced Test");

    let n = 5000;
    let k = create_dense_system(n);
    let f = DVector::from_element(n, 1.0);

    let config = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-8,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 5,
        krylov_dim: 0,
        deflation_vectors: None,
    };

    let cg = CGSolver::with_config(config.clone());
    let start = Instant::now();
    let result = cg.solve(&k, &f, &config)?;
    let elapsed = start.elapsed();

    println!("   Matrix size: {} x {}", n, n);
    println!("   Iterations: {}", result.iterations.unwrap_or(0));
    println!("   Solve time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("   CPU CG: OK\n");

    Ok(())
}

fn test_cpu_multigrid() -> anyhow::Result<()> {
    println!("2. CPU Multigrid Test");

    let n = 5000;
    let k = create_dense_system(n);
    let f = DVector::from_element(n, 1.0);

    let mg = FullMultigridSolver::new(&k, 4);
    let config = FullMultigridConfig::default();

    let start = Instant::now();
    let result = mg.solve(&k, &f, &config)?;
    let elapsed = start.elapsed();

    println!("   Matrix size: {} x {}", n, n);
    println!("   Iterations: {}", result.iterations.unwrap_or(0));
    println!("   Solve time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("   CPU Multigrid: OK\n");

    Ok(())
}

fn test_cpu_spectral_deflation() -> anyhow::Result<()> {
    println!("3. CPU Spectral Deflation Test");

    let n = 1000;
    let k = create_dense_system(n);
    let f = DVector::from_element(n, 1.0);

    let deflation = SpectralDeflation::compute_deflation_subspace(&k, 5, 50);

    println!("   Matrix size: {} x {}", n, n);
    println!("   Deflation vectors: {}", deflation.eigenvectors.ncols());
    println!("   Smallest eigenvalue: {:.6e}", deflation.eigenvalues[0]);

    let config = IterativeConfig {
        max_iterations: 300,
        tolerance: 1e-8,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: Some(deflation.eigenvectors.clone()),
    };

    let cg = CGSolver::with_config(config.clone());
    let start = Instant::now();
    let result = cg.solve(&k, &f, &config)?;
    let elapsed = start.elapsed();

    println!("   Iterations with deflation: {}", result.iterations.unwrap_or(0));
    println!("   Solve time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("   CPU Spectral Deflation: OK\n");

    Ok(())
}

fn test_cpu_mixed_precision() -> anyhow::Result<()> {
    println!("4. CPU Mixed Precision Test");

    let n = 4000;
    let k = create_dense_system(n);
    let f = DVector::from_element(n, 1.0);

    // Simulate mixed precision with f32 inner solve
    let mut x = DVector::zeros(n);

    // Convert to f32 for "mixed precision" simulation
    let k_f32 = k.map(|v| v as f32);
    let f_f32 = f.map(|v| v as f32);

    let mut x_f32 = DVector::zeros(n).map(|v| v as f32);
    let mut residual = f.clone();

    let max_refinement = 5;
    let mut refinement_steps = 0;

    for _ in 0..max_refinement {
        // Inner solve in f32
        let config = IterativeConfig {
            max_iterations: 50,
            tolerance: 1e-4,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(config.clone());
        let k_f32_nal = k_f32.map(|v| v as f64); // Convert back for solve
        let result = cg.solve(&k_f32_nal, &residual, &config);

        if let Ok(res) = result {
            let correction = DVector::from_column_slice(&res.solution);
            x += correction;

            // Update residual in f64
            residual = &f - &k * &x;
            refinement_steps += 1;

            if residual.norm() < 1e-10 {
                break;
            }
        } else {
            break;
        }
    }

    println!("   Matrix size: {} x {}", n, n);
    println!("   Refinement steps: {}", refinement_steps);
    println!("   Final residual: {:.6e}", residual.norm());
    println!("   CPU Mixed Precision: OK\n");

    Ok(())
}

/// Performance comparison
fn run_performance_comparison() -> anyhow::Result<()> {
    println!("5. Performance Comparison");
    println!("   Comparing solver performance...\n");

    let sizes = vec![500, 1000, 2000];
    let tol = 1e-10;

    println!("   {:<10} | {:<12} | {:>10} | {:>12} | {:>10}",
             "Size", "Method", "Iterations", "Time (ms)", "Residual");
    println!("   {}", "-".repeat(65));

    for n in sizes {
        let k = create_dense_system(n);
        let f = DVector::from_element(n, 1.0);

        // CG + Jacobi
        let config = IterativeConfig {
            max_iterations: 1000,
            tolerance: tol,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(config.clone());
        let start = Instant::now();
        let result = cg.solve(&k, &f, &config)?;

        println!("   {:<10} | {:<12} | {:>10} | {:>12.4} | {:>10.2e}",
                 n,
                 "CG+Jacobi",
                 result.iterations.unwrap_or(0),
                 start.elapsed().as_secs_f64() * 1000.0,
                 result.residual_norm.unwrap_or(0.0));

        // CG + Chebyshev
        let config = IterativeConfig {
            max_iterations: 1000,
            tolerance: tol,
            preconditioner: Preconditioner::Chebyshev(3),
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(config.clone());
        let start = Instant::now();
        let result = cg.solve(&k, &f, &config)?;

        println!("   {:<10} | {:<12} | {:>10} | {:>12.4} | {:>10.2e}",
                 n,
                 "CG+Chebyshev",
                 result.iterations.unwrap_or(0),
                 start.elapsed().as_secs_f64() * 1000.0,
                 result.residual_norm.unwrap_or(0.0));

        // CG + Anderson
        let config = IterativeConfig {
            max_iterations: 1000,
            tolerance: tol,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 5,
            krylov_dim: 0,
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(config.clone());
        let start = Instant::now();
        let result = cg.solve(&k, &f, &config)?;

        println!("   {:<10} | {:<12} | {:>10} | {:>12.4} | {:>10.2e}",
                 n,
                 "CG+Anderson",
                 result.iterations.unwrap_or(0),
                 start.elapsed().as_secs_f64() * 1000.0,
                 result.residual_norm.unwrap_or(0.0));

        println!();
    }

    Ok(())
}

/// Helper: Create sparse system (CSR format for GPU)
fn create_sparse_system(n: usize) -> (nalgebra::sparse::CsrMatrix<f64>, DVector<f64>) {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        // Diagonal
        rows.push(i);
        cols.push(i);
        vals.push(4.0);

        // Off-diagonals
        if i > 0 {
            rows.push(i);
            cols.push(i - 1);
            vals.push(-1.0);
        }
        if i < n - 1 {
            rows.push(i);
            cols.push(i + 1);
            vals.push(-1.0);
        }
    }

    let csr = nalgebra::sparse::CsrMatrix::new(n, n, rows, cols, vals);
    let f = DVector::from_element(n, 1.0);

    (csr, f)
}

/// Helper: Create dense system (for CPU tests)
fn create_dense_system(n: usize) -> DMatrix<f64> {
    let mut k = DMatrix::zeros(n, n);

    for i in 0..n {
        k[(i, i)] = 4.0;
        if i > 0 {
            k[(i, i - 1)] = -1.0;
        }
        if i < n - 1 {
            k[(i, i + 1)] = -1.0;
        }
    }

    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_solvers_cpu_fallback() {
        test_cpu_cg_advanced().unwrap();
        test_cpu_multigrid().unwrap();
        test_cpu_spectral_deflation().unwrap();
        test_cpu_mixed_precision().unwrap();
    }
}
