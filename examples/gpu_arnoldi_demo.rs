//! GPU Arnoldi Iteration demonstration.

use fea::gpu::gpu_arnoldi::{GPUArnoldi, IRAM, run_arnoldi_demo};
use fea::gpu::gpu_eigen_enhanced::GPULanczosEigen;
use fea::gpu::GPUCSRMatrix;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("GPU Arnoldi Iteration Demo\n");
    run_arnoldi_demo()?;
    compare_arnoldi_lanczos()?;
    demo_iram()?;
    println!("\nDemo Complete");
    Ok(())
}

fn compare_arnoldi_lanczos() -> anyhow::Result<()> {
    println!("\nArnoldi vs Lanczos Comparison\n");

    let n = 300;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0;
    for i in 0..n {
        row_ptr.push(nnz);
        col_ind.push(i);
        values.push(4.0);
        nnz += 1;
        if i > 0 { col_ind.push(i - 1); values.push(-1.0); nnz += 1; }
        if i < n - 1 { col_ind.push(i + 1); values.push(-1.0); nnz += 1; }
    }
    row_ptr.push(nnz);

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);

    let start = Instant::now();
    let arnoldi = GPUArnoldi::new(0, 50, 1e-10);
    let arnoldi_result = arnoldi.iterate(&matrix, 20)?;
    let arnoldi_time = start.elapsed();

    println!("Arnoldi: {:.2}ms, {} iterations", arnoldi_time.as_secs_f64() * 1000.0, arnoldi_result.iterations);

    let start = Instant::now();
    let lanczos = GPULanczosEigen::new(0, 1e-10, 50, 10);
    let lanczos_result = lanczos.compute_largest(&matrix)?;
    let lanczos_time = start.elapsed();

    println!("Lanczos: {:.2}ms, {} iterations, {} eigenvalues",
        lanczos_time.as_secs_f64() * 1000.0,
        lanczos_result.num_iterations,
        lanczos_result.eigenvalues.len());

    Ok(())
}

fn demo_iram() -> anyhow::Result<()> {
    println!("\nIRAM Demo\n");

    let n = 500;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0;
    for i in 0..n {
        row_ptr.push(nnz);
        col_ind.push(i);
        values.push(2.0);
        nnz += 1;
        if i > 0 { col_ind.push(i - 1); values.push(-1.0); nnz += 1; }
        if i < n - 1 { col_ind.push(i + 1); values.push(-1.0); nnz += 1; }
    }
    row_ptr.push(nnz);

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);

    let start = Instant::now();
    let iram = IRAM::new(0, 10, 100, 1e-10);
    let result = iram.compute(&matrix)?;
    let elapsed = start.elapsed();

    println!("IRAM: {:.2}ms, {} eigenvalues", elapsed.as_secs_f64() * 1000.0, result.eigenvalues.len());
    for (i, &lam) in result.eigenvalues.iter().take(5).enumerate() {
        println!("  λ{} = {:.6}", i + 1, lam);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_arnoldi_lanczos() {
        assert!(compare_arnoldi_lanczos().is_ok());
    }

    #[test]
    fn test_demo_iram() {
        assert!(demo_iram().is_ok());
    }
}
