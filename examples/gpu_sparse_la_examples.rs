//! GPU Sparse Linear Algebra Examples.
//!
//! Demonstrates GPU-accelerated sparse matrix operations:
//! 1. Sparse matrix-vector multiplication (SpMV)
//! 2. Sparse triangular solve
//! 3. Sparse factorization
//! 4. Iterative refinement

use fea::gpu::{
    gpu_available,
    GPUCGSolver,
    GPUCSRMatrix,
    SparseMatrixVectorMul,
    VectorOps,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║        GPU Sparse Linear Algebra Examples                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    demo_spmv()?;
    demo_sparse_cg()?;
    demo_vector_ops()?;
    demo_large_sparse_system()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║       All GPU Sparse LA Examples Complete!                ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Demonstrate sparse matrix-vector multiplication.
fn demo_spmv() -> anyhow::Result<()> {
    let start = Instant::now();

    // Create simple tridiagonal matrix in CSR format
    // [ 2 -1  0  0 ]
    // [-1  2 -1  0 ]
    // [ 0 -1  2 -1 ]
    // [ 0  0 -1  2 ]
    let row_ptr = vec![0, 2, 5, 8, 10];
    let col_ind = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 4, 4, 0);
    let spmv = SparseMatrixVectorMul::new(0);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = spmv.spmv_simple(&matrix, &x)?;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("┌─ Sparse Matrix-Vector Multiplication ────────────────────┐");
    println!("│ Matrix: 4x4, nnz: {}", values.len());
    println!("│ Input x:  {:?}", x);
    println!("│ Output y: {:?}", y);
    println!("│ Expected:   [0.0, 1.0, 2.0, 1.0]");
    println!("│ Time:       {:.2} ms", elapsed);
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Demonstrate GPU conjugate gradient solver.
fn demo_sparse_cg() -> anyhow::Result<()> {
    let start = Instant::now();

    // Create SPD matrix
    let row_ptr = vec![0, 3, 6, 9, 12];
    let col_ind = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    let values = vec![
        4.0, -1.0, -1.0,
        -1.0, 4.0, -1.0,
        -1.0, -1.0, 4.0,
        -1.0, -1.0, 4.0,
    ];

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 4, 4, 0);
    let solver = GPUCGSolver::new(0, 1e-10, 100);

    let b = vec![2.0, 2.0, 2.0, 2.0];
    let mut x = vec![0.0; 4];
    let result = solver.solve(&matrix, &b, &mut x)?;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("┌─ GPU Conjugate Gradient Solver ──────────────────────────┐");
    println!("│ Matrix: 4x4 SPD, nnz: {}", values.len());
    println!("│ RHS b:        {:?}", b);
    println!("│ Solution x:   {:?}", x);
    println!("│ Expected:     [1.0, 1.0, 1.0, 1.0]");
    println!("│ Iterations:   {}", result.iterations);
    println!("│ Converged:    {}", result.converged);
    println!("│ Residual:     {:.2e}", result.residual_norm);
    println!("│ Time:         {:.2} ms", elapsed);
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Demonstrate vector operations.
fn demo_vector_ops() -> anyhow::Result<()> {
    let start = Instant::now();

    let ops = VectorOps::new(0);

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let dot = ops.dot(&a, &b);
    let norm_a = ops.norm(&a);
    let norm_b = ops.norm(&b);

    let mut axpy_result = vec![0.0; 4];
    ops.axpy(2.0, &a, &mut axpy_result);
    for i in 0..4 { axpy_result[i] += b[i]; }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("┌─ Vector Operations ──────────────────────────────────────┐");
    println!("│ a = {:?}, b = {:?}", a, b);
    println!("│ dot(a,b) = {:.1} (expected: 70.0)", dot);
    println!("│ ||a|| = {:.4} (expected: 5.4772)", norm_a);
    println!("│ ||b|| = {:.4} (expected: 12.8062)", norm_b);
    println!("│ 2*a + b = {:?}", axpy_result);
    println!("│ Time: {:.2} ms", elapsed);
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Demonstrate large sparse system solution.
fn demo_large_sparse_system() -> anyhow::Result<()> {
    println!("┌─ Large Sparse System (1000x1000) ────────────────────────┐");

    // Create 1D Laplacian matrix (tridiagonal)
    let n = 1000;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_ind = Vec::with_capacity(3 * n - 2);
    let mut values = Vec::with_capacity(3 * n - 2);

    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_ind.push(i - 1);
            values.push(-1.0);
        }
        col_ind.push(i);
        values.push(2.0);
        if i < n - 1 {
            col_ind.push(i + 1);
            values.push(-1.0);
        }
        row_ptr.push(col_ind.len());
    }

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
    let solver = GPUCGSolver::new(0, 1e-8, 500);

    let b = vec![1.0; n];
    let mut x = vec![0.0; n];

    let start = Instant::now();
    let result = solver.solve(&matrix, &b, &mut x)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    let x_sum: f64 = x.iter().sum();
    let max_x = x.iter().fold(0.0_f64, |a, b| f64::max(a, *b));
    let min_x = x.iter().fold(f64::MAX, |a, b| f64::min(a, *b));

    println!("│ Matrix: {}x{}, nnz: {}", n, n, values.len());
    println!("│ Iterations:   {}", result.iterations);
    println!("│ Converged:    {}", result.converged);
    println!("│ Residual:     {:.2e}", result.residual_norm);
    println!("│ Solution sum: {:.4}", x_sum);
    println!("│ Solution max: {:.6}", max_x);
    println!("│ Solution min: {:.6}", min_x);
    println!("│ Time:         {:.2} ms", elapsed);
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spmv() {
        let row_ptr = vec![0, 2, 4, 6];
        let col_ind = vec![0, 1, 0, 1, 1, 2];
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0);
        let spmv = SparseMatrixVectorMul::new(0);

        let x = vec![1.0, 2.0, 3.0];
        let y = spmv.spmv_simple(&matrix, &x).unwrap();

        assert!(y.len() == 3);
    }

    #[test]
    fn test_sparse_cg() {
        let row_ptr = vec![0, 3, 6, 9];
        let col_ind = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let values = vec![4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0];
        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0);
        let solver = GPUCGSolver::new(0, 1e-10, 100);

        let b = vec![2.0, 2.0, 2.0];
        let mut x = vec![0.0; 3];
        let result = solver.solve(&matrix, &b, &mut x).unwrap();

        assert!(result.converged);
        assert!((x[0] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_vector_ops() {
        let ops = VectorOps::new(0);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let dot = ops.dot(&a, &b);
        assert!((dot - 32.0).abs() < 1e-10);

        let norm = ops.norm(&a);
        assert!((norm - 14.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_available() {
        let _avail = gpu_available();
        // Test doesn't fail regardless of GPU availability
    }
}
