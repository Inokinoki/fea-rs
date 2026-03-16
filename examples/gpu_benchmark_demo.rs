//! GPU Benchmark Demo.

use fea::gpu::{gpu_available, GPUCGSolver, GPUCSRMatrix, VectorOps};

fn main() -> anyhow::Result<()> {
    println!("=== GPU Benchmark Demo ===\n");
    println!("GPU Available: {}", gpu_available());
    
    benchmark_system(100, "Small")?;
    benchmark_system(500, "Medium")?;
    
    println!("\nBenchmark Complete!");
    Ok(())
}

fn benchmark_system(n: usize, label: &str) -> anyhow::Result<()> {
    let mut row_ptr = Vec::new();
    let mut col_ind = Vec::new();
    let mut values = Vec::new();
    
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 { col_ind.push(i - 1); values.push(-1.0_f64); }
        col_ind.push(i); values.push(2.0_f64);
        if i < n - 1 { col_ind.push(i + 1); values.push(-1.0_f64); }
        row_ptr.push(col_ind.len());
    }
    
    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
    let solver = GPUCGSolver::new(0, 1e-8, 500);
    let b = vec![1.0_f64; n];
    let mut x = vec![0.0_f64; n];
    
    let start = std::time::Instant::now();
    let result = solver.solve(&matrix, &b, &mut x)?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    
    println!("{} ({} DOF): {} iter, {:.1} ms", label, n, result.iterations, ms);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_benchmark() {
        let n = 50;
        let mut row_ptr = Vec::new();
        let mut col_ind = Vec::new();
        let mut values = Vec::new();
        
        row_ptr.push(0);
        for i in 0..n {
            if i > 0 { col_ind.push(i - 1); values.push(-1.0_f64); }
            col_ind.push(i); values.push(2.0_f64);
            if i < n - 1 { col_ind.push(i + 1); values.push(-1.0_f64); }
            row_ptr.push(col_ind.len());
        }
        
        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
        let solver = GPUCGSolver::new(0, 1e-6, 100);
        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let result = solver.solve(&matrix, &b, &mut x);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_vector_ops() {
        let ops = VectorOps::new(0);
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![4.0_f64, 5.0, 6.0];
        let dot = ops.dot(&a, &b);
        assert!((dot - 32.0_f64).abs() < 1e-10);
    }
}
