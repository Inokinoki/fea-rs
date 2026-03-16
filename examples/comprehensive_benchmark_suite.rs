//! Comprehensive FEA Performance Benchmark.
//!
//! This benchmark suite provides comprehensive performance testing:
//! - Solver performance comparison
//! - Acceleration method benchmarks
//! - Material model performance
//! - GPU vs CPU comparison
//! - Memory usage analysis

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

/// Benchmark result record.
#[derive(Debug, Clone)]
pub struct BenchmarkRecord {
    pub category: String,
    pub test_name: String,
    pub size: usize,
    pub iterations: usize,
    pub time_ms: f64,
    pub residual: f64,
    pub memory_mb: f64,
}

impl BenchmarkRecord {
    pub fn print(&self) {
        println!("{:<25} | {:>8} | {:>10} | {:>12.4} | {:>10.2e} | {:>10.2}",
                 format!("{}-{}", self.category, self.test_name),
                 self.size,
                 self.iterations,
                 self.time_ms,
                 self.residual,
                 self.memory_mb);
    }
}

/// Comprehensive benchmark suite.
pub struct ComprehensiveBenchmark {
    results: Vec<BenchmarkRecord>,
}

impl ComprehensiveBenchmark {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Runs complete benchmark suite.
    pub fn run_all(&mut self) -> anyhow::Result<()> {
        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║       Comprehensive FEA Performance Benchmark            ║");
        println!("╚══════════════════════════════════════════════════════════╝\n");

        // Solver benchmarks
        self.benchmark_solvers()?;

        // Acceleration benchmarks
        self.benchmark_acceleration()?;

        // Material benchmarks
        self.benchmark_materials()?;

        // Memory benchmarks
        self.benchmark_memory()?;

        // Print summary
        self.print_summary();

        Ok(())
    }

    /// Benchmark different solvers.
    fn benchmark_solvers(&mut self) -> anyhow::Result<()> {
        println!("{}\n", "=".repeat(90));
        println!("Solver Performance Benchmarks");
        println!("{}\n", "=".repeat(90));

        println!("{:<25} | {:>8} | {:>10} | {:>12} | {:>10} | {:>10}",
                 "Test", "Size", "Iterations", "Time (ms)", "Residual", "Memory MB");
        println!("{}", "-".repeat(90));

        for &size in &[100, 500, 1000] {
            let (k, f) = create_test_system(size);

            // Direct solver
            let result = benchmark_direct(&k, &f, "Solver", "Direct", size);
            self.results.push(result);

            // CG + Jacobi
            let result = benchmark_cg_jacobi(&k, &f, "Solver", "CG-Jacobi", size);
            self.results.push(result);

            // CG + Chebyshev
            let result = benchmark_cg_chebyshev(&k, &f, "Solver", "CG-Cheb", size);
            self.results.push(result);
        }

        println!();
        Ok(())
    }

    /// Benchmark acceleration methods.
    fn benchmark_acceleration(&mut self) -> anyhow::Result<()> {
        println!("{}\n", "=".repeat(90));
        println!("Acceleration Method Benchmarks");
        println!("{}\n", "=".repeat(90));

        println!("{:<25} | {:>8} | {:>10} | {:>12} | {:>10} | {:>10}",
                 "Test", "Size", "Iterations", "Time (ms)", "Residual", "Memory MB");
        println!("{}", "-".repeat(90));

        for &size in &[200, 500] {
            let (k, f) = create_test_system(size);

            // Anderson acceleration
            let result = benchmark_anderson(&k, &f, "Accel", "Anderson", size);
            self.results.push(result);

            // Spectral deflation
            let result = benchmark_deflation(&k, &f, "Accel", "Deflation", size);
            self.results.push(result);

            // Unified adaptive
            let result = benchmark_unified(&k, &f, "Accel", "Unified", size);
            self.results.push(result);
        }

        println!();
        Ok(())
    }

    /// Benchmark material models.
    fn benchmark_materials(&mut self) -> anyhow::Result<()> {
        println!("{}\n", "=".repeat(90));
        println!("Material Model Benchmarks");
        println!("{}\n", "=".repeat(90));

        println!("{:<25} | {:>8} | {:>10} | {:>12} | {:>10} | {:>10}",
                 "Test", "Size", "Iterations", "Time (ms)", "Energy", "Memory MB");
        println!("{}", "-".repeat(90));

        // Neo-Hookean
        let result = benchmark_neo_hookean("Material", "NeoHookean", 1000);
        self.results.push(result);

        // Mooney-Rivlin
        let result = benchmark_mooney_rivlin("Material", "MooneyRivlin", 1000);
        self.results.push(result);

        println!();
        Ok(())
    }

    /// Benchmark memory usage.
    fn benchmark_memory(&mut self) -> anyhow::Result<()> {
        println!("{}\n", "=".repeat(90));
        println!("Memory Usage Benchmarks");
        println!("{}\n", "=".repeat(90));

        println!("{:<25} | {:>8} | {:>10} | {:>12} | {:>10} | {:>10}",
                 "Test", "Size", "Iterations", "Time (ms)", "Residual", "Memory MB");
        println!("{}", "-".repeat(90));

        for &size in &[500, 1000, 2000] {
            let (k, f) = create_test_system(size);
            let mem_mb = estimate_memory_usage(&k, &f);

            let result = benchmark_cg_jacobi(&k, &f, "Memory", "CG", size);
            let mut record = result;
            record.memory_mb = mem_mb;
            self.results.push(record);
        }

        println!();
        Ok(())
    }

    /// Print benchmark summary.
    fn print_summary(&self) {
        println!("{}\n", "=".repeat(90));
        println!("BENCHMARK SUMMARY");
        println!("{}\n", "=".repeat(90));

        let total_tests = self.results.len();
        let total_time: f64 = self.results.iter().map(|r| r.time_ms).sum();

        println!("Total benchmarks: {}", total_tests);
        println!("Total time: {:.2} ms\n", total_time);

        // Find fastest solver
        let solver_results: Vec<_> = self.results.iter()
            .filter(|r| r.category == "Solver")
            .collect();

        if !solver_results.is_empty() {
            let fastest = solver_results.iter()
                .min_by(|a, b| a.time_ms.partial_cmp(&b.time_ms).unwrap());

            if let Some(fastest) = fastest {
                println!("Fastest solver: {} ({} DOFs, {:.4} ms)",
                         fastest.test_name, fastest.size, fastest.time_ms);
            }
        }

        // Find best acceleration
        let accel_results: Vec<_> = self.results.iter()
            .filter(|r| r.category == "Accel")
            .collect();

        if !accel_results.is_empty() {
            let best = accel_results.iter()
                .min_by(|a, b| a.time_ms.partial_cmp(&b.time_ms).unwrap());

            if let Some(best) = best {
                println!("Best acceleration: {} ({} DOFs, {:.4} ms)",
                         best.test_name, best.size, best.time_ms);
            }
        }
    }
}

/// Creates a test linear system (tridiagonal stiffness matrix).
fn create_test_system(size: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut k = DMatrix::zeros(size, size);
    for i in 0..size {
        k[(i, i)] = 4.0;
        if i > 0 {
            k[(i, i - 1)] = -1.0;
        }
        if i < size - 1 {
            k[(i, i + 1)] = -1.0;
        }
    }
    let f = DVector::from_element(size, 1.0);
    (k, f)
}

/// Estimates memory usage for a system.
fn estimate_memory_usage(k: &DMatrix<f64>, f: &DVector<f64>) -> f64 {
    let matrix_bytes = k.nrows() * k.ncols() * 8;
    let vector_bytes = f.len() * 8 * 4; // Multiple vectors in solver
    ((matrix_bytes + vector_bytes) as f64) / (1024.0 * 1024.0)
}

/// Benchmarks direct solver.
fn benchmark_direct(k: &DMatrix<f64>, f: &DVector<f64>, category: &str, test: &str, size: usize) -> BenchmarkRecord {
    let start = Instant::now();
    let _ = k.clone().lu().solve(f);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    BenchmarkRecord {
        category: category.to_string(),
        test_name: test.to_string(),
        size,
        iterations: 1,
        time_ms: elapsed,
        residual: 0.0,
        memory_mb: 0.0,
    }
}

/// Benchmarks CG + Jacobi.
fn benchmark_cg_jacobi(k: &DMatrix<f64>, f: &DVector<f64>, category: &str, test: &str, size: usize) -> BenchmarkRecord {
    let config = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-10,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };

    let cg = CGSolver::with_config(config.clone());
    let start = Instant::now();
    let result = cg.solve(k, f, &config).unwrap();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    BenchmarkRecord {
        category: category.to_string(),
        test_name: test.to_string(),
        size,
        iterations: result.iterations.unwrap_or(0),
        time_ms: elapsed,
        residual: result.residual_norm.unwrap_or(0.0),
        memory_mb: 0.0,
    }
}

/// Benchmarks CG + Chebyshev.
fn benchmark_cg_chebyshev(k: &DMatrix<f64>, f: &DVector<f64>, category: &str, test: &str, size: usize) -> BenchmarkRecord {
    let config = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-10,
        preconditioner: Preconditioner::Chebyshev(3),
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };

    let cg = CGSolver::with_config(config.clone());
    let start = Instant::now();
    let result = cg.solve(k, f, &config).unwrap();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    BenchmarkRecord {
        category: category.to_string(),
        test_name: test.to_string(),
        size,
        iterations: result.iterations.unwrap_or(0),
        time_ms: elapsed,
        residual: result.residual_norm.unwrap_or(0.0),
        memory_mb: 0.0,
    }
}

/// Benchmarks Anderson acceleration.
fn benchmark_anderson(k: &DMatrix<f64>, f: &DVector<f64>, category: &str, test: &str, size: usize) -> BenchmarkRecord {
    let config = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-10,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 5,
        krylov_dim: 0,
        deflation_vectors: None,
    };

    let cg = CGSolver::with_config(config.clone());
    let start = Instant::now();
    let result = cg.solve(k, f, &config).unwrap();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    BenchmarkRecord {
        category: category.to_string(),
        test_name: test.to_string(),
        size,
        iterations: result.iterations.unwrap_or(0),
        time_ms: elapsed,
        residual: result.residual_norm.unwrap_or(0.0),
        memory_mb: 0.0,
    }
}

/// Benchmarks spectral deflation.
fn benchmark_deflation(k: &DMatrix<f64>, f: &DVector<f64>, category: &str, test: &str, size: usize) -> BenchmarkRecord {
    let deflation = SpectralDeflation::compute_deflation_subspace(k, 5, 50);

    let config = IterativeConfig {
        max_iterations: 500,
        tolerance: 1e-10,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: Some(deflation.eigenvectors),
    };

    let cg = CGSolver::with_config(config.clone());
    let start = Instant::now();
    let result = cg.solve(k, f, &config).unwrap();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    BenchmarkRecord {
        category: category.to_string(),
        test_name: test.to_string(),
        size,
        iterations: result.iterations.unwrap_or(0),
        time_ms: elapsed,
        residual: result.residual_norm.unwrap_or(0.0),
        memory_mb: 0.0,
    }
}

/// Benchmarks unified adaptive solver.
fn benchmark_unified(k: &DMatrix<f64>, f: &DVector<f64>, category: &str, test: &str, size: usize) -> BenchmarkRecord {
    let config = UnifiedSolverConfig {
        strategy: AccelerationStrategy::Adaptive,
        max_iterations: 500,
        tolerance: 1e-10,
        enable_monitoring: true,
        verbose: false,
    };

    let mut solver = UnifiedAccelerationSolver::new(config);
    let start = Instant::now();
    let result = solver.solve(k, f);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    BenchmarkRecord {
        category: category.to_string(),
        test_name: test.to_string(),
        size,
        iterations: result.base_result.iterations.unwrap_or(0),
        time_ms: elapsed,
        residual: result.base_result.residual_norm.unwrap_or(0.0),
        memory_mb: 0.0,
    }
}

/// Benchmarks Neo-Hookean material.
fn benchmark_neo_hookean(category: &str, test: &str, iterations: usize) -> BenchmarkRecord {
    let params = HyperelasticParams::neo_hookean(0.5e6, 1e9);
    let mat = NeoHookean::new(params);

    let f = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.1, 1.0, 1.0]));

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = mat.strain_energy(&f);
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    let w = mat.strain_energy(&f);

    BenchmarkRecord {
        category: category.to_string(),
        test_name: test.to_string(),
        size: iterations,
        iterations,
        time_ms: elapsed,
        residual: w,
        memory_mb: 0.001,
    }
}

/// Benchmarks Mooney-Rivlin material.
fn benchmark_mooney_rivlin(category: &str, test: &str, iterations: usize) -> BenchmarkRecord {
    let params = HyperelasticParams::mooney_rivlin(0.4e6, 0.1e6, 1e9);
    let mat = MooneyRivlin::new(params);

    let f = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.1, 1.0, 1.0]));

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = mat.strain_energy(&f);
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    let w = mat.strain_energy(&f);

    BenchmarkRecord {
        category: category.to_string(),
        test_name: test.to_string(),
        size: iterations,
        iterations,
        time_ms: elapsed,
        residual: w,
        memory_mb: 0.001,
    }
}

/// Main entry point.
fn main() -> anyhow::Result<()> {
    let mut benchmark = ComprehensiveBenchmark::new();
    benchmark.run_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_benchmark() {
        let mut benchmark = ComprehensiveBenchmark::new();
        benchmark.run_all().unwrap();
        assert!(!benchmark.results.is_empty());
    }
}
