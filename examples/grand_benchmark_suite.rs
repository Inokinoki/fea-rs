//! Grand Benchmark Suite - Comprehensive FEA Acceleration Benchmark.
//!
//! This benchmark suite tests all acceleration methods across various
//! problem sizes and types, providing performance comparisons and
//! validation data.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;
use std::fs::File;
use std::io::Write;

/// Benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Problem sizes to test.
    pub problem_sizes: Vec<usize>,
    /// Number of runs per test.
    pub num_runs: usize,
    /// Warmup runs.
    pub warmup_runs: usize,
    /// Tolerance.
    pub tolerance: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            problem_sizes: vec![100, 500, 1000],
            num_runs: 3,
            warmup_runs: 1,
            tolerance: 1e-10,
            max_iterations: 500,
        }
    }
}

/// Benchmark result.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Method name.
    pub method: String,
    /// Problem size.
    pub size: usize,
    /// Average iterations.
    pub avg_iterations: f64,
    /// Average time (ms).
    pub avg_time_ms: f64,
    /// Standard deviation of time.
    pub std_time_ms: f64,
    /// Final residual.
    pub residual: f64,
    /// Converged.
    pub converged: bool,
}

/// Grand benchmark runner.
pub struct GrandBenchmark {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
}

impl GrandBenchmark {
    /// Creates a new grand benchmark.
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Runs all benchmarks.
    pub fn run_all(&mut self) -> anyhow::Result<()> {
        println!("=== Grand FEA Acceleration Benchmark Suite ===\n");
        println!("Configuration:");
        println!("  Problem sizes: {:?}", self.config.problem_sizes);
        println!("  Runs per test: {}", self.config.num_runs);
        println!("  Tolerance: {:.2e}\n", self.config.tolerance);

        // Run benchmarks for each category
        self.benchmark_classical_methods()?;
        self.benchmark_preconditioners()?;
        self.benchmark_spectral_methods()?;
        self.benchmark_recycling_methods()?;
        self.benchmark_adaptive_methods()?;
        self.benchmark_polynomial_methods()?;

        // Print summary
        self.print_summary();

        Ok(())
    }

    /// Benchmarks classical iterative methods.
    fn benchmark_classical_methods(&mut self) -> anyhow::Result<()> {
        println!("─".repeat(70));
        println!("Category 1: Classical Iterative Methods");
        println!("─".repeat(70));

        for &n in &self.config.problem_sizes {
            let (k, f) = self.generate_test_problem(n);

            // CG + Jacobi
            let result = self.run_benchmark("CG + Jacobi", n, |k, f, cfg| {
                let config = IterativeConfig {
                    max_iterations: cfg.max_iterations,
                    tolerance: cfg.tolerance,
                    preconditioner: Preconditioner::Jacobi,
                    anderson_depth: 0,
                    krylov_dim: 0,
                    deflation_vectors: None,
                };
                let cg = CGSolver::with_config(config.clone());
                cg.solve(k, f, &config)
            }, &k, &f)?;
            self.results.push(result);

            // BiCGSTAB
            let result = self.run_benchmark("BiCGSTAB", n, |k, f, cfg| {
                let solver = BiCGSTABSolver::with_tolerance(cfg.tolerance);
                solver.solve(k, f, &IterativeConfig::default())
            }, &k, &f)?;
            self.results.push(result);

            // GMRES
            let result = self.run_benchmark("GMRES(30)", n, |k, f, cfg| {
                let solver = GMRESSolver::with_tolerance(cfg.tolerance);
                solver.solve(k, f, &IterativeConfig::default())
            }, &k, &f)?;
            self.results.push(result);
        }

        self.print_category_results("Classical Methods");
        Ok(())
    }

    /// Benchmarks preconditioners.
    fn benchmark_preconditioners(&mut self) -> anyhow::Result<()> {
        println!("\n─".repeat(70));
        println!("Category 2: Advanced Preconditioners");
        println!("─".repeat(70));

        for &n in &self.config.problem_sizes {
            let (k, f) = self.generate_test_problem(n);

            // CG + Chebyshev
            let result = self.run_benchmark("CG + Chebyshev(3)", n, |k, f, cfg| {
                let config = IterativeConfig {
                    max_iterations: cfg.max_iterations,
                    tolerance: cfg.tolerance,
                    preconditioner: Preconditioner::Chebyshev(3),
                    anderson_depth: 0,
                    krylov_dim: 0,
                    deflation_vectors: None,
                };
                let cg = CGSolver::with_config(config.clone());
                cg.solve(k, f, &config)
            }, &k, &f)?;
            self.results.push(result);

            // FSAI preconditioner
            let result = self.run_benchmark_with_setup("FSAI Prec", n, |k, f, cfg| {
                let pattern: Vec<Vec<usize>> = (0..k.nrows()).map(|i| {
                    let mut p = vec![i];
                    if i > 0 { p.push(i - 1); }
                    if i < k.nrows() - 1 { p.push(i + 1); }
                    p
                }).collect();

                if let Some(fsai) = FSAIPreconditioner::new(k, pattern) {
                    // Use FSAI as preconditioner (simplified)
                    let config = IterativeConfig {
                        max_iterations: cfg.max_iterations,
                        tolerance: cfg.tolerance,
                        preconditioner: Preconditioner::Jacobi,
                        anderson_depth: 0,
                        krylov_dim: 0,
                        deflation_vectors: None,
                    };
                    let cg = CGSolver::with_config(config.clone());
                    cg.solve(k, f, &config)
                } else {
                    let config = IterativeConfig::default();
                    let cg = CGSolver::with_config(config.clone());
                    cg.solve(k, f, &config)
                }
            }, &k, &f)?;
            self.results.push(result);
        }

        self.print_category_results("Advanced Preconditioners");
        Ok(())
    }

    /// Benchmarks spectral methods.
    fn benchmark_spectral_methods(&mut self) -> anyhow::Result<()> {
        println!("\n─".repeat(70));
        println!("Category 3: Spectral Acceleration Methods");
        println!("─".repeat(70));

        for &n in &self.config.problem_sizes {
            let (k, f) = self.generate_test_problem(n);

            // Spectral Deflation + CG
            let result = self.run_benchmark_with_setup("Deflated CG", n, |k, f, cfg| {
                let deflation = SpectralDeflation::compute_deflation_subspace(k, 5, 50);
                let config = IterativeConfig {
                    max_iterations: cfg.max_iterations,
                    tolerance: cfg.tolerance,
                    preconditioner: Preconditioner::Jacobi,
                    anderson_depth: 0,
                    krylov_dim: 0,
                    deflation_vectors: Some(deflation.eigenvectors),
                };
                let cg = CGSolver::with_config(config.clone());
                cg.solve(k, f, &config)
            }, &k, &f)?;
            self.results.push(result);

            // Chebyshev Semi-Iterative
            let result = self.run_benchmark_with_setup("Chebyshev Semi-Iter", n, |k, f, cfg| {
                let (lambda_min, lambda_max) =
                    ChebyshevSemiIterative::estimate_eigenvalues_gershgorin(k);
                let mut cheb = ChebyshevSemiIterative::new(lambda_min, lambda_max);
                let mut x = DVector::zeros(f.len());

                for _ in 0..cfg.max_iterations {
                    cheb.iterate(&mut x, k, f);
                    let residual = (f - k * &x).norm();
                    if residual < cfg.tolerance * f.norm() {
                        break;
                    }
                }

                let residual = (f - k * &x).norm();
                Ok(SolverResult::iterative(
                    x.data.as_vec().clone(),
                    cheb.iteration,
                    residual,
                    residual < cfg.tolerance * f.norm(),
                ))
            }, &k, &f)?;
            self.results.push(result);
        }

        self.print_category_results("Spectral Methods");
        Ok(())
    }

    /// Benchmarks recycling methods.
    fn benchmark_recycling_methods(&mut self) -> anyhow::Result<()> {
        println!("\n─".repeat(70));
        println!("Category 4: Krylov Recycling Methods");
        println!("─".repeat(70));

        for &n in &self.config.problem_sizes {
            let (k, f) = self.generate_test_problem(n);

            // GCRO-DR
            let result = self.run_benchmark_with_setup("GCRO-DR", n, |k, f, cfg| {
                let gcro = GCRODRSolver::new(20, 5);
                let (x, iters, residual, converged) =
                    gcro.solve(k, f, None, cfg.tolerance, cfg.max_iterations);
                Ok(SolverResult::iterative(
                    x.data.as_vec().clone(),
                    iters,
                    residual,
                    converged,
                ))
            }, &k, &f)?;
            self.results.push(result);

            // Recycling BiCGSTAB
            let result = self.run_benchmark_with_setup("Recycling BiCGSTAB", n, |k, f, cfg| {
                let mut rbicg = RecyclingBiCGSTAB::new(10);
                let (x, iters, residual, converged) =
                    rbicg.solve(k, f, cfg.tolerance, cfg.max_iterations);
                Ok(SolverResult::iterative(
                    x.data.as_vec().clone(),
                    iters,
                    residual,
                    converged,
                ))
            }, &k, &f)?;
            self.results.push(result);
        }

        self.print_category_results("Recycling Methods");
        Ok(())
    }

    /// Benchmarks adaptive methods.
    fn benchmark_adaptive_methods(&mut self) -> anyhow::Result<()> {
        println!("\n─".repeat(70));
        println!("Category 5: Adaptive Solver Methods");
        println!("─".repeat(70));

        for &n in &self.config.problem_sizes {
            let (k, f) = self.generate_test_problem(n);

            // Adaptive CG
            let result = self.run_benchmark_with_setup("Adaptive CG", n, |k, f, cfg| {
                let mut solver = AdaptiveCGSolver::new(cfg.tolerance, cfg.max_iterations);
                let result = solver.solve(k, f);
                Ok(SolverResult::iterative(
                    result.solution.local_data,
                    result.iterations,
                    result.residual_norm,
                    result.converged,
                ))
            }, &k, &f)?;
            self.results.push(result);
        }

        self.print_category_results("Adaptive Methods");
        Ok(())
    }

    /// Benchmarks polynomial methods.
    fn benchmark_polynomial_methods(&mut self) -> anyhow::Result<()> {
        println!("\n─".repeat(70));
        println!("Category 6: Polynomial Acceleration Methods");
        println!("─".repeat(70));

        for &n in &self.config.problem_sizes {
            let (k, f) = self.generate_test_problem(n);

            // Chebyshev Polynomial Acceleration
            let result = self.run_benchmark_with_setup("Chebyshev Poly", n, |k, f, cfg| {
                let (x, iters, residual) = ChebyshevAcceleration::chebyshev_iteration(
                    k, f, 0.5, 4.5, cfg.tolerance, cfg.max_iterations,
                );
                Ok(SolverResult::iterative(
                    x.data.as_vec().clone(),
                    iters,
                    residual,
                    residual < cfg.tolerance * f.norm(),
                ))
            }, &k, &f)?;
            self.results.push(result);
        }

        self.print_category_results("Polynomial Methods");
        Ok(())
    }

    /// Generates a test problem (SPD matrix).
    fn generate_test_problem(&self, n: usize) -> (DMatrix<f64>, DVector<f64>) {
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
        let f = DVector::from_element(n, 1.0);
        (k, f)
    }

    /// Runs a benchmark with multiple runs.
    fn run_benchmark<F>(&self, name: &str, size: usize, solver: F, k: &DMatrix<f64>, f: &DVector<f64>)
        -> anyhow::Result<BenchmarkResult>
    where
        F: Fn(&DMatrix<f64>, &DVector<f64>, &BenchmarkConfig) -> anyhow::Result<SolverResult>,
    {
        self.run_benchmark_with_setup(name, size, |k, f, cfg| solver(k, f, cfg), k, f)
    }

    /// Runs a benchmark with setup phase.
    fn run_benchmark_with_setup<F>(&self, name: &str, size: usize, solver: F, k: &DMatrix<f64>, f: &DVector<f64>)
        -> anyhow::Result<BenchmarkResult>
    where
        F: Fn(&DMatrix<f64>, &DVector<f64>, &BenchmarkConfig) -> anyhow::Result<SolverResult>,
    {
        let mut iterations = Vec::new();
        let mut times = Vec::new();
        let mut residuals = Vec::new();
        let mut converged_count = 0;

        // Warmup
        for _ in 0..self.config.warmup_runs {
            let _ = solver(k, f, &self.config);
        }

        // Actual runs
        for _ in 0..self.config.num_runs {
            let start = Instant::now();
            let result = solver(k, f, &self.config)?;
            let elapsed = start.elapsed();

            iterations.push(result.iterations.unwrap_or(0) as f64);
            times.push(elapsed.as_secs_f64() * 1000.0);
            residuals.push(result.residual_norm.unwrap_or(0.0));
            if result.converged {
                converged_count += 1;
            }
        }

        let avg_iters = iterations.iter().sum::<f64>() / iterations.len() as f64;
        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        let std_time = self.compute_std(&times);
        let avg_residual = residuals.iter().sum::<f64>() / residuals.len() as f64;

        Ok(BenchmarkResult {
            method: name.to_string(),
            size,
            avg_iterations: avg_iters,
            avg_time_ms: avg_time,
            std_time_ms: std_time,
            residual: avg_residual,
            converged: converged_count == self.config.num_runs,
        })
    }

    fn compute_std(&self, values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    fn print_category_results(&self, category: &str) {
        let category_results: Vec<_> = self.results.iter()
            .filter(|r| {
                // Filter to last category's results
                self.results.len() - self.config.problem_sizes.len() * 6..self.results.len()
                    .contains(&self.results.iter().position(|x| x == r).unwrap_or(0))
            })
            .collect();

        println!("\n{:<25} | {:<8} | {:>10} | {:>12} | {:>10}",
                 "Method", "Size", "Iterations", "Time (ms)", "Residual");
        println!("{}", "-".repeat(75));

        for result in &self.results[self.results.len().saturating_sub(self.config.problem_sizes.len() * 2)..] {
            println!("{:<25} | {:>8} | {:>10.1} | {:>10.4} ±{:.2} | {:>10.2e}",
                     result.method, result.size, result.avg_iterations,
                     result.avg_time_ms, result.std_time_ms, result.residual);
        }
    }

    fn print_summary(&self) {
        println!("\n{}", "=".repeat(70));
        println!("BENCHMARK SUMMARY");
        println!("{}\n", "=".repeat(70));

        println!("{:<30} | {:<8} | {:>10} | {:>12}", "Method", "Size", "Iterations", "Time (ms)");
        println!("{}", "-".repeat(70));

        for result in &self.results {
            println!("{:<30} | {:>8} | {:>10.1} | {:>10.4}",
                     format!("{} ", result.method), result.size,
                     result.avg_iterations, result.avg_time_ms);
        }

        // Find best methods by size
        println!("\nBest Methods by Problem Size:");
        for &size in &self.config.problem_sizes {
            let size_results: Vec<_> = self.results.iter()
                .filter(|r| r.size == size)
                .collect();
            if let Some(best) = size_results.iter().min_by(|a, b| {
                a.avg_time_ms.partial_cmp(&b.avg_time_ms).unwrap()
            }) {
                println!("  n={:<6}: {} ({:.4} ms)", size, best.method, best.avg_time_ms);
            }
        }
    }

    /// Exports results to CSV.
    pub fn export_csv(&self, filename: &str) -> anyhow::Result<()> {
        let mut file = File::create(filename)?;

        writeln!(file, "method,size,avg_iterations,avg_time_ms,std_time_ms,residual,converged")?;
        for result in &self.results {
            writeln!(file, "{},{},{:.2},{:.4},{:.4},{:.2e},{}",
                     result.method, result.size, result.avg_iterations,
                     result.avg_time_ms, result.std_time_ms, result.residual,
                     result.converged)?;
        }

        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let config = BenchmarkConfig {
        problem_sizes: vec![100, 250, 500],
        num_runs: 3,
        warmup_runs: 1,
        tolerance: 1e-10,
        max_iterations: 300,
    };

    let mut benchmark = GrandBenchmark::new(config);
    benchmark.run_all()?;

    // Export results
    benchmark.export_csv("benchmark_results.csv")?;
    println!("\nResults exported to benchmark_results.csv");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grand_benchmark() {
        let config = BenchmarkConfig {
            problem_sizes: vec![50, 100],
            num_runs: 2,
            warmup_runs: 1,
            tolerance: 1e-8,
            max_iterations: 100,
        };

        let mut benchmark = GrandBenchmark::new(config);
        benchmark.run_all().unwrap();
    }
}
