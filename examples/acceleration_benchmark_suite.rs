//! Comprehensive benchmark suite for FEA acceleration algorithms.
//!
//! This benchmark suite provides:
//! - Performance benchmarks for all acceleration methods
//! - Scalability tests for different problem sizes
//! - Comparison against baseline methods
//! - Statistical analysis of results

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;
use std::collections::HashMap;

/// Benchmark result statistics.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub problem_size: usize,
    pub iterations: usize,
    pub elapsed_ms: f64,
    pub residual_norm: f64,
    pub converged: bool,
    pub iterations_per_second: f64,
    pub memory_mb: f64,
}

impl BenchmarkResult {
    pub fn efficiency(&self, baseline: &Self) -> f64 {
        if baseline.elapsed_ms > 0.0 {
            baseline.elapsed_ms / self.elapsed_ms
        } else {
            1.0
        }
    }
}

/// Benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub problem_sizes: Vec<usize>,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub num_runs: usize,
    pub warmup_runs: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            problem_sizes: vec![100, 500, 1000, 2000],
            max_iterations: 1000,
            tolerance: 1e-10,
            num_runs: 5,
            warmup_runs: 2,
        }
    }
}

/// Main benchmark runner.
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    results: HashMap<String, Vec<BenchmarkResult>>,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: HashMap::new(),
        }
    }

    /// Run all benchmarks.
    pub fn run_all(&mut self) -> anyhow::Result<()> {
        println!("=== FEA Acceleration Benchmark Suite ===\n");
        println!("Configuration:");
        println!("  Problem sizes: {:?}", self.config.problem_sizes);
        println!("  Max iterations: {}", self.config.max_iterations);
        println!("  Tolerance: {:.2e}", self.config.tolerance);
        println!("  Runs per test: {}\n", self.config.num_runs);

        self.benchmark_preconditioners()?;
        self.benchmark_acceleration_methods()?;
        self.benchmark_spectral_methods()?;
        self.benchmark_multigrid()?;
        self.scalability_test()?;

        self.print_summary();

        Ok(())
    }

    /// Benchmark different preconditioners.
    pub fn benchmark_preconditioners(&mut self) -> anyhow::Result<()> {
        println!("1. Preconditioner Benchmarks");
        println!("   Testing various preconditioning strategies...\n");

        let preconditioners = vec![
            ("None", Preconditioner::None),
            ("Jacobi", Preconditioner::Jacobi),
            ("Chebyshev(2)", Preconditioner::Chebyshev(2)),
            ("Chebyshev(3)", Preconditioner::Chebyshev(3)),
        ];

        for &n in &self.config.problem_sizes {
            let k = generate_stiffness_matrix(n);
            let f = DVector::from_element(n, 1.0);

            for (name, prec) in &preconditioners {
                let results = self.run_benchmark_series(
                    &format!("{}_prec_{}", name, n),
                    n,
                    |k, f, config| {
                        let cg = CGSolver::with_config(IterativeConfig {
                            max_iterations: config.max_iterations,
                            tolerance: config.tolerance,
                            preconditioner: *prec,
                            anderson_depth: 0,
                            krylov_dim: 0,
                            deflation_vectors: None,
                        });
                        cg.solve(k, f, &IterativeConfig::default())
                    },
                    &k,
                    &f,
                )?;

                self.results.insert(format!("{}_prec_{}", name, n), results);
            }
        }

        self.print_preconditioner_results();
        println!();

        Ok(())
    }

    /// Benchmark acceleration methods.
    pub fn benchmark_acceleration_methods(&mut self) -> anyhow::Result<()> {
        println!("2. Acceleration Method Benchmarks");
        println!("   Testing acceleration techniques...\n");

        for &n in &self.config.problem_sizes {
            let k = generate_stiffness_matrix(n);
            let f = DVector::from_element(n, 1.0);

            // Baseline: CG + Jacobi
            let baseline = self.run_benchmark_series(
                &format!("cg_jacobi_{}", n),
                n,
                |k, f, config| {
                    let cg = CGSolver::with_config(IterativeConfig {
                        max_iterations: config.max_iterations,
                        tolerance: config.tolerance,
                        preconditioner: Preconditioner::Jacobi,
                        anderson_depth: 0,
                        krylov_dim: 0,
                        deflation_vectors: None,
                    });
                    cg.solve(k, f, &IterativeConfig::default())
                },
                &k,
                &f,
            )?;

            self.results.insert(format!("cg_jacobi_{}", n), baseline.clone());

            // Anderson acceleration
            for &depth in &[3, 5, 10] {
                let results = self.run_benchmark_series(
                    &format!("anderson_{}_{}", depth, n),
                    n,
                    |k, f, config| {
                        let cg = CGSolver::with_config(IterativeConfig {
                            max_iterations: config.max_iterations,
                            tolerance: config.tolerance,
                            preconditioner: Preconditioner::Jacobi,
                            anderson_depth: depth,
                            krylov_dim: 0,
                            deflation_vectors: None,
                        });
                        cg.solve(k, f, &IterativeConfig::default())
                    },
                    &k,
                    &f,
                )?;

                self.results.insert(format!("anderson_{}_{}", depth, n), results);
            }

            // Krylov recycling
            for &dim in &[10, 20, 30] {
                let results = self.run_benchmark_series(
                    &format!("krylov_{}_{}", dim, n),
                    n,
                    |k, f, config| {
                        let cg = CGSolver::with_config(IterativeConfig {
                            max_iterations: config.max_iterations,
                            tolerance: config.tolerance,
                            preconditioner: Preconditioner::Jacobi,
                            anderson_depth: 0,
                            krylov_dim: dim,
                            deflation_vectors: None,
                        });
                        cg.solve(k, f, &IterativeConfig::default())
                    },
                    &k,
                    &f,
                )?;

                self.results.insert(format!("krylov_{}_{}", dim, n), results);
            }
        }

        self.print_acceleration_results();
        println!();

        Ok(())
    }

    /// Benchmark spectral methods.
    pub fn benchmark_spectral_methods(&mut self) -> anyhow::Result<()> {
        println!("3. Spectral Method Benchmarks");
        println!("   Testing spectral acceleration techniques...\n");

        for &n in &self.config.problem_sizes {
            let k = generate_stiffness_matrix(n);

            // Eigenvalue estimation
            let start = Instant::now();
            let (lambda_min, lambda_max) =
                ChebyshevSemiIterative::estimate_eigenvalues_gershgorin(&k);
            let gershgorin_time = start.elapsed();

            println!("   Size {}: Gershgorin eigenvalue estimation: {:.4} ms",
                     n, gershgorin_time.as_secs_f64() * 1000.0);
            println!("     lambda_min: {:.6e}, lambda_max: {:.6e}",
                     lambda_min, lambda_max);

            // Lanczos estimation
            let start = Instant::now();
            let (lambda_min_l, lambda_max_l) =
                ChebyshevSemiIterative::estimate_eigenvalues_lanczos(&k, 20);
            let lanczos_time = start.elapsed();

            println!("   Size {}: Lanczos eigenvalue estimation: {:.4} ms",
                     n, lanczos_time.as_secs_f64() * 1000.0);
            println!("     lambda_min: {:.6e}, lambda_max: {:.6e}",
                     lambda_min_l, lambda_max_l);

            // Spectral deflation
            let start = Instant::now();
            let deflation = SpectralDeflation::compute_deflation_subspace(&k, 5, 50);
            let deflation_setup = start.elapsed();

            println!("   Size {}: Spectral deflation setup: {:.4} ms",
                     n, deflation_setup.as_secs_f64() * 1000.0);
            println!("     Deflation vectors: {}", deflation.eigenvectors.ncols());

            println!();
        }

        Ok(())
    }

    /// Benchmark multigrid methods.
    pub fn benchmark_multigrid(&mut self) -> anyhow::Result<()> {
        println!("4. Multigrid Benchmarks");
        println!("   Testing multigrid solvers...\n");

        for &n in &self.config.problem_sizes {
            let k = generate_stiffness_matrix(n);
            let f = DVector::from_element(n, 1.0);

            // Full Multigrid
            let mg = FullMultigridSolver::new(&k, 4);
            let config = FullMultigridConfig::default();

            let start = Instant::now();
            let result = mg.solve(&k, &f, &config);
            let elapsed = start.elapsed();

            match result {
                Ok(res) => {
                    println!("   Size {}: Full Multigrid", n);
                    println!("     Iterations: {}", res.iterations.unwrap_or(0));
                    println!("     Time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
                    println!("     Residual: {:.6e}", res.residual_norm.unwrap_or(0.0));
                }
                Err(e) => {
                    println!("   Size {}: Full Multigrid - Error: {}", n, e);
                }
            }

            // V-cycle Multigrid preconditioner
            let start = Instant::now();
            let mut mg_prec = MultigridPreconditioner::new(&k, 3);
            let setup_time = start.elapsed();

            println!("   Size {}: Multigrid setup time: {:.4} ms",
                     n, setup_time.as_secs_f64() * 1000.0);

            println!();
        }

        Ok(())
    }

    /// Scalability test.
    pub fn scalability_test(&mut self) -> anyhow::Result<()> {
        println!("5. Scalability Test");
        println!("   Testing strong scaling with problem size...\n");

        let sizes = vec![100, 200, 500, 1000, 2000, 5000];
        println!("   {:<10} | {:>12} | {:>12} | {:>15} | {:>10}",
                 "Size", "Iterations", "Time (ms)", "Iter/Second", "Memory MB");
        println!("   {}", "-".repeat(70));

        for n in sizes {
            let k = generate_stiffness_matrix(n);
            let f = DVector::from_element(n, 1.0);

            let config = IterativeConfig {
                max_iterations: self.config.max_iterations,
                tolerance: self.config.tolerance,
                preconditioner: Preconditioner::Jacobi,
                anderson_depth: 0,
                krylov_dim: 0,
                deflation_vectors: None,
            };

            let cg = CGSolver::with_config(config.clone());
            let start = Instant::now();
            let result = cg.solve(&k, &f, &config)?;

            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let iterations = result.iterations.unwrap_or(1);
            let iter_per_sec = iterations as f64 / (elapsed / 1000.0);
            let memory_mb = (k.len() * 8) as f64 / (1024.0 * 1024.0);

            println!("   {:<10} | {:>12} | {:>12.4} | {:>15.2} | {:>10.4}",
                     n,
                     iterations,
                     elapsed,
                     iter_per_sec,
                     memory_mb);
        }

        println!();
        Ok(())
    }

    /// Run a series of benchmark runs.
    fn run_benchmark_series<F>(
        &self,
        name: &str,
        n: usize,
        mut solver: F,
        k: &DMatrix<f64>,
        f: &DVector<f64>,
    ) -> anyhow::Result<Vec<BenchmarkResult>>
    where
        F: FnMut(&DMatrix<f64>, &DVector<f64>, &BenchmarkConfig) -> anyhow::Result<SolverResult>,
    {
        let mut results = Vec::with_capacity(self.config.num_runs);

        // Warmup runs
        for _ in 0..self.config.warmup_runs {
            let _ = solver(k, f, &self.config);
        }

        // Actual runs
        for _ in 0..self.config.num_runs {
            let start = Instant::now();
            let result = solver(k, f, &self.config)?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let iterations = result.iterations.unwrap_or(1);
            let iter_per_sec = iterations as f64 / (elapsed / 1000.0);
            let memory_mb = (k.len() * 8) as f64 / (1024.0 * 1024.0);

            results.push(BenchmarkResult {
                name: name.to_string(),
                problem_size: n,
                iterations,
                elapsed_ms: elapsed,
                residual_norm: result.residual_norm.unwrap_or(0.0),
                converged: result.converged,
                iterations_per_second: iter_per_sec,
                memory_mb,
            });
        }

        Ok(results)
    }

    /// Print preconditioner results.
    fn print_preconditioner_results(&self) {
        println!("   {:<20} | {:>10} | {:>10} | {:>10} | {:>8}",
                 "Method", "Iters", "Time (ms)", "Residual", "Conv");
        println!("   {}", "-".repeat(65));

        for &n in &self.config.problem_sizes {
            println!("   Problem size: {}", n);
            for prec in &["None", "Jacobi", "Chebyshev(2)", "Chebyshev(3)"] {
                let key = format!("{}_prec_{}", prec.replace("(", "_").replace(")", ""), n);
                if let Some(results) = self.results.get(&key) {
                    let avg_iters = results.iter().map(|r| r.iterations).sum::<usize>() / results.len();
                    let avg_time = results.iter().map(|r| r.elapsed_ms).sum::<f64>() / results.len() as f64;
                    let avg_residual = results.iter().map(|r| r.residual_norm).sum::<f64>() / results.len() as f64;

                    println!("   {:<20} | {:>10} | {:>10.4} | {:>10.2e} | {:>8}",
                             prec,
                             avg_iters,
                             avg_time,
                             avg_residual,
                             if results[0].converged { "Yes" } else { "No" });
                }
            }
            println!();
        }
    }

    /// Print acceleration results.
    fn print_acceleration_results(&self) {
        println!("   {:<20} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8}",
                 "Method", "Iters", "Time (ms)", "Residual", "Speedup", "Eff");
        println!("   {}", "-".repeat(75));

        for &n in &self.config.problem_sizes {
            println!("   Problem size: {}", n);

            // Get baseline
            let baseline_key = format!("cg_jacobi_{}", n);
            let baseline = self.results.get(&baseline_key);
            let baseline_iters = baseline.map(|r| {
                r.iter().map(|res| res.iterations).sum::<usize>() / r.len()
            }).unwrap_or(1);

            for method in &["anderson_3", "anderson_5", "anderson_10",
                           "krylov_10", "krylov_20", "krylov_30"] {
                let key = format!("{}_{}", method, n);
                if let Some(results) = self.results.get(&key) {
                    let avg_iters = results.iter().map(|r| r.iterations).sum::<usize>() / results.len();
                    let avg_time = results.iter().map(|r| r.elapsed_ms).sum::<f64>() / results.len() as f64;
                    let avg_residual = results.iter().map(|r| r.residual_norm).sum::<f64>() / results.len() as f64;
                    let speedup = baseline_iters as f64 / avg_iters as f64;
                    let efficiency = if avg_iters > 0 { speedup } else { 1.0 };

                    println!("   {:<20} | {:>10} | {:>10.4} | {:>10.2e} | {:>8.2f}x | {:>8.2f}",
                             method.replace("_", " "),
                             avg_iters,
                             avg_time,
                             avg_residual,
                             speedup,
                             efficiency);
                }
            }
            println!();
        }
    }

    /// Print summary of all results.
    pub fn print_summary(&self) {
        println!("=== Benchmark Summary ===\n");

        let total_tests = self.results.len();
        let successful_tests = self.results.values()
            .flatten()
            .filter(|r| r.converged)
            .count();

        println!("Total test configurations: {}", total_tests);
        println!("Successful convergences: {}", successful_tests);
        println!("Convergence rate: {:.1}%",
                 100.0 * successful_tests as f64 / total_tests.max(1) as f64);

        // Find best performing methods
        println!("\nBest performing configurations by problem size:");
        for &n in &self.config.problem_sizes {
            let mut best_time = f64::INFINITY;
            let mut best_method = String::new();

            for (key, results) in &self.results {
                if key.contains(&format!("_{}", n)) && !key.contains("cg_jacobi") {
                    let avg_time = results.iter().map(|r| r.elapsed_ms).sum::<f64>() / results.len() as f64;
                    if avg_time < best_time {
                        best_time = avg_time;
                        best_method = key.clone();
                    }
                }
            }

            if !best_method.is_empty() {
                println!("  Size {}: {} (avg time: {:.4} ms)",
                         n, best_method.replace("_", " "), best_time);
            }
        }
    }
}

/// Generate a stiffness-like matrix.
fn generate_stiffness_matrix(n: usize) -> DMatrix<f64> {
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

fn main() -> anyhow::Result<()> {
    let config = BenchmarkConfig {
        problem_sizes: vec![100, 500, 1000],
        max_iterations: 500,
        tolerance: 1e-10,
        num_runs: 3,
        warmup_runs: 1,
    };

    let mut runner = BenchmarkRunner::new(config);
    runner.run_all()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runner() {
        let config = BenchmarkConfig {
            problem_sizes: vec![50, 100],
            max_iterations: 200,
            tolerance: 1e-8,
            num_runs: 2,
            warmup_runs: 1,
        };

        let mut runner = BenchmarkRunner::new(config);
        runner.run_all().unwrap();
    }
}
