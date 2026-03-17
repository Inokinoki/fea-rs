//! Acceleration Performance Utilities.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! This module provides utilities for measuring, analyzing, and optimizing
//! the performance of acceleration methods.

use std::time::Instant;
use nalgebra::{DMatrix, DVector};

/// Performance metrics for a solver run.
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Number of iterations.
    pub iterations: usize,
    /// Total time in milliseconds.
    pub time_ms: f64,
    /// Matrix-vector products count.
    pub matvec_count: usize,
    /// Final residual norm.
    pub residual_norm: f64,
    /// Convergence achieved.
    pub converged: bool,
    /// Memory usage estimate (MB).
    pub memory_mb: f64,
}

impl PerformanceMetrics {
    /// Creates new performance metrics.
    pub fn new() -> Self {
        Self {
            iterations: 0,
            time_ms: 0.0,
            matvec_count: 0,
            residual_norm: 0.0,
            converged: false,
            memory_mb: 0.0,
        }
    }

    /// Computes iterations per second.
    pub fn iterations_per_second(&self) -> f64 {
        if self.time_ms > 0.0 {
            self.iterations as f64 / (self.time_ms / 1000.0)
        } else {
            0.0
        }
    }

    /// Computes matvecs per second.
    pub fn matvecs_per_second(&self) -> f64 {
        if self.time_ms > 0.0 {
            self.matvec_count as f64 / (self.time_ms / 1000.0)
        } else {
            0.0
        }
    }

    /// Computes time per iteration.
    pub fn time_per_iteration(&self) -> f64 {
        if self.iterations > 0 {
            self.time_ms / self.iterations as f64
        } else {
            0.0
        }
    }

    /// Computes time per matvec.
    pub fn time_per_matvec(&self) -> f64 {
        if self.matvec_count > 0 {
            self.time_ms / self.matvec_count as f64
        } else {
            0.0
        }
    }

    /// Computes speedup compared to baseline.
    pub fn speedup(&self, baseline: &PerformanceMetrics) -> f64 {
        if self.time_ms > 0.0 {
            baseline.time_ms / self.time_ms
        } else {
            f64::INFINITY
        }
    }

    /// Computes efficiency (speedup / cores).
    pub fn efficiency(&self, baseline: &PerformanceMetrics, cores: usize) -> f64 {
        let speedup = self.speedup(baseline);
        if cores > 0 {
            speedup / cores as f64
        } else {
            speedup
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance comparison between two solver configurations.
#[derive(Debug, Clone)]
pub struct PerformanceComparison {
    /// Name of baseline method.
    pub baseline_name: String,
    /// Name of accelerated method.
    pub accelerated_name: String,
    /// Baseline metrics.
    pub baseline_metrics: PerformanceMetrics,
    /// Accelerated metrics.
    pub accelerated_metrics: PerformanceMetrics,
}

impl PerformanceComparison {
    /// Computes speedup.
    pub fn speedup(&self) -> f64 {
        self.accelerated_metrics.speedup(&self.baseline_metrics)
    }

    /// Computes iteration reduction.
    pub fn iteration_reduction(&self) -> f64 {
        if self.baseline_metrics.iterations > 0 {
            1.0 - (self.accelerated_metrics.iterations as f64 / self.baseline_metrics.iterations as f64)
        } else {
            0.0
        }
    }

    /// Prints comparison report.
    pub fn print_report(&self) {
        println!("Performance Comparison Report");
        println!("=============================\n");
        println!("Baseline: {}", self.baseline_name);
        println!("  Iterations: {}", self.baseline_metrics.iterations);
        println!("  Time: {:.4} ms", self.baseline_metrics.time_ms);
        println!("  MatVecs: {}", self.baseline_metrics.matvec_count);
        println!();
        println!("Accelerated: {}", self.accelerated_name);
        println!("  Iterations: {}", self.accelerated_metrics.iterations);
        println!("  Time: {:.4} ms", self.accelerated_metrics.time_ms);
        println!("  MatVecs: {}", self.accelerated_metrics.matvec_count);
        println!();
        println!("Results:");
        println!("  Speedup: {:.2}x", self.speedup());
        println!("  Iteration reduction: {:.1}%", self.iteration_reduction() * 100.0);
    }
}

/// Benchmark timer for fine-grained timing.
pub struct BenchmarkTimer {
    start_time: Option<Instant>,
    lap_times: Vec<f64>,
}

impl BenchmarkTimer {
    /// Creates a new benchmark timer.
    pub fn new() -> Self {
        Self {
            start_time: None,
            lap_times: Vec::new(),
        }
    }

    /// Starts the timer.
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Stops the timer and returns elapsed time in ms.
    pub fn stop(&mut self) -> f64 {
        if let Some(start) = self.start_time.take() {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            self.lap_times.push(elapsed);
            elapsed
        } else {
            0.0
        }
    }

    /// Records a lap time.
    pub fn lap(&mut self) -> f64 {
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            self.lap_times.push(elapsed);
            self.start_time = Some(Instant::now());
            elapsed
        } else {
            0.0
        }
    }

    /// Gets all lap times.
    pub fn lap_times(&self) -> &[f64] {
        &self.lap_times
    }

    /// Gets average lap time.
    pub fn average_lap(&self) -> f64 {
        if self.lap_times.is_empty() {
            0.0
        } else {
            self.lap_times.iter().sum::<f64>() / self.lap_times.len() as f64
        }
    }

    /// Gets total time.
    pub fn total_time(&self) -> f64 {
        self.lap_times.iter().sum()
    }
}

impl Default for BenchmarkTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance profiler for tracking solver components.
#[derive(Debug, Clone)]
pub struct PerformanceProfiler {
    /// Component name.
    pub name: String,
    /// Total time spent.
    pub total_time_ms: f64,
    /// Number of calls.
    pub call_count: usize,
    /// Minimum time.
    pub min_time_ms: f64,
    /// Maximum time.
    pub max_time_ms: f64,
}

impl PerformanceProfiler {
    /// Creates a new profiler.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            total_time_ms: 0.0,
            call_count: 0,
            min_time_ms: f64::INFINITY,
            max_time_ms: 0.0,
        }
    }

    /// Records a timing.
    pub fn record(&mut self, time_ms: f64) {
        self.total_time_ms += time_ms;
        self.call_count += 1;
        self.min_time_ms = self.min_time_ms.min(time_ms);
        self.max_time_ms = self.max_time_ms.max(time_ms);
    }

    /// Gets average time per call.
    pub fn average_time(&self) -> f64 {
        if self.call_count > 0 {
            self.total_time_ms / self.call_count as f64
        } else {
            0.0
        }
    }
}

/// Memory tracker for estimating memory usage.
pub struct MemoryTracker {
    allocations: Vec<usize>,
}

impl MemoryTracker {
    /// Creates a new memory tracker.
    pub fn new() -> Self {
        Self {
            allocations: Vec::new(),
        }
    }

    /// Tracks an allocation.
    pub fn track(&mut self, bytes: usize) {
        self.allocations.push(bytes);
    }

    /// Gets total allocated memory in MB.
    pub fn total_mb(&self) -> f64 {
        self.allocations.iter().sum::<usize>() as f64 / (1024.0 * 1024.0)
    }

    /// Gets peak memory usage in MB.
    pub fn peak_mb(&self) -> f64 {
        self.allocations.iter().max().copied().unwrap_or(0) as f64 / (1024.0 * 1024.0)
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimates memory usage of a matrix.
pub fn estimate_matrix_memory(nrows: usize, ncols: usize) -> usize {
    nrows * ncols * 8 // f64 is 8 bytes
}

/// Estimates memory usage of a vector.
pub fn estimate_vector_memory(size: usize) -> usize {
    size * 8
}

/// Measures matrix-vector multiplication performance.
pub fn benchmark_matvec(a: &DMatrix<f64>, x: &DVector<f64>, iterations: usize) -> PerformanceMetrics {
    let mut metrics = PerformanceMetrics::new();
    let mut timer = BenchmarkTimer::new();

    timer.start();
    for _ in 0..iterations {
        let _y = a * x;
    }
    metrics.time_ms = timer.stop();

    metrics.matvec_count = iterations;
    metrics.iterations = iterations;
    metrics.memory_mb = (estimate_matrix_memory(a.nrows(), a.ncols())
        + estimate_vector_memory(x.len()) * 2) as f64 / (1024.0 * 1024.0);

    metrics
}

/// Compares multiple solver configurations.
pub fn compare_solvers<'a, F>(
    _name: &'a str,
    k: &DMatrix<f64>,
    f: &DVector<f64>,
    mut solvers: Vec<(&'a str, F)>,
) -> Vec<(&'a str, PerformanceMetrics)>
where
    F: FnMut(&DMatrix<f64>, &DVector<f64>) -> (usize, f64, bool),
{
    let mut results = Vec::new();

    for (solver_name, mut solver) in solvers {
        let mut timer = BenchmarkTimer::new();
        timer.start();

        let (iterations, residual, converged) = solver(k, f);
        let time_ms = timer.stop();

        let metrics = PerformanceMetrics {
            iterations,
            time_ms,
            matvec_count: iterations * 2, // Approximate
            residual_norm: residual,
            converged,
            memory_mb: 0.0,
        };

        println!("{:<25} | {:>8} | {:>12.4} | {:>10.2e} | {:>8}",
                 solver_name, iterations, time_ms, residual,
                 if converged { "Yes" } else { "No" });

        results.push((solver_name, metrics));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics {
            iterations: 50,
            time_ms: 100.0,
            matvec_count: 100,
            residual_norm: 1e-10,
            converged: true,
            memory_mb: 10.0,
        };

        assert!((metrics.iterations_per_second() - 500.0).abs() < 0.1);
        assert!((metrics.time_per_iteration() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_benchmark_timer() {
        let mut timer = BenchmarkTimer::new();

        timer.start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.stop();

        assert!(elapsed >= 10.0);
        assert_eq!(timer.lap_times().len(), 1);
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();

        tracker.track(1024 * 1024); // 1 MB
        tracker.track(2 * 1024 * 1024); // 2 MB

        assert!((tracker.total_mb() - 3.0).abs() < 0.1);
        assert!((tracker.peak_mb() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_matvec_benchmark() {
        let n = 100;
        let a = DMatrix::from_fn(n, n, |i, j| if i == j { 4.0 } else { 0.0 });
        let x = DVector::from_element(n, 1.0);

        let metrics = benchmark_matvec(&a, &x, 10);

        assert_eq!(metrics.matvec_count, 10);
        assert!(metrics.time_ms > 0.0);
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new("Test");

        profiler.record(10.0);
        profiler.record(20.0);
        profiler.record(15.0);

        assert_eq!(profiler.call_count, 3);
        assert!((profiler.total_time_ms - 45.0).abs() < 0.1);
        assert!((profiler.average_time() - 15.0).abs() < 0.1);
        assert!((profiler.min_time_ms - 10.0).abs() < 0.1);
        assert!((profiler.max_time_ms - 20.0).abs() < 0.1);
    }
}
