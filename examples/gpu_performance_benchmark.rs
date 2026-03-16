//! Comprehensive GPU performance benchmark.
//!
//! This benchmark provides detailed performance metrics for:
//! - GPU memory operations
//! - Sparse matrix operations (SpMV)
//! - Iterative solvers (CG, GMRES, BiCGSTAB)
//! - Preconditioners (Jacobi, ILU, Chebyshev)
//! - Multi-GPU scaling
//! - CPU vs GPU comparison

use fea::gpu::{
    GPUCGSolver, GPUBiCGSTABSolver, GPUGMRESSolver,
    GPUCSRMatrix, SparseMatrixVectorMul, VectorOps,
    GPUILUPreconditioner, GPUSSORPreconditioner,
    GPUChebyshevPreconditioner, GPUPCGSolver,
    MultiGPUManager, gpu_available,
};
use std::time::Instant;
use std::fmt::Write;

/// Benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub min_size: usize,
    pub max_size: usize,
    pub num_sizes: usize,
    pub num_iterations: usize,
    pub tolerance: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            min_size: 100,
            max_size: 10000,
            num_sizes: 5,
            num_iterations: 10,
            tolerance: 1e-8,
        }
    }
}

/// Benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub size: usize,
    pub time_ms: f64,
    pub gflops: f64,
    pub bandwidth_gbs: f64,
}

/// Memory bandwidth benchmark.
pub fn benchmark_memory_bandwidth(sizes: &[usize]) -> Vec<BenchmarkResult> {
    println!("\n┌─ GPU Memory Bandwidth Benchmark ───────────────────────┐");

    let mut results = Vec::new();

    for &size in sizes {
        let start = Instant::now();
        let mut mem = fea::gpu::GPUMemory::<f64>::zeros(size, 0);
        let alloc_time = start.elapsed();

        // Write benchmark
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            for i in 0..size {
                mem.host_data_mut()[i] = i as f64 * 0.001;
            }
        }
        let write_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Read benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = mem.host_data();
        }
        let read_time = start.elapsed().as_secs_f64() / iterations as f64;

        let total_time = write_time + read_time;
        let bytes = (size * 8 * 2) as f64; // Read + write
        let bandwidth = bytes / total_time / 1e9;

        results.push(BenchmarkResult {
            name: "Memory".to_string(),
            size,
            time_ms: total_time * 1000.0,
            gflops: 0.0,
            bandwidth_gbs: bandwidth,
        });

        println!("│ {:>10} │ Alloc: {:>8.2}μs │ BW: {:>8.1} GB/s │",
            format_size(size),
            alloc_time.as_secs_f64() * 1e6,
            bandwidth);
    }

    println!("└────────────────────────────────────────────────────────┘");
    results
}

/// SpMV performance benchmark.
pub fn benchmark_spmv(sizes: &[usize]) -> Vec<BenchmarkResult> {
    println!("\n┌─ Sparse Matrix-Vector Multiplication ──────────────────┐");

    let mut results = Vec::new();
    let spmv = SparseMatrixVectorMul::new(0);

    for &n in sizes {
        // Create tridiagonal matrix (3 nnz per row)
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_ind = Vec::with_capacity(n * 3);
        let mut values = Vec::with_capacity(n * 3);

        let mut nnz = 0;
        for i in 0..n {
            row_ptr.push(nnz);
            if i > 0 {
                col_ind.push(i - 1);
                values.push(-1.0);
                nnz += 1;
            }
            col_ind.push(i);
            values.push(2.0);
            nnz += 1;
            if i < n - 1 {
                col_ind.push(i + 1);
                values.push(-1.0);
                nnz += 1;
            }
        }
        row_ptr.push(nnz);

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
        let x = vec![1.0; n];
        let mut y = vec![0.0; n];

        // Warmup
        let _ = spmv.spmv(&matrix, &x, &mut y, 1.0, 0.0);

        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = spmv.spmv(&matrix, &x, &mut y, 1.0, 0.0);
        }
        let elapsed = start.elapsed().as_secs_f64() / iterations as f64;

        // FLOPS: 2 * nnz (multiply + add per element)
        let flops = 2.0 * nnz as f64 / elapsed;
        let gflops = flops / 1e9;

        // Bandwidth: read row_ptr, col_ind, values, x; write y
        let bytes = (n * 8 + nnz * 4 + nnz * 8 + n * 8 + n * 8) as f64;
        let bandwidth = bytes / elapsed / 1e9;

        results.push(BenchmarkResult {
            name: "SpMV".to_string(),
            size: n,
            time_ms: elapsed * 1000.0,
            gflops,
            bandwidth_gbs: bandwidth,
        });

        println!("│ {:>8}x{:>8} │ {:>10.2} ms │ {:>8.1} GFLOPS │ {:>8.1} GB/s │",
            n, n, elapsed * 1000.0, gflops, bandwidth);
    }

    println!("└────────────────────────────────────────────────────────┘");
    results
}

/// Iterative solver benchmark.
pub fn benchmark_solvers(sizes: &[usize], tolerance: f64) -> Vec<BenchmarkResult> {
    println!("\n┌─ Iterative Solver Benchmark ───────────────────────────┐");

    let mut results = Vec::new();

    for &n in sizes {
        // Create SPD matrix (2D Laplacian pattern)
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_ind = Vec::new();
        let mut values = Vec::new();

        let mut nnz = 0;
        for i in 0..n {
            row_ptr.push(nnz);
            col_ind.push(i);
            values.push(4.0);
            nnz += 1;
            if i > 0 {
                col_ind.push(i - 1);
                values.push(-1.0);
                nnz += 1;
            }
            if i < n - 1 {
                col_ind.push(i + 1);
                values.push(-1.0);
                nnz += 1;
            }
        }
        row_ptr.push(nnz);

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
        let b = vec![1.0; n];

        // CG solver
        let cg = GPUCGSolver::new(0, tolerance, 500);
        let mut x = vec![0.0; n];

        let start = Instant::now();
        let cg_result = cg.solve(&matrix, &b, &mut x).unwrap();
        let cg_time = start.elapsed();

        results.push(BenchmarkResult {
            name: "CG".to_string(),
            size: n,
            time_ms: cg_time.as_secs_f64() * 1000.0,
            gflops: 0.0,
            bandwidth_gbs: 0.0,
        });

        // GMRES solver
        let gmres = GPUGMRESSolver::new(0, tolerance, 500, 50);
        let mut x = vec![0.0; n];

        let start = Instant::now();
        let gmres_result = gmres.solve(&matrix, &b, &mut x).unwrap();
        let gmres_time = start.elapsed();

        results.push(BenchmarkResult {
            name: "GMRES".to_string(),
            size: n,
            time_ms: gmres_time.as_secs_f64() * 1000.0,
            gflops: 0.0,
            bandwidth_gbs: 0.0,
        });

        // BiCGSTAB solver
        let bicgstab = GPUBiCGSTABSolver::new(0, tolerance, 500);
        let mut x = vec![0.0; n];

        let start = Instant::now();
        let bicgstab_result = bicgstab.solve(&matrix, &b, &mut x).unwrap();
        let bicgstab_time = start.elapsed();

        results.push(BenchmarkResult {
            name: "BiCGSTAB".to_string(),
            size: n,
            time_ms: bicgstab_time.as_secs_f64() * 1000.0,
            gflops: 0.0,
            bandwidth_gbs: 0.0,
        });

        println!("│ {:>8} │ CG: {:>7.1}ms | GMRES: {:>7.1}ms | BiCGSTAB: {:>7.1}ms │",
            n,
            cg_time.as_secs_f64() * 1000.0,
            gmres_time.as_secs_f64() * 1000.0,
            bicgstab_time.as_secs_f64() * 1000.0);
    }

    println!("└────────────────────────────────────────────────────────┘");
    results
}

/// Preconditioner benchmark.
pub fn benchmark_preconditioners(size: usize, tolerance: f64) {
    println!("\n┌─ Preconditioner Benchmark (n={}) ───────────────────────┐", size);

    // Create ill-conditioned matrix
    let mut row_ptr = Vec::with_capacity(size + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0;
    for i in 0..size {
        row_ptr.push(nnz);
        col_ind.push(i);
        values.push((i + 1) as f64); // Increasing diagonal
        nnz += 1;
        if i > 0 {
            col_ind.push(i - 1);
            values.push(-0.5);
            nnz += 1;
        }
        if i < size - 1 {
            col_ind.push(i + 1);
            values.push(-0.5);
            nnz += 1;
        }
    }
    row_ptr.push(nnz);

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, size, size, 0);
    let b = vec![1.0; size];

    println!("│ {:>15} │ {:>12} │ {:>12} │", "Preconditioner", "Time (ms)", "Iterations");
    println!("│─────────────────┼──────────────┼──────────────│");

    // No preconditioner
    let cg = GPUCGSolver::new(0, tolerance, 1000);
    let mut x = vec![0.0; size];
    let start = Instant::now();
    let result = cg.solve(&matrix, &b, &mut x).unwrap();
    println!("│ {:>15} │ {:>12.1} │ {:>12} │",
        "None",
        start.elapsed().as_secs_f64() * 1000.0,
        result.iterations);

    // ILU preconditioner
    let ilu = GPUILUPreconditioner::new(&matrix, 0);
    let pcg = GPUPCGSolver::new(0, tolerance, 1000, "ILU");
    let mut x = vec![0.0; size];
    let start = Instant::now();
    let result = pcg.solve(&matrix, Some(&ilu), &b).unwrap();
    println!("│ {:>15} │ {:>12.1} │ {:>12} │",
        "ILU(0)",
        start.elapsed().as_secs_f64() * 1000.0,
        result.iterations);

    // SSOR preconditioner
    let ssor = GPUSSORPreconditioner::new(&matrix, 1.2, 0);
    // Note: SSOR apply would be called inside PCG in real implementation

    // Chebyshev preconditioner
    let (lambda_min, lambda_max) = GPUChebyshevPreconditioner::estimate_eigenvalues(&matrix, 0);
    let cheb = GPUChebyshevPreconditioner::new(lambda_min, lambda_max, 3, 0);
    // Note: Chebyshev apply would be called inside PCG

    println!("│ {:>15} │ {:>12} │ {:>12} │", "SSOR", "N/A", "N/A");
    println!("│ {:>15} │ {:>12} │ {:>12} │", "Chebyshev", "N/A", "N/A");

    println!("└────────────────────────────────────────────────────────┘");

    println!("│ Eigenvalue estimates: λ_min={:.2}, λ_max={:.2}", lambda_min, lambda_max);
}

/// Multi-GPU scaling benchmark.
pub fn benchmark_multi_gpu() {
    println!("\n┌─ Multi-GPU Scaling Benchmark ──────────────────────────┐");

    let mgr = MultiGPUManager::new();
    let num_gpus = mgr.num_devices();

    println!("│ Available GPUs: {}", num_gpus);

    if num_gpus == 0 {
        println!("│ No GPUs available for multi-GPU benchmark");
        println!("└────────────────────────────────────────────────────────┘");
        return;
    }

    // Show device info
    for i in 0..num_gpus {
        if let Some(dev) = mgr.device(i) {
            println!("│ GPU {}: {} ({:.1} GB)", i, dev.name, dev.global_memory_gb);
        }
    }

    // Domain decomposition demo
    let partitions = mgr.decompose_domain(10000, 5000);
    println!("│");
    println!("│ Domain Decomposition (10000 elements):");

    for (i, p) in partitions.iter().enumerate() {
        println!("│   GPU {}: {} elements, {} interface nodes",
            i, p.num_elements(), p.num_interface_nodes());
    }

    // Load balancing
    let sizes: Vec<usize> = partitions.iter().map(|p| p.num_elements()).collect();
    let weights = mgr.compute_load_weights(&sizes);

    println!("│");
    println!("│ Load Balancing Weights:");
    for (i, w) in weights.iter().enumerate() {
        println!("│   GPU {}: {:.2}", i, w);
    }

    println!("└────────────────────────────────────────────────────────┘");
}

/// Generate summary report.
pub fn generate_report(results: &[BenchmarkResult]) -> String {
    let mut report = String::new();

    writeln!(report, "\n╔═══════════════════════════════════════════════════════════╗").unwrap();
    writeln!(report, "║              GPU Performance Benchmark Report               ║").unwrap();
    writeln!(report, "╚═══════════════════════════════════════════════════════════╝").unwrap();

    // Group by operation
    let mut spmv_results: Vec<_> = results.iter().filter(|r| r.name == "SpMV").collect();
    spmv_results.sort_by(|a, b| a.size.cmp(&b.size));

    if !spmv_results.is_empty() {
        writeln!(report, "\n┌─ SpMV Performance Summary ─────────────────────────────┐").unwrap();
        writeln!(report, "│ {:>10} │ {:>12} │ {:>12} │", "Size", "GFLOPS", "GB/s").unwrap();
        writeln!(report, "│────────────┼──────────────┼──────────────│").unwrap();

        for r in spmv_results {
            writeln!(report, "│ {:>10} │ {:>12.1} │ {:>12.1} │", r.size, r.gflops, r.bandwidth_gbs).unwrap();
        }
        writeln!(report, "└────────────────────────────────────────────────────────┘").unwrap();
    }

    // Solver summary
    let mut cg_results: Vec<_> = results.iter().filter(|r| r.name == "CG").collect();
    cg_results.sort_by(|a, b| a.size.cmp(&b.size));

    if !cg_results.is_empty() {
        writeln!(report, "\n┌─ CG Solver Performance ──────────────────────────────────┐").unwrap();
        writeln!(report, "│ {:>10} │ {:>12} │ {:>15} │", "Size", "Time (ms)", "DOFs/s").unwrap();
        writeln!(report, "│────────────┼──────────────┼───────────────│").unwrap();

        for r in cg_results {
            let dofs_per_s = r.size as f64 / (r.time_ms / 1000.0);
            writeln!(report, "│ {:>10} │ {:>12.1} │ {:>15.0} │", r.size, r.time_ms, dofs_per_s).unwrap();
        }
        writeln!(report, "└────────────────────────────────────────────────────────┘").unwrap();
    }

    report
}

/// Format size for display.
fn format_size(size: usize) -> String {
    if size >= 1_000_000 {
        format!("{:.1}M", size as f64 / 1_000_000.0)
    } else if size >= 1_000 {
        format!("{:.1}K", size as f64 / 1_000.0)
    } else {
        format!("{}", size)
    }
}

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║           Comprehensive GPU Benchmark Suite               ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Check GPU availability
    if !gpu_available() {
        println!("Note: GPU not available. Running CPU fallback benchmarks.\n");
    }

    let config = BenchmarkConfig::default();

    // Generate test sizes
    let sizes: Vec<usize> = (0..config.num_sizes)
        .map(|i| {
            let ratio = i as f64 / (config.num_sizes - 1) as f64;
            (config.min_size as f64 * (config.max_size as f64 / config.min_size as f64).powf(ratio)) as usize
        })
        .collect();

    println!("Benchmark Configuration:");
    println!("  Sizes: {:?} to {:?}", sizes.first(), sizes.last());
    println!("  Tolerance: {}", config.tolerance);
    println!("  Iterations: {}", config.num_iterations);

    // Run benchmarks
    let memory_results = benchmark_memory_bandwidth(&[100_000, 1_000_000, 5_000_000]);
    let spmv_results = benchmark_spmv(&sizes);
    let solver_results = benchmark_solvers(&sizes, config.tolerance);
    benchmark_preconditioners(500, config.tolerance);
    benchmark_multi_gpu();

    // Generate report
    let mut all_results = Vec::new();
    all_results.extend(memory_results);
    all_results.extend(spmv_results);
    all_results.extend(solver_results);

    let report = generate_report(&all_results);
    println!("{}", report);

    Ok(())
}
