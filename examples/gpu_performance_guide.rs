//! GPU performance comparison and optimization guide.
//!
//! This example provides:
//! - Detailed GPU vs CPU performance comparison
//! - Optimization tips for GPU kernels
//! - Memory bandwidth analysis
//! - Occupancy optimization
//! - Best practices guide

use fea::gpu::{
    GPUCGSolver, GPUGMRESSolver, GPUBiCGSTABSolver, GPUPCGSolver,
    GPUILUPreconditioner, GPUCSRMatrix, SparseMatrixVectorMul,
    gpu_available, list_gpu_devices,
};
use fea::gpu::gpu_kernel_optimizer::{
    KernelLaunchOptimizer, MemoryCoalescingAnalyzer, KernelPerformancePredictor,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       GPU Performance & Optimization Guide                ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Chapter 1: Hardware Analysis
    analyze_hardware();

    // Chapter 2: GPU vs CPU Comparison
    compare_gpu_cpu()?;

    // Chapter 3: Memory Bandwidth Analysis
    analyze_memory_bandwidth()?;

    // Chapter 4: Kernel Optimization
    optimize_kernels();

    // Chapter 5: Best Practices
    print_best_practices();

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Guide Complete                               ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Chapter 1: Hardware Analysis
fn analyze_hardware() {
    println!("┌─ Hardware Analysis ──────────────────────────────────────┐");

    let devices = list_gpu_devices();

    if devices.is_empty() {
        println!("│ No GPU devices detected");
        println!("└────────────────────────────────────────────────────────┘\n");
        return;
    }

    for (i, dev) in devices.iter().enumerate() {
        println!("│ GPU {}: {}", i, dev.name);
        println!("│   Type: {:?}", dev.device_type);
        println!("│   Compute Capability: {}.{}",
            dev.compute_capability.0, dev.compute_capability.1);
        println!("│   Global Memory: {:.1} GB", dev.global_memory_gb);
        println!("│   Multiprocessors: {}", dev.num_multiprocessors);
        println!("│   Max Threads/Block: {}", dev.max_threads_per_block);

        // Estimate theoretical performance
        let compute_capability = format!("{}.{}", dev.compute_capability.0, dev.compute_capability.1);
        let tflops = estimate_tflops(&compute_capability, dev.num_multiprocessors);
        println!("│   Est. Peak FP64: {:.1} TFLOPS", tflops);

        // Memory bandwidth estimate
        let bandwidth = estimate_bandwidth(&compute_capability);
        println!("│   Est. Bandwidth: {:.0} GB/s", bandwidth);

        println!("│");
    }

    println!("└────────────────────────────────────────────────────────┘\n");
}

/// Estimate TFLOPS based on compute capability.
fn estimate_tflops(compute_cap: &str, num_sm: u32) -> f64 {
    // Rough estimates based on typical GPU specs
    let flops_per_sm = match compute_cap {
        "9.0" => 500.0,  // H100
        "8.6" => 250.0,  // RTX 30xx
        "8.0" => 300.0,  // A100
        "7.5" => 150.0,  // RTX 20xx
        "7.0" => 200.0,  // V100
        _ => 100.0,
    };
    num_sm as f64 * flops_per_sm / 1000.0 // Convert to TFLOPS
}

/// Estimate memory bandwidth.
fn estimate_bandwidth(compute_cap: &str) -> f64 {
    match compute_cap {
        "9.0" => 3000.0,  // H100
        "8.6" => 760.0,   // RTX 3090
        "8.0" => 1555.0,  // A100
        "7.5" => 448.0,   // RTX 2080
        "7.0" => 900.0,   // V100
        _ => 300.0,
    }
}

/// Chapter 2: GPU vs CPU Comparison
fn compare_gpu_cpu() -> anyhow::Result<()> {
    println!("┌─ GPU vs CPU Performance Comparison ──────────────────────┐");

    if !gpu_available() {
        println!("│ GPU not available - skipping comparison");
        println!("└────────────────────────────────────────────────────────┘\n");
        return Ok(());
    }

    let sizes = [100, 500, 1000, 2000, 5000];

    println!("│ {:>8} │ {:>12} │ {:>12} │ {:>10} │", "Size", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("│──────────┼──────────────┼──────────────┼────────────│");

    for &n in &sizes {
        // Create test matrix
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

        let b_cpu = vec![1.0f64; n];
        let b_gpu = b_cpu.clone();

        // CPU CG (simple implementation)
        let mut x_cpu = vec![0.0f64; n];
        let mut r_cpu = b_cpu.clone();
        let mut p_cpu = b_cpu.clone();
        let mut rho_cpu = 1.0;

        let start = Instant::now();
        let mut iter_cpu = 0;
        for _ in 0..500 {
            // Simplified CG iteration
            let mut ap_cpu = vec![0.0f64; n];
            for i in 0..n {
                ap_cpu[i] = 4.0 * p_cpu[i];
                if i > 0 { ap_cpu[i] -= p_cpu[i - 1]; }
                if i < n - 1 { ap_cpu[i] -= p_cpu[i + 1]; }
            }

            let p_ap: f64 = p_cpu.iter().zip(ap_cpu.iter()).map(|(a, b)| a * b).sum();
            if p_ap.abs() < 1e-15 { break; }

            let alpha = rho_cpu / p_ap;
            for i in 0..n { x_cpu[i] += alpha * p_cpu[i]; }
            for i in 0..n { r_cpu[i] -= alpha * ap_cpu[i]; }

            let r_norm: f64 = r_cpu.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm < 1e-8 { iter_cpu = _; break; }

            let rho_new: f64 = r_cpu.iter().map(|v| v * v).sum();
            let beta = rho_new / rho_cpu;
            for i in 0..n { p_cpu[i] = r_cpu[i] + beta * p_cpu[i]; }
            rho_cpu = rho_new;
        }
        let cpu_time = start.elapsed().as_secs_f64() * 1000.0;

        // GPU CG
        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
        let cg = GPUCGSolver::new(0, 1e-8, 500);
        let mut x_gpu = vec![0.0f64; n];

        let start = Instant::now();
        let gpu_result = cg.solve(&matrix, &b_gpu, &mut x_gpu)?;
        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

        let speedup = if gpu_time > 0.0 { cpu_time / gpu_time } else { 0.0 };

        println!("│ {:>8} │ {:>12.2} │ {:>12.2} │ {:>10.2}x │",
            n, cpu_time, gpu_time, speedup);
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Chapter 3: Memory Bandwidth Analysis
fn analyze_memory_bandwidth() -> anyhow::Result<()> {
    println!("┌─ Memory Bandwidth Analysis ──────────────────────────────┐");

    if !gpu_available() {
        println!("│ GPU not available - skipping analysis");
        println!("└────────────────────────────────────────────────────────┘\n");
        return Ok(());
    }

    let sizes = [1_000_000, 5_000_000, 10_000_000, 50_000_000];

    println!("│ {:>12} │ {:>14} │ {:>14} │", "Size", "Time (ms)", "Bandwidth");
    println!("│──────────────┼────────────────┼────────────────│");

    for &size in &sizes {
        let spmv = SparseMatrixVectorMul::new(0);

        // Create simple matrix (diagonal for bandwidth test)
        let mut row_ptr = Vec::with_capacity(size + 1);
        let mut col_ind = Vec::with_capacity(size);
        let mut values = Vec::with_capacity(size);

        for i in 0..size {
            row_ptr.push(i);
            col_ind.push(i);
            values.push(1.0);
        }
        row_ptr.push(size);

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, size, size, 0);
        let x = vec![1.0f64; size];
        let mut y = vec![0.0f64; size];

        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = spmv.spmv(&matrix, &x, &mut y, 1.0, 0.0);
        }
        let elapsed = start.elapsed().as_secs_f64() / iterations as f64;

        // Memory traffic: read row_ptr, col_ind, values, x; write y
        let bytes = (size * 8 * 2 + size * 4 + size * 8) as f64;
        let bandwidth = bytes / elapsed / 1e9;

        let size_str = if size >= 1_000_000 {
            format!("{:.1}M", size as f64 / 1_000_000.0)
        } else {
            format!("{}", size)
        };

        println!("│ {:>12} │ {:>14.2} │ {:>14.1} GB/s │",
            size_str, elapsed * 1000.0, bandwidth);
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Chapter 4: Kernel Optimization
fn optimize_kernels() {
    println!("┌─ Kernel Optimization Tips ───────────────────────────────┐");

    let optimizer = KernelLaunchOptimizer::default_v100();

    println!("│ 1. Block Size Selection:");
    println!("│    For vector operations (n=10000):");
    let config = optimizer.optimize_block_size_1d(10000, 16);
    println!("│      Optimal block size: {} threads", config.threads_per_block());
    println!("│      Grid size: {} blocks", config.num_blocks());

    println!("│");
    println!("│ 2. 2D Kernel Optimization:");
    println!("│    For matrix operations (1024x1024):");
    let config = optimizer.optimize_block_size_2d(1024, 1024);
    println!("│      Block: {}x{} threads", config.block_dim.0, config.block_dim.1);
    println!("│      Grid: {}x{} blocks", config.grid_dim.0, config.grid_dim.1);

    println!("│");
    println!("│ 3. Occupancy Analysis:");
    let config = optimizer.optimize_block_size_1d(10000, 16);
    for regs in [16, 32, 64] {
        let occupancy = optimizer.estimate_occupancy(&config, regs);
        println!("│      {} regs/thread: {:.0}% occupancy", regs, occupancy * 100.0);
    }

    println!("│");
    println!("│ 4. Memory Coalescing:");

    // Coalesced access
    let indices: Vec<usize> = (0..100).collect();
    let report = MemoryCoalescingAnalyzer::analyze_access_pattern(&indices, 8);
    println!("│    Sequential access: {:.0}% efficiency", report.efficiency * 100.0);

    // Strided access
    let indices: Vec<usize> = (0..100).map(|i| i * 32).collect();
    let report = MemoryCoalescingAnalyzer::analyze_access_pattern(&indices, 8);
    println!("│    Strided access: {:.0}% efficiency", report.efficiency * 100.0);

    println!("└────────────────────────────────────────────────────────┘\n");
}

/// Chapter 5: Best Practices
fn print_best_practices() {
    println!("┌─ GPU Programming Best Practices ─────────────────────────┐");
    println!("│");
    println!("│ MEMORY OPTIMIZATION:");
    println!("│  ✓ Use coalesced memory access patterns");
    println!("│  ✓ Minimize host-device transfers");
    println!("│  ✓ Use shared memory for repeated accesses");
    println!("│  ✓ Align data to 128-byte boundaries");
    println!("│  ✓ Use pinned memory for async transfers");
    println!("│");
    println!("│ KERNEL OPTIMIZATION:");
    println!("│  ✓ Use block sizes of 128, 256, or 512");
    println!("│  ✓ Minimize register usage per thread");
    println!("│  ✓ Use warp-level primitives when possible");
    println!("│  ✓ Avoid thread divergence in warps");
    println!("│  ✓ Unroll small loops");
    println!("│");
    println!("│ MULTI-GPU OPTIMIZATION:");
    println!("│  ✓ Balance workload across devices");
    println!("│  ✓ Use GPU Direct P2P when available");
    println!("│  ✓ Overlap computation with communication");
    println!("│  ✓ Minimize synchronization points");
    println!("│");
    println!("│ PERFORMANCE MONITORING:");
    println!("│  ✓ Profile with Nsight Compute/Systems");
    println!("│  ✓ Monitor occupancy and throughput");
    println!("│  ✓ Check for memory bottlenecks");
    println!("│  ✓ Analyze instruction mix");
    println!("│");
    println!("│ FEA-SPECIFIC TIPS:");
    println!("│  ✓ Use CSR format for sparse matrices");
    println!("│  ✓ Precondition on GPU (avoid host sync)");
    println!("│  ✓ Batch multiple RHS solves");
    println!("│  ✓ Use mixed precision when applicable");
    println!("│");
    println!("└────────────────────────────────────────────────────────┘");
}
