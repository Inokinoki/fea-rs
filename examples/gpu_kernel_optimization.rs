//! GPU kernel optimization demonstration.
//!
//! This example demonstrates GPU kernel launch optimization:
//! - Block size optimization
//! - Occupancy analysis
//! - Memory coalescing analysis
//! - Performance prediction

use fea::gpu::gpu_kernel_optimizer::{
    KernelLaunchOptimizer, MemoryCoalescingAnalyzer, KernelPerformancePredictor,
    GPUHardwareInfo,
};

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       GPU Kernel Launch Optimization Demo                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Create optimizer for V100 GPU
    let optimizer = KernelLaunchOptimizer::default_v100();

    // Display hardware info
    display_hardware_info(optimizer.hardware());

    // Block size optimization
    optimize_block_sizes(&optimizer);

    // Memory coalescing analysis
    analyze_memory_patterns();

    // Performance prediction
    predict_performance();

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Optimization Demo Complete                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Display GPU hardware information.
fn display_hardware_info(hw: &GPUHardwareInfo) {
    println!("┌─ GPU Hardware Information ───────────────────────────────┐");
    println!("│ Compute Capability: {}.{}", hw.compute_capability.0, hw.compute_capability.1);
    println!("│ Multiprocessors:    {}", hw.num_multiprocessors);
    println!("│ Max Threads/Block:  {}", hw.max_threads_per_block);
    println!("│ Registers/Block:    {}", hw.max_registers_per_block);
    println!("│ Shared Memory:      {} KB", hw.shared_memory_per_block / 1024);
    println!("│ Warp Size:          {}", hw.warp_size);
    println!("└────────────────────────────────────────────────────────┘\n");
}

/// Demonstrate block size optimization.
fn optimize_block_sizes(optimizer: &KernelLaunchOptimizer) {
    println!("┌─ Block Size Optimization ────────────────────────────────┐");

    // 1D kernel (e.g., vector operations)
    println!("│ 1D Kernel (vector axpy, n=10000):");
    let config = optimizer.optimize_block_size_1d(10000, 16);
    println!("│   Block size:     {}", config.threads_per_block());
    println!("│   Grid size:      {:?}", config.grid_dim);
    println!("│   Total threads:  {}", config.total_threads());

    // 2D kernel (e.g., matrix operations)
    println!("│");
    println!("│ 2D Kernel (matrix 1024x1024):");
    let config = optimizer.optimize_block_size_2d(1024, 1024);
    println!("│   Block size:     {}x{}", config.block_dim.0, config.block_dim.1);
    println!("│   Grid size:      {:?}x{}", config.grid_dim.0, config.grid_dim.1);
    println!("│   Total threads:  {}", config.total_threads());

    // SpMV kernel
    println!("│");
    println!("│ SpMV Kernel (n=50000, nnz=200000):");
    let config = optimizer.optimize_spmv(50000, 200000);
    println!("│   Block size:     {}", config.threads_per_block());
    println!("│   Grid size:      {:?}", config.grid_dim);

    // Reduction kernel
    println!("│");
    println!("│ Reduction Kernel (n=1000000):");
    let config = optimizer.optimize_reduction(1000000);
    println!("│   Block size:     {}", config.threads_per_block());
    println!("│   Shared memory:  {} bytes", config.shared_memory_bytes);

    // Occupancy analysis
    println!("│");
    println!("│ Occupancy Analysis:");
    let config = optimizer.optimize_block_size_1d(10000, 16);
    let occupancy = optimizer.estimate_occupancy(&config, 16);
    println!("│   With 16 regs:   {:.0}%", occupancy * 100.0);

    let occupancy = optimizer.estimate_occupancy(&config, 32);
    println!("│   With 32 regs:   {:.0}%", occupancy * 100.0);

    let occupancy = optimizer.estimate_occupancy(&config, 64);
    println!("│   With 64 regs:   {:.0}%", occupancy * 100.0);

    println!("└────────────────────────────────────────────────────────┘\n");
}

/// Analyze memory access patterns.
fn analyze_memory_patterns() {
    println!("┌─ Memory Coalescing Analysis ─────────────────────────────┐");

    // Sequential (coalesced) access
    println!("│ Sequential Access (0, 1, 2, ..., 99):");
    let indices: Vec<usize> = (0..100).collect();
    let report = MemoryCoalescingAnalyzer::analyze_access_pattern(&indices, 8);
    println!("│   Coalesced:      {}", if report.is_coalesced { "Yes ✓" } else { "No ✗" });
    println!("│   Efficiency:     {:.0}%", report.efficiency * 100.0);
    if !report.recommendations.is_empty() {
        println!("│   Recommendations:");
        for rec in &report.recommendations {
            println!("│     • {}", rec);
        }
    }

    // Strided access
    println!("│");
    println!("│ Strided Access (0, 32, 64, ..., 3168):");
    let indices: Vec<usize> = (0..100).map(|i| i * 32).collect();
    let report = MemoryCoalescingAnalyzer::analyze_access_pattern(&indices, 8);
    println!("│   Coalesced:      {}", if report.is_coalesced { "Yes ✓" } else { "No ✗" });
    println!("│   Efficiency:     {:.0}%", report.efficiency * 100.0);
    if !report.recommendations.is_empty() {
        println!("│   Recommendations:");
        for rec in &report.recommendations {
            println!("│     • {}", rec);
        }
    }

    // Random access
    println!("│");
    println!("│ Random Access (reversed indices):");
    let indices: Vec<usize> = (0..100).rev().collect();
    let report = MemoryCoalescingAnalyzer::analyze_access_pattern(&indices, 8);
    println!("│   Coalesced:      {}", if report.is_coalesced { "Yes ✓" } else { "No ✗" });
    println!("│   Efficiency:     {:.0}%", report.efficiency * 100.0);
    if !report.recommendations.is_empty() {
        println!("│   Recommendations:");
        for rec in &report.recommendations {
            println!("│     • {}", rec);
        }
    }

    println!("└────────────────────────────────────────────────────────┘\n");
}

/// Predict kernel performance.
fn predict_performance() {
    println!("┌─ Performance Prediction (V100) ──────────────────────────┐");

    let predictor = KernelPerformancePredictor::v100();

    // SpMV predictions
    println!("│ SpMV Performance:");
    let sizes = [(10_000, 50_000), (50_000, 200_000), (100_000, 500_000)];

    for (n, nnz) in sizes {
        let estimate = predictor.predict_spmv(nnz);
        println!("│   n={:>6}, nnz={:>7}: {:.2} ms, {:.1} GFLOPS ({})",
            n, nnz, estimate.estimated_time_ms, estimate.achieved_gflops, estimate.bottleneck);
    }

    // GEMM predictions
    println!("│");
    println!("│ GEMM Performance (C = A × B):");
    let sizes = [256, 512, 1024, 2048];

    for &n in &sizes {
        let estimate = predictor.predict_gemm(n, n, n);
        println!("│   {}x{}x{}: {:.2} ms, {:.1} GFLOPS ({})",
            n, n, n, estimate.estimated_time_ms, estimate.achieved_gflops, estimate.bottleneck);
    }

    println!("└────────────────────────────────────────────────────────┘\n");
}
