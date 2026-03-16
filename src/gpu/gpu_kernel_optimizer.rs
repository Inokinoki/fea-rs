//! GPU kernel launch configuration optimizer.
//!
//! This module provides utilities for optimizing CUDA/OpenCL kernel
//! launch parameters based on hardware characteristics and problem size.

use std::collections::HashMap;

/// GPU hardware characteristics.
#[derive(Debug, Clone)]
pub struct GPUHardwareInfo {
    pub compute_capability: (u32, u32),
    pub num_multiprocessors: u32,
    pub max_threads_per_block: usize,
    pub max_registers_per_block: u32,
    pub shared_memory_per_block: usize,
    pub warp_size: usize,
}

impl Default for GPUHardwareInfo {
    fn default() -> Self {
        Self {
            compute_capability: (7, 0), // V100 default
            num_multiprocessors: 80,
            max_threads_per_block: 1024,
            max_registers_per_block: 65536,
            shared_memory_per_block: 98304,
            warp_size: 32,
        }
    }
}

/// Kernel launch configuration.
#[derive(Debug, Clone)]
pub struct KernelLaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_memory_bytes: usize,
}

impl KernelLaunchConfig {
    /// Returns total number of threads.
    pub fn total_threads(&self) -> usize {
        (self.grid_dim.0 * self.grid_dim.1 * self.grid_dim.2
            * self.block_dim.0 * self.block_dim.1 * self.block_dim.2) as usize
    }

    /// Returns number of blocks.
    pub fn num_blocks(&self) -> usize {
        (self.grid_dim.0 * self.grid_dim.1 * self.grid_dim.2) as usize
    }

    /// Returns threads per block.
    pub fn threads_per_block(&self) -> usize {
        (self.block_dim.0 * self.block_dim.1 * self.block_dim.2) as usize
    }
}

/// Kernel launch optimizer.
pub struct KernelLaunchOptimizer {
    hardware: GPUHardwareInfo,
    occupancy_table: HashMap<String, f64>,
}

impl KernelLaunchOptimizer {
    /// Creates a new optimizer with hardware info.
    pub fn new(hardware: GPUHardwareInfo) -> Self {
        Self {
            hardware,
            occupancy_table: HashMap::new(),
        }
    }

    /// Creates optimizer with default hardware (V100).
    pub fn default_v100() -> Self {
        Self::new(GPUHardwareInfo::default())
    }

    /// Computes optimal block size for 1D kernel.
    pub fn optimize_block_size_1d(&self, problem_size: usize, regs_per_thread: u32) -> KernelLaunchConfig {
        // Start with common block sizes
        let candidates = [32, 64, 128, 256, 512, 1024];

        let mut best_block_size = 256;
        let mut best_occupancy = 0.0;

        for &block_size in &candidates {
            if block_size > self.hardware.max_threads_per_block {
                continue;
            }

            // Calculate occupancy based on register usage
            let total_regs = block_size as u32 * regs_per_thread;
            let max_blocks_per_sm = self.hardware.max_registers_per_block / total_regs.max(1);
            let warps_per_sm = (max_blocks_per_sm * block_size as u32) / self.hardware.warp_size as u32;
            let max_warps = self.hardware.num_multiprocessors * 32; // 32 warps per SM typical

            let occupancy = (warps_per_sm as f64 / 32.0).min(1.0);

            if occupancy > best_occupancy && block_size <= problem_size {
                best_occupancy = occupancy;
                best_block_size = block_size;
            }
        }

        let blocks = ((problem_size + best_block_size - 1) / best_block_size) as u32;

        KernelLaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (best_block_size as u32, 1, 1),
            shared_memory_bytes: 0,
        }
    }

    /// Computes optimal block size for 2D kernel (e.g., matrix operations).
    pub fn optimize_block_size_2d(&self, width: usize, height: usize) -> KernelLaunchConfig {
        // For 2D kernels, square blocks are often optimal
        let block_size = 16; // 16x16 = 256 threads

        let grid_x = ((width + block_size - 1) / block_size) as u32;
        let grid_y = ((height + block_size - 1) / block_size) as u32;

        KernelLaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_memory_bytes: 0,
        }
    }

    /// Computes optimal configuration for SpMV kernel.
    pub fn optimize_spmv(&self, n_rows: usize, nnz: usize) -> KernelLaunchConfig {
        // For SpMV, one thread per row is common
        self.optimize_block_size_1d(n_rows, 16) // Typical SpMV uses ~16 regs
    }

    /// Computes optimal configuration for reduction kernel.
    pub fn optimize_reduction(&self, input_size: usize) -> KernelLaunchConfig {
        // Reduction benefits from max occupancy
        let block_size = self.hardware.max_threads_per_block.min(1024);
        let blocks = ((input_size + block_size - 1) / block_size) as u32;

        KernelLaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_memory_bytes: block_size * 8, // For shared memory reduction
        }
    }

    /// Estimates achieved occupancy for a given configuration.
    pub fn estimate_occupancy(&self, config: &KernelLaunchConfig, regs_per_thread: u32) -> f64 {
        let threads_per_block = config.threads_per_block();
        let total_regs = threads_per_block as u32 * regs_per_thread;

        if total_regs == 0 {
            return 0.0;
        }

        let max_blocks_per_sm = (self.hardware.max_registers_per_block / total_regs)
            .min(16); // Max 16 blocks per SM typical
        let active_warps = (max_blocks_per_sm * threads_per_block as u32) / self.hardware.warp_size as u32;
        let max_warps_per_sm = 32; // Typical

        (active_warps as f64 / max_warps_per_sm as f64).min(1.0)
    }

    /// Records actual occupancy for a kernel.
    pub fn record_occupancy(&mut self, kernel_name: &str, occupancy: f64) {
        self.occupancy_table.insert(kernel_name.to_string(), occupancy);
    }

    /// Gets recorded occupancy for a kernel.
    pub fn get_occupancy(&self, kernel_name: &str) -> Option<f64> {
        self.occupancy_table.get(kernel_name).copied()
    }

    /// Returns hardware info.
    pub fn hardware(&self) -> &GPUHardwareInfo {
        &self.hardware
    }
}

/// Memory coalescing analyzer.
pub struct MemoryCoalescingAnalyzer;

impl MemoryCoalescingAnalyzer {
    /// Analyzes memory access pattern for coalescing efficiency.
    pub fn analyze_access_pattern(indices: &[usize], element_size: usize) -> CoalescingReport {
        if indices.is_empty() {
            return CoalescingReport {
                is_coalesced: true,
                efficiency: 1.0,
                recommendations: vec!["Empty access pattern".to_string()],
            };
        }

        let mut consecutive_count = 1;
        let mut max_consecutive = 1;
        let mut misaligned_count = 0;

        for i in 1..indices.len() {
            if indices[i] == indices[i - 1] + 1 {
                consecutive_count += 1;
                max_consecutive = max_consecutive.max(consecutive_count);
            } else {
                consecutive_count = 1;
            }

            // Check 128-byte alignment
            if (indices[i] * element_size) % 128 != 0 {
                misaligned_count += 1;
            }
        }

        let efficiency = max_consecutive as f64 / 32.0; // 32 threads per warp
        let is_coalesced = efficiency > 0.8;

        let mut recommendations = Vec::new();

        if !is_coalesced {
            recommendations.push("Consider reordering data for sequential access".to_string());
        }

        if misaligned_count > indices.len() / 2 {
            recommendations.push("Add padding to achieve 128-byte alignment".to_string());
        }

        if efficiency < 0.5 {
            recommendations.push("Use shared memory to reduce global memory accesses".to_string());
        }

        CoalescingReport {
            is_coalesced,
            efficiency: efficiency.min(1.0),
            recommendations,
        }
    }
}

/// Memory coalescing analysis report.
#[derive(Debug, Clone)]
pub struct CoalescingReport {
    pub is_coalesced: bool,
    pub efficiency: f64,
    pub recommendations: Vec<String>,
}

/// Kernel performance predictor.
pub struct KernelPerformancePredictor {
    hardware: GPUHardwareInfo,
    peak_bandwidth_gbs: f64,
    peak_tflops: f64,
}

impl KernelPerformancePredictor {
    /// Creates predictor for V100 GPU.
    pub fn v100() -> Self {
        Self {
            hardware: GPUHardwareInfo::default(),
            peak_bandwidth_gbs: 900.0,
            peak_tflops: 7.0,
        }
    }

    /// Predicts SpMV performance.
    pub fn predict_spmv(&self, nnz: usize) -> PerformanceEstimate {
        let bytes = (nnz * (8 + 4 + 8)) as f64; // value + col_ind + x
        let flops = (nnz * 2) as f64; // multiply + add

        let bandwidth_time = bytes / self.peak_bandwidth_gbs / 1e9;
        let compute_time = flops / self.peak_tflops / 1e12;

        let estimated_time = bandwidth_time.max(compute_time) * 1.2; // 20% overhead

        PerformanceEstimate {
            estimated_time_ms: estimated_time * 1000.0,
            achieved_gflops: flops / estimated_time / 1e9,
            achieved_gbs: bytes / estimated_time / 1e9,
            bottleneck: if bandwidth_time > compute_time { "Memory" } else { "Compute" },
        }
    }

    /// Predicts GEMM performance.
    pub fn predict_gemm(&self, m: usize, n: usize, k: usize) -> PerformanceEstimate {
        let flops = (2 * m * n * k) as f64;
        let bytes = ((m * k + k * n + m * n) * 8) as f64;

        let bandwidth_time = bytes / self.peak_bandwidth_gbs / 1e9;
        let compute_time = flops / self.peak_tflops / 1e12;

        let estimated_time = bandwidth_time.max(compute_time) * 1.1; // 10% overhead for GEMM

        PerformanceEstimate {
            estimated_time_ms: estimated_time * 1000.0,
            achieved_gflops: flops / estimated_time / 1e9,
            achieved_gbs: bytes / estimated_time / 1e9,
            bottleneck: if bandwidth_time > compute_time { "Memory" } else { "Compute" },
        }
    }
}

/// Performance estimation result.
#[derive(Debug, Clone)]
pub struct PerformanceEstimate {
    pub estimated_time_ms: f64,
    pub achieved_gflops: f64,
    pub achieved_gbs: f64,
    pub bottleneck: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_launch_optimizer() {
        let optimizer = KernelLaunchOptimizer::default_v100();

        let config = optimizer.optimize_block_size_1d(10000, 16);
        assert!(config.threads_per_block() >= 32);
        assert!(config.threads_per_block() <= 1024);
        assert!(config.total_threads() >= 10000);
    }

    #[test]
    fn test_2d_block_optimization() {
        let optimizer = KernelLaunchOptimizer::default_v100();

        let config = optimizer.optimize_block_size_2d(1024, 1024);
        assert_eq!(config.block_dim.0, 16);
        assert_eq!(config.block_dim.1, 16);
        assert_eq!(config.threads_per_block(), 256);
    }

    #[test]
    fn test_occupancy_estimation() {
        let optimizer = KernelLaunchOptimizer::default_v100();

        let config = optimizer.optimize_block_size_1d(10000, 16);
        let occupancy = optimizer.estimate_occupancy(&config, 16);

        assert!(occupancy > 0.0);
        assert!(occupancy <= 1.0);
    }

    #[test]
    fn test_memory_coalescing_analysis() {
        // Coalesced access
        let indices: Vec<usize> = (0..100).collect();
        let report = MemoryCoalescingAnalyzer::analyze_access_pattern(&indices, 8);

        assert!(report.is_coalesced);
        assert!(report.efficiency > 0.9);

        // Strided access
        let indices: Vec<usize> = (0..100).map(|i| i * 32).collect();
        let report = MemoryCoalescingAnalyzer::analyze_access_pattern(&indices, 8);

        assert!(!report.is_coalesced || report.efficiency < 0.5);
    }

    #[test]
    fn test_performance_predictor() {
        let predictor = KernelPerformancePredictor::v100();

        let estimate = predictor.predict_spmv(1_000_000);
        assert!(estimate.estimated_time_ms > 0.0);
        assert!(estimate.achieved_gflops > 0.0);
        assert!(!estimate.bottleneck.is_empty());

        let estimate = predictor.predict_gemm(1000, 1000, 1000);
        assert!(estimate.estimated_time_ms > 0.0);
        assert!(estimate.achieved_gflops > 0.0);
    }

    #[test]
    fn test_kernel_config() {
        let config = KernelLaunchConfig {
            grid_dim: (10, 10, 1),
            block_dim: (16, 16, 1),
            shared_memory_bytes: 4096,
        };

        assert_eq!(config.num_blocks(), 100);
        assert_eq!(config.threads_per_block(), 256);
        assert_eq!(config.total_threads(), 25600);
    }
}
