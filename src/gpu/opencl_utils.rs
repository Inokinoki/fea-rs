//! OpenCL optimization utilities and examples.
//!
//! This module provides:
//! - OpenCL device query utilities
//! - Kernel compilation and caching
//! - Performance tuning helpers
//! - Memory optimization utilities

use std::collections::HashMap;
use std::time::Instant;

/// OpenCL device information.
#[derive(Debug, Clone)]
pub struct OpenCLDeviceInfo {
    pub name: String,
    pub vendor: String,
    pub device_type: String,
    pub compute_units: u32,
    pub max_work_group_size: usize,
    pub global_memory_mb: usize,
    pub local_memory_kb: usize,
    pub max_clock_mhz: u32,
    pub version: String,
}

/// OpenCL kernel cache entry.
#[derive(Debug, Clone)]
pub struct KernelCacheEntry {
    pub source_hash: u64,
    pub compiled_at: Instant,
    pub build_time_ms: f64,
}

/// OpenCL performance tuner.
pub struct OpenCLTuner {
    device_info: OpenCLDeviceInfo,
    kernel_cache: HashMap<String, KernelCacheEntry>,
    optimal_work_sizes: HashMap<String, usize>,
}

impl OpenCLTuner {
    /// Creates a new OpenCL tuner.
    pub fn new(device_info: OpenCLDeviceInfo) -> Self {
        Self {
            device_info,
            kernel_cache: HashMap::new(),
            optimal_work_sizes: HashMap::new(),
        }
    }

    /// Finds optimal work group size for a kernel.
    pub fn find_optimal_work_size(&mut self, kernel_name: &str, n: usize) -> usize {
        let max_wg = self.device_info.max_work_group_size;

        // Try common work group sizes
        let candidates = [32, 64, 128, 256, 512];

        let mut best_size = 128; // Default

        for &size in &candidates {
            if size <= max_wg && n >= size {
                best_size = size;
                break;
            }
        }

        self.optimal_work_sizes.insert(kernel_name.to_string(), best_size);
        best_size
    }

    /// Estimates optimal number of compute units to use.
    pub fn estimate_compute_units(&self, problem_size: usize) -> u32 {
        let total_units = self.device_info.compute_units;

        // For small problems, use fewer units to reduce overhead
        if problem_size < 10_000 {
            (total_units / 4).max(1)
        } else if problem_size < 100_000 {
            (total_units / 2).max(1)
        } else {
            total_units
        }
    }

    /// Returns device info.
    pub fn device_info(&self) -> &OpenCLDeviceInfo {
        &self.device_info
    }
}

/// Memory optimization utilities.
pub mod memory_optimization {
    /// Calculates optimal buffer size for coalesced access.
    pub fn optimal_buffer_size(element_size: usize, prefer_alignment: usize) -> usize {
        // Align to 128 bytes for optimal memory transactions
        let alignment = prefer_alignment.max(128);
        ((element_size + alignment - 1) / alignment) * alignment
    }

    /// Calculates padding needed for coalesced access.
    pub fn calculate_padding(width: usize, element_size: usize) -> usize {
        // Ensure each row starts at aligned address
        let row_size = width * element_size;
        let alignment = 128;
        (alignment - (row_size % alignment)) % alignment
    }

    /// Estimates memory bandwidth utilization.
    pub fn estimate_bandwidth_utilization(
        access_pattern: &str,
        element_size: usize,
    ) -> f64 {
        match access_pattern {
            "coalesced" => 0.85,  // 85% of peak
            "strided_2" => 0.50,  // 50% of peak (stride 2)
            "strided_4" => 0.25,  // 25% of peak (stride 4)
            "random" => 0.10,     // 10% of peak (random access)
            _ => 0.50,            // Default estimate
        }
    }
}

/// Kernel optimization utilities.
pub mod kernel_optimization {
    /// Generates optimized SpMV kernel source for given matrix format.
    pub fn generate_spmv_kernel(format: &str, use_shared_mem: bool) -> String {
        let mut kernel = String::new();

        kernel.push_str("__kernel void spmv_optimized(\n");
        kernel.push_str("    const int n,\n");
        kernel.push_str("    __global const double* values,\n");
        kernel.push_str("    __global const int* col_ind,\n");
        kernel.push_str("    __global const int* row_ptr,\n");
        kernel.push_str("    __global const double* x,\n");
        kernel.push_str("    __global double* y,\n");
        kernel.push_str("    const double alpha,\n");
        kernel.push_str("    const double beta\n");
        kernel.push_str(") {\n");

        if use_shared_mem {
            kernel.push_str("    __local double shared_x[256];\n");
            kernel.push_str("    __local int shared_col[256];\n");
        }

        kernel.push_str("    int row = get_global_id(0);\n");
        kernel.push_str("    if (row >= n) return;\n\n");

        if use_shared_mem {
            kernel.push_str("    // Load to shared memory\n");
            kernel.push_str("    int row_start = row_ptr[row];\n");
            kernel.push_str("    int row_end = row_ptr[row + 1];\n");
        } else {
            kernel.push_str("    double sum = 0.0;\n");
            kernel.push_str("    for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {\n");
            kernel.push_str("        sum += values[j] * x[col_ind[j]];\n");
            kernel.push_str("    }\n");
        }

        kernel.push_str("    y[row] = alpha * sum + beta * y[row];\n");
        kernel.push_str("}\n");

        kernel
    }

    /// Calculates optimal vectorization factor.
    pub fn optimal_vectorization(element_size: usize) -> usize {
        // float4 = 16 bytes, float2 = 8 bytes
        if element_size == 8 {
            2 // Use float2
        } else if element_size == 4 {
            4 // Use float4
        } else {
            1 // No vectorization
        }
    }

    /// Generates loop unrolling pragma.
    pub fn generate_unroll_pragma(unroll_factor: usize) -> String {
        if unroll_factor > 1 {
            format!("__attribute__((reqd_work_group_size({}, 1, 1)))\n", unroll_factor)
        } else {
            String::new()
        }
    }
}

/// Benchmark utilities for OpenCL.
pub mod benchmark {
    use std::time::Instant;

    /// Measures kernel execution time.
    pub fn measure_kernel_time<F>(kernel_fn: F, iterations: usize) -> f64
    where
        F: Fn() -> (),
    {
        // Warmup
        kernel_fn();

        // Measure
        let start = Instant::now();
        for _ in 0..iterations {
            kernel_fn();
        }
        let elapsed = start.elapsed();

        elapsed.as_secs_f64() / iterations as f64 * 1000.0 // ms
    }

    /// Calculates achieved GFLOPS.
    pub fn calculate_gflops(flops: f64, time_ms: f64) -> f64 {
        flops / time_ms / 1e6
    }

    /// Calculates achieved bandwidth.
    pub fn calculate_bandwidth(bytes: f64, time_ms: f64) -> f64 {
        bytes / time_ms / 1e6 // GB/s
    }

    /// Prints benchmark results.
    pub fn print_results(name: &str, time_ms: f64, gflops: f64, bandwidth_gbs: f64) {
        println!("│ {:<25} │ {:>10.2} ms │ {:>10.1} GFLOPS │ {:>10.1} GB/s │",
            name, time_ms, gflops, bandwidth_gbs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info() {
        let info = OpenCLDeviceInfo {
            name: "Test GPU".to_string(),
            vendor: "Test Vendor".to_string(),
            device_type: "GPU".to_string(),
            compute_units: 40,
            max_work_group_size: 1024,
            global_memory_mb: 8192,
            local_memory_kb: 64,
            max_clock_mhz: 1500,
            version: "OpenCL 3.0".to_string(),
        };

        assert_eq!(info.compute_units, 40);
        assert!(info.max_work_group_size >= 256);
    }

    #[test]
    fn test_tuner_work_size() {
        let info = OpenCLDeviceInfo {
            name: "Test".to_string(),
            vendor: "Test".to_string(),
            device_type: "GPU".to_string(),
            compute_units: 40,
            max_work_group_size: 256,
            global_memory_mb: 4096,
            local_memory_kb: 32,
            max_clock_mhz: 1000,
            version: "OpenCL 2.0".to_string(),
        };

        let mut tuner = OpenCLTuner::new(info);
        let optimal = tuner.find_optimal_work_size("test_kernel", 10000);

        assert!(optimal <= 256);
        assert!(optimal >= 32);
    }

    #[test]
    fn test_memory_padding() {
        let width = 100;
        let element_size = 8; // double

        let padding = memory_optimization::calculate_padding(width, element_size);
        assert!(padding < 128);
    }

    #[test]
    fn test_bandwidth_estimation() {
        let coalesced = memory_optimization::estimate_bandwidth_utilization("coalesced", 8);
        let random = memory_optimization::estimate_bandwidth_utilization("random", 8);

        assert!(coalesced > random);
        assert!(coalesced > 0.8);
        assert!(random < 0.2);
    }

    #[test]
    fn test_kernel_generation() {
        let kernel = kernel_optimization::generate_spmv_kernel("CSR", false);
        assert!(kernel.contains("spmv_optimized"));
        assert!(kernel.contains("row_ptr"));
        assert!(kernel.contains("col_ind"));

        let kernel_shared = kernel_optimization::generate_spmv_kernel("CSR", true);
        assert!(kernel_shared.contains("__local"));
    }

    #[test]
    fn test_benchmark_utils() {
        let gflops = benchmark::calculate_gflops(1e9, 10.0);
        assert!((gflops - 100.0).abs() < 0.1);

        let bw = benchmark::calculate_bandwidth(1e9, 1000.0); // 1 GB in 1000 ms = 1 GB/s
        assert!((bw - 1.0).abs() < 0.1);
    }
}
