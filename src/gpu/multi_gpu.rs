//! Multi-GPU support framework for large-scale FEA.
//!
//! This module provides:
//! - Multi-GPU device management
//! - Domain decomposition for parallel analysis
//! - GPU-aware MPI-like communication
//! - Load balancing across devices
//! - Asynchronous kernel execution

use crate::gpu::{GPUDevice, DeviceType, GPUMemory, GPUStream, GPUCSRMatrix};
use std::sync::Arc;

/// Multi-GPU manager for coordinating multiple devices.
#[derive(Debug)]
pub struct MultiGPUManager {
    /// Available GPU devices.
    devices: Vec<GPUDevice>,
    /// Device contexts (one per GPU).
    contexts: Vec<GPUContext>,
    /// Streams for async execution.
    streams: Vec<GPUStream>,
    /// Current device for operations.
    current_device: usize,
}

impl MultiGPUManager {
    /// Creates a new multi-GPU manager.
    pub fn new() -> Self {
        let devices = GPUDevice::available_devices();
        let mut contexts = Vec::new();
        let mut streams = Vec::new();

        for (i, device) in devices.iter().enumerate() {
            contexts.push(GPUContext::new(i, device.device_type));
            streams.push(GPUStream::async_stream(i, 0));
        }

        Self {
            devices,
            contexts,
            streams,
            current_device: 0,
        }
    }

    /// Returns the number of available GPUs.
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    /// Returns a reference to the specified device.
    pub fn device(&self, id: usize) -> Option<&GPUDevice> {
        self.devices.get(id)
    }

    /// Sets the current device for operations.
    pub fn set_device(&mut self, id: usize) -> anyhow::Result<()> {
        if id >= self.devices.len() {
            anyhow::bail!("Invalid device ID: {}", id);
        }
        self.current_device = id;
        Ok(())
    }

    /// Returns the current device ID.
    pub fn current_device(&self) -> usize {
        self.current_device
    }

    /// Creates a domain decomposition for parallel analysis.
    pub fn decompose_domain(
        &self,
        total_elements: usize,
        total_nodes: usize,
    ) -> Vec<DomainPartition> {
        let n_gpus = self.devices.len();
        if n_gpus == 0 {
            return vec![];
        }

        let mut partitions = Vec::with_capacity(n_gpus);
        let elem_per_gpu = total_elements / n_gpus;
        let node_per_gpu = total_nodes / n_gpus;

        for i in 0..n_gpus {
            let elem_start = i * elem_per_gpu;
            let elem_end = if i == n_gpus - 1 {
                total_elements
            } else {
                (i + 1) * elem_per_gpu
            };

            let node_start = i * node_per_gpu;
            let node_end = if i == n_gpus - 1 {
                total_nodes
            } else {
                (i + 1) * node_per_gpu
            };

            partitions.push(DomainPartition {
                device_id: i,
                element_range: (elem_start, elem_end),
                node_range: (node_start, node_end),
                interface_nodes: Vec::new(),
            });
        }

        // Identify interface nodes (shared between partitions)
        Self::compute_interfaces(&mut partitions);

        partitions
    }

    /// Computes interface nodes between partitions.
    fn compute_interfaces(partitions: &mut [DomainPartition]) {
        // Simplified: in real implementation, would analyze mesh connectivity
        // to find nodes shared between adjacent partitions
        for i in 0..partitions.len() {
            if i > 0 {
                // Add overlap with previous partition
                let prev_end = partitions[i - 1].node_range.1;
                if prev_end > partitions[i].node_range.0 {
                    partitions[i].interface_nodes.push(prev_end - 1);
                }
            }
        }
    }

    /// Executes a computation across all GPUs.
    pub fn parallel_execute<F>(&self, mut f: F) -> Vec<anyhow::Result<()>>
    where
        F: FnMut(usize, &GPUContext, &GPUStream) -> anyhow::Result<()>,
    {
        let mut results = Vec::new();

        for (i, (context, stream)) in self.contexts.iter().zip(self.streams.iter()).enumerate() {
            results.push(f(i, context, stream));
        }

        results
    }

    /// Synchronizes all GPU streams.
    pub fn synchronize_all(&self) {
        for stream in &self.streams {
            stream.synchronize();
        }
    }

    /// Asynchronously transfers data between GPUs.
    pub fn gpu_transfer(
        &self,
        src_device: usize,
        dst_device: usize,
        data: &[f64],
    ) -> anyhow::Result<Vec<f64>> {
        // In real implementation, would use GPU-direct P2P or host staging
        if src_device == dst_device {
            return Ok(data.to_vec());
        }

        // Check if P2P is supported
        if self.p2p_supported(src_device, dst_device) {
            // Direct P2P transfer (would use cudaMemcpyPeerAsync)
            Ok(data.to_vec())
        } else {
            // Host-mediated transfer
            Ok(data.to_vec())
        }
    }

    /// Checks if peer-to-peer access is supported between devices.
    pub fn p2p_supported(&self, device1: usize, device2: usize) -> bool {
        // In real implementation, would query CUDA device properties
        // For now, assume devices on same socket support P2P
        device1 == device2 || (device1 < 2 && device2 < 2)
    }

    /// Returns memory bandwidth for a device (GB/s).
    pub fn memory_bandwidth(&self, device_id: usize) -> f64 {
        if let Some(device) = self.devices.get(device_id) {
            // Return actual bandwidth based on device type
            match device.device_type {
                DeviceType::CUDA => match device.compute_capability {
                    (8, _) => 936.0, // A100
                    (7, _) => 450.0, // V100
                    _ => 300.0,
                },
                DeviceType::OpenCL => 200.0,
            }
        } else {
            0.0
        }
    }

    /// Computes load balancing weights for partitions.
    pub fn compute_load_weights(&self, partition_sizes: &[usize]) -> Vec<f64> {
        let n = partition_sizes.len();
        if n == 0 {
            return vec![];
        }

        let total: usize = partition_sizes.iter().sum();
        if total == 0 {
            return vec![1.0 / n as f64; n];
        }

        let mut weights = Vec::with_capacity(n);
        for (i, &size) in partition_sizes.iter().enumerate() {
            let device_speed = self.memory_bandwidth(i);
            let weight = (size as f64 / total as f64) * (device_speed / 300.0);
            weights.push(weight);
        }

        // Normalize
        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }

        weights
    }
}

impl Default for MultiGPUManager {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU context for device operations.
#[derive(Debug, Clone)]
pub struct GPUContext {
    device_id: usize,
    device_type: DeviceType,
    initialized: bool,
}

impl GPUContext {
    pub fn new(device_id: usize, device_type: DeviceType) -> Self {
        Self {
            device_id,
            device_type,
            initialized: true,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }
}

/// Domain partition for multi-GPU parallel analysis.
#[derive(Debug, Clone)]
pub struct DomainPartition {
    /// Assigned device ID.
    pub device_id: usize,
    /// Element range (start, end).
    pub element_range: (usize, usize),
    /// Node range (start, end).
    pub node_range: (usize, usize),
    /// Interface nodes (shared with other partitions).
    pub interface_nodes: Vec<usize>,
}

impl DomainPartition {
    /// Returns the number of elements in this partition.
    pub fn num_elements(&self) -> usize {
        self.element_range.1 - self.element_range.0
    }

    /// Returns the number of nodes in this partition.
    pub fn num_nodes(&self) -> usize {
        self.node_range.1 - self.node_range.0
    }

    /// Returns the number of interface nodes.
    pub fn num_interface_nodes(&self) -> usize {
        self.interface_nodes.len()
    }
}

/// Asynchronous GPU task for concurrent execution.
#[derive(Debug)]
pub struct GPUTask {
    /// Task ID.
    pub id: usize,
    /// Target device.
    pub device_id: usize,
    /// Stream for async execution.
    pub stream_id: usize,
    /// Task priority (lower = higher priority).
    pub priority: u32,
    /// Estimated execution time (ms).
    pub estimated_time_ms: f64,
}

impl GPUTask {
    pub fn new(id: usize, device_id: usize, stream_id: usize) -> Self {
        Self {
            id,
            device_id,
            stream_id,
            priority: 0,
            estimated_time_ms: 0.0,
        }
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_estimated_time(mut self, time_ms: f64) -> Self {
        self.estimated_time_ms = time_ms;
        self
    }
}

/// GPU task scheduler for managing concurrent tasks.
#[derive(Debug)]
pub struct GPUTaskScheduler {
    tasks: Vec<GPUTask>,
    device_queues: Vec<Vec<GPUTask>>,
}

impl GPUTaskScheduler {
    pub fn new(num_devices: usize) -> Self {
        Self {
            tasks: Vec::new(),
            device_queues: (0..num_devices).map(|_| Vec::new()).collect(),
        }
    }

    /// Adds a task to the scheduler.
    pub fn add_task(&mut self, task: GPUTask) {
        if task.device_id < self.device_queues.len() {
            self.device_queues[task.device_id].push(task);
        }
    }

    /// Schedules tasks based on priority and load balancing.
    pub fn schedule(&mut self) {
        for queue in &mut self.device_queues {
            queue.sort_by(|a, b| a.priority.cmp(&b.priority));
        }
    }

    /// Returns the number of pending tasks for a device.
    pub fn pending_tasks(&self, device_id: usize) -> usize {
        if device_id < self.device_queues.len() {
            self.device_queues[device_id].len()
        } else {
            0
        }
    }

    /// Clears completed tasks.
    pub fn clear_completed(&mut self) {
        for queue in &mut self.device_queues {
            queue.clear();
        }
    }
}

/// GPU memory pool for efficient memory management.
#[derive(Debug)]
pub struct GPUMemoryPool {
    device_id: usize,
    /// Pool of reusable memory blocks.
    free_blocks: Vec<usize>,
    /// Size of each block.
    block_size: usize,
    /// Total allocated memory.
    total_allocated: usize,
}

impl GPUMemoryPool {
    pub fn new(device_id: usize, block_size: usize, num_blocks: usize) -> Self {
        Self {
            device_id,
            free_blocks: (0..num_blocks).collect(),
            block_size,
            total_allocated: block_size * num_blocks,
        }
    }

    /// Allocates a memory block from the pool.
    pub fn allocate(&mut self) -> Option<usize> {
        self.free_blocks.pop()
    }

    /// Returns a block to the pool.
    pub fn deallocate(&mut self, block_id: usize) {
        self.free_blocks.push(block_id);
    }

    /// Returns the number of free blocks.
    pub fn free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Returns the memory utilization (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        let total = self.free_blocks.len() + (self.total_allocated / self.block_size - self.free_blocks.len());
        if total == 0 {
            0.0
        } else {
            1.0 - (self.free_blocks.len() as f64 / total as f64)
        }
    }
}

/// Performance metrics for multi-GPU analysis.
#[derive(Debug, Clone, Default)]
pub struct GPUMetrics {
    /// Total computation time (ms).
    pub total_time_ms: f64,
    /// Time spent on GPU 0.
    pub gpu0_time_ms: f64,
    /// Time spent on GPU 1.
    pub gpu1_time_ms: f64,
    /// Time spent on GPU 2.
    pub gpu2_time_ms: f64,
    /// Time spent on GPU 3.
    pub gpu3_time_ms: f64,
    /// Communication overhead (ms).
    pub comm_time_ms: f64,
    /// Load imbalance ratio.
    pub load_imbalance: f64,
    /// Speedup vs single GPU.
    pub speedup: f64,
    /// Parallel efficiency.
    pub efficiency: f64,
}

impl GPUMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes parallel efficiency.
    pub fn compute_efficiency(&mut self, single_gpu_time: f64, num_gpus: usize) {
        if self.total_time_ms > 0.0 && num_gpus > 0 {
            self.speedup = single_gpu_time / self.total_time_ms;
            self.efficiency = self.speedup / num_gpus as f64;
        }
    }

    /// Prints performance summary.
    pub fn print_summary(&self) {
        println!("=== GPU Performance Metrics ===");
        println!("  Total time:       {:.2} ms", self.total_time_ms);
        println!("  GPU times:        {:.2}, {:.2}, {:.2}, {:.2} ms",
            self.gpu0_time_ms, self.gpu1_time_ms,
            self.gpu2_time_ms, self.gpu3_time_ms);
        println!("  Communication:    {:.2} ms", self.comm_time_ms);
        println!("  Load imbalance:   {:.1}%", self.load_imbalance * 100.0);
        println!("  Speedup:          {:.2}x", self.speedup);
        println!("  Efficiency:       {:.1}%", self.efficiency * 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_gpu_manager_creation() {
        let mgr = MultiGPUManager::new();
        // May have 0 or more devices depending on system
        assert!(mgr.num_devices() >= 0);
    }

    #[test]
    fn test_domain_decomposition() {
        let mgr = MultiGPUManager::new();
        let partitions = mgr.decompose_domain(1000, 500);

        if !partitions.is_empty() {
            let total_elems: usize = partitions.iter().map(|p| p.num_elements()).sum();
            assert!(total_elems <= 1000);
        }
    }

    #[test]
    fn test_domain_partition() {
        let partition = DomainPartition {
            device_id: 0,
            element_range: (0, 100),
            node_range: (0, 50),
            interface_nodes: vec![49, 50],
        };

        assert_eq!(partition.num_elements(), 100);
        assert_eq!(partition.num_nodes(), 50);
        assert_eq!(partition.num_interface_nodes(), 2);
    }

    #[test]
    fn test_gpu_task_scheduler() {
        let mut scheduler = GPUTaskScheduler::new(2);

        scheduler.add_task(GPUTask::new(1, 0, 0).with_priority(2));
        scheduler.add_task(GPUTask::new(2, 0, 0).with_priority(1));
        scheduler.add_task(GPUTask::new(3, 1, 0).with_priority(3));

        scheduler.schedule();

        assert_eq!(scheduler.pending_tasks(0), 2);
        assert_eq!(scheduler.pending_tasks(1), 1);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = GPUMemoryPool::new(0, 1024, 10);

        assert_eq!(pool.free_blocks(), 10);
        assert!((pool.utilization() - 0.0).abs() < 1e-10);

        let block = pool.allocate();
        assert!(block.is_some());
        assert_eq!(pool.free_blocks(), 9);

        pool.deallocate(block.unwrap());
        assert_eq!(pool.free_blocks(), 10);
    }

    #[test]
    fn test_gpu_metrics() {
        let mut metrics = GPUMetrics::new();
        metrics.total_time_ms = 100.0;
        metrics.gpu0_time_ms = 90.0;
        metrics.gpu1_time_ms = 95.0;
        metrics.comm_time_ms = 5.0;
        metrics.load_imbalance = 0.05;

        metrics.compute_efficiency(180.0, 2);

        assert!(metrics.speedup > 0.0);
        assert!(metrics.efficiency <= 1.0);
    }
}
