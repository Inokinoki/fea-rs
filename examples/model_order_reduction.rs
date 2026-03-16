//! Model Order Reduction (ROM) demonstration.
//!
//! This example demonstrates:
//! - Proper Orthogonal Decomposition (POD)
//! - Reduced basis method
//! - Offline-online decomposition
//! - Error estimation
//! - Computational savings

use fea::gpu::gpu_rom::{
    ReducedOrderModel, SnapshotCollector, ROMErrorEstimator,
};
use fea::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       Model Order Reduction Demonstration                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Demo 1: POD from synthetic snapshots
    demo_pod_synthetic()?;

    // Demo 2: ROM for parametric problem
    demo_rom_parametric()?;

    // Demo 3: Computational savings
    demo_computational_savings()?;

    // Demo 4: Error estimation
    demo_error_estimation()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Demonstration Complete                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demo 1: POD from synthetic snapshots.
fn demo_pod_synthetic() -> anyhow::Result<()> {
    println!("┌─ Demo 1: POD from Synthetic Snapshots ───────────────────┐");

    // Generate synthetic snapshots
    let n = 200; // Full DOFs
    let m = 30;  // Number of snapshots

    println!("│ Generating {} snapshots ({} DOFs)...", m, n);

    let mut snapshots = DMatrix::zeros(n, m);
    for i in 0..m {
        let freq = 1.0 + i as f64 * 0.2;
        let phase = i as f64 * 0.1;
        for j in 0..n {
            snapshots[(j, i)] = (2.0 * std::f64::consts::PI * freq * j as f64 / n as f64 + phase).sin();
        }
    }

    // Create ROM with different energy thresholds
    for threshold in [0.90, 0.95, 0.99, 0.999] {
        let rom = ReducedOrderModel::from_snapshots(&snapshots, threshold);

        println!("│");
        println!("│ Energy threshold: {:.0}%", threshold * 100.0);
        println!("│   Retained modes: {}", rom.num_modes);
        println!("│   Reduction: {:.1}%", rom.reduction_ratio(n) * 100.0);
        println!("│   Energy captured: {:.1}%", rom.energy_captured * 100.0);
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 2: ROM for parametric problem.
fn demo_rom_parametric() -> anyhow::Result<()> {
    println!("┌─ Demo 2: ROM for Parametric Problem ─────────────────────┐");

    let n = 100;
    let num_params = 20;

    println!("│ Parametric Study:");
    println!("│   Full DOFs: {}", n);
    println!("│   Parameters: {}", num_params);
    println!("│");

    // Collect snapshots for different parameter values
    let mut collector = SnapshotCollector::new(0);

    for i in 0..num_params {
        let param = 1.0 + i as f64 * 0.5;

        // Simulate solution for this parameter
        let mut solution = vec![0.0; n];
        for j in 0..n {
            solution[j] = (param * j as f64 * 0.1).sin() * (1.0 - j as f64 / n as f64);
        }

        collector.collect(&solution);
    }

    println!("│ Collected {} snapshots", collector.num_snapshots());

    // Create ROM
    let snapshots = collector.to_matrix();
    let rom = ReducedOrderModel::from_snapshots(&snapshots, 0.99);

    println!("│");
    println!("│ Reduced Order Model:");
    println!("│   Basis dimensions: {} × {}", rom.basis.nrows(), rom.basis.ncols());
    println!("│   Compression ratio: {:.1}x", n as f64 / rom.num_modes as f64);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 3: Computational savings.
fn demo_computational_savings() -> anyhow::Result<()> {
    println!("┌─ Demo 3: Computational Savings ──────────────────────────┐");

    let sizes = [50, 100, 200, 500];

    println!("│ {:>8} │ {:>12} │ {:>12} │ {:>10} │", "Size", "Full (ms)", "ROM (ms)", "Speedup");
    println!("│──────────┼──────────────┼──────────────┼────────────│");

    for &n in &sizes {
        // Create reduced model
        let num_modes = (n as f64 * 0.1) as usize;
        let basis = DMatrix::identity(n, num_modes.max(3));

        // Full solve
        let k_full = DMatrix::identity(n, n).scale(100.0);
        let f_full = DVector::from_element(n, 1.0);

        let start = Instant::now();
        let _ = k_full.lu().solve(&f_full);
        let full_time = start.elapsed().as_secs_f64() * 1000.0;

        // ROM solve
        let k_reduced = DMatrix::identity(num_modes, num_modes).scale(100.0);
        let f_reduced = DVector::from_element(num_modes, 1.0);

        let start = Instant::now();
        let _ = k_reduced.lu().solve(&f_reduced);
        let rom_solve = start.elapsed().as_secs_f64() * 1000.0;

        // Total ROM time (projection + solve)
        let start = Instant::now();
        let _ = basis.transpose() * &k_full;
        let proj_time = start.elapsed().as_secs_f64() * 1000.0;
        let rom_total = proj_time + rom_solve;

        let speedup = if rom_total > 0.0 { full_time / rom_total } else { 0.0 };

        println!("│ {:>8} │ {:>12.2} │ {:>12.2} │ {:>10.2}x │",
            n, full_time, rom_total, speedup);
    }

    println!("│");
    println!("│ Note: ROM benefits increase with:");
    println!("│   - More queries (amortize projection cost)");
    println!("│   - Larger systems");
    println!("│   - Many parameter evaluations");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 4: Error estimation.
fn demo_error_estimation() -> anyhow::Result<()> {
    println!("┌─ Demo 4: Error Estimation ───────────────────────────────┐");

    let n = 100;

    // Create test problem
    let k_full = DMatrix::identity(n, n).scale(100.0);
    let f = DVector::from_element(n, 1.0);

    // Full solution (reference)
    let u_full = k_full.lu().solve(&f).unwrap();

    // Approximate ROM solution (perturbed)
    let mut u_rom = u_full.clone();
    for i in 0..n {
        u_rom[i] *= 0.95 + (i as f64 / n as f64) * 0.1; // 5% variation
    }

    // Estimate error
    let estimator = ROMErrorEstimator::estimate(&k_full, &f, &u_rom);

    // True error
    let true_error = (&u_full - &u_rom).norm() / u_full.norm() * 100.0;

    println!("│ Problem size: {} DOFs", n);
    println!("│");
    println!("│ Error Analysis:");
    println!("│   Residual norm: {:.2e}", estimator.residual_norm);
    println!("│   Error bound: {:.2}%", estimator.error_bound);
    println!("│   True error: {:.2}%", true_error);
    println!("│   Effectivity: {:.2}", estimator.effectivity);
    println!("│");
    println!("│ Error estimator provides:");
    println!("│   - Rigorous bounds without full solve");
    println!("│   - Adaptive basis enrichment criteria");
    println!("│   - Online certification");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_pod_synthetic() {
        assert!(demo_pod_synthetic().is_ok());
    }

    #[test]
    fn test_demo_rom_parametric() {
        assert!(demo_rom_parametric().is_ok());
    }

    #[test]
    fn test_demo_computational_savings() {
        assert!(demo_computational_savings().is_ok());
    }

    #[test]
    fn test_demo_error_estimation() {
        assert!(demo_error_estimation().is_ok());
    }
}
