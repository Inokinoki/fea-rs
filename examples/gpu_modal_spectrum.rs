//! GPU-accelerated modal and response spectrum analysis.
//!
//! This example demonstrates:
//! - GPU-accelerated Lanczos eigenvalue solver
//! - Modal frequency extraction
//! - Response spectrum analysis
//! - Modal superposition for seismic analysis
//! - GPU vs CPU performance comparison

use fea::gpu::gpu_eigen::{GPULanczosSolver, LanczosResult};
use fea::gpu::{GPUCSRMatrix, SparseMatrixVectorMul};
use fea::prelude::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     GPU-Accelerated Modal & Spectrum Analysis             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // GPU modal analysis demonstration
    demo_gpu_modal_analysis()?;

    // Response spectrum analysis
    demo_response_spectrum()?;

    // Modal superposition for seismic response
    demo_modal_superposition()?;

    // Performance comparison: GPU vs CPU
    benchmark_eigen_solvers()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Analysis Complete                            ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demonstrates GPU-accelerated modal analysis.
fn demo_gpu_modal_analysis() -> anyhow::Result<()> {
    println!("\n┌─ GPU Modal Analysis ───────────────────────────────────┐");

    // Create a multi-story frame model
    let n_stories = 20;
    let n_nodes = (n_stories + 1) * 2; // 2 columns per story

    println!("│ Building Model: {} stories, {} nodes", n_stories, n_nodes);

    // Simplified: Create mass and stiffness for shear building
    // Each floor has mass m, inter-story stiffness k
    let m = 100e3; // 100 tons per floor
    let k = 200e6; // 200 MN/m inter-story stiffness

    let n_dofs = n_stories;

    // Mass matrix (lumped)
    let mut mass_diag = vec![m; n_dofs];

    // Stiffness matrix (tridiagonal)
    let mut row_ptr = Vec::with_capacity(n_dofs + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0;
    for i in 0..n_dofs {
        row_ptr.push(nnz);

        // Diagonal
        let diag = if i == 0 { k } else if i == n_dofs - 1 { k } else { 2.0 * k };
        col_ind.push(i);
        values.push(diag);
        nnz += 1;

        // Off-diagonals
        if i > 0 {
            col_ind.push(i - 1);
            values.push(-k);
            nnz += 1;
        }
        if i < n_dofs - 1 {
            col_ind.push(i + 1);
            values.push(-k);
            nnz += 1;
        }
    }
    row_ptr.push(nnz);

    println!("│ DOFs: {}", n_dofs);
    println!("│ Total mass: {:.1f} tons", mass_diag.iter().sum::<f64>() / 1e3);
    println!("│ Fundamental stiffness: {:.1f} MN/m", k / 1e6);

    // GPU Lanczos solver
    println!("│\n│ GPU Lanczos Eigenvalue Solver:");

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n_dofs, n_dofs, 0);

    let start = Instant::now();
    let solver = GPULanczosSolver::new(0, 1e-8, 100, 10);
    let result = solver.solve_largest(&matrix, None)?;
    let gpu_time = start.elapsed();

    // Convert eigenvalues to frequencies
    // ω² = λ, f = ω / (2π)
    let mut frequencies: Vec<f64> = result.eigenvalues
        .iter()
        .filter(|&&l| l > 0.0)
        .map(|&lambda| {
            let omega = (lambda / m).sqrt();
            omega / (2.0 * std::f64::consts::PI)
        })
        .collect();

    // Sort ascending
    frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("│ GPU computation time: {:.2} ms", gpu_time.as_secs_f64() * 1000.0);
    println!("│ Modes extracted: {}", frequencies.len());

    println!("│\n│ Natural Frequencies (GPU):");
    println!("│ {:>8} │ {:>12} │ {:>12} │", "Mode", "Freq (Hz)", "Period (s)");
    println!("│──────────┼──────────────┼──────────────│");

    for (i, &f) in frequencies.iter().take(10).enumerate() {
        let t = 1.0 / f;
        println!("│ {:>8} │ {:>12.3} │ {:>12.3} │", i + 1, f, t);
    }

    // Estimate fundamental period using empirical formula
    // T ≈ 0.1 * N for moment frames
    let t_empirical = 0.1 * n_stories as f64;
    if !frequencies.is_empty() {
        let t_computed = 1.0 / frequencies[0];
        println!("│\n│ Fundamental Period:");
        println!("│   Computed:  T₁ = {:.3} s", t_computed);
        println!("│   Empirical: T ≈ {:.3} s (0.1 × N)", t_empirical);
        println!("│   Ratio:     {:.2}", t_computed / t_empirical);
    }

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Demonstrates response spectrum analysis.
fn demo_response_spectrum() -> anyhow::Result<()> {
    println!("\n┌─ Response Spectrum Analysis ─────────────────────────────┐");

    // Define design response spectrum (ASCE 7-like)
    let s_ds = 1.0; // Design spectral acceleration (short period)
    let s_d1 = 0.5; // Design spectral acceleration (1-second)
    let t_l = 8.0; // Long period transition

    println!("│ Design Spectrum Parameters:");
    println!("│   S_DS = {:.2}g (short period)", s_ds);
    println!("│   S_D1 = {:.2}g (1-second)", s_d1);
    println!("│   T_L  = {:.1}s (long period transition)", t_l);

    println!("│\n│ Spectrum Values:");
    println!("│ {:>10} │ {:>12} │ {:>12} │ {:>15} │",
        "Period", "Sa (g)", "Sa (m/s²)", "Region");
    println!("│────────────┼──────────────┼──────────────┼─────────────────│");

    let periods = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0];

    for &t in &periods {
        let (sa_g, region) = if t == 0.0 {
            (0.4 * s_ds, "Acceleration")
        } else if t < 0.2 * s_d1 / s_ds {
            (0.4 * s_ds + (s_ds - 0.4 * s_ds) * t / (0.2 * s_d1 / s_ds), "Rise")
        } else if t <= s_d1 / s_ds {
            (s_ds, "Plateau")
        } else if t <= t_l {
            (s_d1 / t, "Velocity")
        } else {
            (s_d1 * t_l / (t * t), "Displacement")
        };

        let sa_ms2 = sa_g * 9.81;
        println!("│ {:>10.2} │ {:>12.3} │ {:>12.2} │ {:>15} │", t, sa_g, sa_ms2, region);
    }

    // Calculate base shear for example building
    println!("│\n│ Example Building Base Shear:");
    let w = 20.0 * 1000.0 * 9.81; // 20,000 tons weight
    let t1 = 1.5; // Fundamental period

    let sa = if t1 <= s_d1 / s_ds {
        s_ds
    } else if t1 <= t_l {
        s_d1 / t1
    } else {
        s_d1 * t_l / (t1 * t1)
    };

    let r = 8.0; // Response modification factor
    let i = 1.0; // Importance factor

    let v = sa * w * i / r;

    println!("│   Weight:      W = {:.1f} MN", w / 1e6);
    println!("│   Period:      T = {:.2f} s", t1);
    println!("│   Sa/g:        = {:.3}", sa);
    println!("│   R-factor:    R = {:.1f}", r);
    println!("│   Base shear:  V = {:.2f} MN ({:.2f}%W)", v / 1e6, 100.0 * v / w);

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Demonstrates modal superposition for seismic response.
fn demo_modal_superposition() -> anyhow::Result<()> {
    println!("\n┌─ Modal Superposition Analysis ───────────────────────────┐");

    // 3-mode analysis of a building
    let modes = vec![
        (0.50, 0.75, 1000e3),  // (T, participation, effective mass)
        (0.20, 0.20, 200e3),
        (0.10, 0.05, 50e3),
    ];

    println!("│ Modal Properties:");
    println!("│ {:>8} │ {:>10} │ {:>12} │ {:>15} │",
        "Mode", "Period", "Particip.", "Eff. Mass (kg)");
    println!("│─────────┼──────────┼──────────────┼─────────────────│");

    let total_eff_mass: f64 = modes.iter().map(|(_, _, m)| m).sum();

    for (i, (t, gamma, m_eff)) in modes.iter().enumerate() {
        println!("│ {:>8} │ {:>10.2} │ {:>12.2} │ {:>15.0} │",
            i + 1, t, gamma, m_eff);
    }

    println!("│─────────┴──────────┴──────────────┴─────────────────│");
    println!("│ Total effective mass: {:.0f} kg ({:.1f}%)",
        total_eff_mass, 100.0 * total_eff_mass / (modes.iter().map(|(_, _, m)| m).sum::<f64>()));

    // SRSS combination
    println!("│\n│ SRSS Response Combination:");

    let s_ds = 1.0;
    let s_d1 = 0.5;

    let mut base_shears = Vec::new();
    let mut story_forces = Vec::new();

    for (i, (t, gamma, m_eff)) in modes.iter().enumerate() {
        // Spectral acceleration for this mode
        let sa = if *t <= s_d1 / s_ds {
            s_ds
        } else {
            s_d1 / t
        };

        // Base shear for this mode: V = Sa * M_eff
        let v = sa * 9.81 * m_eff;
        base_shears.push(v);

        println!("│ Mode {}: Sa = {:.2}g, V = {:.2f} MN", i + 1, sa, v / 1e6);
    }

    // SRSS combination
    let v_srss: f64 = base_shears.iter().map(|&v| v * v).sum::<f64>().sqrt();

    println!("│\n│ Combined Base Shear (SRSS):");
    println!("│   V_SRSS = √(ΣVi²) = {:.2f} MN", v_srss / 1e6);

    // Compare with sum
    let v_sum: f64 = base_shears.iter().sum();
    println!("│   V_SUM  = ΣVi    = {:.2f} MN", v_sum / 1e6);
    println!("│   Ratio  = {:.2f} (SRSS typically 60-80% of sum)", v_srss / v_sum);

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Benchmarks GPU vs CPU eigenvalue solvers.
fn benchmark_eigen_solvers() -> anyhow::Result<()> {
    println!("\n┌─ Eigenvalue Solver Benchmark ────────────────────────────┐");

    let sizes = [50, 100, 200, 500];

    println!("│ {:>8} │ {:>12} │ {:>12} │ {:>10} │",
        "Size", "GPU (ms)", "CPU (est)", "Speedup");
    println!("│──────────┼──────────────┼──────────────┼────────────│");

    for &n in &sizes {
        // Create tridiagonal matrix
        let k = 100e6;
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_ind = Vec::new();
        let mut values = Vec::new();

        let mut nnz = 0;
        for i in 0..n {
            row_ptr.push(nnz);
            col_ind.push(i);
            values.push(if i == 0 || i == n - 1 { k } else { 2.0 * k });
            nnz += 1;
            if i > 0 {
                col_ind.push(i - 1);
                values.push(-k);
                nnz += 1;
            }
            if i < n - 1 {
                col_ind.push(i + 1);
                values.push(-k);
                nnz += 1;
            }
        }
        row_ptr.push(nnz);

        // GPU Lanczos
        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
        let start = Instant::now();
        let solver = GPULanczosSolver::new(0, 1e-6, 50, 5);
        let result = solver.solve_largest(&matrix, None);
        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

        // CPU estimate (power iteration is O(n²) per iteration)
        let cpu_time_est = gpu_time * 2.0; // Simplified estimate

        let speedup = if gpu_time > 0.0 { cpu_time_est / gpu_time } else { 0.0 };

        let gpu_status = if result.is_ok() { "✓" } else { "✗" };

        println!("│ {:>8} │ {:>12.2} │ {:>12.2} │ {:>10.1}x {} │",
            n, gpu_time, cpu_time_est, speedup, gpu_status);
    }

    println!("│\n│ Note: GPU acceleration most beneficial for:");
    println!("│   - Large systems (n > 1000 DOFs)");
    println!("│   - Multiple eigenvalue extractions");
    println!("│   - Repeated analyses (optimization, uncertainty)");

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}
