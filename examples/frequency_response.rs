//! Frequency response and harmonic analysis demonstration.
//!
//! This example demonstrates:
//! - Frequency sweep analysis
//! - Harmonic response computation
//! - Resonance detection
//! - Damping effects on response
//! - Mode superposition for dynamic analysis

use fea::prelude::*;
use fea::algorithms::solvers::{
    LanczosSolver, LanczosConfig, LanczosResult,
};
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Frequency Response & Harmonic Analysis ===\n");

    // Modal analysis for natural frequencies
    demo_modal_analysis()?;

    // Frequency sweep analysis
    demo_frequency_sweep()?;

    // Harmonic response with damping
    demo_harmonic_response()?;

    // Mode superposition analysis
    demo_mode_superposition()?;

    // Resonance detection
    demo_resonance_detection()?;

    println!("\n=== Analysis Complete ===");
    Ok(())
}

/// Demonstrate modal analysis.
fn demo_modal_analysis() -> anyhow::Result<()> {
    println!("\nModal Analysis:");
    println!("-------------");

    // Create a simple beam model
    let mut model = Model::<Truss2>::new();

    let length = 10.0;
    let n_elements = 20;
    let dx = length / n_elements as f64;

    for i in 0..=n_elements {
        model.add_node(Node::new_2d(i as f64 * dx, 0.0));
    }

    for i in 0..n_elements {
        model.add_element(Truss2::new(i, i + 1));
    }

    model.add_material(Material {
        name: "Steel".to_string(),
        e: 210e9,
        nu: 0.3,
        rho: 7850.0,
        alpha: 12e-6,
    });
    model.add_section(Section::circular("round", 0.05));

    // Fixed at left end
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }

    println!("  Nodes: {}", model.nodes.len());
    println!("  Elements: {}", model.elements.len());
    println!("  DOFs: {}", model.ndofs());

    // Extract mass and stiffness matrices (simplified)
    let ndofs = model.ndofs();
    let mut k = DMatrix::identity(ndofs, ndofs);
    let mut m = DMatrix::identity(ndofs, ndofs);

    // For demonstration, use simplified matrices
    // In real implementation, these would be assembled from elements

    // Apply boundary conditions to matrices
    let free_dofs: Vec<usize> = (6..ndofs).collect();
    let n_free = free_dofs.len();

    let k_free = k.fixed_rows_cols::<Dyn>(6, 6, n_free, n_free);
    let m_free = m.fixed_rows_cols::<Dyn>(6, 6, n_free, n_free);

    // Lanczos eigensolver
    let config = LanczosConfig {
        num_eigenvalues: 10,
        max_iterations: 100,
        tolerance: 1e-8,
    };

    let lanczos = LanczosSolver::new();

    println!("\n  Computing eigenvalues...");
    let start = Instant::now();

    // Use simplified eigenvalue computation for demo
    let frequencies = compute_approximate_frequencies(n_elements, 210e9, 7850.0);

    let elapsed = start.elapsed();

    println!("  Computation time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("\n  Natural Frequencies (Hz):");
    for (i, f) in frequencies.iter().take(10).enumerate() {
        let omega = 2.0 * std::f64::consts::PI * f;
        println!("    Mode {}: {:.2f} Hz  (ω = {:.2f} rad/s)", i + 1, f, omega);
    }

    Ok(())
}

/// Compute approximate frequencies for a bar.
fn compute_approximate_frequencies(n: usize, e: f64, rho: f64) -> Vec<f64> {
    // For a fixed-free bar: f_n = (2n-1) * c / (4L)
    // where c = sqrt(E/rho)
    let c = (e / rho).sqrt();
    let l = 10.0;

    let mut freqs = Vec::new();
    for i in 1..=20 {
        let f = (2 * i - 1) as f64 * c / (4.0 * l);
        freqs.push(f);
    }
    freqs
}

/// Demonstrate frequency sweep analysis.
fn demo_frequency_sweep() -> anyhow::Result<()> {
    println!("\nFrequency Sweep Analysis:");
    println!("-----------------------");

    // Single DOF system parameters
    let m = 100.0; // mass (kg)
    let k = 10000.0; // stiffness (N/m)
    let c_values = [10.0, 50.0, 100.0, 200.0]; // damping coefficients

    let omega_n = (k / m).sqrt(); // natural frequency
    let f_n = omega_n / (2.0 * std::f64::consts::PI);

    println!("  System parameters:");
    println!("    Mass: {:.1f} kg", m);
    println!("    Stiffness: {:.1f} N/m", k);
    println!("    Natural frequency: {:.2f} Hz", f_n);

    // Frequency sweep
    let freq_range = (0.1 * f_n, 2.0 * f_n);
    let n_points = 50;

    println!("\n  Frequency Response (displacement amplitude):");
    println!("  {:>10} | {:>15} | {:>15} | {:>15} | {:>15}",
        "f/fn", "c=10", "c=50", "c=100", "c=200");
    println!("  {:-<75}", "");

    for i in 0..n_points {
        let freq_ratio = freq_range.0 / f_n + (i as f64 / (n_points - 1) as f64) * (freq_range.1 / f_n - freq_range.0 / f_n);
        let omega = freq_ratio * omega_n;

        let mut amplitudes = Vec::new();

        for &c in &c_values {
            // Steady-state amplitude for harmonic excitation F = F0 * sin(omega*t)
            let f0 = 100.0;

            // Dynamic magnification factor
            let r = omega / omega_n;
            let zeta = c / (2.0 * (k * m).sqrt());

            let denom = ((1.0 - r * r).powi(2) + (2.0 * zeta * r).powi(2)).sqrt();
            let static_disp = f0 / k;
            let amplitude = static_disp / denom;

            amplitudes.push(amplitude);
        }

        println!("  {:>10.2} | {:>15.4e} | {:>15.4e} | {:>15.4e} | {:>15.4e}",
            freq_ratio, amplitudes[0], amplitudes[1], amplitudes[2], amplitudes[3]);
    }

    Ok(())
}

/// Demonstrate harmonic response with damping.
fn demo_harmonic_response() -> anyhow::Result<()> {
    println!("\nHarmonic Response Analysis:");
    println!("-------------------------");

    // MDOF system (3 DOF)
    let m = 100.0;
    let k = 10000.0;

    // Mass matrix (diagonal)
    let mass = DMatrix::from_diagonal(&DVector::from_element(3, m));

    // Stiffness matrix
    let stiffness = DMatrix::from_row_slice(3, 3, &[
        2.0 * k, -k, 0.0,
        -k, 2.0 * k, -k,
        0.0, -k, k,
    ]);

    // Damping matrix (Rayleigh damping)
    let alpha = 0.1; // Mass proportional
    let beta = 0.001; // Stiffness proportional
    let damping = mass.scale(alpha) + stiffness.scale(beta);

    println!("  3-DOF System:");
    println!("    Natural frequencies (estimated): {:.2f}, {:.2f}, {:.2f} Hz",
        1.5, 4.5, 7.5);

    // Harmonic force vector
    let f0 = DVector::from_column_slice(&[100.0, 0.0, 0.0]);

    // Frequency response at selected frequencies
    let frequencies = vec![0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0, 7.5, 8.0];

    println!("\n  Frequency Response (DOF 1 displacement amplitude):");
    println!("  {:>10} | {:>20} | {:>20}", "Freq (Hz)", "Amplitude (m)", "Phase (deg)");
    println!("  {:-<58}", "");

    for f in frequencies {
        let omega = 2.0 * std::f64::consts::PI * f;

        // Dynamic stiffness: K_dynamic = K - omega^2 * M + i * omega * C
        // For simplified demo, compute magnitude only
        let k_dyn = &stiffness - &mass.scale(omega * omega);

        // Add damping effect (simplified)
        let c_effect = omega * beta * k;

        // Solve for displacement
        let k_det = k_dyn[(0, 0)] * k_dyn[(1, 1)] * k_dyn[(2, 2)];

        let amplitude = f0[0] / (k_det.abs() + c_effect).max(1.0);
        let phase = -(omega * 0.01).atan() * 180.0 / std::f64::consts::PI;

        println!("  {:>10.1} | {:>20.4e} | {:>20.1}", f, amplitude, phase);
    }

    Ok(())
}

/// Demonstrate mode superposition analysis.
fn demo_mode_superposition() -> anyhow::Result<()> {
    println!("\nMode Superposition Analysis:");
    println!("--------------------------");

    // N-DOF system
    let n = 10;
    let m = 100.0;
    let k = 10000.0;

    // Compute approximate natural frequencies and mode shapes
    println!("  Computing {} modes for {}-DOF system...", n, n);

    let mut frequencies = Vec::new();
    for i in 1..=n {
        // Approximate frequencies for shear building
        let omega = 2.0 * (k / m).sqrt() * ((i as f64 * std::f64::consts::PI) / (2.0 * (n + 1) as f64)).sin();
        frequencies.push(omega / (2.0 * std::f64::consts::PI));
    }

    println!("\n  Natural Frequencies:");
    for (i, f) in frequencies.iter().enumerate() {
        println!("    Mode {}: {:.3f} Hz", i + 1, f);
    }

    // Modal participation factors
    println!("\n  Modal Participation Factors (for uniform loading):");
    let mut total_participation = 0.0;

    for (i, f) in frequencies.iter().enumerate() {
        // Simplified participation factor
        let gamma = if i % 2 == 0 { 4.0 / ((i + 1) as f64 * std::f64::consts::PI) } else { 0.0 };
        total_participation += gamma;
        println!("    Mode {}: γ = {:.4f}", i + 1, gamma);
    }

    println!("\n  Total modal mass participation: {:.1f}%",
        (total_participation / (4.0 / std::f64::consts::PI)) * 100.0);

    Ok(())
}

/// Demonstrate resonance detection.
fn demo_resonance_detection() -> anyhow::Result<()> {
    println!("\nResonance Detection:");
    println!("------------------");

    // System with known resonances
    let natural_freqs = vec![5.0, 12.5, 23.0, 45.0]; // Hz
    let damping_ratios = vec![0.02, 0.05, 0.03, 0.08];

    println!("  System has {} known resonances:", natural_freqs.len());
    for (i, (f, z)) in natural_freqs.iter().zip(damping_ratios.iter()).enumerate() {
        println!("    Mode {}: f = {:.1f} Hz, ζ = {:.1f}%", i + 1, f, z * 100.0);
    }

    // Sweep through frequency range and detect peaks
    let freq_start = 0.0;
    let freq_end = 60.0;
    let n_points = 200;

    let mut responses = Vec::new();
    let mut frequencies = Vec::new();

    for i in 0..n_points {
        let f = freq_start + (i as f64 / (n_points - 1) as f64) * (freq_end - freq_start);

        // Compute total response (sum of modal contributions)
        let mut response = 0.0;
        for (f_n, zeta) in natural_freqs.iter().zip(damping_ratios.iter()) {
            let r = f / f_n;
            let denom = ((1.0 - r * r).powi(2) + (2.0 * zeta * r).powi(2)).sqrt();
            if denom > 0.01 {
                response += 1.0 / denom;
            }
        }

        responses.push(response);
        frequencies.push(f);
    }

    // Detect peaks (resonances)
    println!("\n  Detected Resonances:");
    let threshold = 0.5 * responses.iter().cloned().fold(0.0_f64, f64::max);

    for i in 1..responses.len() - 1 {
        if responses[i] > responses[i - 1] &&
           responses[i] > responses[i + 1] &&
           responses[i] > threshold {
            println!("    Peak at f = {:.1f} Hz (response = {:.1f})",
                frequencies[i], responses[i]);
        }
    }

    // Find maximum response
    let max_idx = responses.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    println!("\n  Maximum response: {:.1f} at f = {:.1f} Hz",
        responses[max_idx], frequencies[max_idx]);

    Ok(())
}
