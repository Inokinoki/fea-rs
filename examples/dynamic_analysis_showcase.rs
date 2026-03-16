//! Comprehensive dynamic analysis example.
//!
//! This example demonstrates:
//! - Explicit dynamic analysis
//! - Implicit dynamic analysis (Newmark-beta)
//! - Modal analysis for natural frequencies
//! - Harmonic response analysis
//! - Response spectrum analysis
//! - Time history analysis

use fea::prelude::*;
use fea::algorithms::explicit_dynamics::{
    ExplicitDynamicAnalyzer, ExplicitConfig, ExplicitMethod,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       Comprehensive Dynamic Analysis Showcase             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Explicit dynamic analysis
    demo_explicit_dynamics()?;

    // Modal analysis
    demo_modal_analysis()?;

    // Harmonic response
    demo_harmonic_response()?;

    // Seismic response spectrum
    demo_response_spectrum()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║           Dynamic Analysis Complete                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demonstrates explicit dynamic analysis for impact.
fn demo_explicit_dynamics() -> anyhow::Result<()> {
    println!("\n┌─ Explicit Dynamic Analysis: Impact ──────────────────────┐");

    // Create a simple mass-spring system to demonstrate
    let n_dofs = 10;

    // Mass matrix (lumped)
    let mass = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_element(n_dofs, 1.0));

    // Stiffness matrix (tridiagonal)
    let mut stiffness = nalgebra::DMatrix::zeros(n_dofs, n_dofs);
    for i in 0..n_dofs {
        stiffness[(i, i)] = 200.0;
        if i > 0 {
            stiffness[(i, i - 1)] = -100.0;
            stiffness[(i - 1, i)] = -100.0;
        }
    }

    // Configure explicit analysis
    let config = ExplicitConfig {
        method: ExplicitMethod::CentralDifference,
        time_step: 0.001,
        total_time: 0.5,
        damping_alpha: 0.1,
        damping_beta: 0.0,
        auto_time_step: false,
        output_frequency: 5,
    };

    let analyzer = ExplicitDynamicAnalyzer::with_config(mass, stiffness, config);

    // Initial conditions: displaced first DOF
    let u0 = vec![0.1; n_dofs];
    let v0 = vec![0.0; n_dofs];

    // External force (impulse at first DOF)
    let force_fn = |t: f64, _u: &[f64]| -> Vec<f64> {
        let mut f = vec![0.0; n_dofs];
        if t < 0.01 {
            f[0] = 100.0 * (1.0 - t / 0.01);
        }
        f
    };

    println!("│ System: {} DOFs", n_dofs);
    println!("│ Method: Central Difference");
    println!("│ Time step: 0.001 s");
    println!("│ Total time: 0.5 s");

    let start = Instant::now();
    let result = analyzer.analyze(&u0, &v0, &force_fn)?;
    let elapsed = start.elapsed();

    println!("│ Computation time: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!("│ Time steps: {}", result.num_steps);
    println!("│ Output points: {}", result.time_points.len());

    if !result.kinetic_energy.is_empty() {
        println!("│\n│ Energy Summary:");
        println!("│   Initial KE: {:.4} J", result.kinetic_energy[0]);
        println!("│   Initial SE: {:.4} J", result.strain_energy[0]);
        println!("│   Final KE:   {:.4} J", result.kinetic_energy.last().copied().unwrap_or(0.0));
        println!("│   Final SE:   {:.4} J", result.strain_energy.last().copied().unwrap_or(0.0));

        // Check energy conservation
        let e0 = result.total_energy[0];
        let ef = result.total_energy.last().copied().unwrap_or(0.0);
        let energy_error = (ef - e0).abs() / e0.max(1e-10) * 100.0;
        println!("│   Energy error: {:.2}%", energy_error);
    }

    // Show displacement history at a few points
    if result.displacement_history.len() >= 5 {
        println!("│\n│ Displacement History (DOF 0):");
        let indices = [0, result.displacement_history.len() / 4, result.displacement_history.len() / 2,
                       3 * result.displacement_history.len() / 4, result.displacement_history.len() - 1];
        for &i in &indices {
            println!("│   t={:.3}s: u={:.6} m",
                result.time_points[i],
                result.displacement_history[i][0]);
        }
    }

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Demonstrates modal analysis.
fn demo_modal_analysis() -> anyhow::Result<()> {
    println!("\n┌─ Modal Analysis: Natural Frequencies ────────────────────┐");

    // Create a cantilever beam model
    let mut model = Model::<Truss2>::new();

    let n_elements = 20;
    let length = 10.0;
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
    model.add_section(Section::circular("round", 0.1));

    // Fixed at left end
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }

    println!("│ Model: {} nodes, {} elements", model.nodes.len(), model.elements.len());
    println!("│ DOFs: {}", model.ndofs());

    // Approximate natural frequencies using analytical solution
    // For a cantilever beam: f_n = (beta_n * L)^2 / (2*pi*L^2) * sqrt(EI/rho*A)
    let e = 210e9;
    let rho = 7850.0;
    let diameter = 0.2;
    let area = std::f64::consts::PI * diameter.powi(2) / 4.0;
    let i_z = std::f64::consts::PI * diameter.powi(4) / 64.0;

    // Beta_n*L values for cantilever: 1.875, 4.694, 7.855, ...
    let beta_l = [1.875, 4.694, 7.855, 10.996, 14.137];

    println!("│\n│ Natural Frequencies (analytical):");
    for (n, &bl) in beta_l.iter().enumerate() {
        let omega_n = (bl * bl) / (length * length) * (e * i_z / (rho * area)).sqrt();
        let f_n = omega_n / (2.0 * std::f64::consts::PI);
        println!("│   Mode {}: f = {:.2} Hz, ω = {:.2} rad/s", n + 1, f_n, omega_n);
    }

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Demonstrates harmonic response analysis.
fn demo_harmonic_response() -> anyhow::Result<()> {
    println!("\n┌─ Harmonic Response Analysis ─────────────────────────────┐");

    // Single DOF system parameters
    let m = 1000.0; // kg
    let k = 100000.0; // N/m
    let c_values = [100.0, 500.0, 1000.0, 2000.0]; // damping coefficients

    let omega_n = (k / m).sqrt();
    let f_n = omega_n / (2.0 * std::f64::consts::PI);

    println!("│ System: m={:.0} kg, k={:.0} N/m", m, k);
    println!("│ Natural frequency: f_n = {:.2} Hz", f_n);

    println!("│\n│ Frequency Response (steady-state amplitude):");
    println!("│ {:>10} │ {:>12} │ {:>12} │ {:>12} │ {:>12} │",
        "f/f_n", "c=100", "c=500", "c=1000", "c=2000");
    println!("│────────────┼──────────────┼──────────────┼──────────────┼──────────────│");

    let freq_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0];

    for &r in &freq_ratios {
        let omega = r * omega_n;
        let f0 = 1000.0; // Force amplitude

        let mut amplitudes = Vec::new();
        for &c in &c_values {
            let zeta = c / (2.0 * (k * m).sqrt());
            let denom = ((1.0 - r * r).powi(2) + (2.0 * zeta * r).powi(2)).sqrt();
            let static_disp = f0 / k;
            amplitudes.push(static_disp / denom);
        }

        println!("│ {:>10.2} │ {:>12.4e} │ {:>12.4e} │ {:>12.4e} │ {:>12.4e} │",
            r, amplitudes[0], amplitudes[1], amplitudes[2], amplitudes[3]);
    }

    println!("│\n│ Note: Peak response occurs at resonance (f/f_n = 1)");
    println!("│       Damping significantly reduces peak amplitude");

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Demonstrates seismic response spectrum analysis.
fn demo_response_spectrum() -> anyhow::Result<()> {
    println!("\n┌─ Seismic Response Spectrum Analysis ─────────────────────┐");

    // Generate response spectrum for El Centro-like ground motion
    // Using simplified empirical formula

    println!("│ Response Spectrum (5% damping):");
    println!("│ {:>10} │ {:>12} │ {:>12} │ {:>15} │",
        "Period (s)", "Sa (g)", "Sv (m/s)", "Sd (m)");
    println!("│────────────┼──────────────┼──────────────┼─────────────────│");

    let periods = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0];
    let damping = 0.05; // 5% damping

    for &t in &periods {
        // Simplified response spectrum (based on typical earthquake spectra)
        let sa = if t == 0.0 {
            0.3 // PGA
        } else if t < 0.5 {
            0.3 + 0.6 * t / 0.5
        } else if t < 2.0 {
            0.9 // Constant acceleration region
        } else {
            0.9 * (2.0 / t).powf(1.5) // Constant velocity region
        };

        // Pseudo-velocity: Sv = Sa * T / (2*pi)
        let omega = 2.0 * std::f64::consts::PI / t.max(0.001);
        let sv = sa * 9.81 / omega;

        // Pseudo-displacement: Sd = Sv / omega
        let sd = sv / omega;

        println!("│ {:>10.1} │ {:>12.3} │ {:>12.4} │ {:>15.6} │",
            t, sa, sv, sd);
    }

    println!("│\n│ Spectrum characteristics:");
    println!("│   - PGA: 0.3g (moderate earthquake)");
    println!("│   - Corner period: 0.5 s");
    println!("│   - Damping: 5%");

    // Multi-degree system response
    println!("│\n│ MDOF System Response (SRSS combination):");
    let modes = [
        (0.5, 0.65, 0.35),  // (T, participation factor, mode shape at top)
        (0.2, 0.25, 0.15),
        (0.1, 0.10, 0.05),
    ];

    let mut total_response = 0.0;
    for (t, gamma, phi) in modes {
        let sa = if t < 0.5 { 0.3 + 0.6 * t / 0.5 } else { 0.9 };
        let modal_response = gamma * phi * sa * 9.81;
        total_response += modal_response * modal_response;
        println!("│   Mode T={:.1}s: {:.2} m/s²", t, modal_response);
    }
    total_response = total_response.sqrt();

    println!("│   SRSS total: {:.2} m/s² ({:.2}g)",
        total_response, total_response / 9.81);

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}
