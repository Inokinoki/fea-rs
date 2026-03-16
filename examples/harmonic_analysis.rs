//! Harmonic (frequency response) analysis example.
//!
//! This example demonstrates frequency response analysis of a simple structure
//! subjected to harmonic excitation.

use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("=== Harmonic Analysis Example ===\n");

    // Create a simple cantilever beam model
    let mut model = Model::<Truss2>::new();

    // Material and section
    model.add_material(STEEL_A36);
    model.add_section(Section::circular("round", 0.02)); // 20mm diameter

    // Create nodes for a cantilever truss
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(0.5, 0.0));
    let n2 = model.add_node(Node::new_2d(1.0, 0.0));
    let n3 = model.add_node(Node::new_2d(1.5, 0.0));
    let n4 = model.add_node(Node::new_2d(2.0, 0.0));

    // Add elements
    for i in 0..4 {
        model.add_element(Truss2::new(i, i + 1));
    }

    // Boundary conditions: fixed at node 0
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
    }
    // Constrain transverse DOFs at other nodes
    for &node in &[n1, n2, n3, n4] {
        for dof in [Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(node, dof));
        }
    }

    // First, run modal analysis to find natural frequencies
    println!("Running modal analysis...");
    let modal_analysis = ModalAnalysis::new();
    let modal_config = ModalConfig {
        num_modes: 5,
        consistent_mass: false,
        max_iterations: 1000,
        tolerance: 1e-10,
    };
    let modal_result = modal_analysis.run_modal(&mut model, &modal_config)?;

    println!("Natural frequencies:");
    for (i, (&w, &f_hz)) in modal_result.frequencies.iter()
        .zip(modal_result.frequencies_hz.iter())
        .enumerate()
    {
        println!("  Mode {}: {:.2} rad/s ({:.2} Hz)", i + 1, w, f_hz);
    }

    // Find first natural frequency for excitation range
    let first_freq = modal_result.frequencies.first().copied().unwrap_or(100.0);
    println!("\nFirst natural frequency: {:.2} rad/s", first_freq);

    // Run harmonic analysis
    println!("\nRunning harmonic analysis...");

    let harmonic_analysis = HarmonicAnalysis::new();

    // Apply harmonic load at the tip (node 4) in X direction
    let mut load_amplitude = vec![0.0; model.ndofs()];
    if let Some(dof_idx) = model.dof_index(n4, Dof::Ux) {
        load_amplitude[dof_idx] = 100.0; // 100 N amplitude
    }

    let harmonic_config = HarmonicConfig {
        freq_start: 0.0,
        freq_end: first_freq * 2.5, // Sweep past first resonance
        num_points: 100,
        damping_ratio: 0.02, // 2% damping
        load_amplitude,
    };

    let harmonic_result = harmonic_analysis.run_static(&mut model, &harmonic_config)?;

    // Find peak response
    let mut max_amplitude = 0.0;
    let mut peak_freq = 0.0;
    let mut peak_dof = 0;

    for (i, (&freq, amps)) in harmonic_result.frequencies.iter()
        .zip(harmonic_result.displacement_amplitudes.iter())
        .enumerate()
    {
        for (dof, &amp) in amps.iter().enumerate() {
            if amp > max_amplitude {
                max_amplitude = amp;
                peak_freq = freq;
                peak_dof = dof;
            }
        }
    }

    println!("\nFrequency Response Results:");
    println!("  - Frequency range: {:.1} to {:.1} rad/s",
        harmonic_result.frequencies.first().unwrap_or(&0.0),
        harmonic_result.frequencies.last().unwrap_or(&0.0));
    println!("  - Number of frequency points: {}", harmonic_result.frequencies.len());
    println!("\nPeak Response:");
    println!("  - Maximum amplitude: {:.6e} m", max_amplitude);
    println!("  - Peak frequency: {:.2} rad/s ({:.2} Hz)",
        peak_freq, peak_freq / (2.0 * std::f64::consts::PI));
    println!("  - Peak DOF index: {}", peak_dof);

    // Print frequency response table (sample points)
    println!("\nFrequency Response Table (sample):");
    println!("  Freq (rad/s) | Freq (Hz) | Ampl (m)    | Phase (deg)");
    println!("  -------------|-----------|-------------|------------");

    let sample_indices = [0, harmonic_result.frequencies.len() / 4,
                          harmonic_result.frequencies.len() / 2,
                          3 * harmonic_result.frequencies.len() / 4,
                          harmonic_result.frequencies.len() - 1];

    for &idx in &sample_indices {
        let freq = harmonic_result.frequencies[idx];
        let freq_hz = freq / (2.0 * std::f64::consts::PI);
        let amp = harmonic_result.displacement_amplitudes[idx][peak_dof];
        let phase = harmonic_result.phase_angles[idx][peak_dof];
        let phase_deg = phase * 180.0 / std::f64::consts::PI;
        println!("  {:12.2} | {:9.2} | {:11.6e} | {:10.1}",
            freq, freq_hz, amp, phase_deg);
    }

    // Verify resonance detection
    let expected_resonance = first_freq; // Should be near first natural freq
    let resonance_error = (peak_freq - expected_resonance).abs() / expected_resonance * 100.0;

    println!("\nValidation:");
    println!("  - Expected resonance: {:.2} rad/s", expected_resonance);
    println!("  - Detected peak: {:.2} rad/s", peak_freq);
    println!("  - Error: {:.2}%", resonance_error);

    if resonance_error < 10.0 {
        println!("  - VALIDATION PASSED (peak near natural frequency)");
    } else {
        println!("  - Note: Peak may be at higher mode or damping effect");
    }

    Ok(())
}
