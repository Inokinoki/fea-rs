//! Dynamic Analysis Acceleration Example.
//!
//! This example demonstrates:
//! - Modal superposition for frequency response
//! - Component Mode Synthesis (Craig-Bampton)
//! - Proper Orthogonal Decomposition
//! - Krylov subspace reduction

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("=== Dynamic Analysis Acceleration Example ===\n");

    // Part 1: Modal Acceleration
    demonstrate_modal_acceleration()?;

    // Part 2: Component Mode Synthesis
    demonstrate_cms()?;

    // Part 3: Proper Orthogonal Decomposition
    demonstrate_pod()?;

    // Part 4: Krylov Reduction
    demonstrate_krylov()?;

    // Part 5: Complete Dynamic Analysis
    demonstrate_complete_dynamic_analysis()?;

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Demonstrates modal acceleration for frequency response
fn demonstrate_modal_acceleration() -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Part 1: Modal Acceleration for Dynamic Response");
    println!("─".repeat(60));

    // Create a 10-DOF system
    let n = 10;
    let mut k = DMatrix::zeros(n, n);
    let mut m = DMatrix::zeros(n, n);

    for i in 0..n {
        k[(i, i)] = 2.0;
        m[(i, i)] = 1.0;
        if i > 0 {
            k[(i, i - 1)] = -1.0;
        }
        if i < n - 1 {
            k[(i, i + 1)] = -1.0;
        }
    }

    // Compute eigenvalues and eigenvectors
    let eigen = nalgebra::SymmetricEigen::new(k.clone());
    let frequencies: DVector<f64> = eigen.eigenvalues.map(|e| e.sqrt().max(0.0));
    let mode_shapes = eigen.eigenvectors.clone();

    println!("\nSystem Properties:");
    println!("  DOFs: {}", n);
    println!("  Natural frequencies (rad/s):");
    for i in 0..n {
        println!("    Mode {}: {:.4}", i + 1, frequencies[i]);
    }

    // Create modal accelerator with first 5 modes
    let mut modal = ModalAcceleration::new(frequencies, mode_shapes);
    modal.truncate(5);

    println!("\nModal Truncation:");
    println!("  Retained modes: {}", modal.num_modes);

    // Frequency response
    let force = DVector::from_column_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    println!("\nFrequency Response (DOF 1):");
    println!("  {:<15} | {:<15}", "Omega (rad/s)", "Response");
    println!("  {}", "-".repeat(35));

    for omega in [5.0, 10.0, 15.0, 20.0, 25.0] {
        let response = modal.frequency_response(omega, &force);
        println!("  {:<15.4} | {:<15.6}", omega, response[0]);
    }

    // Effective modal mass
    let direction = DVector::from_column_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let eff_masses = modal.effective_modal_mass(&direction);

    println!("\nEffective Modal Mass:");
    let mut cumulative = 0.0;
    for i in 0..modal.num_modes {
        cumulative += eff_masses[i];
        println!("  Mode {}: {:.4} (cumulative: {:.4})", i + 1, eff_masses[i], cumulative);
    }

    // Mass participation for 90% threshold
    let modes_needed = modal.mass_participation_ratio(&eff_masses, 0.90);
    println!("\nModes for 90% mass participation: {}", modes_needed);

    println!("\n  Modal Acceleration: OK");
    Ok(())
}

/// Demonstrates Component Mode Synthesis
fn demonstrate_cms() -> anyhow::Result<()> {
    println!("\n─".repeat(60));
    println!("Part 2: Component Mode Synthesis (Craig-Bampton)");
    println!("─".repeat(60));

    // Create a 20-DOF system
    let n = 20;
    let mut k = DMatrix::zeros(n, n);
    let m = DMatrix::identity(n, n);

    for i in 0..n {
        k[(i, i)] = 2.0;
        if i > 0 {
            k[(i, i - 1)] = -1.0;
        }
        if i < n - 1 {
            k[(i, i + 1)] = -1.0;
        }
    }

    println!("\nOriginal System:");
    println!("  DOFs: {}", n);

    // Select boundary DOFs (ends)
    let boundary = vec![0, n - 1];

    // Perform Craig-Bampton reduction
    let mut cms = ComponentModeSynthesis::new(boundary.len(), 5);
    let (k_r, m_r) = cms.craig_bampton_reduction(&k, &m, &boundary);

    println!("\nCraig-Bampton Reduction:");
    println!("  Boundary DOFs: {}", boundary.len());
    println!("  Internal modes retained: 5");
    println!("  Reduced DOFs: {}", k_r.nrows());
    println!("  Reduction ratio: {:.2}%", 100.0 * (1.0 - k_r.nrows() as f64 / n as f64));

    // Compute reduced frequencies
    let k_r_clone = k_r.clone();
    let m_r_clone = m_r.clone();

    if let Some(m_r_inv) = m_r_clone.try_inverse() {
        let generalized_k = m_r_inv * k_r_clone;
        let eigen = nalgebra::SymmetricEigen::new(generalized_k);

        println!("\nReduced System Frequencies (rad/s):");
        for i in 0..eigen.eigenvalues.len().min(5) {
            let freq = eigen.eigenvalues[i].sqrt().max(0.0);
            println!("  Mode {}: {:.4}", i + 1, freq);
        }
    }

    // Guyan reduction comparison
    let (k_guyan, m_guyan) = cms.guyan_reduction(&k, &m, &boundary);

    println!("\nGuyan Reduction:");
    println!("  Reduced DOFs: {}", k_guyan.nrows());

    println!("\n  Component Mode Synthesis: OK");
    Ok(())
}

/// Demonstrates Proper Orthogonal Decomposition
fn demonstrate_pod() -> anyhow::Result<()> {
    println!("\n─".repeat(60));
    println!("Part 3: Proper Orthogonal Decomposition");
    println!("─".repeat(60));

    // Generate synthetic snapshot data
    // Simulating displacement snapshots from a dynamic simulation
    let n_spatial = 50;
    let n_snapshots = 20;

    let mut snapshots = DMatrix::zeros(n_spatial, n_snapshots);

    for s in 0..n_snapshots {
        let t = s as f64 * 0.1;
        for i in 0..n_spatial {
            let x = i as f64 / n_spatial as f64;
            // Superposition of modes
            snapshots[(i, s)] =
                (std::f64::consts::PI * x).sin() * (5.0 * t).cos()
                + 0.5 * (2.0 * std::f64::consts::PI * x).sin() * (10.0 * t).cos()
                + 0.25 * (3.0 * std::f64::consts::PI * x).sin() * (15.0 * t).cos();
        }
    }

    println!("\nSnapshot Data:");
    println!("  Spatial points: {}", n_spatial);
    println!("  Snapshots: {}", n_snapshots);

    // Compute POD with 99% energy threshold
    let pod = ProperOrthogonalDecomposition::from_snapshots(&snapshots, 0.99);

    println!("\nPOD Results:");
    println!("  Modes retained: {}", pod.modes.ncols());
    println!("  Energy captured: {:.2}%", pod.energy_captured * 100.0);
    println!("  Original DOFs: {}", n_spatial);
    println!("  Reduced DOFs: {}", pod.modes.ncols());
    println!("  Reduction ratio: {:.2}%", 100.0 * (1.0 - pod.modes.ncols() as f64 / n_spatial as f64));

    // Show singular values
    println!("\nSingular Values:");
    for i in 0..pod.singular_values.len().min(5) {
        println!("  Mode {}: {:.6}", i + 1, pod.singular_values[i]);
    }

    // Demonstrate reconstruction
    let q_reduced = DVector::from_column_slice(&[1.0, 0.5, 0.25]);
    if q_reduced.len() <= pod.modes.ncols() {
        let reconstructed = pod.reconstruct(&q_reduced);
        println!("\nReconstruction:");
        println!("  Reduced coordinates: {:?}", q_reduced.as_slice());
        println!("  Reconstructed DOFs: {}", reconstructed.len());
    }

    println!("\n  POD: OK");
    Ok(())
}

/// Demonstrates Krylov subspace reduction
fn demonstrate_krylov() -> anyhow::Result<()> {
    println!("\n─".repeat(60));
    println!("Part 4: Krylov Subspace Reduction");
    println!("─".repeat(60));

    // Create a 100-DOF system
    let n = 100;
    let mut a = DMatrix::zeros(n, n);
    let b = DVector::from_column_slice(&vec![1.0; n]);

    for i in 0..n {
        a[(i, i)] = 4.0;
        if i > 0 {
            a[(i, i - 1)] = -1.0;
        }
        if i < n - 1 {
            a[(i, i + 1)] = -1.0;
        }
    }

    println!("\nOriginal System:");
    println!("  DOFs: {}", n);

    // Create Krylov reduction with order 10
    let krylov = KrylovReduction::new(&a, &b, 10);

    println!("\nKrylov Reduction:");
    println!("  Reduction order: {}", krylov.order);
    println!("  Reduced DOFs: {}", krylov.order);
    println!("  Reduction ratio: {:.2}%", 100.0 * (1.0 - krylov.order as f64 / n as f64));

    // Reduce system
    let (a_r, b_r) = krylov.reduce(&a, &b);

    println!("\nReduced System:");
    println!("  Matrix size: {}x{}", a_r.nrows(), a_r.ncols());
    println!("  Input vector size: {}", b_r.len());

    // Verify moment matching (first few moments should match)
    let original_moment0 = b.norm();
    let reduced_moment0 = b_r.norm();

    println!("\nMoment Matching:");
    println!("  Original ||b||: {:.6}", original_moment0);
    println!("  Reduced ||b_r||: {:.6}", reduced_moment0);
    println!("  Relative error: {:.6e}", (original_moment0 - reduced_moment0) / original_moment0);

    println!("\n  Krylov Reduction: OK");
    Ok(())
}

/// Complete dynamic analysis workflow
fn demonstrate_complete_dynamic_analysis() -> anyhow::Result<()> {
    println!("\n─".repeat(60));
    println!("Part 5: Complete Dynamic Analysis Workflow");
    println!("─".repeat(60));

    // Create a structural model
    let n = 30;
    let mut k = DMatrix::zeros(n, n);
    let mut m = DMatrix::zeros(n, n);

    for i in 0..n {
        k[(i, i)] = 2.0 * (i + 1) as f64;
        m[(i, i)] = 1.0;
        if i > 0 {
            k[(i, i - 1)] = -1.0 * i as f64;
        }
        if i < n - 1 {
            k[(i, i + 1)] = -1.0 * (i + 1) as f64;
        }
    }

    println!("\nStructural Model:");
    println!("  DOFs: {}", n);

    // Step 1: Compute eigenpairs
    let eigen = nalgebra::SymmetricEigen::new(k.clone());
    let frequencies: DVector<f64> = eigen.eigenvalues.map(|e| e.sqrt().max(0.0));
    let mode_shapes = eigen.eigenvectors;

    println!("\nStep 1: Eigenvalue Analysis");
    println!("  First 5 natural frequencies:");
    for i in 0..5 {
        println!("    Mode {}: {:.4} rad/s", i + 1, frequencies[i]);
    }

    // Step 2: Modal truncation
    let mut modal = ModalAcceleration::new(frequencies, mode_shapes);
    modal.truncate(10);

    println!("\nStep 2: Modal Truncation");
    println!("  Retained modes: {}", modal.num_modes);

    // Step 3: Frequency response analysis
    let force = DVector::from_column_slice(&vec![1.0; n]);

    println!("\nStep 3: Frequency Response Analysis");
    println!("  {:<12} | {:<12} | {:<12}", "Omega", "DOF 1", "DOF 15");
    println!("  {}", "-".repeat(42));

    for omega in [1.0, 5.0, 10.0, 15.0, 20.0] {
        let response = modal.frequency_response(omega, &force);
        println!("  {:<12.4} | {:<12.6} | {:<12.6}", omega, response[0], response[14]);
    }

    // Step 4: Model reduction with POD
    // Generate snapshots at different frequencies
    let n_snapshots = 15;
    let mut snapshots = DMatrix::zeros(n, n_snapshots);

    for (s, &omega) in [1.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0].iter().enumerate() {
        let response = modal.frequency_response(omega, &force);
        for i in 0..n {
            snapshots[(i, s)] = response[i];
        }
    }

    let pod = ProperOrthogonalDecomposition::from_snapshots(&snapshots, 0.95);

    println!("\nStep 4: POD Model Reduction");
    println!("  Snapshots: {}", n_snapshots);
    println!("  POD modes: {}", pod.modes.ncols());
    println!("  Energy captured: {:.2}%", pod.energy_captured * 100.0);

    // Step 5: Reduced system
    let (k_r, m_r) = pod.project_system(&k, &m);

    println!("\nStep 5: Reduced System Analysis");
    println!("  Original DOFs: {}", n);
    println!("  Reduced DOFs: {}", k_r.nrows());

    // Reduced eigenvalue analysis
    if let Some(m_r_inv) = m_r.try_inverse() {
        let gen_k = m_r_inv * k_r;
        let eigen_r = nalgebra::SymmetricEigen::new(gen_k);

        println!("\n  Reduced frequencies vs Full:");
        println!("  {:<12} | {:<12} | {:<12}", "Mode", "Full", "Reduced");
        println!("  {}", "-".repeat(42));

        for i in 0..pod.modes.ncols().min(5) {
            let freq_full = frequencies[i];
            let freq_red = eigen_r.eigenvalues[i].sqrt().max(0.0);
            let error = (freq_full - freq_red) / freq_full * 100.0;
            println!("  {:<12} | {:<12.4} | {:<12.4} ({:.2}%)",
                     i + 1, freq_full, freq_red, error.abs());
        }
    }

    println!("\n  Complete Dynamic Analysis: OK");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_acceleration_example() {
        demonstrate_modal_acceleration().unwrap();
        demonstrate_cms().unwrap();
        demonstrate_pod().unwrap();
        demonstrate_krylov().unwrap();
        demonstrate_complete_dynamic_analysis().unwrap();
    }
}
