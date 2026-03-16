//! Thermal-Stress Coupling Example.
//!
//! This example demonstrates coupled thermal-stress analysis:
//! - Temperature-dependent material properties
//! - Thermal expansion effects
//! - Thermal stress calculation

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Thermal-Stress Coupling Example                  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Example 1: Thermal expansion
    example_thermal_expansion()?;

    // Example 2: Temperature-dependent properties
    example_temperature_dependent()?;

    // Example 3: Coupled thermal-stress analysis
    example_thermal_stress_analysis()?;

    // Example 4: Thermal contact effects
    example_thermal_contact()?;

    println!("\n=== Thermal-Stress Examples Complete ===");
    Ok(())
}

/// Example 1: Thermal expansion.
fn example_thermal_expansion() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 1: Thermal Expansion");
    println!("{}\n", "=".repeat(60));

    // Create isotropic thermal expansion tensor (steel)
    let alpha = ThermalExpansionTensor::isotropic(12e-6);

    println!("Material: Steel");
    println!("  Thermal expansion coefficient: {:.2e} /K", alpha.alpha_x);
    println!();

    // Compute thermal strain for various temperature changes
    let delta_temps = vec![20.0, 50.0, 100.0, 200.0, 500.0];

    println!("Thermal Strain vs Temperature Change:");
    println!("  {:>12} | {:>15}", "ΔT (K)", "ε_thermal");
    println!("  {}", "-".repeat(32));

    for dt in delta_temps {
        let strain = alpha.thermal_strain(dt);
        println!("  {:>12.1} | {:>15.6}", dt, strain[0]);
    }
    println!();

    // Orthotropic material (composite)
    println!("Material: Orthotropic Composite");
    let alpha_ortho = ThermalExpansionTensor::orthotropic(1e-6, 10e-6, 10e-6);
    println!("  α_x: {:.2e} /K", alpha_ortho.alpha_x);
    println!("  α_y: {:.2e} /K", alpha_ortho.alpha_y);
    println!("  α_z: {:.2e} /K", alpha_ortho.alpha_z);

    let strain_100 = alpha_ortho.thermal_strain(100.0);
    println!("\n  Thermal strain at ΔT=100K:");
    println!("    ε_x: {:.6}", strain_100[0]);
    println!("    ε_y: {:.6}", strain_100[1]);
    println!("    ε_z: {:.6}", strain_100[2]);
    println!();

    Ok(())
}

/// Example 2: Temperature-dependent material properties.
fn example_temperature_dependent() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 2: Temperature-Dependent Properties");
    println!("{}\n", "=".repeat(60));

    // Create temperature-dependent properties for steel
    let props = TemperatureDependentProperties::new(
        20.0,        // Reference temp (C)
        210e9,       // Young's modulus at ref (Pa)
        -50e6,       // Modulus decreases ~50 MPa per degree C
        ThermalExpansionTensor::isotropic(12e-6),
    );

    println!("Material: Temperature-Dependent Steel");
    println!("  Reference temperature: {:.1} C", props.reference_temp);
    println!("  E at reference: {:.1} GPa", props.youngs_modulus_ref / 1e9);
    println!();

    // Properties at various temperatures
    let temps = vec![20.0, 100.0, 200.0, 400.0, 600.0];

    println!("Properties vs Temperature:");
    println!("  {:>8} | {:>12} | {:>15}", "T (C)", "E (GPa)", "α (10^-6/K)");
    println!("  {}", "-".repeat(42));

    for t in temps {
        let e = props.youngs_modulus(t);
        let alpha = props.thermal_expansion.alpha_x * 1e6;
        println!("  {:>8.1} | {:>12.2} | {:>15.2}", t, e / 1e9, alpha);
    }
    println!();

    Ok(())
}

/// Example 3: Coupled thermal-stress analysis.
fn example_thermal_stress_analysis() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 3: Coupled Thermal-Stress Analysis");
    println!("{}\n", "=".repeat(60));

    // Create simple 1D bar model
    println!("Model: 1D Bar with Thermal Loading");
    println!("  Length: 1.0 m");
    println!("  Elements: 5");
    println!("  Fixed at left end");
    println!();

    // Create stiffness matrix (simplified 1D bar)
    let n_nodes = 6;
    let n_dof = n_nodes - 1; // Left end fixed
    let ea_l = 210e9 * 0.01 / 0.2; // EA/L

    let mut k = DMatrix::zeros(n_dof, n_dof);
    for i in 0..n_dof {
        if i > 0 {
            k[(i, i - 1)] = -ea_l;
            k[(i - 1, i)] = -ea_l;
        }
        k[(i, i)] = 2.0 * ea_l;
    }
    k[(0, 0)] = ea_l; // First node only connected to one element

    // Apply temperature distribution (linear gradient)
    let temps: Vec<f64> = (0..n_nodes)
        .map(|i| 20.0 + 100.0 * (i as f64 / (n_nodes - 1) as f64))
        .collect();

    println!("Temperature Distribution:");
    println!("  T_min: {:.1} C", temps[0]);
    println!("  T_max: {:.1} C", temps[n_nodes - 1]);
    println!();

    // Create thermal-stress solver
    let solver = ThermalStressSolver::new(1, 1);
    let thermal_expansion = ThermalExpansionTensor::isotropic(12e-6);
    let f = DVector::zeros(n_dof); // No mechanical load

    // Solve thermal-stress problem
    let start = std::time::Instant::now();
    let result = solver.solve(&k, &f, &temps, &thermal_expansion, 20.0);
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  Computation time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Max displacement: {:.6e} m", result.max_displacement);

    // Show displacements at selected nodes
    println!("\n  Displacements at selected nodes:");
    let node_indices = vec![0, 2, 4, 5];
    for &node in &node_indices {
        if node < n_nodes {
            let dof = if node == 0 { 0 } else { node - 1 };
            if dof < result.displacements.len() {
                println!("    Node {}: {:.6e} m (T={:.1} C)",
                         node, result.displacements[dof], temps[node]);
            }
        }
    }
    println!();

    Ok(())
}

/// Example 4: Thermal contact effects.
fn example_thermal_contact() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 4: Thermal Effects on Contact");
    println!("{}\n", "=".repeat(60));

    // Create thermal contact modifier
    let modifier = ThermalContactModifier::new(20.0, 1e-5);

    println!("Thermal Contact Modifier:");
    println!("  Reference temperature: {:.1} C", modifier.reference_temp);
    println!("  Gap thermal coefficient: {:.2e} m/K", modifier.gap_thermal_coeff);
    println!();

    // Gap evolution with temperature
    let original_gap = -0.001; // Initial penetration (closed contact)
    let temps = vec![20.0, 50.0, 100.0, 150.0, 200.0];

    println!("Contact Status vs Temperature:");
    println!("  {:>8} | {:>12} | {:>10}", "T (C)", "Gap (mm)", "Status");
    println!("  {}", "-".repeat(35));

    for t in temps {
        let gap = modifier.modify_gap(original_gap, t);
        let state = modifier.update_contact_state(
            ContactState::Sticking,
            original_gap,
            t,
        );
        let status = match state {
            ContactState::Open => "Open",
            ContactState::Sticking => "Sticking",
            ContactState::Sliding => "Sliding",
        };
        println!("  {:>8.1} | {:>12.6} | {:>10}", t, gap * 1000.0, status);
    }
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_stress_examples() {
        example_thermal_expansion().unwrap();
        example_temperature_dependent().unwrap();
        example_thermal_contact().unwrap();
    }
}
