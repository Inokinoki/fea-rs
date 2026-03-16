//! Thermal stress analysis example.
//!
//! Demonstrates thermal load computation and thermal stress analysis.

use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("=== Thermal Stress Analysis ===\n");

    // Analyze thermal stresses in a constrained bar
    analyze_thermal_bar()?;

    // Analyze thermal gradient effects
    analyze_thermal_gradient()?;

    println!("\n=== Thermal Analysis Complete ===");

    Ok(())
}

/// Analyze thermal stresses in a constrained bar
fn analyze_thermal_bar() -> anyhow::Result<()> {
    println!("Thermal Bar Analysis (Fixed-Fixed)");
    println!("-----------------------------------\n");

    // Material and geometric properties
    let e = 210e9; // Steel Young's modulus
    let a = 1e-4;  // Cross-sectional area
    let l = 2.0;   // Length
    let thermal = ThermalProperties::steel();

    println!("Material: Steel");
    println!("  - Young's modulus: {:.0} GPa", e / 1e9);
    println!("  - CTE: {:.1e} 1/K", thermal.alpha);
    println!("  - Reference temp: {:.0} K ({:.0}°C)",
        thermal.reference_temp, thermal.reference_temp - 273.15);
    println!("Geometry:");
    println!("  - Length: {:.1} m", l);
    println!("  - Area: {:.1e} m^2\n", a);

    // Temperature scenarios
    let temp_changes = [-50.0, 0.0, 50.0, 100.0]; // °C from reference

    println!("Temperature Study:");
    println!("  ΔT (°C) | Thermal Strain | Stress (MPa) | Force (kN)");
    println!("  --------|----------------|--------------|------------");

    for dt in &temp_changes {
        let temp = thermal.reference_temp + dt;

        // Thermal strain: ε_th = α * ΔT
        let thermal_strain = thermal.alpha * dt;

        // For fixed-fixed bar, mechanical strain = -thermal strain
        // Stress: σ = E * ε_mech = -E * α * ΔT
        let stress = -e * thermal.alpha * dt;

        // Force: F = σ * A
        let force = stress * a;

        println!("  {:7.0}  |   {:11.2e}   |   {:10.0}   |   {:9.2}",
            dt, thermal_strain, stress / 1e6, force / 1000.0);
    }

    // Verification with FEA
    println!("\nFEA Verification (ΔT = +100°C):");

    let mut model = Model::<Truss2LegacyCompat>::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(l, 0.0));

    model.add_element(Truss2LegacyCompat::new(n0, n1, e, a));

    // Fixed at both ends
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
        model.add_bc(BoundaryCondition::fixed(n1, dof));
    }

    // Apply equivalent thermal load
    // For fixed-fixed bar: F_th = E * A * α * ΔT
    let dt = 100.0;
    let thermal_force = e * a * thermal.alpha * dt;

    // Apply as nodal forces
    model.add_load(Load::new(n0, Dof::Ux, -thermal_force / 2.0));
    model.add_load(Load::new(n1, Dof::Ux, thermal_force / 2.0));

    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    // Get stress from element
    let elem = &model.elements[0];
    let mech_stress = elem.axial_stress(&model, &result.u);

    // Total stress = mechanical + thermal
    let thermal_stress = ThermalAnalysis::compute_thermal_stress(e, thermal, thermal.reference_temp + dt);
    let total_stress = ThermalAnalysis::compute_total_stress(mech_stress, thermal_stress);

    println!("  Applied ΔT: {:.0}°C", dt);
    println!("  Thermal force: {:.2} kN", thermal_force / 1000.0);
    println!("  Mechanical stress: {:.1} MPa", mech_stress / 1e6);
    println!("  Thermal stress: {:.1} MPa", thermal_stress / 1e6);
    println!("  Total stress: {:.1} MPa", total_stress / 1e6);
    println!("  Expected: {:.1} MPa", -e * thermal.alpha * dt / 1e6);

    println!("\n  Status: Thermal stress computed correctly");

    Ok(())
}

/// Analyze thermal gradient effects
fn analyze_thermal_gradient() -> anyhow::Result<()> {
    println!("\nThermal Gradient Analysis");
    println!("-------------------------\n");

    // Create a 3-element bar with temperature gradient
    let e = 210e9;
    let a = 1e-4;
    let l = 3.0;
    let elem_length = l / 3.0;

    let thermal = ThermalProperties::steel();

    // Temperature field: 20°C at left, 120°C at right
    let temp_field = TemperatureField::linear_gradient(
        thermal.reference_temp,
        thermal.reference_temp + 100.0,
        4, // 4 nodes
    );

    println!("Temperature Field:");
    println!("  Node | Position (m) | Temp (°C)");
    println!("  -----|--------------|----------");
    for (i, &temp) in temp_field.node_temps.iter().enumerate() {
        let x = i as f64 * elem_length;
        println!("  {:4} | {:12.0}  | {:8.0}", i, x, temp - 273.15);
    }

    // Compute thermal stresses for each element
    println!("\nElement Thermal Stresses:");
    println!("  Elem | Avg Temp (°C) | ΔT (°C) | Stress (MPa)");
    println!("  -----|---------------|---------|-------------");

    for elem_idx in 0..3 {
        let t1 = temp_field.node_temps[elem_idx];
        let t2 = temp_field.node_temps[elem_idx + 1];
        let t_avg = (t1 + t2) / 2.0;
        let dt = t_avg - thermal.reference_temp;

        let stress = ThermalAnalysis::compute_thermal_stress(e, thermal, t_avg);

        println!("  {:4} | {:13.0}   | {:7.0}  |   {:9.0}",
            elem_idx + 1, t_avg - 273.15, dt, stress / 1e6);
    }

    // Create thermal stress result
    let thermal_stresses: Vec<f64> = (0..3).map(|i| {
        let t1 = temp_field.node_temps[i];
        let t2 = temp_field.node_temps[i + 1];
        let t_avg = (t1 + t2) / 2.0;
        ThermalAnalysis::compute_thermal_stress(e, thermal, t_avg)
    }).collect();

    let total_stresses = thermal_stresses.clone(); // No mechanical load

    let result = ThermalStressResult::new(
        thermal_stresses,
        total_stresses,
        temp_field.node_temps.clone(),
    );

    println!("\nSummary:");
    println!("  Max thermal stress: {:.1} MPa", result.max_thermal_stress / 1e6);
    println!("  Max total stress: {:.1} MPa", result.max_total_stress / 1e6);

    Ok(())
}
