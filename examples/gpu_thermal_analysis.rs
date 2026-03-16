//! GPU Thermal Analysis Example.
//!
//! This example demonstrates GPU-accelerated thermal analysis:
//! - Steady-state heat conduction
//! - Transient thermal analysis
//! - Heat flux computation

use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║          GPU Thermal Analysis Example                    ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Example 1: Steady-state thermal analysis
    example_steady_thermal()?;

    // Example 2: Transient thermal analysis
    example_transient_thermal()?;

    // Example 3: Heat flux computation
    example_heat_flux()?;

    println!("\n=== GPU Thermal Examples Complete ===");
    Ok(())
}

/// Example 1: Steady-state thermal analysis.
fn example_steady_thermal() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 1: Steady-State Thermal Analysis");
    println!("{}\n", "=".repeat(60));

    // Create thermal context (simulating GPU context)
    let ctx = GPUThermalContext::new(6, 5)
        .with_material(50.0, 500.0, 8000.0);

    println!("Thermal Context:");
    println!("  Nodes: {}", ctx.num_nodes);
    println!("  Elements: {}", ctx.num_elements);
    println!("  Conductivity: {:.1} W/(m·K)", ctx.conductivity);
    println!();

    // Define elements (1D chain)
    let elements = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];

    // Define boundary conditions
    let bc_list = vec![
        (0, ThermalBoundaryCondition::FixedTemperature(400.0)),
        (5, ThermalBoundaryCondition::FixedTemperature(300.0)),
    ];

    println!("Boundary Conditions:");
    println!("  Node 0: T = 400 K (fixed)");
    println!("  Node 5: T = 300 K (fixed)");
    println!();

    // Solve steady-state thermal problem
    let result = solve_steady_thermal_gpu(&ctx, &elements, &bc_list, 1e-10, 100);

    println!("Results:");
    println!("  Iterations: {}", result.iterations);
    println!("  Residual: {:.2e}", result.residual_norm);
    println!("  Computation time: {:.4} ms", result.computation_time_ms);
    println!();

    println!("Nodal Temperatures:");
    for i in 0..ctx.num_nodes {
        println!("  Node {}: T = {:.2} K", i, result.temperatures[i]);
    }
    println!();

    Ok(())
}

/// Example 2: Transient thermal analysis.
fn example_transient_thermal() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 2: Transient Thermal Analysis");
    println!("{}\n", "=".repeat(60));

    // Create thermal context
    let ctx = GPUThermalContext::new(6, 5)
        .with_material(50.0, 500.0, 8000.0);

    println!("Material Properties:");
    println!("  Conductivity: {:.1} W/(m·K)", ctx.conductivity);
    println!("  Specific heat: {:.1} J/(kg·K)", ctx.specific_heat);
    println!("  Density: {:.0} kg/m³", ctx.density);
    println!("  Thermal diffusivity: {:.2e} m²/s",
             ctx.conductivity / (ctx.density * ctx.specific_heat));
    println!();

    // Define elements
    let elements = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];

    // Boundary conditions
    let bc_list = vec![
        (0, ThermalBoundaryCondition::FixedTemperature(500.0)),
        (5, ThermalBoundaryCondition::Convection { h: 10.0, t_inf: 300.0 }),
    ];

    // Initial temperature (uniform)
    let t_initial = DVector::from_element(6, 300.0);

    println!("Initial/Boundary Conditions:");
    println!("  Initial temperature: 300 K (uniform)");
    println!("  Node 0: T = 500 K (fixed)");
    println!("  Node 5: Convection (h=10 W/(m²·K), T∞=300 K)");
    println!();

    // Solve transient problem
    let dt = 1.0; // 1 second time step
    let num_steps = 5;

    let results = solve_transient_thermal_gpu(
        &ctx, &elements, &bc_list, &t_initial, dt, num_steps, 1e-10, 100
    );

    println!("Transient Results (selected nodes):");
    println!("  Time (s) | Node 1 (K) | Node 3 (K)");
    println!("  {}", "-".repeat(35));

    for (step, result) in results.iter().enumerate() {
        let time = (step + 1) as f64 * dt;
        println!("  {:>8.1} | {:>10.2} | {:>10.2}",
                 time,
                 result.temperatures[1],
                 result.temperatures[3]);
    }
    println!();

    Ok(())
}

/// Example 3: Heat flux computation.
fn example_heat_flux() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 3: Heat Flux Computation");
    println!("{}\n", "=".repeat(60));

    // Simulated temperature distribution
    let temperatures = DVector::from_column_slice(&[400.0, 375.0, 350.0, 325.0, 300.0]);
    let elements = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
    let conductivity = 50.0; // W/(m·K)

    println!("Input:");
    println!("  Conductivity: {:.1} W/(m·K)", conductivity);
    println!();

    println!("Nodal Temperatures:");
    for i in 0..temperatures.len() {
        println!("  Node {}: T = {:.1} K", i, temperatures[i]);
    }
    println!();

    // Compute heat fluxes
    let fluxes = compute_heat_flux(&temperatures, &elements, conductivity);

    println!("Element Heat Fluxes:");
    println!("  Element | Heat Flux (W/m²) | Direction");
    println!("  {}", "-".repeat(40));

    for (i, &q) in fluxes.iter().enumerate() {
        let direction = if q > 0.0 { "Hot→Cold" } else { "Cold→Hot" };
        println!("  {:>7} | {:>14.2} | {}", i + 1, q.abs(), direction);
    }
    println!();

    // Verify energy balance
    let total_in = fluxes.first().copied().unwrap_or(0.0);
    let total_out = fluxes.last().copied().unwrap_or(0.0);
    let imbalance = (total_in - total_out).abs();

    println!("Energy Balance:");
    println!("  Heat entering (element 1): {:.2} W/m²", total_in.abs());
    println!("  Heat leaving (element 4): {:.2} W/m²", total_out.abs());
    println!("  Imbalance: {:.6} W/m²", imbalance);

    if imbalance < 1e-6 {
        println!("  Status: BALANCED ✓");
    } else {
        println!("  Status: Small numerical imbalance (expected)");
    }
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_thermal_examples() {
        example_steady_thermal().unwrap();
        example_transient_thermal().unwrap();
        example_heat_flux().unwrap();
    }
}
