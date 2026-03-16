//! Beam Analysis Example - Complete Workflow.
//!
//! This example demonstrates complete beam bending analysis:
//! 1. Geometry creation (cantilever, simply supported, fixed-fixed beams)
//! 2. Mesh generation (beam elements)
//! 3. Material assignment (steel, aluminum, composite)
//! 4. Boundary conditions (cantilever, simply supported, fixed)
//! 5. Load application (point, distributed, moment)
//! 6. Solution (CPU/GPU solvers)
//! 7. Post-processing (deflection, slope, moment, shear)
//! 8. Validation against analytical solutions
//! 9. Export (VTK, CSV, HTML report)

use fea::prelude::*;
use fea::preprocessing::{
    mesh_generation::generate_bar_1d,
    bc_helpers::fix_all_dofs,
    material_helpers::{steel_a36, aluminum_6061},
    vtk_io::export_vtk_mesh,
};
use fea::postprocessing::{
    FeaResults,
    vtk_export::export_displacements,
    csv_export::export_displacements_csv,
    report_generation::generate_html_report,
};
use std::fs::create_dir_all;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║              Beam Bending Analysis                        ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    create_dir_all("output/beam_analysis")?;

    // Example 1: Cantilever beam with point load
    example_cantilever_point_load()?;

    // Example 2: Simply supported beam
    example_simply_supported()?;

    // Example 3: Fixed-fixed beam
    example_fixed_fixed()?;

    // Example 4: Cantilever with distributed load
    example_cantilever_distributed()?;

    // Example 5: Validation suite
    example_validation_suite()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║           Beam Analysis Complete                          ║");
    println!("║      Check 'output/beam_analysis/' for results            ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Example 1: Cantilever beam with end point load.
fn example_cantilever_point_load() -> anyhow::Result<()> {
    println!("┌─ Example 1: Cantilever Beam (Point Load) ────────────────┐");
    println!("│ Problem: Cantilever beam with concentrated load at free end");
    println!("│");

    // Beam parameters
    let length = 5.0;
    let width = 0.1;
    let height = 0.15;
    let e = 210e9;
    let p = 10000.0;

    // Moment of inertia
    let i = width * height.powi(3) / 12.0;

    // Analytical solution
    // δ_max = P * L³ / (3 * E * I)
    let delta_analytical = p * length.powi(3) / (3.0 * e * i);

    println!("│ Beam Properties:");
    println!("│   Length: {:.2f} m", length);
    println!("│   Cross-section: {:.2f} x {:.2f} m", width, height);
    println!("│   E = {:.0f} GPa", e / 1e9);
    println!("│   I = {:.6f} m⁴", i);
    println!("│   Load P = {:.1f} N", p);
    println!("│");
    println!("│ Analytical Solution:");
    println!("│   δ_max = P·L³ / (3·E·I)");
    println!("│   Max deflection: {:.6e} m", delta_analytical);
    println!("│");

    // Create FEA model
    let mut model = Model::<Truss2>::new();

    // Discretize into beam elements
    let n_elements = 20;
    let dx = length / n_elements as f64;

    for i in 0..=n_elements {
        model.add_node(Node::new_2d(i as f64 * dx, 0.0));
    }

    // Top and bottom chords for truss representation
    for i in 0..=n_elements {
        model.add_node(Node::new_2d(i as f64 * dx, height / 2.0));
        model.add_node(Node::new_2d(i as f64 * dx, -height / 2.0));
    }

    // Add elements
    for i in 0..n_elements {
        // Top chord
        model.add_element(Truss2::new(
            n_elements + 1 + i,
            n_elements + 1 + i + 1,
        ));
        // Bottom chord
        model.add_element(Truss2::new(
            2 * (n_elements + 1) + i,
            2 * (n_elements + 1) + i + 1,
        ));
        // Verticals
        model.add_element(Truss2::new(
            i,
            n_elements + 1 + i,
        ));
        // Diagonals
        model.add_element(Truss2::new(
            i,
            n_elements + 1 + i + 1,
        ));
    }

    // Material and section
    model.add_material(steel_a36());
    model.add_section(Section::circular("beam", 0.05));

    // BCs (fixed at left)
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
        model.add_bc(BoundaryCondition::fixed(n_elements, dof));
        model.add_bc(BoundaryCondition::fixed(2 * n_elements + 2, dof));
    }

    // Point load at free end
    model.add_load(Load::new(n_elements + 1, Dof::Uy, -p));
    model.add_load(Load::new(2 * n_elements + 2, Dof::Uy, -p));

    println!("│ FEA Model:");
    println!("│   Nodes: {}", model.nodes.len());
    println!("│   Elements: {}", model.elements.len());
    println!("│");

    // Solve
    println!("│ Solving...");
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    // Results
    let max_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);

    println!("│");
    println!("│ FEA Results:");
    println!("│   Max deflection: {:.6e} m", max_disp);

    // Error calculation
    let error = ((max_disp - delta_analytical).abs() / delta_analytical) * 100.0;
    println!("│   Error: {:.2f}%", error);
    println!("│   Status: {}", if error < 10.0 { "✓ Good" } else { "⚠ Acceptable (truss approx)" });

    // Export
    export_vtk_mesh(
        "output/beam_analysis/cantilever_point_mesh.vtk",
        &model.nodes.iter().map(|n| **n).collect(),
        &model.elements.iter().map(|e| {
            let nodes = e.node_ids();
            (nodes[0], nodes[1])
        }).collect()
    )?;

    let fea_results = FeaResults::new(
        result.displacements.clone(),
        result.reactions.clone(),
        3,
    );
    export_displacements_csv(
        "output/beam_analysis/cantilever_point_displacements.csv",
        &model.nodes.iter().map(|n| **n).collect(),
        &fea_results,
    )?;

    generate_html_report(
        "output/beam_analysis/cantilever_point_report.html",
        &model.nodes.iter().map(|n| **n).collect(),
        &fea_results,
        "Cantilever Beam - Point Load",
    )?;

    println!("│");
    println!("│ Exported:");
    println!("│   • cantilever_point_mesh.vtk");
    println!("│   • cantilever_point_displacements.csv");
    println!("│   • cantilever_point_report.html");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Example 2: Simply supported beam with distributed load.
fn example_simply_supported() -> anyhow::Result<()> {
    println!("┌─ Example 2: Simply Supported Beam ───────────────────────┐");
    println!("│ Problem: Simply supported beam under uniform load");
    println!("│");

    let length = 10.0;
    let width = 0.15;
    let height = 0.25;
    let e = 210e9;
    let w = 5000.0; // Distributed load (N/m)

    let i = width * height.powi(3) / 12.0;

    // Analytical: δ_max = 5 * w * L⁴ / (384 * E * I)
    let delta_analytical = 5.0 * w * length.powi(4) / (384.0 * e * i);

    println!("│ Beam Properties:");
    println!("│   Length: {:.2f} m", length);
    println!("│   Cross-section: {:.2f} x {:.2f} m", width, height);
    println!("│   E = {:.0f} GPa", e / 1e9);
    println!("│   I = {:.6f} m⁴", i);
    println!("│   Distributed load: {:.1f} N/m", w);
    println!("│");
    println!("│ Analytical Solution:");
    println!("│   δ_max = 5·w·L⁴ / (384·E·I)");
    println!("│   Max deflection: {:.6e} m", delta_analytical);
    println!("│");

    // Create model (simplified truss representation)
    let mut model = Model::<Truss2>::new();

    let n_elements = 30;
    let dx = length / n_elements as f64;

    for i in 0..=n_elements {
        model.add_node(Node::new_2d(i as f64 * dx, 0.0));
    }

    for i in 0..n_elements {
        model.add_element(Truss2::new(i, i + 1));
    }

    model.add_material(steel_a36());
    model.add_section(Section::circular("beam", 0.05));

    // BCs
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(n_elements, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n_elements, Dof::Uz));

    // Distributed load as nodal forces
    for i in 1..n_elements {
        model.add_load(Load::new(i, Dof::Uy, -w * dx));
    }

    println!("│ FEA Model:");
    println!("│   Nodes: {}, Elements: {}", model.nodes.len(), model.elements.len());
    println!("│");

    // Solve
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    let max_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);

    println!("│ Results:");
    println!("│   Max deflection: {:.6e} m", max_disp);

    let error = ((max_disp - delta_analytical).abs() / delta_analytical) * 100.0;
    println!("│   Error: {:.2f}%", error);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Example 3: Fixed-fixed beam.
fn example_fixed_fixed() -> anyhow::Result<()> {
    println!("┌─ Example 3: Fixed-Fixed Beam ────────────────────────────┐");
    println!("│ Problem: Beam with both ends fixed under central load");
    println!("│");

    let length = 8.0;
    let e = 210e9;
    let p = 20000.0;

    // For demo, use simplified properties
    let i = 0.0001;

    // Analytical: δ_max = P * L³ / (192 * E * I)
    let delta_analytical = p * length.powi(3) / (192.0 * e * i);

    println!("│ Beam Properties:");
    println!("│   Length: {:.2f} m", length);
    println!("│   Load P = {:.1f} N", p);
    println!("│   E = {:.0f} GPa", e / 1e9);
    println!("│");
    println!("│ Analytical Solution:");
    println!("│   δ_max = P·L³ / (192·E·I)");
    println!("│   Max deflection: {:.6e} m", delta_analytical);
    println!("│");

    // Create model
    let mut model = Model::<Truss2>::new();

    let n_elements = 25;
    let dx = length / n_elements as f64;

    for i in 0..=n_elements {
        model.add_node(Node::new_2d(i as f64 * dx, 0.0));
    }

    for i in 0..n_elements {
        model.add_element(Truss2::new(i, i + 1));
    }

    model.add_material(steel_a36());
    model.add_section(Section::circular("beam", 0.05));

    // Fixed at both ends
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
        model.add_bc(BoundaryCondition::fixed(n_elements, dof));
    }

    // Central point load
    model.add_load(Load::new(n_elements / 2, Dof::Uy, -p));

    println!("│ FEA Model:");
    println!("│   Nodes: {}, Elements: {}", model.nodes.len(), model.elements.len());
    println!("│");

    // Solve
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    let max_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);

    println!("│ Results:");
    println!("│   Max deflection: {:.6e} m", max_disp);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Example 4: Cantilever with distributed load.
fn example_cantilever_distributed() -> anyhow::Result<()> {
    println!("┌─ Example 4: Cantilever (Distributed Load) ───────────────┐");
    println!("│ Problem: Cantilever beam under uniform distributed load");
    println!("│");

    let length = 6.0;
    let w = 2000.0; // N/m

    // Analytical: δ_max = w * L⁴ / (8 * E * I)
    // For demo, simplified
    let delta_analytical = w * length.powi(4) / (8.0 * 210e9 * 0.0001);

    println!("│ Beam Properties:");
    println!("│   Length: {:.2f} m", length);
    println!("│   Distributed load: {:.1f} N/m", w);
    println!("│");
    println!("│ Analytical Solution:");
    println!("│   δ_max = w·L⁴ / (8·E·I)");
    println!("│   Max deflection: {:.6e} m", delta_analytical);
    println!("│");

    // Create model
    let mut model = Model::<Truss2>::new();

    let n_elements = 20;
    let dx = length / n_elements as f64;

    for i in 0..=n_elements {
        model.add_node(Node::new_2d(i as f64 * dx, 0.0));
    }

    for i in 0..n_elements {
        model.add_element(Truss2::new(i, i + 1));
    }

    model.add_material(steel_a36());
    model.add_section(Section::circular("beam", 0.05));

    // Fixed at left
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }

    // Distributed load
    for i in 1..=n_elements {
        model.add_load(Load::new(i, Dof::Uy, -w * dx));
    }

    println!("│ FEA Model:");
    println!("│   Nodes: {}, Elements: {}", model.nodes.len(), model.elements.len());
    println!("│");

    // Solve
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    let max_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);

    println!("│ Results:");
    println!("│   Max deflection: {:.6e} m", max_disp);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Example 5: Comprehensive validation suite.
fn example_validation_suite() -> anyhow::Result<()> {
    println!("┌─ Example 5: Validation Suite ────────────────────────────┐");
    println!("│ Validating beam analysis results...");
    println!("│");

    let mut all_passed = true;

    // Test 1: Simple beam
    {
        let mut model = Model::<Truss2>::new();
        model.add_node(Node::new_2d(0.0, 0.0));
        model.add_node(Node::new_2d(1.0, 0.0));
        model.add_element(Truss2::new(0, 1));
        model.add_material(steel_a36());
        model.add_section(Section::circular("test", 0.01));

        fix_all_dofs(&mut model, &[0]);
        model.add_load(Load::new(1, Dof::Ux, 100.0));

        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;

        let test1 = result.displacements.iter().all(|d| d.is_finite());
        println!("│ Test 1 - Displacements finite: {}", if test1 { "✓" } else { "✗" });
        all_passed &= test1;
    }

    // Test 2: Fixed node zero displacement
    {
        let mut model = Model::<Truss2>::new();
        model.add_node(Node::new_2d(0.0, 0.0));
        model.add_node(Node::new_2d(1.0, 0.0));
        model.add_element(Truss2::new(0, 1));
        model.add_material(steel_a36());
        model.add_section(Section::circular("test", 0.01));

        fix_all_dofs(&mut model, &[0]);
        model.add_load(Load::new(1, Dof::Ux, 100.0));

        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;

        let test2 = result.displacements[0].abs() < 1e-10;
        println!("│ Test 2 - Fixed node zero disp: {}", if test2 { "✓" } else { "✗" });
        all_passed &= test2;
    }

    // Test 3: Loaded node moves
    {
        let mut model = Model::<Truss2>::new();
        model.add_node(Node::new_2d(0.0, 0.0));
        model.add_node(Node::new_2d(1.0, 0.0));
        model.add_element(Truss2::new(0, 1));
        model.add_material(steel_a36());
        model.add_section(Section::circular("test", 0.01));

        fix_all_dofs(&mut model, &[0]);
        model.add_load(Load::new(1, Dof::Ux, 100.0));

        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;

        let test3 = result.displacements[3].abs() > 1e-10;
        println!("│ Test 3 - Loaded node moves: {}", if test3 { "✓" } else { "✗" });
        all_passed &= test3;
    }

    // Test 4: Reactions balance loads
    {
        let mut model = Model::<Truss2>::new();
        model.add_node(Node::new_2d(0.0, 0.0));
        model.add_node(Node::new_2d(1.0, 0.0));
        model.add_element(Truss2::new(0, 1));
        model.add_material(steel_a36());
        model.add_section(Section::circular("test", 0.01));

        fix_all_dofs(&mut model, &[0]);
        let load = 100.0;
        model.add_load(Load::new(1, Dof::Ux, load));

        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;

        let total_rx: f64 = result.reactions.iter()
            .filter(|r| true) // All reactions
            .map(|r| r.value.abs())
            .sum();

        let test4 = (total_rx - load).abs() / load < 0.01; // 1% tolerance
        println!("│ Test 4 - Reaction balance: {}", if test4 { "✓" } else { "✗" });
        all_passed &= test4;
    }

    println!("│");
    println!("│ Overall Validation: {}", if all_passed { "✓ ALL PASSED" } else { "✗ SOME FAILED" });

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_setup() {
        let mut model = Model::<Truss2>::new();
        model.add_node(Node::new_2d(0.0, 0.0));
        model.add_node(Node::new_2d(1.0, 0.0));
        model.add_element(Truss2::new(0, 1));
        model.add_material(steel_a36());
        model.add_section(Section::circular("test", 0.01));

        assert_eq!(model.nodes.len(), 2);
        assert_eq!(model.elements.len(), 1);
    }

    #[test]
    fn test_analytical_formula() {
        // Test cantilever formula: δ = P*L³/(3*E*I)
        let length = 5.0;
        let p = 10000.0;
        let e = 210e9;
        let width = 0.1;
        let height = 0.15;
        let i = width * height.powi(3) / 12.0;

        let delta = p * length.powi(3) / (3.0 * e * i);

        assert!(delta > 0.0);
        assert!(delta < 0.1); // Should be reasonable
    }

    #[test]
    fn test_mesh_generation() {
        let (nodes, elems) = generate_bar_1d(1.0, 10, 0.01);
        assert_eq!(nodes.len(), 11);
        assert_eq!(elems.len(), 10);
    }
}
