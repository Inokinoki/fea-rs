//! Modal analysis example: Multi-element truss with mode shape visualization.
//!
//! This example demonstrates:
//! - Modal analysis for natural frequencies and mode shapes
//! - JSON export with mode shapes for web viewer animation
//! - Mass matrix formulation comparison

use fea::prelude::*;
use fea::core::Node;
use fea::modal::{ModalConfig, ModalSolver, MassFormulation};

fn main() -> anyhow::Result<()> {
    println!("Modal Analysis Example");
    println!("======================\n");

    // Material and geometry
    let e = 210e9; // Steel Young's modulus (Pa)
    let a = 1e-4;  // Cross-sectional area (m²)
    let l = 0.5;   // Element length (m)

    // Create a 4-element cantilever truss
    let mut model = Model::<Truss2>::new();

    // Create 5 nodes in a line
    let nodes: Vec<_> = (0..=4)
        .map(|i| model.add_node(Node::new_2d(i as f64 * l, 0.0)))
        .collect();

    // Add 4 elements
    for i in 0..4 {
        model.add_element(Truss2::new(nodes[i], nodes[i + 1], e, a));
    }

    // Boundary conditions: fixed at node 0
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: nodes[0],
            dof,
            value: 0.0,
        });
    }
    // Constrain Y and Z for all other nodes (only X motion free)
    for &node in &nodes[1..] {
        for dof in [Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition { node, dof, value: 0.0 });
        }
    }

    println!("Nodes: {}", model.nodes.len());
    println!("Elements: {}", model.elements.len());
    println!("Free DOFs: {}\n", model.nodes.len() * 3 - model.bcs.len());

    // Run modal analysis with lumped mass
    let config = ModalConfig {
        num_modes: 3,
        mass_formulation: MassFormulation::Lumped,
        max_iterations: 1000,
        tolerance: 1e-10,
    };

    let solver = ModalSolver::with_config(config.clone());
    let result = solver.analyze_truss2(&mut model)?;

    println!("Natural Frequencies (Lumped Mass):");
    println!("----------------------------------");
    for (i, &freq) in result.frequencies.iter().enumerate() {
        let freq_hz = freq / (2.0 * std::f64::consts::PI);
        println!("  Mode {}: {:8.2} rad/s = {:8.2} Hz", i + 1, freq, freq_hz);
    }

    // Also run with consistent mass for comparison
    let config_consistent = ModalConfig {
        mass_formulation: MassFormulation::Consistent,
        ..config
    };
    let solver_consistent = ModalSolver::with_config(config_consistent);
    let result_consistent = solver_consistent.analyze_truss2(&mut model)?;

    println!("\nNatural Frequencies (Consistent Mass):");
    println!("--------------------------------------");
    for (i, &freq) in result_consistent.frequencies.iter().enumerate() {
        let freq_hz = freq / (2.0 * std::f64::consts::PI);
        let lumped_hz = result.frequencies[i] / (2.0 * std::f64::consts::PI);
        let diff = (freq_hz - lumped_hz).abs() / lumped_hz * 100.0;
        println!("  Mode {}: {:8.2} Hz (diff from lumped: {:.1}%)", i + 1, freq_hz, diff);
    }

    // Export for web visualization (use lumped mass results)
    std::fs::create_dir_all("web")?;
    result.write_json("web/modes.json")?;

    // Also run static analysis for comparison
    model.add_load(Load {
        node: nodes[4],
        dof: Dof::Ux,
        value: 1000.0,
    });

    let static_result = LinearStaticSolver::new().solve_truss2(&mut model)?;

    // Export static results with modal data
    JsonWriter::new().write_truss2("web/model.json", &model, &static_result.u)?;

    println!("\nResults exported:");
    println!("  - web/model.json (static deflection)");
    println!("  - web/modes.json (mode shapes for animation)");
    println!("\nTo view: open http://localhost:8000 and check mode shapes");

    Ok(())
}
