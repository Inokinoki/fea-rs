//! 3D truss tower example.
//!
//! This example demonstrates:
//! - 3D truss modeling
//! - Multiple load cases
//! - Material property usage
//! - Result export to VTK and JSON

use fea::prelude::*;
use fea::core::Node;
use fea::materials::STEEL_A36;

fn main() -> anyhow::Result<()> {
    println!("3D Truss Tower Analysis");
    println!("=======================\n");

    // Material properties
    let e = STEEL_A36.young_modulus;
    let _rho = STEEL_A36.density;
    println!("Material: {}", STEEL_A36.name);
    println!("Young's modulus: {:.1} GPa", e / 1e9);
    println!("Density: {:.0} kg/m^3\n", _rho);

    // Tower geometry
    let base_width = 2.0; // meters
    let height = 6.0; // meters
    let num_levels = 3;

    // Cross-sectional properties
    let area = 5e-4; // m^2 (500 mm^2)

    // Create model
    let mut model = Model::<Truss2>::new();

    // Create nodes at each level
    let mut level_nodes = Vec::new();
    for level in 0..=num_levels {
        let z = level as f64 * height / num_levels as f64;
        let scale = 1.0 - 0.3 * (level as f64 / num_levels as f64); // Tapering
        let w = base_width * scale;

        let nodes = [
            model.add_node(Node::new_3d(w / 2.0, w / 2.0, z)),   // Node 0: front-right
            model.add_node(Node::new_3d(-w / 2.0, w / 2.0, z)),  // Node 1: front-left
            model.add_node(Node::new_3d(-w / 2.0, -w / 2.0, z)), // Node 2: back-left
            model.add_node(Node::new_3d(w / 2.0, -w / 2.0, z)),  // Node 3: back-right
        ];
        level_nodes.push(nodes);
    }

    // Add elements
    for level in 0..num_levels {
        let bottom = level_nodes[level];
        let top = level_nodes[level + 1];

        // Vertical/horizontal members at bottom of this level
        for i in 0..4 {
            // Horizontal ring at bottom
            let j = (i + 1) % 4;
            model.add_element(Truss2::new(bottom[i], bottom[j], e, area));

            // Vertical member
            model.add_element(Truss2::new(bottom[i], top[i], e, area));
        }

        // Diagonal bracing
        for i in 0..4 {
            let j = (i + 1) % 4;
            model.add_element(Truss2::new(bottom[i], top[j], e, area));
        }
    }

    // Top horizontal ring
    let top = level_nodes[num_levels];
    for i in 0..4 {
        let j = (i + 1) % 4;
        model.add_element(Truss2::new(top[i], top[j], e, area));
    }

    println!("Nodes: {}", model.nodes.len());
    println!("Elements: {}", model.elements.len());

    // Boundary conditions: fix base (level 0)
    let base = level_nodes[0];
    for &node in &base {
        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition { node, dof, value: 0.0 });
        }
    }

    // Load case 1: Vertical load at top center
    println!("\nLoad Case 1: Vertical load at top");
    let top_center_load = -5000.0; // 5 kN downward
    for &node in &top {
        model.add_load(Load {
            node,
            dof: Dof::Uz,
            value: top_center_load / 4.0, // Distribute among 4 nodes
        });
    }

    // Solve
    model.build_dofs_3d();
    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    // Report results
    let top_displacement = result.u[model.dof_index(top[0], Dof::Uz).unwrap()];
    println!("Top displacement (Z): {:.4} mm", top_displacement * 1000.0);

    // Find max stress
    let mut max_stress = 0.0;
    let mut max_stress_elem = 0;
    for (i, elem) in model.elements.iter().enumerate() {
        let stress = elem.axial_stress(&model, &result.u);
        if stress.abs() > max_stress {
            max_stress = stress.abs();
            max_stress_elem = i;
        }
    }
    println!("Max axial stress: {:.2} MPa", max_stress / 1e6);
    println!("Max stress element: {}", max_stress_elem);

    // Check yield
    let utilization = max_stress / STEEL_A36.yield_strength;
    println!("Yield utilization: {:.1}%", utilization * 100.0);
    if utilization > 1.0 {
        println!("WARNING: Material yield exceeded!");
    } else if utilization > 0.6 {
        println!("WARNING: High utilization (>60%)");
    }

    // Export results
    std::fs::create_dir_all("out")?;
    let config = VizConfig::with_auto_scale(&model, &result.u, 0.2);
    VtkLegacyWriter::new().write_truss2_with_states("out/tower.vtk", &model, &result.u, &config)?;
    JsonWriter::new().write_truss2_enhanced("out/tower.json", &model, &result.u, &config)?;

    println!("\nResults exported to:");
    println!("  - out/tower.vtk (ParaView)");
    println!("  - out/tower.json (web viewer)");

    Ok(())
}
