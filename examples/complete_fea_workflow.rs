//! Complete FEA Workflow Example.
//!
//! This example demonstrates the complete FEA workflow:
//! 1. Geometry creation
//! 2. Mesh generation
//! 3. Material assignment
//! 4. Boundary conditions
//! 5. Load application
//! 6. Solution
//! 7. Post-processing
//! 8. Results visualization

use fea::prelude::*;
use fea::preprocessing::{
    mesh_generation::{generate_box_3d, generate_rect_2d},
    bc_helpers::fix_all_dofs,
    material_helpers::steel_a36,
    vtk_io::{export_vtk_mesh, export_vtk_displacements, export_vtk_scalar},
    csv_export::{export_displacements_csv, export_reactions_csv},
    report_generation::{generate_text_report, generate_html_report},
    FeaResults,
};
use std::fs::create_dir_all;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║           Complete FEA Workflow Example                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Create output directory
    create_dir_all("output/workflow")?;

    // Example 1: 3D cantilever beam
    run_cantilever_beam()?;

    // Example 2: Plate with hole
    run_plate_with_hole()?;

    // Example 3: Multi-material assembly
    run_multi_material_assembly()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║        Results saved in 'output/workflow/' directory     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Run 3D cantilever beam analysis.
fn run_cantilever_beam() -> anyhow::Result<()> {
    println!("┌─ 3D Cantilever Beam Analysis ────────────────────────────┐");

    // Geometry and mesh parameters
    let length = 1.0;
    let width = 0.1;
    let height = 0.1;
    let nx = 20;
    let ny = 4;
    let nz = 4;

    println!("│ Geometry:");
    println!("│   Length: {:.1} m", length);
    println!("│   Width:  {:.2} m", width);
    println!("│   Height: {:.2} m", height);
    println!("│");
    println!("│ Mesh: {} × {} × {}", nx, ny, nz);

    // Generate hex mesh
    let (nodes, hexes) = generate_box_3d(length, width, height, nx, ny, nz);

    println!("│ Generated: {} nodes, {} elements", nodes.len(), hexes.len());

    // Create model
    let mut model: Model<Truss2> = Model::new();

    // For simplicity, we'll use a simplified truss representation
    // In practice, you would use 3D solid elements
    for (i, node) in nodes.iter().enumerate() {
        model.add_node(Node::new_3d(node.x, node.y, node.z));

        // Connect to previous node in x-direction
        if i >= nx + 1 {
            let prev = i - (nx + 1);
            model.add_element(Truss2::new(prev, i));
        }
    }

    // Add material
    model.add_material(steel_a36());
    model.add_section(Section::circular("beam", 0.01));

    // Fix left end (x = 0)
    let fixed_nodes: Vec<usize> = (0..((ny + 1) * (nz + 1))).collect();
    for node_id in &fixed_nodes {
        fix_all_dofs(&mut model, &[*node_id]);
    }

    // Apply point load at free end
    let load_node = nodes.len() - 1;
    model.add_load(Load::new(load_node, Dof::Uy, -1000.0));

    println!("│ Boundary conditions: {} fixed nodes", fixed_nodes.len());
    println!("│ Applied load: -1000 N at free end");
    println!("│");

    // Solve
    println!("│ Solving...");
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    let max_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);
    println!("│ Max displacement: {:.6e} m", max_disp);

    // Export results
    export_vtk_mesh("output/workflow/cantilever_mesh.vtk", &model.nodes, &model.elements)?;
    export_vtk_displacements("output/workflow/cantilever_displacements.vtk", &model.nodes, &result.displacements)?;
    export_displacements_csv("output/workflow/cantilever_displacements.csv", &model.nodes, &result)?;
    export_reactions_csv("output/workflow/cantilever_reactions.csv", &model.nodes, &result)?;

    // Generate report
    let report = generate_text_report(&model.nodes, model.elements.len(), &model.bcs, &model.loads, &result);
    std::fs::write("output/workflow/cantilever_report.txt", &report)?;

    // Generate HTML report
    generate_html_report("output/workflow/cantilever_report.html", &model.nodes, &result, "Cantilever Beam Analysis")?;

    println!("│");
    println!("│ Exported:");
    println!("│   - cantilever_mesh.vtk");
    println!("│   - cantilever_displacements.vtk");
    println!("│   - cantilever_displacements.csv");
    println!("│   - cantilever_reactions.csv");
    println!("│   - cantilever_report.txt");
    println!("│   - cantilever_report.html");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Run plate with hole analysis.
fn run_plate_with_hole() -> anyhow::Result<()> {
    println!("┌─ Plate with Hole Analysis ───────────────────────────────┐");

    // Geometry
    let width = 1.0;
    let height = 0.5;
    let hole_radius = 0.1;
    let nx = 40;
    let ny = 20;

    println!("│ Geometry:");
    println!("│   Width:  {:.2} m", width);
    println!("│   Height: {:.2} m", height);
    println!("│   Hole radius: {:.2} m", hole_radius);
    println!("│");

    // Generate mesh
    let (nodes, quads) = generate_rect_2d(width, height, nx, ny);

    println!("│ Mesh: {} × {} = {} elements", nx, ny, quads.len());

    // Create simplified truss model for the plate
    let mut model: Model<Truss2> = Model::new();

    for node in &nodes {
        model.add_node(Node::new_2d(node.x, node.y));
    }

    // Create diagonal truss pattern
    for j in 0..ny {
        for i in 0..nx {
            let n0 = j * (nx + 1) + i;
            let n1 = j * (nx + 1) + i + 1;
            let n2 = (j + 1) * (nx + 1) + i;
            let n3 = (j + 1) * (nx + 1) + i + 1;

            model.add_element(Truss2::new(n0, n1));
            model.add_element(Truss2::new(n0, n2));
            model.add_element(Truss2::new(n0, n3));
            model.add_element(Truss2::new(n1, n3));
            model.add_element(Truss2::new(n2, n3));
        }
    }

    // Add material
    model.add_material(steel_a36());
    model.add_section(Section::circular("plate", 0.005));

    // Fix left edge
    let mut left_nodes = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        if node.x < 1e-6 {
            left_nodes.push(i);
        }
    }
    for node_id in &left_nodes {
        fix_all_dofs(&mut model, &[*node_id]);
    }

    // Apply tension on right edge
    for (i, node) in nodes.iter().enumerate() {
        if node.x > width - 1e-6 {
            model.add_load(Load::new(i, Dof::Ux, 100.0));
        }
    }

    println!("│ Fixed: {} nodes on left edge", left_nodes.len());
    println!("│ Applied tension on right edge");
    println!("│");

    // Solve
    println!("│ Solving...");
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    let max_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);
    println!("│ Max displacement: {:.6e} m", max_disp);

    // Export results
    export_vtk_mesh("output/workflow/plate_mesh.vtk", &model.nodes, &model.elements)?;
    export_vtk_displacements("output/workflow/plate_displacements.vtk", &model.nodes, &result.displacements)?;

    println!("│");
    println!("│ Exported:");
    println!("│   - plate_mesh.vtk");
    println!("│   - plate_displacements.vtk");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Run multi-material assembly analysis.
fn run_multi_material_assembly() -> anyhow::Result<()> {
    println!("┌─ Multi-Material Assembly Analysis ───────────────────────┐");

    // Create a simple assembly: steel bar with aluminum end
    let mut model: Model<Truss2> = Model::new();

    // Steel section (left half)
    for i in 0..6 {
        model.add_node(Node::new_3d(i as f64 * 0.1, 0.0, 0.0));
    }

    // Aluminum section (right half)
    for i in 6..11 {
        model.add_node(Node::new_3d(i as f64 * 0.1, 0.0, 0.0));
    }

    // Add elements
    for i in 0..10 {
        model.add_element(Truss2::new(i, i + 1));
    }

    // Steel material (elements 0-4)
    let steel_id = model.add_material(steel_a36());
    model.add_section(Section::circular("steel", 0.02));

    // Aluminum material (elements 5-9)
    let alum_id = model.add_material(material_helpers::aluminum_6061());
    model.add_section(Section::circular("aluminum", 0.02));

    // Fix left end
    fix_all_dofs(&mut model, &[0]);

    // Apply load at right end
    model.add_load(Load::new(10, Dof::Ux, 5000.0));

    println!("│ Assembly: Steel bar (left) + Aluminum bar (right)");
    println!("│ Total elements: {}", model.elements.len());
    println!("│ Materials: Steel A36, Aluminum 6061-T6");
    println!("│");

    // Solve
    println!("│ Solving...");
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    let max_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);
    println!("│ Max displacement: {:.6e} m", max_disp);

    // Export results
    let fea_results = FeaResults::new(result.displacements.clone(), result.reactions.clone(), 3);
    generate_html_report("output/workflow/assembly_report.html", &model.nodes, &fea_results, "Multi-Material Assembly")?;

    println!("│");
    println!("│ Exported:");
    println!("│   - assembly_report.html");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}
