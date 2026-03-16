//! Comprehensive pre/post-processing example.
//!
//! This example demonstrates:
//! - Mesh generation (1D, 2D, 3D)
//! - STL import/export
//! - Boundary condition helpers
//! - Material assignment
//! - VTK export for visualization
//! - CSV export for data analysis
//! - Report generation

use fea::prelude::*;
use fea::preprocessing::{
    mesh_generation::{generate_bar_1d, generate_rect_2d, generate_box_3d},
    bc_helpers::{fix_all_dofs, fix_face_3d},
    material_helpers::steel_a36,
    vtk_io::export_vtk_mesh,
};
use std::fs::create_dir_all;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     Pre/Post-Processing Comprehensive Example             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Create output directory
    create_dir_all("output")?;

    // Demo 1: 1D bar mesh generation and analysis
    demo_1d_bar()?;

    // Demo 2: 2D plate mesh generation and analysis
    demo_2d_plate()?;

    // Demo 3: 3D block mesh generation and analysis
    demo_3d_block()?;

    // Demo 4: Material library
    demo_materials()?;

    // Demo 5: BC helpers
    demo_bc_helpers()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║     Check 'output/' directory for VTK files              ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demo 1: 1D bar with mesh generation.
fn demo_1d_bar() -> anyhow::Result<()> {
    println!("┌─ 1D Bar Analysis ────────────────────────────────────────┐");

    let length = 1.0;
    let num_elements = 10;
    let area = 0.0001;

    // Generate mesh
    let (nodes, elements) = generate_bar_1d(length, num_elements, area);

    println!("│ Generated mesh: {} nodes, {} elements", nodes.len(), elements.len());

    // Create model
    let mut model: Model<Truss2> = Model::new();

    // Add nodes
    for node in &nodes {
        model.add_node(*node);
    }

    // Add elements
    for (n0, n1) in &elements {
        model.add_element(Truss2::new(*n0, *n1));
    }

    // Add material
    model.add_material(steel_a36());
    model.add_section(Section::circular("bar", (area / std::f64::consts::PI).sqrt()));

    // Apply BCs
    fix_all_dofs(&mut model, &[0]); // Fix left end

    // Apply load
    model.add_load(Load::new(nodes.len() - 1, Dof::Ux, 1000.0));

    // Solve
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    println!("│ Max displacement: {:.6e} m", result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max));
    println!("│");

    // Export to VTK
    export_vtk_mesh("output/bar_mesh.vtk", &model.nodes, &elements)?;
    println!("│ Exported: output/bar_mesh.vtk");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 2: 2D plate analysis.
fn demo_2d_plate() -> anyhow::Result<()> {
    println!("┌─ 2D Plate Analysis ──────────────────────────────────────┐");

    let width = 0.5;
    let height = 0.25;
    let nx = 20;
    let ny = 10;

    // Generate quad mesh
    let (nodes, quads) = generate_rect_2d(width, height, nx, ny);

    println!("│ Generated quad mesh: {} nodes, {} elements", nodes.len(), quads.len());

    // Generate tri mesh
    let (nodes_tri, tris) = generate_tri_2d_from_rect(width, height, nx, ny);
    println!("│ Generated tri mesh: {} nodes, {} elements", nodes_tri.len(), tris.len());
    println!("│");

    // Export to VTK
    export_vtk_mesh("output/plate_quad.vtk", &nodes, &quads)?;
    println!("│ Exported: output/plate_quad.vtk");
    export_vtk_mesh("output/plate_tri.vtk", &nodes_tri, &tris)?;
    println!("│ Exported: output/plate_tri.vtk");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 3: 3D block analysis.
fn demo_3d_block() -> anyhow::Result<()> {
    println!("┌─ 3D Block Analysis ──────────────────────────────────────┐");

    let width = 0.1;
    let height = 0.1;
    let depth = 0.5;
    let nx = 5;
    let ny = 5;
    let nz = 20;

    // Generate hex mesh
    let (nodes, hexes) = generate_box_3d(width, height, depth, nx, ny, nz);

    println!("│ Generated hex mesh: {} nodes, {} elements", nodes.len(), hexes.len());

    // Generate tet mesh
    let (nodes_tet, tets) = generate_tet_3d_from_hex(width, height, depth, nx, ny, nz);
    println!("│ Generated tet mesh: {} nodes, {} elements", nodes_tet.len(), tets.len());
    println!("│");

    // Export to VTK
    export_vtk_mesh("output/block_hex.vtk", &nodes, &hexes)?;
    println!("│ Exported: output/block_hex.vtk");
    export_vtk_mesh("output/block_tet.vtk", &nodes_tet, &tets)?;
    println!("│ Exported: output/block_tet.vtk");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 4: Material library.
fn demo_materials() -> anyhow::Result<()> {
    println!("┌─ Material Library ───────────────────────────────────────┐");

    let materials = [
        ("Steel A36", steel_a36()),
        ("Aluminum 6061", material_helpers::aluminum_6061()),
        ("Titanium Ti-64", material_helpers::titanium_ti64()),
        ("Concrete", material_helpers::concrete_normal()),
    ];

    println!("│ {:<20} │ {:>10} │ {:>10} │ {:>10} │", "Material", "E (GPa)", "ν", "ρ (kg/m³)");
    println!("│─────────────────────┼────────────┼────────────┼────────────│");

    for (name, mat) in &materials {
        println!("│ {:<20} │ {:>10.1} │ {:>10.2} │ {:>10.0} │",
            name, mat.young_modulus / 1e9, mat.poisson_ratio, mat.density);
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 5: BC helpers.
fn demo_bc_helpers() -> anyhow::Result<()> {
    println!("┌─ Boundary Condition Helpers ─────────────────────────────┐");

    let mut model: Model<Truss2> = Model::new();

    // Create a simple 3D structure
    let nodes = [
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
        Node::new_3d(0.0, 0.0, 1.0),
    ];

    for node in &nodes {
        model.add_node(*node);
    }

    for i in 0..4 {
        model.add_element(Truss2::new(0, i));
    }

    println!("│ Created model with {} nodes, {} elements", model.nodes.len(), model.elements.len());

    // Fix face at x=0
    fix_face_3d(&mut model, Some(0.0), None, None, 1e-6);
    println!("│ Applied fix_face_3d at x=0");

    println!("│ Total BCs applied: {}", model.bcs.len());
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}
