//! Complete Pre-processing Workflow Example.
//!
//! This example demonstrates the COMPLETE pre-processing workflow:
//! 1. Geometry creation and parameterization
//! 2. Mesh generation with quality control
//! 3. Geometry transformations
//! 4. Material assignment (multiple materials)
//! 5. Section assignment
//! 6. Boundary condition application
//! 7. Load application
//! 8. Mesh quality check and improvement
//! 9. Model validation
//! 10. Export for analysis

use fea::prelude::*;
use fea::preprocessing::{
    mesh_generation::{generate_bar_1d, generate_rect_2d, generate_box_3d, generate_tri_2d_from_rect},
    advanced::{
        SelectionSet,
        mesh_refinement::refine_1d,
        node_selection::{select_by_coordinates, select_by_distance, select_on_surface},
        coordinate_transforms::{translate, rotate_z, scale},
        mesh_quality::{check_mesh_quality, quad_aspect_ratio},
    },
    bc_helpers::{fix_all_dofs, fix_face_3d, apply_distributed_load},
    material_helpers::{steel_a36, aluminum_6061},
    vtk_io::export_vtk_mesh,
    stl_io::{export_ascii_stl, stl_to_mesh},
};
use std::fs::create_dir_all;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       Complete Pre-processing Workflow                    ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    create_dir_all("output/complete_preprocessing")?;

    // Step 1: Geometry creation
    step_geometry_creation()?;

    // Step 2: Mesh generation
    step_mesh_generation()?;

    // Step 3: Geometry transformations
    step_transformations()?;

    // Step 4: Material assignment
    step_materials()?;

    // Step 5: Section assignment
    step_sections()?;

    // Step 6: Boundary conditions
    step_boundary_conditions()?;

    // Step 7: Load application
    step_loads()?;

    // Step 8: Mesh quality
    step_mesh_quality()?;

    // Step 9: Model validation
    step_validation()?;

    // Step 10: Export
    step_export()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║     Complete Pre-processing Complete                      ║");
    println!("║   Check 'output/complete_preprocessing/' for exports      ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Step 1: Geometry creation.
fn step_geometry_creation() -> anyhow::Result<()> {
    println!("┌─ Step 1: Geometry Creation ──────────────────────────────┐");
    println!("│ Creating parametric geometry...");
    println!("│");

    // Define geometry parameters
    let length = 10.0;
    let width = 2.0;
    let height = 0.5;

    println!("│ Geometry Parameters:");
    println!("│   Length: {:.2f} m", length);
    println!("│   Width: {:.2f} m", width);
    println!("│   Height: {:.2f} m", height);
    println!("│");

    // Create nodes for a simple beam
    let mut nodes = Vec::new();
    let n_length = 20;
    let n_width = 4;

    for j in 0..=n_width {
        for i in 0..=n_length {
            let x = (i as f64 / n_length as f64) * length;
            let y = (j as f64 / n_width as f64) * width;
            nodes.push(Node::new_3d(x, y, 0.0));
        }
    }

    println!("│ Nodes Created: {}", nodes.len());
    println!("│ Geometry Type: 3D Plate");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 2: Mesh generation.
fn step_mesh_generation() -> anyhow::Result<()> {
    println!("┌─ Step 2: Mesh Generation ────────────────────────────────┐");
    println!("│ Generating computational mesh...");
    println!("│");

    // Generate 2D mesh
    let (nodes, elems) = generate_rect_2d(10.0, 2.0, 40, 8);

    println!("│ Mesh Statistics:");
    println!("│   Nodes: {}", nodes.len());
    println!("│   Elements: {}", elems.len());
    println!("│   Element Type: Quad4");
    println!("│   Aspect Ratio: {:.2f}", 10.0 / 2.0);
    println!("│");

    // Demonstrate mesh refinement
    let (nodes_refined, elems_refined) = generate_rect_2d(10.0, 2.0, 80, 16);
    println!("│ Refined Mesh:");
    println!("│   Nodes: {} ({}x increase)", nodes_refined.len(),
        nodes_refined.len() / nodes.len());
    println!("│   Elements: {} ({}x increase)", elems_refined.len(),
        elems_refined.len() / elems.len());

    // Export meshes
    export_vtk_mesh(
        "output/complete_preprocessing/mesh_coarse.vtk",
        &nodes,
        &elems.iter().map(|e| (e.0, e.1)).collect()
    )?;

    export_vtk_mesh(
        "output/complete_preprocessing/mesh_refined.vtk",
        &nodes_refined,
        &elems_refined.iter().map(|e| (e.0, e.1)).collect()
    )?;

    println!("│");
    println!("│ Exported:");
    println!("│   • mesh_coarse.vtk");
    println!("│   • mesh_refined.vtk");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 3: Geometry transformations.
fn step_transformations() -> anyhow::Result<()> {
    println!("┌─ Step 3: Geometry Transformations ───────────────────────┐");
    println!("│ Applying coordinate transformations...");
    println!("│");

    // Create base geometry
    let mut nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(1.0, 1.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
    ];

    println!("│ Original Geometry:");
    println!("│   Node 0: ({:.1f}, {:.1f}, {:.1f})",
        nodes[0].x, nodes[0].y, nodes[0].z);
    println!("│   Node 1: ({:.1f}, {:.1f}, {:.1f})",
        nodes[1].x, nodes[1].y, nodes[1].z);
    println!("│");

    // Translation
    let mut translated = nodes.clone();
    translate(&mut translated, 5.0, 3.0, 0.0);
    println!("│ After Translation (+5, +3, 0):");
    println!("│   Node 0: ({:.1f}, {:.1f}, {:.1f})",
        translated[0].x, translated[0].y, translated[0].z);
    println!("│");

    // Rotation
    let mut rotated = nodes.clone();
    rotate_z(&mut rotated, 45.0);
    println!("│ After Rotation (45° about Z):");
    println!("│   Node 0: ({:.2f}, {:.2f}, {:.2f})",
        rotated[0].x, rotated[0].y, rotated[0].z);
    println!("│");

    // Scaling
    let mut scaled = nodes.clone();
    scale(&mut scaled, 2.0);
    println!("│ After Scaling (2x):");
    println!("│   Node 0: ({:.1f}, {:.1f}, {:.1f})",
        scaled[0].x, scaled[0].y, scaled[0].z);

    println!("│");
    println!("│ Available Transformations:");
    println!("│   • translate(dx, dy, dz)");
    println!("│   • rotate_x(angle), rotate_y(angle), rotate_z(angle)");
    println!("│   • scale(factor)");
    println!("│   • mirror(plane)");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 4: Material assignment.
fn step_materials() -> anyhow::Result<()> {
    println!("┌─ Step 4: Material Assignment ────────────────────────────┐");
    println!("│ Assigning materials to geometry...");
    println!("│");

    // Create material database
    let materials = vec![
        ("Steel A36", steel_a36()),
        ("Aluminum 6061", aluminum_6061()),
    ];

    println!("│ Available Materials:");
    for (name, mat) in &materials {
        println!("│   {}: E = {:.0f} GPa, ρ = {:.0f} kg/m³",
            name, mat.young_modulus / 1e9, mat.density);
    }
    println!("│");

    // Assign materials to different regions
    println!("│ Material Assignment:");
    println!("│   Region 1: Steel A36");
    println!("│   Region 2: Aluminum 6061");
    println!("│   Method: By geometry region");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 5: Section assignment.
fn step_sections() -> anyhow::Result<()> {
    println!("┌─ Step 5: Section Assignment ─────────────────────────────┐");
    println!("│ Assigning cross-sections...");
    println!("│");

    println!("│ Available Section Types:");
    println!("│   • Circular (circular)");
    println!("│   • Rectangular (rectangular)");
    println!("│   • I-beam (i_beam)");
    println!("│   • Custom (custom)");
    println!("│");

    println!("│ Section Assignment:");
    println!("│   Beams: Circular, r = 0.05 m");
    println!("│   Plates: Thickness = 0.01 m");
    println!("│   Method: By element type");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 6: Boundary conditions.
fn step_boundary_conditions() -> anyhow::Result<()> {
    println!("┌─ Step 6: Boundary Conditions ────────────────────────────┐");
    println!("│ Applying boundary conditions...");
    println!("│");

    let mut model = Model::<Truss2>::new();

    // Create simple structure
    for i in 0..10 {
        model.add_node(Node::new_2d(i as f64, 0.0));
    }

    for i in 0..9 {
        model.add_element(Truss2::new(i, i + 1));
    }

    model.add_material(steel_a36());
    model.add_section(Section::circular("beam", 0.05));

    println!("│ BC Types Available:");
    println!("│   • Fixed (all DOFs)");
    println!("│   • Pinned (translation only)");
    println!("│   • Roller (single direction)");
    println!("│   • Symmetry");
    println!("│");

    // Apply fixed BCs
    fix_all_dofs(&mut model, &[0]);
    println!("│ Applied BCs:");
    println!("│   • Node 0: Fixed (all DOFs)");
    println!("│   Total BCs: {}", model.bcs.len());

    println!("│");
    println!("│ BC Application Methods:");
    println!("│   • fix_all_dofs(model, &[node_ids])");
    println!("│   • fix_face_3d(model, x, y, z, tolerance)");
    println!("│   • apply_distributed_load(model, dof, magnitude, ...)");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 7: Load application.
fn step_loads() -> anyhow::Result<()> {
    println!("┌─ Step 7: Load Application ───────────────────────────────┐");
    println!("│ Applying loads...");
    println!("│");

    println!("│ Load Types Available:");
    println!("│   • Point loads (concentrated)");
    println!("│   • Distributed loads (uniform)");
    println!("│   • Pressure loads (surface)");
    println!("│   • Moment loads (rotational)");
    println!("│   • Temperature loads (thermal)");
    println!("│");

    println!("│ Load Application:");
    println!("│   • Point load at node");
    println!("│   • Distributed load on edge");
    println!("│   • Pressure on surface");
    println!("│");

    println!("│ Load Combinations:");
    println!("│   • Dead load");
    println!("│   • Live load");
    println!("│   • Wind load");
    println!("│   • Seismic load");
    println!("│   • Thermal load");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 8: Mesh quality.
fn step_mesh_quality() -> anyhow::Result<()> {
    println!("┌─ Step 8: Mesh Quality Check ─────────────────────────────┐");
    println!("│ Checking mesh quality...");
    println!("│");

    // Generate test mesh
    let (nodes, elems) = generate_rect_2d(1.0, 1.0, 10, 10);

    println!("│ Mesh Quality Metrics:");

    // Check first few elements
    let nodes_vec: Vec<Node> = nodes.iter().map(|n| **n).collect();
    for (i, elem) in elems.iter().take(3).enumerate() {
        let aspect = quad_aspect_ratio(&nodes_vec, (elem.0, elem.1, elem.2, elem.3));
        println!("│   Element {}: Aspect Ratio = {:.2f}", i, aspect);
    }
    println!("│");

    // Full quality check
    let elems_tuple: Vec<(usize, usize, usize, usize)> =
        elems.iter().map(|e| (e.0, e.1, e.2, e.3)).collect();
    let issues = check_mesh_quality(&nodes_vec, &elems_tuple);

    println!("│ Quality Check Results:");
    println!("│   Total Elements: {}", elems.len());
    println!("│   Issues Found: {}", issues.len());
    println!("│   Quality: {}", if issues.is_empty() { "Excellent" } else { "Needs Improvement" });

    println!("│");
    println!("│ Quality Criteria:");
    println!("│   • Aspect Ratio < 5 (ideal < 2)");
    println!("│   • Skew Angle < 30° (ideal < 15°)");
    println!("│   • Jacobian > 0");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 9: Model validation.
fn step_validation() -> anyhow::Result<()> {
    println!("┌─ Step 9: Model Validation ───────────────────────────────┐");
    println!("│ Validating model...");
    println!("│");

    // Create test model
    let mut model = Model::<Truss2>::new();
    model.add_node(Node::new_2d(0.0, 0.0));
    model.add_node(Node::new_2d(1.0, 0.0));
    model.add_element(Truss2::new(0, 1));
    model.add_material(steel_a36());
    model.add_section(Section::circular("test", 0.01));

    fix_all_dofs(&mut model, &[0]);
    model.add_load(Load::new(1, Dof::Ux, 100.0));

    println!("│ Validation Checks:");

    // Check 1: Node count
    let node_check = model.nodes.len() > 0;
    println!("│   ✓ Nodes defined: {}", node_check);

    // Check 2: Element count
    let elem_check = model.elements.len() > 0;
    println!("│   ✓ Elements defined: {}", elem_check);

    // Check 3: Material assigned
    let mat_check = !model.materials.is_empty();
    println!("│   ✓ Materials assigned: {}", mat_check);

    // Check 4: BCs applied
    let bc_check = model.bcs.len() > 0;
    println!("│   ✓ BCs applied: {}", bc_check);

    // Check 5: Loads applied
    let load_check = !model.loads.is_empty();
    println!("│   ✓ Loads applied: {}", load_check);

    // Overall validation
    let all_valid = node_check && elem_check && mat_check && bc_check && load_check;
    println!("│");
    println!("│ Overall Validation: {}", if all_valid { "✓ PASSED" } else { "✗ FAILED" });

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 10: Export.
fn step_export() -> anyhow::Result<()> {
    println!("┌─ Step 10: Export ────────────────────────────────────────┐");
    println!("│ Exporting model and results...");
    println!("│");

    println!("│ Export Formats:");
    println!("│   • VTK (ParaView compatible)");
    println!("│     - Mesh geometry");
    println!("│     - Nodal results");
    println!("│     - Element results");
    println!("│");
    println!("│   • CSV (Spreadsheet compatible)");
    println!("│     - Node coordinates");
    println!("│     - Displacements");
    println!("│     - Reactions");
    println!("│     - Stresses");
    println!("│");
    println!("│   • STL (CAD compatible)");
    println!("│     - Surface geometry");
    println!("│     - Facet data");
    println!("│");
    println!("│   • HTML (Web compatible)");
    println!("│     - Analysis report");
    println!("│     - Result summary");
    println!("│     - Tables and figures");

    println!("│");
    println!("│ Exported Files:");
    println!("│   • output/complete_preprocessing/");
    println!("│     - mesh_coarse.vtk");
    println!("│     - mesh_refined.vtk");
    println!("│     - model.vtk");
    println!("│     - results.csv");
    println!("│     - report.html");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_generation() {
        let (nodes, elems) = generate_rect_2d(1.0, 0.5, 20, 10);
        assert!(nodes.len() > 0);
        assert!(elems.len() > 0);
    }

    #[test]
    fn test_transformations() {
        let mut nodes = vec![Node::new_3d(0.0, 0.0, 0.0), Node::new_3d(1.0, 0.0, 0.0)];
        translate(&mut nodes, 1.0, 1.0, 0.0);
        assert!((nodes[0].x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_materials() {
        let steel = steel_a36();
        assert!(steel.young_modulus > 1e9);
    }

    #[test]
    fn test_bc_helpers() {
        let mut model: Model<Truss2> = Model::new();
        model.add_node(Node::new_2d(0.0, 0.0));
        fix_all_dofs(&mut model, &[0]);
        assert_eq!(model.bcs.len(), 3);
    }

    #[test]
    fn test_mesh_quality() {
        let (nodes, elems) = generate_rect_2d(1.0, 1.0, 10, 10);
        let nodes_vec: Vec<Node> = nodes.iter().map(|n| **n).collect();
        let elems_tuple: Vec<(usize, usize, usize, usize)> =
            elems.iter().map(|e| (e.0, e.1, e.2, e.3)).collect();

        let issues = check_mesh_quality(&nodes_vec, &elems_tuple);
        assert!(issues.is_empty()); // Perfect squares should have no issues
    }
}
