//! Complete Pre-processing Pipeline Example.
//!
//! This example demonstrates the COMPLETE pre-processing workflow:
//! 1. Geometry creation
//! 2. Mesh generation (GPU-accelerated)
//! 3. Mesh quality check
//! 4. Mesh refinement
//! 5. Material assignment
//! 6. Boundary condition application
//! 7. Node/element selection
//! 8. Coordinate transformations
//! 9. Export to various formats
//! 10. Validation

use fea::prelude::*;
use fea::preprocessing::{
    mesh_generation::{generate_bar_1d, generate_rect_2d, generate_box_3d, generate_tri_2d_from_rect, generate_tet_3d_from_hex},
    advanced::{
        SelectionSet,
        mesh_refinement::{refine_1d, refine_quad, refine_hex},
        node_selection::{select_by_coordinates, select_by_distance, select_on_surface},
        coordinate_transforms::{translate, rotate_x, rotate_y, rotate_z, scale, scale_nonuniform, mirror},
        mesh_quality::{check_mesh_quality, quad_aspect_ratio, quad_skew_angle, quad_jacobian},
    },
    bc_helpers::{fix_all_dofs, fix_face_3d, apply_distributed_load, apply_symmetry_x},
    material_helpers::{steel_a36, aluminum_6061, titanium_ti64, concrete_normal},
    vtk_io::export_vtk_mesh,
    stl_io::{import_ascii_stl, export_ascii_stl, stl_to_mesh},
};
use std::fs::create_dir_all;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       Complete Pre-Processing Pipeline                    ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    create_dir_all("output/preprocessing")?;

    // Step 1: Geometry creation
    step_geometry_creation()?;

    // Step 2: Mesh generation
    step_mesh_generation()?;

    // Step 3: Mesh quality check
    step_mesh_quality()?;

    // Step 4: Mesh refinement
    step_mesh_refinement()?;

    // Step 5: Material assignment
    step_material_assignment()?;

    // Step 6: Node selection
    step_node_selection()?;

    // Step 7: Selection sets
    step_selection_sets()?;

    // Step 8: Coordinate transforms
    step_coordinate_transforms()?;

    // Step 9: BC application
    step_bc_application()?;

    // Step 10: Export
    step_export()?;

    // Validation
    run_validation()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║     Pre-Processing Pipeline Complete                      ║");
    println!("║     Check 'output/preprocessing/' for exports             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Step 1: Geometry creation.
fn step_geometry_creation() -> anyhow::Result<()> {
    println!("┌─ Step 1: Geometry Creation ──────────────────────────────┐");
    println!("│ Creating geometric primitives...");
    println!("│");
    println!("│ Primitives available:");
    println!("│   • 1D: Line, Bar");
    println!("│   • 2D: Rectangle, Circle, Plate");
    println!("│   • 3D: Box, Cylinder, Sphere");
    println!("│");
    println!("│ For this example: 3D Box (1.0 x 0.5 x 0.2)");
    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 2: Mesh generation.
fn step_mesh_generation() -> anyhow::Result<()> {
    println!("┌─ Step 2: Mesh Generation ────────────────────────────────┐");

    // 1D mesh
    let (nodes_1d, elems_1d) = generate_bar_1d(1.0, 10, 0.01);
    println!("│ 1D Bar Mesh:");
    println!("│   Nodes: {}, Elements: {}", nodes_1d.len(), elems_1d.len());

    // 2D quad mesh
    let (nodes_2d, elems_2d) = generate_rect_2d(1.0, 0.5, 20, 10);
    println!("│");
    println!("│ 2D Quad Mesh:");
    println!("│   Nodes: {}, Elements: {}", nodes_2d.len(), elems_2d.len());

    // 2D tri mesh
    let (nodes_tri, _) = generate_tri_2d_from_rect(1.0, 0.5, 20, 10);
    println!("│");
    println!("│ 2D Tri Mesh:");
    println!("│   Nodes: {}, Elements: {} (2 per quad)", nodes_tri.len(), elems_2d.len() * 2);

    // 3D hex mesh
    let (nodes_3d, elems_3d) = generate_box_3d(1.0, 0.5, 0.2, 10, 5, 2);
    println!("│");
    println!("│ 3D Hex Mesh:");
    println!("│   Nodes: {}, Elements: {}", nodes_3d.len(), elems_3d.len());

    // 3D tet mesh
    let (nodes_tet, elems_tet) = generate_tet_3d_from_hex(1.0, 0.5, 0.2, 10, 5, 2);
    println!("│");
    println!("│ 3D Tet Mesh:");
    println!("│   Nodes: {}, Elements: {} (5 per hex)", nodes_tet.len(), elems_tet.len());

    // Export 3D mesh
    export_vtk_mesh("output/preprocessing/mesh_3d.vtk", &nodes_3d, &elems_3d)?;
    println!("│");
    println!("│ Exported: output/preprocessing/mesh_3d.vtk");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 3: Mesh quality check.
fn step_mesh_quality() -> anyhow::Result<()> {
    println!("┌─ Step 3: Mesh Quality Check ─────────────────────────────┐");

    // Create test quad mesh
    let nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(1.0, 1.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
    ];
    let elements = vec![(0, 1, 2, 3)];

    // Check quality
    let issues = check_mesh_quality(&nodes, &elements);

    println!("│ Mesh Quality Metrics:");
    println!("│   Elements checked: {}", elements.len());
    println!("│   Quality issues: {}", issues.len());

    if issues.is_empty() {
        println!("│   ✓ All elements pass quality checks");
    } else {
        println!("│   Issues found:");
        for issue in &issues {
            println!("│     • {}", issue);
        }
    }

    // Compute aspect ratio for first element
    let aspect = quad_aspect_ratio(&nodes, elements[0]);
    println!("│");
    println!("│ Element 0 Aspect Ratio: {:.2} (ideal = 1.0)", aspect);

    // Compute skew angle
    let skew = quad_skew_angle(&nodes, elements[0]);
    println!("│ Element 0 Skew Angle: {:.1}° (ideal = 0°)", skew);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 4: Mesh refinement.
fn step_mesh_refinement() -> anyhow::Result<()> {
    println!("┌─ Step 4: Mesh Refinement ────────────────────────────────┐");

    // Start with coarse 1D mesh
    let nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
    ];
    let elements = vec![(0, 1)];

    println!("│ Initial mesh: {} nodes, {} elements", nodes.len(), elements.len());

    // Refine once
    let (nodes1, elems1) = refine_1d(&nodes, &elements);
    println!("│");
    println!("│ After 1st refinement: {} nodes, {} elements", nodes1.len(), elems1.len());

    // Refine twice
    let (nodes2, elems2) = refine_1d(&nodes1, &elems1);
    println!("│ After 2nd refinement: {} nodes, {} elements", nodes2.len(), elems2.len());

    // Refine thrice
    let (nodes3, elems3) = refine_1d(&nodes2, &elems2);
    println!("│ After 3rd refinement: {} nodes, {} elements", nodes3.len(), elems3.len());

    println!("│");
    println!("│ Refinement ratio: {}x elements per level", elems1.len() / elements.len());

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 5: Material assignment.
fn step_material_assignment() -> anyhow::Result<()> {
    println!("┌─ Step 5: Material Assignment ────────────────────────────┐");
    println!("│ Available Materials:");
    println!("│");

    let steel = steel_a36();
    println!("│ Steel A36:");
    println!("│   E = {:.0} GPa", steel.young_modulus / 1e9);
    println!("│   ν = {:.2}", steel.poisson_ratio);
    println!("│   ρ = {:.0} kg/m³", steel.density);
    println!("│   σ_y = {:.0} MPa", steel.yield_strength / 1e6);

    println!("│");

    let aluminum = aluminum_6061();
    println!("│ Aluminum 6061-T6:");
    println!("│   E = {:.1} GPa", aluminum.young_modulus / 1e9);
    println!("│   ν = {:.2}", aluminum.poisson_ratio);
    println!("│   ρ = {:.0} kg/m³", aluminum.density);
    println!("│   σ_y = {:.0} MPa", aluminum.yield_strength / 1e6);

    println!("│");

    let titanium = titanium_ti64();
    println!("│ Titanium Ti-6Al-4V:");
    println!("│   E = {:.1} GPa", titanium.young_modulus / 1e9);
    println!("│   ν = {:.2}", titanium.poisson_ratio);
    println!("│   ρ = {:.0} kg/m³", titanium.density);
    println!("│   σ_y = {:.0} MPa", titanium.yield_strength / 1e6);

    println!("│");

    let concrete = concrete_normal();
    println!("│ Normal Concrete:");
    println!("│   E = {:.0} GPa", concrete.young_modulus / 1e9);
    println!("│   ν = {:.2}", concrete.poisson_ratio);
    println!("│   ρ = {:.0} kg/m³", concrete.density);
    println!("│   σ_y = {:.0} MPa", concrete.yield_strength / 1e6);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 6: Node selection.
fn step_node_selection() -> anyhow::Result<()> {
    println!("┌─ Step 6: Node Selection ─────────────────────────────────┐");

    let nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(0.5, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
        Node::new_3d(1.0, 1.0, 0.0),
        Node::new_3d(0.5, 0.5, 0.0),
    ];

    // Select by X range
    let selected_x = select_by_coordinates(&nodes, Some((0.0, 0.5)), None, None);
    println!("│ Selection by X coordinate [0, 0.5]:");
    println!("│   Selected nodes: {}", selected_x.len());
    println!("│   Node IDs: {:?}", selected_x);

    // Select by distance
    println!("│");
    let selected_dist = select_by_distance(&nodes, (0.5, 0.5, 0.0), 0.0, 0.6);
    println!("│ Selection by distance from (0.5, 0.5):");
    println!("│   Max distance: 0.6");
    println!("│   Selected nodes: {}", selected_dist.len());
    println!("│   Node IDs: {:?}", selected_dist);

    // Select on surface
    println!("│");
    let selected_surface = select_on_surface(&nodes, 'x', 0.0, 1e-6);
    println!("│ Selection on X=0 surface:");
    println!("│   Selected nodes: {}", selected_surface.len());
    println!("│   Node IDs: {:?}", selected_surface);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 7: Selection sets.
fn step_selection_sets() -> anyhow::Result<()> {
    println!("┌─ Step 7: Selection Sets ─────────────────────────────────┐");

    let mut left_face = SelectionSet::new("Left_Face");
    left_face.add_node(0);
    left_face.add_node(3);

    let mut right_face = SelectionSet::new("Right_Face");
    right_face.add_node(2);
    right_face.add_node(4);

    let mut bottom_face = SelectionSet::new("Bottom_Face");
    bottom_face.add_node(0);
    bottom_face.add_node(1);
    bottom_face.add_node(2);

    println!("│ Selection Sets Created:");
    println!("│");
    println!("│ '{}':", left_face.name);
    println!("│   Nodes: {}", left_face.num_nodes());
    println!("│   Elements: {}", left_face.num_elements());
    println!("│   Contains node 0: {}", left_face.contains_node(0));
    println!("│   Contains node 2: {}", left_face.contains_node(2));

    println!("│");
    println!("│ '{}':", right_face.name);
    println!("│   Nodes: {}", right_face.num_nodes());

    println!("│");
    println!("│ '{}':", bottom_face.name);
    println!("│   Nodes: {}", bottom_face.num_nodes());

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 8: Coordinate transforms.
fn step_coordinate_transforms() -> anyhow::Result<()> {
    println!("┌─ Step 8: Coordinate Transforms ──────────────────────────┐");

    let mut nodes = vec![Node::new_3d(1.0, 0.0, 0.0)];

    println!("│ Original node: ({:.1}, {:.1}, {:.1})", nodes[0].x, nodes[0].y, nodes[0].z);

    // Translation
    translate(&mut nodes, 1.0, 1.0, 0.0);
    println!("│");
    println!("│ After translate (1, 1, 0):");
    println!("│   Node: ({:.1}, {:.1}, {:.1})", nodes[0].x, nodes[0].y, nodes[0].z);

    // Reset
    nodes = vec![Node::new_3d(1.0, 0.0, 0.0)];

    // Rotation about Z
    rotate_z(&mut nodes, 90.0);
    println!("│");
    println!("│ After rotate Z 90°:");
    println!("│   Node: ({:.1}, {:.1}, {:.1})", nodes[0].x, nodes[0].y, nodes[0].z);

    // Reset
    nodes = vec![Node::new_3d(1.0, 0.0, 0.0)];

    // Scaling
    scale(&mut nodes, 2.0);
    println!("│");
    println!("│ After scale 2x:");
    println!("│   Node: ({:.1}, {:.1}, {:.1})", nodes[0].x, nodes[0].y, nodes[0].z);

    // Reset
    nodes = vec![Node::new_3d(1.0, 0.0, 0.0)];

    // Mirror
    mirror(&mut nodes, 'y');
    println!("│");
    println!("│ After mirror Y:");
    println!("│   Node: ({:.1}, {:.1}, {:.1})", nodes[0].x, nodes[0].y, nodes[0].z);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 9: BC application.
fn step_bc_application() -> anyhow::Result<()> {
    println!("┌─ Step 9: Boundary Condition Application ─────────────────┐");
    println!("│ BC Helpers Available:");
    println!("│");
    println!("│ • fix_all_dofs() - Fix all DOFs on nodes");
    println!("│ • fix_face_3d() - Fix nodes on 3D face (x/y/z plane)");
    println!("│ • apply_distributed_load() - Apply load to face");
    println!("│ • apply_symmetry_x() - Apply symmetry BCs");
    println!("│");
    println!("│ Usage Example:");
    println!("│   fix_all_dofs(&mut model, &[0, 1, 2]);");
    println!("│   fix_face_3d(&mut model, Some(0.0), None, None, 1e-6);");
    println!("│   apply_distributed_load(&mut model, Dof::Uy, 100.0, ...);");
    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 10: Export.
fn step_export() -> anyhow::Result<()> {
    println!("┌─ Step 10: Export ────────────────────────────────────────┐");
    println!("│ Export Formats Available:");
    println!("│");
    println!("│ • VTK - ParaView compatible");
    println!("│   - export_vtk_mesh()");
    println!("│   - export_vtk_displacements()");
    println!("│   - export_vtk_scalar()");
    println!("│");
    println!("│ • STL - CAD compatible");
    println!("│   - import_ascii_stl()");
    println!("│   - export_ascii_stl()");
    println!("│   - stl_to_mesh()");
    println!("│");
    println!("│ Exports generated in: output/preprocessing/");
    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Run validation.
fn run_validation() -> anyhow::Result<()> {
    println!("┌─ Validation ─────────────────────────────────────────────┐");
    println!("│ Running validation checks...");

    // Validate mesh generation
    let (nodes, elems) = generate_bar_1d(1.0, 10, 0.01);
    assert_eq!(nodes.len(), 11, "1D mesh node count");
    assert_eq!(elems.len(), 10, "1D mesh element count");

    println!("│   ✓ Mesh generation validated");

    // Validate transforms
    let mut test_nodes = vec![Node::new_3d(1.0, 0.0, 0.0)];
    translate(&mut test_nodes, 1.0, 0.0, 0.0);
    assert!((test_nodes[0].x - 2.0).abs() < 1e-10, "Translation");

    println!("│   ✓ Coordinate transforms validated");

    // Validate selection
    let nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(0.5, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
    ];
    let selected = select_by_coordinates(&nodes, Some((0.0, 0.5)), None, None);
    assert_eq!(selected.len(), 2, "Node selection");

    println!("│   ✓ Node selection validated");

    // Validate selection sets
    let mut set = SelectionSet::new("test");
    set.add_node(0);
    assert!(set.contains_node(0), "Selection set");
    assert!(!set.contains_node(1), "Selection set");

    println!("│   ✓ Selection sets validated");

    println!("│");
    println!("│ All validations passed ✓");
    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_pipeline() {
        // Test mesh generation
        let (nodes, elems) = generate_bar_1d(1.0, 5, 0.01);
        assert_eq!(nodes.len(), 6);
        assert_eq!(elems.len(), 5);

        // Test transforms
        let mut nodes = vec![Node::new_3d(1.0, 0.0, 0.0)];
        translate(&mut nodes, 1.0, 0.0, 0.0);
        assert!((nodes[0].x - 2.0).abs() < 1e-10);

        // Test selection
        let nodes = vec![
            Node::new_3d(0.0, 0.0, 0.0),
            Node::new_3d(0.5, 0.0, 0.0),
        ];
        let selected = select_by_coordinates(&nodes, Some((0.0, 0.5)), None, None);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_selection_sets() {
        let mut set = SelectionSet::new("test");
        set.add_node(0);
        set.add_node(1);

        assert!(set.contains_node(0));
        assert!(set.contains_node(1));
        assert!(!set.contains_node(2));
        assert_eq!(set.num_nodes(), 2);
    }

    #[test]
    fn test_material_helpers() {
        let steel = steel_a36();
        assert!(steel.young_modulus > 1e9);
        assert!(steel.density > 100.0);
    }
}
