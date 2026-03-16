//! Pre-processing Comprehensive Example.
//!
//! This example demonstrates all pre-processing capabilities:
//! - Mesh generation (1D, 2D, 3D)
//! - Mesh refinement
//! - Node selection
//! - Selection sets
//! - Coordinate transformations
//! - Mesh quality analysis
//! - Material assignment

use fea::prelude::*;
use fea::preprocessing::{
    mesh_generation::{generate_bar_1d, generate_rect_2d, generate_box_3d, generate_tet_3d_from_hex},
    advanced::{
        SelectionSet,
        mesh_refinement::refine_1d,
        node_selection::{select_by_coordinates, select_by_distance, select_on_surface},
        coordinate_transforms::{translate, rotate_z, scale},
        mesh_quality::{check_mesh_quality, quad_aspect_ratio},
    },
    material_helpers::{steel_a36, aluminum_6061},
    bc_helpers::fix_all_dofs,
};

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║        Pre-Processing Comprehensive Example               ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Demo 1: Mesh generation
    demo_mesh_generation()?;

    // Demo 2: Mesh refinement
    demo_mesh_refinement()?;

    // Demo 3: Node selection
    demo_node_selection()?;

    // Demo 4: Selection sets
    demo_selection_sets()?;

    // Demo 5: Coordinate transforms
    demo_coordinate_transforms()?;

    // Demo 6: Mesh quality
    demo_mesh_quality()?;

    // Demo 7: Material assignment
    demo_material_assignment()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║            Pre-Processing Complete                        ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demo 1: Mesh generation.
fn demo_mesh_generation() -> anyhow::Result<()> {
    println!("┌─ Mesh Generation ────────────────────────────────────────┐");

    // 1D bar
    let (nodes_1d, elems_1d) = generate_bar_1d(1.0, 10, 0.01);
    println!("│ 1D Bar: {} nodes, {} elements", nodes_1d.len(), elems_1d.len());

    // 2D rectangle
    let (nodes_2d, elems_2d) = generate_rect_2d(1.0, 0.5, 20, 10);
    println!("│ 2D Quad: {} nodes, {} elements", nodes_2d.len(), elems_2d.len());

    // 2D triangles
    let (nodes_tri, elems_tri) = generate_rect_2d(1.0, 0.5, 20, 10);
    let tri_count = elems_tri.len() * 2;
    println!("│ 2D Tri: {} nodes, {} elements", nodes_tri.len(), tri_count);

    // 3D box
    let (nodes_3d, elems_3d) = generate_box_3d(1.0, 0.5, 0.2, 10, 5, 2);
    println!("│ 3D Hex: {} nodes, {} elements", nodes_3d.len(), elems_3d.len());

    // 3D tetrahedra
    let (nodes_tet, elems_tet) = generate_tet_3d_from_hex(1.0, 0.5, 0.2, 10, 5, 2);
    println!("│ 3D Tet: {} nodes, {} elements", nodes_tet.len(), elems_tet.len());

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 2: Mesh refinement.
fn demo_mesh_refinement() -> anyhow::Result<()> {
    println!("┌─ Mesh Refinement ────────────────────────────────────────┐");

    // Start with coarse 1D mesh
    let nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
    ];
    let elements = vec![(0, 1)];

    println!("│ Initial: {} nodes, {} elements", nodes.len(), elements.len());

    // Refine once
    let (nodes1, elems1) = refine_1d(&nodes, &elements);
    println!("│ After 1 refinement: {} nodes, {} elements", nodes1.len(), elems1.len());

    // Refine twice
    let (nodes2, elems2) = refine_1d(&nodes1, &elems1);
    println!("│ After 2 refinements: {} nodes, {} elements", nodes2.len(), elems2.len());

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 3: Node selection.
fn demo_node_selection() -> anyhow::Result<()> {
    println!("┌─ Node Selection ─────────────────────────────────────────┐");

    // Create test nodes
    let nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(0.5, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
        Node::new_3d(1.0, 1.0, 0.0),
    ];

    // Select by X range
    let selected_x = select_by_coordinates(&nodes, Some((0.0, 0.5)), None, None);
    println!("│ Nodes with X in [0, 0.5]: {} nodes", selected_x.len());

    // Select by distance
    let selected_dist = select_by_distance(&nodes, (0.5, 0.5, 0.0), 0.0, 0.6);
    println!("│ Nodes within 0.6 of (0.5, 0.5): {} nodes", selected_dist.len());

    // Select on surface
    let selected_surface = select_on_surface(&nodes, 'x', 0.0, 1e-6);
    println!("│ Nodes on X=0 surface: {} nodes", selected_surface.len());

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 4: Selection sets.
fn demo_selection_sets() -> anyhow::Result<()> {
    println!("┌─ Selection Sets ─────────────────────────────────────────┐");

    let mut left_face = SelectionSet::new("Left_Face");
    left_face.add_node(0);
    left_face.add_node(3);

    let mut right_face = SelectionSet::new("Right_Face");
    right_face.add_node(2);
    right_face.add_node(4);

    println!("│ Selection Set: '{}'", left_face.name);
    println!("│   Nodes: {}", left_face.num_nodes());
    println!("│   Elements: {}", left_face.num_elements());

    println!("│ Selection Set: '{}'", right_face.name);
    println!("│   Nodes: {}", right_face.num_nodes());
    println!("│   Elements: {}", right_face.num_elements());

    println!("│");
    println!("│ Contains node 0 in Left_Face: {}", left_face.contains_node(0));
    println!("│ Contains node 2 in Left_Face: {}", left_face.contains_node(2));

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 5: Coordinate transforms.
fn demo_coordinate_transforms() -> anyhow::Result<()> {
    println!("┌─ Coordinate Transforms ──────────────────────────────────┐");

    let mut nodes = vec![
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
    ];

    println!("│ Original nodes:");
    println!("│   Node 0: ({:.1}, {:.1}, {:.1})", nodes[0].x, nodes[0].y, nodes[0].z);
    println!("│   Node 1: ({:.1}, {:.1}, {:.1})", nodes[1].x, nodes[1].y, nodes[1].z);

    // Translate
    translate(&mut nodes, 1.0, 1.0, 0.0);
    println!("│");
    println!("│ After translate (1, 1, 0):");
    println!("│   Node 0: ({:.1}, {:.1}, {:.1})", nodes[0].x, nodes[0].y, nodes[0].z);

    // Reset
    nodes = vec![
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
    ];

    // Rotate about Z
    rotate_z(&mut nodes, 90.0);
    println!("│");
    println!("│ After rotate Z 90°:");
    println!("│   Node 0: ({:.1}, {:.1}, {:.1})", nodes[0].x, nodes[0].y, nodes[0].z);

    // Reset
    nodes = vec![
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
    ];

    // Scale
    scale(&mut nodes, 2.0);
    println!("│");
    println!("│ After scale 2x:");
    println!("│   Node 0: ({:.1}, {:.1}, {:.1})", nodes[0].x, nodes[0].y, nodes[0].z);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 6: Mesh quality.
fn demo_mesh_quality() -> anyhow::Result<()> {
    println!("┌─ Mesh Quality Analysis ──────────────────────────────────┐");

    // Good quad element (square)
    let good_nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(1.0, 1.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
    ];
    let good_elem = (0, 1, 2, 3);

    let aspect_good = quad_aspect_ratio(&good_nodes, good_elem);
    println!("│ Square element aspect ratio: {:.2} (ideal = 1.0)", aspect_good);

    // Bad quad element (stretched)
    let bad_nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(5.0, 0.0, 0.0),
        Node::new_3d(5.0, 1.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
    ];
    let bad_elem = (0, 1, 2, 3);

    let aspect_bad = quad_aspect_ratio(&bad_nodes, bad_elem);
    println!("│ Stretched element aspect ratio: {:.2} (poor > 10)", aspect_bad);

    // Check mesh quality
    let elements = vec![good_elem, bad_elem];
    let all_nodes = [good_nodes, bad_nodes].concat();
    let issues = check_mesh_quality(&all_nodes, &elements);

    println!("│");
    println!("│ Quality issues found: {}", issues.len());
    for issue in &issues {
        println!("│   • {}", issue);
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 7: Material assignment.
fn demo_material_assignment() -> anyhow::Result<()> {
    println!("┌─ Material Assignment ────────────────────────────────────┐");

    let steel = steel_a36();
    println!("│ Steel A36:");
    println!("│   Young's Modulus: {:.0} GPa", steel.young_modulus / 1e9);
    println!("│   Poisson's Ratio: {:.2}", steel.poisson_ratio);
    println!("│   Density: {:.0} kg/m³", steel.density);
    println!("│   Yield Strength: {:.0} MPa", steel.yield_strength / 1e6);

    println!("│");

    let aluminum = aluminum_6061();
    println!("│ Aluminum 6061-T6:");
    println!("│   Young's Modulus: {:.1} GPa", aluminum.young_modulus / 1e9);
    println!("│   Poisson's Ratio: {:.2}", aluminum.poisson_ratio);
    println!("│   Density: {:.0} kg/m³", aluminum.density);
    println!("│   Yield Strength: {:.0} MPa", aluminum.yield_strength / 1e6);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_generation() {
        let (nodes, elems) = generate_bar_1d(1.0, 5, 0.01);
        assert_eq!(nodes.len(), 6);
        assert_eq!(elems.len(), 5);
    }

    #[test]
    fn test_node_selection() {
        let nodes = vec![
            Node::new_3d(0.0, 0.0, 0.0),
            Node::new_3d(1.0, 0.0, 0.0),
        ];

        let selected = select_by_coordinates(&nodes, Some((0.0, 0.5)), None, None);
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_selection_set() {
        let mut set = SelectionSet::new("test");
        set.add_node(0);
        set.add_node(1);

        assert!(set.contains_node(0));
        assert!(!set.contains_node(2));
    }

    #[test]
    fn test_coordinate_transforms() {
        let mut nodes = vec![Node::new_3d(1.0, 0.0, 0.0)];
        translate(&mut nodes, 1.0, 0.0, 0.0);
        assert!((nodes[0].x - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_mesh_quality() {
        let nodes = vec![
            Node::new_3d(0.0, 0.0, 0.0),
            Node::new_3d(1.0, 0.0, 0.0),
            Node::new_3d(1.0, 1.0, 0.0),
            Node::new_3d(0.0, 1.0, 0.0),
        ];
        let elem = (0, 1, 2, 3);

        let aspect = quad_aspect_ratio(&nodes, elem);
        assert!((aspect - 1.0).abs() < 1e-10);
    }
}
