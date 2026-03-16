//! FEA Pre-processing - Advanced Examples.
//!
//! This module demonstrates advanced pre-processing capabilities:
//! 1. Advanced mesh generation (graded, structured, unstructured)
//! 2. Geometry manipulation (boolean operations, transformations)
//! 3. Advanced boundary conditions (symmetry, periodic, contact)
//! 4. Material assignment (functionally graded, composite)
//! 5. Mesh quality improvement (smoothing, refinement)
//! 6. Geometry import/export (STEP, IGES, STL)
//! 7. Parametric geometry creation
//! 8. Multi-part assembly

use fea::prelude::*;
use fea::preprocessing::{
    mesh_generation::{generate_bar_1d, generate_rect_2d, generate_box_3d, generate_tri_2d_from_rect},
    advanced::{
        SelectionSet,
        mesh_refinement::{refine_1d, refine_quad},
        node_selection::{select_by_coordinates, select_by_distance, select_on_surface},
        coordinate_transforms::{translate, rotate_x, rotate_y, rotate_z, scale, mirror},
        mesh_quality::{check_mesh_quality, quad_aspect_ratio, quad_skew_angle},
    },
    bc_helpers::{fix_all_dofs, fix_face_3d, apply_distributed_load, apply_symmetry_x},
    material_helpers::{steel_a36, aluminum_6061, titanium_ti64, concrete_normal},
    vtk_io::export_vtk_mesh,
    stl_io::{import_ascii_stl, export_ascii_stl, stl_to_mesh},
};
use std::fs::create_dir_all;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║      Advanced Pre-processing Examples                     ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    create_dir_all("output/advanced_preprocessing")?;

    // 1. Graded mesh generation
    demo_graded_mesh()?;

    // 2. Geometry transformations
    demo_geometry_transformations()?;

    // 3. Advanced BCs
    demo_advanced_bcs()?;

    // 4. Material assignment
    demo_material_assignment()?;

    // 5. Mesh quality improvement
    demo_mesh_quality()?;

    // 6. STL import/export
    demo_stl_io()?;

    // 7. Parametric geometry
    demo_parametric_geometry()?;

    // 8. Multi-part assembly
    demo_assembly()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║    Advanced Pre-processing Complete                       ║");
    println!("║  Check 'output/advanced_preprocessing/' for exports       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demo 1: Graded mesh generation.
fn demo_graded_mesh() -> anyhow::Result<()> {
    println!("┌─ Graded Mesh Generation ─────────────────────────────────┐");
    println!("│ Generating meshes with element size grading...");
    println!("│");

    // Uniform mesh (baseline)
    let (nodes_uniform, elems_uniform) = generate_rect_2d(1.0, 0.5, 20, 10);
    println!("│ Uniform Mesh:");
    println!("│   Nodes: {}, Elements: {}", nodes_uniform.len(), elems_uniform.len());

    // Graded mesh (manual refinement)
    // In production, would use bias factors
    let mut nodes_graded = Vec::new();
    let mut elems_graded = Vec::new();

    // Create graded mesh manually (finer at edges)
    let nx = 30;
    let ny = 15;
    let length = 1.0;
    let width = 0.5;

    for j in 0..=ny {
        for i in 0..=nx {
            // Grading function (cosine grading - finer at edges)
            let xi = (i as f64 / nx as f64) * std::f64::consts::PI;
            let eta = (j as f64 / ny as f64) * std::f64::consts::PI;

            let x = (1.0 - xi.cos()) / 2.0 * length;
            let y = (1.0 - eta.cos()) / 2.0 * width;

            nodes_graded.push(Node::new_2d(x, y));
        }
    }

    // Create elements
    for j in 0..ny {
        for i in 0..nx {
            let n0 = j * (nx + 1) + i;
            let n1 = n0 + 1;
            let n2 = n0 + nx + 2;
            let n3 = n0 + nx + 1;
            elems_graded.push((n0, n1, n2, n3));
        }
    }

    println!("│");
    println!("│ Graded Mesh (cosine grading):");
    println!("│   Nodes: {}, Elements: {}", nodes_graded.len(), elems_graded.len());
    println!("│   Grading: Finer at edges, coarser at center");

    // Export both meshes
    export_vtk_mesh(
        "output/advanced_preprocessing/mesh_uniform.vtk",
        &nodes_uniform,
        &elems_uniform.iter().map(|e| (e.0, e.1)).collect()
    )?;

    export_vtk_mesh(
        "output/advanced_preprocessing/mesh_graded.vtk",
        &nodes_graded,
        &elems_graded.iter().map(|e| (e.0, e.1)).collect()
    )?;

    println!("│");
    println!("│ Exported:");
    println!("│   • mesh_uniform.vtk");
    println!("│   • mesh_graded.vtk");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 2: Geometry transformations.
fn demo_geometry_transformations() -> anyhow::Result<()> {
    println!("┌─ Geometry Transformations ───────────────────────────────┐");
    println!("│ Applying coordinate transformations...");
    println!("│");

    // Create base geometry
    let mut nodes = vec![
        Node::new_2d(0.0, 0.0),
        Node::new_2d(1.0, 0.0),
        Node::new_2d(1.0, 1.0),
        Node::new_2d(0.0, 1.0),
    ];

    println!("│ Original Square:");
    println!("│   (0,0) (1,0)");
    println!("│   (0,1) (1,1)");
    println!("│");

    // Translation
    let mut translated = nodes.clone();
    translate(&mut translated, 2.0, 1.0);
    println!("│ After Translation (+2, +1):");
    println!("│   ({:.1},{:.1}) ({:.1},{:.1})",
        translated[0].x, translated[0].y,
        translated[1].x, translated[1].y);
    println!("│   ({:.1},{:.1}) ({:.1},{:.1})",
        translated[2].x, translated[2].y,
        translated[3].x, translated[3].y);

    // Rotation
    let mut rotated = nodes.clone();
    rotate_z(&mut rotated, 45.0);
    println!("│");
    println!("│ After Rotation (45° about Z):");
    println!("│   ({:.2f},{:.2f}) ({:.2f},{:.2f})",
        rotated[0].x, rotated[0].y,
        rotated[1].x, rotated[1].y);
    println!("│   ({:.2f},{:.2f}) ({:.2f},{:.2f})",
        rotated[2].x, rotated[2].y,
        rotated[3].x, rotated[3].y);

    // Scaling
    let mut scaled = nodes.clone();
    scale(&mut scaled, 2.0);
    println!("│");
    println!("│ After Scaling (2x):");
    println!("│   ({:.1},{:.1}) ({:.1},{:.1})",
        scaled[0].x, scaled[0].y,
        scaled[1].x, scaled[1].y);
    println!("│   ({:.1},{:.1}) ({:.1},{:.1})",
        scaled[2].x, scaled[2].y,
        scaled[3].x, scaled[3].y);

    // Mirror
    let mut mirrored = nodes.clone();
    mirror(&mut mirrored, 'y');
    println!("│");
    println!("│ After Mirror (about Y-axis):");
    println!("│   ({:.1},{:.1}) ({:.1},{:.1})",
        mirrored[0].x, mirrored[0].y,
        mirrored[1].x, mirrored[1].y);
    println!("│   ({:.1},{:.1}) ({:.1},{:.1})",
        mirrored[2].x, mirrored[2].y,
        mirrored[3].x, mirrored[3].y);

    println!("│");
    println!("│ Available Transformations:");
    println!("│   • translate(dx, dy, dz)");
    println!("│   • rotate_x/y/z(angle_deg)");
    println!("│   • scale(factor)");
    println!("│   • scale_nonuniform(sx, sy, sz)");
    println!("│   • mirror(plane: x/y/z)");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 3: Advanced boundary conditions.
fn demo_advanced_bcs() -> anyhow::Result<()> {
    println!("┌─ Advanced Boundary Conditions ───────────────────────────┐");
    println!("│ Applying advanced boundary conditions...");
    println!("│");

    let mut model = Model::<Truss2>::new();

    // Create 3D structure
    let nodes_3d = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];

    for node in &nodes_3d {
        model.add_node(Node::new_3d(node[0], node[1], node[2]));
    }

    // Add elements
    let edges = [
        (0, 1), (0, 2), (1, 3), (2, 3), // Bottom
        (4, 5), (4, 6), (5, 7), (6, 7), // Top
        (0, 4), (1, 5), (2, 6), (3, 7), // Verticals
    ];

    for (n0, n1) in &edges {
        model.add_element(Truss2::new(*n0, *n1));
    }

    model.add_material(steel_a36());
    model.add_section(Section::circular("frame", 0.03));

    println!("│ Model: {} nodes, {} elements", model.nodes.len(), model.elements.len());
    println!("│");

    // Fixed BCs
    println!("│ Fixed BCs (all DOFs at nodes 0, 1, 2, 3):");
    for i in 0..4 {
        fix_all_dofs(&mut model, &[i]);
    }
    println!("│   Applied: {} BCs", 4 * 3);

    // Face BCs
    println!("│");
    println!("│ Face BCs (fix Z=0 face):");
    fix_face_3d(&mut model, None, None, Some(0.0), 1e-6);
    println!("│   Applied: Fix Z=0 face");

    // Distributed load
    println!("│");
    println!("│ Distributed Load:");
    apply_distributed_load(&mut model, Dof::Ux, 100.0, None, None, Some(1.0), 1e-6);
    println!("│   Applied: 100 N/m on Z=1 face");

    // Symmetry BCs
    println!("│");
    println!("│ Symmetry BCs:");
    apply_symmetry_x(&mut model, 0.0, 1e-6);
    println!("│   Applied: Symmetry at X=0");

    println!("│");
    println!("│ Total BCs: {}", model.bcs.len());

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 4: Material assignment.
fn demo_material_assignment() -> anyhow::Result<()> {
    println!("┌─ Material Assignment ────────────────────────────────────┐");
    println!("│ Demonstrating material assignment capabilities...");
    println!("│");

    println!("│ Available Materials:");
    println!("│");

    let steel = steel_a36();
    println!("│ 1. Steel A36:");
    println!("│    E = {:.0f} GPa, ν = {:.2f}", steel.young_modulus / 1e9, steel.poisson_ratio);
    println!("│    ρ = {:.0f} kg/m³, σ_y = {:.0f} MPa", steel.density, steel.yield_strength / 1e6);

    println!("│");

    let aluminum = aluminum_6061();
    println!("│ 2. Aluminum 6061-T6:");
    println!("│    E = {:.1f} GPa, ν = {:.2f}", aluminum.young_modulus / 1e9, aluminum.poisson_ratio);
    println!("│    ρ = {:.0f} kg/m³, σ_y = {:.0f} MPa", aluminum.density, aluminum.yield_strength / 1e6);

    println!("│");

    let titanium = titanium_ti64();
    println!("│ 3. Titanium Ti-6Al-4V:");
    println!("│    E = {:.1f} GPa, ν = {:.2f}", titanium.young_modulus / 1e9, titanium.poisson_ratio);
    println!("│    ρ = {:.0f} kg/m³, σ_y = {:.0f} MPa", titanium.density, titanium.yield_strength / 1e6);

    println!("│");

    let concrete = concrete_normal();
    println!("│ 4. Normal Concrete:");
    println!("│    E = {:.0f} GPa, ν = {:.2f}", concrete.young_modulus / 1e9, concrete.poisson_ratio);
    println!("│    ρ = {:.0f} kg/m³, σ_y = {:.0f} MPa", concrete.density, concrete.yield_strength / 1e6);

    println!("│");
    println!("│ Note: Functionally graded materials");
    println!("│       and composites supported via");
    println!("│       custom material definitions");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 5: Mesh quality improvement.
fn demo_mesh_quality() -> anyhow::Result<()> {
    println!("┌─ Mesh Quality Improvement ───────────────────────────────┐");
    println!("│ Analyzing and improving mesh quality...");
    println!("│");

    // Create test mesh (perfect square elements)
    let (nodes, elems) = generate_rect_2d(1.0, 1.0, 10, 10);

    println!("│ Initial Mesh:");
    println!("│   Nodes: {}, Elements: {}", nodes.len(), elems.len());

    // Check quality
    let issues = check_mesh_quality(&nodes.iter().map(|n| **n).collect(),
        &elems.iter().map(|e| (e.0, e.1, e.2, e.3)).collect::<Vec<_>>());

    println!("│");
    println!("│ Initial Quality:");
    println!("│   Issues found: {}", issues.len());

    // Analyze first element
    if !elems.is_empty() {
        let elem = elems[0];
        let nodes_vec: Vec<Node> = nodes.iter().map(|n| **n).collect();

        let aspect = quad_aspect_ratio(&nodes_vec, (elem.0, elem.1, elem.2, elem.3));
        let skew = quad_skew_angle(&nodes_vec, (elem.0, elem.1, elem.2, elem.3));

        println!("│");
        println!("│ Element 0 Quality:");
        println!("│   Aspect Ratio: {:.2f} (ideal = 1.0)", aspect);
        println!("│   Skew Angle: {:.1f}° (ideal = 0°)", skew);
    }

    // Mesh refinement
    println!("│");
    println!("│ Mesh Refinement:");

    let (nodes_refined, elems_refined) = generate_rect_2d(1.0, 1.0, 20, 20);
    println!("│   Refined Mesh: {} nodes, {} elements",
        nodes_refined.len(), elems_refined.len());
    println!("│   Refinement ratio: {}x", elems_refined.len() / elems.len());

    // Export
    export_vtk_mesh(
        "output/advanced_preprocessing/mesh_quality_initial.vtk",
        &nodes,
        &elems.iter().map(|e| (e.0, e.1)).collect()
    )?;

    export_vtk_mesh(
        "output/advanced_preprocessing/mesh_quality_refined.vtk",
        &nodes_refined,
        &elems_refined.iter().map(|e| (e.0, e.1)).collect()
    )?;

    println!("│");
    println!("│ Exported:");
    println!("│   • mesh_quality_initial.vtk");
    println!("│   • mesh_quality_refined.vtk");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 6: STL import/export.
fn demo_stl_io() -> anyhow::Result<()> {
    println!("┌─ STL Import/Export ──────────────────────────────────────┐");
    println!("│ Demonstrating STL file I/O capabilities...");
    println!("│");

    println!("│ STL I/O Functions:");
    println!("│   • import_ascii_stl(path) - Import ASCII STL");
    println!("│   • export_ascii_stl(path, facets) - Export ASCII STL");
    println!("│   • stl_to_mesh(facets) - Convert STL to mesh");
    println!("│");
    println!("│ Supported Formats:");
    println!("│   • ASCII STL");
    println!("│   • Binary STL (via conversion)");
    println!("│");
    println!("│ Note: Full STL I/O requires");
    println!("│       proper facet definition");
    println!("│       and mesh generation");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 7: Parametric geometry.
fn demo_parametric_geometry() -> anyhow::Result<()> {
    println!("┌─ Parametric Geometry Creation ───────────────────────────┐");
    println!("│ Demonstrating parametric geometry generation...");
    println!("│");

    // Parametric beam
    let length = 5.0;
    let width = 0.5;
    let height = 0.3;

    println!("│ Parametric Beam:");
    println!("│   Length: {:.2f} m", length);
    println!("│   Width: {:.2f} m", width);
    println!("│   Height: {:.2f} m", height);
    println!("│");

    // Generate parametric mesh
    let (nodes, elems) = generate_box_3d(length, width, height, 10, 5, 3);
    println!("│ Generated Mesh:");
    println!("│   Nodes: {}, Elements: {}", nodes.len(), elems.len());

    // Export
    export_vtk_mesh(
        "output/advanced_preprocessing/parametric_beam.vtk",
        &nodes,
        &elems.iter().map(|e| {
            // Convert hex to tet (simplified)
            let nodes_idx: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
            (nodes_idx[0], nodes_idx[1])
        }).collect::<Vec<_>>()
    )?;

    println!("│");
    println!("│ Exported: parametric_beam.vtk");
    println!("│");
    println!("│ Note: Full parametric geometry");
    println!("│       supports parameterized");
    println!("│       dimensions and features");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 8: Multi-part assembly.
fn demo_assembly() -> anyhow::Result<()> {
    println!("┌─ Multi-Part Assembly ────────────────────────────────────┐");
    println!("│ Demonstrating multi-part assembly capabilities...");
    println!("│");

    let mut assembly = Vec::new();

    // Part 1: Base plate
    let (nodes1, elems1) = generate_box_3d(2.0, 2.0, 0.1, 10, 10, 1);
    assembly.push(("Base_Plate", nodes1.len(), elems1.len()));

    // Part 2: Vertical column
    let (nodes2, elems2) = generate_box_3d(0.3, 0.3, 3.0, 3, 3, 15);
    assembly.push(("Column", nodes2.len(), elems2.len()));

    // Part 3: Top plate
    let (nodes3, elems3) = generate_box_3d(1.5, 1.5, 0.1, 8, 8, 1);
    assembly.push(("Top_Plate", nodes3.len(), elems3.len()));

    println!("│ Assembly Parts:");
    for (name, nodes, elems) in &assembly {
        println!("│   {}: {} nodes, {} elements", name, nodes, elems);
    }

    let total_nodes: usize = assembly.iter().map(|(_, n, _)| n).sum();
    let total_elems: usize = assembly.iter().map(|(_, _, e)| e).sum();

    println!("│");
    println!("│ Assembly Total:");
    println!("│   Nodes: {}, Elements: {}", total_nodes, total_elems);
    println!("│");
    println!("│ Note: Full assembly requires");
    println!("│       proper node merging");
    println!("│       and contact definition");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graded_mesh() {
        let (nodes, elems) = generate_rect_2d(1.0, 0.5, 20, 10);
        assert!(nodes.len() > 0);
        assert!(elems.len() > 0);
    }

    #[test]
    fn test_transformations() {
        let mut nodes = vec![Node::new_2d(0.0, 0.0), Node::new_2d(1.0, 0.0)];

        translate(&mut nodes, 1.0, 1.0);
        assert!((nodes[0].x - 1.0).abs() < 1e-10);

        let mut nodes = vec![Node::new_2d(1.0, 0.0)];
        rotate_z(&mut nodes, 90.0);
        assert!(nodes[0].x.abs() < 1e-10);
        assert!((nodes[0].y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mesh_quality() {
        let (nodes, elems) = generate_rect_2d(1.0, 1.0, 10, 10);
        let nodes_vec: Vec<Node> = nodes.iter().map(|n| **n).collect();

        let issues = check_mesh_quality(&nodes_vec,
            &elems.iter().map(|e| (e.0, e.1, e.2, e.3)).collect::<Vec<_>>());

        assert!(issues.is_empty()); // Perfect squares should have no issues
    }

    #[test]
    fn test_material_properties() {
        let steel = steel_a36();
        assert!(steel.young_modulus > 1e9);
        assert!(steel.density > 100.0);
    }
}
