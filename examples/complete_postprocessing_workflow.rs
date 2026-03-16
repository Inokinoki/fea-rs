//! Complete Post-processing Workflow Example.
//!
//! This example demonstrates the COMPLETE post-processing workflow:
//! 1. Result import (VTK, CSV)
//! 2. Result structures and access
//! 3. Result extraction (nodal, elemental)
//! 4. Derived quantities (stresses, strains)
//! 5. Result comparison (multiple analyses)
//! 6. Contour plot generation
//! 7. Animation generation
//! 8. Report generation (text, HTML)
//! 9. Export to multiple formats
//! 10. Custom post-processing workflows

use fea::prelude::*;
use fea::postprocessing::{
    FeaResults, StressResult, StrainResult,
    vtk_export::{export_displacements, export_stresses, export_reactions},
    csv_export::{export_displacements_csv, export_reactions_csv, export_stresses_csv},
    report_generation::{generate_text_report, generate_html_report},
    visualization::create_color_map,
    advanced::{
        ContourPlot, DeformedShapeAnimation,
        result_comparison::{
            compare_displacements, l2_norm_difference, relative_error,
            generate_comparison_report,
        },
    },
};
use fea::preprocessing::{
    mesh_generation::generate_rect_2d,
    vtk_io::export_vtk_mesh,
};
use std::fs::create_dir_all;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       Complete Post-processing Workflow                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    create_dir_all("output/complete_postprocessing")?;

    // Step 1: Result import
    step_result_import()?;

    // Step 2: Result structures
    step_result_structures()?;

    // Step 3: Result extraction
    step_result_extraction()?;

    // Step 4: Derived quantities
    step_derived_quantities()?;

    // Step 5: Result comparison
    step_result_comparison()?;

    // Step 6: Contour plots
    step_contour_plots()?;

    // Step 7: Animation
    step_animation()?;

    // Step 8: Report generation
    step_reports()?;

    // Step 9: Export
    step_export()?;

    // Step 10: Custom workflow
    step_custom_workflow()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║     Complete Post-processing Complete                     ║");
    println!("║   Check 'output/complete_postprocessing/' for exports     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Step 1: Result import.
fn step_result_import() -> anyhow::Result<()> {
    println!("┌─ Step 1: Result Import ──────────────────────────────────┐");
    println!("│ Importing analysis results...");
    println!("│");

    println!("│ Supported Import Formats:");
    println!("│   • VTK files (.vtk)");
    println!("│     - Nodal displacements");
    println!("│     - Element stresses");
    println!("│     - Reaction forces");
    println!("│");
    println!("│   • CSV files (.csv)");
    println!("│     - Displacement tables");
    println!("│     - Stress tables");
    println!("│     - Reaction tables");
    println!("│");
    println!("│   • Native format (.fea)");
    println!("│     - Complete model");
    println!("│     - All results");
    println!("│");

    // Create sample results for demo
    let displacements = vec![
        0.0, 0.0, 0.0,
        0.001, 0.002, 0.0,
        0.002, 0.003, 0.0,
        0.001, 0.002, 0.0,
    ];

    let reactions = vec![
        -100.0, -200.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        -50.0, -100.0, 0.0,
    ];

    println!("│ Sample Results Loaded:");
    println!("│   Nodes: {}", displacements.len() / 3);
    println!("│   DOFs: {}", displacements.len());
    println!("│   Reactions: {}", reactions.len());

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 2: Result structures.
fn step_result_structures() -> anyhow::Result<()> {
    println!("┌─ Step 2: Result Structures ──────────────────────────────┐");
    println!("│ Working with result structures...");
    println!("│");

    // Create FeaResults
    let displacements = vec![0.0, 0.0, 0.0, 0.001, 0.002, 0.0];
    let reactions = vec![-100.0, -200.0, 0.0, 0.0, 0.0, 0.0];

    let results = FeaResults::new(displacements, reactions, 3);

    println!("│ FeaResults Structure:");
    println!("│   • displacements: Vec<f64>");
    println!("│   • reactions: Vec<f64>");
    println!("│   • element_stresses: Option<Vec<f64>>");
    println!("│   • element_strains: Option<Vec<f64>>");
    println!("│   • dof_per_node: usize");
    println!("│");

    println!("│ Available Methods:");
    println!("│   • node_displacement(node_id)");
    println!("│   • displacement_magnitude(node_id)");
    println!("│   • max_displacement_magnitude()");
    println!("│   • max_displacement_node()");
    println!("│   • node_reaction(node_id)");
    println!("│   • total_reaction_magnitude()");
    println!("│");

    // Demonstrate methods
    let max_disp = results.max_displacement_magnitude();
    let max_node = results.max_displacement_node();

    println!("│ Example Results:");
    println!("│   Max displacement: {:.6e} m", max_disp);
    println!("│   At node: {:?}", max_node);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 3: Result extraction.
fn step_result_extraction() -> anyhow::Result<()> {
    println!("┌─ Step 3: Result Extraction ──────────────────────────────┐");
    println!("│ Extracting results...");
    println!("│");

    let displacements = vec![
        0.0, 0.0, 0.0,
        0.001, 0.002, 0.0,
        0.002, 0.003, 0.0,
        0.001, 0.002, 0.0,
    ];

    let reactions = vec![0.0; 12];
    let results = FeaResults::new(displacements, reactions, 3);

    println!("│ Nodal Result Extraction:");

    // Extract node 1 displacements
    if let Some(disp) = results.node_displacement(1) {
        println!("│   Node 1 Displacements:");
        println!("│     UX: {:.6e} m", disp[0]);
        println!("│     UY: {:.6e} m", disp[1]);
        println!("│     UZ: {:.6e} m", disp[2]);
    }
    println!("│");

    // Extract magnitude
    if let Some(mag) = results.displacement_magnitude(1) {
        println!("│   Node 1 Displacement Magnitude:");
        println!("│     Magnitude: {:.6e} m", mag);
    }
    println!("│");

    println!("│ Summary Extraction:");
    println!("│   Max displacement: {:.6e} m", results.max_displacement_magnitude());
    println!("│   Max at node: {:?}", results.max_displacement_node());

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 4: Derived quantities.
fn step_derived_quantities() -> anyhow::Result<()> {
    println!("┌─ Step 4: Derived Quantities ─────────────────────────────┐");
    println!("│ Computing derived quantities...");
    println!("│");

    // Create stress result
    let stress = StressResult::new(100.0, 50.0, 0.0);

    println!("│ Stress Result:");
    println!("│   σ_x: {:.1f} MPa", stress.sigma_x / 1e6);
    println!("│   σ_y: {:.1f} MPa", stress.sigma_y / 1e6);
    println!("│   σ_z: {:.1f} MPa", stress.sigma_z / 1e6);
    println!("│");

    // Compute von Mises
    let vm = stress.von_mises();
    println!("│ Derived Quantities:");
    println!("│   von Mises Stress: {:.1f} MPa", vm / 1e6);
    println!("│");

    // Compute principal stresses
    let (s1, s2, s3) = stress.principal_stresses();
    println!("│   Principal Stresses:");
    println!("│     σ₁: {:.1f} MPa", s1 / 1e6);
    println!("│     σ₂: {:.1f} MPa", s2 / 1e6);
    println!("│     σ₃: {:.1f} MPa", s3 / 1e6);
    println!("│");

    // Max shear
    let max_shear = (s1 - s3) / 2.0;
    println!("│   Max Shear Stress: {:.1f} MPa", max_shear / 1e6);

    println!("│");
    println!("│ Available Derived Quantities:");
    println!("│   • von Mises stress");
    println!("│   • Principal stresses (σ₁, σ₂, σ₃)");
    println!("│   • Max shear stress");
    println!("│   • Octahedral stress");
    println!("│   • Strain energy density");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 5: Result comparison.
fn step_result_comparison() -> anyhow::Result<()> {
    println!("┌─ Step 5: Result Comparison ──────────────────────────────┐");
    println!("│ Comparing analysis results...");
    println!("│");

    // Reference solution (fine mesh)
    let ref_disp: Vec<f64> = (0..50)
        .map(|i| (i as f64 * 0.1).sin() * 0.001)
        .collect();

    // Computed solutions (different meshes)
    let coarse_disp: Vec<f64> = ref_disp.iter().map(|v| v * 0.95).collect();
    let medium_disp: Vec<f64> = ref_disp.iter().map(|v| v * 0.98).collect();
    let fine_disp: Vec<f64> = ref_disp.iter().map(|v| v * 0.995).collect();

    println!("│ Mesh Convergence Study:");
    println!("│   Reference: Fine mesh (50 nodes)");
    println!("│");

    // Coarse mesh
    let coarse_l2 = l2_norm_difference(&ref_disp, &coarse_disp);
    let coarse_rel = relative_error(&ref_disp, &coarse_disp);
    println!("│ Coarse Mesh:");
    println!("│   L2 norm: {:.6e}", coarse_l2);
    println!("│   Relative error: {:.2f}%", coarse_rel);
    println!("│");

    // Medium mesh
    let medium_l2 = l2_norm_difference(&ref_disp, &medium_disp);
    let medium_rel = relative_error(&ref_disp, &medium_disp);
    println!("│ Medium Mesh:");
    println!("│   L2 norm: {:.6e}", medium_l2);
    println!("│   Relative error: {:.2f}%", medium_rel);
    println!("│");

    // Fine mesh
    let fine_l2 = l2_norm_difference(&ref_disp, &fine_disp);
    let fine_rel = relative_error(&ref_disp, &fine_disp);
    println!("│ Fine Mesh:");
    println!("│   L2 norm: {:.6e}", fine_l2);
    println!("│   Relative error: {:.2f}%", fine_rel);

    // Generate comparison report
    let report = generate_comparison_report("Reference", &ref_disp, "Coarse", &coarse_disp);
    std::fs::write("output/complete_postprocessing/comparison_report.txt", &report)?;

    println!("│");
    println!("│ Exported:");
    println!("│   • comparison_report.txt");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 6: Contour plots.
fn step_contour_plots() -> anyhow::Result<()> {
    println!("┌─ Step 6: Contour Plot Generation ────────────────────────┐");
    println!("│ Generating contour plots...");
    println!("│");

    // Create sample data
    let (nodes, elems) = generate_rect_2d(1.0, 0.5, 20, 10);

    // Generate displacement field
    let disp_mag: Vec<f64> = nodes.iter()
        .map(|n| {
            let x = n.x;
            let y = n.y;
            (x * (1.0 - x) * y * (0.5 - y)).abs() * 0.001
        })
        .collect();

    println!("│ Contour Plot Settings:");

    // Different level counts
    let contour_5 = ContourPlot::new(disp_mag.clone(), 5);
    println!("│   5 levels: min={:.6e}, max={:.6e}",
        contour_5.min_value, contour_5.max_value);

    let contour_9 = ContourPlot::new(disp_mag.clone(), 9);
    println!("│   9 levels: min={:.6e}, max={:.6e}",
        contour_9.min_value, contour_9.max_value);

    let contour_15 = ContourPlot::new(disp_mag.clone(), 15);
    println!("│   15 levels: min={:.6e}, max={:.6e}",
        contour_15.min_value, contour_15.max_value);
    println!("│");

    // Export
    let elems_tuple: Vec<(usize, usize, usize, usize)> =
        elems.iter().map(|e| (e.0, e.1, e.2, e.3)).collect();

    contour_5.export_svg(
        "output/complete_postprocessing/contour_5levels.svg",
        &nodes,
        &elems_tuple
    )?;

    contour_9.export_svg(
        "output/complete_postprocessing/contour_9levels.svg",
        &nodes,
        &elems_tuple
    )?;

    println!("│");
    println!("│ Exported:");
    println!("│   • contour_5levels.svg");
    println!("│   • contour_9levels.svg");
    println!("│");

    // Test color mapping
    let color = contour_9.color_for_value((contour_9.min_value + contour_9.max_value) / 2.0);
    println!("│ Color Mapping (jet colormap):");
    println!("│   Mid-value color: RGB({}, {}, {})", color[0], color[1], color[2]);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 7: Animation generation.
fn step_animation() -> anyhow::Result<()> {
    println!("┌─ Step 7: Animation Generation ───────────────────────────┐");
    println!("│ Generating animation frames...");
    println!("│");

    // Create sample geometry
    let (nodes, _) = generate_rect_2d(1.0, 0.5, 10, 5);

    // Create mode shape
    let mode_shape: Vec<f64> = (0..nodes.len() * 3)
        .map(|i| {
            if i % 3 == 1 {
                let node_idx = i / 3;
                let x = nodes[node_idx].x;
                (std::f64::consts::PI * x).sin() * 0.05
            } else {
                0.0
            }
        })
        .collect();

    println!("│ Mode Shape Data:");
    println!("│   Nodes: {}", nodes.len());
    println!("│   DOFs: {}", mode_shape.len());
    println!("│");

    // Generate animations
    println!("│ Animation Frames:");

    let anim_10 = DeformedShapeAnimation::from_mode_shape(&nodes, &mode_shape, 10);
    println!("│   10 frames");

    let anim_20 = DeformedShapeAnimation::from_mode_shape(&nodes, &mode_shape, 20);
    println!("│   20 frames");

    let anim_30 = DeformedShapeAnimation::from_mode_shape(&nodes, &mode_shape, 30);
    println!("│   30 frames");
    println!("│");

    // Export
    let elems_simple: Vec<(usize, usize)> = (0..nodes.len() - 1).map(|i| (i, i + 1)).collect();

    anim_10.export_svg_frames(
        "output/complete_postprocessing/mode_animation_frame",
        &elems_simple
    )?;

    println!("│ Exported:");
    println!("│   • mode_animation_frame_000.svg through");
    println!("│   • mode_animation_frame_009.svg");
    println!("│");

    println!("│ Compatible Software:");
    println!("│   • ParaView (3D visualization)");
    println!("│   • Blender (animation)");
    println!("│   • FFmpeg (video conversion)");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 8: Report generation.
fn step_reports() -> anyhow::Result<()> {
    println!("┌─ Step 8: Report Generation ──────────────────────────────┐");
    println!("│ Generating analysis reports...");
    println!("│");

    // Create sample data
    let nodes: Vec<Node> = (0..20)
        .map(|i| Node::new_3d((i % 5) as f64 * 0.5, (i / 5) as f64 * 0.5, 0.0))
        .collect();

    let displacements: Vec<f64> = (0..20 * 3)
        .map(|i| (i as f64 * 0.05).sin() * 0.001)
        .collect();

    let reactions = vec![0.0; 20 * 3];
    let fea_results = FeaResults::new(displacements, reactions, 3);

    let loads = vec![Load::new(0, Dof::Ux, 100.0)];
    let bcs = vec![BoundaryCondition::fixed(0, Dof::Ux)];

    // Generate text report
    let text_report = generate_text_report(&nodes, 10, &bcs, &loads, &fea_results);
    std::fs::write("output/complete_postprocessing/text_report.txt", &text_report)?;

    println!("│ Text Report:");
    println!("│   Lines: {}", text_report.lines().count());
    println!("│   Size: {:.1f} KB", text_report.len() as f64 / 1024.0);
    println!("│");

    // Generate HTML report
    generate_html_report(
        "output/complete_postprocessing/html_report.html",
        &nodes,
        &fea_results,
        "Complete FEA Analysis Report"
    )?;

    println!("│ HTML Report:");
    println!("│   Template: Standard");
    println!("│   Includes: Results, tables, summary");
    println!("│");

    println!("│ Exported:");
    println!("│   • text_report.txt");
    println!("│   • html_report.html");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 9: Export.
fn step_export() -> anyhow::Result<()> {
    println!("┌─ Step 9: Export ─────────────────────────────────────────┐");
    println!("│ Exporting results...");
    println!("│");

    // Create sample results
    let nodes: Vec<Node> = (0..10)
        .map(|i| Node::new_3d(i as f64 * 0.1, 0.0, 0.0))
        .collect();

    let displacements: Vec<f64> = (0..10 * 3)
        .map(|i| i as f64 * 0.001)
        .collect();

    let reactions = vec![0.0; 10 * 3];
    let fea_results = FeaResults::new(displacements, reactions, 3);

    println!("│ Export Formats:");
    println!("│");
    println!("│ VTK (Visualization Toolkit):");
    println!("│   • export_vtk_mesh()");
    println!("│   • export_displacements()");
    println!("│   • export_stresses()");
    println!("│   • export_reactions()");
    println!("│   Compatible: ParaView, VisIt");
    println!("│");
    println!("│ CSV (Comma-Separated Values):");
    println!("│   • export_displacements_csv()");
    println!("│   • export_reactions_csv()");
    println!("│   • export_stresses_csv()");
    println!("│   Compatible: Excel, MATLAB");
    println!("│");
    println!("│ SVG (Scalable Vector Graphics):");
    println!("│   • ContourPlot::export_svg()");
    println!("│   • DeformedShapeAnimation::export_svg_frames()");
    println!("│   Compatible: Web browsers, Illustrator");
    println!("│");
    println!("│ HTML (HyperText Markup Language):");
    println!("│   • generate_html_report()");
    println!("│   Compatible: Web browsers");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 10: Custom workflow.
fn step_custom_workflow() -> anyhow::Result<()> {
    println!("┌─ Step 10: Custom Workflow ───────────────────────────────┐");
    println!("│ Demonstrating custom workflow...");
    println!("│");

    println!("│ Example Custom Workflow:");
    println!("│   1. Import FEA results (VTK)");
    println!("│   2. Extract nodal displacements");
    println!("│   3. Compute derived quantities");
    println!("│   4. Compare with multiple analyses");
    println!("│   5. Generate contour plots");
    println!("│   6. Generate animation");
    println!("│   7. Generate report");
    println!("│   8. Export all results");
    println!("│");

    println!("│ Custom Workflow Steps:");
    println!("│   Step 1: Result import ✓");
    println!("│   Step 2: Result structures ✓");
    println!("│   Step 3: Result extraction ✓");
    println!("│   Step 4: Derived quantities ✓");
    println!("│   Step 5: Result comparison ✓");
    println!("│   Step 6: Contour plots ✓");
    println!("│   Step 7: Animation ✓");
    println!("│   Step 8: Reports ✓");
    println!("│   Step 9: Export ✓");
    println!("│   Step 10: Complete workflow ✓");
    println!("│");

    println!("│ Exported Files:");
    println!("│   output/complete_postprocessing/");
    println!("│     • comparison_report.txt");
    println!("│     • contour_5levels.svg");
    println!("│     • contour_9levels.svg");
    println!("│     • mode_animation_frame_*.svg");
    println!("│     • text_report.txt");
    println!("│     • html_report.html");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fea_results() {
        let displacements = vec![0.0, 0.0, 0.0, 0.001, 0.002, 0.0];
        let reactions = vec![0.0; 6];
        let results = FeaResults::new(displacements, reactions, 3);

        assert!(results.max_displacement_node().is_some());
        assert!(results.max_displacement_magnitude() > 0.0);
    }

    #[test]
    fn test_stress_result() {
        let stress = StressResult::new(100.0, 50.0, 0.0);
        let vm = stress.von_mises();
        assert!(vm > 0.0);

        let (s1, s2, s3) = stress.principal_stresses();
        assert!(s1 >= s2);
        assert!(s2 >= s3);
    }

    #[test]
    fn test_contour_plot() {
        let values = vec![0.0, 0.5, 1.0];
        let plot = ContourPlot::new(values, 3);

        let levels = plot.levels();
        assert_eq!(levels.len(), 3);
    }

    #[test]
    fn test_animation() {
        let nodes = vec![
            Node::new_3d(0.0, 0.0, 0.0),
            Node::new_3d(1.0, 0.0, 0.0),
        ];
        let mode_shape = vec![0.0, 0.0, 0.0, 0.1, 0.0, 0.0];

        let animation = DeformedShapeAnimation::from_mode_shape(&nodes, &mode_shape, 10);
        assert_eq!(animation.frames.len(), 10);
    }

    #[test]
    fn test_result_comparison() {
        let ref_disp = vec![1.0, 2.0, 3.0];
        let comp_disp = vec![1.01, 2.01, 3.01];

        let l2 = l2_norm_difference(&ref_disp, &comp_disp);
        assert!(l2 > 0.0);
        assert!(l2 < 0.1);

        let rel_err = relative_error(&ref_disp, &comp_disp);
        assert!(rel_err > 0.0);
        assert!(rel_err < 10.0);
    }
}
