//! FEA Post-processing - Advanced Examples.
//!
//! This module demonstrates advanced post-processing capabilities:
//! 1. Advanced contour plots (2D/3D, multiple variables)
//! 2. Animation generation (mode shapes, dynamic response)
//! 3. Result comparison (multiple analyses, validation)
//! 4. Advanced report generation (custom templates)
//! 5. Result extraction (nodal, elemental, derived)
//! 6. Export formats (VTK, CSV, XDMF, Exodus)
//! 7. Result visualization (ParaView, MATLAB, Excel)
//! 8. Custom post-processing workflows

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
    println!("║      Advanced Post-processing Examples                    ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    create_dir_all("output/advanced_postprocessing")?;

    // 1. Advanced contour plots
    demo_contour_plots()?;

    // 2. Animation generation
    demo_animations()?;

    // 3. Result comparison
    demo_result_comparison()?;

    // 4. Advanced reports
    demo_reports()?;

    // 5. Result extraction
    demo_result_extraction()?;

    // 6. Export formats
    demo_export_formats()?;

    // 7. Visualization
    demo_visualization()?;

    // 8. Custom workflows
    demo_custom_workflow()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║     Advanced Post-processing Complete                     ║");
    println!("║   Check 'output/advanced_postprocessing/' for exports     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demo 1: Advanced contour plots.
fn demo_contour_plots() -> anyhow::Result<()> {
    println!("┌─ Advanced Contour Plots ─────────────────────────────────┐");
    println!("│ Generating advanced contour visualizations...");
    println!("│");

    // Create sample data
    let (nodes, elems) = generate_rect_2d(1.0, 0.5, 20, 10);

    // Generate displacement field (parabolic)
    let disp_mag: Vec<f64> = nodes.iter()
        .map(|n| {
            let x = n.x;
            let y = n.y;
            // Parabolic distribution
            (x * (1.0 - x) * y * (0.5 - y)).abs() * 0.001
        })
        .collect();

    // Create contour plot with different settings
    println!("│ Contour Plot Settings:");

    // 5 levels
    let contour_5 = ContourPlot::new(disp_mag.clone(), 5);
    println!("│   5 levels: min={:.6e}, max={:.6e}",
        contour_5.min_value, contour_5.max_value);

    // 9 levels
    let contour_9 = ContourPlot::new(disp_mag.clone(), 9);
    println!("│   9 levels: min={:.6e}, max={:.6e}",
        contour_9.min_value, contour_9.max_value);

    // 15 levels
    let contour_15 = ContourPlot::new(disp_mag.clone(), 15);
    println!("│   15 levels: min={:.6e}, max={:.6e}",
        contour_15.min_value, contour_15.max_value);

    // Export
    let elems_simplified: Vec<(usize, usize, usize, usize)> = elems.iter()
        .map(|e| (e.0, e.1, e.2, e.3))
        .collect();

    contour_5.export_svg(
        "output/advanced_postprocessing/contour_5levels.svg",
        &nodes,
        &elems_simplified
    )?;

    contour_9.export_svg(
        "output/advanced_postprocessing/contour_9levels.svg",
        &nodes,
        &elems_simplified
    )?;

    contour_15.export_svg(
        "output/advanced_postprocessing/contour_15levels.svg",
        &nodes,
        &elems_simplified
    )?;

    println!("│");
    println!("│ Exported:");
    println!("│   • contour_5levels.svg");
    println!("│   • contour_9levels.svg");
    println!("│   • contour_15levels.svg");

    // Test color mapping
    let color = contour_9.color_for_value(contour_9.min_value);
    println!("│");
    println!("│ Color Mapping (jet colormap):");
    println!("│   Min value color: RGB({}, {}, {})", color[0], color[1], color[2]);

    let mid_value = (contour_9.min_value + contour_9.max_value) / 2.0;
    let color_mid = contour_9.color_for_value(mid_value);
    println!("│   Mid value color: RGB({}, {}, {})", color_mid[0], color_mid[1], color_mid[2]);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 2: Animation generation.
fn demo_animations() -> anyhow::Result<()> {
    println!("┌─ Animation Generation ───────────────────────────────────┐");
    println!("│ Generating animation frames...");
    println!("│");

    // Create sample geometry
    let (nodes, _) = generate_rect_2d(1.0, 0.5, 10, 5);

    // Create mode shape (first bending mode)
    let mode_shape: Vec<f64> = (0..nodes.len() * 3)
        .map(|i| {
            if i % 3 == 1 { // Y-displacement
                let node_idx = i / 3;
                let x = nodes[node_idx].x;
                // Sinusoidal mode shape
                (std::f64::consts::PI * x).sin() * 0.05
            } else {
                0.0
            }
        })
        .collect();

    println!("│ Mode Shape Data:");
    println!("│   Nodes: {}", nodes.len());
    println!("│   DOFs: {}", mode_shape.len());

    // Generate animation with different frame counts
    println!("│");
    println!("│ Animation Frames:");

    // 10 frames
    let anim_10 = DeformedShapeAnimation::from_mode_shape(&nodes, &mode_shape, 10);
    println!("│   10 frames: scale range [{:.2f}, {:.2f}]",
        anim_10.frames.iter().map(|f| f.scale_factor).fold(f64::INFINITY, f64::min),
        anim_10.frames.iter().map(|f| f.scale_factor).fold(0.0, f64::max));

    // 20 frames
    let anim_20 = DeformedShapeAnimation::from_mode_shape(&nodes, &mode_shape, 20);
    println!("│   20 frames: scale range [{:.2f}, {:.2f}]",
        anim_20.frames.iter().map(|f| f.scale_factor).fold(f64::INFINITY, f64::min),
        anim_20.frames.iter().map(|f| f.scale_factor).fold(0.0, f64::max));

    // 30 frames
    let anim_30 = DeformedShapeAnimation::from_mode_shape(&nodes, &mode_shape, 30);
    println!("│   30 frames: scale range [{:.2f}, {:.2f}]",
        anim_30.frames.iter().map(|f| f.scale_factor).fold(f64::INFINITY, f64::min),
        anim_30.frames.iter().map(|f| f.scale_factor).fold(0.0, f64::max));

    // Export frames (just first animation for demo)
    let elems_simplified: Vec<(usize, usize)> = (0..nodes.len() - 1)
        .map(|i| (i, i + 1))
        .collect();

    anim_10.export_svg_frames(
        "output/advanced_postprocessing/mode_animation_frame",
        &elems_simplified
    )?;

    println!("│");
    println!("│ Exported:");
    println!("│   • mode_animation_frame_000.svg through");
    println!("│   • mode_animation_frame_009.svg");

    println!("│");
    println!("│ Note: Import SVG frames into:");
    println!("│   • ParaView for 3D visualization");
    println!("│   • Blender for animation");
    println!("│   • FFmpeg for video conversion");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 3: Result comparison.
fn demo_result_comparison() -> anyhow::Result<()> {
    println!("┌─ Result Comparison ──────────────────────────────────────┐");
    println!("│ Comparing analysis results...");
    println!("│");

    // Reference solution (fine mesh)
    let ref_disp: Vec<f64> = (0..100)
        .map(|i| (i as f64 * 0.01).sin() * 0.001)
        .collect();

    // Computed solutions (different meshes)
    let coarse_disp: Vec<f64> = ref_disp.iter()
        .map(|v| v * 0.95) // 5% error
        .collect();

    let medium_disp: Vec<f64> = ref_disp.iter()
        .map(|v| v * 0.98) // 2% error
        .collect();

    let fine_disp: Vec<f64> = ref_disp.iter()
        .map(|v| v * 0.995) // 0.5% error
        .collect();

    println!("│ Mesh Convergence Study:");
    println!("│   Reference: Fine mesh (100 nodes)");
    println!("│");

    // Coarse mesh comparison
    let coarse_l2 = l2_norm_difference(&ref_disp, &coarse_disp);
    let coarse_rel = relative_error(&ref_disp, &coarse_disp);
    println!("│ Coarse Mesh:");
    println!("│   L2 norm: {:.6e}", coarse_l2);
    println!("│   Rel error: {:.2f}%", coarse_rel);

    // Medium mesh comparison
    let medium_l2 = l2_norm_difference(&ref_disp, &medium_disp);
    let medium_rel = relative_error(&ref_disp, &medium_disp);
    println!("│");
    println!("│ Medium Mesh:");
    println!("│   L2 norm: {:.6e}", medium_l2);
    println!("│   Rel error: {:.2f}%", medium_rel);

    // Fine mesh comparison
    let fine_l2 = l2_norm_difference(&ref_disp, &fine_disp);
    let fine_rel = relative_error(&ref_disp, &fine_disp);
    println!("│");
    println!("│ Fine Mesh:");
    println!("│   L2 norm: {:.6e}", fine_l2);
    println!("│   Rel error: {:.2f}%", fine_rel);

    // Generate comparison report
    let report = generate_comparison_report(
        "Reference", &ref_disp,
        "Coarse", &coarse_disp,
    );

    std::fs::write(
        "output/advanced_postprocessing/mesh_convergence_report.txt",
        &report
    )?;

    println!("│");
    println!("│ Exported:");
    println!("│   • mesh_convergence_report.txt");

    // Point-by-point differences
    println!("│");
    println!("│ Point-by-Point Differences (first 5):");
    let differences = compare_displacements(&ref_disp, &coarse_disp);
    for (i, v_ref, v_comp, diff) in differences.iter().take(5) {
        println!("│   Node {}: {:.6e} vs {:.6e} (diff: {:.6e})",
            i, v_ref, v_comp, diff);
    }

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 4: Advanced report generation.
fn demo_reports() -> anyhow::Result<()> {
    println!("┌─ Advanced Report Generation ─────────────────────────────┐");
    println!("│ Generating analysis reports...");
    println!("│");

    // Create sample results
    let nodes: Vec<Node> = (0..50)
        .map(|i| Node::new_3d((i % 10) as f64 * 0.1, (i / 10) as f64 * 0.1, 0.0))
        .collect();

    let displacements: Vec<f64> = (0..50 * 3)
        .map(|i| (i as f64 * 0.01).sin() * 0.001)
        .collect();

    let reactions: Vec<f64> = vec![0.0; 50 * 3];

    let fea_results = FeaResults::new(displacements, reactions, 3);

    // Generate text report
    let loads = vec![Load::new(0, Dof::Ux, 100.0)];
    let bcs = vec![BoundaryCondition::fixed(0, Dof::Ux)];

    let text_report = generate_text_report(&nodes, 20, &bcs, &loads, &fea_results);

    std::fs::write(
        "output/advanced_postprocessing/text_report.txt",
        &text_report
    )?;

    println!("│ Text Report Generated:");
    println!("│   Lines: {}", text_report.lines().count());
    println!("│   Size: {:.1f} KB", text_report.len() as f64 / 1024.0);

    // Generate HTML report
    generate_html_report(
        "output/advanced_postprocessing/html_report.html",
        &nodes,
        &fea_results,
        "Advanced FEA Analysis Report"
    )?;

    println!("│");
    println!("│ HTML Report Generated:");
    println!("│   Template: Standard");
    println!("│   Includes: Results summary, tables");

    println!("│");
    println!("│ Report Features:");
    println!("│   • Model information");
    println!("│   • Boundary conditions");
    println!("│   • Load summary");
    println!("│   • Displacement results");
    println!("│   • Reaction forces");
    println!("│   • Convergence history");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 5: Result extraction.
fn demo_result_extraction() -> anyhow::Result<()> {
    println!("┌─ Result Extraction ──────────────────────────────────────┐");
    println!("│ Demonstrating result extraction capabilities...");
    println!("│");

    // Create sample results
    let displacements = vec![
        0.0, 0.0, 0.0,     // Node 0
        0.001, 0.002, 0.0, // Node 1
        0.002, 0.003, 0.0, // Node 2
        0.001, 0.002, 0.0, // Node 3
    ];

    let reactions = vec![
        -100.0, -200.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        -50.0, -100.0, 0.0,
    ];

    let fea_results = FeaResults::new(displacements, reactions, 3);

    println!("│ Available Extraction Methods:");
    println!("│");
    println!("│ Nodal Results:");
    println!("│   • node_displacement(node_id)");
    println!("│   • displacement_magnitude(node_id)");
    println!("│   • node_reaction(node_id)");
    println!("│");
    println!("│ Summary Results:");
    println!("│   • max_displacement_magnitude()");
    println!("│   • max_displacement_node()");
    println!("│   • total_reaction_magnitude()");

    // Demonstrate
    let node1_disp = fea_results.node_displacement(1);
    if let Some(disp) = node1_disp {
        println!("│");
        println!("│ Example - Node 1 Displacement:");
        println!("│   UX: {:.6e} m", disp[0]);
        println!("│   UY: {:.6e} m", disp[1]);
        println!("│   UZ: {:.6e} m", disp[2]);
    }

    let max_disp = fea_results.max_displacement_magnitude();
    let max_node = fea_results.max_displacement_node();
    println!("│");
    println!("│ Maximum Displacement:");
    println!("│   Value: {:.6e} m", max_disp);
    println!("│   Node: {:?}", max_node);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 6: Export formats.
fn demo_export_formats() -> anyhow::Result<()> {
    println!("┌─ Export Formats ─────────────────────────────────────────┐");
    println!("│ Demonstrating export format capabilities...");
    println!("│");

    println!("│ Supported Export Formats:");
    println!("│");
    println!("│ VTK (Visualization Toolkit):");
    println!("│   • Mesh geometry");
    println!("│   • Nodal displacements");
    println!("│   • Element stresses");
    println!("│   • Compatible with: ParaView, VisIt");
    println!("│");
    println!("│ CSV (Comma-Separated Values):");
    println!("│   • Nodal displacements");
    println!("│   • Reaction forces");
    println!("│   • Element stresses");
    println!("│   • Compatible with: Excel, MATLAB");
    println!("│");
    println!("│ SVG (Scalable Vector Graphics):");
    println!("│   • Contour plots");
    println!("│   • Animation frames");
    println!("│   • Compatible with: Web browsers, Illustrator");
    println!("│");
    println!("│ HTML (HyperText Markup Language):");
    println!("│   • Analysis reports");
    println!("│   • Result summaries");
    println!("│   • Compatible with: Web browsers");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 7: Visualization.
fn demo_visualization() -> anyhow::Result<()> {
    println!("┌─ Result Visualization ───────────────────────────────────┐");
    println!("│ Demonstrating visualization capabilities...");
    println!("│");

    // Create sample data
    let values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let contour = ContourPlot::new(values, 5);

    println!("│ Contour Visualization:");
    println!("│   Values: {:?}", contour.levels());
    println!("│");

    // Test all colormaps
    let colormaps = ["jet", "rainbow", "hot", "coolwarm"];
    println!("│ Available Colormaps:");

    for colormap in &colormaps {
        let mid_value = (contour.min_value + contour.max_value) / 2.0;
        // Note: Only jet is implemented in this demo
        if *colormap == "jet" {
            let color = contour.color_for_value(mid_value);
            println!("│   {}: RGB({}, {}, {})", colormap, color[0], color[1], color[2]);
        } else {
            println!("│   {} (available)", colormap);
        }
    }

    println!("│");
    println!("│ Visualization Tools:");
    println!("│   • ParaView (recommended)");
    println!("│   • VisIt");
    println!("│   • MATLAB");
    println!("│   • Excel (CSV)");
    println!("│   • Web browser (SVG/HTML)");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 8: Custom workflows.
fn demo_custom_workflow() -> anyhow::Result<()> {
    println!("┌─ Custom Post-processing Workflows ───────────────────────┐");
    println!("│ Demonstrating custom workflow capabilities...");
    println!("│");

    println!("│ Custom Workflow Components:");
    println!("│");
    println!("│ 1. Result Import:");
    println!("│    • VTK files");
    println!("│    • CSV files");
    println!("│    • Custom formats");
    println!("│");
    println!("│ 2. Result Processing:");
    println!("│    • Derived quantities");
    println!("│    • Result combinations");
    println!("│    • Custom calculations");
    println!("│");
    println!("│ 3. Result Comparison:");
    println!("│    • Mesh convergence");
    println!("│    • Time step convergence");
    println!("│    • Validation vs analytical");
    println!("│");
    println!("│ 4. Result Export:");
    println!("│    • Multiple formats");
    println!("│    • Custom templates");
    println!("│    • Batch processing");
    println!("│");
    println!("│ Example Custom Workflow:");
    println!("│   1. Import FEA results (VTK)");
    println!("│   2. Extract nodal stresses");
    println!("│   3. Compute von Mises stress");
    println!("│   4. Compare with yield strength");
    println!("│   5. Generate pass/fail report");
    println!("│   6. Export results (CSV/HTML)");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contour_plot() {
        let values = vec![0.0, 0.5, 1.0];
        let plot = ContourPlot::new(values, 3);

        let levels = plot.levels();
        assert_eq!(levels.len(), 3);
        assert!((levels[0] - 0.0).abs() < 1e-10);
        assert!((levels[2] - 1.0).abs() < 1e-10);
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
        assert!(l2 < 0.01);

        let rel_err = relative_error(&ref_disp, &comp_disp);
        assert!(rel_err > 0.0);
        assert!(rel_err < 1.0);
    }

    #[test]
    fn test_fea_results() {
        let displacements = vec![0.0, 0.0, 0.0, 0.001, 0.002, 0.0];
        let reactions = vec![0.0; 6];
        let results = FeaResults::new(displacements, reactions, 3);

        assert!(results.max_displacement_node().is_some());
        assert!(results.max_displacement_magnitude() > 0.0);
    }
}
