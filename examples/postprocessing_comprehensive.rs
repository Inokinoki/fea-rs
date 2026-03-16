//! Post-processing Comprehensive Example.
//!
//! This example demonstrates all post-processing capabilities:
//! - Result structures
//! - VTK export
//! - CSV export
//! - Contour plots
//! - Animations
//! - Result comparison

use fea::prelude::*;
use fea::postprocessing::{
    FeaResults, StressResult, StrainResult,
    vtk_export::{export_displacements, export_stresses, export_reactions},
    csv_export::{export_displacements_csv, export_reactions_csv, export_stresses_csv},
    report_generation::{generate_text_report, generate_html_report},
    visualization::create_color_map,
    advanced::{
        ContourPlot, DeformedShapeAnimation,
        result_comparison::{compare_displacements, l2_norm_difference, relative_error, generate_comparison_report},
    },
};
use std::fs::create_dir_all;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       Post-Processing Comprehensive Example               ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Create output directory
    create_dir_all("output/postprocessing")?;

    // Demo 1: Result structures
    demo_result_structures()?;

    // Demo 2: VTK export
    demo_vtk_export()?;

    // Demo 3: CSV export
    demo_csv_export()?;

    // Demo 4: Report generation
    demo_report_generation()?;

    // Demo 5: Contour plots
    demo_contour_plots()?;

    // Demo 6: Animation
    demo_animation()?;

    // Demo 7: Result comparison
    demo_result_comparison()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║        Results saved in 'output/postprocessing/'         ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demo 1: Result structures.
fn demo_result_structures() -> anyhow::Result<()> {
    println!("┌─ Result Structures ──────────────────────────────────────┐");

    // Create FeaResults
    let displacements = vec![0.0, 0.0, 0.0, 0.001, 0.002, 0.0, 0.003, 0.004, 0.0];
    let reactions = vec![-100.0, -200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let results = FeaResults::new(displacements, reactions, 3);

    println!("│ FeaResults:");
    println!("│   Displacements: {} values", results.displacements.len());
    println!("│   Reactions: {} values", results.reactions.len());
    println!("│   DOFs per node: {}", results.dof_per_node);

    // Max displacement
    let max_disp_node = results.max_displacement_node();
    let max_disp = results.max_displacement_magnitude();
    println!("│");
    println!("│ Max displacement: {:.6e} m at node {:?}", max_disp, max_disp_node);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 2: VTK export.
fn demo_vtk_export() -> anyhow::Result<()> {
    println!("┌─ VTK Export ─────────────────────────────────────────────┐");

    let nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(1.0, 1.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
    ];

    let elements = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ];

    let displacements = vec![
        0.0, 0.0, 0.0,
        0.001, 0.002, 0.0,
        0.002, 0.003, 0.0,
        0.001, 0.002, 0.0,
    ];

    // Export mesh
    export_displacements("output/postprocessing/demo_displacements.vtk", &nodes, &displacements)?;
    println!("│ Exported: output/postprocessing/demo_displacements.vtk");

    // Export stresses (example)
    let stresses = vec![
        StressResult::new(100.0, 50.0, 0.0),
        StressResult::new(150.0, 75.0, 0.0),
        StressResult::new(120.0, 60.0, 0.0),
        StressResult::new(130.0, 65.0, 0.0),
    ];

    export_stresses("output/postprocessing/demo_stresses.vtk", &nodes, &stresses)?;
    println!("│ Exported: output/postprocessing/demo_stresses.vtk");

    // Export reactions
    let reactions = vec![
        -100.0, -200.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        -50.0, -100.0, 0.0,
    ];

    let fea_results = FeaResults::new(displacements.clone(), reactions, 3);
    export_reactions("output/postprocessing/demo_reactions.vtk", &nodes, &fea_results)?;
    println!("│ Exported: output/postprocessing/demo_reactions.vtk");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 3: CSV export.
fn demo_csv_export() -> anyhow::Result<()> {
    println!("┌─ CSV Export ─────────────────────────────────────────────┐");

    let nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(1.0, 1.0, 0.0),
    ];

    let displacements = vec![
        0.0, 0.0, 0.0,
        0.001, 0.002, 0.0,
        0.002, 0.003, 0.0,
    ];

    let reactions = vec![
        -100.0, -200.0, 0.0,
        0.0, 0.0, 0.0,
        -50.0, -100.0, 0.0,
    ];

    let fea_results = FeaResults::new(displacements, reactions, 3);

    // Export displacements CSV
    export_displacements_csv("output/postprocessing/displacements.csv", &nodes, &fea_results)?;
    println!("│ Exported: output/postprocessing/displacements.csv");

    // Export reactions CSV
    export_reactions_csv("output/postprocessing/reactions.csv", &nodes, &fea_results)?;
    println!("│ Exported: output/postprocessing/reactions.csv");

    // Export stresses CSV
    let node_ids = vec![0, 1, 2];
    let stresses = vec![
        StressResult::new(100.0, 50.0, 0.0),
        StressResult::new(150.0, 75.0, 0.0),
        StressResult::new(120.0, 60.0, 0.0),
    ];
    export_stresses_csv("output/postprocessing/stresses.csv", &node_ids, &stresses)?;
    println!("│ Exported: output/postprocessing/stresses.csv");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 4: Report generation.
fn demo_report_generation() -> anyhow::Result<()> {
    println!("┌─ Report Generation ──────────────────────────────────────┐");

    let nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(1.0, 1.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
    ];

    let elements = vec![(0, 1), (1, 2), (2, 3), (3, 0)];

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

    let loads = vec![Load::new(2, Dof::Ux, 100.0)];
    let bcs = vec![BoundaryCondition::fixed(0, Dof::Ux)];

    let fea_results = FeaResults::new(displacements, reactions, 3);

    // Generate text report
    let text_report = generate_text_report(&nodes, elements.len(), &bcs, &loads, &fea_results);
    std::fs::write("output/postprocessing/report.txt", &text_report)?;
    println!("│ Generated: output/postprocessing/report.txt");

    // Generate HTML report
    generate_html_report("output/postprocessing/report.html", &nodes, &fea_results, "Demo Analysis")?;
    println!("│ Generated: output/postprocessing/report.html");

    // Preview text report
    println!("│");
    println!("│ Text Report Preview:");
    for line in text_report.lines().take(10) {
        println!("│   {}", line);
    }
    println!("│   ...");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 5: Contour plots.
fn demo_contour_plots() -> anyhow::Result<()> {
    println!("┌─ Contour Plots ──────────────────────────────────────────┐");

    // Create sample values
    let values = vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.3, 0.6, 0.9];
    let contour = ContourPlot::new(values.clone(), 5);

    println!("│ Contour Plot:");
    println!("│   Min value: {:.2}", contour.min_value);
    println!("│   Max value: {:.2}", contour.max_value);
    println!("│   Levels: {}", contour.num_levels);

    let levels = contour.levels();
    println!("│   Level values: {:?}", levels);

    // Test color mapping
    let color = contour.color_for_value(0.5);
    println!("│   Color at 0.5: RGB({}, {}, {})", color[0], color[1], color[2]);

    // Export SVG contour plot (simplified - needs proper elements)
    let nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(1.0, 1.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
    ];

    let elements = vec![(0, 1, 2, 3)];
    contour.export_svg("output/postprocessing/contour.svg", &nodes, &elements)?;
    println!("│");
    println!("│ Exported: output/postprocessing/contour.svg");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 6: Animation.
fn demo_animation() -> anyhow::Result<()> {
    println!("┌─ Animation Generation ───────────────────────────────────┐");

    let nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(1.0, 1.0, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
    ];

    // Mode shape (simplified)
    let mode_shape = vec![
        0.0, 0.0, 0.0,
        0.1, 0.0, 0.0,
        0.1, 0.1, 0.0,
        0.0, 0.1, 0.0,
    ];

    // Create animation
    let animation = DeformedShapeAnimation::from_mode_shape(&nodes, &mode_shape, 10);

    println!("│ Animation:");
    println!("│   Frames: {}", animation.frames.len());
    println!("│   First frame scale: {:.2}", animation.frames[0].scale_factor);
    println!("│   Last frame scale: {:.2}", animation.frames.last().unwrap().scale_factor);

    // Export SVG frames
    let elements = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
    animation.export_svg_frames("output/postprocessing/animation_frame", &elements)?;
    println!("│");
    println!("│ Exported: output/postprocessing/animation_frame_000.svg through");
    println!("│           output/postprocessing/animation_frame_009.svg");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 7: Result comparison.
fn demo_result_comparison() -> anyhow::Result<()> {
    println!("┌─ Result Comparison ──────────────────────────────────────┐");

    // Reference results
    let ref_disp = vec![0.0, 0.0, 0.0, 0.001, 0.002, 0.0, 0.002, 0.003, 0.0];

    // Computed results (slightly different)
    let comp_disp = vec![0.0, 0.0, 0.0, 0.00101, 0.00201, 0.0, 0.00201, 0.00301, 0.0];

    println!("│ Comparison:");
    println!("│   Reference DOFs: {}", ref_disp.len());
    println!("│   Computed DOFs: {}", comp_disp.len());

    // L2 norm difference
    let l2 = l2_norm_difference(&ref_disp, &comp_disp);
    println!("│");
    println!("│ L2 Norm Difference: {:.6e}", l2);

    // Relative error
    let rel_err = relative_error(&ref_disp, &comp_disp);
    println!("│ Relative Error: {:.4}%", rel_err);

    // Find significant differences
    let differences = compare_displacements(&ref_disp, &comp_disp);
    println!("│");
    println!("│ Significant differences (>1e-6): {}", differences.len());
    for (i, val_ref, val_comp, diff) in &differences {
        println!("│   DOF {}: {:.6e} vs {:.6e} (diff: {:.6e})", i, val_ref, val_comp, diff);
    }

    // Generate comparison report
    let report = generate_comparison_report(
        "Reference", &ref_disp,
        "Computed", &comp_disp,
    );

    std::fs::write("output/postprocessing/comparison_report.txt", &report)?;
    println!("│");
    println!("│ Exported: output/postprocessing/comparison_report.txt");

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
    fn test_result_comparison() {
        let ref_disp = vec![1.0, 2.0, 3.0];
        let comp_disp = vec![1.001, 2.001, 3.001];

        let l2 = l2_norm_difference(&ref_disp, &comp_disp);
        assert!(l2 > 0.0);
        assert!(l2 < 0.01);
    }
}
