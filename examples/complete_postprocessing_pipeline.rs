//! Complete Post-processing Pipeline Example.
//!
//! This example demonstrates the COMPLETE post-processing workflow:
//! 1. Result structures
//! 2. VTK export (mesh, displacements, stresses)
//! 3. CSV export (tables)
//! 4. Contour plot generation (SVG)
//! 5. Animation generation (SVG frames)
//! 6. Result comparison
//! 7. Report generation (text, HTML)
//! 8. Validation

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
use std::fs::create_dir_all;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       Complete Post-Processing Pipeline                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    create_dir_all("output/postprocessing_pipeline")?;

    // Create sample results
    let (nodes, displacements, reactions, stresses) = create_sample_results();
    let fea_results = FeaResults::new(displacements.clone(), reactions.clone(), 3);

    // Step 1: Result structures
    step_result_structures(&fea_results)?;

    // Step 2: VTK export
    step_vtk_export(&nodes, &displacements, &stresses)?;

    // Step 3: CSV export
    step_csv_export(&nodes, &fea_results, &stresses)?;

    // Step 4: Contour plots
    step_contour_plots(&nodes, &displacements)?;

    // Step 5: Animation
    step_animation(&nodes)?;

    // Step 6: Result comparison
    step_result_comparison(&displacements)?;

    // Step 7: Report generation
    step_report_generation(&nodes, &fea_results)?;

    // Validation
    run_validation(&fea_results)?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║     Post-Processing Pipeline Complete                     ║");
    println!("║     Check 'output/postprocessing_pipeline/' for exports   ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Create sample results for demonstration.
fn create_sample_results() -> (
    Vec<Node>,
    Vec<f64>,
    Vec<f64>,
    Vec<StressResult>,
) {
    // Create nodes
    let nodes = vec![
        Node::new_3d(0.0, 0.0, 0.0),
        Node::new_3d(0.5, 0.0, 0.0),
        Node::new_3d(1.0, 0.0, 0.0),
        Node::new_3d(0.0, 0.5, 0.0),
        Node::new_3d(0.5, 0.5, 0.0),
        Node::new_3d(1.0, 0.5, 0.0),
        Node::new_3d(0.0, 1.0, 0.0),
        Node::new_3d(0.5, 1.0, 0.0),
        Node::new_3d(1.0, 1.0, 0.0),
    ];

    // Create sample displacements
    let displacements = vec![
        0.0, 0.0, 0.0,
        0.0001, 0.0002, 0.0,
        0.0002, 0.0003, 0.0,
        0.0002, 0.0001, 0.0,
        0.0003, 0.0003, 0.0,
        0.0004, 0.0004, 0.0,
        0.0003, 0.0, 0.0,
        0.0004, 0.0001, 0.0,
        0.0005, 0.0002, 0.0,
    ];

    // Create sample reactions
    let reactions = vec![
        -100.0, -200.0, 0.0,
        0.0, 0.0, 0.0,
        -50.0, -100.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        -50.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ];

    // Create sample stresses
    let stresses = vec![
        StressResult::new(100.0, 50.0, 0.0),
        StressResult::new(150.0, 75.0, 0.0),
        StressResult::new(120.0, 60.0, 0.0),
        StressResult::new(130.0, 65.0, 0.0),
        StressResult::new(140.0, 70.0, 0.0),
        StressResult::new(160.0, 80.0, 0.0),
        StressResult::new(110.0, 55.0, 0.0),
        StressResult::new(170.0, 85.0, 0.0),
        StressResult::new(180.0, 90.0, 0.0),
    ];

    (nodes, displacements, reactions, stresses)
}

/// Step 1: Result structures.
fn step_result_structures(results: &FeaResults) -> anyhow::Result<()> {
    println!("┌─ Step 1: Result Structures ──────────────────────────────┐");
    println!("│ FeaResults Structure:");
    println!("│   Displacements: {} values", results.displacements.len());
    println!("│   Reactions: {} values", results.reactions.len());
    println!("│   DOFs per node: {}", results.dof_per_node);

    let max_disp_node = results.max_displacement_node();
    let max_disp = results.max_displacement_magnitude();
    println!("│");
    println!("│ Maximum Displacement:");
    println!("│   Value: {:.6e} m", max_disp);
    println!("│   Node: {:?}", max_disp_node);

    println!("│");
    println!("│ Available Methods:");
    println!("│   • node_displacement(node_id)");
    println!("│   • displacement_magnitude(node_id)");
    println!("│   • max_displacement_magnitude()");
    println!("│   • max_displacement_node()");
    println!("│   • node_reaction(node_id)");
    println!("│   • total_reaction_magnitude()");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 2: VTK export.
fn step_vtk_export(
    nodes: &[Node],
    displacements: &[f64],
    stresses: &[StressResult],
) -> anyhow::Result<()> {
    println!("┌─ Step 2: VTK Export ─────────────────────────────────────┐");

    // Export displacements
    export_displacements(
        "output/postprocessing_pipeline/displacements.vtk",
        nodes,
        displacements,
    )?;
    println!("│ Exported: displacements.vtk");

    // Export stresses
    export_stresses(
        "output/postprocessing_pipeline/stresses.vtk",
        nodes,
        stresses,
    )?;
    println!("│ Exported: stresses.vtk");

    // Export reactions
    let reactions = vec![0.0; displacements.len()];
    let fea_results = FeaResults::new(displacements.to_vec(), reactions, 3);
    export_reactions(
        "output/postprocessing_pipeline/reactions.vtk",
        nodes,
        &fea_results,
    )?;
    println!("│ Exported: reactions.vtk");

    println!("│");
    println!("│ VTK Format: ParaView compatible");
    println!("│   - Node coordinates");
    println!("│   - Displacement vectors");
    println!("│   - Stress tensors");
    println!("│   - Reaction forces");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 3: CSV export.
fn step_csv_export(
    nodes: &[Node],
    results: &FeaResults,
    stresses: &[StressResult],
) -> anyhow::Result<()> {
    println!("┌─ Step 3: CSV Export ─────────────────────────────────────┐");

    // Export displacements
    export_displacements_csv(
        "output/postprocessing_pipeline/displacements.csv",
        nodes,
        results,
    )?;
    println!("│ Exported: displacements.csv");

    // Export reactions
    export_reactions_csv(
        "output/postprocessing_pipeline/reactions.csv",
        nodes,
        results,
    )?;
    println!("│ Exported: reactions.csv");

    // Export stresses
    let node_ids: Vec<usize> = (0..nodes.len()).collect();
    export_stresses_csv(
        "output/postprocessing_pipeline/stresses.csv",
        &node_ids,
        stresses,
    )?;
    println!("│ Exported: stresses.csv");

    println!("│");
    println!("│ CSV Format: Spreadsheet compatible");
    println!("│   - Displacements (UX, UY, UZ, Magnitude)");
    println!("│   - Reactions (RX, RY, RZ, Magnitude)");
    println!("│   - Stresses (SX, SY, SZ, SXY, SYZ, SXZ, VonMises)");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 4: Contour plots.
fn step_contour_plots(nodes: &[Node], displacements: &[f64]) -> anyhow::Result<()> {
    println!("┌─ Step 4: Contour Plots ──────────────────────────────────┐");

    // Create contour plot from displacement magnitude
    let disp_mag: Vec<f64> = (0..displacements.len() / 3)
        .map(|i| {
            let dx = displacements[i * 3];
            let dy = displacements[i * 3 + 1];
            let dz = displacements[i * 3 + 2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect();

    let contour = ContourPlot::new(disp_mag, 9);

    println!("│ Contour Plot:");
    println!("│   Min value: {:.6e}", contour.min_value);
    println!("│   Max value: {:.6e}", contour.max_value);
    println!("│   Levels: {}", contour.num_levels);

    let levels = contour.levels();
    println!("│   Level values:");
    for (i, level) in levels.iter().enumerate() {
        println!("│     Level {}: {:.6e}", i + 1, level);
    }

    // Test color mapping
    let color = contour.color_for_value((contour.min_value + contour.max_value) / 2.0);
    println!("│");
    println!("│ Color at midpoint: RGB({}, {}, {})", color[0], color[1], color[2]);

    // Export SVG (simplified - requires proper elements)
    let elements = vec![(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)];
    contour.export_svg(
        "output/postprocessing_pipeline/contour_displacement.svg",
        nodes,
        &elements,
    )?;
    println!("│");
    println!("│ Exported: contour_displacement.svg");

    println!("│");
    println!("│ Colormaps Available:");
    println!("│   • jet (default)");
    println!("│   • rainbow");
    println!("│   • hot");
    println!("│   • coolwarm");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 5: Animation.
fn step_animation(nodes: &[Node]) -> anyhow::Result<()> {
    println!("┌─ Step 5: Animation Generation ───────────────────────────┐");

    // Create sample mode shape
    let mode_shape: Vec<f64> = (0..nodes.len() * 3)
        .map(|i| {
            if i % 3 == 1 {
                (i as f64 * 0.1).sin() * 0.01
            } else {
                0.0
            }
        })
        .collect();

    // Create animation
    let animation = DeformedShapeAnimation::from_mode_shape(nodes, &mode_shape, 20);

    println!("│ Animation:");
    println!("│   Frames: {}", animation.frames.len());
    println!("│   Duration: 20 frames");
    println!("│   Scale range: [{:.2}, {:.2}]",
        animation.frames.iter().map(|f| f.scale_factor).fold(f64::INFINITY, f64::min),
        animation.frames.iter().map(|f| f.scale_factor).fold(0.0, f64::max),
    );

    // Export SVG frames
    let elements = vec![(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)];
    animation.export_svg_frames(
        "output/postprocessing_pipeline/animation_frame",
        &elements,
    )?;
    println!("│");
    println!("│ Exported: animation_frame_000.svg through");
    println!("│           animation_frame_019.svg");

    println!("│");
    println!("│ Usage: Import frames into animation software");
    println!("│   - Blender");
    println!("│   - Adobe After Effects");
    println!("│   - FFmpeg (for video conversion)");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 6: Result comparison.
fn step_result_comparison(displacements: &[f64]) -> anyhow::Result<()> {
    println!("┌─ Step 6: Result Comparison ──────────────────────────────┐");

    // Create reference and computed results
    let ref_disp = displacements.to_vec();
    let mut comp_disp = displacements.to_vec();

    // Add small perturbation to computed
    for val in &mut comp_disp {
        *val *= 1.001;
    }

    println!("│ Comparison: Reference vs Computed");
    println!("│   DOFs: {}", ref_disp.len());

    // L2 norm
    let l2 = l2_norm_difference(&ref_disp, &comp_disp);
    println!("│");
    println!("│ L2 Norm Difference: {:.6e}", l2);

    // Relative error
    let rel_err = relative_error(&ref_disp, &comp_disp);
    println!("│ Relative Error: {:.4}%", rel_err);

    // Find significant differences
    let differences = compare_displacements(&ref_disp, &comp_disp);
    println!("│");
    println!("│ Significant Differences (>1e-6): {}", differences.len());

    if !differences.is_empty() {
        println!("│   First 5 differences:");
        for (i, val_ref, val_comp, diff) in differences.iter().take(5) {
            println!("│     DOF {}: {:.6e} vs {:.6e} (diff: {:.6e})",
                i, val_ref, val_comp, diff);
        }
    }

    // Generate comparison report
    let report = generate_comparison_report(
        "Reference", &ref_disp,
        "Computed", &comp_disp,
    );
    std::fs::write(
        "output/postprocessing_pipeline/comparison_report.txt",
        &report,
    )?;
    println!("│");
    println!("│ Exported: comparison_report.txt");

    println!("│");
    println!("│ Comparison Metrics:");
    println!("│   • L2 norm difference");
    println!("│   • Relative error (%)");
    println!("│   • Point-by-point differences");
    println!("│   • Full comparison report");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Step 7: Report generation.
fn step_report_generation(nodes: &[Node], results: &FeaResults) -> anyhow::Result<()> {
    println!("┌─ Step 7: Report Generation ──────────────────────────────┐");

    // Create sample loads and BCs
    let loads = vec![Load::new(0, Dof::Uy, 100.0)];
    let bcs = vec![BoundaryCondition::fixed(0, Dof::Ux)];

    // Generate text report
    let text_report = generate_text_report(nodes, 10, &bcs, &loads, results);
    std::fs::write("output/postprocessing_pipeline/report.txt", &text_report)?;
    println!("│ Generated: report.txt (text format)");

    // Generate HTML report
    generate_html_report(
        "output/postprocessing_pipeline/report.html",
        nodes,
        results,
        "FEA Analysis Report",
    )?;
    println!("│ Generated: report.html (HTML format)");

    println!("│");
    println!("│ Report Contents:");
    println!("│   • Model summary (nodes, elements, BCs, loads)");
    println!("│   • Displacement summary (max, location)");
    println!("│   • Reaction summary");
    println!("│   • Convergence history (if available)");
    println!("│   • HTML: Interactive with styling");

    // Preview text report
    println!("│");
    println!("│ Text Report Preview (first 10 lines):");
    for line in text_report.lines().take(10) {
        println!("│   {}", line);
    }
    println!("│   ...");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Run validation.
fn run_validation(results: &FeaResults) -> anyhow::Result<()> {
    println!("┌─ Validation ─────────────────────────────────────────────┐");
    println!("│ Running validation checks...");

    // Validate result structures
    assert!(results.max_displacement_node().is_some(), "Max displacement node");
    assert!(results.max_displacement_magnitude() > 0.0, "Max displacement");
    println!("│   ✓ Result structures validated");

    // Validate stress result
    let stress = StressResult::new(100.0, 50.0, 0.0);
    let vm = stress.von_mises();
    assert!(vm > 0.0, "Von Mises stress");
    let (s1, s2, s3) = stress.principal_stresses();
    assert!(s1 >= s2, "Principal stress order");
    assert!(s2 >= s3, "Principal stress order");
    println!("│   ✓ Stress results validated");

    // Validate contour plot
    let values = vec![0.0, 0.5, 1.0];
    let plot = ContourPlot::new(values, 3);
    assert_eq!(plot.levels().len(), 3, "Contour levels");
    println!("│   ✓ Contour plots validated");

    // Validate result comparison
    let ref_disp = vec![1.0, 2.0, 3.0];
    let comp_disp = vec![1.001, 2.001, 3.001];
    let l2 = l2_norm_difference(&ref_disp, &comp_disp);
    assert!(l2 > 0.0 && l2 < 0.01, "L2 norm");
    println!("│   ✓ Result comparison validated");

    println!("│");
    println!("│ All validations passed ✓");
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
        assert!((levels[0] - 0.0).abs() < 1e-10);
        assert!((levels[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_result_comparison() {
        let ref_disp = vec![1.0, 2.0, 3.0];
        let comp_disp = vec![1.001, 2.001, 3.001];

        let l2 = l2_norm_difference(&ref_disp, &comp_disp);
        assert!(l2 > 0.0);
        assert!(l2 < 0.01);

        let rel_err = relative_error(&ref_disp, &comp_disp);
        assert!(rel_err > 0.0);
        assert!(rel_err < 1.0);
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
}
