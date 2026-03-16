//! Adaptive Mesh Refinement and Mesh Quality Example.
//!
//! This example demonstrates:
//! - Mesh quality assessment
//! - Error estimation
//! - Adaptive mesh refinement
//! - Solution transfer between meshes

use fea::prelude::*;
use nalgebra::Point2;

fn main() -> anyhow::Result<()> {
    println!("=== Adaptive Mesh Refinement Example ===\n");

    // Part 1: Mesh Quality Assessment
    demonstrate_mesh_quality()?;

    // Part 2: Error Estimation
    demonstrate_error_estimation()?;

    // Part 3: Adaptive Refinement Loop
    demonstrate_adaptive_loop()?;

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Demonstrates mesh quality assessment
fn demonstrate_mesh_quality() -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Part 1: Mesh Quality Assessment");
    println!("─".repeat(60));

    let mq = MeshQuality::new();

    // Good mesh - uniform square elements
    println!("\n1. Good Quality Mesh (uniform squares):");
    let good_nodes = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(1.0, 1.0),
        Point2::new(0.0, 1.0),
    ];
    let good_elements = vec![(0, 1, 2, 3)];
    let good_quality = mq.compute_quality_2d(&good_nodes, &good_elements);

    println!("  Min quality: {:.4}", good_quality.min_quality);
    println!("  Avg quality: {:.4}", good_quality.avg_quality);
    println!("  Max quality: {:.4}", good_quality.max_quality);
    println!("  Poor elements: {}", good_quality.poor_elements);

    // Poor mesh - highly skewed element
    println!("\n2. Poor Quality Mesh (skewed):");
    let poor_nodes = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(1.1, 0.1),  // Very skewed
        Point2::new(0.0, 1.0),
    ];
    let poor_elements = vec![(0, 1, 2, 3)];
    let poor_quality = mq.compute_quality_2d(&poor_nodes, &poor_elements);

    println!("  Min quality: {:.4}", poor_quality.min_quality);
    println!("  Avg quality: {:.4}", poor_quality.avg_quality);
    println!("  Max quality: {:.4}", poor_quality.max_quality);
    println!("  Poor elements: {}", poor_quality.poor_elements);

    // Mesh with multiple elements
    println!("\n3. Multi-element Mesh (2x2):");
    let multi_nodes = vec![
        Point2::new(0.0, 0.0),
        Point2::new(0.5, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(0.0, 0.5),
        Point2::new(0.5, 0.5),
        Point2::new(1.0, 0.5),
        Point2::new(0.0, 1.0),
        Point2::new(0.5, 1.0),
        Point2::new(1.0, 1.0),
    ];
    let multi_elements = vec![
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (3, 4, 7, 6),
        (4, 5, 8, 7),
    ];
    let multi_quality = mq.compute_quality_2d(&multi_nodes, &multi_elements);

    println!("  Min quality: {:.4}", multi_quality.min_quality);
    println!("  Avg quality: {:.4}", multi_quality.avg_quality);
    println!("  Max quality: {:.4}", multi_quality.max_quality);
    println!("  Poor elements: {}", multi_quality.poor_elements);

    println!("\n  Mesh Quality Assessment: OK");
    Ok(())
}

/// Demonstrates error estimation
fn demonstrate_error_estimation() -> anyhow::Result<()> {
    println!("\n─".repeat(60));
    println!("Part 2: Error Estimation");
    println!("─".repeat(60));

    let est = ErrorEstimator::new();

    // Zienkiewicz-Zhu estimator
    println!("\n1. Zienkiewicz-Zhu Error Estimator:");

    let raw_stresses = vec![
        DVector::from_column_slice(&[100.0, 50.0, 25.0]),
        DVector::from_column_slice(&[110.0, 55.0, 27.0]),
        DVector::from_column_slice(&[105.0, 52.0, 26.0]),
        DVector::from_column_slice(&[150.0, 75.0, 37.0]),  // Higher stress region
    ];

    let smoothed_stresses = vec![
        DVector::from_column_slice(&[102.0, 51.0, 25.5]),
        DVector::from_column_slice(&[108.0, 54.0, 27.0]),
        DVector::from_column_slice(&[106.0, 53.0, 26.5]),
        DVector::from_column_slice(&[145.0, 72.0, 36.0]),
    ];

    let volumes = vec![1.0, 1.0, 1.0, 1.0];

    let error = est.zz_estimator(&raw_stresses, &smoothed_stresses, &volumes);

    println!("  Global error: {:.6e}", error.global_error);
    println!("  Energy norm error: {:.6e}", error.energy_norm_error);
    println!("  Element errors:");
    for (i, e) in error.element_errors.iter().enumerate() {
        println!("    Element {}: {:.6e}", i, e);
    }

    // Mark elements for refinement
    let marked = error.mark_for_refinement(0.5);
    println!("\n  Elements marked for refinement (threshold=50%): {:?}", marked);

    // Residual-based estimator
    println!("\n2. Residual-Based Error Estimator:");

    let displacements = vec![
        DVector::from_column_slice(&[0.0, 0.0]),
        DVector::from_column_slice(&[0.1, 0.05]),
        DVector::from_column_slice(&[0.2, 0.1]),
    ];

    let forces = vec![
        DVector::from_column_slice(&[0.0, 0.0]),
        DVector::from_column_slice(&[10.0, 5.0]),
        DVector::from_column_slice(&[20.0, 10.0]),
    ];

    // Simple stiffness matrix
    let k = DMatrix::from_row_slice(6, 6, &[
        100.0, 0.0, -10.0, 0.0, 0.0, 0.0,
        0.0, 100.0, 0.0, -10.0, 0.0, 0.0,
        -10.0, 0.0, 100.0, 0.0, -10.0, 0.0,
        0.0, -10.0, 0.0, 100.0, 0.0, -10.0,
        0.0, 0.0, -10.0, 0.0, 100.0, 0.0,
        0.0, 0.0, 0.0, -10.0, 0.0, 100.0,
    ]);

    let residual_error = est.residual_estimator(&displacements, &forces, &k);

    println!("  Global residual error: {:.6e}", residual_error.global_error);
    println!("  Element errors:");
    for (i, e) in residual_error.element_errors.iter().enumerate() {
        println!("    Element {}: {:.6e}", i, e);
    }

    println!("\n  Error Estimation: OK");
    Ok(())
}

/// Demonstrates adaptive refinement loop
fn demonstrate_adaptive_loop() -> anyhow::Result<()> {
    println!("\n─".repeat(60));
    println!("Part 3: Adaptive Refinement Loop");
    println!("─".repeat(60));

    // Initial coarse mesh
    let mut nodes = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(1.0, 1.0),
        Point2::new(0.0, 1.0),
    ];
    let mut elements = vec![(0, 1, 2, 3)];

    println!("\nInitial mesh:");
    println!("  Nodes: {}", nodes.len());
    println!("  Elements: {}", elements.len());

    // Create AMR controller
    let mut amr = AdaptiveMeshRefiner::new(4, 0.001)
        .with_strategy(RefinementStrategy::ErrorBased(0.5));

    // Simulated element errors (in practice, computed from solution)
    let mut errors = vec![1.0];

    // Adaptive refinement loop
    for iteration in 0..3 {
        println!("\n--- Iteration {} ---", iteration + 1);

        // Refine mesh
        let (new_nodes, new_elements) = amr.refine_quad_mesh(&nodes, &elements, &errors);

        println!("  Refined mesh:");
        println!("    Nodes: {} (was {})", new_nodes.len(), nodes.len());
        println!("    Elements: {} (was {})", new_elements.len(), elements.len());

        // Simulate new errors (would be computed from new solution)
        errors = vec![0.5; new_elements.len()];
        // Make one element have higher error
        if !errors.is_empty() {
            errors[errors.len() / 2] = 1.0;
        }

        // Update for next iteration
        nodes = new_nodes;
        elements = new_elements;

        // Check continuation
        let max_error = errors.iter().copied().fold(0.0, f64::max);
        if !amr.should_continue(max_error) {
            println!("  Convergence achieved!");
            break;
        }

        amr.increment_level();
    }

    // Mesh quality check after refinement
    let mq = MeshQuality::new();
    let quality = mq.compute_quality_2d(&nodes, &elements);

    println!("\nFinal mesh quality:");
    println!("  Min quality: {:.4}", quality.min_quality);
    println!("  Avg quality: {:.4}", quality.avg_quality);
    println!("  Poor elements: {}", quality.poor_elements);

    // Solution transfer demonstration
    println!("\n4. Solution Transfer:");

    // Coarse solution
    let coarse_nodes = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(0.0, 1.0),
    ];
    let coarse_solution = vec![1.0, 2.0, 3.0];

    // Fine nodes ( denser sampling)
    let fine_nodes = vec![
        Point2::new(0.0, 0.0),
        Point2::new(0.25, 0.0),
        Point2::new(0.5, 0.0),
        Point2::new(0.75, 0.0),
        Point2::new(1.0, 0.0),
    ];

    let fine_solution = SolutionInterpolator::interpolate_coarse_to_fine(
        &coarse_nodes, &coarse_solution, &fine_nodes,
    );

    println!("  Coarse solution: {:?}", coarse_solution);
    println!("  Fine solution: {:?}", fine_solution);

    // Project back to coarse
    let projected = SolutionInterpolator::project_fine_to_coarse(
        &fine_nodes, &fine_solution, &coarse_nodes,
    );

    println!("  Projected back: {:?}", projected);

    println!("\n  Adaptive Refinement Loop: OK");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amr_example() {
        demonstrate_mesh_quality().unwrap();
        demonstrate_error_estimation().unwrap();
        demonstrate_adaptive_loop().unwrap();
    }
}
