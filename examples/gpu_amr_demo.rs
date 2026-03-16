//! GPU Adaptive Mesh Refinement (AMR) demonstration.
//!
//! This example demonstrates:
//! - Error estimation for mesh adaptation
//! - Gradient-based refinement criteria
//! - Solution-based mesh adaptation
//! - AMR iteration loops

use fea::gpu::gpu_amr::{
    GPUAdaptiveMeshRefinement, MeshElement, GradientRecovery,
    MeshInterpolation, run_amr_demo,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║        GPU Adaptive Mesh Refinement Demo                  ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Built-in demo
    run_amr_demo()?;

    // Error estimation demo
    demo_error_estimation()?;

    // Multi-level AMR demo
    demo_multi_level_amr()?;

    // Performance comparison
    demo_amr_performance()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Demo Complete                                ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demo: Error estimation
fn demo_error_estimation() -> anyhow::Result<()> {
    println!("┌─ Demo: Error Estimation ─────────────────────────────────┐");

    // Create simple mesh
    let mut amr = GPUAdaptiveMeshRefinement::new(0, 4);

    // Add elements with varying expected error
    for i in 0..20 {
        let expected_error = (i as f64 / 20.0).sin();
        amr.add_element(MeshElement {
            id: i,
            node_ids: vec![i, i + 1],
            error_indicator: expected_error,
            refinement_level: 0,
        });
    }

    // Simulate solution with high gradient in middle
    let solution: Vec<f64> = (0..21).map(|i| {
        if i < 10 { i as f64 * 0.1 } else { 1.0 + (i - 10) as f64 * 0.05 }
    }).collect();

    let gradient: Vec<f64> = (0..21).map(|i| {
        if i >= 9 && i <= 11 { 0.5 } else { 0.1 }
    }).collect();

    // Error estimation
    let errors = amr.estimate_error(&solution, &gradient);

    println!("│ Error Estimation Results:");
    println!("│   Min error: {:.4}", errors.iter().fold(f64::INFINITY, f64::min));
    println!("│   Max error: {:.4}", errors.iter().fold(f64::NEG_INFINITY, f64::max));
    println!("│   Avg error: {:.4}", errors.iter().sum::<f64>() / errors.len() as f64);

    // Mark for refinement
    let threshold = 0.3;
    let to_refine = amr.mark_for_refinement(&errors, threshold);

    println!("│");
    println!("│ Refinement (threshold = {:.2}):", threshold);
    println!("│   Elements to refine: {}", to_refine.len());
    println!("│   Refined elements: {:?}", to_refine.iter().take(10).collect::<Vec<_>>());

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo: Multi-level AMR
fn demo_multi_level_amr() -> anyhow::Result<()> {
    println!("┌─ Demo: Multi-Level AMR ──────────────────────────────────┐");

    let mut amr = GPUAdaptiveMeshRefinement::new(0, 5);

    // Initial coarse mesh
    for i in 0..10 {
        amr.add_element(MeshElement {
            id: i,
            node_ids: vec![i, i + 1],
            error_indicator: 0.0,
            refinement_level: 0,
        });
    }

    println!("│ AMR Iteration Loop:");
    println!("│");

    for iter in 0..4 {
        // Simulate gradient (high in middle region)
        let gradient: Vec<f64> = (0..11).map(|i| {
            if i >= 4 && i <= 6 { 1.0 } else { 0.2 }
        }).collect();

        let solution = vec![0.0f64; 11];
        let errors = amr.estimate_error(&solution, &gradient);

        let to_refine = amr.mark_for_refinement(&errors, 0.5);
        let to_derefine = amr.mark_for_derefine(&errors, 0.1);

        amr.refine_elements(&to_refine);
        amr.derefine_elements(&to_derefine);

        let stats = amr.get_statistics();

        println!("│ Iter {}: refine={}, derefine={}, total={}, avg_level={:.2}",
            iter + 1, to_refine.len(), to_derefine.len(),
            stats.total_elements, stats.average_level);
    }

    let stats = amr.get_statistics();
    println!("│");
    println!("│ Final Mesh Statistics:");
    println!("│   Total elements: {}", stats.total_elements);
    println!("│   Average level: {:.2}", stats.average_level);
    println!("│   Max level: {}", stats.max_level);
    println!("│   Level distribution: {:?}", stats.refinement_levels);

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo: Performance comparison
fn demo_amr_performance() -> anyhow::Result<()> {
    println!("┌─ Demo: AMR Performance ──────────────────────────────────┐");

    println!("│ Comparing Uniform vs Adaptive Refinement:");
    println!("│");

    // Uniform refinement (all elements refined)
    let uniform_elements = 10 * 2usize.pow(3); // 3 levels of uniform refinement
    println!("│ Uniform Refinement (3 levels):");
    println!("│   Elements: {}", uniform_elements);
    println!("│   DOFs: ~{}", uniform_elements * 2);

    // Adaptive refinement (only refine where needed)
    let refined_elements = 10 + 4 + 2; // Only refine high-error region
    println!("│");
    println!("│ Adaptive Refinement (3 levels):");
    println!("│   Elements: {}", refined_elements);
    println!("│   DOFs: ~{}", refined_elements * 2);

    let savings = (1.0 - refined_elements as f64 / uniform_elements as f64) * 100.0;
    println!("│");
    println!("│ Computational Savings:");
    println!("│   Element reduction: {:.1}%", savings);
    println!("│   Estimated time savings: {:.1}%", savings * 0.8); // Assume 80% linear scaling
    println!("│   Memory savings: {:.1}%", savings);

    println!("│");
    println!("│ Note: AMR is most beneficial for:");
    println!("│   • Localized phenomena (cracks, shocks)");
    println!("│   • Boundary layers");
    println!("│   • Moving fronts");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_error_estimation() {
        assert!(demo_error_estimation().is_ok());
    }

    #[test]
    fn test_demo_multi_level_amr() {
        assert!(demo_multi_level_amr().is_ok());
    }

    #[test]
    fn test_demo_amr_performance() {
        assert!(demo_amr_performance().is_ok());
    }
}
