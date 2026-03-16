//! Frictional Contact Analysis Example.
//!
//! This example demonstrates how to use frictional contact algorithms
//! for FEA simulations with Coulomb friction.

use fea::prelude::*;
use nalgebra::Vector3;

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Frictional Contact Analysis Example              ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Example 1: Coulomb friction model
    example_coulomb_friction()?;

    // Example 2: Penalty frictional contact
    example_penalty_friction()?;

    // Example 3: Augmented Lagrangian friction
    example_augmented_lagrangian_friction()?;

    // Example 4: Contact detection
    example_contact_detection()?;

    println!("\n=== Frictional Contact Examples Complete ===");
    Ok(())
}

/// Example 1: Coulomb friction model.
fn example_coulomb_friction() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 1: Coulomb Friction Model");
    println!("{}\n", "=".repeat(60));

    // Create Coulomb friction model
    let friction = CoulombFrictionModel::new(0.3, 0.25);

    println!("Friction Model Parameters:");
    println!("  Static coefficient: {}", friction.mu_static);
    println!("  Dynamic coefficient: {}", friction.mu_dynamic);
    println!();

    // Test friction at different velocities
    let velocities = vec![0.0, 0.005, 0.01, 0.05, 0.1, 1.0];
    println!("Friction Coefficient vs Velocity:");
    for v in velocities {
        let mu = friction.friction_coefficient(v);
        println!("  v = {:.3}: μ = {:.4}", v, mu);
    }
    println!();

    // Test stick/slip condition
    let normal_force = 100.0;
    let tangential_forces = vec![20.0, 25.0, 30.0, 35.0];

    println!("Stick/Slip Test (Normal force = {:.1} N):", normal_force);
    for ft in tangential_forces {
        let is_sticking = friction.is_sticking(ft, normal_force);
        let state = if is_sticking { "Sticking" } else { "Sliding" };
        println!("  F_t = {:.1} N: {}", ft, state);
    }
    println!();

    Ok(())
}

/// Example 2: Penalty frictional contact.
fn example_penalty_friction() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 2: Penalty Frictional Contact");
    println!("{}\n", "=".repeat(60));

    // Create penalty solver
    let solver = PenaltyFrictionSolver::new(1e6, 1e6);

    // Create contact constraint
    let normal = Vector3::new(0.0, 1.0, 0.0);
    let mut constraint = FrictionalContactConstraint::new(0, 0, normal);

    println!("Penalty Friction Solver:");
    println!("  Normal penalty: {:.2e}", solver.normal_penalty);
    println!("  Tangential penalty: {:.2e}", solver.tangential_penalty);
    println!();

    // Test case 1: Open contact (no penetration)
    println!("Test 1: Open Contact");
    let disp_open = Vector3::new(0.0, 0.01, 0.0);
    let force = solver.solve_constraint(&mut constraint, &disp_open);
    println!("  Gap: {:.6}", constraint.gap);
    println!("  State: {:?}", constraint.state);
    println!("  Normal force: {:.2} N", constraint.normal_force);
    println!("  Friction force: {:.2} N", constraint.friction_force);
    println!("  Contact force magnitude: {:.2} N", force.norm());
    println!();

    // Test case 2: Penetrating contact with sticking
    println!("Test 2: Penetrating Contact (Sticking)");
    let disp_penetrate = Vector3::new(0.0, -0.001, 0.0001);
    let force = solver.solve_constraint(&mut constraint, &disp_penetrate);
    println!("  Gap: {:.6}", constraint.gap);
    println!("  State: {:?}", constraint.state);
    println!("  Normal force: {:.2} N", constraint.normal_force);
    println!("  Friction force: {:.2} N", constraint.friction_force);
    println!("  Contact force magnitude: {:.2} N", force.norm());

    // Get stiffness contribution
    let k_contact = solver.assemble_stiffness_contribution(&constraint);
    println!("  Contact stiffness (diagonal): [{:.2e}, {:.2e}, {:.2e}]",
             k_contact[(0, 0)], k_contact[(1, 1)], k_contact[(2, 2)]);
    println!();

    // Test case 3: Sliding contact
    println!("Test 3: Sliding Contact");
    let disp_slide = Vector3::new(0.0, -0.001, 0.01);
    let _force = solver.solve_constraint(&mut constraint, &disp_slide);
    println!("  Gap: {:.6}", constraint.gap);
    println!("  Slip: {:.6}", constraint.slip);
    println!("  State: {:?}", constraint.state);
    println!("  Normal force: {:.2} N", constraint.normal_force);
    println!("  Friction force: {:.2} N", constraint.friction_force);
    println!();

    Ok(())
}

/// Example 3: Augmented Lagrangian frictional contact.
fn example_augmented_lagrangian_friction() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 3: Augmented Lagrangian Friction");
    println!("{}\n", "=".repeat(60));

    // Create augmented Lagrangian solver
    let mut solver = AugmentedLagrangianFrictionSolver::new(1e5, 1e5);

    // Create contact constraint
    let normal = Vector3::new(0.0, 1.0, 0.0);
    let mut constraint = FrictionalContactConstraint::new(0, 0, normal);

    println!("Augmented Lagrangian Solver:");
    println!("  Normal penalty: {:.2e}", solver.normal_penalty);
    println!("  Tangential penalty: {:.2e}", solver.tangential_penalty);
    println!();

    // Simulate multiple iterations (typical AL approach)
    println!("Iterative Solution Process:");
    let disp = Vector3::new(0.0, -0.001, 0.0);

    for iter in 1..=5 {
        let _force = solver.solve_constraint(&mut constraint, &disp);
        println!("  Iter {}: Normal={:.2} N, Friction={:.2} N, State={:?}",
                 iter, constraint.normal_force, constraint.friction_force, constraint.state);
    }
    println!();

    Ok(())
}

/// Example 4: Contact detection.
fn example_contact_detection() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 4: Contact Detection");
    println!("{}\n", "=".repeat(60));

    // Create contact detector
    let detector = NodeToSurfaceDetection::default();

    // Define nodes (potential contact points)
    let node_positions = vec![
        Vector3::new(0.0, 0.0, 0.0),       // On surface
        Vector3::new(0.5, -0.001, 0.0),    // Penetrating
        Vector3::new(0.5, 0.01, 0.0),      // Close but not contacting
        Vector3::new(0.5, 0.1, 0.0),       // Far from surface
    ];

    // Define target surface (single triangle)
    let target_triangles = vec![(0, 1, 2)];
    let target_positions = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.5, 0.0, 1.0),
    ];

    println!("Contact Detection Setup:");
    println!("  Nodes: {}", node_positions.len());
    println!("  Target triangles: {}", target_triangles.len());
    println!("  Search tolerance: {:.2e}", detector.search_tolerance);
    println!();

    // Detect contact
    let result = detector.detect(&node_positions, &target_triangles, &target_positions);

    println!("Detection Results:");
    println!("  Total potential contacts: {}", result.constraints.len());
    println!("  Active contacts (penetrating): {}", result.num_active);
    println!("  Maximum penetration: {:.6}", result.max_penetration);
    println!();

    if !result.constraints.is_empty() {
        println!("Contact Details:");
        for (i, c) in result.constraints.iter().enumerate() {
            println!("  Contact {}: Node {} -> Target {}, Gap={:.6}, State={:?}",
                     i + 1, c.node_id, c.target_id, c.gap, c.state);
        }
    }
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frictional_contact_examples() {
        example_coulomb_friction().unwrap();
        example_penalty_friction().unwrap();
        example_augmented_lagrangian_friction().unwrap();
        example_contact_detection().unwrap();
    }
}
