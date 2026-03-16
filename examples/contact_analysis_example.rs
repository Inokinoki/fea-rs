//! Contact Analysis Example.
//!
//! This example demonstrates:
//! - Contact detection algorithms
//! - Penalty method for contact
//! - Augmented Lagrangian method
//! - Contact constraint enforcement

use fea::prelude::*;
use nalgebra::Point2;

fn main() -> anyhow::Result<()> {
    println!("=== Contact Analysis Example ===\n");

    // Part 1: Contact Detection
    demonstrate_contact_detection()?;

    // Part 2: Penalty Method
    demonstrate_penalty_method()?;

    // Part 3: Augmented Lagrangian
    demonstrate_augmented_lagrangian()?;

    // Part 4: Contact with Constraint Handler
    demonstrate_constraint_handler()?;

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Demonstrates contact detection
fn demonstrate_contact_detection() -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Part 1: Contact Detection");
    println!("─".repeat(60));

    let params = ContactParameters::default();
    let detector = NodeToSurfaceContact::new(params);

    // Master surface: horizontal line at y=0
    let master_nodes = vec![
        Point2::new(0.0, 0.0),
        Point2::new(0.5, 0.0),
        Point2::new(1.0, 0.0),
    ];
    let master_segments = vec![(0, 1), (1, 2)];

    println!("\nMaster surface: 3 nodes, 2 segments");
    println!("  Segment 0: ({}, {}) to ({}, {})",
             master_nodes[0].x, master_nodes[0].y,
             master_nodes[1].x, master_nodes[1].y);
    println!("  Segment 1: ({}, {}) to ({}, {})",
             master_nodes[1].x, master_nodes[1].y,
             master_nodes[2].x, master_nodes[2].y);

    // Slave nodes at various distances
    let slave_nodes = vec![
        Point2::new(0.25, 0.001),   // Very close (in contact)
        Point2::new(0.5, 0.01),     // Close
        Point2::new(0.75, 0.1),     // Far
        Point2::new(0.5, -0.001),   // Penetrating
    ];

    println!("\nSlave nodes:");
    for (i, node) in slave_nodes.iter().enumerate() {
        println!("  Node {}: ({:.3}, {:.3})", i, node.x, node.y);
    }

    let detection = detector.detect_2d(&slave_nodes, &master_segments, &master_nodes);

    println!("\nDetection Results:");
    println!("  Total contact pairs: {}", detection.pairs.len());
    println!("  Active contacts: {}", detection.num_active);
    println!("  Max penetration: {:.6}", detection.max_penetration);

    println!("\nContact Pairs:");
    for (i, pair) in detection.pairs.iter().enumerate() {
        println!("  Pair {}: slave={}, master={}, gap={:.6}",
                 i, pair.slave_id, pair.master_id, pair.gap);
    }

    println!("\n  Contact Detection: OK");
    Ok(())
}

/// Demonstrates penalty contact method
fn demonstrate_penalty_method() -> anyhow::Result<()> {
    println!("\n─".repeat(60));
    println!("Part 2: Penalty Contact Method");
    println!("─".repeat(60));

    // Create penalty contact with stiffness and no friction
    let penalty = PenaltyContact::new(1e8, 0.0);

    println!("\nPenalty Parameters:");
    println!("  Stiffness: {:.2e}", penalty.stiffness);
    println!("  Friction coefficient: {}", penalty.friction);

    // Test with various gap values
    let test_gaps = vec![-0.001, -0.0001, 0.0, 0.001];

    println!("\nContact Force vs Gap:");
    println!("  {:<15} | {:<15} | {:<15}", "Gap", "Normal Force", "Status");
    println!("  {}", "-".repeat(50));

    for gap in test_gaps {
        let pair = ContactPair {
            master_id: 0,
            slave_id: 0,
            normal: DVector::from_column_slice(&[0.0, 1.0]),
            gap,
            stiffness: penalty.stiffness,
        };

        let (force, _friction) = penalty.compute_force_2d(&pair, 0.0);
        let status = if gap < 0.0 { "Contact" } else { "Open" };

        println!("  {:<15.6} | {:<15.2} | {:<15}", gap, force.norm(), status);
    }

    // Test with friction
    let penalty_friction = PenaltyContact::new(1e8, 0.3);

    println!("\nWith Friction (μ=0.3):");
    let pair = ContactPair {
        master_id: 0,
        slave_id: 0,
        normal: DVector::from_column_slice(&[0.0, 1.0]),
        gap: -0.001,
        stiffness: 1e8,
    };

    let tangential_displacements = vec![0.0, 0.0001, 0.001, 0.01];

    println!("  {:<15} | {:<15}", "Tangential Disp", "Friction Force");
    println!("  {}", "-".repeat(35));

    for disp in tangential_displacements {
        let (_, friction) = penalty_friction.compute_force_2d(&pair, disp);
        println!("  {:<15.6} | {:<15.2}", disp, friction);
    }

    println!("\n  Penalty Method: OK");
    Ok(())
}

/// Demonstrates augmented Lagrangian method
fn demonstrate_augmented_lagrangian() -> anyhow::Result<()> {
    println!("\n─".repeat(60));
    println!("Part 3: Augmented Lagrangian Method");
    println!("─".repeat(60));

    let mut al = AugmentedLagrangianContact::new(1e8);

    println!("\nAugmented Lagrangian Parameters:");
    println!("  Stiffness: {:.2e}", al.stiffness);
    println!("  Update parameter: {}", al.update_param);

    // Create contact pairs with penetration
    let mut pairs = vec![
        ContactPair {
            master_id: 0,
            slave_id: 0,
            normal: DVector::from_column_slice(&[0.0, 1.0]),
            gap: -0.001,
            stiffness: 1e8,
        },
        ContactPair {
            master_id: 1,
            slave_id: 1,
            normal: DVector::from_column_slice(&[1.0, 0.0]),
            gap: -0.0005,
            stiffness: 1e8,
        },
    ];

    al.init_multipliers(pairs.len());

    println!("\nIteration History:");
    println!("  {:<8} | {:<15} | {:<15} | {:<15}",
             "Iter", "Multiplier 1", "Multiplier 2", "Total Force");
    println!("  {}", "-".repeat(60));

    for iter in 0..5 {
        // Update multipliers
        al.update_multipliers(&pairs);

        // Compute forces
        let force1 = al.compute_augmented_force(&pairs[0], 0);
        let force2 = al.compute_augmented_force(&pairs[1], 1);

        let mult1 = if iter < al.multipliers.len() { al.multipliers[iter % al.multipliers.len()] } else { 0.0 };
        let mult2 = if iter + 1 < al.multipliers.len() { al.multipliers[iter + 1] } else { 0.0 };

        println!("  {:<8} | {:<15.4} | {:<15.4} | {:<15.4}",
                 iter, mult1, mult2, force1.norm() + force2.norm());
    }

    // Demonstrate contact release
    println!("\nContact Release Test:");
    pairs[0].gap = 0.001; // Open the contact
    al.update_multipliers(&pairs);

    let force_released = al.compute_augmented_force(&pairs[0], 0);
    println!("  Gap after release: {:.6}", pairs[0].gap);
    println!("  Force after release: {:.6}", force_released.norm());

    println!("\n  Augmented Lagrangian: OK");
    Ok(())
}

/// Demonstrates constraint handler
fn demonstrate_constraint_handler() -> anyhow::Result<()> {
    println!("\n─".repeat(60));
    println!("Part 4: Contact Constraint Handler");
    println!("─".repeat(60));

    let mut handler = ContactConstraintHandler::new(1e-6);

    // Simulate contact detection results
    let detection = ContactDetection {
        pairs: vec![
            ContactPair {
                master_id: 0,
                slave_id: 2,
                normal: DVector::from_column_slice(&[0.0, 1.0]),
                gap: -0.001,
                stiffness: 1e8,
            },
            ContactPair {
                master_id: 1,
                slave_id: 3,
                normal: DVector::from_column_slice(&[0.0, 1.0]),
                gap: -0.0005,
                stiffness: 1e8,
            },
            ContactPair {
                master_id: 2,
                slave_id: 4,
                normal: DVector::from_column_slice(&[1.0, 0.0]),
                gap: 0.01, // Open - should not be active
                stiffness: 1e8,
            },
        ],
        num_active: 2,
        max_penetration: 0.001,
    };

    println!("\nInitial Detection:");
    println!("  Total pairs: {}", detection.pairs.len());
    println!("  Active contacts: {}", detection.num_active);

    handler.update_active_set(&detection);

    println!("\nAfter Constraint Handler:");
    println!("  Active pairs: {}", handler.active_pairs.len());

    for (i, pair) in handler.active_pairs.iter().enumerate() {
        println!("    Pair {}: slave={}, gap={:.6}", i, pair.slave_id, pair.gap);
    }

    // Apply constraints to a simple system
    let dof_per_node = 2;
    let n_nodes = 5;
    let n_dof = n_nodes * dof_per_node;

    let mut k = DMatrix::identity(n_dof, n_dof);
    let mut f = DVector::from_element(n_dof, 1.0);

    // Scale diagonal to represent actual stiffness
    for i in 0..n_dof {
        k[(i, i)] = 1e6;
    }

    println!("\nOriginal System:");
    println!("  DOFs: {}", n_dof);
    println!("  Stiffness matrix: {}x{}", k.nrows(), k.ncols());

    let (k_ext, f_ext) = handler.apply_lagrange_constraints(&mut k, &mut f, dof_per_node);

    println!("\nExtended System (with Lagrange multipliers):");
    println!("  DOFs: {} (original + {} constraints)", k_ext.nrows(), k_ext.nrows() - n_dof);
    println!("  Stiffness matrix: {}x{}", k_ext.nrows(), k_ext.ncols());

    // Verify constraint equations are added
    println!("\nConstraint Equations:");
    for (c, pair) in handler.active_pairs.iter().enumerate() {
        let constraint_row = n_dof + c;
        let has_constraint = k_ext.row(constraint_row).iter().any(|&v| v.abs() > 1e-10);
        println!("  Constraint {}: {} (slave node {}, gap={:.6})",
                 c, if has_constraint { "Added" } else { "Missing" },
                 pair.slave_id, pair.gap);
    }

    println!("\n  Constraint Handler: OK");
    Ok(())
}

/// Full contact simulation example
fn run_full_contact_simulation() -> anyhow::Result<()> {
    println!("\n─".repeat(60));
    println!("Bonus: Full Contact Simulation");
    println!("─".repeat(60));

    // Setup
    let params = ContactParameters {
        method: ContactMethod::Penalty,
        penalty_stiffness: 1e8,
        friction_coefficient: 0.1,
        tolerance: 1e-6,
        max_lagrange_iter: 10,
    };

    let detector = NodeToSurfaceContact::new(params.clone());
    let penalty = PenaltyContact::new(params.penalty_stiffness, params.friction_coefficient);

    // Simple block-on-surface problem
    let master_nodes = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
    ];
    let master_segments = vec![(0, 1)];

    // Block nodes (will move down)
    let mut block_positions = vec![
        Point2::new(0.3, 0.1),
        Point2::new(0.7, 0.1),
    ];

    println!("\nSimulation: Block descending onto surface");
    println!("  Initial block height: 0.1");

    // Simulate descent
    for step in 0..10 {
        let height = 0.1 - step as f64 * 0.015;
        block_positions[0].y = height;
        block_positions[1].y = height;

        // Detect contact
        let detection = detector.detect_2d(
            &block_positions,
            &master_segments,
            &master_nodes,
        );

        // Compute contact forces
        let mut total_force = 0.0;
        for pair in &detection.pairs {
            let (force, _friction) = penalty.compute_force_2d(pair, 0.0);
            total_force += force.norm();
        }

        println!("  Step {}: height={:.4}, contacts={}, force={:.2}",
                 step + 1, height, detection.pairs.len(), total_force);
    }

    println!("\n  Full Simulation: OK");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contact_example() {
        demonstrate_contact_detection().unwrap();
        demonstrate_penalty_method().unwrap();
        demonstrate_augmented_lagrangian().unwrap();
        demonstrate_constraint_handler().unwrap();
        run_full_contact_simulation().unwrap();
    }
}
