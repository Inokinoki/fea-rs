//! Contact Analysis Demo - Demonstrates contact mechanics capabilities.

use fea::algorithms::contact::{
    ContactManager, ContactPair, ContactType, FrictionModel,
};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    println!("=== Contact Analysis Demo ===\n");

    // Create contact manager
    let mut contact_mgr = ContactManager::new();
    
    // Define contact pair (node-to-surface)
    let mut pair = ContactPair::new(
        vec![0, 1, 2],  // Master nodes
        vec![3, 4],     // Slave nodes
        ContactType::NodeToSurface,
    );
    
    // Set penalty stiffness and tolerance
    pair.penalty_stiffness = 1e6;
    pair.tolerance = 0.01;
    
    // Add friction
    pair = pair.with_friction(FrictionModel::coulomb(0.3));
    
    contact_mgr.add_pair(pair);
    println!("Contact pairs defined: {}", contact_mgr.num_pairs());
    
    // Define node positions for contact detection
    let mut positions: HashMap<usize, [f64; 3]> = HashMap::new();
    positions.insert(0, [0.0, 0.0, 0.0]);
    positions.insert(1, [1.0, 0.0, 0.0]);
    positions.insert(2, [0.5, 1.0, 0.0]);
    positions.insert(3, [0.5, 0.5, 0.0]);  // Near master surface
    positions.insert(4, [0.3, 0.3, 0.0]);
    
    // Detect contact
    let detections = contact_mgr.detect(&positions);
    println!("Contact detections: {}", detections.len());
    
    for detection in &detections {
        println!("  Slave node {}: gap={:.4}, in_contact={}",
                 detection.slave_node, detection.gap, detection.in_contact);
    }

    // Compute contact forces
    let forces = contact_mgr.compute_penalty_forces(&positions);
    println!("\nContact forces on {} nodes", forces.len());

    for (node, force) in &forces {
        let mag = (force[0].powi(2) + force[1].powi(2) + force[2].powi(2)).sqrt();
        println!("  Node {}: |F| = {:.2} N", node, mag);
    }
    
    println!("\nContact Analysis Complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_contact_pair_creation() {
        let pair = ContactPair::new(
            vec![0, 1],
            vec![2],
            ContactType::NodeToSurface,
        );
        assert_eq!(pair.master_nodes.len(), 2);
        assert_eq!(pair.slave_nodes.len(), 1);
    }
    
    #[test]
    fn test_friction_models() {
        let frictionless = FrictionModel::frictionless();
        assert!(matches!(frictionless, FrictionModel::Frictionless));
        
        let coulomb = FrictionModel::coulomb(0.3);
        if let FrictionModel::Coulomb { mu } = coulomb {
            assert!((mu - 0.3).abs() < 1e-10);
        } else {
            panic!("Expected Coulomb friction");
        }
    }
    
    #[test]
    fn test_contact_detection() {
        let mut mgr = ContactManager::new();
        let mut pair = ContactPair::new(
            vec![0, 1],
            vec![2],
            ContactType::NodeToNode,
        );
        pair.tolerance = 1.0;  // Large tolerance
        mgr.add_pair(pair);
        
        let mut positions = HashMap::new();
        positions.insert(0, [0.0, 0.0, 0.0]);
        positions.insert(1, [1.0, 0.0, 0.0]);
        positions.insert(2, [0.5, 0.5, 0.0]);
        
        let detections = mgr.detect(&positions);
        assert!(!detections.is_empty());
    }
}
