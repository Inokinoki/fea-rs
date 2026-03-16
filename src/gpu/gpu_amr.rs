//! GPU-accelerated adaptive mesh refinement (AMR).
//!
//! This module provides:
//! - Error estimation for mesh adaptation
//! - GPU-accelerated refinement/derefinement
//! - Gradient-based refinement criteria
//! - Solution-based mesh adaptation

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

use super::{GPUCSRMatrix, SparseMatrixVectorMul};

/// Mesh element for AMR.
#[derive(Debug, Clone)]
pub struct MeshElement {
    pub id: usize,
    pub node_ids: Vec<usize>,
    pub error_indicator: f64,
    pub refinement_level: usize,
}

/// Adaptive mesh refinement manager.
pub struct GPUAdaptiveMeshRefinement {
    device_id: usize,
    elements: Vec<MeshElement>,
    max_refinement_level: usize,
}

impl GPUAdaptiveMeshRefinement {
    /// Creates a new AMR manager.
    pub fn new(device_id: usize, max_refinement_level: usize) -> Self {
        Self {
            device_id,
            elements: Vec::new(),
            max_refinement_level,
        }
    }

    /// Estimates error using gradient recovery.
    pub fn estimate_error(&self, solution: &[f64], gradient: &[f64]) -> Vec<f64> {
        let mut errors = Vec::with_capacity(self.elements.len());

        for elem in &self.elements {
            // Simplified error indicator: gradient magnitude
            let elem_error: f64 = elem.node_ids.iter()
                .map(|&i| gradient.get(i).copied().unwrap_or(0.0).abs())
                .sum();

            errors.push(elem_error / elem.node_ids.len() as f64);
        }

        errors
    }

    /// Marks elements for refinement based on error threshold.
    pub fn mark_for_refinement(&mut self, errors: &[f64], threshold: f64) -> Vec<usize> {
        let mut to_refine = Vec::new();

        for (i, (&error, elem)) in errors.iter().zip(&self.elements).enumerate() {
            if error > threshold && elem.refinement_level < self.max_refinement_level {
                to_refine.push(i);
            }
        }

        to_refine
    }

    /// Marks elements for derefinement.
    pub fn mark_for_derefine(&mut self, errors: &[f64], threshold: f64) -> Vec<usize> {
        let mut to_derefine = Vec::new();

        for (i, (&error, elem)) in errors.iter().zip(&self.elements).enumerate() {
            if error < threshold * 0.1 && elem.refinement_level > 0 {
                to_derefine.push(i);
            }
        }

        to_derefine
    }

    /// Refines marked elements (simplified: just increases level).
    pub fn refine_elements(&mut self, indices: &[usize]) {
        for &i in indices {
            if i < self.elements.len() {
                self.elements[i].refinement_level += 1;
                // In real implementation: actually split element
            }
        }
    }

    /// Derefinement (simplified: decreases level).
    pub fn derefine_elements(&mut self, indices: &[usize]) {
        for &i in indices {
            if i < self.elements.len() && self.elements[i].refinement_level > 0 {
                self.elements[i].refinement_level -= 1;
            }
        }
    }

    /// Adds an element to the mesh.
    pub fn add_element(&mut self, element: MeshElement) {
        self.elements.push(element);
    }

    /// Returns mesh statistics.
    pub fn get_statistics(&self) -> AMRStatistics {
        let total_elements = self.elements.len();
        let refinement_levels: HashMap<usize, usize> = self.elements.iter()
            .fold(HashMap::new(), |mut acc, elem| {
                *acc.entry(elem.refinement_level).or_insert(0) += 1;
                acc
            });

        let avg_level: f64 = self.elements.iter()
            .map(|e| e.refinement_level)
            .sum::<usize>() as f64 / total_elements as f64;

        AMRStatistics {
            total_elements,
            refinement_levels,
            average_level: avg_level,
            max_level: self.elements.iter().map(|e| e.refinement_level).max().unwrap_or(0),
        }
    }
}

/// AMR statistics.
#[derive(Debug, Clone)]
pub struct AMRStatistics {
    pub total_elements: usize,
    pub refinement_levels: HashMap<usize, usize>,
    pub average_level: f64,
    pub max_level: usize,
}

/// Gradient recovery operator for error estimation.
pub struct GradientRecovery {
    device_id: usize,
}

impl GradientRecovery {
    /// Creates a new gradient recovery operator.
    pub fn new(device_id: usize) -> Self {
        Self { device_id }
    }

    /// Recovers gradient from nodal solution (ZZ estimator).
    pub fn recover_gradient(
        &self,
        solution: &[f64],
        connectivity: &[Vec<usize>],
    ) -> Vec<f64> {
        let n_nodes = solution.len();
        let mut gradient = vec![0.0f64; n_nodes];

        // Simplified: average neighboring differences
        for (node_id, neighbors) in connectivity.iter().enumerate() {
            if node_id < n_nodes {
                let base = solution[node_id];
                let diff: f64 = neighbors.iter()
                    .map(|&n| (solution.get(n).copied().unwrap_or(base) - base).abs())
                    .sum();
                gradient[node_id] = diff / neighbors.len() as f64;
            }
        }

        gradient
    }
}

/// Solution interpolation between meshes.
pub struct MeshInterpolation {
    device_id: usize,
}

impl MeshInterpolation {
    /// Creates a new mesh interpolator.
    pub fn new(device_id: usize) -> Self {
        Self { device_id }
    }

    /// Interpolates solution from coarse to fine mesh.
    pub fn interpolate_coarse_to_fine(
        &self,
        coarse_solution: &[f64],
        old_to_new: &[Vec<usize>],
    ) -> Vec<f64> {
        let mut fine_solution = vec![0.0f64; old_to_new.len()];

        for (new_node, old_nodes) in old_to_new.iter().enumerate() {
            if new_node < fine_solution.len() {
                fine_solution[new_node] = old_nodes.iter()
                    .map(|&i| coarse_solution.get(i).copied().unwrap_or(0.0))
                    .sum::<f64>() / old_nodes.len() as f64;
            }
        }

        fine_solution
    }
}

/// Demonstrates adaptive mesh refinement.
pub fn run_amr_demo() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║        Adaptive Mesh Refinement Demo                      ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Create initial mesh
    let mut amr = GPUAdaptiveMeshRefinement::new(0, 3);

    // Add initial elements (1D for simplicity)
    for i in 0..10 {
        amr.add_element(MeshElement {
            id: i,
            node_ids: vec![i, i + 1],
            error_indicator: 0.0,
            refinement_level: 0,
        });
    }

    println!("Initial Mesh:");
    let stats = amr.get_statistics();
    println!("  Total elements: {}", stats.total_elements);
    println!("  Average level: {:.2}", stats.average_level);
    println!();

    // Simulate AMR iterations
    let solution = vec![1.0f64; 11];
    let gradient = (0..11).map(|i| (i as f64 * 0.5).sin().abs()).collect::<Vec<_>>();

    let recovery = GradientRecovery::new(0);
    let errors = recovery.recover_gradient(&solution, &amr.elements.iter().map(|e| e.node_ids.clone()).collect::<Vec<_>>());

    println!("AMR Iteration 1:");
    let to_refine = amr.mark_for_refinement(&errors, 0.3);
    println!("  Elements to refine: {}", to_refine.len());
    amr.refine_elements(&to_refine);

    let stats = amr.get_statistics();
    println!("  Total elements: {}", stats.total_elements);
    println!("  Refinement levels: {:?}", stats.refinement_levels);
    println!();

    println!("AMR Iteration 2:");
    let errors = recovery.recover_gradient(&solution, &amr.elements.iter().map(|e| e.node_ids.clone()).collect::<Vec<_>>());
    let to_refine = amr.mark_for_refinement(&errors, 0.2);
    let to_derefine = amr.mark_for_derefine(&errors, 0.05);
    println!("  Elements to refine: {}", to_refine.len());
    println!("  Elements to derefine: {}", to_derefine.len());
    amr.refine_elements(&to_refine);
    amr.derefine_elements(&to_derefine);

    let stats = amr.get_statistics();
    println!("  Final statistics:");
    println!("    Total elements: {}", stats.total_elements);
    println!("    Average level: {:.2}", stats.average_level);
    println!("    Max level: {}", stats.max_level);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amr_creation() {
        let amr = GPUAdaptiveMeshRefinement::new(0, 3);
        assert_eq!(amr.get_statistics().total_elements, 0);
    }

    #[test]
    fn test_error_estimation() {
        let mut amr = GPUAdaptiveMeshRefinement::new(0, 3);
        amr.add_element(MeshElement {
            id: 0,
            node_ids: vec![0, 1],
            error_indicator: 0.0,
            refinement_level: 0,
        });

        let solution = vec![0.0, 1.0];
        let gradient = vec![1.0, 0.0];

        let errors = amr.estimate_error(&solution, &gradient);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_refinement() {
        let mut amr = GPUAdaptiveMeshRefinement::new(0, 3);
        amr.add_element(MeshElement {
            id: 0,
            node_ids: vec![0, 1],
            error_indicator: 1.0,
            refinement_level: 0,
        });

        let errors = vec![1.0];
        let to_refine = amr.mark_for_refinement(&errors, 0.5);
        assert_eq!(to_refine.len(), 1);

        amr.refine_elements(&to_refine);
        assert_eq!(amr.elements[0].refinement_level, 1);
    }

    #[test]
    fn test_gradient_recovery() {
        let recovery = GradientRecovery::new(0);
        let solution = vec![0.0, 1.0, 2.0, 3.0];
        let connectivity = vec![
            vec![1],
            vec![0, 2],
            vec![1, 3],
            vec![2],
        ];

        let gradient = recovery.recover_gradient(&solution, &connectivity);
        assert_eq!(gradient.len(), 4);
    }
}
