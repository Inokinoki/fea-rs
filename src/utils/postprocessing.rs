//! Result post-processing utilities.

use serde::Serialize;

/// Nodal displacement result.
#[derive(Debug, Clone, Serialize)]
pub struct NodalDisplacement {
    pub node: usize,
    pub ux: f64,
    pub uy: f64,
    pub uz: f64,
    pub magnitude: f64,
}

/// Element result for post-processing.
#[derive(Debug, Clone, Serialize)]
pub struct ElementResult {
    pub element_id: usize,
    pub node_ids: [usize; 2],
    pub axial_force: f64,
    pub axial_stress: f64,
    pub axial_strain: f64,
    pub elongation: f64,
}

/// Reaction force at a constrained DOF.
#[derive(Debug, Clone, Serialize)]
pub struct ReactionForce {
    pub node: usize,
    pub dof: String,
    pub value: f64,
}

/// Result statistics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ResultStatistics {
    pub max_displacement: f64,
    pub max_displacement_node: usize,
    pub max_stress: f64,
    pub max_stress_element: usize,
    pub min_stress: f64,
    pub min_stress_element: usize,
    pub max_axial_force: f64,
    pub max_axial_force_element: usize,
}

impl ResultStatistics {
    /// Computes statistics from results.
    pub fn from_results(displacements: &[NodalDisplacement], element_results: &[ElementResult]) -> Self {
        let mut stats = Self::default();

        for (i, d) in displacements.iter().enumerate() {
            if d.magnitude > stats.max_displacement {
                stats.max_displacement = d.magnitude;
                stats.max_displacement_node = i;
            }
        }

        for (i, r) in element_results.iter().enumerate() {
            if r.axial_stress.abs() > stats.max_stress.abs() {
                stats.max_stress = r.axial_stress;
                stats.max_stress_element = i;
            }
            if r.axial_stress < stats.min_stress {
                stats.min_stress = r.axial_stress;
                stats.min_stress_element = i;
            }
            if r.axial_force.abs() > stats.max_axial_force.abs() {
                stats.max_axial_force = r.axial_force;
                stats.max_axial_force_element = i;
            }
        }

        stats
    }
}
