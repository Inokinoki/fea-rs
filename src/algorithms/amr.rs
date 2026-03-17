//! Adaptive Mesh Refinement (AMR) and error estimation.
#![allow(unused_variables)]
//!
//! This module provides:
//! - Mesh quality indicators
//! - Error estimators (Zienkiewicz-Zhu, residual-based)
//! - Mesh refinement strategies
//! - Solution adaptive refinement

use nalgebra::{DMatrix, DVector, Point2};

/// Mesh quality metrics.
#[derive(Debug, Clone)]
pub struct MeshQuality {
    /// Minimum element quality (0-1, 1 is best).
    pub min_quality: f64,
    /// Average element quality.
    pub avg_quality: f64,
    /// Maximum element quality.
    pub max_quality: f64,
    /// Number of poor quality elements.
    pub poor_elements: usize,
}

impl MeshQuality {
    /// Creates a new mesh quality report.
    pub fn new() -> Self {
        Self {
            min_quality: 1.0,
            avg_quality: 0.0,
            max_quality: 1.0,
            poor_elements: 0,
        }
    }

    /// Computes quality metrics for a mesh.
    pub fn compute_quality_2d(
        &self,
        nodes: &[Point2<f64>],
        elements: &[(usize, usize, usize, usize)],
    ) -> MeshQuality {
        let mut qualities = Vec::with_capacity(elements.len());

        for (n0, n1, n2, n3) in elements {
            let q = self.element_quality_quad_2d(
                &nodes[*n0], &nodes[*n1], &nodes[*n2], &nodes[*n3],
            );
            qualities.push(q);
        }

        self.compute_stats(&qualities)
    }

    /// Computes quality for a single quad element.
    pub fn element_quality_quad_2d(
        &self,
        p0: &Point2<f64>,
        p1: &Point2<f64>,
        p2: &Point2<f64>,
        p3: &Point2<f64>,
    ) -> f64 {
        // Compute aspect ratio and skew
        let e01 = p1 - p0;
        let e12 = p2 - p1;
        let e23 = p3 - p2;
        let e30 = p0 - p3;

        let l01 = e01.norm();
        let l12 = e12.norm();
        let l23 = e23.norm();
        let l30 = e30.norm();

        if l01 < 1e-15 || l12 < 1e-15 || l23 < 1e-15 || l30 < 1e-15 {
            return 0.0;
        }

        // Aspect ratio
        let aspect = l01.max(l12).max(l23).max(l30)
            / l01.min(l12).min(l23).min(l30);

        // Skew angle (deviation from 90 degrees)
        let cos_angle_01_12 = (e01.x * e12.x + e01.y * e12.y) / (l01 * l12);
        let angle_01_12 = cos_angle_01_12.abs().acos();
        let skew = (angle_01_12 - std::f64::consts::FRAC_PI_2).abs();

        // Quality is combination of aspect ratio and skew
        let aspect_quality = 1.0 / aspect;
        let skew_quality = 1.0 - 2.0 * skew / std::f64::consts::PI;

        aspect_quality.min(skew_quality.max(0.0))
    }

    fn compute_stats(&self, qualities: &[f64]) -> MeshQuality {
        let mut mq = MeshQuality::new();

        if qualities.is_empty() {
            return mq;
        }

        mq.min_quality = qualities.iter().copied().fold(1.0, f64::min);
        mq.max_quality = qualities.iter().copied().fold(0.0, f64::max);
        mq.avg_quality = qualities.iter().copied().sum::<f64>() / qualities.len() as f64;
        mq.poor_elements = qualities.iter().filter(|&&q| q < 0.3).count();

        mq
    }
}

/// Error estimator for adaptive refinement.
#[derive(Debug, Clone)]
pub struct ErrorEstimator {
    /// Element-wise error indicators.
    pub element_errors: Vec<f64>,
    /// Global error estimate.
    pub global_error: f64,
    /// Estimated energy norm error.
    pub energy_norm_error: f64,
}

impl ErrorEstimator {
    /// Creates a new error estimator.
    pub fn new() -> Self {
        Self {
            element_errors: Vec::new(),
            global_error: 0.0,
            energy_norm_error: 0.0,
        }
    }

    /// Zienkiewicz-Zhu error estimator.
    /// Compares smoothed and raw stress fields.
    pub fn zz_estimator(
        &self,
        raw_stresses: &[DVector<f64>],
        smoothed_stresses: &[DVector<f64>],
        element_volumes: &[f64],
    ) -> ErrorEstimator {
        let mut element_errors = Vec::with_capacity(raw_stresses.len());
        let mut total_error_sq = 0.0;
        let mut energy_error_sq = 0.0;

        for (i, (raw, smooth)) in raw_stresses.iter().zip(smoothed_stresses.iter()).enumerate() {
            let diff = raw - smooth;
            let err_sq = diff.dot(&diff) * element_volumes.get(i).copied().unwrap_or(1.0);
            element_errors.push(err_sq.sqrt());
            total_error_sq += err_sq;
            energy_error_sq += err_sq;
        }

        ErrorEstimator {
            element_errors,
            global_error: total_error_sq.sqrt(),
            energy_norm_error: energy_error_sq.sqrt(),
        }
    }

    /// Residual-based error estimator.
    pub fn residual_estimator(
        &self,
        displacements: &[DVector<f64>],
        forces: &[DVector<f64>],
        stiffness: &DMatrix<f64>,
    ) -> ErrorEstimator {
        let n = displacements.len();
        let mut u_global = DVector::zeros(stiffness.nrows());

        // Assemble global displacement
        for (i, u) in displacements.iter().enumerate() {
            let dof_start = i * u.len();
            for (j, &val) in u.iter().enumerate() {
                if dof_start + j < u_global.len() {
                    u_global[dof_start + j] = val;
                }
            }
        }

        // Compute residual: r = f - Ku
        let ku = stiffness * &u_global;
        let mut f_global = DVector::zeros(stiffness.nrows());
        for (i, f) in forces.iter().enumerate() {
            let dof_start = i * f.len();
            for (j, &val) in f.iter().enumerate() {
                if dof_start + j < f_global.len() {
                    f_global[dof_start + j] = val;
                }
            }
        }

        let residual = f_global - ku;
        let global_error = residual.norm();

        // Element-wise errors
        let dof_per_element = if !displacements.is_empty() {
            displacements[0].len()
        } else {
            2
        };

        let num_elements = n;
        let mut element_errors = Vec::with_capacity(num_elements);

        for e in 0..num_elements {
            let dof_start = e * dof_per_element;
            let dof_end = (dof_start + dof_per_element).min(residual.len());
            let mut err_sq = 0.0;
            for i in dof_start..dof_end {
                err_sq += residual[i] * residual[i];
            }
            element_errors.push(err_sq.sqrt());
        }

        ErrorEstimator {
            element_errors,
            global_error,
            energy_norm_error: global_error,
        }
    }

    /// Returns elements marked for refinement.
    pub fn mark_for_refinement(&self, threshold_ratio: f64) -> Vec<usize> {
        if self.element_errors.is_empty() {
            return Vec::new();
        }

        let max_error = self.element_errors.iter().copied().fold(0.0, f64::max);
        let threshold = max_error * threshold_ratio;

        self.element_errors
            .iter()
            .enumerate()
            .filter(|(_, &e)| e > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns efficiency index (ratio of estimated to true error).
    pub fn efficiency_index(&self, true_error: f64) -> f64 {
        if true_error > 1e-15 {
            self.global_error / true_error
        } else {
            1.0
        }
    }
}

/// Mesh refinement strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RefinementStrategy {
    /// Uniform refinement of all elements.
    Uniform,
    /// Refine only marked elements.
    Marked,
    /// Refine based on error threshold.
    ErrorBased(f64),
    /// Refine a fixed percentage of worst elements.
    Percentage(f64),
}

/// Adaptive mesh refiner.
#[derive(Debug, Clone)]
pub struct AdaptiveMeshRefiner {
    /// Current refinement level.
    pub level: usize,
    /// Maximum refinement level.
    pub max_level: usize,
    /// Refinement strategy.
    pub strategy: RefinementStrategy,
    /// Error tolerance.
    pub tolerance: f64,
}

impl AdaptiveMeshRefiner {
    /// Creates a new adaptive mesh refiner.
    pub fn new(max_level: usize, tolerance: f64) -> Self {
        Self {
            level: 0,
            max_level,
            strategy: RefinementStrategy::ErrorBased(0.5),
            tolerance,
        }
    }

    /// Sets the refinement strategy.
    pub fn with_strategy(mut self, strategy: RefinementStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Performs adaptive refinement iteration.
    pub fn refine_quad_mesh(
        &self,
        nodes: &[Point2<f64>],
        elements: &[(usize, usize, usize, usize)],
        element_errors: &[f64],
    ) -> (Vec<Point2<f64>>, Vec<(usize, usize, usize, usize)>) {
        let threshold_ratio = match self.strategy {
            RefinementStrategy::ErrorBased(r) => r,
            RefinementStrategy::Percentage(p) => {
                // Find threshold for top p% elements
                let mut sorted_errors = element_errors.to_vec();
                sorted_errors.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let idx = (element_errors.len() as f64 * p / 100.0) as usize;
                if idx < element_errors.len() && element_errors[idx] > 0.0 {
                    element_errors[idx + 1] / element_errors[idx].max(1e-15)
                } else {
                    0.5
                }
            }
            _ => 0.5,
        };

        let marked: Vec<usize> = element_errors
            .iter()
            .enumerate()
            .filter(|(_, &e)| {
                let max_e = element_errors.iter().copied().fold(0.0, f64::max);
                e > max_e * threshold_ratio
            })
            .map(|(i, _)| i)
            .collect();

        self.refine_marked_elements(nodes, elements, &marked)
    }

    /// Refines marked elements.
    pub fn refine_marked_elements(
        &self,
        nodes: &[Point2<f64>],
        elements: &[(usize, usize, usize, usize)],
        marked: &[usize],
    ) -> (Vec<Point2<f64>>, Vec<(usize, usize, usize, usize)>) {
        let mut new_nodes = nodes.to_vec();
        let mut new_elements = Vec::new();

        let marked_set: std::collections::HashSet<usize> = marked.iter().copied().collect();
        let mut edge_midpoints: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();

        for (e_idx, elem) in elements.iter().enumerate() {
            if !marked_set.contains(&e_idx) {
                // Keep unrefined element
                new_elements.push(*elem);
                continue;
            }

            // Refine quad into 4 quads
            let (n0, n1, n2, n3) = *elem;

            // Get or create edge midpoints
            let m01 = self.get_or_create_midpoint(
                &mut new_nodes, &mut edge_midpoints, n0, n1,
            );
            let m12 = self.get_or_create_midpoint(
                &mut new_nodes, &mut edge_midpoints, n1, n2,
            );
            let m23 = self.get_or_create_midpoint(
                &mut new_nodes, &mut edge_midpoints, n2, n3,
            );
            let m30 = self.get_or_create_midpoint(
                &mut new_nodes, &mut edge_midpoints, n3, n0,
            );

            // Create center point
            let c0 = (nodes[n0].x + nodes[n1].x + nodes[n2].x + nodes[n3].x) / 4.0;
            let c1 = (nodes[n0].y + nodes[n1].y + nodes[n2].y + nodes[n3].y) / 4.0;
            let center = new_nodes.len();
            new_nodes.push(Point2::new(c0, c1));

            // Create 4 new quads
            new_elements.push((n0, m01, center, m30));
            new_elements.push((m01, n1, m12, center));
            new_elements.push((center, m12, n2, m23));
            new_elements.push((m30, center, m23, n3));
        }

        (new_nodes, new_elements)
    }

    fn get_or_create_midpoint(
        &self,
        nodes: &mut Vec<Point2<f64>>,
        edge_midpoints: &mut std::collections::HashMap<(usize, usize), usize>,
        n0: usize,
        n1: usize,
    ) -> usize {
        let key = if n0 < n1 { (n0, n1) } else { (n1, n0) };

        *edge_midpoints.entry(key).or_insert_with(|| {
            let mx = (nodes[n0].x + nodes[n1].x) / 2.0;
            let my = (nodes[n0].y + nodes[n1].y) / 2.0;
            nodes.push(Point2::new(mx, my));
            nodes.len() - 1
        })
    }

    /// Checks if refinement should continue.
    pub fn should_continue(&self, error: f64) -> bool {
        self.level < self.max_level && error > self.tolerance
    }

    /// Increments refinement level.
    pub fn increment_level(&mut self) {
        self.level += 1;
    }
}

/// Solution interpolator for mesh transfers.
pub struct SolutionInterpolator;

impl SolutionInterpolator {
    /// Interpolates solution from coarse to fine mesh.
    pub fn interpolate_coarse_to_fine(
        coarse_nodes: &[Point2<f64>],
        coarse_solution: &[f64],
        fine_nodes: &[Point2<f64>],
    ) -> Vec<f64> {
        let mut fine_solution = Vec::with_capacity(fine_nodes.len());

        for fine_node in fine_nodes {
            // Find nearest coarse node (simplified - should use shape functions)
            let mut min_dist = f64::INFINITY;
            let mut coarse_idx = 0;

            for (i, coarse_node) in coarse_nodes.iter().enumerate() {
                let dist = (fine_node.x - coarse_node.x).powi(2)
                    + (fine_node.y - coarse_node.y).powi(2);
                if dist < min_dist {
                    min_dist = dist;
                    coarse_idx = i;
                }
            }

            fine_solution.push(coarse_solution[coarse_idx]);
        }

        fine_solution
    }

    /// Projects fine solution to coarse mesh (L2 projection, simplified).
    pub fn project_fine_to_coarse(
        fine_nodes: &[Point2<f64>],
        fine_solution: &[f64],
        coarse_nodes: &[Point2<f64>],
    ) -> Vec<f64> {
        let mut coarse_solution = Vec::with_capacity(coarse_nodes.len());

        for coarse_node in coarse_nodes {
            // Find nearest fine node
            let mut min_dist = f64::INFINITY;
            let mut fine_idx = 0;

            for (i, fine_node) in fine_nodes.iter().enumerate() {
                let dist = (coarse_node.x - fine_node.x).powi(2)
                    + (coarse_node.y - fine_node.y).powi(2);
                if dist < min_dist {
                    min_dist = dist;
                    fine_idx = i;
                }
            }

            coarse_solution.push(fine_solution[fine_idx]);
        }

        coarse_solution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_quality() {
        let mq = MeshQuality::new();

        // Perfect square
        let nodes = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let elements = vec![(0, 1, 2, 3)];

        let quality = mq.compute_quality_2d(&nodes, &elements);
        assert!(quality.min_quality > 0.5);
        assert!(quality.avg_quality > 0.5);
    }

    #[test]
    fn test_error_estimator() {
        let est = ErrorEstimator::new();

        let raw = vec![
            DVector::from_column_slice(&[1.0, 2.0, 3.0]),
            DVector::from_column_slice(&[2.0, 3.0, 4.0]),
        ];
        let smooth = vec![
            DVector::from_column_slice(&[1.1, 2.1, 3.1]),
            DVector::from_column_slice(&[2.1, 3.1, 4.1]),
        ];
        let volumes = vec![1.0, 1.0];

        let error = est.zz_estimator(&raw, &smooth, &volumes);
        assert!(error.global_error > 0.0);
        assert_eq!(error.element_errors.len(), 2);
    }

    #[test]
    fn test_adaptive_refinement() {
        let amr = AdaptiveMeshRefiner::new(3, 0.01);

        let nodes = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let elements = vec![(0, 1, 2, 3)];
        let errors = vec![1.0];

        let (new_nodes, new_elements) = amr.refine_quad_mesh(&nodes, &elements, &errors);

        // One element refined into 4
        assert_eq!(new_elements.len(), 4);
        assert!(new_nodes.len() > nodes.len());
    }

    #[test]
    fn test_solution_interpolation() {
        let coarse_nodes = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ];
        let coarse_sol = vec![1.0, 2.0, 3.0];

        let fine_nodes = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.5, 0.0),
            Point2::new(1.0, 0.0),
        ];

        let fine_sol = SolutionInterpolator::interpolate_coarse_to_fine(
            &coarse_nodes, &coarse_sol, &fine_nodes,
        );

        assert_eq!(fine_sol.len(), fine_nodes.len());
    }
}
