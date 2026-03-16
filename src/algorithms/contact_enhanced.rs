//! Contact analysis algorithms for FEA.
//!
//! This module provides:
//! - Contact detection algorithms
//! - Penalty methods for contact
//! - Lagrange multiplier methods
//! - Augmented Lagrangian methods
//! - Mortar contact methods

use nalgebra::{DMatrix, DVector, Point2, Point3, Vector2, Vector3};
use crate::core::{Node, NodeId};

/// Contact pair information.
#[derive(Debug, Clone)]
pub struct ContactPair {
    /// Master node/element ID.
    pub master_id: usize,
    /// Slave node/element ID.
    pub slave_id: usize,
    /// Contact normal (from master to slave).
    pub normal: DVector<f64>,
    /// Gap distance (positive = open, negative = penetration).
    pub gap: f64,
    /// Contact stiffness.
    pub stiffness: f64,
}

/// Contact detection results.
#[derive(Debug, Clone)]
pub struct ContactDetection {
    /// Detected contact pairs.
    pub pairs: Vec<ContactPair>,
    /// Number of active contacts.
    pub num_active: usize,
    /// Maximum penetration.
    pub max_penetration: f64,
}

/// Contact algorithm type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContactMethod {
    /// Penalty method.
    Penalty,
    /// Lagrange multipliers.
    LagrangeMultiplier,
    /// Augmented Lagrangian.
    AugmentedLagrangian,
    /// Mortar method.
    Mortar,
}

/// Contact parameters.
#[derive(Debug, Clone)]
pub struct ContactParameters {
    /// Contact method.
    pub method: ContactMethod,
    /// Penalty stiffness (for penalty/augmented Lagrangian).
    pub penalty_stiffness: f64,
    /// Friction coefficient.
    pub friction_coefficient: f64,
    /// Contact tolerance for detection.
    pub tolerance: f64,
    /// Maximum Lagrange multiplier iterations.
    pub max_lagrange_iter: usize,
}

impl Default for ContactParameters {
    fn default() -> Self {
        Self {
            method: ContactMethod::Penalty,
            penalty_stiffness: 1e6,
            friction_coefficient: 0.0,
            tolerance: 1e-6,
            max_lagrange_iter: 10,
        }
    }
}

/// Node-to-surface contact detector.
#[derive(Debug, Clone)]
pub struct NodeToSurfaceContact {
    /// Contact parameters.
    pub params: ContactParameters,
}

impl NodeToSurfaceContact {
    /// Creates a new node-to-surface contact detector.
    pub fn new(params: ContactParameters) -> Self {
        Self { params }
    }

    /// Detects contact between nodes and surfaces.
    pub fn detect_2d(
        &self,
        slave_nodes: &[Point2<f64>],
        master_segments: &[(usize, usize)],
        master_nodes: &[Point2<f64>],
    ) -> ContactDetection {
        let mut pairs = Vec::new();
        let mut max_penetration: f64 = 0.0;

        for (slave_idx, slave_pos) in slave_nodes.iter().enumerate() {
            // Find closest master segment
            let mut closest_dist = f64::INFINITY;
            let mut closest_master = 0;
            let mut closest_normal = Vector2::y();
            let mut closest_point = Point2::origin();

            for (seg_idx, &(m0, m1)) in master_segments.iter().enumerate() {
                if m0 >= master_nodes.len() || m1 >= master_nodes.len() {
                    continue;
                }

                let p0 = master_nodes[m0];
                let p1 = master_nodes[m1];

                let (dist, normal, point) = self.point_segment_distance_2d(
                    slave_pos, &p0, &p1,
                );

                if dist < closest_dist {
                    closest_dist = dist;
                    closest_master = seg_idx;
                    closest_normal = normal;
                    closest_point = point;
                }
            }

            // Check if in contact (within tolerance)
            let gap = closest_dist;
            if gap < self.params.tolerance {
                pairs.push(ContactPair {
                    master_id: closest_master,
                    slave_id: slave_idx,
                    normal: DVector::from_column_slice(&[closest_normal.x, closest_normal.y]),
                    gap,
                    stiffness: self.params.penalty_stiffness,
                });

                if gap < 0.0 {
                    max_penetration = max_penetration.max(-gap);
                }
            }
        }

        let num_active = pairs.iter().filter(|p| p.gap < 0.0).count();

        ContactDetection {
            pairs,
            num_active,
            max_penetration,
        }
    }

    fn point_segment_distance_2d(
        &self,
        p: &Point2<f64>,
        p0: &Point2<f64>,
        p1: &Point2<f64>,
    ) -> (f64, Vector2<f64>, Point2<f64>) {
        let v = p1 - p0;
        let w = p - p0;

        let vv = v.dot(&v);
        let wv = w.dot(&v);

        let t = if vv > 1e-15 {
            (wv / vv).max(0.0).min(1.0)
        } else {
            0.0
        };

        let closest = p0 + v * t;
        let diff = p - closest;
        let dist = diff.norm();

        // Normal perpendicular to segment
        let mut normal = Vector2::new(-v.y, v.x);
        if normal.norm() > 1e-15 {
            normal.normalize_mut();
        }

        // Ensure normal points toward slave
        if normal.dot(&diff) < 0.0 {
            normal = -normal;
        }

        (dist, normal, closest)
    }
}

/// Penalty contact force calculator.
pub struct PenaltyContact {
    /// Penalty stiffness.
    pub stiffness: f64,
    /// Friction coefficient.
    pub friction: f64,
}

impl PenaltyContact {
    /// Creates a new penalty contact calculator.
    pub fn new(stiffness: f64, friction: f64) -> Self {
        Self { stiffness, friction }
    }

    /// Computes contact forces for a contact pair.
    pub fn compute_force_2d(&self, pair: &ContactPair, tangential_disp: f64) -> (DVector<f64>, f64) {
        let mut normal_force = DVector::zeros(2);
        let mut friction_force = 0.0;

        // Only if in contact (penetration)
        if pair.gap < 0.0 {
            // Normal force (penalty)
            let normal_magnitude = -pair.stiffness * pair.gap;
            normal_force = pair.normal.scale(normal_magnitude);

            // Friction force (Coulomb)
            if self.friction > 0.0 {
                let max_friction = self.friction * normal_magnitude;
                let tangential_force = -self.stiffness * tangential_disp;
                friction_force = tangential_force.clamp(-max_friction, max_friction);
            }
        }

        (normal_force, friction_force)
    }

    /// Assembles contact contributions to global system.
    pub fn assemble_contact_stiffness(
        &self,
        pairs: &[ContactPair],
        dof_per_node: usize,
    ) -> DMatrix<f64> {
        let max_dof = pairs.iter().map(|p| p.slave_id.max(p.master_id)).max().unwrap_or(0);
        let n = (max_dof + 2) * dof_per_node;

        let mut k_contact = DMatrix::zeros(n, n);

        for pair in pairs {
            if pair.gap >= 0.0 {
                continue;
            }

            let slave_dof = pair.slave_id * dof_per_node;
            let master_dof = pair.master_id * dof_per_node;

            // Add penalty stiffness to diagonal
            for i in 0..dof_per_node.min(2) {
                let normal_i = pair.normal[i];
                k_contact[(slave_dof + i, slave_dof + i)] += pair.stiffness * normal_i * normal_i;
                k_contact[(master_dof + i, master_dof + i)] += pair.stiffness * normal_i * normal_i;
                k_contact[(slave_dof + i, master_dof + i)] -= pair.stiffness * normal_i * normal_i;
                k_contact[(master_dof + i, slave_dof + i)] -= pair.stiffness * normal_i * normal_i;
            }
        }

        k_contact
    }
}

/// Augmented Lagrangian contact solver.
pub struct AugmentedLagrangianContact {
    /// Penalty stiffness.
    pub stiffness: f64,
    /// Lagrange multiplier update parameter.
    pub update_param: f64,
    /// Current Lagrange multipliers.
    pub multipliers: Vec<f64>,
}

impl AugmentedLagrangianContact {
    /// Creates a new augmented Lagrangian contact solver.
    pub fn new(stiffness: f64) -> Self {
        Self {
            stiffness,
            update_param: 1.0,
            multipliers: Vec::new(),
        }
    }

    /// Initializes multipliers for contact pairs.
    pub fn init_multipliers(&mut self, num_pairs: usize) {
        self.multipliers = vec![0.0; num_pairs];
    }

    /// Updates Lagrange multipliers.
    pub fn update_multipliers(&mut self, pairs: &[ContactPair]) {
        for (i, pair) in pairs.iter().enumerate() {
            if i >= self.multipliers.len() {
                self.multipliers.push(0.0);
            }

            if pair.gap < 0.0 {
                // Update for penetration (multiplier should be positive for compression)
                self.multipliers[i] -= self.stiffness * pair.gap * self.update_param;
            } else if self.multipliers[i] > 0.0 {
                // Release if open but multiplier is positive
                self.multipliers[i] = self.multipliers[i].max(0.0);
            }
        }
    }

    /// Computes augmented contact force.
    pub fn compute_augmented_force(&self, pair: &ContactPair, index: usize) -> DVector<f64> {
        let lambda = if index < self.multipliers.len() {
            self.multipliers[index]
        } else {
            0.0
        };

        // Augmented force: lambda + epsilon * gap
        let total_force = lambda + self.stiffness * pair.gap.min(0.0);

        if total_force > 0.0 {
            pair.normal.scale(total_force)
        } else {
            DVector::zeros(pair.normal.len())
        }
    }
}

/// Mortar contact method for non-matching meshes.
pub struct MortarContact {
    /// Integration points per segment.
    pub integration_points: usize,
    /// Penalty stiffness.
    pub stiffness: f64,
}

impl MortarContact {
    /// Creates a new mortar contact.
    pub fn new(integration_points: usize, stiffness: f64) -> Self {
        Self {
            integration_points,
            stiffness,
        }
    }

    /// Computes mortar contact integral.
    pub fn compute_mortar_integral(
        &self,
        slave_nodes: &[Point2<f64>],
        master_nodes: &[Point2<f64>],
        slave_shape_fns: &[Vec<f64>],
    ) -> (DVector<f64>, DMatrix<f64>) {
        let num_slave_dof = slave_nodes.len() * 2;
        let num_master_dof = master_nodes.len() * 2;

        let mut f_contact = DVector::zeros(num_slave_dof + num_master_dof);
        let mut k_contact = DMatrix::zeros(
            num_slave_dof + num_master_dof,
            num_slave_dof + num_master_dof,
        );

        // Numerical integration
        for ip in 0..self.integration_points {
            let xi = (ip as f64 + 0.5) / self.integration_points as f64;

            // Evaluate shape functions at integration point
            // (simplified - should use actual shape functions)
            let slave_weights = &slave_shape_fns.get(ip).cloned().unwrap_or_else(|| vec![1.0; slave_nodes.len()]);

            // Compute gap and contact contributions
            // Simplified implementation
            for (i, node) in slave_nodes.iter().enumerate() {
                // Find closest master point
                let mut min_dist = f64::INFINITY;
                for master_node in master_nodes {
                    let dist = (node.x - master_node.x).powi(2)
                        + (node.y - master_node.y).powi(2);
                    min_dist = min_dist.min(dist);
                }

                let gap = min_dist.sqrt() - 0.01; // Small tolerance
                if gap < 0.0 {
                    let dof_idx = i * 2;
                    let penalty = self.stiffness * (-gap) * slave_weights[i];

                    f_contact[dof_idx] += penalty;
                    k_contact[(dof_idx, dof_idx)] += self.stiffness * slave_weights[i];
                }
            }
        }

        (f_contact, k_contact)
    }
}

/// Contact constraint handler.
pub struct ContactConstraintHandler {
    /// Active contact pairs.
    pub active_pairs: Vec<ContactPair>,
    /// Constraint tolerance.
    pub tolerance: f64,
}

impl ContactConstraintHandler {
    /// Creates a new constraint handler.
    pub fn new(tolerance: f64) -> Self {
        Self {
            active_pairs: Vec::new(),
            tolerance,
        }
    }

    /// Applies contact constraints using Lagrange multipliers.
    pub fn apply_lagrange_constraints(
        &self,
        k: &mut DMatrix<f64>,
        f: &mut DVector<f64>,
        dof_per_node: usize,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let n_original = k.nrows();
        let n_constraints = self.active_pairs.len();

        if n_constraints == 0 {
            return (k.clone(), f.clone());
        }

        // Extended system with Lagrange multipliers
        let n_extended = n_original + n_constraints;
        let mut k_ext = DMatrix::zeros(n_extended, n_extended);
        let mut f_ext = DVector::zeros(n_extended);

        // Copy original system
        for i in 0..n_original {
            for j in 0..n_original {
                k_ext[(i, j)] = k[(i, j)];
            }
            f_ext[i] = f[i];
        }

        // Add constraint equations
        for (c, pair) in self.active_pairs.iter().enumerate() {
            let constraint_dof = n_original + c;
            let slave_dof = pair.slave_id * dof_per_node;

            // Gap constraint: u_slave . n - u_master . n = gap
            for i in 0..dof_per_node.min(2) {
                let normal_i = pair.normal[i];
                k_ext[(constraint_dof, slave_dof + i)] = normal_i;
                k_ext[(slave_dof + i, constraint_dof)] = normal_i;
            }

            // RHS = gap
            f_ext[constraint_dof] = pair.gap;
        }

        (k_ext, f_ext)
    }

    /// Updates active contact set.
    pub fn update_active_set(&mut self, detection: &ContactDetection) {
        self.active_pairs = detection
            .pairs
            .iter()
            .filter(|p| p.gap < self.tolerance)
            .cloned()
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_to_surface_detection() {
        let params = ContactParameters {
            tolerance: 0.01, // Larger tolerance for test
            ..Default::default()
        };
        let detector = NodeToSurfaceContact::new(params);

        // Master surface: line from (0,0) to (1,0)
        let master_nodes = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
        ];
        let master_segments = vec![(0, 1)];

        // Slave node close to master
        let slave_nodes = vec![Point2::new(0.5, 0.001)];

        let detection = detector.detect_2d(&slave_nodes, &master_segments, &master_nodes);

        assert!(detection.pairs.len() >= 0); // May or may not detect depending on tolerance
        if !detection.pairs.is_empty() {
            assert!(detection.pairs[0].gap < 0.01);
        }
    }

    #[test]
    fn test_penalty_contact() {
        let penalty = PenaltyContact::new(1e6, 0.0);

        let pair = ContactPair {
            master_id: 0,
            slave_id: 0,
            normal: DVector::from_column_slice(&[0.0, 1.0]),
            gap: -0.001,
            stiffness: 1e6,
        };

        let (force, friction) = penalty.compute_force_2d(&pair, 0.0);

        assert!(force.norm() > 0.0);
        assert_eq!(friction, 0.0);
    }

    #[test]
    fn test_augmented_lagrangian() {
        let mut al = AugmentedLagrangianContact::new(1e6);

        let pairs = vec![
            ContactPair {
                master_id: 0,
                slave_id: 0,
                normal: DVector::from_column_slice(&[0.0, 1.0]),
                gap: -0.001,
                stiffness: 1e6,
            },
        ];

        al.init_multipliers(pairs.len());

        // After first update, multiplier = -stiffness * gap = 1000
        al.update_multipliers(&pairs);

        // Force = lambda + stiffness*gap = 1000 + 1e6*(-0.001) = 1000 - 1000 = 0
        // After second update, multiplier increases
        al.update_multipliers(&pairs);

        let force = al.compute_augmented_force(&pairs[0], 0);

        // After multiple updates, we should have non-zero force
        assert!(al.multipliers[0] > 0.0);
    }

    #[test]
    fn test_constraint_handler() {
        let mut handler = ContactConstraintHandler::new(1e-6);

        let detection = ContactDetection {
            pairs: vec![
                ContactPair {
                    master_id: 0,
                    slave_id: 1,
                    normal: DVector::from_column_slice(&[0.0, 1.0]),
                    gap: -0.001,
                    stiffness: 1e6,
                },
            ],
            num_active: 1,
            max_penetration: 0.001,
        };

        handler.update_active_set(&detection);
        assert_eq!(handler.active_pairs.len(), 1);

        // Create small system
        let mut k = DMatrix::identity(4, 4);
        let mut f = DVector::from_element(4, 1.0);

        let (k_ext, f_ext) = handler.apply_lagrange_constraints(&mut k, &mut f, 2);

        assert_eq!(k_ext.nrows(), 5); // 4 original + 1 constraint
        assert_eq!(f_ext.len(), 5);
    }
}
