//! Frictional Contact Algorithms.
//!
//! This module provides frictional contact formulations including:
//! - Coulomb friction model
//! - Penalty friction method
//! - Augmented Lagrangian friction
//! - Stick-slip transition detection

use nalgebra::{DMatrix, DVector, Vector3};

/// Coulomb friction model parameters.
#[derive(Debug, Clone)]
pub struct CoulombFrictionModel {
    /// Static friction coefficient.
    pub mu_static: f64,
    /// Dynamic friction coefficient.
    pub mu_dynamic: f64,
    /// Stribeck velocity (for smooth transition).
    pub stribeck_velocity: f64,
}

impl Default for CoulombFrictionModel {
    fn default() -> Self {
        Self {
            mu_static: 0.3,
            mu_dynamic: 0.25,
            stribeck_velocity: 0.01,
        }
    }
}

impl CoulombFrictionModel {
    /// Creates a new Coulomb friction model.
    pub fn new(mu_static: f64, mu_dynamic: f64) -> Self {
        Self {
            mu_static,
            mu_dynamic,
            stribeck_velocity: 0.01,
        }
    }

    /// Computes friction coefficient based on slip velocity.
    pub fn friction_coefficient(&self, slip_velocity: f64) -> f64 {
        let v_abs = slip_velocity.abs();
        if v_abs < self.stribeck_velocity {
            // Stribeck region - smooth transition
            let ratio = v_abs / self.stribeck_velocity;
            self.mu_static - (self.mu_static - self.mu_dynamic) * ratio
        } else {
            self.mu_dynamic
        }
    }

    /// Computes friction force.
    pub fn friction_force(&self, normal_force: f64, slip_velocity: f64) -> f64 {
        let mu = self.friction_coefficient(slip_velocity);
        let friction_magnitude = mu * normal_force.abs();

        // Friction opposes slip direction
        let sign = if slip_velocity > 0.0 { -1.0 } else { 1.0 };
        friction_magnitude * sign
    }

    /// Checks if contact is sticking.
    pub fn is_sticking(&self, tangential_force: f64, normal_force: f64) -> bool {
        tangential_force.abs() < self.mu_static * normal_force.abs()
    }
}

/// Contact state enumeration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContactState {
    /// No contact.
    Open,
    /// Contact with no slip (sticking).
    Sticking,
    /// Contact with slip (sliding).
    Sliding,
}

/// Frictional contact constraint.
#[derive(Debug, Clone)]
pub struct FrictionalContactConstraint {
    /// Contact node ID.
    pub node_id: usize,
    /// Target surface ID.
    pub target_id: usize,
    /// Contact normal (pointing from target to contact).
    pub normal: Vector3<f64>,
    /// Contact tangent direction.
    pub tangent: Vector3<f64>,
    /// Current contact state.
    pub state: ContactState,
    /// Normal gap (negative = penetration).
    pub gap: f64,
    /// Tangential slip displacement.
    pub slip: f64,
    /// Normal contact force.
    pub normal_force: f64,
    /// Tangential friction force.
    pub friction_force: f64,
}

impl FrictionalContactConstraint {
    /// Creates a new frictional contact constraint.
    pub fn new(node_id: usize, target_id: usize, normal: Vector3<f64>) -> Self {
        let tangent = Self::compute_tangent(&normal);
        Self {
            node_id,
            target_id,
            normal,
            tangent,
            state: ContactState::Open,
            gap: 0.0,
            slip: 0.0,
            normal_force: 0.0,
            friction_force: 0.0,
        }
    }

    /// Computes tangent direction from normal.
    fn compute_tangent(normal: &Vector3<f64>) -> Vector3<f64> {
        // Find a vector perpendicular to normal
        if normal.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0).cross(normal).normalize()
        } else {
            Vector3::new(0.0, 1.0, 0.0).cross(normal).normalize()
        }
    }

    /// Updates contact state based on gap and forces.
    pub fn update_state(&mut self, friction_model: &CoulombFrictionModel) {
        if self.gap > 0.0 {
            self.state = ContactState::Open;
            self.normal_force = 0.0;
            self.friction_force = 0.0;
        } else if friction_model.is_sticking(self.friction_force, self.normal_force) {
            self.state = ContactState::Sticking;
        } else {
            self.state = ContactState::Sliding;
        }
    }
}

/// Penalty frictional contact solver.
#[derive(Debug, Clone)]
pub struct PenaltyFrictionSolver {
    /// Normal penalty parameter.
    pub normal_penalty: f64,
    /// Tangential penalty parameter.
    pub tangential_penalty: f64,
    /// Friction model.
    pub friction_model: CoulombFrictionModel,
    /// Maximum iterations for contact detection.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl Default for PenaltyFrictionSolver {
    fn default() -> Self {
        Self {
            normal_penalty: 1e6,
            tangential_penalty: 1e6,
            friction_model: CoulombFrictionModel::default(),
            max_iterations: 50,
            tolerance: 1e-6,
        }
    }
}

impl PenaltyFrictionSolver {
    /// Creates a new penalty friction solver.
    pub fn new(normal_penalty: f64, tangential_penalty: f64) -> Self {
        Self {
            normal_penalty,
            tangential_penalty,
            friction_model: CoulombFrictionModel::default(),
            max_iterations: 50,
            tolerance: 1e-6,
        }
    }

    /// Solves frictional contact for a single constraint.
    pub fn solve_constraint(
        &self,
        constraint: &mut FrictionalContactConstraint,
        displacement: &Vector3<f64>,
    ) -> Vector3<f64> {
        // Compute gap
        constraint.gap = displacement.dot(&constraint.normal);

        if constraint.gap > 0.0 {
            // No contact
            constraint.normal_force = 0.0;
            constraint.friction_force = 0.0;
            constraint.state = ContactState::Open;
            return Vector3::zeros();
        }

        // Compute normal force (penalty method)
        constraint.normal_force = -self.normal_penalty * constraint.gap;

        // Compute tangential slip
        let tangential_disp = displacement.dot(&constraint.tangent);
        constraint.slip += tangential_disp;

        // Compute friction force
        let mu = self.friction_model.friction_coefficient(constraint.slip);
        let max_friction = mu * constraint.normal_force;

        // Trial friction force (elastic predictor)
        let trial_friction = self.tangential_penalty * constraint.slip;

        if trial_friction.abs() <= max_friction {
            // Sticking
            constraint.friction_force = trial_friction;
            constraint.state = ContactState::Sticking;
        } else {
            // Sliding - return to yield surface
            constraint.friction_force = max_friction * trial_friction.signum();
            constraint.slip = constraint.friction_force / self.tangential_penalty;
            constraint.state = ContactState::Sliding;
        }

        // Compute contact force vector
        constraint.normal * constraint.normal_force + constraint.tangent * constraint.friction_force
    }

    /// Assembles contact contribution to global stiffness matrix.
    pub fn assemble_stiffness_contribution(
        &self,
        constraint: &FrictionalContactConstraint,
    ) -> DMatrix<f64> {
        let mut k_contact = DMatrix::zeros(3, 3);

        if constraint.state != ContactState::Open {
            // Normal stiffness (outer product)
            for i in 0..3 {
                for j in 0..3 {
                    k_contact[(i, j)] += constraint.normal[i] * constraint.normal[j] * self.normal_penalty;
                }
            }

            // Tangential stiffness (only for sticking)
            if constraint.state == ContactState::Sticking {
                for i in 0..3 {
                    for j in 0..3 {
                        k_contact[(i, j)] += constraint.tangent[i] * constraint.tangent[j] * self.tangential_penalty;
                    }
                }
            }
        }

        k_contact
    }
}

/// Augmented Lagrangian frictional contact solver.
#[derive(Debug, Clone)]
pub struct AugmentedLagrangianFrictionSolver {
    /// Normal penalty parameter.
    pub normal_penalty: f64,
    /// Tangential penalty parameter.
    pub tangential_penalty: f64,
    /// Friction model.
    pub friction_model: CoulombFrictionModel,
    /// Normal Lagrange multiplier.
    pub lambda_normal: f64,
    /// Tangential Lagrange multiplier.
    pub lambda_tangent: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl Default for AugmentedLagrangianFrictionSolver {
    fn default() -> Self {
        Self {
            normal_penalty: 1e5,
            tangential_penalty: 1e5,
            friction_model: CoulombFrictionModel::default(),
            lambda_normal: 0.0,
            lambda_tangent: 0.0,
            max_iterations: 50,
            tolerance: 1e-6,
        }
    }
}

impl AugmentedLagrangianFrictionSolver {
    /// Creates a new augmented Lagrangian friction solver.
    pub fn new(normal_penalty: f64, tangential_penalty: f64) -> Self {
        Self {
            normal_penalty,
            tangential_penalty,
            friction_model: CoulombFrictionModel::default(),
            lambda_normal: 0.0,
            lambda_tangent: 0.0,
            max_iterations: 50,
            tolerance: 1e-6,
        }
    }

    /// Solves frictional contact using augmented Lagrangian method.
    pub fn solve_constraint(
        &mut self,
        constraint: &mut FrictionalContactConstraint,
        displacement: &Vector3<f64>,
    ) -> Vector3<f64> {
        // Compute gap
        constraint.gap = displacement.dot(&constraint.normal);

        if constraint.gap > 0.0 {
            // No contact
            constraint.normal_force = 0.0;
            constraint.friction_force = 0.0;
            constraint.state = ContactState::Open;
            self.lambda_normal = 0.0;
            self.lambda_tangent = 0.0;
            return Vector3::zeros();
        }

        // Compute normal force (augmented Lagrangian)
        constraint.normal_force = self.lambda_normal - self.normal_penalty * constraint.gap;

        // Compute tangential slip
        let tangential_disp = displacement.dot(&constraint.tangent);
        constraint.slip += tangential_disp;

        // Trial friction force
        let trial_friction = self.lambda_tangent + self.tangential_penalty * constraint.slip;

        // Compute friction coefficient
        let mu = self.friction_model.friction_coefficient(constraint.slip);
        let max_friction = mu * constraint.normal_force;

        if trial_friction.abs() <= max_friction {
            // Sticking
            constraint.friction_force = trial_friction;
            constraint.state = ContactState::Sticking;
            self.lambda_tangent = trial_friction;
        } else {
            // Sliding
            constraint.friction_force = max_friction * trial_friction.signum();
            constraint.state = ContactState::Sliding;
            self.lambda_tangent = constraint.friction_force;
        }

        // Update normal Lagrange multiplier
        self.lambda_normal = constraint.normal_force;

        // Compute contact force vector
        constraint.normal * constraint.normal_force + constraint.tangent * constraint.friction_force
    }

    /// Resets Lagrange multipliers.
    pub fn reset_multipliers(&mut self) {
        self.lambda_normal = 0.0;
        self.lambda_tangent = 0.0;
    }
}

/// Contact detection result.
#[derive(Debug, Clone)]
pub struct ContactDetectionResult {
    /// Detected contact constraints.
    pub constraints: Vec<FrictionalContactConstraint>,
    /// Number of active contacts.
    pub num_active: usize,
    /// Maximum penetration.
    pub max_penetration: f64,
}

/// Node-to-surface contact detection.
pub struct NodeToSurfaceDetection {
    /// Search tolerance.
    pub search_tolerance: f64,
}

impl Default for NodeToSurfaceDetection {
    fn default() -> Self {
        Self {
            search_tolerance: 1e-3,
        }
    }
}

impl NodeToSurfaceDetection {
    /// Creates a new node-to-surface contact detector.
    pub fn new(search_tolerance: f64) -> Self {
        Self { search_tolerance }
    }

    /// Detects contact between nodes and target surface.
    pub fn detect(
        &self,
        node_positions: &[Vector3<f64>],
        target_triangles: &[(usize, usize, usize)],
        target_positions: &[Vector3<f64>],
    ) -> ContactDetectionResult {
        let mut constraints = Vec::new();
        let mut max_penetration: f64 = 0.0;

        for (node_id, node_pos) in node_positions.iter().enumerate() {
            // Find closest point on target surface
            let (closest_point, normal, target_id) =
                self.find_closest_point(node_pos, target_triangles, target_positions);

            // Compute gap
            let gap = (node_pos - closest_point).dot(&normal);

            if gap < self.search_tolerance {
                let mut constraint = FrictionalContactConstraint::new(node_id, target_id, normal);
                constraint.gap = gap;

                if gap < 0.0 {
                    max_penetration = max_penetration.max(-gap);
                }

                constraints.push(constraint);
            }
        }

        let num_active = constraints.iter().filter(|c| c.gap < 0.0).count();

        ContactDetectionResult {
            constraints,
            num_active,
            max_penetration,
        }
    }

    /// Finds closest point on target surface.
    fn find_closest_point(
        &self,
        point: &Vector3<f64>,
        triangles: &[(usize, usize, usize)],
        positions: &[Vector3<f64>],
    ) -> (Vector3<f64>, Vector3<f64>, usize) {
        let mut min_distance = f64::INFINITY;
        let mut closest_point = *point;
        let mut normal = Vector3::z();
        let mut target_id = 0;

        for (tri_id, &(i, j, k)) in triangles.iter().enumerate() {
            let p1 = positions[i];
            let p2 = positions[j];
            let p3 = positions[k];

            // Compute triangle normal
            let v1 = p2 - p1;
            let v2 = p3 - p1;
            let tri_normal = v1.cross(&v2).normalize();

            // Project point onto triangle plane
            let d = (point - p1).dot(&tri_normal);

            if d.abs() < min_distance {
                // Check if projection is inside triangle (simplified)
                min_distance = d.abs();
                closest_point = *point - tri_normal * d;
                normal = tri_normal;
                target_id = tri_id;
            }
        }

        (closest_point, normal, target_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coulomb_friction_model() {
        let model = CoulombFrictionModel::new(0.3, 0.25);

        // Test static friction
        assert!(model.is_sticking(0.2, 1.0));
        assert!(!model.is_sticking(0.4, 1.0));

        // Test friction coefficient
        assert!((model.friction_coefficient(0.0) - 0.3).abs() < 0.01);
        assert!((model.friction_coefficient(1.0) - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_penalty_friction_solver() {
        let solver = PenaltyFrictionSolver::new(1e6, 1e6);
        let mut constraint = FrictionalContactConstraint::new(
            0,
            0,
            Vector3::new(0.0, 1.0, 0.0),
        );

        // Test open contact
        let disp = Vector3::new(0.0, 0.01, 0.0);
        let force = solver.solve_constraint(&mut constraint, &disp);
        assert_eq!(constraint.state, ContactState::Open);
        assert!(force.norm() < 1e-10);

        // Test penetrating contact
        let disp = Vector3::new(0.0, -0.001, 0.0);
        let force = solver.solve_constraint(&mut constraint, &disp);
        assert_ne!(constraint.state, ContactState::Open);
        assert!(constraint.normal_force > 0.0);
    }

    #[test]
    fn test_contact_detection() {
        let detector = NodeToSurfaceDetection::default();

        let node_positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.5, -0.001, 0.0),  // Penetrating
        ];

        let target_triangles = vec![(0, 1, 2)];
        let target_positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.5, 0.0, 1.0),
        ];

        let result = detector.detect(&node_positions, &target_triangles, &target_positions);

        assert!(result.constraints.len() >= 1);
    }
}
