//! Contact mechanics algorithms for FEA.
#![allow(unused_variables)]
//!
//! This module provides:
//! - Contact detection algorithms
//! - Penalty method for contact constraints
//! - Lagrange multiplier method
//! - Augmented Lagrangian method
//! - Friction models (Coulomb, stick-slip)
//! - Node-to-surface and surface-to-surface contact

pub mod frictional;

use nalgebra::{DMatrix, DVector, Vector3};
use std::collections::HashMap;

// Re-export frictional contact types
pub use frictional::{
    CoulombFrictionModel, ContactState, FrictionalContactConstraint,
    PenaltyFrictionSolver, AugmentedLagrangianFrictionSolver,
    ContactDetectionResult, NodeToSurfaceDetection,
};

/// Contact constraint types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContactType {
    /// Node-to-surface contact.
    NodeToSurface,
    /// Surface-to-surface contact.
    SurfaceToSurface,
    /// Node-to-node contact.
    NodeToNode,
}

/// Friction model for contact.
#[derive(Debug, Clone)]
pub enum FrictionModel {
    /// No friction (frictionless contact).
    Frictionless,
    /// Coulomb friction with coefficient.
    Coulomb { mu: f64 },
    /// Stick-slip friction.
    StickSlip { mu_static: f64, mu_dynamic: f64 },
    /// Viscous friction.
    Viscous { viscosity: f64 },
}

impl FrictionModel {
    /// Creates a frictionless model.
    pub fn frictionless() -> Self {
        Self::Frictionless
    }

    /// Creates a Coulomb friction model.
    pub fn coulomb(mu: f64) -> Self {
        Self::Coulomb { mu: mu.max(0.0) }
    }

    /// Creates a stick-slip friction model.
    pub fn stick_slip(mu_static: f64, mu_dynamic: f64) -> Self {
        Self::StickSlip {
            mu_static: mu_static.max(0.0),
            mu_dynamic: mu_dynamic.max(0.0).min(mu_static),
        }
    }
}

/// Contact pair definition.
#[derive(Debug, Clone)]
pub struct ContactPair {
    /// Master surface node indices.
    pub master_nodes: Vec<usize>,
    /// Slave surface node indices.
    pub slave_nodes: Vec<usize>,
    /// Contact type.
    pub contact_type: ContactType,
    /// Friction model.
    pub friction: FrictionModel,
    /// Penalty stiffness parameter.
    pub penalty_stiffness: f64,
    /// Contact tolerance (gap threshold).
    pub tolerance: f64,
}

impl ContactPair {
    /// Creates a new contact pair.
    pub fn new(
        master_nodes: Vec<usize>,
        slave_nodes: Vec<usize>,
        contact_type: ContactType,
    ) -> Self {
        Self {
            master_nodes,
            slave_nodes,
            contact_type,
            friction: FrictionModel::frictionless(),
            penalty_stiffness: 1e6,
            tolerance: 1e-6,
        }
    }

    /// Sets the friction model.
    pub fn with_friction(mut self, friction: FrictionModel) -> Self {
        self.friction = friction;
        self
    }

    /// Sets the penalty stiffness.
    pub fn with_penalty(mut self, stiffness: f64) -> Self {
        self.penalty_stiffness = stiffness;
        self
    }

    /// Sets the contact tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
}

/// Contact detection result.
#[derive(Debug, Clone)]
pub struct ContactDetection {
    /// Slave node index.
    pub slave_node: usize,
    /// Closest master surface point.
    pub projection_point: [f64; 3],
    /// Master element indices.
    pub master_element: [usize; 3],
    /// Shape functions at projection.
    pub shape_functions: [f64; 3],
    /// Gap distance (positive = separation, zero or negative = penetration).
    pub gap: f64,
    /// Contact tolerance.
    pub tolerance: f64,
    /// Contact normal (from master to slave).
    pub normal: [f64; 3],
    /// Is in contact.
    pub in_contact: bool,
}

/// Contact constraint enforcement method.
#[derive(Debug, Clone, Copy)]
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

/// Contact manager for handling multiple contact pairs.
#[derive(Debug)]
pub struct ContactManager {
    /// Contact pairs.
    pairs: Vec<ContactPair>,
    /// Contact detection results.
    detections: Vec<ContactDetection>,
    /// Contact method.
    method: ContactMethod,
    /// Augmented Lagrange multipliers.
    lagrange_multipliers: HashMap<(usize, usize), f64>,
    /// Contact stiffness matrix.
    contact_stiffness: Option<DMatrix<f64>>,
    /// Contact force vector.
    contact_forces: Option<DVector<f64>>,
}

impl ContactManager {
    /// Creates a new contact manager.
    pub fn new() -> Self {
        Self {
            pairs: Vec::new(),
            detections: Vec::new(),
            method: ContactMethod::Penalty,
            lagrange_multipliers: HashMap::new(),
            contact_stiffness: None,
            contact_forces: None,
        }
    }

    /// Creates with specified contact method.
    pub fn with_method(method: ContactMethod) -> Self {
        Self {
            pairs: Vec::new(),
            detections: Vec::new(),
            method,
            lagrange_multipliers: HashMap::new(),
            contact_stiffness: None,
            contact_forces: None,
        }
    }

    /// Adds a contact pair.
    pub fn add_pair(&mut self, pair: ContactPair) {
        self.pairs.push(pair);
    }

    /// Returns the number of contact pairs.
    pub fn num_pairs(&self) -> usize {
        self.pairs.len()
    }

    /// Sets the contact method.
    pub fn set_method(&mut self, method: ContactMethod) {
        self.method = method;
    }

    /// Performs contact detection.
    pub fn detect(
        &mut self,
        node_positions: &HashMap<usize, [f64; 3]>,
    ) -> Vec<ContactDetection> {
        self.detections.clear();

        for pair in &self.pairs {
            for &slave_node in &pair.slave_nodes {
                if let Some(slave_pos) = node_positions.get(&slave_node) {
                    // Find closest point on master surface
                    let detection = self.find_closest_point(
                        slave_node,
                        slave_pos,
                        &pair.master_nodes,
                        node_positions,
                        pair.tolerance,
                    );

                    if let Some(det) = detection {
                        if det.in_contact {
                            self.detections.push(det);
                        }
                    }
                }
            }
        }

        self.detections.clone()
    }

    /// Finds the closest point on master surface to a slave node.
    ///
    /// This implementation projects the slave node onto the master surface
    /// by finding the closest point on master surface triangles.
    fn find_closest_point(
        &self,
        slave_node: usize,
        slave_pos: &[f64; 3],
        master_nodes: &[usize],
        node_positions: &HashMap<usize, [f64; 3]>,
        tolerance: f64,
    ) -> Option<ContactDetection> {
        // Build master surface triangles from node connectivity
        // For simplicity, we assume master nodes form a triangulated surface
        // In a full implementation, the master surface elements would be explicitly defined

        let master_positions: Vec<(usize, [f64; 3])> = master_nodes
            .iter()
            .filter_map(|&n| node_positions.get(&n).map(|&pos| (n, pos)))
            .collect();

        if master_positions.len() < 3 {
            // Fall back to closest node if not enough points for a triangle
            return self.find_closest_node(slave_node, slave_pos, master_nodes, node_positions, tolerance);
        }

        // Try to project onto master surface triangles
        let mut min_dist = f64::INFINITY;
        let mut closest_point = *slave_pos;
        let mut master_element = [0, 0, 0];
        let mut shape_functions = [1.0, 0.0, 0.0];
        let mut normal = [0.0, 0.0, 1.0];

        // Generate triangles from master nodes (simplified: use consecutive triplets)
        for i in 0..master_positions.len().saturating_sub(2) {
            for j in (i + 1)..master_positions.len().saturating_sub(1) {
                for k in (j + 1)..master_positions.len() {
                    let (n1, p1) = &master_positions[i];
                    let (n2, p2) = &master_positions[j];
                    let (n3, p3) = &master_positions[k];

                    // Check if points form a valid triangle (not collinear)
                    let edge1 = Vector3::new(
                        p2[0] - p1[0],
                        p2[1] - p1[1],
                        p2[2] - p1[2],
                    );
                    let edge2 = Vector3::new(
                        p3[0] - p1[0],
                        p3[1] - p1[1],
                        p3[2] - p1[2],
                    );
                    let triangle_normal = edge1.cross(&edge2);
                    let area = triangle_normal.norm();

                    if area < 1e-15 {
                        continue; // Degenerate triangle
                    }

                    // Project slave point onto triangle plane
                    let n = triangle_normal.normalize();
                    let v = Vector3::new(
                        slave_pos[0] - p1[0],
                        slave_pos[1] - p1[1],
                        slave_pos[2] - p1[2],
                    );

                    // Distance to plane
                    let dist_to_plane = v.dot(&n).abs();

                    // Project point onto plane
                    let projected = Vector3::new(
                        slave_pos[0] - dist_to_plane * n.x,
                        slave_pos[1] - dist_to_plane * n.y,
                        slave_pos[2] - dist_to_plane * n.z,
                    );

                    // Check if projected point is inside triangle using barycentric coordinates
                    let proj = [projected.x, projected.y, projected.z];
                    if let Some((bary, dist)) = self.point_triangle_closest(
                        slave_pos, &proj, p1, p2, p3, &n
                    ) {
                        if dist < min_dist {
                            min_dist = dist;
                            closest_point = proj;
                            master_element = [*n1, *n2, *n3];
                            shape_functions = bary;
                            normal = [n.x, n.y, n.z];
                        }
                    }
                }
            }
        }

        // If no valid projection found, fall back to closest node
        if min_dist == f64::INFINITY {
            return self.find_closest_node(slave_node, slave_pos, master_nodes, node_positions, tolerance);
        }

        let gap = min_dist;
        let in_contact = gap <= tolerance;

        // Ensure normal points from master to slave
        let normal_to_slave = if gap > 1e-15 {
            [
                (slave_pos[0] - closest_point[0]) / gap,
                (slave_pos[1] - closest_point[1]) / gap,
                (slave_pos[2] - closest_point[2]) / gap,
            ]
        } else {
            normal
        };

        Some(ContactDetection {
            slave_node,
            projection_point: closest_point,
            master_element,
            shape_functions,
            gap,
            tolerance,
            normal: normal_to_slave,
            in_contact,
        })
    }

    /// Finds closest point using simple node-to-node distance (fallback).
    fn find_closest_node(
        &self,
        slave_node: usize,
        slave_pos: &[f64; 3],
        master_nodes: &[usize],
        node_positions: &HashMap<usize, [f64; 3]>,
        tolerance: f64,
    ) -> Option<ContactDetection> {
        let mut min_dist = f64::INFINITY;
        let mut closest_node: Option<usize> = None;
        let mut closest_point = *slave_pos;

        for &master_node in master_nodes {
            if let Some(master_pos) = node_positions.get(&master_node) {
                let dist = self.distance(slave_pos, master_pos);
                if dist < min_dist {
                    min_dist = dist;
                    closest_node = Some(master_node);
                    closest_point = *master_pos;
                }
            }
        }

        if let Some(master_node) = closest_node {
            let gap = min_dist;
            let in_contact = gap <= tolerance;

            let normal = if gap > 1e-15 {
                [
                    (slave_pos[0] - closest_point[0]) / gap,
                    (slave_pos[1] - closest_point[1]) / gap,
                    (slave_pos[2] - closest_point[2]) / gap,
                ]
            } else {
                [0.0, 0.0, 1.0]
            };

            Some(ContactDetection {
                slave_node,
                projection_point: closest_point,
                master_element: [master_node, master_node, master_node],
                shape_functions: [1.0, 0.0, 0.0],
                gap,
                tolerance,
                normal,
                in_contact,
            })
        } else {
            None
        }
    }

    /// Computes closest point on triangle to a given point.
    /// Returns barycentric coordinates and distance if projection is valid.
    fn point_triangle_closest(
        &self,
        point: &[f64; 3],
        projected: &[f64; 3],
        p1: &[f64; 3],
        p2: &[f64; 3],
        p3: &[f64; 3],
        normal: &Vector3<f64>,
    ) -> Option<([f64; 3], f64)> {
        // Compute barycentric coordinates of projected point
        let v0 = Vector3::new(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
        let v1 = Vector3::new(p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]);
        let v2 = Vector3::new(
            projected[0] - p1[0],
            projected[1] - p1[1],
            projected[2] - p1[2],
        );

        let d00 = v0.dot(&v0);
        let d01 = v0.dot(&v1);
        let d11 = v1.dot(&v1);
        let d20 = v2.dot(&v0);
        let d21 = v2.dot(&v1);

        let denom = d00 * d11 - d01 * d01;
        if denom.abs() < 1e-15 {
            return None;
        }

        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        let u = 1.0 - v - w;

        // Check if point is inside triangle
        let inside = u >= 0.0 && v >= 0.0 && w >= 0.0;

        let bary = [u, v, w];
        let dist = self.distance(point, projected);

        if inside {
            Some((bary, dist))
        } else {
            // Check edges
            self.check_triangle_edges(point, p1, p2, p3, &bary)
        }
    }

    /// Checks triangle edges for closest point when projection is outside.
    fn check_triangle_edges(
        &self,
        point: &[f64; 3],
        p1: &[f64; 3],
        p2: &[f64; 3],
        p3: &[f64; 3],
        bary: &[f64; 3],
    ) -> Option<([f64; 3], f64)> {
        // Find closest point on each edge
        let (c1, d1) = self.closest_point_on_edge(point, p1, p2);
        let (c2, d2) = self.closest_point_on_edge(point, p2, p3);
        let (c3, d3) = self.closest_point_on_edge(point, p3, p1);

        // Return the closest one
        let (closest, dist) = if d1 <= d2 && d1 <= d3 {
            (c1, d1)
        } else if d2 <= d1 && d2 <= d3 {
            (c2, d2)
        } else {
            (c3, d3)
        };

        // Compute barycentric coordinates for the closest point
        let bary = if d1 <= d2 && d1 <= d3 {
            let t = ((point[0] - p1[0]) * (p2[0] - p1[0])
                + (point[1] - p1[1]) * (p2[1] - p1[1])
                + (point[2] - p1[2]) * (p2[2] - p1[2]))
                / ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2)).max(1e-15);
            let t = t.max(0.0).min(1.0);
            [1.0 - t, t, 0.0]
        } else if d2 <= d1 && d2 <= d3 {
            let t = ((point[0] - p2[0]) * (p3[0] - p2[0])
                + (point[1] - p2[1]) * (p3[1] - p2[1])
                + (point[2] - p2[2]) * (p3[2] - p2[2]))
                / ((p3[0] - p2[0]).powi(2) + (p3[1] - p2[1]).powi(2) + (p3[2] - p2[2]).powi(2)).max(1e-15);
            let t = t.max(0.0).min(1.0);
            [0.0, 1.0 - t, t]
        } else {
            let t = ((point[0] - p3[0]) * (p1[0] - p3[0])
                + (point[1] - p3[1]) * (p1[1] - p3[1])
                + (point[2] - p3[2]) * (p1[2] - p3[2]))
                / ((p1[0] - p3[0]).powi(2) + (p1[1] - p3[1]).powi(2) + (p1[2] - p3[2]).powi(2)).max(1e-15);
            let t = t.max(0.0).min(1.0);
            [t, 0.0, 1.0 - t]
        };

        Some((bary, dist))
    }

    /// Computes closest point on edge and distance.
    fn closest_point_on_edge(
        &self,
        point: &[f64; 3],
        a: &[f64; 3],
        b: &[f64; 3],
    ) -> ([f64; 3], f64) {
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ab_len_sq = ab[0].powi(2) + ab[1].powi(2) + ab[2].powi(2);

        if ab_len_sq < 1e-15 {
            return (*a, self.distance(point, a));
        }

        let t = ((point[0] - a[0]) * ab[0]
            + (point[1] - a[1]) * ab[1]
            + (point[2] - a[2]) * ab[2])
            / ab_len_sq;

        let t = t.max(0.0).min(1.0);

        let closest = [
            a[0] + t * ab[0],
            a[1] + t * ab[1],
            a[2] + t * ab[2],
        ];

        (closest, self.distance(point, &closest))
    }

    /// Computes distance between two points.
    fn distance(&self, p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        ((p1[0] - p2[0]).powi(2)
            + (p1[1] - p2[1]).powi(2)
            + (p1[2] - p2[2]).powi(2))
        .sqrt()
    }

    /// Computes contact forces using penalty method.
    pub fn compute_penalty_forces(&self, _node_positions: &HashMap<usize, [f64; 3]>) -> HashMap<usize, [f64; 3]> {
        let mut forces: HashMap<usize, [f64; 3]> = HashMap::new();

        for detection in &self.detections {
            if detection.in_contact {
                // Get penalty stiffness from contact pair
                let penalty = self.pairs.iter()
                    .find(|p| p.slave_nodes.contains(&detection.slave_node))
                    .map(|p| p.penalty_stiffness)
                    .unwrap_or(1e6);

                // Penalty force: F = k * (tolerance - gap) * n when gap < tolerance
                let penetration = detection.tolerance - detection.gap;
                if penetration > 0.0 {
                    let force_mag = penalty * penetration;
                    let slave_force = [
                        force_mag * detection.normal[0],
                        force_mag * detection.normal[1],
                        force_mag * detection.normal[2],
                    ];

                    // Add force to slave node
                    forces.entry(detection.slave_node)
                        .and_modify(|f| {
                            f[0] += slave_force[0];
                            f[1] += slave_force[1];
                            f[2] += slave_force[2];
                        })
                        .or_insert(slave_force);
                }
            }
        }

        forces
    }

    /// Computes contact forces using augmented Lagrangian method.
    pub fn compute_augmented_lagrange_forces(
        &mut self,
        node_positions: &HashMap<usize, [f64; 3]>,
    ) -> HashMap<usize, [f64; 3]> {
        let mut forces: HashMap<usize, [f64; 3]> = HashMap::new();

        for detection in &self.detections {
            if detection.in_contact {
                let penalty = self.pairs.iter()
                    .find(|p| p.slave_nodes.contains(&detection.slave_node))
                    .map(|p| p.penalty_stiffness)
                    .unwrap_or(1e6);

                let penetration = -detection.gap;
                let key = (detection.slave_node, detection.master_element[0]);

                // Augmented Lagrangian: F = -(lambda + k * gap) * n
                let lambda = self.lagrange_multipliers.get(&key).copied().unwrap_or(0.0);
                let force_mag = lambda + penalty * penetration;

                if force_mag > 0.0 {
                    let slave_force = [
                        force_mag * detection.normal[0],
                        force_mag * detection.normal[1],
                        force_mag * detection.normal[2],
                    ];

                    forces.entry(detection.slave_node)
                        .and_modify(|f| {
                            f[0] += slave_force[0];
                            f[1] += slave_force[1];
                            f[2] += slave_force[2];
                        })
                        .or_insert(slave_force);

                    // Update Lagrange multiplier
                    self.lagrange_multipliers.insert(key, force_mag.max(0.0));
                }
            }
        }

        forces
    }

    /// Computes friction forces.
    pub fn compute_friction_forces(
        &self,
        contact_forces: &HashMap<usize, [f64; 3]>,
        tangential_velocities: &HashMap<usize, [f64; 3]>,
    ) -> HashMap<usize, [f64; 3]> {
        let mut friction_forces: HashMap<usize, [f64; 3]> = HashMap::new();

        for (node, normal_force) in contact_forces {
            let pair = self.pairs.iter()
                .find(|p| p.slave_nodes.contains(node));

            if let Some(p) = pair {
                let friction_force = match &p.friction {
                    FrictionModel::Frictionless => [0.0, 0.0, 0.0],
                    FrictionModel::Coulomb { mu } => {
                        if let Some(vel) = tangential_velocities.get(node) {
                            let vel_mag = (vel[0].powi(2) + vel[1].powi(2) + vel[2].powi(2)).sqrt();
                            if vel_mag > 1e-15 {
                                let f_mag = mu * (normal_force[0].powi(2) + normal_force[1].powi(2) + normal_force[2].powi(2)).sqrt();
                                [-f_mag * vel[0] / vel_mag, -f_mag * vel[1] / vel_mag, -f_mag * vel[2] / vel_mag]
                            } else {
                                [0.0, 0.0, 0.0]
                            }
                        } else {
                            [0.0, 0.0, 0.0]
                        }
                    }
                    FrictionModel::StickSlip { mu_static, mu_dynamic } => {
                        if let Some(vel) = tangential_velocities.get(node) {
                            let vel_mag = (vel[0].powi(2) + vel[1].powi(2) + vel[2].powi(2)).sqrt();
                            let mu = if vel_mag > 0.01 { *mu_dynamic } else { *mu_static };
                            if vel_mag > 1e-15 {
                                let f_mag = mu * (normal_force[0].powi(2) + normal_force[1].powi(2) + normal_force[2].powi(2)).sqrt();
                                [-f_mag * vel[0] / vel_mag, -f_mag * vel[1] / vel_mag, -f_mag * vel[2] / vel_mag]
                            } else {
                                [0.0, 0.0, 0.0]
                            }
                        } else {
                            [0.0, 0.0, 0.0]
                        }
                    }
                    FrictionModel::Viscous { viscosity } => {
                        if let Some(vel) = tangential_velocities.get(node) {
                            [-viscosity * vel[0], -viscosity * vel[1], -viscosity * vel[2]]
                        } else {
                            [0.0, 0.0, 0.0]
                        }
                    }
                };

                friction_forces.insert(*node, friction_force);
            }
        }

        friction_forces
    }

    /// Assembles contact stiffness matrix contribution.
    pub fn assemble_contact_stiffness(&mut self, total_dofs: usize) -> DMatrix<f64> {
        let mut kc = DMatrix::zeros(total_dofs, total_dofs);

        for detection in &self.detections {
            if detection.in_contact {
                let penalty = self.pairs.iter()
                    .find(|p| p.slave_nodes.contains(&detection.slave_node))
                    .map(|p| p.penalty_stiffness)
                    .unwrap_or(1e6);

                let dof_base = detection.slave_node * 3;

                // Simplified contact stiffness (normal direction only)
                for i in 0..3 {
                    for j in 0..3 {
                        let k_ij = penalty * detection.normal[i] * detection.normal[j];
                        kc[(dof_base + i, dof_base + j)] += k_ij;
                    }
                }
            }
        }

        self.contact_stiffness = Some(kc.clone());
        kc
    }

    /// Resets contact state.
    pub fn reset(&mut self) {
        self.detections.clear();
        self.lagrange_multipliers.clear();
        self.contact_stiffness = None;
        self.contact_forces = None;
    }

    /// Updates Lagrange multipliers for augmented Lagrangian.
    pub fn update_lagrange_multipliers(&mut self) {
        // Multiplier update is done during force computation
    }
}

impl Default for ContactManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Contact results for post-processing.
#[derive(Debug, Clone)]
pub struct ContactResults {
    /// Number of active contacts.
    pub num_active_contacts: usize,
    /// Total contact force magnitude.
    pub total_contact_force: f64,
    /// Maximum contact pressure.
    pub max_contact_pressure: f64,
    /// Total friction force.
    pub total_friction_force: f64,
    /// Contact detections.
    pub detections: Vec<ContactDetection>,
}

impl ContactResults {
    pub fn new(detections: &[ContactDetection], contact_forces: &HashMap<usize, [f64; 3]>) -> Self {
        let num_active = detections.iter().filter(|d| d.in_contact).count();

        let total_force: f64 = contact_forces.values()
            .map(|f| (f[0].powi(2) + f[1].powi(2) + f[2].powi(2)).sqrt())
            .sum();

        Self {
            num_active_contacts: num_active,
            total_contact_force: total_force,
            max_contact_pressure: 0.0, // Would need area information
            total_friction_force: 0.0,
            detections: detections.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contact_pair_creation() {
        let pair = ContactPair::new(
            vec![0, 1, 2],
            vec![3, 4, 5],
            ContactType::NodeToSurface,
        );

        assert_eq!(pair.master_nodes.len(), 3);
        assert_eq!(pair.slave_nodes.len(), 3);
        assert_eq!(pair.contact_type, ContactType::NodeToSurface);
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

        let stick_slip = FrictionModel::stick_slip(0.5, 0.3);
        if let FrictionModel::StickSlip { mu_static, mu_dynamic } = stick_slip {
            assert!((mu_static - 0.5).abs() < 1e-10);
            assert!((mu_dynamic - 0.3).abs() < 1e-10);
        } else {
            panic!("Expected stick-slip friction");
        }
    }

    #[test]
    fn test_contact_manager_creation() {
        let mgr = ContactManager::new();
        assert_eq!(mgr.num_pairs(), 0);
    }

    #[test]
    fn test_contact_detection() {
        let mut mgr = ContactManager::new();
        let mut pair = ContactPair::new(
            vec![0, 1, 2],
            vec![3],
            ContactType::NodeToSurface,
        );
        pair.tolerance = 1.0; // Large tolerance for test
        mgr.add_pair(pair);

        let mut positions = HashMap::new();
        positions.insert(0, [0.0, 0.0, 0.0]);
        positions.insert(1, [1.0, 0.0, 0.0]);
        positions.insert(2, [0.5, 1.0, 0.0]);
        positions.insert(3, [0.5, 0.5, 0.0]); // Slave node near master surface

        let detections = mgr.detect(&positions);
        assert!(!detections.is_empty());
    }

    #[test]
    fn test_penalty_forces() {
        let mut mgr = ContactManager::new();
        let mut pair = ContactPair::new(
            vec![0, 1],
            vec![2],
            ContactType::NodeToNode,
        );
        pair.penalty_stiffness = 1e6;
        pair.tolerance = 0.1;
        mgr.add_pair(pair);

        let mut positions = HashMap::new();
        positions.insert(0, [0.0, 0.0, 0.0]);
        positions.insert(1, [0.0, 0.0, 0.0]);
        positions.insert(2, [0.0, 0.0, 0.0]); // Same position = penetration

        let _ = mgr.detect(&positions);
        let forces = mgr.compute_penalty_forces(&positions);

        // Should have contact force on penetrating node
        assert!(!forces.is_empty());
    }

    #[test]
    fn test_contact_results() {
        let detections = vec![
            ContactDetection {
                slave_node: 0,
                projection_point: [0.0, 0.0, 0.0],
                master_element: [0, 0, 0],
                shape_functions: [1.0, 0.0, 0.0],
                gap: -0.01,
                tolerance: 0.1,
                normal: [0.0, 0.0, 1.0],
                in_contact: true,
            },
        ];

        let mut forces = HashMap::new();
        forces.insert(0, [100.0, 0.0, 0.0]);

        let results = ContactResults::new(&detections, &forces);
        assert_eq!(results.num_active_contacts, 1);
        assert!(results.total_contact_force > 0.0);
    }
}
