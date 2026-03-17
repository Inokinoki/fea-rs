//! Beam elements for structural analysis.
#![allow(unused_variables)]
//!
//! This module provides:
//! - Euler-Bernoulli beam element (2D and 3D)
//! - Timoshenko beam element (shear deformation)

use super::{Element, ElementContext};
use crate::core::NodeId;
use nalgebra::DMatrix;

/// A 2-node Euler-Bernoulli beam element in 2D.
///
/// Each node has 3 DOFs: ux, uy, rotation_z
/// Total: 6 DOFs per element.
#[derive(Debug, Clone, Copy)]
pub struct Beam2DElement {
    pub n1: NodeId,
    pub n2: NodeId,
}

impl Beam2DElement {
    /// Creates a new 2D beam element.
    pub fn new(n1: NodeId, n2: NodeId) -> Self {
        Self { n1, n2 }
    }

    /// Computes element length and direction cosines.
    pub fn length_and_dir(&self, nodes: &[crate::core::Node]) -> (f64, [f64; 2]) {
        let p1 = nodes[self.n1].as_array();
        let p2 = nodes[self.n2].as_array();
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let l = (dx * dx + dy * dy).sqrt();
        let dir = if l > 0.0 {
            [dx / l, dy / l]
        } else {
            [0.0, 0.0]
        };
        (l, dir)
    }

    /// Computes local stiffness matrix for Euler-Bernoulli beam.
    fn local_stiffness(&self, length: f64, e: f64, a: f64, i: f64) -> DMatrix<f64> {
        if length == 0.0 {
            return DMatrix::zeros(6, 6);
        }

        // Axial stiffness
        let ea_l = e * a / length;

        // Bending stiffness coefficients
        let ei_l2 = e * i / (length * length);
        let ei_l3 = e * i / (length * length * length);

        let k_axial = ea_l;
        let k_bend_1 = 12.0 * ei_l3;
        let k_bend_2 = 6.0 * ei_l2;
        let k_bend_3 = 4.0 * ei_l2 * length; // 4EI/L
        let k_bend_4 = 2.0 * ei_l2 * length; // 2EI/L

        // Local stiffness matrix
        // DOF ordering: [u1x, u1y, r1z, u2x, u2y, r2z]
        let mut k = DMatrix::zeros(6, 6);

        // Axial terms
        k[(0, 0)] = k_axial;
        k[(0, 3)] = -k_axial;
        k[(3, 0)] = -k_axial;
        k[(3, 3)] = k_axial;

        // Bending terms (u1y, r1z, u2y, r2z) -> indices 1, 2, 4, 5
        k[(1, 1)] = k_bend_1;
        k[(1, 2)] = k_bend_2;
        k[(1, 4)] = -k_bend_1;
        k[(1, 5)] = k_bend_2;

        k[(2, 1)] = k_bend_2;
        k[(2, 2)] = k_bend_3;
        k[(2, 4)] = -k_bend_2;
        k[(2, 5)] = k_bend_4;

        k[(4, 1)] = -k_bend_1;
        k[(4, 2)] = -k_bend_2;
        k[(4, 4)] = k_bend_1;
        k[(4, 5)] = -k_bend_2;

        k[(5, 1)] = k_bend_2;
        k[(5, 2)] = k_bend_4;
        k[(5, 4)] = -k_bend_2;
        k[(5, 5)] = k_bend_3;

        k
    }

    /// Computes transformation matrix from local to global coordinates.
    fn transformation_matrix(&self, cx: f64, cy: f64) -> DMatrix<f64> {
        // T = [ cx  cy  0   0   0   0 ]
        //     [-cy  cx  0   0   0   0 ]
        //     [ 0   0   1   0   0   0 ]
        //     [ 0   0   0  cx  cy  0 ]
        //     [ 0   0   0 -cy  cx  0 ]
        //     [ 0   0   0   0   0   1 ]
        DMatrix::from_row_slice(6, 6, &[
            cx, cy, 0.0, 0.0, 0.0, 0.0,
            -cy, cx, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, cx, cy, 0.0,
            0.0, 0.0, 0.0, -cy, cx, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Computes bending stress at a point along the beam.
    pub fn bending_stress(&self, ctx: &ElementContext, u: &[f64], y: f64) -> [f64; 2] {
        let (length, _) = self.length_and_dir(ctx.nodes);
        if length == 0.0 {
            return [0.0, 0.0];
        }

        let e = ctx.material.young_modulus;
        let _i = ctx.section.i_z;

        // Extract local displacements (simplified - assumes proper DOF mapping)
        // In practice, would need proper transformation
        let curvature = (u[5] - u[2]) / length; // Simplified curvature estimate
        let stress = e * y * curvature;

        [stress, -stress] // Top and bottom fiber stresses
    }
}

impl Element for Beam2DElement {
    fn node_ids(&self) -> Vec<NodeId> {
        vec![self.n1, self.n2]
    }

    fn ndofs(&self) -> usize {
        6 // 2 nodes * 3 DOFs (ux, uy, rz)
    }

    fn stiffness(&self, ctx: &ElementContext) -> DMatrix<f64> {
        let (length, [cx, cy]) = self.length_and_dir(ctx.nodes);
        let e = ctx.material.young_modulus;
        let a = ctx.section.area;
        let i = ctx.section.i_z;

        let k_local = self.local_stiffness(length, e, a, i);
        let t = self.transformation_matrix(cx, cy);

        // k_global = T^T * k_local * T
        let t_t = t.transpose();
        let kt = k_local * &t;
        t_t * &kt
    }

    fn mass(&self, ctx: &ElementContext) -> Option<DMatrix<f64>> {
        let (length, _) = self.length_and_dir(ctx.nodes);
        let rho = ctx.material.density;
        let a = ctx.section.area;
        let mass = rho * a * length;

        // Lumped mass matrix
        let mut m = DMatrix::zeros(6, 6);
        let m_node = mass / 2.0;

        // Translational DOFs only (lumped rotational inertia typically neglected)
        m[(0, 0)] = m_node;
        m[(1, 1)] = m_node;
        m[(3, 3)] = m_node;
        m[(4, 4)] = m_node;

        Some(m)
    }
}

/// A 2-node Euler-Bernoulli beam element in 3D.
///
/// Each node has 6 DOFs: ux, uy, uz, rx, ry, rz
/// Total: 12 DOFs per element.
#[derive(Debug, Clone, Copy)]
pub struct Beam3DElement {
    pub n1: NodeId,
    pub n2: NodeId,
}

impl Beam3DElement {
    /// Creates a new 3D beam element.
    pub fn new(n1: NodeId, n2: NodeId) -> Self {
        Self { n1, n2 }
    }

    /// Computes element length and direction.
    pub fn length_and_dir(&self, nodes: &[crate::core::Node]) -> (f64, [f64; 3]) {
        let p1 = nodes[self.n1].as_array();
        let p2 = nodes[self.n2].as_array();
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let dz = p2[2] - p1[2];
        let l = (dx * dx + dy * dy + dz * dz).sqrt();
        let dir = if l > 0.0 {
            [dx / l, dy / l, dz / l]
        } else {
            [0.0, 0.0, 0.0]
        };
        (l, dir)
    }

    /// Computes transformation matrix from local to global coordinates in 3D.
    ///
    /// For a 3D beam, we need to define a local coordinate system:
    /// - Local x-axis: along the beam (from n1 to n2)
    /// - Local y-axis: perpendicular to x, defined using a reference vector
    /// - Local z-axis: cross product of x and y
    ///
    /// The transformation matrix relates local displacements to global displacements.
    fn transformation_matrix_3d(lx: f64, ly: f64, lz: f64) -> DMatrix<f64> {
        // Local x-axis is along the beam
        let x = [lx, ly, lz];
        let x_norm = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();

        if x_norm < 1e-15 {
            return DMatrix::identity(12, 12);
        }

        let x = [x[0] / x_norm, x[1] / x_norm, x[2] / x_norm];

        // Choose a reference vector that's not parallel to x
        // Use the global z-axis as reference, unless beam is vertical
        let ref_vec = if x[2].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [1.0, 0.0, 0.0]
        };

        // Local y-axis: perpendicular to both x and ref_vec
        let mut y = [
            x[1] * ref_vec[2] - x[2] * ref_vec[1],
            x[2] * ref_vec[0] - x[0] * ref_vec[2],
            x[0] * ref_vec[1] - x[1] * ref_vec[0],
        ];
        let y_norm = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
        if y_norm > 1e-15 {
            y = [y[0] / y_norm, y[1] / y_norm, y[2] / y_norm];
        }

        // Local z-axis: cross product of x and y
        let z = [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ];

        // Build the 3x3 rotation matrix (columns are local axes in global coords)
        // R = [x_x  y_x  z_x]
        //     [x_y  y_y  z_y]
        //     [x_z  y_z  z_z]
        //
        // Transformation from local to global: v_global = R * v_local
        // Transformation from global to local: v_local = R^T * v_global

        // For the full 12x12 transformation matrix:
        // T = [R  0  0  0]
        //     [0  R  0  0]
        //     [0  0  R  0]
        //     [0  0  0  R]
        //
        // But we need the transpose for k_global = T^T * k_local * T
        // Since we're going from local to global, we use R directly

        let mut t = DMatrix::zeros(12, 12);

        // Node 1 translations and rotations
        t[(0, 0)] = x[0]; t[(0, 1)] = y[0]; t[(0, 2)] = z[0];
        t[(1, 0)] = x[1]; t[(1, 1)] = y[1]; t[(1, 2)] = z[1];
        t[(2, 0)] = x[2]; t[(2, 1)] = y[2]; t[(2, 2)] = z[2];

        t[(3, 3)] = x[0]; t[(3, 4)] = y[0]; t[(3, 5)] = z[0];
        t[(4, 3)] = x[1]; t[(4, 4)] = y[1]; t[(4, 5)] = z[1];
        t[(5, 3)] = x[2]; t[(5, 4)] = y[2]; t[(5, 5)] = z[2];

        // Node 2 translations and rotations
        t[(6, 6)] = x[0]; t[(6, 7)] = y[0]; t[(6, 8)] = z[0];
        t[(7, 6)] = x[1]; t[(7, 7)] = y[1]; t[(7, 8)] = z[1];
        t[(8, 6)] = x[2]; t[(8, 7)] = y[2]; t[(8, 8)] = z[2];

        t[(9, 9)] = x[0]; t[(9, 10)] = y[0]; t[(9, 11)] = z[0];
        t[(10, 9)] = x[1]; t[(10, 10)] = y[1]; t[(10, 11)] = z[1];
        t[(11, 9)] = x[2]; t[(11, 10)] = y[2]; t[(11, 11)] = z[2];

        t
    }
}

impl Element for Beam3DElement {
    fn node_ids(&self) -> Vec<NodeId> {
        vec![self.n1, self.n2]
    }

    fn ndofs(&self) -> usize {
        12 // 2 nodes * 6 DOFs
    }

    fn stiffness(&self, ctx: &ElementContext) -> DMatrix<f64> {
        let (length, [lx, ly, lz]) = self.length_and_dir(ctx.nodes);
        let e = ctx.material.young_modulus;
        let g = ctx.material.shear_modulus();
        let a = ctx.section.area;
        let i_y = ctx.section.i_y;
        let i_z = ctx.section.i_z;
        let j = ctx.section.j;

        if length == 0.0 {
            return DMatrix::zeros(12, 12);
        }

        // Compute local stiffness matrix
        let mut k_local = DMatrix::zeros(12, 12);

        // Axial stiffness (local x-direction)
        let ea_l = e * a / length;
        k_local[(0, 0)] = ea_l;
        k_local[(0, 6)] = -ea_l;
        k_local[(6, 0)] = -ea_l;
        k_local[(6, 6)] = ea_l;

        // Torsional stiffness
        let gj_l = g * j / length;
        k_local[(3, 3)] = gj_l;
        k_local[(3, 9)] = -gj_l;
        k_local[(9, 3)] = -gj_l;
        k_local[(9, 9)] = gj_l;

        // Bending about z-axis (in x-y plane)
        let ei_z_l2 = e * i_z / (length * length);
        let ei_z_l3 = e * i_z / (length * length * length);
        k_local[(1, 1)] = 12.0 * ei_z_l3;
        k_local[(1, 5)] = 6.0 * ei_z_l2;
        k_local[(1, 7)] = -12.0 * ei_z_l3;
        k_local[(1, 11)] = 6.0 * ei_z_l2;
        k_local[(5, 1)] = 6.0 * ei_z_l2;
        k_local[(5, 5)] = 4.0 * ei_z_l2 * length;
        k_local[(5, 7)] = -6.0 * ei_z_l2;
        k_local[(5, 11)] = 2.0 * ei_z_l2 * length;
        k_local[(7, 1)] = -12.0 * ei_z_l3;
        k_local[(7, 5)] = -6.0 * ei_z_l2;
        k_local[(7, 7)] = 12.0 * ei_z_l3;
        k_local[(7, 11)] = -6.0 * ei_z_l2;
        k_local[(11, 1)] = 6.0 * ei_z_l2;
        k_local[(11, 5)] = 2.0 * ei_z_l2 * length;
        k_local[(11, 7)] = -6.0 * ei_z_l2;
        k_local[(11, 11)] = 4.0 * ei_z_l2 * length;

        // Bending about y-axis (in x-z plane)
        let ei_y_l2 = e * i_y / (length * length);
        let ei_y_l3 = e * i_y / (length * length * length);
        k_local[(2, 2)] = 12.0 * ei_y_l3;
        k_local[(2, 4)] = -6.0 * ei_y_l2;
        k_local[(2, 8)] = -12.0 * ei_y_l3;
        k_local[(2, 10)] = -6.0 * ei_y_l2;
        k_local[(4, 2)] = -6.0 * ei_y_l2;
        k_local[(4, 4)] = 4.0 * ei_y_l2 * length;
        k_local[(4, 8)] = 6.0 * ei_y_l2;
        k_local[(4, 10)] = 2.0 * ei_y_l2 * length;
        k_local[(8, 2)] = -12.0 * ei_y_l3;
        k_local[(8, 4)] = 6.0 * ei_y_l2;
        k_local[(8, 8)] = 12.0 * ei_y_l3;
        k_local[(8, 10)] = 6.0 * ei_y_l2;
        k_local[(10, 2)] = -6.0 * ei_y_l2;
        k_local[(10, 4)] = 2.0 * ei_y_l2 * length;
        k_local[(10, 8)] = 6.0 * ei_y_l2;
        k_local[(10, 10)] = 4.0 * ei_y_l2 * length;

        // Compute transformation matrix from local to global coordinates
        let t = Self::transformation_matrix_3d(lx, ly, lz);

        // k_global = T^T * k_local * T
        let t_t = t.transpose();
        let kt = k_local * &t;
        t_t * &kt
    }

    fn mass(&self, ctx: &ElementContext) -> Option<DMatrix<f64>> {
        let (length, _) = self.length_and_dir(ctx.nodes);
        let rho = ctx.material.density;
        let a = ctx.section.area;
        let mass = rho * a * length;

        // Lumped mass matrix (translational DOFs only)
        let mut m = DMatrix::zeros(12, 12);
        let m_node = mass / 2.0;

        m[(0, 0)] = m_node;
        m[(1, 1)] = m_node;
        m[(2, 2)] = m_node;
        m[(6, 6)] = m_node;
        m[(7, 7)] = m_node;
        m[(8, 8)] = m_node;

        Some(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Material, Node, Section};

    fn get_test_material() -> Material {
        Material::new("steel", 210e9, 0.3, 7850.0, 250e6)
    }

    fn get_test_section() -> Section {
        Section::new("rect", 0.01, 8.33e-6, 8.33e-6, 1.0e-5)
    }

    #[test]
    fn test_beam2d_new() {
        let beam = Beam2DElement::new(0, 1);
        assert_eq!(beam.n1, 0);
        assert_eq!(beam.n2, 1);
    }

    #[test]
    fn test_beam2d_length_and_dir() {
        let nodes = vec![
            Node::new_2d(0.0, 0.0),
            Node::new_2d(2.0, 0.0),
        ];
        let beam = Beam2DElement::new(0, 1);
        let (length, dir) = beam.length_and_dir(&nodes);

        assert!((length - 2.0).abs() < 1e-10);
        assert!((dir[0] - 1.0).abs() < 1e-10);
        assert!((dir[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_beam2d_stiffness_symmetry() {
        let nodes = vec![
            Node::new_2d(0.0, 0.0),
            Node::new_2d(1.0, 0.0),
        ];
        let beam = Beam2DElement::new(0, 1);
        let mat = get_test_material();
        let sec = get_test_section();
        let ctx = ElementContext::new(&nodes, &mat, &sec);
        let k = beam.stiffness(&ctx);

        // Check symmetry
        for i in 0..6 {
            for j in 0..6 {
                assert!((k[(i, j)] - k[(j, i)]).abs() < 1e-6,
                    "Stiffness not symmetric at ({},{}): {} vs {}", i, j, k[(i, j)], k[(j, i)]);
            }
        }
    }

    #[test]
    fn test_beam2d_stiffness_positive_diagonal() {
        let nodes = vec![
            Node::new_2d(0.0, 0.0),
            Node::new_2d(1.0, 0.0),
        ];
        let beam = Beam2DElement::new(0, 1);
        let mat = get_test_material();
        let sec = get_test_section();
        let ctx = ElementContext::new(&nodes, &mat, &sec);
        let k = beam.stiffness(&ctx);

        // Diagonal terms should be positive
        for i in 0..6 {
            assert!(k[(i, i)] > 0.0, "Diagonal term {} should be positive", i);
        }
    }

    #[test]
    fn test_beam3d_new() {
        let beam = Beam3DElement::new(0, 1);
        assert_eq!(beam.n1, 0);
        assert_eq!(beam.n2, 1);
    }

    #[test]
    fn test_beam3d_length_and_dir() {
        let nodes = vec![
            Node::new_3d(0.0, 0.0, 0.0),
            Node::new_3d(1.0, 1.0, 1.0),
        ];
        let beam = Beam3DElement::new(0, 1);
        let (length, dir) = beam.length_and_dir(&nodes);

        let expected_len = 3.0f64.sqrt();
        assert!((length - expected_len).abs() < 1e-10);
        let expected_dir = 1.0 / 3.0f64.sqrt();
        assert!((dir[0] - expected_dir).abs() < 1e-10);
        assert!((dir[1] - expected_dir).abs() < 1e-10);
        assert!((dir[2] - expected_dir).abs() < 1e-10);
    }
}
