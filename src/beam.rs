//! 2D Beam element with rotational degrees of freedom.
//!
//! This module provides:
//! - Euler-Bernoulli beam element (2 nodes, 3 DOFs each: ux, uy, rotation)
//! - Stiffness matrix formulation
//! - Stress recovery

use crate::core::NodeId;
use nalgebra::{DMatrix, DVector};

/// Additional DOF for rotational degrees of freedom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RotationalDof {
    /// Rotation about X axis.
    Rx,
    /// Rotation about Y axis.
    Ry,
    /// Rotation about Z axis.
    Rz,
}

/// A 2D beam element (Euler-Bernoulli) with 6 DOFs:
/// - Node 1: ux, uy, rz
/// - Node 2: ux, uy, rz
#[derive(Debug, Clone, Copy)]
pub struct Beam2D {
    pub n1: NodeId,
    pub n2: NodeId,
    /// Young's modulus
    pub e: f64,
    /// Cross-sectional area
    pub a: f64,
    /// Area moment of inertia (about z-axis for 2D bending)
    pub i: f64,
}

impl Beam2D {
    pub fn new(n1: NodeId, n2: NodeId, e: f64, a: f64, i: f64) -> Self {
        Self { n1, n2, e, a, i }
    }

    /// Computes element length and direction cosines.
    pub fn length_and_dir(&self, model: &BeamModel) -> (f64, [f64; 2]) {
        let p1 = model.nodes[self.n1];
        let p2 = model.nodes[self.n2];
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

    /// Computes local stiffness matrix for the beam element.
    ///
    /// Local DOF ordering: [u1x, u1y, r1z, u2x, u2y, r2z]
    pub fn local_stiffness(&self, model: &BeamModel) -> DMatrix<f64> {
        let (l, [cx, cy]) = self.length_and_dir(model);
        if l == 0.0 {
            return DMatrix::zeros(6, 6);
        }

        let e = self.e;
        let a = self.a;
        let i = self.i;

        // Axial stiffness
        let ea_l = e * a / l;

        // Bending stiffness coefficients
        let ei_l2 = e * i / (l * l);
        let ei_l3 = e * i / (l * l * l);

        let k_axial = ea_l;
        let k_bend_1 = 12.0 * ei_l3;
        let k_bend_2 = 6.0 * ei_l2;
        let k_bend_3 = 4.0 * ei_l2 * l; // = 4EI/L
        let k_bend_4 = 2.0 * ei_l2 * l; // = 2EI/L

        // Local stiffness matrix in local coordinates
        // Then transformed to global
        let mut k_local = DMatrix::zeros(6, 6);

        // Axial terms (local x-direction)
        k_local[(0, 0)] = k_axial;
        k_local[(0, 3)] = -k_axial;
        k_local[(3, 0)] = -k_axial;
        k_local[(3, 3)] = k_axial;

        // Bending terms (local y-direction and rotation)
        // Node 1 DOFs: 1 (y), 2 (rz)
        // Node 2 DOFs: 4 (y), 5 (rz)
        k_local[(1, 1)] = k_bend_1;
        k_local[(1, 2)] = k_bend_2;
        k_local[(1, 4)] = -k_bend_1;
        k_local[(1, 5)] = k_bend_2;

        k_local[(2, 1)] = k_bend_2;
        k_local[(2, 2)] = k_bend_3;
        k_local[(2, 4)] = -k_bend_2;
        k_local[(2, 5)] = k_bend_4;

        k_local[(4, 1)] = -k_bend_1;
        k_local[(4, 2)] = -k_bend_2;
        k_local[(4, 4)] = k_bend_1;
        k_local[(4, 5)] = -k_bend_2;

        k_local[(5, 1)] = k_bend_2;
        k_local[(5, 2)] = k_bend_4;
        k_local[(5, 4)] = -k_bend_2;
        k_local[(5, 5)] = k_bend_3;

        // Transform to global coordinates
        self.transform_stiffness(&k_local, cx, cy)
    }

    /// Transforms local stiffness to global coordinates.
    fn transform_stiffness(&self, k_local: &DMatrix<f64>, cx: f64, cy: f64) -> DMatrix<f64> {
        // Transformation matrix for 2D beam
        // T = [ cx  cy  0   0   0   0 ]
        //     [-cy  cx  0   0   0   0 ]
        //     [ 0   0   1   0   0   0 ]
        //     [ 0   0   0  cx  cy  0 ]
        //     [ 0   0   0 -cy  cx  0 ]
        //     [ 0   0   0   0   0   1 ]

        let t = DMatrix::from_row_slice(6, 6, &[
            cx, cy, 0.0, 0.0, 0.0, 0.0,
            -cy, cx, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, cx, cy, 0.0,
            0.0, 0.0, 0.0, -cy, cx, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);

        // k_global = T^T * k_local * T
        let t_t = t.transpose();
        let kt = k_local * &t;
        t_t * &kt
    }

    /// Computes bending stress at a point along the element.
    ///
    /// - `xi`: Natural coordinate (-1 to 1)
    /// - `y`: Distance from neutral axis
    /// - `u`: Global displacement vector
    pub fn bending_stress(&self, model: &BeamModel, u: &[f64], xi: f64, y: f64) -> f64 {
        let (l, _) = self.length_and_dir(model);
        if l == 0.0 {
            return 0.0;
        }

        // Get DOF indices (simplified - assumes standard ordering)
        // In a full implementation, would need proper DOF mapping
        let e = self.e;

        // Simplified stress calculation
        // Full implementation would interpolate displacements and compute curvature
        e * y * self.curvature(model, u, xi)
    }

    /// Computes curvature at a point along the element.
    fn curvature(&self, _model: &BeamModel, _u: &[f64], _xi: f64) -> f64 {
        // Simplified - full implementation would compute from shape functions
        0.0
    }
}

/// Node coordinate type for beam model.
pub type Coord2D = [f64; 2];

/// A beam-specific model structure.
#[derive(Debug)]
pub struct BeamModel {
    pub nodes: Vec<Coord2D>,
    pub elements: Vec<Beam2D>,
}

impl BeamModel {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            elements: Vec::new(),
        }
    }

    pub fn add_node(&mut self, coord: Coord2D) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(coord);
        id
    }

    pub fn add_element(&mut self, element: Beam2D) {
        self.elements.push(element);
    }
}

impl Default for BeamModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Assembles global stiffness matrix for beam model.
pub fn assemble_beam_stiffness(model: &BeamModel, ndof: usize) -> DMatrix<f64> {
    let mut k = DMatrix::<f64>::zeros(ndof, ndof);

    for e in &model.elements {
        let ke = e.local_stiffness(model);
        // Simplified assembly - assumes consecutive DOFs
        let n1_dof = e.n1 * 3;
        let n2_dof = e.n2 * 3;

        let indices = [
            n1_dof, n1_dof + 1, n1_dof + 2, // u1x, u1y, r1z
            n2_dof, n2_dof + 1, n2_dof + 2, // u2x, u2y, r2z
        ];

        for (a, &ia) in indices.iter().enumerate() {
            for (b, &ib) in indices.iter().enumerate() {
                if ia < ndof && ib < ndof {
                    k[(ia, ib)] += ke[(a, b)];
                }
            }
        }
    }

    k
}

/// Assembles global load vector for beam model.
pub fn assemble_beam_load(_model: &BeamModel, loads: &[(NodeId, usize, f64)], ndof: usize) -> DVector<f64> {
    let mut f = DVector::<f64>::zeros(ndof);
    for &(node, dof, value) in loads {
        let idx = node * 3 + dof;
        if idx < ndof {
            f[idx] += value;
        }
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_beam2d_new() {
        let beam = Beam2D::new(0, 1, 210e9, 1e-4, 1e-8);
        assert_eq!(beam.n1, 0);
        assert_eq!(beam.n2, 1);
        assert_eq!(beam.e, 210e9);
        assert_eq!(beam.a, 1e-4);
        assert_eq!(beam.i, 1e-8);
    }

    #[test]
    fn test_beam2d_length_and_dir_horizontal() {
        let mut model = BeamModel::new();
        let n0 = model.add_node([0.0, 0.0]);
        let n1 = model.add_node([2.0, 0.0]);

        let beam = Beam2D::new(n0, n1, 210e9, 1e-4, 1e-8);
        let (length, dir) = beam.length_and_dir(&model);

        assert_relative_eq!(length, 2.0, epsilon = 1e-12);
        assert_relative_eq!(dir[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(dir[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_beam2d_length_and_dir_vertical() {
        let mut model = BeamModel::new();
        let n0 = model.add_node([0.0, 0.0]);
        let n1 = model.add_node([0.0, 3.0]);

        let beam = Beam2D::new(n0, n1, 210e9, 1e-4, 1e-8);
        let (length, dir) = beam.length_and_dir(&model);

        assert_relative_eq!(length, 3.0, epsilon = 1e-12);
        assert_relative_eq!(dir[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(dir[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_beam2d_length_and_dir_diagonal() {
        let mut model = BeamModel::new();
        let n0 = model.add_node([0.0, 0.0]);
        let n1 = model.add_node([3.0, 4.0]);

        let beam = Beam2D::new(n0, n1, 210e9, 1e-4, 1e-8);
        let (length, dir) = beam.length_and_dir(&model);

        assert_relative_eq!(length, 5.0, epsilon = 1e-12);
        assert_relative_eq!(dir[0], 0.6, epsilon = 1e-12);
        assert_relative_eq!(dir[1], 0.8, epsilon = 1e-12);
    }

    #[test]
    fn test_beam_stiffness_symmetry() {
        let mut model = BeamModel::new();
        let n0 = model.add_node([0.0, 0.0]);
        let n1 = model.add_node([1.0, 0.0]);

        let beam = Beam2D::new(n0, n1, 210e9, 1e-4, 1e-8);
        let k = beam.local_stiffness(&model);

        // Check symmetry
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(k[(i, j)], k[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_beam_stiffness_positive_diagonal() {
        let mut model = BeamModel::new();
        let n0 = model.add_node([0.0, 0.0]);
        let n1 = model.add_node([1.0, 0.0]);

        let beam = Beam2D::new(n0, n1, 210e9, 1e-4, 1e-8);
        let k = beam.local_stiffness(&model);

        // Diagonal terms should be positive (stability)
        for i in 0..6 {
            assert!(k[(i, i)] > 0.0, "Diagonal term {} should be positive", i);
        }
    }

    #[test]
    fn test_beam_model_assembly() {
        let mut model = BeamModel::new();
        let n0 = model.add_node([0.0, 0.0]);
        let n1 = model.add_node([1.0, 0.0]);
        let n2 = model.add_node([2.0, 0.0]);

        model.add_element(Beam2D::new(n0, n1, 210e9, 1e-4, 1e-8));
        model.add_element(Beam2D::new(n1, n2, 210e9, 1e-4, 1e-8));

        let k = assemble_beam_stiffness(&model, 9); // 3 nodes * 3 DOFs

        assert_eq!(k.nrows(), 9);
        assert_eq!(k.ncols(), 9);
    }

    #[test]
    fn test_cantilever_beam_stiffness() {
        // Simple cantilever beam test
        let mut model = BeamModel::new();
        let n0 = model.add_node([0.0, 0.0]);
        let n1 = model.add_node([1.0, 0.0]);

        let e = 210e9;
        let a = 1e-4;
        let i = 1e-8;
        let l = 1.0;

        model.add_element(Beam2D::new(n0, n1, e, a, i));

        let k = assemble_beam_stiffness(&model, 6);

        // Check axial stiffness at node 1
        let expected_axial = e * a / l;
        assert_relative_eq!(k[(0, 0)], expected_axial, max_relative = 1e-10);

        // Check bending stiffness (vertical DOF at node 1)
        // For cantilever: k_yy = 12EI/L^3
        let expected_bending = 12.0 * e * i / (l * l * l);
        assert_relative_eq!(k[(1, 1)], expected_bending, max_relative = 1e-10);

        // Check rotational stiffness at node 1
        // For cantilever: k_rr = 4EI/L
        let expected_rotation = 4.0 * e * i / l;
        assert_relative_eq!(k[(2, 2)], expected_rotation, max_relative = 1e-10);
    }
}
