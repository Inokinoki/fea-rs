//! 2-node truss element in 3D space.
#![allow(unused_variables)]

use super::{Element, ElementContext};
use crate::core::NodeId;
use nalgebra::{DMatrix, DVector};

/// A 2-node truss/bar element in 3D (use z=0 for 2D).
///
/// This element uses material and section properties from the context.
#[derive(Debug, Clone, Copy)]
pub struct Truss2 {
    pub n1: NodeId,
    pub n2: NodeId,
}

impl Truss2 {
    /// Creates a new truss element.
    pub fn new(n1: NodeId, n2: NodeId) -> Self {
        Self { n1, n2 }
    }

    /// Computes element length and direction cosines.
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

    /// Element axial stress (positive in tension) based on nodal displacements.
    pub fn axial_stress(&self, ctx: &ElementContext, u: &[f64]) -> f64 {
        let (l, dir) = self.length_and_dir(ctx.nodes);
        if l == 0.0 {
            return 0.0;
        }

        let dof_indices = self.dof_indices();
        let u1 = [
            u.get(dof_indices[0]).copied().unwrap_or(0.0),
            u.get(dof_indices[1]).copied().unwrap_or(0.0),
            u.get(dof_indices[2]).copied().unwrap_or(0.0),
        ];
        let u2 = [
            u.get(dof_indices[3]).copied().unwrap_or(0.0),
            u.get(dof_indices[4]).copied().unwrap_or(0.0),
            u.get(dof_indices[5]).copied().unwrap_or(0.0),
        ];

        let du = [u2[0] - u1[0], u2[1] - u1[1], u2[2] - u1[2]];
        let delta_l = du[0] * dir[0] + du[1] * dir[1] + du[2] * dir[2];
        ctx.material.young_modulus * (delta_l / l)
    }

    /// Returns the DOF indices for this element (assumes 3 DOFs per node).
    fn dof_indices(&self) -> [usize; 6] {
        [
            self.n1 * 3,
            self.n1 * 3 + 1,
            self.n1 * 3 + 2,
            self.n2 * 3,
            self.n2 * 3 + 1,
            self.n2 * 3 + 2,
        ]
    }
}

impl Element for Truss2 {
    fn node_ids(&self) -> Vec<NodeId> {
        vec![self.n1, self.n2]
    }

    fn ndofs(&self) -> usize {
        6 // 2 nodes * 3 DOFs each
    }

    fn stiffness(&self, ctx: &ElementContext) -> DMatrix<f64> {
        let (l, [lx, ly, lz]) = self.length_and_dir(ctx.nodes);
        if l == 0.0 {
            return DMatrix::zeros(6, 6);
        }

        let e = ctx.material.young_modulus;
        let a = ctx.section.area;
        let k = a * e / l;

        // 3D truss stiffness in global coordinates: k * (n n^T) embedded in 6x6.
        let nn = [
            [lx * lx, lx * ly, lx * lz],
            [ly * lx, ly * ly, ly * lz],
            [lz * lx, lz * ly, lz * lz],
        ];

        let mut ke = DMatrix::zeros(6, 6);
        for i in 0..3 {
            for j in 0..3 {
                let v = k * nn[i][j];
                ke[(i, j)] += v;
                ke[(i + 3, j + 3)] += v;
                ke[(i, j + 3)] -= v;
                ke[(i + 3, j)] -= v;
            }
        }
        ke
    }

    fn mass(&self, ctx: &ElementContext) -> Option<DMatrix<f64>> {
        let (length, _) = self.length_and_dir(ctx.nodes);
        let rho = ctx.material.density;
        let mass = rho * ctx.section.area * length;

        // Lumped mass matrix - diagonal, half mass at each node
        let mut me = DMatrix::zeros(6, 6);
        let m_node = mass / 2.0;

        for dof in 0..3 {
            me[(dof, dof)] += m_node;
            me[(dof + 3, dof + 3)] += m_node;
        }

        Some(me)
    }

    fn internal_force(&self, ctx: &ElementContext) -> Option<DVector<f64>> {
        let displacements = ctx.displacements?;
        let (l, dir) = self.length_and_dir(ctx.nodes);
        if l == 0.0 {
            return Some(DVector::zeros(6));
        }

        let dof_indices = self.dof_indices();
        let u1 = [
            displacements.get(dof_indices[0]).copied().unwrap_or(0.0),
            displacements.get(dof_indices[1]).copied().unwrap_or(0.0),
            displacements.get(dof_indices[2]).copied().unwrap_or(0.0),
        ];
        let u2 = [
            displacements.get(dof_indices[3]).copied().unwrap_or(0.0),
            displacements.get(dof_indices[4]).copied().unwrap_or(0.0),
            displacements.get(dof_indices[5]).copied().unwrap_or(0.0),
        ];

        let du = [u2[0] - u1[0], u2[1] - u1[1], u2[2] - u1[2]];
        let delta_l = du[0] * dir[0] + du[1] * dir[1] + du[2] * dir[2];
        let strain = delta_l / l;
        let stress = ctx.material.young_modulus * strain;
        let force = stress * ctx.section.area;

        // Internal force vector (opposite direction at each node)
        let mut f = DVector::zeros(6);
        for i in 0..3 {
            f[i] = -force * dir[i];
            f[i + 3] = force * dir[i];
        }

        Some(f)
    }
}
