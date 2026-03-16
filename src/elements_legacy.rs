//! Legacy element module for backward compatibility.
//!
//! This module re-exports the legacy element types.

pub use crate::elements::{ElementLegacy, Truss2Legacy};

// Also provide the original Truss2 with embedded properties
use crate::core::{Dof, Model, NodeId};
use nalgebra::DMatrix;

/// Legacy Element trait for backward compatibility.
pub trait ElementOld {
    fn node_ids(&self) -> Vec<NodeId>;
    fn stiffness(&self, model: &Model<Self>) -> DMatrix<f64>
    where
        Self: Sized;
}

/// A 2-node truss/bar element in 3D with embedded material properties.
#[derive(Debug, Clone, Copy)]
pub struct Truss2 {
    pub n1: NodeId,
    pub n2: NodeId,
    pub e: f64,
    pub a: f64,
}

impl Truss2 {
    pub fn new(n1: NodeId, n2: NodeId, e: f64, a: f64) -> Self {
        Self { n1, n2, e, a }
    }

    pub fn length_and_dir(&self, model: &Model<Truss2>) -> (f64, [f64; 3]) {
        let p1 = model.nodes[self.n1].as_array();
        let p2 = model.nodes[self.n2].as_array();
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

    pub fn axial_stress(&self, model: &Model<Truss2>, u: &[f64]) -> f64 {
        let (l, dir) = self.length_and_dir(model);
        if l == 0.0 {
            return 0.0;
        }

        let dof = |nid: NodeId, d: Dof| model.dof_index(nid, d);
        let u1 = [
            dof(self.n1, Dof::Ux).and_then(|i| u.get(i)).copied().unwrap_or(0.0),
            dof(self.n1, Dof::Uy).and_then(|i| u.get(i)).copied().unwrap_or(0.0),
            dof(self.n1, Dof::Uz).and_then(|i| u.get(i)).copied().unwrap_or(0.0),
        ];
        let u2 = [
            dof(self.n2, Dof::Ux).and_then(|i| u.get(i)).copied().unwrap_or(0.0),
            dof(self.n2, Dof::Uy).and_then(|i| u.get(i)).copied().unwrap_or(0.0),
            dof(self.n2, Dof::Uz).and_then(|i| u.get(i)).copied().unwrap_or(0.0),
        ];

        let du = [u2[0] - u1[0], u2[1] - u1[1], u2[2] - u1[2]];
        let delta_l = du[0] * dir[0] + du[1] * dir[1] + du[2] * dir[2];
        self.e * (delta_l / l)
    }
}

impl ElementOld for Truss2 {
    fn node_ids(&self) -> Vec<NodeId> {
        vec![self.n1, self.n2]
    }

    fn stiffness(&self, model: &Model<Self>) -> DMatrix<f64> {
        let (l, [lx, ly, lz]) = self.length_and_dir(model);
        if l == 0.0 {
            return DMatrix::zeros(6, 6);
        }

        let k = self.a * self.e / l;
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
}
