//! Legacy element trait and wrapper for backward compatibility.

use crate::core::{Dof, Model, NodeId};
use nalgebra::DMatrix;

/// Legacy element trait (for backward compatibility).
pub trait ElementLegacy {
    /// Returns the element's node connectivity.
    fn node_ids(&self) -> Vec<NodeId>;

    /// Returns the element stiffness matrix in the global DOF ordering.
    fn stiffness(&self, model: &Model<Self>) -> DMatrix<f64>
    where
        Self: Sized;
}

/// A 2-node truss/bar element in 3D with embedded material properties.
///
/// This is the legacy version that stores E and A directly on the element.
/// For new code, use `Truss2` with `Material` and `Section` in the context.
#[derive(Debug, Clone, Copy)]
pub struct Truss2Legacy {
    pub n1: NodeId,
    pub n2: NodeId,
    pub e: f64,
    pub a: f64,
}

impl Truss2Legacy {
    pub fn new(n1: NodeId, n2: NodeId, e: f64, a: f64) -> Self {
        Self { n1, n2, e, a }
    }

    pub fn length_and_dir(&self, model: &Model<Self>) -> (f64, [f64; 3]) {
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

    /// Element axial stress (positive in tension) based on nodal displacements.
    pub fn axial_stress(&self, model: &Model<Self>, u: &[f64]) -> f64 {
        let (l, dir) = self.length_and_dir(model);
        if l == 0.0 {
            return 0.0;
        }

        let dof = |nid: NodeId, d: Dof| model.dof_index(nid, d);
        let Some(i11) = dof(self.n1, Dof::Ux) else { return 0.0; };
        let Some(i12) = dof(self.n1, Dof::Uy) else { return 0.0; };
        let Some(i13) = dof(self.n1, Dof::Uz) else { return 0.0; };
        let Some(i21) = dof(self.n2, Dof::Ux) else { return 0.0; };
        let Some(i22) = dof(self.n2, Dof::Uy) else { return 0.0; };
        let Some(i23) = dof(self.n2, Dof::Uz) else { return 0.0; };

        let u1 = [
            *u.get(i11).unwrap_or(&0.0),
            *u.get(i12).unwrap_or(&0.0),
            *u.get(i13).unwrap_or(&0.0),
        ];
        let u2 = [
            *u.get(i21).unwrap_or(&0.0),
            *u.get(i22).unwrap_or(&0.0),
            *u.get(i23).unwrap_or(&0.0),
        ];

        let du = [u2[0] - u1[0], u2[1] - u1[1], u2[2] - u1[2]];
        let delta_l = du[0] * dir[0] + du[1] * dir[1] + du[2] * dir[2];
        self.e * (delta_l / l)
    }
}

impl ElementLegacy for Truss2Legacy {
    fn node_ids(&self) -> Vec<NodeId> {
        vec![self.n1, self.n2]
    }

    fn stiffness(&self, model: &Model<Self>) -> DMatrix<f64> {
        let (l, [lx, ly, lz]) = self.length_and_dir(model);
        if l == 0.0 {
            return DMatrix::zeros(6, 6);
        }

        let k = self.a * self.e / l;

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
}
