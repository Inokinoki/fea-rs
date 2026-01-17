use crate::core::{Dof, Model, NodeId};
use nalgebra::DMatrix;

/// An FEA element that can contribute stiffness to the global system.
pub trait Element {
    /// Returns the element's node connectivity.
    fn node_ids(&self) -> Vec<NodeId>;

    /// Returns the element stiffness matrix in the global DOF ordering for the element.
    ///
    /// For truss-like elements, the local ordering is typically:
    /// [n1.Ux, n1.Uy, n1.Uz, n2.Ux, n2.Uy, n2.Uz]
    fn stiffness(&self, model: &Model<Self>) -> DMatrix<f64>
    where
        Self: Sized;
}

/// A 2-node truss/bar element in 3D (use z=0 for 2D).
///
/// - `e`: Young's modulus
/// - `a`: Cross-sectional area
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

    /// Element axial stress (positive in tension) based on nodal displacements.
    pub fn axial_stress(&self, model: &Model<Truss2>, u: &[f64]) -> f64 {
        let (l, dir) = self.length_and_dir(model);
        if l == 0.0 {
            return 0.0;
        }
        let dof = |nid: NodeId, d: Dof| model.dof_index(nid, d).unwrap();
        let u1 = [
            u[dof(self.n1, Dof::Ux)],
            u[dof(self.n1, Dof::Uy)],
            u[dof(self.n1, Dof::Uz)],
        ];
        let u2 = [
            u[dof(self.n2, Dof::Ux)],
            u[dof(self.n2, Dof::Uy)],
            u[dof(self.n2, Dof::Uz)],
        ];
        let du = [u2[0] - u1[0], u2[1] - u1[1], u2[2] - u1[2]];
        let delta_l = du[0] * dir[0] + du[1] * dir[1] + du[2] * dir[2];
        self.e * (delta_l / l)
    }
}

impl Element for Truss2 {
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
        // Where n = [lx, ly, lz].
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

