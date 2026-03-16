//! Boundary conditions and loads.

use crate::core::{Dof, NodeId};

/// A nodal boundary condition (Dirichlet/essential condition).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryCondition {
    pub node: NodeId,
    pub dof: Dof,
    pub value: f64,
}

impl BoundaryCondition {
    /// Creates a new boundary condition.
    pub fn new(node: NodeId, dof: Dof, value: f64) -> Self {
        Self { node, dof, value }
    }

    /// Creates a fixed (zero displacement) boundary condition.
    pub fn fixed(node: NodeId, dof: Dof) -> Self {
        Self { node, dof, value: 0.0 }
    }
}

/// A nodal load (Neumann/natural condition).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Load {
    pub node: NodeId,
    pub dof: Dof,
    pub value: f64,
}

impl Load {
    /// Creates a new nodal load.
    pub fn new(node: NodeId, dof: Dof, value: f64) -> Self {
        Self { node, dof, value }
    }
}
