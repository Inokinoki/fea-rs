//! Element traits and implementations.
//!
//! This module provides:
//! - The Element trait for all finite elements
//! - Element context for passing state to element methods
//! - Truss element implementations

use crate::core::{Material, Node, NodeId, Section};
use nalgebra::{DMatrix, DVector};

/// Context passed to element methods for computing matrices and vectors.
#[derive(Debug, Clone)]
pub struct ElementContext<'a> {
    /// Node coordinates.
    pub nodes: &'a [Node],
    /// Nodal displacements (optional, for nonlinear analysis).
    pub displacements: Option<&'a [f64]>,
    /// Nodal velocities (optional, for dynamic analysis).
    pub velocities: Option<&'a [f64]>,
    /// Nodal accelerations (optional, for dynamic analysis).
    pub accelerations: Option<&'a [f64]>,
    /// Material properties.
    pub material: &'a Material,
    /// Section properties.
    pub section: &'a Section,
}

impl<'a> ElementContext<'a> {
    /// Creates a new element context.
    pub fn new(
        nodes: &'a [Node],
        material: &'a Material,
        section: &'a Section,
    ) -> Self {
        Self {
            nodes,
            displacements: None,
            velocities: None,
            accelerations: None,
            material,
            section,
        }
    }

    /// Sets the displacements.
    pub fn with_displacements(mut self, displacements: &'a [f64]) -> Self {
        self.displacements = Some(displacements);
        self
    }

    /// Sets the velocities.
    pub fn with_velocities(mut self, velocities: &'a [f64]) -> Self {
        self.velocities = Some(velocities);
        self
    }

    /// Sets the accelerations.
    pub fn with_accelerations(mut self, accelerations: &'a [f64]) -> Self {
        self.accelerations = Some(accelerations);
        self
    }
}

/// An FEA element that can contribute stiffness to the global system.
pub trait Element: Clone + std::fmt::Debug {
    /// Returns the element's node connectivity.
    fn node_ids(&self) -> Vec<NodeId>;

    /// Returns the number of DOFs for this element.
    fn ndofs(&self) -> usize;

    /// Returns the element stiffness matrix in the global DOF ordering.
    fn stiffness(&self, ctx: &ElementContext) -> DMatrix<f64>;

    /// Returns the element mass matrix (for dynamic analysis).
    ///
    /// Default implementation returns None (no mass).
    fn mass(&self, _ctx: &ElementContext) -> Option<DMatrix<f64>> {
        None
    }

    /// Returns the internal force vector (for nonlinear analysis).
    ///
    /// Default implementation returns None (linear elements only).
    fn internal_force(&self, _ctx: &ElementContext) -> Option<DVector<f64>> {
        None
    }

    /// Returns the element damping matrix (for dynamic analysis).
    ///
    /// Default implementation returns None (no damping).
    fn damping(&self, _ctx: &ElementContext) -> Option<DMatrix<f64>> {
        None
    }
}

pub mod truss;
pub mod legacy;
pub mod plate;
pub mod beam;

pub use truss::Truss2;
pub use legacy::{Truss2Legacy, ElementLegacy};
pub use plate::{Plate4, Plate8};
pub use beam::{Beam2DElement, Beam3DElement};
