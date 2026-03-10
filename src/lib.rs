//! Finite Element Analysis (FEA) methods and visualization utilities.
//!
//! Current focus:
//! - Linear static analysis for 2-node truss/bar elements
//! - Modal analysis for natural frequencies and mode shapes
//! - 2D beam elements with rotational DOFs
//! - Simple VTK export for visualization (ParaView)
//! - Iterative solvers with acceleration techniques
//!
//! The API is intentionally small and composable. It will expand over time.

pub mod beam;
pub mod core;
pub mod elements;
pub mod modal;
pub mod solver;
pub mod solvers;
pub mod viz;

/// Common imports for convenience.
pub mod prelude {
    pub use crate::beam::{Beam2D, BeamModel};
    pub use crate::core::{BoundaryCondition, Dof, Load, Model, NodeId};
    pub use crate::elements::{Element, Truss2};
    pub use crate::modal::{ModalConfig, ModalSolver, ModalResult, MassFormulation};
    pub use crate::solver::{LinearStaticResult, LinearStaticSolver};
    pub use crate::solvers::{ConjugateGradient, ConjugateGradientConfig, GaussSeidel, Preconditioner};
    pub use crate::viz::{JsonWriter, VtkLegacyWriter, VtkMesh, VizConfig};
}
