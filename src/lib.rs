//! Finite Element Analysis (FEA) methods and visualization utilities.
//!
//! Current focus:
//! - Linear static analysis for 2-node truss/bar elements
//! - Modal analysis for natural frequencies and mode shapes
//! - 2D beam elements with rotational DOFs
//! - Sparse matrix support for large-scale problems
//! - Simple VTK export for visualization (ParaView)
//! - Iterative solvers with acceleration techniques
//! - Result post-processing and CSV export
//!
//! The API is intentionally small and composable. It will expand over time.

pub mod beam;
pub mod core;
pub mod elements;
pub mod modal;
pub mod postprocessing;
pub mod solver;
pub mod solvers;
pub mod sparse;
pub mod viz;

/// Common imports for convenience.
pub mod prelude {
    pub use crate::beam::{Beam2D, BeamModel};
    pub use crate::core::{BoundaryCondition, Dof, Load, Model, NodeId};
    pub use crate::elements::{Element, Truss2};
    pub use crate::modal::{MassFormulation, ModalConfig, ModalResult, ModalSolver};
    pub use crate::postprocessing::{
        compute_reactions, extract_element_results, extract_nodal_displacements, CsvWriter,
        ElementResult, NodalDisplacement, ReactionForce, ResultStatistics,
    };
    pub use crate::solver::{LinearStaticResult, LinearStaticSolver};
    pub use crate::solvers::{ConjugateGradient, ConjugateGradientConfig, GaussSeidel, GMRES, PCG, Preconditioner};
    pub use crate::sparse::{CsrMatrix, SparseConjugateGradient};
    pub use crate::viz::{JsonWriter, VtkLegacyWriter, VtkMesh, VizConfig};
}
