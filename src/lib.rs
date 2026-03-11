//! Finite Element Analysis (FEA) methods and visualization utilities.
//!
//! # Overview
//!
//! This crate provides a modular, extensible framework for finite element analysis.
//! It supports multiple analysis types, solver methods, and element formulations.
//!
//! # Features
//!
//! - **Linear static analysis** - Direct and iterative solvers
//! - **Modal analysis** - Natural frequencies and mode shapes
//! - **Buckling analysis** - Linear buckling load factors
//! - **Dynamic analysis** - Transient response using Newmark-beta method
//! - **Nonlinear analysis** - Newton-Raphson and arc-length methods
//!
//! # Module Structure
//!
//! - `core` - Basic data structures (Model, Node, Material, Section)
//! - `elements` - Element traits and implementations
//! - `algorithms` - Solvers, analysis types, and nonlinear methods
//! - `utils` - Sparse matrices, post-processing, and visualization
//!
//! # Example: Linear Static Analysis
//!
//! ```rust,no_run
//! use fea::prelude::*;
//!
//! fn main() -> anyhow::Result<()> {
//!     // Create model
//!     let mut model = Model::<Truss2>::new();
//!
//!     // Add nodes
//!     let n0 = model.add_node(Node::new_2d(0.0, 0.0));
//!     let n1 = model.add_node(Node::new_2d(1.0, 0.0));
//!
//!     // Add material and section
//!     model.add_material(STEEL_A36);
//!     model.add_section(Section::circular("round", 0.01));
//!
//!     // Add element
//!     model.add_element(Truss2::new(n0, n1));
//!
//!     // Add boundary conditions
//!     model.add_bc(BoundaryCondition::fixed(n0, Dof::Ux));
//!     model.add_bc(BoundaryCondition::fixed(n0, Dof::Uy));
//!     model.add_bc(BoundaryCondition::fixed(n0, Dof::Uz));
//!     model.add_bc(BoundaryCondition::fixed(n1, Dof::Uy));
//!     model.add_bc(BoundaryCondition::fixed(n1, Dof::Uz));
//!
//!     // Add load
//!     model.add_load(Load::new(n1, Dof::Ux, 1000.0));
//!
//!     // Run analysis
//!     let analysis = LinearStaticAnalysis::new();
//!     let config = StaticConfig::default();
//!     let result = analysis.run(&mut model, &config)?;
//!
//!     println!("Displacement at node 1: {:?}", result.displacements[3]);
//!     Ok(())
//! }
//! ```
//!
//! # Feature Flags
//!
//! - `static` - Linear static analysis
//! - `modal` - Modal analysis
//! - `buckling` - Buckling analysis
//! - `dynamic` - Transient dynamic analysis
//! - `solver-direct` - Direct solvers (LU, Cholesky)
//! - `solver-iterative` - Iterative solvers (CG, PCG, GMRES)
//! - `element-truss` - Truss elements
//! - `element-beam` - Beam elements
//! - `nonlinear-geometric` - Geometric nonlinearity
//!
//! Preset feature combinations:
//! - `minimal` - Static analysis with direct solver and truss elements
//! - `standard` - Static + modal with direct/iterative solvers
//! - `full` - All features enabled

// Core modules
pub mod core;
pub mod elements;
pub mod algorithms;
pub mod utils;

// Legacy modules for backward compatibility
pub mod beam;
pub mod elements_legacy;
pub mod solver;
pub mod solvers;
pub mod modal;
pub mod sparse;
pub mod materials;
pub mod postprocessing;
pub mod parametric;
pub mod viz;

/// Common imports for convenience.
pub mod prelude {
    // Core types
    pub use crate::core::{
        BoundaryCondition, Dof, Load, Model, Node, NodeId,
        Material, Section,
        STEEL_A36, STAINLESS_STEEL_304, ALUMINUM_6061_T6, ALUMINUM_7075_T6, TITANIUM_TI6AL4V,
    };

    // Elements
    pub use crate::elements::{Element, ElementContext, Truss2};
    pub use crate::elements_legacy::{ElementLegacy, Truss2Legacy};

    // Solvers
    pub use crate::algorithms::solvers::{
        Solver, SolverResult,
        DirectSolver, DirectConfig,
        CGSolver, PCGSolver, GMRESSolver, GaussSeidelSolver,
        IterativeConfig, Preconditioner,
    };

    // Analysis types
    pub use crate::algorithms::analysis::{
        Analysis, AnalysisResult,
        LinearStaticAnalysis, StaticConfig, StaticResult,
        ModalAnalysis, ModalConfig, ModalResult,
        BucklingAnalysis, BucklingConfig, BucklingResult,
        TransientDynamicAnalysis, DynamicConfig, DynamicResult,
    };

    // Nonlinear methods
    pub use crate::algorithms::nonlinear::{
        NewtonRaphsonConfig, NewtonRaphsonResult,
        newton_raphson,
        ArcLengthConfig, ArcLengthResult,
        arc_length,
    };

    // Utilities
    pub use crate::utils::{
        CsrMatrix, SparseConjugateGradient,
        NodalDisplacement, ElementResult, ReactionForce, ResultStatistics,
        VizConfig, VtkMesh, JsonOutput,
    };

    // Legacy re-exports
    pub use crate::beam::{Beam2D, BeamModel};
    pub use crate::solver::{LinearStaticSolver, LinearStaticResult};
    pub use crate::solvers::{
        ConjugateGradient, ConjugateGradientConfig,
        PCG, GMRES, GaussSeidel, Preconditioner as LegacyPreconditioner,
        IterativeResult,
    };
    pub use crate::modal::{ModalSolver, ModalConfig as LegacyModalConfig, ModalResult as LegacyModalResult, MassFormulation};
    pub use crate::postprocessing::{
        compute_reactions, extract_element_results, extract_nodal_displacements,
        CsvWriter, ReactionForce as LegacyReactionForce,
        ElementResult as LegacyElementResult, NodalDisplacement as LegacyNodalDisplacement,
        ResultStatistics as LegacyResultStatistics, ConvergenceHistory,
    };
    pub use crate::viz::{VtkLegacyWriter, JsonWriter, VtkMesh as LegacyVtkMesh, VizConfig as LegacyVizConfig};
}

// Re-export main types at crate level
pub use core::{Model, Node, Dof, BoundaryCondition, Load, Material, Section};
pub use elements::{Element, ElementContext, Truss2};
pub use algorithms::solvers::{Solver, SolverResult};
pub use algorithms::analysis::{Analysis, AnalysisResult};
