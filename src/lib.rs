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
pub mod gpu;
pub mod preprocessing;

// Benchmark modules
pub mod benchmarks;

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
    pub use crate::elements::{Element, ElementContext, Truss2, Plate4, Plate8, Beam2DElement, Beam3DElement};
    pub use crate::elements_legacy::{ElementLegacy, Truss2Legacy, Truss2 as Truss2LegacyCompat};

    // Solvers
    pub use crate::algorithms::solvers::{
        Solver, SolverResult,
        DirectSolver, DirectConfig,
        CGSolver, PCGSolver, CGNRSolver, TFQMRSolver, GMRESSolver, GaussSeidelSolver, BiCGSTABSolver,
        IterativeConfig, Preconditioner,
        ConvergenceHistory,
        lanczos::{LanczosSolver, LanczosConfig, LanczosResult},
        multigrid::{MultigridPreconditioner, MultigridCG, AMGSetup},
        ic::{IncompleteCholesky, ModifiedIncompleteCholesky, BlockIncompleteCholesky},
        fmg::{FullMultigridSolver, FullMultigridConfig},
        gcr::GCRSolver,
        qmr::QMRSolver,
        diis::{DIISAccelerator, DIISConfig, CGSSolver, SORSolver},
        block_precond::{BlockJacobi, AdditiveSchwarz, SPAI, AINV},
        polynomial::{ChebyshevPreconditioner, NewtonSchulz, HotellingBodewig, EisenstatPreconditioner},
        fgmres::{FGMRESSolver, DGMRESSolver},
        recycling::{RecyclingPreconditioner, GCRODRSolver},
        krylov::{DeflatedCG, SpectralPreconditioner},
        extrapolation::{MPE, RRE, MRE, SteffensenExtrapolation},
        extrapolation2::{TEA, VectorEpsilon, RRE2, MPEVariant, HybridAccelerator},
        nonlinear_accel::{EpsilonAlgorithm, LevinU, ThetaAlgorithm, AitkenDeltaSquared, CombinedAccelerator},
        advanced_accel::{AndersonMixing, AndersonTypeII, NGMRES, PipelinedAnderson},
        mp_accel::{MinimalPolynomial, TopologicalTheta, STEA, GVE},
        advanced_extrap::{MMPE, RREFixed, NewtonExtrapolation, PadeAcceleration},
        poly_accel::{ChebyshevAcceleration, MinimalResidualAcceleration, SteepestDescentAcceleration, NonlinearCGAcceleration, BarzilaiBorweinAcceleration},
        poly_accel2::{ChebyshevSemiIterative, MinimalResidualPoly, OptimalSteepestDescent},
        additional_accel::{SQUARED, BarzilaiBorwein2, CyclicBB, AdaptiveCubicReg},
        acceleration::{AitkenAcceleration, MinimalResidualSmoothing, ConvergenceAccelerator, steffensen},
        advanced_methods::{
            RieszAcceleration, MultipointExtrapolation, VectorEpsilonAlgorithm,
            TopologicalEpsilon, GeneralizedVectorExtrapolation, AdaptiveSteffensen,
        },
        block_solvers::{
            BlockJacobiSolver, BlockGaussSeidelSolver, BlockCGSolver,
            SchurComplementSolver, UzawaSolver, BlockSize,
        },
        preconditioners_advanced::{
            FSAIPreconditioner, ElasticityPreconditioner,
            AdditiveSchwarzMultilevel, BlockRecursivePreconditioner,
        },
        spectral_accel::{
            SpectralDeflation, RationalChebyshevFilter,
            MatrixPowerPreconditioner,
        },
        krylov_recycling::{
            RecyclingBiCGSTAB,
        },
        domain_decomposition::{
            BDDPreconditioner, FETIPreconditioner, NeumannNeumann,
        },
        tensor_multilevel::{
            TensorProductPreconditioner, HierarchicalBasis,
            WaveletPreconditioner, MultilevelAccelerator,
        },
        nonlinear_solvers_enhanced::{
            LineSearch, LineSearchMethod, LineSearchResult,
            TrustRegion, BFGS, HomotopySolver,
        },
        adaptive_solvers::{
            PIDController, ConvergenceMonitor, ConvergenceRecommendation,
            AdaptivePreconditionerSelector, PreconditionerType,
            SolverOrchestrator, SolverStrategy, AdaptiveCGSolver, AdaptiveSolverResult,
        },
        parallel_solvers::{
            DistributedMatrix, DistributedVector, ParallelSolverResult,
            ParallelCG, ParallelGMRES, ParallelAdditiveSchwarz, ParallelMultigrid,
        },
        polynomial_acceleration::{
            ChebyshevAcceleration as ChebyshevPolyAccel,
            PadeAcceleration as PadePolyAccel,
            MinimalPolynomialAcceleration, SteffensenAcceleration,
            AndersonAcceleration as AndersonPolyAccel,
        },
        randomized_la::{
            RandomizedSVD, RandomizedEigenSolver, RandomizedSubspaceIteration,
            SketchingPreconditioner, CountSketch,
        },
        unified_framework::{
            AccelerationStrategy, UnifiedSolverConfig,
            UnifiedSolverResult, UnifiedAccelerationSolver, SolverComparator,
        },
        block_iterative::{
            BlockJacobiIterative, BlockGaussSeidelIterative, BlockCGIterative, BlockILU0,
        },
        advanced_krylov::{
            FlexibleGMRES, DeflatedCGAdvanced, AugmentedKrylov, HarmonicRitzExtractor,
        },
        mixed_precision::{
            MixedPrecisionSolver, MixedPrecisionConfig, MixedPrecisionResult,
            simulate_precision, simulate_precision_vector, simulate_precision_matrix,
            fp16_utils,
        },
        nonlinear_acceleration::{
            AcceleratedNonlinearSolver, NonlinearSolverResult,
            vector_extrapolation,
        },
        performance_utils::{
            PerformanceMetrics, PerformanceComparison, BenchmarkTimer,
            PerformanceProfiler, MemoryTracker, compare_solvers, benchmark_matvec,
        },
        accelerated_pcg::{
            AcceleratedPCG, AcceleratedPCGConfig, AcceleratedPCGResult,
            solve_accelerated,
        },
        advanced_eigensolvers::{
            KrylovSchur, KrylovSchurConfig, KrylovSchurResult,
            IRAM, IRAMConfig, IRAMResult,
            ThickRestartedLanczos, TRLConfig, TRLResult,
        },
    };

    // Analysis types
    pub use crate::algorithms::analysis::{
        Analysis, AnalysisResult,
        LinearStaticAnalysis, StaticConfig, StaticResult,
        ModalAnalysis, ModalConfig, ModalResult,
        BucklingAnalysis, BucklingConfig, BucklingResult,
        TransientDynamicAnalysis, DynamicConfig, DynamicResult,
        WilsonThetaAnalysis, WilsonThetaConfig,
        HarmonicAnalysis, HarmonicConfig, HarmonicResult,
        ModalSuperpositionAnalysis, ModalSuperpositionConfig, ModalSuperpositionResult,
    };

    // Thermal analysis
    pub use crate::algorithms::thermal::{
        ThermalProperties, TemperatureField, ThermalAnalysis, ThermalStressResult,
    };

    // Nonlinear methods
    pub use crate::algorithms::nonlinear::{
        NonlinearMethod, NonlinearConfig, ConvergenceCriteria,
        NewtonRaphson, ArcLengthSolver, ArcLengthConfig, RiksConfig,
        NonlinearStaticAnalysis, NonlinearResult,
        material_nonlinearity, geometric_nonlinearity,
    };

    // Utilities
    pub use crate::utils::{
        CsrMatrix, SparseConjugateGradient,
        NodalDisplacement, ElementResult, ReactionForce, ResultStatistics,
        VizConfig, VtkMesh, JsonOutput,
    };

    // GPU acceleration
    pub use crate::gpu::{
        GPUDevice, DeviceType, GPUMemory, GPUStream,
        GPUCSRMatrix, SparseMatrixVectorMul, VectorOps,
        GPUContext, GPUCGSolver, GPUSolverResult,
        gpu_available, list_gpu_devices, create_gpu_context,
        assembly, preconditioners,
        gpu_kernel_library::{
            GPUContext as GPUKernelContext, GPUStream as GPUKernelStream,
            vector_kernels, sparse_kernels, cg_kernel,
        },
    };

    // Pre-processing
    pub use crate::preprocessing::{
        mesh_generation, stl_io, vtk_io, bc_helpers, material_helpers, advanced,
    };

    // Post-processing
    pub use crate::postprocessing::{
        FeaResults, StressResult, StrainResult,
        vtk_export, csv_export, report_generation, visualization,
    };

    // Adaptive Mesh Refinement
    pub use crate::algorithms::amr::{
        MeshQuality, ErrorEstimator, AdaptiveMeshRefiner,
        RefinementStrategy, SolutionInterpolator,
    };

    // Contact Analysis
    pub use crate::algorithms::contact_enhanced::{
        ContactPair, ContactDetection, ContactMethod, ContactParameters,
        NodeToSurfaceContact, PenaltyContact, AugmentedLagrangianContact,
        MortarContact, ContactConstraintHandler,
    };

    // Dynamic Analysis Acceleration
    pub use crate::algorithms::dynamics_accel::{
        ModalAcceleration, ComponentModeSynthesis,
        ProperOrthogonalDecomposition, KrylovReduction,
    };

    // Benchmarks
    pub use crate::benchmarks::{
        NafemsLe1, NafemsLe1Result,
        NafemsTorsion, NafemsTorsionResult,
        NafemsBeam, NafemsBeamResult,
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
    // Legacy postprocessing exports - temporarily disabled during refactoring
    // pub use crate::postprocessing::{
    //     compute_reactions, extract_element_results, extract_nodal_displacements,
    //     CsvWriter, ReactionForce as LegacyReactionForce,
    //     ElementResult as LegacyElementResult, NodalDisplacement as LegacyNodalDisplacement,
    //     ResultStatistics as LegacyResultStatistics, ConvergenceHistory as LegacyConvergenceHistory,
    // };
    pub use crate::viz::{VtkLegacyWriter, JsonWriter, VtkMesh as LegacyVtkMesh, VizConfig as LegacyVizConfig};
}

// Re-export main types at crate level
pub use core::{Model, Node, Dof, BoundaryCondition, Load, Material, Section};
pub use elements::{Element, ElementContext, Truss2};
pub use algorithms::solvers::{Solver, SolverResult};
pub use algorithms::analysis::{Analysis, AnalysisResult};
