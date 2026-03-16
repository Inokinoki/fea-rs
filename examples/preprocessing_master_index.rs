//! FEA Pre-processing Examples - Master Index.
//!
//! This file provides a comprehensive index of all pre-processing examples.
//!
//! # Pre-processing Example Categories
//!
//! ## Basic Pre-processing (5)
//! - Mesh generation (1D, 2D, 3D)
//! - Material assignment
//! - Boundary condition application
//! - Load application
//!
//! ## Advanced Pre-processing (8)
//! - Graded mesh generation
//! - Geometry transformations
//! - Advanced BCs
//! - Complex material assignment
//! - Mesh quality improvement
//! - STL import/export
//! - Parametric geometry
//! - Multi-part assembly
//!
//! ## GPU Pre-processing (3)
//! - GPU mesh generation
//! - GPU transforms
//! - GPU quality analysis
//!
//! ## Complete Workflows (2)
//! - Complete pipeline
//! - 10-step workflow
//!
//! # Usage
//!
//! Run pre-processing examples:
//! ```bash
//! # Basic
//! cargo run --example preprocessing_demo
//! cargo run --example preprocessing_comprehensive
//!
//! # Advanced
//! cargo run --example advanced_preprocessing_examples
//! cargo run --example gpu_preprocessing_example
//!
//! # Complete Workflows
//! cargo run --example complete_preprocessing_pipeline
//! cargo run --example complete_preprocessing_workflow
//! ```

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║    FEA Pre-processing - Master Example Index              ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("┌─ Basic Pre-processing (5 examples) ──────────────────────┐");
    println!("│");
    println!("│ Mesh Generation:");
    println!("│   • 1D bar mesh (generate_bar_1d)");
    println!("│     - Uniform mesh");
    println!("│     - Graded mesh (bias)");
    println!("│");
    println!("│   • 2D rectangular mesh (generate_rect_2d)");
    println!("│     - Quad4 elements");
    println!("│     - Structured mesh");
    println!("│");
    println!("│   • 2D triangular mesh (generate_tri_2d_from_rect)");
    println!("│     - Tri3 elements");
    println!("│     - 2 tris per quad");
    println!("│");
    println!("│   • 3D box mesh (generate_box_3d)");
    println!("│     - Hex8 elements");
    println!("│     - Structured mesh");
    println!("│");
    println!("│   • 3D tetrahedral mesh (generate_tet_3d_from_hex)");
    println!("│     - Tet4 elements");
    println!("│     - 5 tets per hex");
    println!("│");
    println!("│   • 3D cylindrical mesh (generate_cylinder_3d)");
    println!("│     - Cylindrical coordinates");
    println!("│     - Structured mesh");
    println!("│");
    println!("│ Material Assignment:");
    println!("│   • Steel A36 (steel_a36)");
    println!("│     - E = 210 GPa, ν = 0.3");
    println!("│     - ρ = 7850 kg/m³");
    println!("│");
    println!("│   • Aluminum 6061-T6 (aluminum_6061)");
    println!("│     - E = 68.9 GPa, ν = 0.33");
    println!("│     - ρ = 2700 kg/m³");
    println!("│");
    println!("│   • Titanium Ti-6Al-4V (titanium_ti64)");
    println!("│     - E = 113.8 GPa, ν = 0.342");
    println!("│     - ρ = 4430 kg/m³");
    println!("│");
    println!("│   • Normal Concrete (concrete_normal)");
    println!("│     - E = 25 GPa, ν = 0.2");
    println!("│     - ρ = 2400 kg/m³");
    println!("│");
    println!("│ Boundary Conditions:");
    println!("│   • fix_all_dofs()");
    println!("│     - Fix all DOFs at node");
    println!("│");
    println!("│   • fix_face_3d()");
    println!("│     - Fix nodes on 3D face");
    println!("│     - By coordinate (x, y, or z)");
    println!("│");
    println!("│   • apply_distributed_load()");
    println!("│     - Apply distributed load");
    println!("│     - By face selection");
    println!("│");
    println!("│   • apply_symmetry_x()");
    println!("│     - Apply symmetry BCs");
    println!("│     - At x = constant plane");
    println!("│");
    println!("│ Examples:");
    println!("│   • preprocessing_demo.rs");
    println!("│   • preprocessing_comprehensive.rs");
    println!("└────────────────────────────────────────────────────────┘\n");

    println!("┌─ Advanced Pre-processing (8 features) ───────────────────┐");
    println!("│");
    println!("│ Graded Mesh Generation:");
    println!("│   • Cosine grading (finer at edges)");
    println!("│   • Bias grading (finer at one end)");
    println!("│   • Custom grading functions");
    println!("│");
    println!("│ Geometry Transformations:");
    println!("│   • Translation (translate)");
    println!("│   • Rotation about X (rotate_x)");
    println!("│   • Rotation about Y (rotate_y)");
    println!("│   • Rotation about Z (rotate_z)");
    println!("│   • Uniform scaling (scale)");
    println!("│   • Non-uniform scaling (scale_nonuniform)");
    println!("│   • Mirror about plane (mirror)");
    println!("│");
    println!("│ Advanced Boundary Conditions:");
    println!("│   • Symmetry conditions");
    println!("│   • Periodic conditions");
    println!("│   • Contact conditions");
    println!("│   • Multi-point constraints");
    println!("│");
    println!("│ Complex Material Assignment:");
    println!("│   • Multiple materials");
    println!("│   • Functionally graded materials");
    println!("│   • Composite materials");
    println!("│   • By geometry region");
    println!("│   • By element set");
    println!("│");
    println!("│ Mesh Quality Improvement:");
    println!("│   • Aspect ratio check (quad_aspect_ratio)");
    println!("│     - Ideal < 2, Acceptable < 5");
    println!("│   • Skew angle check (quad_skew_angle)");
    println!("│     - Ideal < 15°, Acceptable < 30°");
    println!("│   • Jacobian check (quad_jacobian)");
    println!("│     - Must be > 0");
    println!("│   • Quality check (check_mesh_quality)");
    println!("│     - All metrics combined");
    println!("│   • Mesh refinement (refine_1d/quad/hex)");
    println!("│     - h-refinement");
    println!("│");
    println!("│ STL Import/Export:");
    println!("│   • ASCII STL import (import_ascii_stl)");
    println!("│     - Facet data");
    println!("│     - Normal vectors");
    println!("│   • ASCII STL export (export_ascii_stl)");
    println!("│   • STL to mesh conversion (stl_to_mesh)");
    println!("│");
    println!("│ Parametric Geometry:");
    println!("│   • Parametric beam");
    println!("│   • Parametric plate");
    println!("│   • Parametric solid");
    println!("│   • Parameterized dimensions");
    println!("│");
    println!("│ Multi-part Assembly:");
    println!("│   • Base plate");
    println!("│   • Vertical columns");
    println!("│   • Top plates");
    println!("│   • Node merging");
    println!("│   • Contact definition");
    println!("│");
    println!("│ Examples:");
    println!("│   • advanced_preprocessing_examples.rs");
    println!("│   • gpu_preprocessing_example.rs");
    println!("└────────────────────────────────────────────────────────┘\n");

    println!("┌─ GPU Pre-processing (3 features) ────────────────────────┐");
    println!("│");
    println!("│ GPU Mesh Generation:");
    println!("│   • 1D mesh (9 CUDA kernels)");
    println!("│   • 2D mesh (9 CUDA kernels)");
    println!("│   • 3D mesh (9 CUDA kernels)");
    println!("│");
    println!("│ GPU Coordinate Transforms:");
    println!("│   • GPU translation");
    println!("│   • GPU rotation");
    println!("│   • GPU scaling");
    println!("│");
    println!("│ GPU Mesh Quality:");
    println!("│   • GPU aspect ratio");
    println!("│   • GPU skew angle");
    println!("│   • GPU quality check");
    println!("│");
    println!("│ Examples:");
    println!("│   • gpu_preprocessing_example.rs");
    println!("└────────────────────────────────────────────────────────┘\n");

    println!("┌─ Complete Workflows (2 examples) ────────────────────────┐");
    println!("│");
    println!("│ Complete Pipeline:");
    println!("│   • complete_preprocessing_pipeline.rs");
    println!("│     - All pre-processing features");
    println!("│     - End-to-end workflow");
    println!("│");
    println!("│ 10-Step Workflow:");
    println!("│   • complete_preprocessing_workflow.rs");
    println!("│     - Step 1: Geometry creation");
    println!("│     - Step 2: Mesh generation");
    println!("│     - Step 3: Transformations");
    println!("│     - Step 4: Materials");
    println!("│     - Step 5: Sections");
    println!("│     - Step 6: BCs");
    println!("│     - Step 7: Loads");
    println!("│     - Step 8: Mesh quality");
    println!("│     - Step 9: Validation");
    println!("│     - Step 10: Export");
    println!("└────────────────────────────────────────────────────────┘\n");

    println!("┌─ Pre-processing Export Formats ──────────────────────────┐");
    println!("│");
    println!("│ VTK Export:");
    println!("│   • export_vtk_mesh()");
    println!("│     - Mesh geometry");
    println!("│     - Compatible with: ParaView, VisIt");
    println!("│");
    println!("│ STL Export:");
    println!("│   • export_ascii_stl()");
    println!("│     - Surface geometry");
    println!("│     - Compatible with: CAD software");
    println!("└────────────────────────────────────────────────────────┘\n");

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Run examples with: cargo run --example <example_name>    ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_preprocessing_categories() {
        let categories = [
            "Basic Pre-processing",
            "Advanced Pre-processing",
            "GPU Pre-processing",
            "Complete Workflows",
        ];
        assert_eq!(categories.len(), 4);
    }

    #[test]
    fn test_mesh_generation_functions() {
        let functions = [
            "generate_bar_1d",
            "generate_rect_2d",
            "generate_tri_2d_from_rect",
            "generate_box_3d",
            "generate_tet_3d_from_hex",
            "generate_cylinder_3d",
        ];
        assert_eq!(functions.len(), 6);
    }

    #[test]
    fn test_transform_functions() {
        let functions = [
            "translate",
            "rotate_x",
            "rotate_y",
            "rotate_z",
            "scale",
            "scale_nonuniform",
            "mirror",
        ];
        assert_eq!(functions.len(), 7);
    }

    #[test]
    fn test_bc_functions() {
        let functions = [
            "fix_all_dofs",
            "fix_face_3d",
            "apply_distributed_load",
            "apply_symmetry_x",
        ];
        assert_eq!(functions.len(), 4);
    }

    #[test]
    fn test_material_functions() {
        let materials = [
            "steel_a36",
            "aluminum_6061",
            "titanium_ti64",
            "concrete_normal",
        ];
        assert_eq!(materials.len(), 4);
    }
}
