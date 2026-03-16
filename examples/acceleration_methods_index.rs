//! FEA Acceleration Methods - Complete Index.
//!
//! This example provides a comprehensive index of all acceleration methods
//! available in the FEA library, with links to detailed examples.

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║    FEA Acceleration Methods - Complete Reference Index   ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("This index provides an overview of all acceleration methods");
    println!("available in the FEA library.\n");

    // Section 1: Classical Methods
    print_section_header("1. Classical Acceleration Methods");
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Method                    │ Description                    │");
    println!("├───────────────────────────┼────────────────────────────────┤");
    println!("│ CG + Jacobi               │ Basic preconditioned CG        │");
    println!("│ CG + Chebyshev            │ CG with polynomial precond     │");
    println!("│ BiCGSTAB                  │ Biconjugate gradient stabilized│");
    println!("│ GMRES                     │ Generalized minimal residual   │");
    println!("│ SOR                       │ Successive over-relaxation     │");
    println!("└───────────────────────────┴────────────────────────────────┘");
    println!("See: acceleration_quick_start.rs\n");

    // Section 2: Advanced Preconditioning
    print_section_header("2. Advanced Preconditioning");
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Method                    │ Description                    │");
    println!("├───────────────────────────┼────────────────────────────────┤");
    println!("│ FSAI                      │ Factorized sparse approx inv   │");
    println!("│ Elasticity Prec           │ Physics-based preconditioner   │");
    println!("│ Additive Schwarz          │ Domain decomposition prec      │");
    println!("│ Block Recursive           │ Block-wise preconditioning     │");
    println!("│ Multilevel Schwarz        │ Hierarchical preconditioning   │");
    println!("└───────────────────────────┴────────────────────────────────┘");
    println!("See: practical_fea_acceleration.rs\n");

    // Section 3: Spectral Methods
    print_section_header("3. Spectral Acceleration Methods");
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Method                    │ Description                    │");
    println!("├───────────────────────────┼────────────────────────────────┤");
    println!("│ Chebyshev Semi-Iter       │ Chebyshev polynomial iter      │");
    println!("│ Spectral Deflation        │ Deflate small eigenvalues      │");
    println!("│ Rational Chebyshev        │ Rational filter acceleration   │");
    println!("│ Matrix Power Prec         │ Polynomial preconditioner      │");
    println!("└───────────────────────────┴────────────────────────────────┘");
    println!("See: acceleration_reference_guide.rs\n");

    // Section 4: Recycling Methods
    print_section_header("4. Krylov Subspace Recycling");
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Method                    │ Description                    │");
    println!("├───────────────────────────┼────────────────────────────────┤");
    println!("│ GCRO-DR                   │ GMRES with deflated restarting │");
    println!("│ Recycling BiCGSTAB        │ BiCGSTAB with subspace recyc   │");
    println!("│ Deflated CG               │ CG with deflation vectors      │");
    println!("│ Augmented Krylov          │ Krylov with augmentation       │");
    println!("└───────────────────────────┴────────────────────────────────┘");
    println!("See: comprehensive_solver_comparison.rs\n");

    // Section 5: Adaptive Methods
    print_section_header("5. Adaptive Acceleration");
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Method                    │ Description                    │");
    println!("├───────────────────────────┼────────────────────────────────┤");
    println!("│ Unified Adaptive          │ Auto-select best strategy      │");
    println!("│ Anderson Acceleration     │ Extrapolation from history     │");
    println!("│ PID-Controlled            │ Feedback-controlled params     │");
    println!("│ Convergence Monitor       │ Real-time convergence track    │");
    println!("└───────────────────────────┴────────────────────────────────┘");
    println!("See: master_acceleration_demo.rs\n");

    // Section 6: Mixed Precision
    print_section_header("6. Mixed Precision Methods");
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Method                    │ Description                    │");
    println!("├───────────────────────────┼────────────────────────────────┤");
    println!("│ Mixed Prec Refinement     │ Low prec solve + high prec corr│");
    println!("│ FP16 Simulation           │ Simulated half precision       │");
    println!("│ Iterative Refinement      │ Residual correction iter       │");
    println!("└───────────────────────────┴────────────────────────────────┘");
    println!("See: mixed_precision_example.rs\n");

    // Section 7: Eigenvalue Acceleration
    print_section_header("7. Eigenvalue Acceleration");
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Method                    │ Description                    │");
    println!("├───────────────────────────┼────────────────────────────────┤");
    println!("│ Krylov-Schur              │ Krylov-Schur eigensolver       │");
    println!("│ IRAM                      │ Implicitly restarted Arnoldi   │");
    println!("│ Thick-Restart Lanczos     │ Lanczos with thick restart     │");
    println!("│ Randomized SVD            │ Randomized singular value dec  │");
    println!("└───────────────────────────┴────────────────────────────────┘");
    println!("See: advanced_eigensolvers module\n");

    // Section 8: Performance Utilities
    print_section_header("8. Performance & Benchmarking");
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Utility                   │ Description                    │");
    println!("├───────────────────────────┼────────────────────────────────┤");
    println!("│ PerformanceMetrics        │ Timing and iteration stats     │");
    println!("│ BenchmarkTimer            │ Fine-grained timing            │");
    println!("│ PerformanceProfiler       │ Component-wise profiling       │");
    println!("│ MemoryTracker             │ Memory usage estimation        │");
    println!("│ compare_solvers           │ Multi-solver comparison        │");
    println!("└───────────────────────────┴────────────────────────────────┘");
    println!("See: acceleration_benchmark_suite.rs\n");

    // Section 9: GPU Acceleration
    print_section_header("9. GPU Acceleration");
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Module                    │ Description                    │");
    println!("├───────────────────────────┼────────────────────────────────┤");
    println!("│ gpu_kernel_library        │ GPU vector/sparse/CG kernels   │");
    println!("│ gpu_kernels_complete      │ Complete CUDA kernel library   │");
    println!("│ gpu_advanced_solvers      │ Advanced GPU solver impl       │");
    println!("└───────────────────────────┴────────────────────────────────┘");
    println!("See: gpu_accelerated_solvers.rs\n");

    // Example Quick Reference
    print_section_header("Example Quick Reference");
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Example File              │ Purpose                        │");
    println!("├───────────────────────────┼────────────────────────────────┤");
    println!("│ acceleration_quick_start  │ Get started in 5 minutes       │");
    println!("│ practical_fea_accel       │ Real-world FEA applications    │");
    println!("│ acceleration_reference    │ Complete API reference         │");
    println!("│ acceleration_validation   │ Automated validation suite     │");
    println!("│ acceleration_benchmark    │ Performance benchmarks         │");
    println!("│ comprehensive_integration │ Integration test suite         │");
    println!("│ master_validation_suite   │ Full validation coverage       │");
    println!("│ ultimate_showcase         │ All methods demonstration      │");
    println!("└───────────────────────────┴────────────────────────────────┘\n");

    // Configuration Quick Reference
    print_section_header("Configuration Quick Reference");
    println!("For well-conditioned problems (κ < 100):");
    println!("  → Use CG + Jacobi (simple, effective)");
    println!();
    println!("For moderately conditioned problems (100 < κ < 1000):");
    println!("  → Use CG + Chebyshev(3) or Anderson(5)");
    println!();
    println!("For ill-conditioned problems (κ > 1000):");
    println!("  → Use AcceleratedPCG with all features enabled");
    println!("  → Or Unified Adaptive solver");
    println!();
    println!("For multiple right-hand sides:");
    println!("  → Use GCRO-DR or Recycling BiCGSTAB");
    println!();
    println!("For eigenvalue problems:");
    println!("  → Use Krylov-Schur or IRAM");
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("For more information, see the crate documentation:");
    println!("  cargo doc --open");
    println!("═══════════════════════════════════════════════════════════");
}

fn print_section_header(title: &str) {
    println!("\n{}", "═".repeat(60));
    println!("{}", title);
    println!("{}\n", "═".repeat(60));
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_index_compiles() {
        // Just verify the example compiles
        assert!(true);
    }
}
