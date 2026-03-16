//! FEA Framework Quick Start Guide.
//!
//! This example provides a step-by-step introduction to the FEA framework:
//! - Basic static analysis
//! - Adding GPU acceleration
//! - Dynamic analysis
//! - Material nonlinearity
//! - Next steps

use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║          FEA Framework Quick Start Guide                  ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Step 1: Your First Analysis
    step_1_first_analysis()?;

    // Step 2: Adding GPU Acceleration
    step_2_gpu_acceleration()?;

    // Step 3: Dynamic Analysis
    step_3_dynamic_analysis()?;

    // Step 4: Material Nonlinearity
    step_4_material_nonlinearity()?;

    // Next Steps
    next_steps();

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Quick Start Complete                         ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Step 1: Your First FEA Analysis
fn step_1_first_analysis() -> anyhow::Result<()> {
    println!("┌─ Step 1: Your First FEA Analysis ────────────────────────┐");
    println!("│");
    println!("│ Creating a simple cantilever beam model...");

    // Create the model
    let mut model = Model::<Truss2>::new();

    // Add nodes (simple 2-element cantilever)
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(0.5, 0.0));
    let n2 = model.add_node(Node::new_2d(1.0, 0.0));

    println!("│   Added 3 nodes");

    // Add elements
    model.add_element(Truss2::new(n0, n1));
    model.add_element(Truss2::new(n1, n2));

    println!("│   Added 2 truss elements");

    // Add material
    model.add_material(STEEL_A36);

    // Add section
    model.add_section(Section::circular("round", 0.01));

    println!("│   Added Steel A36 material");
    println!("│   Added circular section (radius = 0.01 m)");

    // Add boundary conditions (fixed at left end)
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
    }

    println!("│   Fixed support at node 0");

    // Add load at free end
    model.add_load(Load::new(n2, Dof::Uy, -1000.0));

    println!("│   Applied 1000 N load at node 2");
    println!("│");

    // Run analysis
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    // Display results
    let max_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);

    println!("│ Results:");
    println!("│   Max displacement: {:.6e} m", max_disp);
    println!("│   Total DOFs: {}", result.displacements.len());
    println!("│   Analysis completed successfully!");
    println!("│");
    println!("│ Key Concepts:");
    println!("│   1. Create Model<Element>");
    println!("│   2. Add nodes and elements");
    println!("│   3. Add material and section");
    println!("│   4. Add boundary conditions");
    println!("│   5. Add loads");
    println!("│   6. Run analysis");
    println!("│   7. Access results");
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Step 2: Adding GPU Acceleration
fn step_2_gpu_acceleration() -> anyhow::Result<()> {
    println!("┌─ Step 2: Adding GPU Acceleration ────────────────────────┐");
    println!("│");

    use fea::gpu::{gpu_available, GPUCGSolver, GPUCSRMatrix};

    // Check GPU availability
    if gpu_available() {
        println!("│ GPU is available!");
    } else {
        println!("│ GPU not available - showing code example only");
        println!("│");
        println!("│ When GPU is available:");
    }

    println!("│");
    println!("│ To use GPU acceleration:");
    println!("│");
    println!("│   1. Import GPU modules:");
    println!("│      use fea::gpu::*;");
    println!("│");
    println!("│   2. Create sparse matrix on GPU:");
    println!("│      let matrix = GPUCSRMatrix::from_csr(...);");
    println!("│");
    println!("│   3. Create GPU solver:");
    println!("│      let solver = GPUCGSolver::new(device_id, tol, max_iter);");
    println!("│");
    println!("│   4. Solve:");
    println!("│      let result = solver.solve(&matrix, &b, &mut x)?;");
    println!("│");

    if gpu_available() {
        // Simple demo
        let row_ptr = vec![0, 2, 4, 6];
        let col_ind = vec![0, 1, 0, 1, 1, 2];
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0];

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0);
        let b = vec![1.0, 1.0, 1.0];

        let solver = GPUCGSolver::new(0, 1e-8, 100);
        let mut x = vec![0.0; 3];
        let result = solver.solve(&matrix, &b, &mut x)?;

        println!("│ Demo Result:");
        println!("│   Iterations: {}", result.iterations);
        println!("│   Converged: {}", result.converged);
    }

    println!("│");
    println!("│ Available GPU Solvers:");
    println!("│   - GPUCGSolver (Conjugate Gradient)");
    println!("│   - GPUGMRESSolver (GMRES with restart)");
    println!("│   - GPUBiCGSTABSolver (BiCGSTAB)");
    println!("│   - GPUPCGSolver (Preconditioned CG)");
    println!("│");
    println!("│ Available Preconditioners:");
    println!("│   - GPUILUPreconditioner (ILU(0))");
    println!("│   - GPUSSORPreconditioner (SSOR)");
    println!("│   - GPUChebyshevPreconditioner");
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Step 3: Dynamic Analysis
fn step_3_dynamic_analysis() -> anyhow::Result<()> {
    println!("┌─ Step 3: Dynamic Analysis ───────────────────────────────┐");
    println!("│");

    use fea::algorithms::explicit_dynamics::{
        ExplicitDynamicAnalyzer, ExplicitConfig, ExplicitMethod,
    };

    println!("│ For dynamic analysis, use ExplicitDynamicAnalyzer:");
    println!("│");

    let n = 50;
    let mass = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_element(n, 1.0));

    let mut stiffness = nalgebra::DMatrix::zeros(n, n);
    for i in 0..n {
        stiffness[(i, i)] = 100.0;
        if i > 0 {
            stiffness[(i, i - 1)] = -50.0;
            stiffness[(i - 1, i)] = -50.0;
        }
    }

    // Configure analysis
    let config = ExplicitConfig {
        method: ExplicitMethod::CentralDifference,
        time_step: 0.001,
        total_time: 0.1,
        damping_alpha: 0.05,
        damping_beta: 0.0,
        auto_time_step: false,
        output_frequency: 10,
    };

    println!("│ Configuration:");
    println!("│   Method: Central Difference");
    println!("│   Time step: {:.3} s", config.time_step);
    println!("│   Total time: {:.1} s", config.total_time);
    println!("│   Damping: α = {:.2}", config.damping_alpha);
    println!("│");

    let analyzer = ExplicitDynamicAnalyzer::with_config(mass, stiffness, config);

    let u0 = vec![0.0; n];
    let v0 = vec![0.0; n];
    let force_fn = |_t: f64, _u: &[f64]| vec![0.0; n];

    let result = analyzer.analyze(&u0, &v0, &force_fn)?;

    println!("│ Results:");
    println!("│   Time steps: {}", result.num_steps);
    println!("│   Output points: {}", result.time_points.len());

    // Energy check
    let e0 = result.energy_history[0].total_energy;
    let ef = result.energy_history.last().unwrap().total_energy;
    let energy_error = if e0 > 1e-15 {
        ((ef - e0) / e0 * 100.0).abs()
    } else {
        0.0
    };

    println!("│   Energy conservation: {:.2}% error", energy_error);
    println!("│");
    println!("│ Available Methods:");
    println!("│   - ExplicitMethod::CentralDifference");
    println!("│   - ExplicitMethod::ForwardEuler");
    println!("│   - ExplicitMethod::RungeKutta4");
    println!("│   - ExplicitMethod::ExplicitNewmark");
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Step 4: Material Nonlinearity
fn step_4_material_nonlinearity() -> anyhow::Result<()> {
    println!("┌─ Step 4: Material Nonlinearity ──────────────────────────┐");
    println!("│");

    use fea::materials::nonlinear::{
        LinearElastic, VonMisesPlasticity, StressState,
    };

    println!("│ For material nonlinearity:");
    println!("│");

    // Linear elastic
    let elastic = LinearElastic::new(210e9, 0.3, 7850.0);
    let strain = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0];
    let stress = elastic.stress(&strain, 0.0);

    println!("│ 1. Linear Elastic:");
    println!("│    E = {:.0} GPa, ν = {:.2}", elastic.e / 1e9, elastic.nu);
    println!("│    σ = {:.1} MPa (ε = {:.4})", stress[0] / 1e6, strain[0]);
    println!("│");

    // Von Mises plasticity
    let plastic = VonMisesPlasticity::new(210e9, 0.3, 250e6, 1e9);
    let state = StressState::default();
    let (new_state, _tangent) = plastic.update(&strain, &state);

    println!("│ 2. Von Mises Plasticity:");
    println!("│    Yield stress: {:.0} MPa", plastic.yield_stress / 1e6);
    println!("│    Hardening: {:.0} GPa", plastic.hardening / 1e9);
    println!("│    Stress after update: {:.1} MPa", new_state.stress[0] / 1e6);
    println!("│    Plastic strain: {:.6}", new_state.plastic_strain);
    println!("│");

    println!("│ Available Material Models:");
    println!("│   - LinearElastic (Hooke's law)");
    println!("│   - VonMisesPlasticity (isotropic hardening)");
    println!("│   - NeoHookean (hyperelastic)");
    println!("│   - JohnsonCook (viscoplastic, rate-dependent)");
    println!("│   - DamageModel (continuum damage mechanics)");
    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(())
}

/// Next Steps
fn next_steps() {
    println!("┌─ Next Steps ─────────────────────────────────────────────┐");
    println!("│");
    println!("│ Congratulations! You've completed the Quick Start Guide.");
    println!("│");
    println!("│ Recommended Next Examples:");
    println!("│");
    println!("│ 1. complete_feature_showcase.rs");
    println!("│    Comprehensive demonstration of all features");
    println!("│");
    println!("│ 2. gpu_performance_guide.rs");
    println!("│    GPU optimization tips and best practices");
    println!("│");
    println!("│ 3. regression_tests.rs");
    println!("│    Automated testing and validation");
    println!("│");
    println!("│ 4. amg_solver_demo.rs");
    println!("│    Advanced preconditioning techniques");
    println!("│");
    println!("│ Documentation:");
    println!("│   - All modules have inline documentation");
    println!("│   - Run 'cargo doc' to generate HTML docs");
    println!("│   - Check individual example files for detailed comments");
    println!("│");
    println!("│ Support:");
    println!("│   - Report issues on GitHub");
    println!("│   - Check existing examples for usage patterns");
    println!("│   - Review unit tests for API details");
    println!("│");
    println!("└────────────────────────────────────────────────────────┘");
}
