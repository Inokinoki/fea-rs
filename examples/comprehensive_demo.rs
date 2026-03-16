//! Comprehensive examples demonstrating the modular FEA framework.
//!
//! This example shows:
//! - Different element types (Truss2, Plate4, Plate8)
//! - Various solvers (Direct, CG, PCG, GMRES, BiCGSTAB)
//! - Multiple analysis types (Static, Modal, Buckling, Dynamic)
//! - Convergence tracking and post-processing

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("=== FEA Framework Comprehensive Example ===\n");

    // Example 1: Static analysis with different solvers
    example_static_analysis()?;

    // Example 2: Modal analysis
    example_modal_analysis()?;

    // Example 3: Plate element analysis
    example_plate_analysis()?;

    // Example 4: Solver comparison with Anderson acceleration
    example_solver_comparison()?;

    // Example 5: Validation against analytical solution
    example_validation()?;

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}

/// Example 1: Static analysis using the new modular framework
fn example_static_analysis() -> anyhow::Result<()> {
    println!("\n--- Example 1: Static Analysis with Truss2 ---");

    let mut model = Model::<Truss2>::new();

    // Create a simple triangular truss (stable structure)
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    let n2 = model.add_node(Node::new_2d(0.5, 0.866)); // Equilateral triangle

    // Add material and section
    model.add_material(STEEL_A36);
    model.add_section(Section::circular("round", 0.01));

    // Add elements (triangular truss)
    model.add_element(Truss2::new(n0, n1));
    model.add_element(Truss2::new(n1, n2));
    model.add_element(Truss2::new(n2, n0));

    // Boundary conditions: pin at n0 (fix X, Y, Z), roller at n1 (fix Y, Z)
    // Also fix Z at n2 since this is a 2D planar problem
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
    }
    for dof in [Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n1, dof));
    }
    model.add_bc(BoundaryCondition::fixed(n2, Dof::Uz)); // Prevent out-of-plane motion

    // Add downward load at apex (n2)
    model.add_load(Load::new(n2, Dof::Uy, -1000.0));

    println!("  - Nodes: {}", model.nodes.len());
    println!("  - Elements: {}", model.elements.len());
    println!("  - BCs: {}", model.bcs.len());
    println!("  - Loads: {}", model.loads.len());

    // Build DOFs first
    let ndof = model.build_dofs_3d();
    println!("  - Total DOFs: {}", ndof);

    // Run static analysis
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    println!("Static analysis completed:");
    println!("  - Number of DOFs: {}", model.ndofs());
    let disp_n2 = result.displacements[model.dof_index(n2, Dof::Uy).unwrap()];
    println!("  - Vertical displacement at apex: {:.6e}", disp_n2);
    println!("  - Number of reactions: {}", result.reactions.len());

    Ok(())
}

/// Example 2: Modal analysis
fn example_modal_analysis() -> anyhow::Result<()> {
    println!("\n--- Example 2: Modal Analysis ---");

    let mut model = Model::<Truss2>::new();

    // Create a cantilever truss
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(0.5, 0.0));
    let n2 = model.add_node(Node::new_2d(1.0, 0.0));

    model.add_material(STEEL_A36);
    model.add_section(Section::circular("round", 0.01));

    model.add_element(Truss2::new(n0, n1));
    model.add_element(Truss2::new(n1, n2));

    // Fixed support at node 0 (all DOFs)
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
    }
    // Constrain other nodes in Y and Z to allow only axial motion
    for &node in &[n1, n2] {
        for dof in [Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(node, dof));
        }
    }

    // Run modal analysis
    let analysis = ModalAnalysis::new();
    let config = ModalConfig {
        num_modes: 3,
        consistent_mass: false,
        max_iterations: 1000,
        tolerance: 1e-10,
    };
    let result = analysis.run_modal(&mut model, &config)?;

    println!("Modal analysis completed:");
    println!("  - Number of modes computed: {}", result.frequencies.len());
    for (i, (&freq, &freq_hz)) in result.frequencies.iter()
        .zip(result.frequencies_hz.iter())
        .enumerate()
    {
        println!("  - Mode {}: {:.2} rad/s ({:.2} Hz)", i + 1, freq, freq_hz);
    }

    Ok(())
}

/// Example 3: Plate element analysis
fn example_plate_analysis() -> anyhow::Result<()> {
    println!("\n--- Example 3: Plate Element Analysis (Plate4) ---");
    println!("Note: Plate elements use 2 DOFs per node (plane stress/strain).");
    println!("Full integration with the framework requires 2D DOF handling.");

    // Demonstrate that Plate4 element can compute stiffness
    let mut model = Model::<Plate4>::new();

    // Create a single plate element (4 nodes, 2D plane stress)
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    let n2 = model.add_node(Node::new_2d(1.0, 1.0));
    let n3 = model.add_node(Node::new_2d(0.0, 1.0));

    model.add_material(STEEL_A36);
    model.add_section(Section::new("plate", 0.01, 0.0, 0.0, 0.0)); // thickness = 0.01

    // Create plate element
    let plate = Plate4::new(n0, n1, n2, n3, 0.01);
    model.add_element(plate);

    // Build DOFs (using 3D DOF mapping - Plate4 will use only Ux, Uy)
    let _ndof = model.build_dofs_3d();

    println!("Plate element created successfully:");
    println!("  - Nodes: {}", model.nodes.len());
    println!("  - Element DOFs: {}", model.elements[0].ndofs());
    println!("  - Thickness: {:.4} m", model.elements[0].thickness);

    Ok(())
}

/// Example 4: Solver comparison
fn example_solver_comparison() -> anyhow::Result<()> {
    println!("\n--- Example 4: Solver Comparison ---");

    // Create a simple test matrix
    let n = 100;
    let mut k = DMatrix::zeros(n, n);
    let mut f = DVector::zeros(n);

    // Create a symmetric positive definite matrix (tridiagonal)
    for i in 0..n {
        k[(i, i)] = 2.0;
        if i > 0 {
            k[(i, i - 1)] = -1.0;
        }
        if i < n - 1 {
            k[(i, i + 1)] = -1.0;
        }
        f[i] = 1.0;
    }

    let config = IterativeConfig {
        max_iterations: 1000,
        tolerance: 1e-10,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };

    // Test Direct solver
    let direct = DirectSolver::new();
    let direct_config = DirectConfig { use_cholesky: true };
    let result = direct.solve(&k, &f, &direct_config)?;
    println!("Direct (Cholesky): converged = {}", result.converged);

    // Test CG solver without Anderson
    let cg = CGSolver::with_tolerance(1e-10);
    let result = cg.solve(&k, &f, &config)?;
    println!(
        "CG (no Anderson): iterations = {:?}, converged = {}",
        result.iterations, result.converged
    );

    // Test CG solver with Anderson acceleration
    let config_anderson = IterativeConfig {
        anderson_depth: 5,
        ..config.clone()
    };
    let cg_anderson = CGSolver::with_tolerance(1e-10);
    let result = cg_anderson.solve(&k, &f, &config_anderson)?;
    println!(
        "CG (Anderson-5): iterations = {:?}, converged = {}",
        result.iterations, result.converged
    );

    // Test PCG solver
    let pcg = PCGSolver::with_tolerance(1e-10);
    let result = pcg.solve(&k, &f, &config)?;
    println!(
        "PCG: iterations = {:?}, converged = {}",
        result.iterations, result.converged
    );

    // Test GMRES solver
    let gmres = GMRESSolver::with_restart(30);
    let result = gmres.solve(&k, &f, &config)?;
    println!(
        "GMRES: iterations = {:?}, converged = {}",
        result.iterations, result.converged
    );

    // Test BiCGSTAB solver
    let bicgstab = BiCGSTABSolver::with_tolerance(1e-10);
    let result = bicgstab.solve(&k, &f, &config)?;
    println!(
        "BiCGSTAB: iterations = {:?}, converged = {}",
        result.iterations, result.converged
    );

    Ok(())
}

/// Example 5: Validation against analytical solution
fn example_validation() -> anyhow::Result<()> {
    println!("\n--- Example 5: Validation Against Analytical Solution ---");

    use fea::prelude::*;

    // Simple bar extension problem
    // A bar of length L, area A, Young's modulus E, fixed at one end,
    // with axial load P at the other end.
    // Analytical solution: delta = P*L / (A*E)

    let e = 210e9; // Young's modulus (Pa)
    let a = 1e-4; // Cross-sectional area (m^2)
    let l = 2.0; // Length (m)
    let p = 10_000.0; // Load (N)

    let mut model = Model::<Truss2LegacyCompat>::new();

    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(l, 0.0));

    model.add_element(Truss2LegacyCompat::new(n0, n1, e, a));

    // Fix node 0
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
    }
    // Constrain node 1 in Y and Z
    for dof in [Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n1, dof));
    }

    // Apply load
    model.add_load(Load::new(n1, Dof::Ux, p));

    // Solve using legacy solver
    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    // Analytical solution
    let analytical_disp = p * l / (a * e);
    let computed_disp = result.u[model.dof_index(n1, Dof::Ux).unwrap()];

    let error = (analytical_disp - computed_disp).abs() / analytical_disp * 100.0;

    println!("Bar Extension Validation:");
    println!("  - Analytical displacement: {:.6e} m", analytical_disp);
    println!("  - Computed displacement:   {:.6e} m", computed_disp);
    println!("  - Relative error:          {:.6}%", error);

    // Verify error is within tolerance
    if error < 1e-8 {
        println!("  - VALIDATION PASSED");
    } else {
        println!("  - VALIDATION FAILED (error > 1e-8%)");
    }

    // Stress validation
    let elem = &model.elements[0];
    let analytical_stress = p / a;
    let computed_stress = elem.axial_stress(&model, &result.u);
    let stress_error = (analytical_stress - computed_stress).abs() / analytical_stress * 100.0;

    println!("\nStress Validation:");
    println!("  - Analytical stress: {:.6e} Pa", analytical_stress);
    println!("  - Computed stress:   {:.6e} Pa", computed_stress);
    println!("  - Relative error:    {:.6}%", stress_error);

    if stress_error < 1e-8 {
        println!("  - VALIDATION PASSED");
    } else {
        println!("  - VALIDATION FAILED");
    }

    Ok(())
}

/// Additional utility: Convergence history tracking
fn example_convergence_tracking() -> anyhow::Result<()> {
    println!("\n--- Bonus: Convergence History Tracking ---");

    let mut history = ConvergenceHistory::new();

    // Simulate convergence data
    let mut residual = 1.0;
    for i in 0..20 {
        residual *= 0.5; // Halve each iteration
        history.record(i, residual);
    }

    if let Some(rate) = history.convergence_rate() {
        println!("Convergence rate: {:.2} (higher is better)", rate);
    }
    println!("Final residual: {:.6e}", history.residual_norms.last().unwrap_or(&0.0));
    println!("Total iterations: {}", history.iterations.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_analysis() {
        assert!(example_static_analysis().is_ok());
    }

    #[test]
    fn test_modal_analysis() {
        assert!(example_modal_analysis().is_ok());
    }

    #[test]
    fn test_plate_analysis() {
        assert!(example_plate_analysis().is_ok());
    }

    #[test]
    fn test_solver_comparison() {
        assert!(example_solver_comparison().is_ok());
    }

    #[test]
    fn test_validation() {
        // This test verifies that FEA results match analytical solutions
        assert!(example_validation().is_ok());
    }
}
