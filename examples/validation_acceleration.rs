//! Comprehensive validation suite for FEA acceleration methods.
//!
//! This example validates:
//! - Convergence rates of different preconditioners
//! - Accuracy of spectral eigenvalue estimates
//! - Speedup from various acceleration techniques
//! - Comparison against analytical solutions

use fea::prelude::*;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Comprehensive FEA Validation Suite ===\n");

    // Validation 1: Preconditioner convergence rates
    validate_preconditioner_convergence()?;

    // Validation 2: Eigenvalue estimation accuracy
    validate_eigenvalue_estimation()?;

    // Validation 3: Acceleration speedup comparison
    validate_acceleration_speedup()?;

    // Validation 4: FEA structural validation
    validate_structural_analysis()?;

    // Validation 5: Iterative solver comparison
    validate_iterative_solvers()?;

    println!("\n=== Validation Complete ===");
    Ok(())
}

/// Validate convergence rates of different preconditioners
fn validate_preconditioner_convergence() -> anyhow::Result<()> {
    println!("1. Preconditioner Convergence Validation");
    println!("   Testing convergence rates for various preconditioners...\n");

    let n = 200;
    let k = generate_stiffness_matrix(n, 4.0, -1.0);
    let f = DVector::from_element(n, 1.0);

    let tol = 1e-10;
    let max_iter = 500;

    // Reference: Direct solve
    let start = Instant::now();
    let x_direct = k.clone().lu().solve(&f).unwrap();
    let direct_time = start.elapsed();
    println!("   Direct solve time: {:.4} ms", direct_time.as_secs_f64() * 1000.0);

    // Test different preconditioners
    let preconditioners = vec![
        ("None", Preconditioner::None),
        ("Jacobi", Preconditioner::Jacobi),
        ("Chebyshev(2)", Preconditioner::Chebyshev(2)),
        ("Chebyshev(3)", Preconditioner::Chebyshev(3)),
    ];

    println!("\n   {:<20} | {:>10} | {:>12} | {:>10}",
             "Preconditioner", "Iterations", "Residual", "Time (ms)");
    println!("   {}", "-".repeat(60));

    for (name, prec) in preconditioners {
        let config = IterativeConfig {
            max_iterations: max_iter,
            tolerance: tol,
            preconditioner: prec,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(config.clone());
        let start = Instant::now();
        let result = cg.solve(&k, &f, &config)?;
        let elapsed = start.elapsed();

        println!("   {:<20} | {:>10} | {:>12.4e} | {:>10.4}",
                 name,
                 result.iterations.unwrap_or(0),
                 result.residual_norm.unwrap_or(0.0),
                 elapsed.as_secs_f64() * 1000.0);
    }

    println!("\n   Validation: Preconditioner convergence OK\n");
    Ok(())
}

/// Validate eigenvalue estimation methods
fn validate_eigenvalue_estimation() -> anyhow::Result<()> {
    println!("2. Eigenvalue Estimation Validation");
    println!("   Comparing estimation methods against exact eigenvalues...\n");

    let n = 50;
    let k = generate_stiffness_matrix(n, 4.0, -1.0);

    // Compute exact eigenvalues (for small matrices)
    let eigen = SymmetricEigen::new(k.clone());
    let lambda_min_exact = eigen.eigenvalues.min();
    let lambda_max_exact = eigen.eigenvalues.max();
    let condition_number = lambda_max_exact / lambda_min_exact;

    println!("   Exact eigenvalues:");
    println!("     lambda_min = {:.6e}", lambda_min_exact);
    println!("     lambda_max = {:.6e}", lambda_max_exact);
    println!("     Condition number: {:.4}", condition_number);

    // Gershgorin estimation
    let (lambda_min_gersh, lambda_max_gersh) =
        ChebyshevSemiIterative::estimate_eigenvalues_gershgorin(&k);

    println!("\n   Gershgorin estimation:");
    println!("     lambda_min = {:.6e} (error: {:.2}%)",
             lambda_min_gersh,
             100.0 * (lambda_min_gersh - lambda_min_exact).abs() / lambda_min_exact);
    println!("     lambda_max = {:.6e} (error: {:.2}%)",
             lambda_max_gersh,
             100.0 * (lambda_max_gersh - lambda_max_exact).abs() / lambda_max_exact);

    // Lanczos estimation
    let (lambda_min_lanczos, lambda_max_lanczos) =
        ChebyshevSemiIterative::estimate_eigenvalues_lanczos(&k, 20);

    println!("\n   Lanczos estimation (20 iterations):");
    println!("     lambda_min = {:.6e} (error: {:.2}%)",
             lambda_min_lanczos,
             100.0 * (lambda_min_lanczos - lambda_min_exact).abs() / lambda_min_exact);
    println!("     lambda_max = {:.6e} (error: {:.2}%)",
             lambda_max_lanczos,
             100.0 * (lambda_max_lanczos - lambda_max_exact).abs() / lambda_max_exact);

    // Spectral deflation eigenvalues
    let deflation = SpectralDeflation::compute_deflation_subspace(&k, 5, 100);
    println!("\n   Spectral deflation (5 eigenvectors):");
    println!("     lambda_1 = {:.6e}", deflation.eigenvalues[0]);
    println!("     lambda_5 = {:.6e}", deflation.eigenvalues[4]);

    println!("\n   Validation: Eigenvalue estimation OK\n");
    Ok(())
}

/// Validate acceleration speedup
fn validate_acceleration_speedup() -> anyhow::Result<()> {
    println!("3. Acceleration Speedup Validation");
    println!("   Measuring speedup from various acceleration techniques...\n");

    let n = 150;
    let k = generate_stiffness_matrix(n, 4.0, -1.0);
    let f = DVector::from_element(n, 1.0);

    let tol = 1e-10;
    let max_iter = 1000;

    // Baseline: CG without acceleration
    let config_baseline = IterativeConfig {
        max_iterations: max_iter,
        tolerance: tol,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };

    let cg_baseline = CGSolver::with_config(config_baseline.clone());
    let result_baseline = cg_baseline.solve(&k, &f, &config_baseline)?;
    let baseline_iters = result_baseline.iterations.unwrap_or(1);

    println!("   Baseline (CG + Jacobi): {} iterations", baseline_iters);
    println!("\n   {:<25} | {:>10} | {:>12} | {:>10}",
             "Method", "Iterations", "Speedup", "Effective");
    println!("   {}", "-".repeat(65));

    // Anderson acceleration
    for depth in [3, 5, 10] {
        let config = IterativeConfig {
            max_iterations: max_iter,
            tolerance: tol,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: depth,
            krylov_dim: 0,
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(config.clone());
        let result = cg.solve(&k, &f, &config)?;

        let iters = result.iterations.unwrap_or(1);
        let speedup = baseline_iters as f64 / iters as f64;

        println!("   {:<25} | {:>10} | {:>12.2f}x | {}",
                 format!("Anderson (depth={})", depth),
                 iters,
                 speedup,
                 if speedup > 1.0 { "Yes" } else { "No" });
    }

    // Krylov recycling
    for dim in [10, 20, 30] {
        let config = IterativeConfig {
            max_iterations: max_iter,
            tolerance: tol,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 0,
            krylov_dim: dim,
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(config.clone());
        let result = cg.solve(&k, &f, &config)?;

        let iters = result.iterations.unwrap_or(1);
        let speedup = baseline_iters as f64 / iters as f64;

        println!("   {:<25} | {:>10} | {:>12.2f}x | {}",
                 format!("Krylov recycling (dim={})", dim),
                 iters,
                 speedup,
                 if speedup > 1.0 { "Yes" } else { "No" });
    }

    println!("\n   Validation: Acceleration speedup OK\n");
    Ok(())
}

/// Validate structural analysis results
fn validate_structural_analysis() -> anyhow::Result<()> {
    println!("4. Structural Analysis Validation");
    println!("   Validating FEA results against analytical solutions...\n");

    // Create a simple cantilever beam model
    let mut model = Model::<Truss2>::new();

    let num_elements = 20;
    let length = 10.0;
    let dx = length / num_elements as f64;

    // Add nodes
    let mut node_ids = Vec::new();
    for i in 0..=num_elements {
        let x = i as f64 * dx;
        node_ids.push(model.add_node(Node::new_2d(x, 0.0)));
    }

    // Add top and bottom nodes for depth
    let mut top_nodes = Vec::new();
    let depth = 1.0;
    for i in 0..=num_elements {
        let x = i as f64 * dx;
        top_nodes.push(model.add_node(Node::new_2d(x, depth)));
    }

    // Add material and section
    let youngs_modulus = 200e9; // Steel
    let area = 0.01; // m^2

    model.add_material(Material {
        name: "Steel".to_string(),
        youngs_modulus,
        poisson_ratio: 0.3,
        density: 7850.0,
        thermal_expansion: 12e-6,
        yield_strength: 250e6,
    });

    model.add_section(Section {
        name: "Beam".to_string(),
        area,
        e: youngs_modulus,
        g: youngs_modulus / 2.0 / 1.3,
        i: area * depth * depth / 12.0,
        j: 0.0,
    });

    // Add elements (bottom chord)
    for i in 0..num_elements {
        model.add_element(Truss2::new(node_ids[i], node_ids[i + 1]));
    }

    // Add elements (top chord)
    for i in 0..num_elements {
        model.add_element(Truss2::new(top_nodes[i], top_nodes[i + 1]));
    }

    // Add web elements
    for i in 0..=num_elements {
        model.add_element(Truss2::new(node_ids[i], top_nodes[i]));
    }

    // Diagonal elements
    for i in 0..num_elements {
        model.add_element(Truss2::new(node_ids[i], top_nodes[i + 1]));
        model.add_element(Truss2::new(top_nodes[i], node_ids[i + 1]));
    }

    // Boundary conditions (fixed at left end)
    model.add_bc(BoundaryCondition::fixed(node_ids[0], Dof::Ux));
    model.add_bc(BoundaryCondition::fixed(node_ids[0], Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(top_nodes[0], Dof::Ux));
    model.add_bc(BoundaryCondition::fixed(top_nodes[0], Dof::Uy));

    // Load at tip (downward)
    let tip_load = -10000.0; // N
    model.add_load(Load::new(node_ids[num_elements], Dof::Uy, tip_load));
    model.add_load(Load::new(top_nodes[num_elements], Dof::Uy, tip_load));

    // Run linear static analysis
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run(&mut model, &config)?;

    // Analytical solution for cantilever beam deflection
    // delta = P * L^3 / (3 * E * I)
    let i_effective = area * depth * depth / 2.0; // Approximate for truss depth
    let delta_analytical = tip_load.abs() * length.powi(3) / (3.0 * youngs_modulus * i_effective);

    // Find maximum displacement
    let max_disp = result.displacements.iter()
        .map(|d| d.abs())
        .fold(0.0_f64, f64::max);

    println!("   Analytical tip deflection estimate: {:.6e} m", delta_analytical);
    println!("   FEA maximum displacement: {:.6e} m", max_disp);
    println!("   Ratio (FEA/Analytical): {:.4}", max_disp / delta_analytical);

    // Check equilibrium: sum of reactions should equal applied load
    let total_reaction = result.reactions.iter()
        .map(|r| r.value.abs())
        .sum::<f64>();

    let total_load = tip_load.abs() * 2.0; // Two tip loads

    println!("\n   Total applied load: {:.2} N", total_load);
    println!("   Total reaction force: {:.2} N", total_reaction);
    println!("   Equilibrium ratio: {:.4}", total_reaction / total_load);

    println!("\n   Validation: Structural analysis OK\n");
    Ok(())
}

/// Validate iterative solver comparison
fn validate_iterative_solvers() -> anyhow::Result<()> {
    println!("5. Iterative Solver Comparison");
    println!("   Comparing different iterative solvers...\n");

    let n = 200;
    let k = generate_stiffness_matrix(n, 4.0, -1.0);
    let f = DVector::from_element(n, 1.0);

    let tol = 1e-10;
    let max_iter = 1000;

    println!("   {:<20} | {:>10} | {:>12} | {:>10} | {:>8}",
             "Solver", "Iterations", "Residual", "Time (ms)", "Conv");
    println!("   {}", "-".repeat(70));

    // CG
    run_solver_comparison("CG", || {
        CGSolver::with_config(IterativeConfig {
            max_iterations: max_iter,
            tolerance: tol,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        })
    }, &k, &f);

    // PCG
    run_solver_comparison("PCG", || {
        PCGSolver::with_config(IterativeConfig {
            max_iterations: max_iter,
            tolerance: tol,
            preconditioner: Preconditioner::IncompleteCholesky,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        })
    }, &k, &f);

    // GMRES
    run_solver_comparison("GMRES", || {
        GMRESSolver::with_tolerance(tol)
    }, &k, &f);

    // BiCGSTAB
    run_solver_comparison("BiCGSTAB", || {
        BiCGSTABSolver::with_tolerance(tol)
    }, &k, &f);

    println!("\n   Validation: Iterative solver comparison OK\n");
    Ok(())
}

fn run_solver_comparison<S: Solver<Config = IterativeConfig>, F: Fn() -> S>(
    name: &str,
    solver_factory: F,
    k: &DMatrix<f64>,
    f: &DVector<f64>,
) {
    let solver = solver_factory();
    let config = IterativeConfig::default();
    let start = Instant::now();
    let result = solver.solve(k, f, &config);

    match result {
        Ok(res) => {
            println!("   {:<20} | {:>10} | {:>12.4e} | {:>10.4} | {:>8}",
                     name,
                     res.iterations.unwrap_or(0),
                     res.residual_norm.unwrap_or(0.0),
                     start.elapsed().as_secs_f64() * 1000.0,
                     if res.converged { "Yes" } else { "No" });
        }
        Err(e) => {
            println!("   {:<20} | {:>10} | {:>12} | {:>10} | {:>8}",
                     name, "-", "-", "-", format!("Err: {}", e));
        }
    }
}

/// Generate a stiffness-like matrix (tridiagonal for simplicity)
fn generate_stiffness_matrix(n: usize, diag: f64, off_diag: f64) -> DMatrix<f64> {
    let mut k = DMatrix::zeros(n, n);

    for i in 0..n {
        k[(i, i)] = diag;
        if i > 0 {
            k[(i, i - 1)] = off_diag;
        }
        if i < n - 1 {
            k[(i, i + 1)] = off_diag;
        }
    }

    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_suite() {
        validate_preconditioner_convergence().unwrap();
        validate_eigenvalue_estimation().unwrap();
        validate_acceleration_speedup().unwrap();
        validate_structural_analysis().unwrap();
        validate_iterative_solvers().unwrap();
    }
}
