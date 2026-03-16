//! Comprehensive solver comparison and benchmarking example.
//!
//! Demonstrates and compares all available iterative solvers and preconditioners.

use fea::prelude::*;
use fea::algorithms::solvers::{
    Preconditioner, IterativeConfig,
};
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("=== FEA Solver Comparison Benchmark ===\n");

    // Test 1: Compare all solvers on a model problem
    compare_solvers()?;

    // Test 2: Preconditioner comparison
    compare_preconditioners()?;

    // Test 3: Acceleration techniques
    demonstrate_acceleration()?;

    // Test 4: FEA model solver comparison
    fea_solver_comparison()?;

    println!("\n=== Benchmark Complete ===");

    Ok(())
}

/// Compare all available solvers
fn compare_solvers() -> anyhow::Result<()> {
    println!("Solver Comparison");
    println!("-----------------\n");

    // Create a test problem: 2D Poisson equation
    let n = 50;
    let (a, b) = create_poisson_problem(n);

    println!("Problem size: {} x {} = {}", n, n, n * n);
    println!("Testing solvers on 2D Poisson equation...\n");

    let config = IterativeConfig::default();

    println!("{:<20} | {:>8} | {:>10} | {:>12}", "Solver", "Iterations", "Residual", "Time");
    println!("---------------------|----------|------------|-------------");

    // Direct solver (reference)
    let direct = DirectSolver::new();
    let direct_config = DirectConfig { use_cholesky: true };
    let start = std::time::Instant::now();
    let _direct_result = direct.solve(&a, &b, &direct_config)?;
    let direct_time = start.elapsed();
    println!("{:<20} | {:>8} | {:>10} | {:>10.1} ms",
        "Direct (Cholesky)", "-", "-", direct_time.as_secs_f64() * 1000.0);

    // CG
    let cg = CGSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = cg.solve(&a, &b, &config)?;
    let time = start.elapsed();
    println!("{:<20} | {:>8} | {:>10.2e} | {:>10.1} ms",
        "CG",
        result.iterations.unwrap_or(0),
        result.residual_norm.unwrap_or(0.0),
        time.as_secs_f64() * 1000.0);

    // PCG
    let pcg = PCGSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = pcg.solve(&a, &b, &config)?;
    let time = start.elapsed();
    println!("{:<20} | {:>8} | {:>10.2e} | {:>10.1} ms",
        "PCG",
        result.iterations.unwrap_or(0),
        result.residual_norm.unwrap_or(0.0),
        time.as_secs_f64() * 1000.0);

    // GMRES
    let gmres = GMRESSolver::with_restart(30);
    let start = std::time::Instant::now();
    let result = gmres.solve(&a, &b, &config)?;
    let time = start.elapsed();
    println!("{:<20} | {:>8} | {:>10.2e} | {:>10.1} ms",
        "GMRES(30)",
        result.iterations.unwrap_or(0),
        result.residual_norm.unwrap_or(0.0),
        time.as_secs_f64() * 1000.0);

    // BiCGSTAB
    let bicgstab = BiCGSTABSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = bicgstab.solve(&a, &b, &config)?;
    let time = start.elapsed();
    println!("{:<20} | {:>8} | {:>10.2e} | {:>10.1} ms",
        "BiCGSTAB",
        result.iterations.unwrap_or(0),
        result.residual_norm.unwrap_or(0.0),
        time.as_secs_f64() * 1000.0);

    // GCR
    let gcr = GCRSolver::new();
    let start = std::time::Instant::now();
    let result = gcr.solve(&a, &b, &config)?;
    let time = start.elapsed();
    println!("{:<20} | {:>8} | {:>10.2e} | {:>10.1} ms",
        "GCR",
        result.iterations.unwrap_or(0),
        result.residual_norm.unwrap_or(0.0),
        time.as_secs_f64() * 1000.0);

    // QMR
    let qmr = QMRSolver::new();
    let start = std::time::Instant::now();
    let result = qmr.solve(&a, &b, &config)?;
    let time = start.elapsed();
    println!("{:<20} | {:>8} | {:>10.2e} | {:>10.1} ms",
        "QMR",
        result.iterations.unwrap_or(0),
        result.residual_norm.unwrap_or(0.0),
        time.as_secs_f64() * 1000.0);

    // SOR
    let sor = SORSolver::new(1.5);
    let start = std::time::Instant::now();
    let (x, iter, residual, converged) = sor.solve(&a, &b)?;
    let time = start.elapsed();
    println!("{:<20} | {:>8} | {:>10.2e} | {:>10.1} ms",
        "SOR(1.5)",
        iter,
        residual,
        time.as_secs_f64() * 1000.0);
    let _ = converged;
    let _ = x;

    Ok(())
}

/// Compare preconditioners
fn compare_preconditioners() -> anyhow::Result<()> {
    println!("\nPreconditioner Comparison");
    println!("-------------------------\n");

    let n = 50;
    let (a, b) = create_poisson_problem(n);

    println!("Testing preconditioners with CG solver...\n");

    println!("{:<20} | {:>10} | {:>12}", "Preconditioner", "Iterations", "Time");
    println!("---------------------|------------|-------------");

    // No preconditioner
    let config = IterativeConfig {
        preconditioner: Preconditioner::None,
        ..Default::default()
    };
    let cg = CGSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = cg.solve(&a, &b, &config)?;
    let time = start.elapsed();
    println!("{:<20} | {:>10} | {:>10.1} ms",
        "None",
        result.iterations.unwrap_or(0),
        time.as_secs_f64() * 1000.0);

    // Jacobi
    let config = IterativeConfig {
        preconditioner: Preconditioner::Jacobi,
        ..Default::default()
    };
    let cg = CGSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = cg.solve(&a, &b, &config)?;
    let time = start.elapsed();
    println!("{:<20} | {:>10} | {:>10.1} ms",
        "Jacobi",
        result.iterations.unwrap_or(0),
        time.as_secs_f64() * 1000.0);

    // SSOR
    let config = IterativeConfig {
        preconditioner: Preconditioner::SSOR(1.5),
        ..Default::default()
    };
    let cg = CGSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = cg.solve(&a, &b, &config)?;
    let time = start.elapsed();
    println!("{:<20} | {:>10} | {:>10.1} ms",
        "SSOR(1.5)",
        result.iterations.unwrap_or(0),
        time.as_secs_f64() * 1000.0);

    // Block-Jacobi (2x2 blocks)
    let bj = BlockJacobi::new(&a, 2);
    if let Some(bj) = bj {
        // Apply preconditioner manually
        let start = std::time::Instant::now();
        let z = bj.apply(&b);
        let time = start.elapsed();
        println!("{:<20} | {:>10} | {:>10.1} ms (apply only)",
            "Block-Jacobi(2)",
            "-",
            time.as_secs_f64() * 1000.0);
        let _ = z;
    }

    // Chebyshev polynomial
    let cheb = ChebyshevPreconditioner::new(0.1, 10.0, 5);
    let start = std::time::Instant::now();
    let z = cheb.apply(&a, &b);
    let time = start.elapsed();
    println!("{:<20} | {:>10} | {:>10.1} ms",
        "Chebyshev(5)",
        "-",
        time.as_secs_f64() * 1000.0);
    let _ = z;

    Ok(())
}

/// Demonstrate acceleration techniques
fn demonstrate_acceleration() -> anyhow::Result<()> {
    println!("\nAcceleration Techniques");
    println!("-----------------------\n");

    let n = 100;
    let (a, b) = create_poisson_problem(n);

    println!("Testing acceleration with CG solver (n={})...\n", n);

    println!("{:<20} | {:>10} | {:>12}", "Method", "Iterations", "Speedup");
    println!("---------------------|------------|-------------");

    // Baseline: Standard CG
    let config = IterativeConfig::default();
    let cg = CGSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let base_result = cg.solve(&a, &b, &config)?;
    let base_time = start.elapsed();
    let baseline_iter = base_result.iterations.unwrap_or(1) as f64;
    println!("{:<20} | {:>10} | {:>10}",
        "CG (baseline)",
        base_result.iterations.unwrap_or(0),
        "1.0x");

    // Anderson acceleration
    let config = IterativeConfig::default().with_anderson(5);
    let cg = CGSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = cg.solve(&a, &b, &config)?;
    let time = start.elapsed();
    let speedup = baseline_iter / result.iterations.unwrap_or(1) as f64;
    println!("{:<20} | {:>10} | {:>10.1}x (time: {:.1} ms)",
        "Anderson(5)",
        result.iterations.unwrap_or(0),
        speedup,
        time.as_secs_f64() * 1000.0);

    // Aitken acceleration demo
    let mut aitken = AitkenAcceleration::new();
    let config = IterativeConfig::default();
    let cg = CGSolver::with_tolerance(1e-10);
    let start = std::time::Instant::now();
    let result = cg.solve(&a, &b, &config)?;
    let time = start.elapsed();

    // Apply Aitken to the solution sequence
    let x_vec = DVector::from_column_slice(&result.solution);
    let r = &b - &a * &x_vec;
    aitken.update(&x_vec.data.as_vec());
    aitken.update(&x_vec.data.as_vec());
    if let Some(_x_accel) = aitken.update(&x_vec.data.as_vec()) {
        // Aitken accelerated solution
    }

    println!("{:<20} | {:>10} | {:>10} (time: {:.1} ms)",
        "Aitken",
        result.iterations.unwrap_or(0),
        "-",
        time.as_secs_f64() * 1000.0);

    Ok(())
}

/// Compare solvers on actual FEA problem
fn fea_solver_comparison() -> anyhow::Result<()> {
    println!("\nFEA Model Solver Comparison");
    println!("---------------------------\n");

    // Create a simple truss model
    let mut model = Model::<Truss2>::new();

    // Create 10-node truss
    let n_nodes = 10;
    for i in 0..n_nodes {
        let x = (i % 5) as f64 * 1.0;
        let y = (i / 5) as f64 * 1.0;
        model.add_node(Node::new_2d(x, y));
    }

    // Add elements
    for i in 0..n_nodes - 1 {
        model.add_element(Truss2::new(i, i + 1));
    }

    // Boundary conditions
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(n_nodes - 1, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n_nodes - 1, Dof::Uz));

    // Load
    model.add_load(Load::new(n_nodes - 1, Dof::Ux, 1000.0));

    println!("Truss model: {} nodes, {} elements", n_nodes, n_nodes - 1);
    println!("\n{:<20} | {:>10} | {:>12}", "Solver", "Iterations", "Displacement");
    println!("---------------------|------------|-------------");

    // Use legacy solver for comparison
    let mut legacy_model = create_legacy_truss_model();
    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut legacy_model)?;
    let disp = result.u[result.u.len() - 3];
    println!("{:<20} | {:>10} | {:>12.6}", "Direct (ref)", "-", disp);

    println!("\nNote: New analysis module uses different DOF ordering.");
    println!("For production use, use the new LinearStaticAnalysis.");

    Ok(())
}

/// Create 2D Poisson problem matrix
fn create_poisson_problem(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let size = n * n;
    let mut a = DMatrix::zeros(size, size);
    let h = 1.0 / (n + 1) as f64;
    let scale = 1.0 / (h * h);

    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            a[(idx, idx)] = 4.0 * scale;

            if i > 0 {
                a[(idx, idx - n)] = -scale;
            }
            if i < n - 1 {
                a[(idx, idx + n)] = -scale;
            }
            if j > 0 {
                a[(idx, idx - 1)] = -scale;
            }
            if j < n - 1 {
                a[(idx, idx + 1)] = -scale;
            }
        }
    }

    let b = DVector::from_element(size, 1.0);
    (a, b)
}

/// Create legacy truss model for comparison
fn create_legacy_truss_model() -> Model<Truss2LegacyCompat> {
    let mut model = Model::<Truss2LegacyCompat>::new();

    let n_nodes = 10;
    for i in 0..n_nodes {
        let x = (i % 5) as f64 * 1.0;
        let y = (i / 5) as f64 * 1.0;
        model.add_node(Node::new_2d(x, y));
    }

    let e = 210e9;
    let a = 1e-4;

    for i in 0..n_nodes - 1 {
        model.add_element(Truss2LegacyCompat::new(i, i + 1, e, a));
    }

    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(n_nodes - 1, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n_nodes - 1, Dof::Uz));

    model.add_load(Load::new(n_nodes - 1, Dof::Ux, 1000.0));

    model
}
