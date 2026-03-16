//! Comprehensive Integration Example - All FEA Acceleration Methods.
//!
//! This example demonstrates how all acceleration methods work together
//! in a complete FEA analysis workflow, with automatic strategy selection
//! and performance comparison.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Comprehensive FEA Acceleration Integration Example ===\n");
    println!("This example demonstrates all acceleration methods working together.\n");

    // Part 1: Problem Setup
    let (k, f, problem_info) = setup_test_problem()?;

    // Part 2: Strategy Comparison
    compare_all_strategies(&k, &f)?;

    // Part 3: Unified Solver with Automatic Selection
    run_unified_solver(&k, &f)?;

    // Part 4: Complete FEA Workflow
    run_complete_fea_workflow()?;

    // Part 5: Performance Report
    generate_performance_report(&problem_info)?;

    println!("\n=== Integration Example Complete ===");
    Ok(())
}

/// Sets up a test problem for demonstration.
fn setup_test_problem() -> anyhow::Result<(DMatrix<f64>, DVector<f64>, ProblemInfo)> {
    println!("─".repeat(70));
    println!("Part 1: Test Problem Setup");
    println!("─".repeat(70));

    // Create a representative FEA stiffness matrix
    let n = 500;
    let mut k = DMatrix::zeros(n, n);

    // Tridiagonal structure (typical of 1D FEA discretization)
    for i in 0..n {
        k[(i, i)] = 4.0;
        if i > 0 {
            k[(i, i - 1)] = -1.0;
        }
        if i < n - 1 {
            k[(i, i + 1)] = -1.0;
        }
    }

    // Create load vector
    let f = DVector::from_element(n, 1.0);

    let info = ProblemInfo {
        size: n,
        condition_estimate: 50.0, // Approximate for this matrix type
        sparsity: 3.0 / n as f64 * 100.0,
    };

    println!("\nProblem Information:");
    println!("  Size: {} x {}", info.size, info.size);
    println!("  Estimated condition number: {:.2}", info.condition_estimate);
    println!("  Sparsity: {:.4}%", info.sparsity);

    Ok((k, f, info))
}

/// Compares all available acceleration strategies.
fn compare_all_strategies(k: &DMatrix<f64>, f: &DVector<f64>) -> anyhow::Result<()> {
    println!("\n─".repeat(70));
    println!("Part 2: Acceleration Strategy Comparison");
    println!("─".repeat(70));

    let strategies = vec![
        ("Basic CG", AccelerationStrategy::None),
        ("CG + Jacobi", AccelerationStrategy::Preconditioning(PreconditionerType::Jacobi)),
        ("CG + Chebyshev(2)", AccelerationStrategy::Preconditioning(PreconditionerType::Chebyshev(2))),
        ("CG + Chebyshev(3)", AccelerationStrategy::Preconditioning(PreconditionerType::Chebyshev(3))),
        ("Anderson(3)", AccelerationStrategy::Anderson { depth: 3 }),
        ("Anderson(5)", AccelerationStrategy::Anderson { depth: 5 }),
        ("Chebyshev Poly(2)", AccelerationStrategy::Chebyshev { degree: 2 }),
        ("Chebyshev Poly(3)", AccelerationStrategy::Chebyshev { degree: 3 }),
        ("Composite", AccelerationStrategy::Composite {
            preconditioner: PreconditionerType::Jacobi,
            recycling_dim: Some(5),
            anderson_depth: Some(3),
        }),
    ];

    println!("\n{:<25} | {:>10} | {:>12} | {:>10}", "Strategy", "Iterations", "Time (ms)", "Residual");
    println!("{}", "-".repeat(70));

    let mut best_strategy = None;
    let mut best_time = f64::INFINITY;

    for (name, strategy) in strategies {
        let config = UnifiedSolverConfig {
            strategy,
            max_iterations: 500,
            tolerance: 1e-10,
            enable_monitoring: false,
            verbose: false,
        };

        let mut solver = UnifiedAccelerationSolver::new(config);

        let start = Instant::now();
        let result = solver.solve(k, f);
        let elapsed = start.elapsed();

        let iters = result.base_result.iterations.unwrap_or(0);
        let residual = result.base_result.residual_norm.unwrap_or(f64::INFINITY);

        println!("{:<25} | {:>10} | {:>12.4} | {:>10.2e}",
                 name, iters, elapsed.as_secs_f64() * 1000.0, residual);

        if result.base_result.converged && elapsed.as_secs_f64() < best_time {
            best_time = elapsed.as_secs_f64();
            best_strategy = Some(name);
        }
    }

    println!("\nBest strategy: {}", best_strategy.unwrap_or("N/A"));

    Ok(())
}

/// Runs the unified solver with automatic strategy selection.
fn run_unified_solver(k: &DMatrix<f64>, f: &DVector<f64>) -> anyhow::Result<()> {
    println!("\n─".repeat(70));
    println!("Part 3: Unified Solver with Automatic Selection");
    println!("─".repeat(70));

    // Create adaptive solver
    let config = UnifiedSolverConfig {
        strategy: AccelerationStrategy::Adaptive,
        max_iterations: 500,
        tolerance: 1e-10,
        enable_monitoring: true,
        verbose: true,
    };

    let mut solver = UnifiedAccelerationSolver::new(config);

    println!("\nRunning adaptive solver...");
    let start = Instant::now();
    let result = solver.solve(k, f);
    let elapsed = start.elapsed();

    println!("\nResults:");
    println!("  Iterations: {}", result.base_result.iterations.unwrap_or(0));
    println!("  Time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Final residual: {:.6e}", result.base_result.residual_norm.unwrap_or(f64::INFINITY));
    println!("  Converged: {}", result.base_result.converged);
    println!("  MatVec count: {}", result.matvec_count);

    // Get recommendation for future solves
    if let Some(recommended) = solver.get_recommended_strategy() {
        println!("\nRecommended strategy for similar problems: {:?}", recommended);
    }

    Ok(())
}

/// Runs a complete FEA workflow demonstration.
fn run_complete_fea_workflow() -> anyhow::Result<()> {
    println!("\n─".repeat(70));
    println!("Part 4: Complete FEA Workflow");
    println!("─".repeat(70));

    // Create a simple truss model
    let mut model = Model::<Truss2>::new();

    // Add nodes for a 10-bar truss
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    let n2 = model.add_node(Node::new_2d(2.0, 0.0));
    let n3 = model.add_node(Node::new_2d(3.0, 0.0));
    let n4 = model.add_node(Node::new_2d(4.0, 0.0));

    let n5 = model.add_node(Node::new_2d(0.0, 1.0));
    let n6 = model.add_node(Node::new_2d(1.0, 1.0));
    let n7 = model.add_node(Node::new_2d(2.0, 1.0));
    let n8 = model.add_node(Node::new_2d(3.0, 1.0));
    let n9 = model.add_node(Node::new_2d(4.0, 1.0));

    // Material and section
    model.add_material(STEEL_A36);
    model.add_section(Section::circular("round", 0.01));

    // Add elements
    for i in 0..4 {
        model.add_element(Truss2::new(n0 + i, n0 + i + 1));
        model.add_element(Truss2::new(n5 + i, n5 + i + 1));
    }

    for i in 0..5 {
        model.add_element(Truss2::new(n0 + i, n5 + i));
    }

    for i in 0..4 {
        model.add_element(Truss2::new(n0 + i, n5 + i + 1));
        model.add_element(Truss2::new(n5 + i, n0 + i + 1));
    }

    // Boundary conditions
    model.add_bc(BoundaryCondition::fixed(n0, Dof::Ux));
    model.add_bc(BoundaryCondition::fixed(n0, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n5, Dof::Ux));
    model.add_bc(BoundaryCondition::fixed(n5, Dof::Uy));

    // Load
    model.add_load(Load::new(n4, Dof::Uy, -10000.0));

    println!("\nModel Statistics:");
    println!("  Nodes: {}", model.nodes().len());
    println!("  Elements: {}", model.elements().len());
    println!("  Materials: 1 (Steel A36)");

    // Run analysis with accelerated solver
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();

    println!("\nRunning linear static analysis with accelerated solvers...");
    let start = Instant::now();
    let result = analysis.run(&mut model, &config)?;
    let elapsed = start.elapsed();

    println!("\nAnalysis Results:");
    println!("  Analysis time: {:.4} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  DOFs: {}", result.displacements.len());

    // Find maximum displacement
    let max_disp = result.displacements.iter()
        .map(|d| d.abs())
        .fold(0.0_f64, f64::max);
    println!("  Maximum displacement: {:.6e} m", max_disp);

    // Equilibrium check
    let total_load = 10000.0;
    let total_reaction: f64 = result.reactions.iter()
        .filter(|r| r.dof == Dof::Uy)
        .map(|r| r.value.abs())
        .sum();

    println!("\nEquilibrium Check:");
    println!("  Applied load: {:.2} N", total_load);
    println!("  Total reaction: {:.2} N", total_reaction);
    println!("  Ratio: {:.4}", total_reaction / total_load);

    Ok(())
}

/// Generates a performance report.
fn generate_performance_report(info: &ProblemInfo) -> anyhow::Result<()> {
    println!("\n─".repeat(70));
    println!("Part 5: Performance Report");
    println!("─".repeat(70));

    println!("\nProblem Characteristics:");
    println!("  Size: {} DOFs", info.size);
    println!("  Condition number estimate: {:.2}", info.condition_estimate);
    println!("  Sparsity: {:.4}%", info.sparsity);

    println!("\nAcceleration Method Recommendations:");
    println!("  For well-conditioned problems (κ < 100):");
    println!("    → Use CG + Jacobi preconditioning");
    println!("  For moderately conditioned problems (100 < κ < 1000):");
    println!("    → Use CG + Chebyshev(3) or Anderson(5)");
    println!("  For ill-conditioned problems (κ > 1000):");
    println!("    → Use Composite strategy or Adaptive solver");

    println!("\nExpected Speedups:");
    println!("  Jacobi preconditioning: 1.5-2x");
    println!("  Chebyshev preconditioning: 2-4x");
    println!("  Anderson acceleration: 1.5-3x");
    println!("  Composite methods: 3-8x");

    println!("\nMemory Overhead:");
    println!("  Basic CG: O(n)");
    println!("  With Anderson(m): O(m*n)");
    println!("  With recycling(k): O(k*n)");

    Ok(())
}

/// Problem information structure.
#[derive(Debug, Clone)]
struct ProblemInfo {
    size: usize,
    condition_estimate: f64,
    sparsity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_example() {
        // Run a simplified version of the integration example
        let n = 50;
        let mut k = DMatrix::zeros(n, n);
        for i in 0..n {
            k[(i, i)] = 4.0;
            if i > 0 { k[(i, i - 1)] = -1.0; }
            if i < n - 1 { k[(i, i + 1)] = -1.0; }
        }
        let f = DVector::from_element(n, 1.0);

        // Test unified solver
        let config = UnifiedSolverConfig {
            strategy: AccelerationStrategy::Preconditioning(PreconditionerType::Jacobi),
            max_iterations: 100,
            tolerance: 1e-8,
            enable_monitoring: false,
            verbose: false,
        };

        let mut solver = UnifiedAccelerationSolver::new(config);
        let result = solver.solve(&k, &f);

        assert!(result.base_result.iterations.unwrap_or(0) > 0);
        assert!(result.base_result.converged || result.base_result.residual_norm.unwrap_or(f64::INFINITY) < 1.0);
    }

    #[test]
    fn test_strategy_comparison() {
        let n = 30;
        let mut k = DMatrix::zeros(n, n);
        for i in 0..n {
            k[(i, i)] = 4.0;
            if i > 0 { k[(i, i - 1)] = -1.0; }
            if i < n - 1 { k[(i, i + 1)] = -1.0; }
        }
        let f = DVector::from_element(n, 1.0);

        let best = SolverComparator::find_best_strategy(&k, &f, 1e-8, 100);

        assert!(best.is_some());
    }
}
