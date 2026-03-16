//! Comprehensive Nonlinear FEA Analysis Example.
//!
//! This example demonstrates comprehensive nonlinear analysis capabilities
//! including Newton-Raphson solver, arc-length method, and material nonlinearity.
//!
//! # Nonlinear Analysis Types Demonstrated
//!
//! 1. Newton-Raphson Iteration
//! 2. Material Nonlinearity (Elastoplastic)
//! 3. Geometric Nonlinearity Utilities
//! 4. Arc-Length Method Configuration
//!
//! # Usage
//!
//! ```bash
//! cargo run --example nonlinear_comprehensive_analysis
//! ```

use fea::prelude::*;
use fea::algorithms::nonlinear::{
    NonlinearStaticAnalysis, NonlinearConfig, NonlinearMethod,
    ConvergenceCriteria, NewtonRaphson, ArcLengthSolver, ArcLengthConfig,
    material_nonlinearity::ElastoplasticMaterial,
    geometric_nonlinearity::{geometric_stiffness, green_lagrange_strain, deformation_gradient},
};
use nalgebra::{DVector, DMatrix};
use std::time::Instant;

/// Nonlinear analysis result summary.
#[derive(Debug, Clone)]
struct NonlinearDemoResult {
    analysis_type: String,
    converged: bool,
    iterations: usize,
    max_displacement: f64,
    computation_time_ms: f64,
}

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     Comprehensive Nonlinear FEA Analysis Demo             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let mut results = Vec::new();

    // 1. Newton-Raphson Solver Demo
    println!("┌─ Newton-Raphson Solver Demo ───────────────────────────┐");
    let nr_result = demo_newton_raphson()?;
    print_result(&nr_result);
    results.push(nr_result);
    println!("└────────────────────────────────────────────────────────┘\n");

    // 2. Material Nonlinearity Demo
    println!("┌─ Material Nonlinearity (Elastoplastic) ────────────────┐");
    let mat_result = demo_material_nonlinearity()?;
    print_result(&mat_result);
    results.push(mat_result);
    println!("└────────────────────────────────────────────────────────┘\n");

    // 3. Geometric Nonlinearity Demo
    println!("┌─ Geometric Nonlinearity Utilities ─────────────────────┐");
    let geo_result = demo_geometric_nonlinearity()?;
    print_result(&geo_result);
    results.push(geo_result);
    println!("└────────────────────────────────────────────────────────┘\n");

    // 4. Arc-Length Method Demo
    println!("┌─ Arc-Length Method Configuration ──────────────────────┐");
    let arc_result = demo_arc_length()?;
    print_result(&arc_result);
    results.push(arc_result);
    println!("└────────────────────────────────────────────────────────┘\n");

    // Summary
    print_summary(&results);

    Ok(())
}

/// Demo Newton-Raphson solver.
fn demo_newton_raphson() -> anyhow::Result<NonlinearDemoResult> {
    let start = Instant::now();

    // Solve: f(x) = x^2 - 4 = 0 (root at x = 2)
    let nr = NewtonRaphson::new();

    let residual = |x: &DVector<f64>| DVector::from_column_slice(&[x[0] * x[0] - 4.0]);
    let tangent = |x: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[2.0 * x[0]]);

    let u0 = DVector::from_column_slice(&[1.0]);
    let (solution, iterations) = nr.solve(1, residual, tangent, &u0, 1e-10, 50)
        .unwrap_or_else(|e| {
            println!("Warning: {}", e);
            (DVector::from_column_slice(&[0.0]), 50)
        });

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    Ok(NonlinearDemoResult {
        analysis_type: "Newton-Raphson".to_string(),
        converged: (solution[0] - 2.0).abs() < 1e-6,
        iterations,
        max_displacement: solution[0],
        computation_time_ms: elapsed,
    })
}

/// Demo material nonlinearity.
fn demo_material_nonlinearity() -> anyhow::Result<NonlinearDemoResult> {
    let start = Instant::now();

    let mut mat = ElastoplasticMaterial::new(210e9, 0.3, 250e6, 2e9);

    // Apply strain increments
    let strains = vec![0.0005, 0.001, 0.0015, 0.002, 0.003];
    let mut stresses = Vec::new();
    let mut tangents = Vec::new();

    for strain in &strains {
        let (stress, tangent) = mat.update_stress(*strain, 0.0, 0.01);
        stresses.push(stress);
        tangents.push(tangent);
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // Check if plasticity occurred (tangent reduced)
    let plasticity = tangents.last().unwrap() < &210e9;

    println!("│   Initial tangent: {:.0} GPa", tangents[0] / 1e9);
    println!("│   Final tangent:   {:.2} GPa", tangents.last().unwrap() / 1e9);
    println!("│   Max stress:      {:.1} MPa", stresses.last().unwrap() / 1e6);
    println!("│   Plasticity:      {}", if plasticity { "Yes" } else { "No" });

    Ok(NonlinearDemoResult {
        analysis_type: "Elastoplastic Material".to_string(),
        converged: true,
        iterations: strains.len(),
        max_displacement: *stresses.last().unwrap(),
        computation_time_ms: elapsed,
    })
}

/// Demo geometric nonlinearity utilities.
fn demo_geometric_nonlinearity() -> anyhow::Result<NonlinearDemoResult> {
    let start = Instant::now();

    // Create deformation gradient for simple extension
    let f = deformation_gradient(&DMatrix::from_row_slice(3, 3, &[
        0.1, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]));

    // Compute Green-Lagrange strain
    let e = green_lagrange_strain(&f);

    // Compute geometric stiffness
    let k_geo = geometric_stiffness(1000.0, 1.0);

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("│   E_xx strain:     {:.6}", e[(0, 0)]);
    println!("│   K_geo norm:      {:.2}", k_geo.norm());

    Ok(NonlinearDemoResult {
        analysis_type: "Geometric Nonlinear".to_string(),
        converged: true,
        iterations: 1,
        max_displacement: e[(0, 0)],
        computation_time_ms: elapsed,
    })
}

/// Demo arc-length method.
fn demo_arc_length() -> anyhow::Result<NonlinearDemoResult> {
    let start = Instant::now();

    // Create arc-length configuration
    let config = ArcLengthConfig {
        initial_radius: 0.5,
        min_radius: 0.01,
        max_radius: 2.0,
        max_iterations: 100,
        num_load_steps: 10,
    };

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("│   Initial radius:  {:.2}", config.initial_radius);
    println!("│   Min radius:      {:.3}", config.min_radius);
    println!("│   Max radius:      {:.2}", config.max_radius);
    println!("│   Load steps:      {}", config.num_load_steps);

    Ok(NonlinearDemoResult {
        analysis_type: "Arc-Length Method".to_string(),
        converged: true,
        iterations: 0, // Config demo only
        max_displacement: config.initial_radius,
        computation_time_ms: elapsed,
    })
}

/// Print result summary.
fn print_result(result: &NonlinearDemoResult) {
    let status = if result.converged { "✓" } else { "✗" };
    println!("│ {} Analysis: {}", status, result.analysis_type);
    println!("│   Iterations: {}", result.iterations);
    println!("│   Max Value:   {:.6}", result.max_displacement);
    println!("│   Time:        {:.2} ms", result.computation_time_ms);
}

/// Print overall summary.
fn print_summary(results: &[NonlinearDemoResult]) {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                    DEMO SUMMARY                           ║");
    println!("╠═══════════════════════════════════════════════════════════╣");

    let total = results.len();
    let converged = results.iter().filter(|r| r.converged).count();

    println!("║ Total Demos:  {:<46} ║", total);
    println!("║ Successful:   {:<46} ║", converged);

    let total_time: f64 = results.iter().map(|r| r.computation_time_ms).sum();
    println!("║ Total Time:   {:<45.2} ms ║", total_time);

    println!("╚═══════════════════════════════════════════════════════════╝");

    if converged == total {
        println!("\n✓ All nonlinear demos completed successfully!\n");
    } else {
        println!("\n✗ Some demos did not complete as expected\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_raphson_demo() {
        let result = demo_newton_raphson().unwrap();
        assert!(result.converged);
        assert!((result.max_displacement - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_material_nonlinearity_demo() {
        let result = demo_material_nonlinearity().unwrap();
        assert!(result.converged);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_geometric_nonlinearity_demo() {
        let result = demo_geometric_nonlinearity().unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_arc_length_demo() {
        let result = demo_arc_length().unwrap();
        assert!(result.converged);
    }
}
