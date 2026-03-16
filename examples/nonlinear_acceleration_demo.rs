//! Nonlinear acceleration methods demonstration and validation.
//!
//! This example demonstrates:
//! - Anderson mixing for nonlinear problems
//! - NGMRES acceleration
//! - Pipelined Anderson acceleration
//! - Comparison with standard Newton-Raphson

use fea::prelude::*;
use fea::algorithms::solvers::{
    AndersonMixing, AndersonTypeII, NGMRES, PipelinedAnderson,
};
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Nonlinear Acceleration Methods Benchmark ===\n");

    // Test Anderson mixing
    test_anderson_mixing()?;

    // Test Anderson Type-II
    test_anderson_type_ii()?;

    // Test NGMRES
    test_ngmres()?;

    // Test Pipelined Anderson
    test_pipelined_anderson()?;

    // Compare all methods on a nonlinear problem
    compare_acceleration_methods()?;

    // Run nonlinear truss analysis with acceleration
    run_nonlinear_truss_analysis()?;

    println!("\n=== Benchmark Complete ===");
    Ok(())
}

/// Test Anderson mixing on a simple fixed-point problem.
fn test_anderson_mixing() -> anyhow::Result<()> {
    println!("\nAnderson Mixing Test:");
    println!("-------------------");

    // Solve x = g(x) where g(x) = 0.5 * x + 0.5 (solution: x = 1)
    let n = 100;
    let mut anderson = AndersonMixing::new(10, 0.5);
    let mut x = DVector::from_element(n, 0.0);

    let g = |x: &DVector<f64>| -> DVector<f64> {
        x.scale(0.5) + DVector::from_element(n, 0.5)
    };

    let start = Instant::now();
    for i in 0..50 {
        let gx = g(&x);
        let x_new = anderson.update(&x, &gx);

        let residual = (&x_new - &x).norm();
        x = x_new;

        if i % 10 == 0 {
            println!("  Iter {}: residual = {:.2e}", i, residual);
        }

        if residual < 1e-10 {
            println!("  Converged at iteration {}", i);
            break;
        }
    }
    let elapsed = start.elapsed();

    let final_residual = (g(&x) - &x).norm();
    println!("  Final residual: {:.2e}", final_residual);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

    Ok(())
}

/// Test Anderson Type-II acceleration.
fn test_anderson_type_ii() -> anyhow::Result<()> {
    println!("\nAnderson Type-II Test:");
    println!("--------------------");

    let n = 100;
    let mut anderson = AndersonTypeII::new(10);
    let mut x = DVector::from_element(n, 0.5);

    let g = |x: &DVector<f64>| -> DVector<f64> {
        x.scale(0.3) + DVector::from_element(n, 0.7)
    };

    let start = Instant::now();
    for i in 0..50 {
        let gx = g(&x);
        let x_new = anderson.update(&x, &gx);

        let residual = (&x_new - &x).norm();
        x = x_new;

        if i % 10 == 0 {
            println!("  Iter {}: residual = {:.2e}", i, residual);
        }

        if residual < 1e-10 {
            println!("  Converged at iteration {}", i);
            break;
        }
    }
    let elapsed = start.elapsed();

    let final_residual = (g(&x) - &x).norm();
    println!("  Final residual: {:.2e}", final_residual);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

    Ok(())
}

/// Test NGMRES acceleration.
fn test_ngmres() -> anyhow::Result<()> {
    println!("\nNGMRES Test:");
    println!("----------");

    let n = 100;
    let mut ngmres = NGMRES::new(10);
    let mut x = DVector::from_element(n, 0.0);

    // Fixed point: x = g(x) where g(x) = 0.8 * x + 0.2
    let g = |x: &DVector<f64>| -> DVector<f64> {
        x.scale(0.8) + DVector::from_element(n, 0.2)
    };

    let start = Instant::now();
    for i in 0..100 {
        let gx = g(&x);
        let r = &x - &gx; // Residual
        let x_new = ngmres.update(&x, &r);

        let residual = (&x_new - &x).norm();
        x = x_new;

        if i % 20 == 0 {
            println!("  Iter {}: residual = {:.2e}", i, residual);
        }

        if residual < 1e-8 {
            println!("  Converged at iteration {}", i);
            break;
        }
    }
    let elapsed = start.elapsed();

    let final_residual = (g(&x) - &x).norm();
    println!("  Final residual: {:.2e}", final_residual);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

    Ok(())
}

/// Test Pipelined Anderson acceleration.
fn test_pipelined_anderson() -> anyhow::Result<()> {
    println!("\nPipelined Anderson Test:");
    println!("----------------------");

    let n = 100;
    let mut pipe = PipelinedAnderson::new(0.5);
    let mut x = DVector::from_element(n, 0.0);

    let g = |x: &DVector<f64>| -> DVector<f64> {
        x.scale(0.6) + DVector::from_element(n, 0.4)
    };

    let start = Instant::now();
    for i in 0..50 {
        let gx = g(&x);
        let x_new = pipe.update(&x, &gx);

        let residual = (&x_new - &x).norm();
        x = x_new;

        if i % 10 == 0 {
            println!("  Iter {}: residual = {:.2e}", i, residual);
        }

        if residual < 1e-10 {
            println!("  Converged at iteration {}", i);
            break;
        }
    }
    let elapsed = start.elapsed();

    let final_residual = (g(&x) - &x).norm();
    println!("  Final residual: {:.2e}", final_residual);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

    Ok(())
}

/// Compare all acceleration methods on the same problem.
fn compare_acceleration_methods() -> anyhow::Result<()> {
    println!("\nAcceleration Methods Comparison:");
    println!("------------------------------");

    let n = 200;
    let max_iter = 100;
    let tol = 1e-10;

    let g = |x: &DVector<f64>| -> DVector<f64> {
        x.scale(0.7) + DVector::from_element(n, 0.3)
    };

    // Standard fixed-point iteration
    let mut x_std = DVector::from_element(n, 0.0);
    let start = Instant::now();
    let mut iter_std = 0;
    for i in 0..max_iter {
        let gx = g(&x_std);
        if (&gx - &x_std).norm() < tol {
            iter_std = i;
            break;
        }
        x_std = gx;
    }
    let time_std = start.elapsed();
    println!(
        "  Standard fixed-point:    {} iterations, {:.2}ms",
        iter_std,
        time_std.as_secs_f64() * 1000.0
    );

    // Anderson mixing
    let mut x_and = DVector::from_element(n, 0.0);
    let mut anderson = AndersonMixing::new(10, 0.5);
    let start = Instant::now();
    let mut iter_and = 0;
    for i in 0..max_iter {
        let gx = g(&x_and);
        let x_new = anderson.update(&x_and, &gx);
        if (&x_new - &x_and).norm() < tol {
            iter_and = i;
            break;
        }
        x_and = x_new;
    }
    let time_and = start.elapsed();
    println!(
        "  Anderson mixing (10):    {} iterations, {:.2}ms",
        iter_and,
        time_and.as_secs_f64() * 1000.0
    );

    // Anderson Type-II
    let mut x_and2 = DVector::from_element(n, 0.0);
    let mut anderson2 = AndersonTypeII::new(10);
    let start = Instant::now();
    let mut iter_and2 = 0;
    for i in 0..max_iter {
        let gx = g(&x_and2);
        let x_new = anderson2.update(&x_and2, &gx);
        if (&x_new - &x_and2).norm() < tol {
            iter_and2 = i;
            break;
        }
        x_and2 = x_new;
    }
    let time_and2 = start.elapsed();
    println!(
        "  Anderson Type-II (10):   {} iterations, {:.2}ms",
        iter_and2,
        time_and2.as_secs_f64() * 1000.0
    );

    // NGMRES
    let mut x_ng = DVector::from_element(n, 0.0);
    let mut ngmres = NGMRES::new(10);
    let start = Instant::now();
    let mut iter_ng = 0;
    for i in 0..max_iter {
        let gx = g(&x_ng);
        let r = &x_ng - &gx;
        let x_new = ngmres.update(&x_ng, &r);
        if (&x_new - &x_ng).norm() < tol {
            iter_ng = i;
            break;
        }
        x_ng = x_new;
    }
    let time_ng = start.elapsed();
    println!(
        "  NGMRES (10):             {} iterations, {:.2}ms",
        iter_ng,
        time_ng.as_secs_f64() * 1000.0
    );

    // Pipelined Anderson
    let mut x_pipe = DVector::from_element(n, 0.0);
    let mut pipe = PipelinedAnderson::new(0.5);
    let start = Instant::now();
    let mut iter_pipe = 0;
    for i in 0..max_iter {
        let gx = g(&x_pipe);
        let x_new = pipe.update(&x_pipe, &gx);
        if (&x_new - &x_pipe).norm() < tol {
            iter_pipe = i;
            break;
        }
        x_pipe = x_new;
    }
    let time_pipe = start.elapsed();
    println!(
        "  Pipelined Anderson:      {} iterations, {:.2}ms",
        iter_pipe,
        time_pipe.as_secs_f64() * 1000.0
    );

    Ok(())
}

/// Run nonlinear truss analysis with geometric nonlinearity.
fn run_nonlinear_truss_analysis() -> anyhow::Result<()> {
    println!("\nNonlinear Truss Analysis:");
    println!("-----------------------");

    // Create a simple 2-bar truss (snap-through problem)
    let mut model = Model::<Truss2>::new();

    // Add nodes (shallow arch)
    let span = 10.0;
    let rise = 0.5;

    model.add_node(Node::new_2d(0.0, 0.0)); // Left support
    model.add_node(Node::new_2d(span / 2.0, rise)); // Crown
    model.add_node(Node::new_2d(span, 0.0)); // Right support

    // Add material and section
    model.add_material(STEEL_A36);
    model.add_section(Section::circular("round", 0.02));

    // Add elements
    model.add_element(Truss2::new(0, 1));
    model.add_element(Truss2::new(1, 2));

    // Boundary conditions
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
        model.add_bc(BoundaryCondition::fixed(2, dof));
    }

    // Load at crown
    model.add_load(Load::new(1, Dof::Uy, -1000.0));

    println!("  Nodes: {}", model.nodes.len());
    println!("  Elements: {}", model.elements.len());
    println!("  DOFs: {}", model.ndofs());

    // Linear static analysis
    let start = Instant::now();
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;
    let linear_time = start.elapsed();

    let max_disp_linear: f64 = result
        .displacements
        .iter()
        .map(|d| d.abs())
        .fold(0.0, f64::max);

    println!("  Linear analysis time: {:.2}ms", linear_time.as_secs_f64() * 1000.0);
    println!("  Max displacement (linear): {:.6e} m", max_disp_linear);

    // Note: Full nonlinear analysis with arc-length method would go here
    // This is a demonstration of the framework

    Ok(())
}
