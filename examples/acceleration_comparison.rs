//! Comprehensive acceleration methods comparison example.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("=== Acceleration Methods Comparison ===\n");

    // Test 1: Linear convergence acceleration
    test_linear_convergence()?;

    // Test 2: Sublinear convergence acceleration
    test_sublinear_convergence()?;

    // Test 3: Oscillatory convergence
    test_oscillatory_convergence()?;

    // Test 4: High-dimensional problem
    test_high_dimensional()?;

    println!("\n=== All Tests Complete ===");
    Ok(())
}

/// Test with linearly converging sequence
fn test_linear_convergence() -> anyhow::Result<()> {
    println!("1. Linear Convergence (x = 0.8x + b)");
    println!("   Target: x* = 5b\n");

    let b = DVector::from_element(50, 1.0);
    let fixed_point = DVector::from_element(50, 5.0);
    let g = |x: &DVector<f64>| x.scale(0.8) + &b;

    run_acceleration_test("Linear", g, &fixed_point, 500)?;
    println!();

    Ok(())
}

/// Test with sublinear convergence
fn test_sublinear_convergence() -> anyhow::Result<()> {
    println!("2. Sublinear Convergence (x = x - 0.1*grad)");
    println!("   Target: minimum at origin\n");

    let fixed_point = DVector::zeros(50);
    let g = |x: &DVector<f64>| x - x.scale(0.1);

    run_acceleration_test("Sublinear", g, &fixed_point, 500)?;
    println!();

    Ok(())
}

/// Test with oscillatory convergence
fn test_oscillatory_convergence() -> anyhow::Result<()> {
    println!("3. Oscillatory Convergence (x = -0.7x + b)");
    println!("   Target: x* = b/1.7\n");

    let b = DVector::from_element(50, 1.0);
    let fixed_point = b.scale(1.0 / 1.7);
    let g = |x: &DVector<f64>| x.scale(-0.7) + &b;

    run_acceleration_test("Oscillatory", g, &fixed_point, 200)?;
    println!();

    Ok(())
}

/// Test with high-dimensional problem
fn test_high_dimensional() -> anyhow::Result<()> {
    println!("4. High-Dimensional (x = 0.95x + b, n=200)");
    println!("   Target: x* = 20b\n");

    let b = DVector::from_element(200, 0.1);
    let fixed_point = DVector::from_element(200, 2.0);
    let g = |x: &DVector<f64>| x.scale(0.95) + &b;

    run_acceleration_test("High-D", g, &fixed_point, 500)?;
    println!();

    Ok(())
}

/// Run acceleration comparison test
fn run_acceleration_test(
    name: &str,
    g: impl Fn(&DVector<f64>) -> DVector<f64>,
    target: &DVector<f64>,
    max_iter: usize,
) -> anyhow::Result<()> {
    let tol = 1e-8;

    // Simple fixed-point
    let mut x = DVector::zeros(target.len());
    let mut iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        x = g(&x);
        iter += 1;
    }
    let baseline_iter = iter;
    println!("{:<20} | {:>8} iterations (baseline)", "Simple", baseline_iter);

    // Aitken
    let mut aitken = AitkenAcceleration::new();
    let mut x = DVector::zeros(target.len());
    iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let x_new = g(&x);
        if let Some(x_accel) = aitken.update(&x_new) {
            x = x_accel;
        } else {
            x = x_new;
        }
        iter += 1;
    }
    print_result("Aitken", iter, baseline_iter);

    // Epsilon
    let mut epsilon = EpsilonAlgorithm::new(10);
    let mut x = DVector::zeros(target.len());
    iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let x_new = g(&x);
        if let Some(x_accel) = epsilon.update(&x_new) {
            x = x_accel;
        } else {
            x = x_new;
        }
        iter += 1;
    }
    print_result("Epsilon", iter, baseline_iter);

    // Theta
    let mut theta = ThetaAlgorithm::new(10);
    let mut x = DVector::zeros(target.len());
    iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let x_new = g(&x);
        if let Some(x_accel) = theta.update(&x_new) {
            x = x_accel;
        } else {
            x = x_new;
        }
        iter += 1;
    }
    print_result("Theta", iter, baseline_iter);

    // Anderson mixing
    let mut anderson = AndersonMixing::new(5, 0.5);
    let mut x = DVector::zeros(target.len());
    iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let x_new = g(&x);
        x = anderson.update(&x, &x_new);
        iter += 1;
    }
    print_result("Anderson", iter, baseline_iter);

    // Combined
    let mut combined = CombinedAccelerator::new(10);
    let mut x = DVector::zeros(target.len());
    iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let x_new = g(&x);
        if let Some(x_accel) = combined.update(&x_new) {
            x = x_accel;
        } else {
            x = x_new;
        }
        iter += 1;
    }
    print_result("Combined", iter, baseline_iter);

    Ok(())
}

fn print_result(name: &str, iter: usize, baseline: usize) {
    if iter >= baseline {
        println!("{:<20} | {:>8} iterations", name, iter);
    } else {
        let speedup = baseline as f64 / iter as f64;
        println!("{:<20} | {:>8} iterations ({:.1f}x speedup)", name, iter, speedup);
    }
}
