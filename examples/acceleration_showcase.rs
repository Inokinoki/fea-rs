//! Comprehensive acceleration methods showcase.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("=== Acceleration Methods Showcase ===\n");

    // Test fixed-point acceleration
    test_fixed_point_acceleration()?;

    // Test vector extrapolation
    test_vector_extrapolation()?;

    // Test gradient acceleration
    test_gradient_acceleration()?;

    println!("\n=== Showcase Complete ===");
    Ok(())
}

/// Test fixed-point iteration acceleration
fn test_fixed_point_acceleration() -> anyhow::Result<()> {
    println!("1. Fixed-Point Acceleration");
    println!("   Problem: x = 0.9x + 0.1 (converges to 1.0)\n");

    let target = DVector::from_element(100, 1.0);
    let g = |x: &DVector<f64>| x.scale(0.9) + DVector::from_element(100, 0.1);

    run_comparison("Fixed-Point", g, &target, 200)?;
    println!();

    Ok(())
}

/// Test vector extrapolation methods
fn test_vector_extrapolation() -> anyhow::Result<()> {
    println!("2. Vector Extrapolation");
    println!("   Problem: x = Ax + b (linear system)\n");

    let n = 50;
    let mut a = DMatrix::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 2.0;
        if i > 0 { a[(i, i - 1)] = -0.3; }
        if i < n - 1 { a[(i, i + 1)] = -0.3; }
    }
    let b = DVector::from_element(n, 1.0);

    // Solve for target
    let target = a.clone().lu().solve(&b).unwrap();

    // Fixed-point iteration: x = x - 0.5*(Ax - b)
    let g = |x: &DVector<f64>| x - (&a * x - &b).scale(0.5);

    run_comparison("Vector Extrapolation", g, &target, 100)?;
    println!();

    Ok(())
}

/// Test gradient-based acceleration
fn test_gradient_acceleration() -> anyhow::Result<()> {
    println!("3. Gradient-Based Acceleration");
    println!("   Problem: minimize f(x) = 0.5*x^T*A*x - b^T*x\n");

    let n = 50;
    let mut a = DMatrix::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 2.0;
        if i > 0 { a[(i, i - 1)] = -0.5; }
        if i < n - 1 { a[(i, i + 1)] = -0.5; }
    }
    let b = DVector::from_element(n, 1.0);

    // Target solution
    let target = a.clone().lu().solve(&b).unwrap();

    // Gradient: grad f = Ax - b
    let grad = |x: &DVector<f64>| &a * x - &b;

    run_gradient_comparison("Gradient", grad, &target, 100)?;
    println!();

    Ok(())
}

/// Run comparison of fixed-point acceleration methods
fn run_comparison<F>(
    name: &str,
    g: F,
    target: &DVector<f64>,
    max_iter: usize,
) -> anyhow::Result<()>
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let tol = 1e-8;

    // Simple
    let (iter, _) = run_simple(&g, target, max_iter, tol);
    println!("{:<20} | {:>8} iterations", "Simple", iter);

    // Aitken
    let (iter, speedup) = run_aitken(&g, target, max_iter, tol, iter);
    println!("{:<20} | {:>8} iterations ({:.1f}x)", "Aitken", iter, speedup);

    // Epsilon
    let (iter, speedup) = run_epsilon(&g, target, max_iter, tol, iter);
    println!("{:<20} | {:>8} iterations ({:.1f}x)", "Epsilon", iter, speedup);

    // Theta
    let (iter, speedup) = run_theta(&g, target, max_iter, tol, iter);
    println!("{:<20} | {:>8} iterations ({:.1f}x)", "Theta", iter, speedup);

    // Anderson
    let (iter, speedup) = run_anderson(&g, target, max_iter, tol, iter);
    println!("{:<20} | {:>8} iterations ({:.1f}x)", "Anderson", iter, speedup);

    // Combined
    let (iter, speedup) = run_combined(&g, target, max_iter, tol, iter);
    println!("{:<20} | {:>8} iterations ({:.1f}x)", "Combined", iter, speedup);

    Ok(())
}

/// Run gradient-based acceleration comparison
fn run_gradient_comparison<F>(
    name: &str,
    grad: F,
    target: &DVector<f64>,
    max_iter: usize,
) -> anyhow::Result<()>
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let tol = 1e-8;

    // Steepest descent
    let (iter, _) = run_steepest_descent(&grad, target, max_iter, tol);
    println!("{:<20} | {:>8} iterations", "Steepest Descent", iter);

    // BB
    let (iter, speedup) = run_bb(&grad, target, max_iter, tol, iter);
    println!("{:<20} | {:>8} iterations ({:.1f}x)", "BB", iter, speedup);

    // BB2
    let (iter, speedup) = run_bb2(&grad, target, max_iter, tol, iter);
    println!("{:<20} | {:>8} iterations ({:.1f}x)", "BB2", iter, speedup);

    // Cyclic BB
    let (iter, speedup) = run_cyclic_bb(&grad, target, max_iter, tol, iter);
    println!("{:<20} | {:>8} iterations ({:.1f}x)", "Cyclic BB", iter, speedup);

    // SQUARED
    let (iter, speedup) = run_squared(&grad, target, max_iter, tol, iter);
    println!("{:<20} | {:>8} iterations ({:.1f}x)", "SQUARED", iter, speedup);

    Ok(())
}

fn run_simple<F>(g: &F, target: &DVector<f64>, max_iter: usize, tol: f64) -> (usize, f64)
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut x = DVector::zeros(target.len());
    let mut iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        x = g(&x);
        iter += 1;
    }
    (iter, 1.0)
}

fn run_aitken<F>(g: &F, target: &DVector<f64>, max_iter: usize, tol: f64, baseline: usize) -> (usize, f64)
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut aitken = AitkenAcceleration::new();
    let mut x = DVector::zeros(target.len());
    let mut iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let x_new = g(&x);
        if let Some(x_accel) = aitken.update(&x_new) {
            x = x_accel;
        } else {
            x = x_new;
        }
        iter += 1;
    }
    (iter, baseline as f64 / iter as f64)
}

fn run_epsilon<F>(g: &F, target: &DVector<f64>, max_iter: usize, tol: f64, baseline: usize) -> (usize, f64)
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut epsilon = EpsilonAlgorithm::new(10);
    let mut x = DVector::zeros(target.len());
    let mut iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let x_new = g(&x);
        if let Some(x_accel) = epsilon.update(&x_new) {
            x = x_accel;
        } else {
            x = x_new;
        }
        iter += 1;
    }
    (iter, baseline as f64 / iter as f64)
}

fn run_theta<F>(g: &F, target: &DVector<f64>, max_iter: usize, tol: f64, baseline: usize) -> (usize, f64)
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut theta = ThetaAlgorithm::new(10);
    let mut x = DVector::zeros(target.len());
    let mut iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let x_new = g(&x);
        if let Some(x_accel) = theta.update(&x_new) {
            x = x_accel;
        } else {
            x = x_new;
        }
        iter += 1;
    }
    (iter, baseline as f64 / iter as f64)
}

fn run_anderson<F>(g: &F, target: &DVector<f64>, max_iter: usize, tol: f64, baseline: usize) -> (usize, f64)
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut anderson = AndersonMixing::new(5, 0.5);
    let mut x = DVector::zeros(target.len());
    let mut iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let x_new = g(&x);
        x = anderson.update(&x, &x_new);
        iter += 1;
    }
    (iter, baseline as f64 / iter as f64)
}

fn run_combined<F>(g: &F, target: &DVector<f64>, max_iter: usize, tol: f64, baseline: usize) -> (usize, f64)
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut combined = CombinedAccelerator::new(10);
    let mut x = DVector::zeros(target.len());
    let mut iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let x_new = g(&x);
        if let Some(x_accel) = combined.update(&x_new) {
            x = x_accel;
        } else {
            x = x_new;
        }
        iter += 1;
    }
    (iter, baseline as f64 / iter as f64)
}

fn run_steepest_descent<F>(grad: &F, target: &DVector<f64>, max_iter: usize, tol: f64) -> (usize, f64)
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut x = DVector::zeros(target.len());
    let mut iter = 0;
    let mut step_size = 0.1;
    while (x - target).norm() > tol && iter < max_iter {
        let g = grad(&x);
        x = x - g.scale(step_size);
        iter += 1;
    }
    (iter, 1.0)
}

fn run_bb<F>(grad: &F, target: &DVector<f64>, max_iter: usize, tol: f64, baseline: usize) -> (usize, f64)
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut bb = BarzilaiBorweinAcceleration::new();
    let mut x = DVector::zeros(target.len());
    let mut x_prev = x.clone();
    let mut iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let g = grad(&x);
        let g_prev = grad(&x_prev);
        let s = &x - &x_prev;
        let y = &g - &g_prev;
        let sy = s.dot(&y);
        let yy = y.dot(&y);
        if sy.abs() > 1e-15 && yy > 1e-15 {
            bb.compute_step(&s, &y);
        }
        x_prev = x.clone();
        x = x - g.scale(bb.optimal_step);
        iter += 1;
    }
    (iter, baseline as f64 / iter as f64)
}

fn run_bb2<F>(grad: &F, target: &DVector<f64>, max_iter: usize, tol: f64, baseline: usize) -> (usize, f64)
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut bb2 = BarzilaiBorwein2::new();
    let mut x = DVector::zeros(target.len());
    let mut x_prev = x.clone();
    let mut iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let g = grad(&x);
        let step = bb2.compute_step(&x, &g);
        x_prev = x.clone();
        x = x - g.scale(step);
        iter += 1;
    }
    (iter, baseline as f64 / iter as f64)
}

fn run_cyclic_bb<F>(grad: &F, target: &DVector<f64>, max_iter: usize, tol: f64, baseline: usize) -> (usize, f64)
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut cbb = CyclicBB::new(5);
    let mut x = DVector::zeros(target.len());
    let mut x_prev = x.clone();
    let mut iter = 0;
    while (x - target).norm() > tol && iter < max_iter {
        let g = grad(&x);
        let alpha = cbb.update_step(&x, &g);
        x_prev = x.clone();
        x = x - g.scale(alpha);
        iter += 1;
    }
    (iter, baseline as f64 / iter as f64)
}

fn run_squared<F>(grad: &F, target: &DVector<f64>, max_iter: usize, tol: f64, baseline: usize) -> (usize, f64)
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut squared = SQUARED::new(0.5);
    let mut x = DVector::zeros(target.len());
    let mut iter = 0;
    while x.clone().norm() < 1e10 && iter < max_iter {
        let g = grad(&x);
        x = squared.update(&x, &g);
        if (x - target).norm() < tol {
            break;
        }
        iter += 1;
    }
    (iter, baseline as f64 / iter as f64)
}
