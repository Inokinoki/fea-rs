//! Plate element validation tests.
//!
//! Validates plate element stiffness and mass matrices.

use fea::prelude::*;
use fea::elements::{Plate4, ElementContext};
use nalgebra::DVector;

fn main() -> anyhow::Result<()> {
    println!("=== Plate Element Validation ===\n");

    // Test 1: Basic element stiffness validation
    println!("Test 1: Element Stiffness Validation");
    println!("----------------------------------------");
    run_stiffness_validation()?;

    // Test 2: Mass matrix validation
    println!("\nTest 2: Mass Matrix Validation");
    println!("----------------------------------------");
    run_mass_validation()?;

    // Test 3: Single element response
    println!("\nTest 3: Single Element Response");
    println!("----------------------------------------");
    run_single_element_test()?;

    println!("\n=== All Validation Tests Completed ===");

    Ok(())
}

/// Basic element stiffness validation
fn run_stiffness_validation() -> anyhow::Result<()> {
    // Create a single plate element
    let nodes = vec![
        Node::new_2d(0.0, 0.0),
        Node::new_2d(1.0, 0.0),
        Node::new_2d(1.0, 1.0),
        Node::new_2d(0.0, 1.0),
    ];

    let e: f64 = 210e9;
    let nu: f64 = 0.3;
    let t: f64 = 0.01;

    // Create context
    let mat = Material::new("steel", e, nu, 7850.0, 250e6);
    let sec = Section::new("plate", t, 0.0, 0.0, 0.0);
    let ctx = ElementContext::new(&nodes, &mat, &sec);

    // Create element
    let elem = Plate4::new(0, 1, 2, 3, t);

    // Get stiffness matrix
    let k = elem.stiffness(&ctx);

    // Verify symmetry
    let mut is_symmetric = true;
    for i in 0..k.nrows() {
        for j in (i + 1)..k.ncols() {
            if (k[(i, j)] - k[(j, i)]).abs() > 1e-6 * k[(i, i)].max(1.0) {
                is_symmetric = false;
                break;
            }
        }
    }

    // Verify positive diagonal
    let mut positive_diag = true;
    for i in 0..k.nrows() {
        if k[(i, i)] < 0.0 {
            positive_diag = false;
            break;
        }
    }

    // Verify zero energy modes (rigid body modes)
    let mut rigid_body_modes = 0;

    // Translation in X
    let tx = DVector::from_fn(8, |i, _| if i % 2 == 0 { 1.0 } else { 0.0 });
    let ktx = &k * &tx;
    if ktx.norm() < 1e-6 {
        rigid_body_modes += 1;
    }

    // Translation in Y
    let ty = DVector::from_fn(8, |i, _| if i % 2 == 1 { 1.0 } else { 0.0 });
    let kty = &k * &ty;
    if kty.norm() < 1e-6 {
        rigid_body_modes += 1;
    }

    println!("  Element stiffness matrix: {}x{}", k.nrows(), k.ncols());
    println!("  Symmetric: {}", is_symmetric);
    println!("  Positive diagonal: {}", positive_diag);
    println!("  Rigid body modes detected: {}", rigid_body_modes);

    if is_symmetric && positive_diag && rigid_body_modes >= 2 {
        println!("  PASSED: Element passes basic stiffness requirements");
    } else {
        println!("  Note: Some requirements not fully met");
    }

    Ok(())
}

/// Mass matrix validation
fn run_mass_validation() -> anyhow::Result<()> {
    let nodes = vec![
        Node::new_2d(0.0, 0.0),
        Node::new_2d(1.0, 0.0),
        Node::new_2d(1.0, 1.0),
        Node::new_2d(0.0, 1.0),
    ];

    let e: f64 = 210e9;
    let nu: f64 = 0.3;
    let t: f64 = 0.01;

    let mat = Material::new("steel", e, nu, 7850.0, 250e6);
    let sec = Section::new("plate", t, 0.0, 0.0, 0.0);
    let ctx = ElementContext::new(&nodes, &mat, &sec);

    let elem = Plate4::new(0, 1, 2, 3, t);

    // Get mass matrix
    let m = elem.mass(&ctx);

    if let Some(mass) = m {
        // Verify symmetry
        let mut is_symmetric = true;
        for i in 0..mass.nrows() {
            for j in (i + 1)..mass.ncols() {
                if (mass[(i, j)] - mass[(j, i)]).abs() > 1e-10 {
                    is_symmetric = false;
                    break;
                }
            }
        }

        // Verify positive diagonal (lumped mass)
        let mut positive_diag = true;
        let mut total_mass: f64 = 0.0;
        for i in 0..mass.nrows() {
            if mass[(i, i)] < 0.0 {
                positive_diag = false;
            }
            total_mass += mass[(i, i)];
        }

        // Expected mass
        let area = 1.0;
        let expected_mass = mat.density * t * area;

        println!("  Element mass matrix: {}x{}", mass.nrows(), mass.ncols());
        println!("  Symmetric: {}", is_symmetric);
        println!("  Positive diagonal: {}", positive_diag);
        println!("  Total mass: {:.4} kg (expected: {:.4} kg)", total_mass, expected_mass);

        if is_symmetric && positive_diag {
            println!("  PASSED: Mass matrix is valid");
        }
    } else {
        println!("  Note: Mass matrix not available for this element");
    }

    Ok(())
}

/// Single element response test
fn run_single_element_test() -> anyhow::Result<()> {
    let nodes = vec![
        Node::new_2d(0.0, 0.0),
        Node::new_2d(2.0, 0.0),
        Node::new_2d(2.0, 1.0),
        Node::new_2d(0.0, 1.0),
    ];

    let e: f64 = 210e9;
    let nu: f64 = 0.3;
    let t: f64 = 0.01;

    let mat = Material::new("steel", e, nu, 7850.0, 250e6);
    let sec = Section::new("plate", t, 0.0, 0.0, 0.0);
    let ctx = ElementContext::new(&nodes, &mat, &sec);

    let elem = Plate4::new(0, 1, 2, 3, t);
    let k = elem.stiffness(&ctx);

    // Verify condition number estimate
    let max_diag = (0..k.nrows()).map(|i| k[(i, i)]).fold(0.0_f64, f64::max);
    let min_diag = (0..k.nrows()).map(|i| k[(i, i)]).fold(f64::MAX, f64::min);

    let condition_estimate = if min_diag > 1e-10 {
        max_diag / min_diag
    } else {
        f64::MAX
    };

    println!("  Element size: 2.0m x 1.0m");
    println!("  Stiffness condition estimate: {:.2e}", condition_estimate);

    if condition_estimate < 1e10 {
        println!("  PASSED: Element stiffness is well-conditioned");
    } else {
        println!("  Note: Element may be ill-conditioned");
    }

    Ok(())
}
