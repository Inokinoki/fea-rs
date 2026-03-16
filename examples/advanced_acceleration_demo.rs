//! Advanced acceleration methods demonstration for FEA solvers.
//!
//! This example showcases:
//! - FSAI preconditioning
//! - Spectral deflation
//! - Chebyshev semi-iterative methods
//! - Multilevel additive Schwarz
//! - Elasticity-based preconditioning

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("=== Advanced FEA Acceleration Methods Demonstration ===\n");

    // Test 1: FSAI Preconditioner
    test_fsai_preconditioner()?;

    // Test 2: Spectral Deflation
    test_spectral_deflation()?;

    // Test 3: Chebyshev Semi-Iterative
    test_chebyshev_semi_iterative()?;

    // Test 4: Multilevel Additive Schwarz
    test_additive_schwarz_multilevel()?;

    // Test 5: Elasticity Preconditioner
    test_elasticity_preconditioner()?;

    // Test 6: Combined acceleration in FEA context
    test_combined_acceleration_fea()?;

    println!("\n=== All Tests Completed ===");
    Ok(())
}

/// Test Factorized Sparse Approximate Inverse (FSAI) preconditioner
fn test_fsai_preconditioner() -> anyhow::Result<()> {
    println!("1. FSAI Preconditioner Test");
    println!("   Testing Factorized Sparse Approximate Inverse...\n");

    // Create a stiffness-like matrix
    let n = 100;
    let mut k = DMatrix::zeros(n, n);

    for i in 0..n {
        k[(i, i)] = 2.0;
        if i > 0 {
            k[(i, i - 1)] = -1.0;
        }
        if i < n - 1 {
            k[(i, i + 1)] = -1.0;
        }
    }

    // Create FSAI preconditioner with automatic pattern
    let fsai = FSAIPreconditioner::with_auto_pattern(&k, 2);

    match fsai {
        Some(fsai) => {
            let f = DVector::from_element(n, 1.0);
            let z = fsai.apply(&f);

            println!("   Matrix size: {} x {}", n, n);
            println!("   Preconditioned residual norm: {:.6e}", z.norm());
            println!("   FSAI preconditioner: OK\n");
        }
        None => {
            println!("   FSAI construction failed (matrix may not be SPD)\n");
        }
    }

    Ok(())
}

/// Test Spectral Deflation preconditioner
fn test_spectral_deflation() -> anyhow::Result<()> {
    println!("2. Spectral Deflation Test");
    println!("   Computing deflation subspace...\n");

    let n = 50;
    let mut k = DMatrix::zeros(n, n);

    for i in 0..n {
        k[(i, i)] = 10.0;
        if i > 0 {
            k[(i, i - 1)] = -1.0;
        }
        if i < n - 1 {
            k[(i, i + 1)] = -1.0;
        }
    }

    // Compute deflation subspace with 5 eigenvectors
    let deflation = SpectralDeflation::compute_deflation_subspace(&k, 5, 100);

    println!("   Matrix size: {} x {}", n, n);
    println!("   Number of deflation vectors: {}", deflation.eigenvectors.ncols());
    println!("   Smallest eigenvalue estimate: {:.6e}", deflation.eigenvalues.min());
    println!("   Largest eigenvalue estimate: {:.6e}", deflation.eigenvalues.max());

    // Test deflation application
    let r = DVector::from_element(n, 1.0);
    let r_deflated = deflation.apply(&r);

    println!("   Original residual norm: {:.6e}", r.norm());
    println!("   Deflated residual norm: {:.6e}", r_deflated.norm());
    println!("   Spectral deflation: OK\n");

    Ok(())
}

/// Test Chebyshev Semi-Iterative method
fn test_chebyshev_semi_iterative() -> anyhow::Result<()> {
    println!("3. Chebyshev Semi-Iterative Test");
    println!("   Solving using Chebyshev iteration...\n");

    let n = 100;
    let mut k = DMatrix::zeros(n, n);

    for i in 0..n {
        k[(i, i)] = 4.0;
        if i > 0 {
            k[(i, i - 1)] = -1.0;
        }
        if i < n - 1 {
            k[(i, i + 1)] = -1.0;
        }
    }

    let f = DVector::from_element(n, 1.0);

    // Estimate eigenvalues using Gershgorin
    let (lambda_min, lambda_max) = ChebyshevSemiIterative::estimate_eigenvalues_gershgorin(&k);
    println!("   Estimated eigenvalue range: [{:.4}, {:.4}]", lambda_min, lambda_max);

    // Create Chebyshev accelerator
    let mut cheb = ChebyshevSemiIterative::new(lambda_min, lambda_max);
    let mut x = DVector::zeros(n);

    // Iterate
    let max_iter = 50;
    for iter in 0..max_iter {
        cheb.iterate(&mut x, &k, &f);
        let residual = &f - &k * &x;
        if residual.norm() < 1e-8 {
            println!("   Converged at iteration {}", iter + 1);
            break;
        }
    }

    let final_residual = &f - &k * &x;
    println!("   Final residual norm: {:.6e}", final_residual.norm());
    println!("   Chebyshev semi-iterative: OK\n");

    Ok(())
}

/// Test Multilevel Additive Schwarz
fn test_additive_schwarz_multilevel() -> anyhow::Result<()> {
    println!("4. Multilevel Additive Schwarz Test");
    println!("   Building multilevel hierarchy...\n");

    let n = 128;
    let mut k = DMatrix::zeros(n, n);

    for i in 0..n {
        k[(i, i)] = 2.0;
        if i > 0 {
            k[(i, i - 1)] = -0.5;
        }
        if i < n - 1 {
            k[(i, i + 1)] = -0.5;
        }
    }

    // Create 4-level hierarchy
    let asm = AdditiveSchwarzMultilevel::new(&k, 4);

    println!("   Matrix size: {} x {}", n, n);
    println!("   Number of levels: {}", asm.levels);
    println!("   Coarsest grid size: {} x {}", asm.coarse_matrices.last().unwrap().nrows(), asm.coarse_matrices.last().unwrap().ncols());

    let f = DVector::from_element(n, 1.0);
    let corr = asm.apply_vcycle(&f);

    println!("   V-cycle correction norm: {:.6e}", corr.norm());
    println!("   Multilevel Additive Schwarz: OK\n");

    Ok(())
}

/// Test Elasticity Preconditioner
fn test_elasticity_preconditioner() -> anyhow::Result<()> {
    println!("5. Elasticity Preconditioner Test");
    println!("   Testing physics-based preconditioning...\n");

    // Simulate a 2D elasticity stiffness matrix
    let nodes_1d = 20;
    let n = nodes_1d * nodes_1d * 2; // 2 DOF per node

    let mut k = DMatrix::zeros(n, n);
    let youngs_modulus = 200e9; // Steel

    // Build simplified stiffness
    for i in 0..n {
        k[(i, i)] = youngs_modulus * 1e-9; // Scaled diagonal

        if i > 0 {
            k[(i, i - 1)] = -youngs_modulus * 1e-10;
        }
        if i < n - 1 {
            k[(i, i + 1)] = -youngs_modulus * 1e-10;
        }
    }

    let prec = ElasticityPreconditioner::new(&k, youngs_modulus);

    let f = DVector::from_element(n, 1000.0); // 1000 N load
    let z = prec.apply(&f);

    println!("   Number of DOFs: {}", n);
    println!("   Young's modulus: {:.2e} Pa", youngs_modulus);
    println!("   Preconditioned force norm: {:.6e}", z.norm());
    println!("   Elasticity preconditioner: OK\n");

    Ok(())
}

/// Test combined acceleration in FEA context
fn test_combined_acceleration_fea() -> anyhow::Result<()> {
    println!("6. Combined Acceleration FEA Test");
    println!("   Solving FEA system with combined methods...\n");

    // Create a simple truss structure model
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

    // Add material and section
    model.add_material(STEEL_A36);
    model.add_section(Section::circular("round", 0.01));

    // Add truss elements
    for i in 0..4 {
        model.add_element(Truss2::new(n0 + i, n0 + i + 1));
        model.add_element(Truss2::new(n5 + i, n5 + i + 1));
    }

    for i in 0..5 {
        model.add_element(Truss2::new(n0 + i, n5 + i));
    }

    // Diagonal elements
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

    // Assemble global stiffness matrix (simplified)
    let k_matrix = model.global_stiffness_matrix();
    let f_vector = model.force_vector();

    println!("   Global stiffness matrix: {} x {}", k_matrix.nrows(), k_matrix.ncols());
    println!("   Force vector size: {}", f_vector.len());

    // Solve with different preconditioners
    let tol = 1e-10;
    let max_iter = 1000;

    // CG with Jacobi
    let cg_jacobi = CGSolver::with_config(IterativeConfig {
        max_iterations: max_iter,
        tolerance: tol,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    });

    let result_jacobi = cg_jacobi.solve(&k_matrix, &f_vector, &IterativeConfig::default());

    if let Ok(res) = result_jacobi {
        println!("   CG + Jacobi: {} iterations, residual = {:.6e}",
                 res.iterations.unwrap_or(0),
                 res.residual_norm.unwrap_or(0.0));
    }

    // CG with Chebyshev preconditioner
    let cg_cheb = CGSolver::with_config(IterativeConfig {
        max_iterations: max_iter,
        tolerance: tol,
        preconditioner: Preconditioner::Chebyshev(3),
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    });

    let result_cheb = cg_cheb.solve(&k_matrix, &f_vector, &IterativeConfig::default());

    if let Ok(res) = result_cheb {
        println!("   CG + Chebyshev: {} iterations, residual = {:.6e}",
                 res.iterations.unwrap_or(0),
                 res.residual_norm.unwrap_or(0.0));
    }

    // CG with Anderson acceleration
    let cg_anderson = CGSolver::with_config(IterativeConfig {
        max_iterations: max_iter,
        tolerance: tol,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 5,
        krylov_dim: 0,
        deflation_vectors: None,
    });

    let result_anderson = cg_anderson.solve(&k_matrix, &f_vector, &IterativeConfig::default());

    if let Ok(res) = result_anderson {
        println!("   CG + Anderson: {} iterations, residual = {:.6e}",
                 res.iterations.unwrap_or(0),
                 res.residual_norm.unwrap_or(0.0));
    }

    println!("   Combined acceleration FEA: OK\n");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_advanced_acceleration() {
        test_fsai_preconditioner().unwrap();
        test_spectral_deflation().unwrap();
        test_chebyshev_semi_iterative().unwrap();
        test_additive_schwarz_multilevel().unwrap();
        test_elasticity_preconditioner().unwrap();
        test_combined_acceleration_fea().unwrap();
    }
}
