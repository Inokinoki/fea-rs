//! Comprehensive FEA validation with advanced solvers and acceleration.
//!
//! This example validates:
//! - All Krylov subspace recycling methods
//! - Domain decomposition preconditioners
//! - Spectral acceleration techniques
//! - Advanced preconditioning strategies
//! - Comparison against analytical solutions

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Comprehensive FEA Validation Suite ===\n");

    // Validation 1: Krylov recycling methods
    validate_krylov_recycling()?;

    // Validation 2: Domain decomposition
    validate_domain_decomposition()?;

    // Validation 3: Spectral acceleration
    validate_spectral_acceleration()?;

    // Validation 4: Complete FEA structural validation
    validate_complete_fea_analysis()?;

    // Validation 5: Performance comparison
    run_performance_comparison()?;

    println!("\n=== All Validations Complete ===");
    Ok(())
}

/// Validate Krylov recycling methods
fn validate_krylov_recycling() -> anyhow::Result<()> {
    println!("1. Krylov Recycling Methods Validation");
    println!("   Testing GCRO-DR, Recycling BiCGSTAB, and Deflated CG...\n");

    let n = 500;
    let k = generate_stiffness_matrix(n, 4.0, -1.0);
    let f = DVector::from_element(n, 1.0);
    let tol = 1e-10;

    // Reference: Direct solve
    let start = Instant::now();
    let x_direct = k.clone().lu().solve(&f).unwrap();
    let direct_time = start.elapsed();
    println!("   Direct solve time: {:.4} ms", direct_time.as_secs_f64() * 1000.0);

    // GCRO-DR
    let gcro = GCRODRSolver::new(20, 5);
    let start = Instant::now();
    let (x_gcro, iters_gcro, res_gcro, conv_gcro) = gcro.solve(&k, &f, None, tol, 200);
    let gcro_time = start.elapsed();

    println!("\n   GCRO-DR:");
    println!("     Iterations: {}", iters_gcro);
    println!("     Final residual: {:.6e}", res_gcro);
    println!("     Converged: {}", conv_gcro);
    println!("     Time: {:.4} ms", gcro_time.as_secs_f64() * 1000.0);

    // Verify solution accuracy
    let error_gcro = (x_gcro - x_direct).norm() / x_direct.norm();
    println!("     Relative error: {:.6e}", error_gcro);

    // Recycling BiCGSTAB
    let mut rbicg = RecyclingBiCGSTAB::new(10);
    let start = Instant::now();
    let (x_rbicg, iters_rbicg, res_rbicg, conv_rbicg) = rbicg.solve(&k, &f, tol, 200);
    let rbicg_time = start.elapsed();

    println!("\n   Recycling BiCGSTAB:");
    println!("     Iterations: {}", iters_rbicg);
    println!("     Final residual: {:.6e}", res_rbicg);
    println!("     Converged: {}", conv_rbicg);
    println!("     Time: {:.4} ms", rbicg_time.as_secs_f64() * 1000.0);

    let error_rbicg = (x_rbicg - x_direct).norm() / x_direct.norm();
    println!("     Relative error: {:.6e}", error_rbicg);

    // Deflated CG
    let defcg = DeflatedCG::new(5);
    let start = Instant::now();
    let (x_defcg, iters_defcg, res_defcg, conv_defcg) = defcg.solve(&k, &f, tol, 200);
    let defcg_time = start.elapsed();

    println!("\n   Deflated CG:");
    println!("     Iterations: {}", iters_defcg);
    println!("     Final residual: {:.6e}", res_defcg);
    println!("     Converged: {}", conv_defcg);
    println!("     Time: {:.4} ms", defcg_time.as_secs_f64() * 1000.0);

    let error_defcg = (x_defcg - x_direct).norm() / x_direct.norm();
    println!("     Relative error: {:.6e}", error_defcg);

    // Validate: All methods should converge to similar solutions
    assert!(error_gcro < 1e-6, "GCRO-DR error too large");
    assert!(error_rbicg < 1e-6, "Recycling BiCGSTAB error too large");
    assert!(error_defcg < 1e-6, "Deflated CG error too large");

    println!("\n   Validation: Krylov recycling methods OK\n");
    Ok(())
}

/// Validate domain decomposition preconditioners
fn validate_domain_decomposition() -> anyhow::Result<()> {
    println!("2. Domain Decomposition Validation");
    println!("   Testing ASM, BDD, and Neumann-Neumann...\n");

    let n = 200;
    let k = generate_stiffness_matrix(n, 4.0, -1.0);
    let f = DVector::from_element(n, 1.0);

    // Additive Schwarz
    let mut asm = AdditiveSchwarz::new(n, 4, 4);
    asm.extract_local_matrices(&k);

    let start = Instant::now();
    let z_asm = asm.apply_restricted(&f.data.as_vec().clone());
    let asm_time = start.elapsed();

    let z_asm_vec = DVector::from_column_slice(&z_asm);
    let asm_reduction = (f.norm()) / (z_asm_vec.norm().max(1e-15));

    println!("   Additive Schwarz:");
    println!("     Subdomains: 4");
    println!("     Overlap: 4");
    println!("     Apply time: {:.4} ms", asm_time.as_secs_f64() * 1000.0);
    println!("     Preconditioned reduction: {:.4}", asm_reduction);

    // BDD
    let mut bdd = BDDPreconditioner::new(4);
    let subdomain_size = n / 4;

    for i in 0..4 {
        let start_idx = i * subdomain_size;
        let end_idx = ((i + 1) * subdomain_size).min(n);
        let size = end_idx - start_idx;

        let dofs: Vec<usize> = (start_idx..end_idx).collect();
        let mut local_k = DMatrix::zeros(size, size);

        for ii in 0..size {
            local_k[(ii, ii)] = 4.0;
            if ii > 0 {
                local_k[(ii, ii - 1)] = -1.0;
            }
            if ii < size - 1 {
                local_k[(ii, ii + 1)] = -1.0;
            }
        }

        bdd.add_subdomain(dofs, local_k);
    }

    bdd.build_coarse_grid();

    let start = Instant::now();
    let z_bdd = bdd.apply(&f.data.as_vec().clone());
    let bdd_time = start.elapsed();

    let z_bdd_vec = DVector::from_column_slice(&z_bdd);
    let bdd_reduction = f.norm() / (z_bdd_vec.norm().max(1e-15));

    println!("\n   BDD Preconditioner:");
    println!("     Subdomains: 4");
    println!("     Apply time: {:.4} ms", bdd_time.as_secs_f64() * 1000.0);
    println!("     Preconditioned reduction: {:.4}", bdd_reduction);

    // Neumann-Neumann
    let mut nn = NeumannNeumann::new(4);
    for _ in 0..4 {
        let size = n / 4;
        let mut local_k = DMatrix::zeros(size, size);
        for ii in 0..size {
            local_k[(ii, ii)] = 4.0;
            if ii > 0 {
                local_k[(ii, ii - 1)] = -1.0;
            }
            if ii < size - 1 {
                local_k[(ii, ii + 1)] = -1.0;
            }
        }
        nn.add_local_matrix(local_k);
    }

    let start = Instant::now();
    let z_nn = nn.apply(&f.data.as_vec().clone());
    let nn_time = start.elapsed();

    let z_nn_vec = DVector::from_column_slice(&z_nn);
    let nn_reduction = f.norm() / (z_nn_vec.norm().max(1e-15));

    println!("\n   Neumann-Neumann:");
    println!("     Subdomains: 4");
    println!("     Apply time: {:.4} ms", nn_time.as_secs_f64() * 1000.0);
    println!("     Preconditioned reduction: {:.4}", nn_reduction);

    // Validate: All should produce finite results
    assert!(z_asm.iter().all(|v| v.is_finite()), "ASM produced non-finite values");
    assert!(z_bdd.iter().all(|v| v.is_finite()), "BDD produced non-finite values");
    assert!(z_nn.iter().all(|v| v.is_finite()), "NN produced non-finite values");

    println!("\n   Validation: Domain decomposition OK\n");
    Ok(())
}

/// Validate spectral acceleration methods
fn validate_spectral_acceleration() -> anyhow::Result<()> {
    println!("3. Spectral Acceleration Validation");
    println!("   Testing Chebyshev, deflation, and rational filters...\n");

    let n = 300;
    let k = generate_stiffness_matrix(n, 10.0, -1.0);
    let f = DVector::from_element(n, 1.0);

    // Eigenvalue estimation
    let (lambda_min, lambda_max) = ChebyshevSemiIterative::estimate_eigenvalues_gershgorin(&k);
    println!("   Estimated eigenvalue range: [{:.4}, {:.4}]", lambda_min, lambda_max);

    // Lanczos estimation
    let (lambda_min_l, lambda_max_l) = ChebyshevSemiIterative::estimate_eigenvalues_lanczos(&k, 30);
    println!("   Lanczos eigenvalue range: [{:.4}, {:.4}]", lambda_min_l, lambda_max_l);

    // Chebyshev semi-iterative
    let mut cheb = ChebyshevSemiIterative::new(lambda_min, lambda_max);
    let mut x = DVector::zeros(n);
    let max_iter = 100;

    let start = Instant::now();
    for _ in 0..max_iter {
        cheb.iterate(&mut x, &k, &f);
        let residual = (f - &k * &x).norm();
        if residual < 1e-10 {
            break;
        }
    }
    let cheb_time = start.elapsed();
    let cheb_residual = (f - &k * &x).norm();

    println!("\n   Chebyshev Semi-Iterative:");
    println!("     Iterations: {}", cheb.iteration);
    println!("     Final residual: {:.6e}", cheb_residual);
    println!("     Time: {:.4} ms", cheb_time.as_secs_f64() * 1000.0);

    // Spectral deflation
    let deflation = SpectralDeflation::compute_deflation_subspace(&k, 5, 100);
    println!("\n   Spectral Deflation:");
    println!("     Deflation vectors: {}", deflation.eigenvectors.ncols());
    println!("     Smallest eigenvalue: {:.6e}", deflation.eigenvalues.min());

    // Apply deflation
    let r = DVector::from_element(n, 1.0);
    let r_deflated = deflation.apply(&r);
    println!("     Deflated residual norm: {:.6e}", r_deflated.norm());

    // Rational Chebyshev filter
    let filter = RationalChebyshevFilter::new(lambda_min_l, (lambda_max_l - lambda_min_l) / 2.0, 4);
    let v = DVector::from_element(n, 1.0);

    let start = Instant::now();
    let filtered = filter.apply(&k, &v);
    let filter_time = start.elapsed();

    println!("\n   Rational Chebyshev Filter:");
    println!("     Filter order: 4");
    println!("     Apply time: {:.4} ms", filter_time.as_secs_f64() * 1000.0);
    println!("     Filtered norm: {:.6e}", filtered.norm());

    // Validate
    assert!(cheb_residual < 1e-6, "Chebyshev did not converge");
    assert!(filtered.iter().all(|v| v.is_finite()), "Filter produced non-finite values");
    assert!(deflation.eigenvectors.ncols() > 0, "Deflation failed");

    println!("\n   Validation: Spectral acceleration OK\n");
    Ok(())
}

/// Validate complete FEA structural analysis
fn validate_complete_fea_analysis() -> anyhow::Result<()> {
    println!("4. Complete FEA Structural Validation");
    println!("   Analyzing 3D truss structure with advanced solvers...\n");

    // Create a 3D truss tower
    let mut model = Model::<Truss2>::new();

    let height = 10.0;
    let width = 2.0;
    let num_levels = 5;
    let dz = height / num_levels as f64;

    // Add nodes
    let mut node_ids = Vec::new();
    for level in 0..=num_levels {
        let z = level as f64 * dz;
        // Four corners at each level
        node_ids.push(model.add_node(Node::new_3d(-width/2.0, -width/2.0, z)));
        node_ids.push(model.add_node(Node::new_3d(width/2.0, -width/2.0, z)));
        node_ids.push(model.add_node(Node::new_3d(width/2.0, width/2.0, z)));
        node_ids.push(model.add_node(Node::new_3d(-width/2.0, width/2.0, z)));
    }

    // Material and section
    model.add_material(STEEL_A36);
    model.add_section(Section::circular("tube", 0.05));

    // Add elements (vertical columns)
    for level in 0..num_levels {
        for corner in 0..4 {
            let n1 = node_ids[level * 4 + corner];
            let n2 = node_ids[(level + 1) * 4 + corner];
            model.add_element(Truss2::new(n1, n2));
        }
    }

    // Horizontal braces
    for level in 0..=num_levels {
        let base = level * 4;
        for i in 0..4 {
            let n1 = node_ids[base + i];
            let n2 = node_ids[base + (i + 1) % 4];
            model.add_element(Truss2::new(n1, n2));
        }
    }

    // Diagonal braces
    for level in 0..num_levels {
        let base = level * 4;
        for i in 0..4 {
            let n1 = node_ids[base + i];
            let n2 = node_ids[(level + 1) * 4 + (i + 1) % 4];
            model.add_element(Truss2::new(n1, n2));
        }
    }

    // Boundary conditions (fixed at base)
    for i in 0..4 {
        let node = node_ids[i];
        model.add_bc(BoundaryCondition::fixed(node, Dof::Ux));
        model.add_bc(BoundaryCondition::fixed(node, Dof::Uy));
        model.add_bc(BoundaryCondition::fixed(node, Dof::Uz));
    }

    // Load at top (horizontal wind load)
    let top_load = 5000.0;
    for i in 0..4 {
        let node = node_ids[num_levels * 4 + i];
        model.add_load(Load::new(node, Dof::Ux, top_load / 4.0));
    }

    // Solve with different methods
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();

    let start = Instant::now();
    let result = analysis.run(&mut model, &config)?;
    let analysis_time = start.elapsed();

    println!("   Model statistics:");
    println!("     Nodes: {}", model.nodes().len());
    println!("     Elements: {}", model.elements().len());
    println!("     DOFs: {}", result.displacements.len());

    println!("\n   Results:");
    println!("     Analysis time: {:.4} ms", analysis_time.as_secs_f64() * 1000.0);

    // Find maximum displacement
    let max_disp = result.displacements.iter()
        .map(|d| d.abs())
        .fold(0.0_f64, f64::max);
    println!("     Maximum displacement: {:.6e} m", max_disp);

    // Find maximum reaction
    let max_reaction = result.reactions.iter()
        .map(|r| r.value.abs())
        .fold(0.0_f64, f64::max);
    println!("     Maximum reaction: {:.2} N", max_reaction);

    // Verify equilibrium
    let total_load = top_load;
    let total_reaction: f64 = result.reactions.iter()
        .filter(|r| r.dof == Dof::Ux)
        .map(|r| r.value.abs())
        .sum();

    let equilibrium_ratio = total_reaction / total_load;
    println!("\n   Equilibrium check:");
    println!("     Total applied load: {:.2} N", total_load);
    println!("     Total reaction: {:.2} N", total_reaction);
    println!("     Ratio: {:.4}", equilibrium_ratio);

    // Validate equilibrium (within 1%)
    assert!((equilibrium_ratio - 1.0).abs() < 0.01, "Equilibrium not satisfied");
    assert!(max_disp > 0.0, "No displacement under load");

    println!("\n   Validation: FEA structural analysis OK\n");
    Ok(())
}

/// Run performance comparison
fn run_performance_comparison() -> anyhow::Result<()> {
    println!("5. Performance Comparison");
    println!("   Comparing all solver configurations...\n");

    let sizes = vec![200, 500, 1000];
    let tol = 1e-10;

    println!("   {:<30} | {:>8} | {:>10} | {:>12} | {:>8}",
             "Method", "Size", "Iterations", "Time (ms)", "Error");
    println!("   {}", "-".repeat(80));

    for n in sizes {
        let k = generate_stiffness_matrix(n, 4.0, -1.0);
        let f = DVector::from_element(n, 1.0);

        // Reference
        let x_ref = k.clone().lu().solve(&f).unwrap();

        // Standard CG
        let config = IterativeConfig {
            max_iterations: 500,
            tolerance: tol,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };
        let cg = CGSolver::with_config(config.clone());
        let start = Instant::now();
        let res = cg.solve(&k, &f, &config)?;
        let cg_time = start.elapsed();
        let cg_error = error_norm(&res, &x_ref);

        println!("   {:<30} | {:>8} | {:>10} | {:>12.4} | {:>8.2e}",
                 "CG + Jacobi", n, res.iterations.unwrap_or(0),
                 cg_time.as_secs_f64() * 1000.0, cg_error);

        // CG + Chebyshev
        let config = IterativeConfig {
            max_iterations: 500,
            tolerance: tol,
            preconditioner: Preconditioner::Chebyshev(3),
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };
        let cg = CGSolver::with_config(config.clone());
        let start = Instant::now();
        let res = cg.solve(&k, &f, &config)?;
        let time = start.elapsed();
        let error = error_norm(&res, &x_ref);

        println!("   {:<30} | {:>8} | {:>10} | {:>12.4} | {:>8.2e}",
                 "CG + Chebyshev(3)", n, res.iterations.unwrap_or(0),
                 time.as_secs_f64() * 1000.0, error);

        // CG + Anderson
        let config = IterativeConfig {
            max_iterations: 500,
            tolerance: tol,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 5,
            krylov_dim: 0,
            deflation_vectors: None,
        };
        let cg = CGSolver::with_config(config.clone());
        let start = Instant::now();
        let res = cg.solve(&k, &f, &config)?;
        let time = start.elapsed();
        let error = error_norm(&res, &x_ref);

        println!("   {:<30} | {:>8} | {:>10} | {:>12.4} | {:>8.2e}",
                 "CG + Anderson(5)", n, res.iterations.unwrap_or(0),
                 time.as_secs_f64() * 1000.0, error);
    }

    println!();
    Ok(())
}

/// Compute relative error norm
fn error_norm(result: &fea::prelude::SolverResult, reference: &DVector<f64>) -> f64 {
    let x = DVector::from_column_slice(&result.solution);
    (x - reference).norm() / reference.norm().max(1e-15)
}

/// Generate a stiffness-like matrix
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
    fn test_comprehensive_validation() {
        validate_krylov_recycling().unwrap();
        validate_domain_decomposition().unwrap();
        validate_spectral_acceleration().unwrap();
        validate_complete_fea_analysis().unwrap();
        run_performance_comparison().unwrap();
    }
}
