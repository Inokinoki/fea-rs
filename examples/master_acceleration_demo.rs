//! Master Acceleration Methods Demonstration.
//!
//! This example provides a comprehensive demonstration of all
//! acceleration methods available in the FEA library.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Master FEA Acceleration Methods Demonstration ===\n");
    println!("This demonstration showcases all acceleration techniques.\n");

    // Part 1: Classical Acceleration
    demonstrate_classical_acceleration()?;

    // Part 2: Advanced Preconditioning
    demonstrate_advanced_preconditioning()?;

    // Part 3: Spectral Methods
    demonstrate_spectral_methods()?;

    // Part 4: Krylov Subspace Recycling
    demonstrate_krylov_recycling()?;

    // Part 5: Domain Decomposition
    demonstrate_domain_decomposition()?;

    // Part 6: Tensor Product and Multilevel
    demonstrate_tensor_multilevel()?;

    // Part 7: Complete FEA Example
    demonstrate_complete_fea_acceleration()?;

    println!("\n=== Demonstration Complete ===");
    println!("All acceleration methods have been demonstrated successfully.");
    Ok(())
}

/// Part 1: Classical Acceleration Methods
fn demonstrate_classical_acceleration() -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Part 1: Classical Acceleration Methods");
    println!("─".repeat(60));

    let n = 200;
    let k = generate_stiffness_matrix(n, 4.0, -1.0);
    let f = DVector::from_element(n, 1.0);
    let tol = 1e-10;

    println!("\nProblem: Solve Kx = f for {} DOF tridiagonal system", n);
    println!("Tolerance: {:.2e}\n", tol);

    // Baseline: CG without acceleration
    let baseline_cfg = IterativeConfig {
        max_iterations: 500,
        tolerance: tol,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 0,
        krylov_dim: 0,
        deflation_vectors: None,
    };

    let start = Instant::now();
    let result = CGSolver::with_config(baseline_cfg.clone()).solve(&k, &f, &baseline_cfg)?;
    let baseline_time = start.elapsed();
    let baseline_iters = result.iterations.unwrap_or(0);

    println!("{:<25} | {:>10} | {:>12.4} ms", "CG + Jacobi (Base)", baseline_iters, baseline_time.as_secs_f64() * 1000.0);

    // Aitken Acceleration
    let mut aitken = AitkenAcceleration::new();
    let mut x = DVector::zeros(n);
    let mut r = f.clone();
    let mut iters = 0;

    while r.norm() > tol && iters < 500 {
        let x_new = x + r.scale(0.1);
        if let Some(x_acc) = aitken.update(&x_new) {
            x = x_acc;
        } else {
            x = x_new;
        }
        r = f.clone() - k * &x;
        iters += 1;
    }
    println!("{:<25} | {:>10} | Aitken Δ²", "Aitken", iters);

    // Anderson Acceleration
    for depth in [3, 5] {
        let cfg = IterativeConfig {
            max_iterations: 500,
            tolerance: tol,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: depth,
            krylov_dim: 0,
            deflation_vectors: None,
        };
        let start = Instant::now();
        let res = CGSolver::with_config(cfg.clone()).solve(&k, &f, &cfg)?;
        let t = start.elapsed();
        let speedup = baseline_iters as f64 / res.iterations.unwrap_or(1) as f64;
        println!("{:<25} | {:>10} | {:>12.4} ms | {:.2}x speedup", format!("Anderson (depth={})", depth), res.iterations.unwrap_or(0), t.as_secs_f64() * 1000.0, speedup);
    }

    println!();
    Ok(())
}

/// Part 2: Advanced Preconditioning
fn demonstrate_advanced_preconditioning() -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Part 2: Advanced Preconditioning Techniques");
    println!("─".repeat(60));

    let n = 100;
    let k = generate_stiffness_matrix(n, 4.0, -1.0);
    let f = DVector::from_element(n, 1.0);

    // FSAI Preconditioner
    let pattern: Vec<Vec<usize>> = (0..n).map(|i| {
        let mut p = vec![i];
        if i > 0 { p.push(i - 1); }
        if i < n - 1 { p.push(i + 1); }
        p
    }).collect();

    if let Some(fsai) = FSAIPreconditioner::new(&k, pattern) {
        let start = Instant::now();
        let z = fsai.apply(&f);
        let time = start.elapsed();
        println!("FSAI Preconditioner:");
        println!("  Apply time: {:.4} ms", time.as_secs_f64() * 1000.0);
        println!("  Result norm: {:.6e}", z.norm());
    }

    // Elasticity Preconditioner
    let prec = ElasticityPreconditioner::new(&k, 200e9);
    let start = Instant::now();
    let z = prec.apply(&f);
    let time = start.elapsed();
    println!("\nElasticity Preconditioner:");
    println!("  Apply time: {:.4} ms", time.as_secs_f64() * 1000.0);
    println!("  Result norm: {:.6e}", z.norm());

    // Block Recursive Preconditioner
    let block_prec = BlockRecursivePreconditioner::new(&k, 10);
    let start = Instant::now();
    let z = block_prec.apply(&f.as_slice());
    let time = start.elapsed();
    println!("\nBlock Recursive Preconditioner:");
    println!("  Apply time: {:.4} ms", time.as_secs_f64() * 1000.0);
    println!("  Coupling strength: {:.4}", block_prec.coupling_strength());

    // Additive Schwarz Multilevel
    let asm = AdditiveSchwarzMultilevel::new(&k, 4);
    let start = Instant::now();
    let corr = asm.apply_vcycle(&f);
    let time = start.elapsed();
    println!("\nAdditive Schwarz Multilevel (4 levels):");
    println!("  V-cycle time: {:.4} ms", time.as_secs_f64() * 1000.0);
    println!("  Correction norm: {:.6e}", corr.norm());

    println!();
    Ok(())
}

/// Part 3: Spectral Methods
fn demonstrate_spectral_methods() -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Part 3: Spectral Acceleration Methods");
    println!("─".repeat(60));

    let n = 100;
    let k = generate_stiffness_matrix(n, 10.0, -1.0);

    // Eigenvalue estimation
    let (lam_min, lam_max) = ChebyshevSemiIterative::estimate_eigenvalues_gershgorin(&k);
    println!("\nEigenvalue Estimation (Gershgorin):");
    println!("  λ_min = {:.6e}", lam_min);
    println!("  λ_max = {:.6e}", lam_max);
    println!("  Condition number estimate: {:.2}", lam_max / lam_min);

    let (lam_min_l, lam_max_l) = ChebyshevSemiIterative::estimate_eigenvalues_lanczos(&k, 20);
    println!("\nEigenvalue Estimation (Lanczos, 20 iter):");
    println!("  λ_min = {:.6e}", lam_min_l);
    println!("  λ_max = {:.6e}", lam_max_l);

    // Spectral Deflation
    let deflation = SpectralDeflation::compute_deflation_subspace(&k, 5, 100);
    println!("\nSpectral Deflation:");
    println!("  Number of deflation vectors: {}", deflation.eigenvectors.ncols());
    println!("  Smallest Ritz value: {:.6e}", deflation.eigenvalues.min());

    // Chebyshev Semi-Iterative
    let f = DVector::from_element(n, 1.0);
    let mut cheb = ChebyshevSemiIterative::new(lam_min, lam_max);
    let mut x = DVector::zeros(n);

    for _ in 0..50 {
        cheb.iterate(&mut x, &k, &f);
    }
    let residual = (f - k * &x).norm();
    println!("\nChebyshev Semi-Iterative (50 iter):");
    println!("  Final residual: {:.6e}", residual);

    // Rational Chebyshev Filter
    let filter = RationalChebyshevFilter::new(lam_min_l, (lam_max_l - lam_min_l) / 2.0, 4);
    let v = DVector::from_element(n, 1.0);
    let start = Instant::now();
    let filtered = filter.apply(&k, &v);
    let time = start.elapsed();
    println!("\nRational Chebyshev Filter:");
    println!("  Apply time: {:.4} ms", time.as_secs_f64() * 1000.0);
    println!("  Filtered norm: {:.6e}", filtered.norm());

    println!();
    Ok(())
}

/// Part 4: Krylov Subspace Recycling
fn demonstrate_krylov_recycling() -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Part 4: Krylov Subspace Recycling");
    println!("─".repeat(60));

    let n = 100;
    let k = generate_stiffness_matrix(n, 4.0, -1.0);
    let f = DVector::from_element(n, 1.0);

    // GCRO-DR
    let gcro = GCRODRSolver::new(20, 5);
    let start = Instant::now();
    let (x, iters, residual, _conv) = gcro.solve(&k, &f, None, 1e-8, 200);
    let time = start.elapsed();
    println!("\nGCRO-DR (subspace=20, eigs=5):");
    println!("  Iterations: {}", iters);
    println!("  Time: {:.4} ms", time.as_secs_f64() * 1000.0);
    println!("  Final residual: {:.6e}", residual);

    // Recycling BiCGSTAB
    let mut rbicg = RecyclingBiCGSTAB::new(10);
    let start = Instant::now();
    let (x, iters, residual, _conv) = rbicg.solve(&k, &f, 1e-8, 200);
    let time = start.elapsed();
    println!("\nRecycling BiCGSTAB (subspace=10):");
    println!("  Iterations: {}", iters);
    println!("  Time: {:.4} ms", time.as_secs_f64() * 1000.0);
    println!("  Final residual: {:.6e}", residual);

    // Deflated CG
    let dcg = DeflatedCG::new(5);
    let start = Instant::now();
    let (x, iters, residual, _conv) = dcg.solve(&k, &f, 1e-8, 200);
    let time = start.elapsed();
    println!("\nDeflated CG (deflation=5):");
    println!("  Iterations: {}", iters);
    println!("  Time: {:.4} ms", time.as_secs_f64() * 1000.0);
    println!("  Final residual: {:.6e}", residual);

    println!();
    Ok(())
}

/// Part 5: Domain Decomposition
fn demonstrate_domain_decomposition() -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Part 5: Domain Decomposition Methods");
    println!("─".repeat(60));

    let n = 200;
    let k = generate_stiffness_matrix(n, 4.0, -1.0);
    let f = DVector::from_element(n, 1.0);
    let f_vec = f.as_slice().to_vec();

    // Additive Schwarz
    let mut asm = AdditiveSchwarz::new(n, 4, 4);
    asm.extract_local_matrices(&k);

    let start = Instant::now();
    let z = asm.apply_restricted(&f_vec);
    let time = start.elapsed();
    println!("\nAdditive Schwarz (4 subdomains, overlap=4):");
    println!("  Apply time: {:.4} ms", time.as_secs_f64() * 1000.0);
    println!("  Result norm: {:.6e}", DVector::from_column_slice(&z).norm());

    // BDD
    let mut bdd = BDDPreconditioner::new(4);
    let subdomain_size = n / 4;
    for i in 0..4 {
        let start_idx = i * subdomain_size;
        let end_idx = ((i + 1) * subdomain_size).min(n);
        let size = end_idx - start_idx;
        let dofs: Vec<usize> = (start_idx..end_idx).collect();

        let mut local_k = DMatrix::zeros(size, size);
        for j in 0..size {
            local_k[(j, j)] = 4.0;
            if j > 0 { local_k[(j, j - 1)] = -1.0; }
            if j < size - 1 { local_k[(j, j + 1)] = -1.0; }
        }
        bdd.add_subdomain(dofs, local_k);
    }
    bdd.build_coarse_grid();

    let start = Instant::now();
    let z = bdd.apply(&f_vec);
    let time = start.elapsed();
    println!("\nBDD Preconditioner:");
    println!("  Apply time: {:.4} ms", time.as_secs_f64() * 1000.0);

    // Neumann-Neumann
    let mut nn = NeumannNeumann::new(4);
    for _ in 0..4 {
        let size = n / 4;
        let mut local_k = DMatrix::zeros(size, size);
        for j in 0..size {
            local_k[(j, j)] = 4.0;
            if j > 0 { local_k[(j, j - 1)] = -1.0; }
            if j < size - 1 { local_k[(j, j + 1)] = -1.0; }
        }
        nn.add_local_matrix(local_k);
    }

    let start = Instant::now();
    let z = nn.apply(&f_vec);
    let time = start.elapsed();
    println!("\nNeumann-Neumann:");
    println!("  Apply time: {:.4} ms", time.as_secs_f64() * 1000.0);
    println!("  Result norm: {:.6e}", DVector::from_column_slice(&z).norm());

    println!();
    Ok(())
}

/// Part 6: Tensor Product and Multilevel
fn demonstrate_tensor_multilevel() -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Part 6: Tensor Product and Multilevel Methods");
    println!("─".repeat(60));

    // Tensor Product Preconditioner
    let sizes = vec![10, 10];
    let tp = TensorProductPreconditioner::from_laplacian_sizes(&sizes);
    let n = sizes[0] * sizes[1];
    let r = DVector::from_element(n, 1.0);

    let start = Instant::now();
    let z = tp.apply(&r, &sizes);
    let time = start.elapsed();
    println!("\nTensor Product Preconditioner (10x10 grid):");
    println!("  Apply time: {:.4} ms", time.as_secs_f64() * 1000.0);
    println!("  Result norm: {:.6e}", z.norm());

    // Hierarchical Basis
    let hb = HierarchicalBasis::new_1d(4, 16);
    println!("\nHierarchical Basis (4 levels, 16 fine nodes):");
    println!("  Number of levels: {}", hb.num_levels());

    let v = DVector::from_fn(16, |i, _| (i as f64).sin());
    let v_coarse = hb.restrict(&v, 0);
    let v_fine = hb.prolongate(&v_coarse, 0);
    let surplus = hb.compute_surplus(&v, 1);

    println!("  Original norm: {:.6e}", v.norm());
    println!("  Coarse norm: {:.6e}", v_coarse.norm());
    println!("  Surplus norm: {:.6e}", surplus.norm());

    // Wavelet Preconditioner
    let wp = WaveletPreconditioner::haar(16);
    let start = Instant::now();
    let z = wp.apply(&v);
    let time = start.elapsed();
    println!("\nHaar Wavelet Preconditioner:");
    println!("  Apply time: {:.4} ms", time.as_secs_f64() * 1000.0);
    println!("  Result norm: {:.6e}", z.norm());

    // Multilevel Accelerator
    let mla = MultilevelAccelerator::new(3);
    let mut a_matrices = Vec::new();
    let mut size = 16;
    for _ in 0..3 {
        let mut a = DMatrix::zeros(size, size);
        for i in 0..size {
            a[(i, i)] = 2.0;
            if i > 0 { a[(i, i - 1)] = -1.0; }
            if i < size - 1 { a[(i, i + 1)] = -1.0; }
        }
        a_matrices.push(a);
        size = (size + 1) / 2;
    }

    let start = Instant::now();
    let correction = mla.vcycle(0, &v, &a_matrices, &hb);
    let time = start.elapsed();
    println!("\nMultilevel V-Cycle (3 levels):");
    println!("  Time: {:.4} ms", time.as_secs_f64() * 1000.0);
    println!("  Correction norm: {:.6e}", correction.norm());

    println!();
    Ok(())
}

/// Part 7: Complete FEA Example
fn demonstrate_complete_fea_acceleration() -> anyhow::Result<()> {
    println!("─".repeat(60));
    println!("Part 7: Complete FEA Structural Analysis");
    println!("─".repeat(60));

    // Create a 3D truss tower
    let mut model = Model::<Truss2>::new();

    let height = 10.0;
    let width = 2.0;
    let num_levels = 5;
    let dz = height / num_levels as f64;

    let mut node_ids = Vec::new();
    for level in 0..=num_levels {
        let z = level as f64 * dz;
        node_ids.push(model.add_node(Node::new_3d(-width/2.0, -width/2.0, z)));
        node_ids.push(model.add_node(Node::new_3d(width/2.0, -width/2.0, z)));
        node_ids.push(model.add_node(Node::new_3d(width/2.0, width/2.0, z)));
        node_ids.push(model.add_node(Node::new_3d(-width/2.0, width/2.0, z)));
    }

    model.add_material(STEEL_A36);
    model.add_section(Section::circular("tube", 0.05));

    // Vertical columns
    for level in 0..num_levels {
        for corner in 0..4 {
            let n1 = node_ids[level * 4 + corner];
            let n2 = node_ids[(level + 1) * 4 + corner];
            model.add_element(Truss2::new(n1, n2));
        }
    }

    // Horizontal and diagonal braces
    for level in 0..=num_levels {
        let base = level * 4;
        for i in 0..4 {
            let n1 = node_ids[base + i];
            let n2 = node_ids[base + (i + 1) % 4];
            model.add_element(Truss2::new(n1, n2));
        }
    }

    // Boundary conditions
    for i in 0..4 {
        model.add_bc(BoundaryCondition::fixed(node_ids[i], Dof::Ux));
        model.add_bc(BoundaryCondition::fixed(node_ids[i], Dof::Uy));
        model.add_bc(BoundaryCondition::fixed(node_ids[i], Dof::Uz));
    }

    // Load at top
    for i in 0..4 {
        model.add_load(Load::new(node_ids[num_levels * 4 + i], Dof::Ux, 1000.0));
    }

    // Solve with accelerated methods
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();

    let start = Instant::now();
    let result = analysis.run(&mut model, &config)?;
    let time = start.elapsed();

    println!("\nTower Model Statistics:");
    println!("  Nodes: {}", model.nodes().len());
    println!("  Elements: {}", model.elements().len());
    println!("  DOFs: {}", result.displacements.len());

    println!("\nSolution with Accelerated Solvers:");
    println!("  Analysis time: {:.4} ms", time.as_secs_f64() * 1000.0);

    let max_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
    println!("  Maximum displacement: {:.6e} m", max_disp);

    // Equilibrium check
    let total_load = 4000.0;
    let total_reaction: f64 = result.reactions.iter()
        .filter(|r| r.dof == Dof::Ux)
        .map(|r| r.value.abs())
        .sum();

    println!("\nEquilibrium Check:");
    println!("  Applied load: {:.2} N", total_load);
    println!("  Reaction force: {:.2} N", total_reaction);
    println!("  Ratio: {:.4}", total_reaction / total_load);

    println!("\nAnalysis completed successfully!");
    Ok(())
}

/// Helper function to generate a stiffness-like matrix
fn generate_stiffness_matrix(n: usize, diag: f64, off_diag: f64) -> DMatrix<f64> {
    let mut k = DMatrix::zeros(n, n);
    for i in 0..n {
        k[(i, i)] = diag;
        if i > 0 { k[(i, i - 1)] = off_diag; }
        if i < n - 1 { k[(i, i + 1)] = off_diag; }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_master_demonstration() {
        demonstrate_classical_acceleration().unwrap();
        demonstrate_advanced_preconditioning().unwrap();
        demonstrate_spectral_methods().unwrap();
        demonstrate_krylov_recycling().unwrap();
        demonstrate_domain_decomposition().unwrap();
        demonstrate_tensor_multilevel().unwrap();
        demonstrate_complete_fea_acceleration().unwrap();
    }
}
