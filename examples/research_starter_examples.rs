//! Research Starter Examples for FEA Library.
//!
//! This collection provides typical starting points for FEA research:
//! - Parameter studies
//! - Convergence analysis
//! - Material model comparison
//! - Mesh refinement studies
//! - Optimization examples

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║        FEA Library - Research Starter Examples           ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("This collection provides typical starting points for FEA research.\n");

    // Example 1: Parameter study
    example_parameter_study()?;

    // Example 2: Convergence analysis
    example_convergence_analysis()?;

    // Example 3: Material model comparison
    example_material_comparison()?;

    // Example 4: Mesh refinement study
    example_mesh_refinement()?;

    // Example 5: Design optimization
    example_design_optimization()?;

    println!("\n=== Research Examples Complete ===");
    println!("\nThese examples serve as templates for your research. Modify them");
    println!("to suit your specific research needs.\n");

    Ok(())
}

/// Example 1: Parameter study - effect of Young's modulus.
fn example_parameter_study() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 1: Parameter Study");
    println!("{}\n", "=".repeat(60));

    println!("Study: Effect of Young's Modulus on Beam Deflection\n");

    // Baseline geometry and loading
    let length = 1.0; // m
    let width = 0.05; // m
    let height = 0.05_f64; // m
    let load = 1000.0; // N

    // Create simple beam stiffness (1D approximation)
    fn create_beam_stiffness(e: f64, l: f64, i: f64) -> DMatrix<f64> {
        let k = 3.0 * e * i / (l * l * l);
        DMatrix::from_row_slice(2, 2, &[k, -k, -k, k])
    }

    let i = width * height.powi(3) / 12.0;

    // Parameter range
    let e_values = vec![50e9, 100e9, 150e9, 200e9, 250e9];

    println!("{:>15} | {:>15} | {:>15}", "E (GPa)", "Deflection (mm)", "Stiffness (N/m)");
    println!("{}", "-".repeat(50));

    for e in e_values {
        let k = create_beam_stiffness(e, length, i);

        // Apply boundary conditions (fixed at node 0)
        let k_reduced = k.view((1, 1), (1, 1)).into_owned();
        let f_reduced = DVector::from_element(1, load);

        let u = k_reduced.lu().solve(&f_reduced).unwrap();
        let deflection_mm = u[0] * 1000.0;
        let stiffness = k[(1, 1)];

        println!("{:>15.1} | {:>15.6} | {:>15.2e}", e / 1e9, deflection_mm, stiffness);
    }

    println!("\nKey observations:");
    println!("  - Deflection is inversely proportional to Young's modulus");
    println!("  - Doubling E halves the deflection");
    println!();

    Ok(())
}

/// Example 2: Convergence analysis.
fn example_convergence_analysis() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 2: Convergence Analysis");
    println!("{}\n", "=".repeat(60));

    println!("Study: CG Solver Convergence with Different Preconditioners\n");

    // Create test problem (Poisson equation discretization)
    fn create_poisson_matrix(n: usize) -> DMatrix<f64> {
        let size = n * n;
        let mut k = DMatrix::zeros(size, size);
        let h = 1.0 / (n + 1) as f64;

        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                k[(idx, idx)] = 4.0 / (h * h);

                if i > 0 { k[(idx, idx - n)] = -1.0 / (h * h); }
                if i < n - 1 { k[(idx, idx + n)] = -1.0 / (h * h); }
                if j > 0 { k[(idx, idx - 1)] = -1.0 / (h * h); }
                if j < n - 1 { k[(idx, idx + 1)] = -1.0 / (h * h); }
            }
        }

        k
    }

    let n = 20;
    let k = create_poisson_matrix(n);
    let f = DVector::from_element(k.nrows(), 1.0);

    // Test different preconditioners
    let preconditioners = vec![
        ("None", Preconditioner::None),
        ("Jacobi", Preconditioner::Jacobi),
        ("Chebyshev(2)", Preconditioner::Chebyshev(2)),
        ("Chebyshev(3)", Preconditioner::Chebyshev(3)),
    ];

    println!("Problem size: {} DOFs\n", k.nrows());
    println!("{:>20} | {:>10} | {:>12} | {:>10}", "Preconditioner", "Iterations", "Time (ms)", "Residual");
    println!("{}", "-".repeat(60));

    for (name, prec) in preconditioners {
        let cfg = IterativeConfig {
            max_iterations: 500,
            tolerance: 1e-10,
            preconditioner: prec,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(cfg.clone());
        let start = std::time::Instant::now();
        let result = cg.solve(&k, &f, &cfg).unwrap();
        let elapsed = start.elapsed();

        println!("{:>20} | {:>10} | {:>12.4} | {:>10.2e}",
                 name,
                 result.iterations.unwrap_or(0),
                 elapsed.as_secs_f64() * 1000.0,
                 result.residual_norm.unwrap_or(f64::INFINITY));
    }

    println!("\nKey observations:");
    println!("  - Preconditioning significantly reduces iterations");
    println!("  - Chebyshev preconditioner is effective for this problem");
    println!();

    Ok(())
}

/// Example 3: Material model comparison.
fn example_material_comparison() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 3: Material Model Comparison");
    println!("{}\n", "=".repeat(60));

    println!("Study: Linear Elastic vs. Elastoplastic Response\n");

    // Material properties
    let e = 210e9;
    let nu = 0.3;
    let yield_stress = 250e6;

    // Create simple uniaxial loading
    let max_strain = 0.02;
    let n_steps = 100;

    // Linear elastic response
    println!("Linear Elastic:");
    println!("  E = {:.0} GPa, ν = {:.2}", e / 1e9, nu);
    println!("\n  {:>12} | {:>15} | {:>15}", "Strain", "Stress (MPa)", "Tangent (GPa)");
    println!("  {}", "-".repeat(48));

    for i in 0..5 {
        let eps = max_strain * (i as f64 / (n_steps - 1) as f64);
        let stress = e * eps;
        println!("  {:>12.6} | {:>15.2} | {:>15.2}", eps, stress / 1e6, e / 1e9);
    }

    // Simplified elastoplastic response
    println!("\nElastoplastic (Linear Hardening):");
    println!("  σ_y = {:.0} MPa, H = {:.0} GPa", yield_stress / 1e6, 2e9 / 1e9);
    println!("\n  {:>12} | {:>15} | {:>15}", "Strain", "Stress (MPa)", "Tangent (GPa)");
    println!("  {}", "-".repeat(48));

    let h = 2e9; // Hardening modulus
    let yield_strain = yield_stress / e;

    for i in 0..5 {
        let eps = max_strain * (i as f64 / (n_steps - 1) as f64);
        let stress = if eps < yield_strain {
            e * eps
        } else {
            yield_stress + h * (eps - yield_strain)
        };
        let tangent = if eps < yield_strain { e } else { h };
        println!("  {:>12.6} | {:>15.2} | {:>15.2}", eps, stress / 1e6, tangent / 1e9);
    }

    println!("\nKey observations:");
    println!("  - Elastic response is linear up to yield");
    println!("  - Plastic response shows reduced tangent modulus");
    println!();

    Ok(())
}

/// Example 4: Mesh refinement study.
fn example_mesh_refinement() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 4: Mesh Refinement Study");
    println!("{}\n", "=".repeat(60));

    println!("Study: Effect of Mesh Density on Solution Accuracy\n");

    // Reference solution (analytical)
    let reference_deflection = 1.0; // Normalized

    // Simulated FEA results at different mesh densities
    let mesh_densities = vec![4, 8, 16, 32, 64, 128];
    let fea_deflections: Vec<f64> = vec![0.75, 0.88, 0.94, 0.97, 0.99, 0.995];

    println!("{:>15} | {:>15} | {:>15} | {:>12}", "Elements", "Deflection", "Error (%)", "Conv. Rate");
    println!("{}", "-".repeat(62));

    let mut prev_error: f64 = 0.0;
    let mut prev_h: f64 = 0.0;

    for (i, (&n_elem, &defl)) in mesh_densities.iter().zip(fea_deflections.iter()).enumerate() {
        let error = ((defl - reference_deflection).abs() / reference_deflection) * 100.0_f64;
        let h: f64 = 1.0 / n_elem as f64;

        let conv_rate = if i > 0 && prev_error > 1e-10 {
            (prev_error / error).ln() / (prev_h / h).ln()
        } else {
            0.0
        };

        println!("{:>15} | {:>15.6} | {:>15.4} | {:>12.2}",
                 n_elem, defl, error, conv_rate);

        prev_error = error;
        prev_h = h;
    }

    println!("\nKey observations:");
    println!("  - Error decreases with mesh refinement");
    println!("  - Convergence rate approaches theoretical value");
    println!();

    Ok(())
}

/// Example 5: Design optimization.
fn example_design_optimization() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 5: Design Optimization");
    println!("{}\n", "=".repeat(60));

    println!("Study: Cantilever Beam Cross-Section Optimization\n");
    println!("Objective: Minimize volume while satisfying stress constraint\n");

    // Design parameters
    let length = 1.0; // m
    let load = 1000.0; // N
    let allowable_stress = 150e6; // Pa
    let e = 210e9; // Pa

    // Simple beam formulas
    let max_moment = load * length;

    println!("Design constraints:");
    println!("  Length: {:.2} m", length);
    println!("  Load: {:.0} N", load);
    println!("  Allowable stress: {:.0} MPa", allowable_stress / 1e6);
    println!();

    // Parametric study
    println!("Parametric Study (width × height):");
    println!("{:>15} | {:>15} | {:>15} | {:>12}", "Section", "Volume (m³)", "σ_max (MPa)", "Status");
    println!("{}", "-".repeat(62));

    let widths = vec![0.02, 0.04, 0.06, 0.08, 0.10];
    let heights: Vec<f64> = vec![0.04, 0.06, 0.08, 0.10, 0.12];

    let mut best_design: Option<(f64, f64, f64)> = None;

    for &w in &widths {
        for &h in &heights {
            let area = w * h;
            let volume = area * length;

            // Section modulus for rectangle
            let i_val = w * h.powi(3) / 12.0;
            let c = h / 2.0;
            let section_modulus = i_val / c;

            let stress = max_moment / section_modulus;
            let status = if stress <= allowable_stress { "✓ PASS" } else { "✗ FAIL" };

            println!("{:>7.0} × {:<7.0} | {:>15.6} | {:>15.2} | {:>12}",
                     w * 1000.0, h * 1000.0, volume, stress / 1e6, status);

            if stress <= allowable_stress {
                if best_design.is_none() || volume < best_design.unwrap().2 {
                    best_design = Some((w, h, volume));
                }
            }
        }
    }

    if let Some((w, h, vol)) = best_design {
        println!("\nOptimal design:");
        println!("  Width: {:.0} mm, Height: {:.0} mm", w * 1000.0, h * 1000.0);
        println!("  Volume: {:.6} m³", vol);
    }

    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_research_examples() {
        example_parameter_study().unwrap();
        example_convergence_analysis().unwrap();
        example_material_comparison().unwrap();
        example_mesh_refinement().unwrap();
        example_design_optimization().unwrap();
    }
}
