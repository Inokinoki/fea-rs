//! Composite material analysis example.
//!
//! Demonstrates analysis of laminated composite structures
//! using equivalent orthotropic properties.

use fea::prelude::*;
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    println!("=== Composite Material Analysis ===\n");

    // Analyze a laminated composite plate
    analyze_laminate()?;

    // Analyze fiber-reinforced composite
    analyze_fiber_composite()?;

    println!("\n=== Analysis Complete ===");

    Ok(())
}

/// Analyze a laminated composite plate
fn analyze_laminate() -> anyhow::Result<()> {
    println!("Laminated Composite Plate Analysis");
    println!("----------------------------------\n");

    // Material properties for carbon/epoxy unidirectional ply
    let e1 = 140e9; // Longitudinal modulus (Pa)
    let e2 = 10e9;  // Transverse modulus (Pa)
    let g12 = 5e9;  // Shear modulus (Pa)
    let nu12 = 0.3; // Major Poisson's ratio

    println!("Carbon/Epoxy Ply Properties:");
    println!("  E1 (longitudinal): {:.0} GPa", e1 / 1e9);
    println!("  E2 (transverse):   {:.0} GPa", e2 / 1e9);
    println!("  G12 (shear):       {:.0} GPa", g12 / 1e9);
    println!("  nu12:              {:.2}\n", nu12);

    // Laminate stacking sequences
    let laminates = [
        ("Unidirectional [0]4", vec![0, 0, 0, 0]),
        ("Cross-ply [0/90]s", vec![0, 90, 90, 0]),
        ("Angle-ply [45/-45]s", vec![45, -45, -45, 45]),
        ("Quasi-isotropic [0/45/-45/90]s", vec![0, 45, -45, 90, 90, -45, 45, 0]),
    ];

    println!("Laminate Stacking Sequences:");
    println!("----------------------------");

    for (name, angles) in &laminates {
        println!("\n  {}", name);

        // Compute equivalent laminate properties (simplified CLT)
        let (ex, ey, gxy, nuxy) = compute_laminate_properties(e1, e2, g12, nu12, angles);

        println!("    Equivalent Properties:");
        println!("      Ex: {:.1} GPa", ex / 1e9);
        println!("      Ey: {:.1} GPa", ey / 1e9);
        println!("      Gxy: {:.1} GPa", gxy / 1e9);
        println!("      nuxy: {:.3}", nuxy);

        // Compute anisotropy ratio
        let anisotropy = ex / ey;
        println!("      Anisotropy (Ex/Ey): {:.1}", anisotropy);
    }

    // FEA model of quasi-isotropic plate
    println!("\n\nFEA Model: Quasi-isotropic Plate");
    println!("------------------------------");

    let qi_props = laminates.iter()
        .find(|(name, _)| name.contains("Quasi-isotropic"))
        .map(|(_, angles)| angles)
        .unwrap();

    let (ex, ey, gxy, nuxy) = compute_laminate_properties(e1, e2, g12, nu12, qi_props);

    // Create simple plate model using truss approximation
    let mut model = Model::<Truss2LegacyCompat>::new();

    // Create equivalent orthotropic truss model
    // Use equivalent modulus for quasi-isotropic laminate
    let e_equiv = (ex + ey) / 2.0;
    let t = 0.001; // 1mm total thickness
    let w = 0.01;  // 10mm width
    let a = t * w;

    println!("  Equivalent modulus: {:.1} GPa", e_equiv / 1e9);
    println!("  Cross-section: {:.1} mm x {:.1} mm", t * 1000.0, w * 1000.0);

    // Simple tension test
    let l = 0.1; // 100mm length
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(l, 0.0));

    model.add_element(Truss2LegacyCompat::new(n0, n1, e_equiv, a));

    // Fixed at left
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uz));

    // Apply tensile load
    let p = 1000.0; // 1kN
    model.add_load(Load::new(n1, Dof::Ux, p));

    // Solve
    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    let disp = result.u[model.dof_index(n1, Dof::Ux).unwrap()];
    let strain = disp / l;
    let stress = p / a;
    let e_measured = stress / strain;

    println!("\n  Tension Test Results:");
    println!("    Applied load: {:.1} N", p);
    println!("    Displacement: {:.4} mm", disp * 1000.0);
    println!("    Strain: {:.4}%", strain * 100.0);
    println!("    Stress: {:.1} MPa", stress / 1e6);
    println!("    Measured E: {:.1} GPa", e_measured / 1e9);

    // Validate against theoretical
    let error = (e_measured - e_equiv).abs() / e_equiv * 100.0;
    println!("    Error vs theory: {:.2}%", error);

    Ok(())
}

/// Analyze fiber-reinforced composite using rule of mixtures
fn analyze_fiber_composite() -> anyhow::Result<()> {
    println!("\nFiber-Reinforced Composite Analysis");
    println!("------------------------------------\n");

    // Fiber properties (carbon)
    let e_f = 230e9; // Fiber modulus
    let nu_f = 0.2;  // Fiber Poisson's ratio

    // Matrix properties (epoxy)
    let e_m = 3.5e9; // Matrix modulus
    let nu_m = 0.35; // Matrix Poisson's ratio

    println!("Constituent Properties:");
    println!("  Fiber (Carbon):");
    println!("    Ef = {:.0} GPa, nu_f = {:.2}", e_f / 1e9, nu_f);
    println!("  Matrix (Epoxy):");
    println!("    Em = {:.1} GPa, nu_m = {:.2}\n", e_m / 1e9, nu_m);

    // Volume fractions to analyze
    let vf_list = [0.3, 0.4, 0.5, 0.6, 0.7];

    println!("Rule of Mixtures Predictions:");
    println!("  Vf   | E1 (GPa) | E2 (GPa) | G12 (GPa) | nu12");
    println!("  -----|----------|----------|-----------|------");

    for &vf in &vf_list {
        let vm = 1.0 - vf;

        // Longitudinal modulus (rule of mixtures)
        let e1 = vf * e_f + vm * e_m;

        // Transverse modulus (inverse rule of mixtures)
        let e2 = 1.0 / (vf / e_f + vm / e_m);

        // Shear modulus (inverse rule of mixtures)
        let g_f = e_f / (2.0 * (1.0 + nu_f));
        let g_m = e_m / (2.0 * (1.0 + nu_m));
        let g12 = 1.0 / (vf / g_f + vm / g_m);

        // Major Poisson's ratio (rule of mixtures)
        let nu12 = vf * nu_f + vm * nu_m;

        println!("  {:.1}  | {:8.1} | {:8.1} | {:9.1} | {:.3}",
            vf, e1 / 1e9, e2 / 1e9, g12 / 1e9, nu12);
    }

    // Halpin-Tsai predictions for comparison
    println!("\nHalpin-Tsai Predictions (xi=2 for E2, xi=1 for G12):");

    let vf = 0.6;
    let vm = 1.0 - vf;

    // Halpin-Tsai for E2
    let xi_e2 = 2.0;
    let eta_e2 = (e_f / e_m - 1.0) / (e_f / e_m + xi_e2);
    let e2_ht = e_m * (1.0 + xi_e2 * eta_e2 * vf) / (1.0 - eta_e2 * vf);

    // Halpin-Tsai for G12
    let g_f = e_f / (2.0 * (1.0 + nu_f));
    let g_m = e_m / (2.0 * (1.0 + nu_m));
    let xi_g12 = 1.0;
    let eta_g12 = (g_f / g_m - 1.0) / (g_f / g_m + xi_g12);
    let g12_ht = g_m * (1.0 + xi_g12 * eta_g12 * vf) / (1.0 - eta_g12 * vf);

    println!("  Vf = {:.1}", vf);
    println!("  E2 (Halpin-Tsai): {:.1} GPa", e2_ht / 1e9);
    println!("  G12 (Halpin-Tsai): {:.1} GPa", g12_ht / 1e9);

    Ok(())
}

/// Compute equivalent laminate properties using Classical Lamination Theory
fn compute_laminate_properties(
    e1: f64, e2: f64, g12: f64, nu12: f64, angles: &[i32],
) -> (f64, f64, f64, f64) {
    // Reduced stiffness matrix for unidirectional ply
    let nu21 = nu12 * e2 / e1;
    let q11 = e1 / (1.0 - nu12 * nu21);
    let q12 = nu12 * e2 / (1.0 - nu12 * nu21);
    let q22 = e2 / (1.0 - nu12 * nu21);
    let q66 = g12;

    // Transform and average for each ply angle
    let mut a11 = 0.0;
    let mut a12 = 0.0;
    let mut a22 = 0.0;
    let mut a66 = 0.0;

    for &angle in angles {
        let theta = angle as f64 * std::f64::consts::PI / 180.0;
        let c = theta.cos();
        let s = theta.sin();
        let c2 = c * c;
        let s2 = s * s;
        let c4 = c2 * c2;
        let s4 = s2 * s2;

        // Transformed reduced stiffness (simplified)
        let q11_bar = q11 * c4 + q22 * s4 + 2.0 * (q12 + 2.0 * q66) * c2 * s2;
        let q12_bar = (q11 + q22 - 4.0 * q66) * c2 * s2 + q12 * (c4 + s4);
        let q22_bar = q11 * s4 + q22 * c4 + 2.0 * (q12 + 2.0 * q66) * c2 * s2;
        let q66_bar = (q11 + q22 - 2.0 * q12 - 2.0 * q66) * c2 * s2 + q66 * (c4 + s4);

        a11 += q11_bar;
        a12 += q12_bar;
        a22 += q22_bar;
        a66 += q66_bar;
    }

    let n = angles.len() as f64;
    a11 /= n;
    a12 /= n;
    a22 /= n;
    a66 /= n;

    // Equivalent engineering constants
    let ex = (a11 * a22 - a12 * a12) / a22;
    let ey = (a11 * a22 - a12 * a12) / a11;
    let gxy = a66;
    let nuxy = a12 / a22;

    (ex, ey, gxy, nuxy)
}
