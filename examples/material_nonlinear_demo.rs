//! Material nonlinear analysis example.
//!
//! This example demonstrates:
//! - Elastic-plastic analysis
//! - Hyperelastic material behavior
//! - Damage mechanics
//! - Stress-strain curve generation
//! - Load-displacement response

use fea::materials::nonlinear::{
    LinearElastic, VonMisesPlasticity, JohnsonCook,
    NeoHookean, DamageModel, StressState,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║        Material Nonlinear Analysis Showcase               ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Linear elastic material demonstration
    demo_linear_elastic()?;

    // Elastic-plastic analysis
    demo_elastic_plastic()?;

    // Johnson-Cook viscoplasticity
    demo_johnson_cook()?;

    // Hyperelastic material (Neo-Hookean)
    demo_hyperelastic()?;

    // Damage mechanics
    demo_damage_mechanics()?;

    // Stress-strain curve generation
    generate_stress_strain_curve()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║          Material Analysis Complete                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demonstrates linear elastic material behavior.
fn demo_linear_elastic() -> anyhow::Result<()> {
    println!("\n┌─ Linear Elastic Material ──────────────────────────────┐");

    let steel = LinearElastic::new(210e9, 0.3, 7850.0);

    println!("│ Material: Steel");
    println!("│ Young's modulus: E = {:.0} GPa", steel.e / 1e9);
    println!("│ Poisson's ratio: ν = {:.2}", steel.nu);
    println!("│ Density: ρ = {:.0} kg/m³", steel.rho);
    println!("│ Shear modulus: G = {:.1} GPa", steel.shear_modulus() / 1e9);
    println!("│ Bulk modulus: K = {:.1} GPa", steel.bulk_modulus() / 1e9);
    println!("│ Wave speed: c = {:.0} m/s", steel.wave_speed());

    // Uniaxial tension
    println!("│\n│ Uniaxial Tension Test:");
    let strains = [0.0001, 0.0005, 0.001, 0.002, 0.005];

    println!("│ {:>12} │ {:>15} │", "Strain", "Stress (MPa)");
    println!("│──────────────┼─────────────────│");

    for &strain in &strains {
        let strain_vec = [strain, 0.0, 0.0, 0.0, 0.0, 0.0];
        let stress = steel.stress(&strain_vec, 0.0);
        println!("│ {:>12.5} │ {:>15.1} │", strain, stress[0] / 1e6);
    }

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Demonstrates elastic-plastic behavior.
fn demo_elastic_plastic() -> anyhow::Result<()> {
    println!("\n┌─ Elastic-Plastic Analysis ─────────────────────────────┐");

    let plastic = VonMisesPlasticity::new(210e9, 0.3, 250e6, 1e9);

    println!("│ Material: Elastic-Plastic Steel");
    println!("│ Young's modulus: E = {:.0} GPa", plastic.e / 1e9);
    println!("│ Yield stress: σ_y = {:.0} MPa", plastic.yield_stress / 1e6);
    println!("│ Hardening modulus: H = {:.0} GPa", plastic.hardening / 1e9);

    println!("│\n│ Loading History:");
    println!("│ {:>10} │ {:>12} │ {:>12} │ {:>10} │",
        "Strain", "Stress", "Eq. Plast.", "Status");
    println!("│────────────┼──────────────┼──────────────┼────────────│");

    let mut state = StressState::default();

    // Monotonic loading
    let strains = vec![0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01];

    for strain in strains {
        let strain_vec = [strain, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (new_state, _tangent) = plastic.update(&strain_vec, &state);

        let status = if new_state.plastic_strain > 0.0 {
            "Plastic"
        } else {
            "Elastic"
        };

        println!("│ {:>10.4} │ {:>12.1} │ {:>12.4} │ {:>10} │",
            strain, new_state.stress[0] / 1e6, new_state.plastic_strain, status);

        state = new_state;
    }

    // Unloading
    println!("│\n│ Unloading:");
    let unload_strains = vec![0.008, 0.005, 0.002, 0.0];

    for strain in unload_strains {
        let strain_vec = [strain, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (new_state, _tangent) = plastic.update(&strain_vec, &state);

        println!("│ {:>10.4} │ {:>12.1} │ {:>12.4} │ {:>10} │",
            strain, new_state.stress[0] / 1e6, new_state.plastic_strain, "Unload");

        state = new_state;
    }

    println!("│\n│ Residual plastic strain after unloading: {:.4}", state.plastic_strain);

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Demonstrates Johnson-Cook viscoplasticity.
fn demo_johnson_cook() -> anyhow::Result<()> {
    println!("\n┌─ Johnson-Cook Viscoplasticity ─────────────────────────┐");

    let jc = JohnsonCook::new(
        210e9,  // E
        0.3,    // nu
        500e6,  // A (reference yield)
        500e6,  // B (hardening)
        0.02,   // C (rate sensitivity)
        0.3,    // n (hardening exponent)
        1.0,    // m (thermal softening)
        1.0,    // reference strain rate
        1800.0, // melting temp (K)
        293.0,  // reference temp (K)
    );

    println!("│ Material: Johnson-Cook Steel");
    println!("│ Reference yield: A = {:.0} MPa", jc.a / 1e6);
    println!("│ Hardening modulus: B = {:.0} MPa", jc.b / 1e6);
    println!("│ Rate sensitivity: C = {:.3}", jc.c);
    println!("│ Hardening exponent: n = {:.2}", jc.n);
    println!("│ Thermal exponent: m = {:.1}", jc.m);

    println!("│\n│ Strain Rate Effects (ε_p = 0.1, T = 293K):");
    println!("│ {:>15} │ {:>15} │", "Strain Rate", "Flow Stress");
    println!("│─────────────────┼─────────────────│");

    let strain_rates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];

    for &rate in &strain_rates {
        let flow = jc.flow_stress(0.1, rate, 293.0);
        println!("│ {:>15.3} │ {:>15.1} │", rate, flow / 1e6);
    }

    println!("│\n│ Temperature Effects (ε_p = 0.1, ε̇ = 1.0):");
    let temperatures = [293.0, 500.0, 800.0, 1000.0, 1200.0, 1500.0];

    println!("│ {:>15} │ {:>15} │", "Temp (K)", "Flow Stress");
    println!("│─────────────────┼─────────────────│");

    for &temp in &temperatures {
        let flow = jc.flow_stress(0.1, 1.0, temp);
        println!("│ {:>15.1} │ {:>15.1} │", temp, flow / 1e6);
    }

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Demonstrates hyperelastic material behavior.
fn demo_hyperelastic() -> anyhow::Result<()> {
    println!("\n┌─ Hyperelastic Material (Neo-Hookean) ──────────────────┐");

    let neo = NeoHookean::from_en(10e6, 0.49); // Rubber-like material

    println!("│ Material: Neo-Hookean (Rubber)");
    println!("│ Shear modulus: μ = {:.2} MPa", neo.mu / 1e6);
    println!("│ Bulk modulus: κ = {:.2} MPa", neo.kappa / 1e6);

    println!("│\n│ Simple Shear Response:");
    println!("│ {:>12} │ {:>15} │", "Shear Strain", "Shear Stress");
    println!("│──────────────┼─────────────────│");

    let shears = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0];

    for &gamma in &shears {
        // Simple shear: F = [[1, γ, 0], [0, 1, 0], [0, 0, 1]]
        // C = F^T * F
        let c11 = 1.0 + gamma * gamma;
        let c12 = gamma;
        let c22 = 1.0;
        let c33 = 1.0;

        // Simplified stress calculation
        let s12 = neo.mu * gamma;

        println!("│ {:>12.2} │ {:>15.2} │", gamma, s12 / 1e6);
    }

    println!("│\n│ Mooney-Rivlin Comparison:");
    let mr = fea::materials::nonlinear::MooneyRivlin::new(0.8e6, 0.2e6, 200e6);

    for &gamma in &[0.2, 0.5, 1.0] {
        let i1 = 3.0 + gamma * gamma;
        let i2 = 3.0 + gamma * gamma;
        let j = 1.0;
        let energy = mr.strain_energy(i1, i2, j);
        println!("│ γ={:.1}: Strain energy density = {:.2} kJ/m³", gamma, energy / 1e3);
    }

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Demonstrates damage mechanics.
fn demo_damage_mechanics() -> anyhow::Result<()> {
    println!("\n┌─ Damage Mechanics ─────────────────────────────────────┐");

    let damage = DamageModel::new(210e9, 0.3, 0.001, 0.02);

    println!("│ Material: Damage Model");
    println!("│ Initiation strain: ε₀ = {:.4}", damage.init_strain);
    println!("│ Failure strain: ε_f = {:.4}", damage.failure_strain);

    println!("│\n│ Damage Evolution:");
    println!("│ {:>12} │ {:>12} │ {:>15} │", "Strain", "Damage", "Effective E");
    println!("│──────────────┼──────────────┼───────────────│");

    let strains = [0.0005, 0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025];

    for &strain in &strains {
        let d = damage.compute_damage(strain);
        let e_eff = damage.e * (1.0 - d);

        let damage_str = if d < 0.01 {
            "None"
        } else if d < 0.3 {
            "Initiated"
        } else if d < 0.7 {
            "Growing"
        } else if d < 0.99 {
            "Critical"
        } else {
            "Failed"
        };

        println!("│ {:>12.4} │ {:>12.3} │ {:>12.1} GPa │ ({})",
            strain, d, e_eff / 1e9, damage_str);
    }

    println!("│\n│ Stress-Softening:");
    let stress_undamaged = [100.0, 200.0, 300.0, 400.0, 500.0]; // MPa

    println!("│ {:>12} │ {:>12} │ {:>12} │", "Damage", "Original", "Damaged");
    println!("│──────────────┼──────────────┼──────────────│");

    for &d in &[0.0, 0.2, 0.4, 0.6, 0.8] {
        println!("│ {:>12.2} │ {:>12} │ {:>12} │", d, "-----", "-----");
        for &s in &stress_undamaged {
            let stress_vec = [s, 0.0, 0.0, 0.0, 0.0, 0.0];
            let damaged = damage.damaged_stress(&stress_vec, d);
            if s == 100.0 {
                println!("│              │ {:>10.1} MPa │ {:>10.1} MPa │",
                    s, damaged[0] / 1e6);
            }
        }
    }

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Generates complete stress-strain curve.
fn generate_stress_strain_curve() -> anyhow::Result<()> {
    println!("\n┌─ Complete Stress-Strain Curve Generation ──────────────┐");

    let plastic = VonMisesPlasticity::new(200e9, 0.3, 300e6, 2e9);

    println!("│ Generating stress-strain curve...");
    println!("│ Material: Elastic-Plastic (σ_y = 300 MPa)");

    let start = Instant::now();

    let n_points = 100;
    let max_strain = 0.05;
    let mut stresses = Vec::with_capacity(n_points);
    let mut strains = Vec::with_capacity(n_points);

    let mut state = StressState::default();

    for i in 0..n_points {
        let strain = (i as f64 / n_points as f64) * max_strain;
        let strain_vec = [strain, 0.0, 0.0, 0.0, 0.0, 0.0];

        let (new_state, _tangent) = plastic.update(&strain_vec, &state);
        stresses.push(new_state.stress[0]);
        strains.push(strain);
        state = new_state;
    }

    let elapsed = start.elapsed();

    println!("│ Generated {} points in {:.2} ms", n_points, elapsed.as_secs_f64() * 1000.0);

    // Key points on curve
    println!("│\n│ Key Points:");

    // Yield point (find first plastic)
    let mut yield_idx = 0;
    for i in 0..n_points {
        if stresses[i] > 300e6 {
            yield_idx = i;
            break;
        }
    }

    // Ultimate point
    let max_stress = stresses.iter().copied().fold(0.0_f64, f64::max);
    let ult_idx = stresses.iter().position(|&s| s == max_stress).unwrap_or(0);

    // Fracture (last point)
    let fracture_idx = n_points - 1;

    println!("│   Proportional limit: ε={:.5}, σ={:.1} MPa",
        strains[0], stresses[0] / 1e6);
    println!("│   Yield point:        ε={:.5}, σ={:.1} MPa",
        strains[yield_idx], stresses[yield_idx] / 1e6);
    println!("│   Ultimate strength:  ε={:.5}, σ={:.1} MPa",
        strains[ult_idx], stresses[ult_idx] / 1e6);
    println!("│   Fracture point:     ε={:.5}, σ={:.1} MPa",
        strains[fracture_idx], stresses[fracture_idx] / 1e6);

    // ASCII art stress-strain curve
    println!("│\n│ Stress-Strain Curve (ASCII):");
    println!("│ σ (MPa)");
    println!("│   │");

    let plot_height = 15;
    let plot_width = 50;

    for row in (0..plot_height).rev() {
        let stress_level = (row as f64 / plot_height as f64) * max_stress;
        print!("│{:>5}│", (stress_level / 1e6) as i32);

        for col in 0..plot_width {
            let strain_idx = (col as f64 / plot_width as f64 * n_points as f64) as usize;
            if strain_idx < stresses.len() && stresses[strain_idx] >= stress_level {
                print!("*");
            } else {
                print!(" ");
            }
        }
        println!("");
    }

    print!("│     └");
    for _ in 0..plot_width {
        print!("-");
    }
    println!("→ ε");

    println!("│      0    0.01   0.02   0.03   0.04   0.05");

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}
