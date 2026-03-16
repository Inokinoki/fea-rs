//! Response spectrum analysis example.
//!
//! Demonstrates seismic analysis using response spectrum method.

use fea::prelude::*;
use nalgebra::DMatrix;

fn main() -> anyhow::Result<()> {
    println!("=== Response Spectrum Analysis ===\n");

    // Analyze a structure under seismic loading
    analyze_seismic_response()?;

    println!("\n=== Analysis Complete ===");

    Ok(())
}

/// Analyze seismic response using response spectrum method
fn analyze_seismic_response() -> anyhow::Result<()> {
    println!("Seismic Analysis of Multi-Story Frame");
    println!("-------------------------------------\n");

    // Material and section properties
    let e: f64 = 200e9; // Steel
    let floor_mass: f64 = 10000.0; // 10 tonnes per floor
    let num_stories = 5;
    let story_height = 3.0;

    // Create a simplified lumped mass model using diagonal mass matrix
    let ndof = num_stories;
    let mut mass = DMatrix::zeros(ndof, ndof);
    let mut stiffness = DMatrix::zeros(ndof, ndof);

    // Story stiffness (simplified: k = 12EI/L^3 for columns)
    let story_stiffness: f64 = 1e7;

    for i in 0..num_stories {
        // Mass
        mass[(i, i)] = floor_mass;

        // Stiffness
        if i > 0 {
            stiffness[(i, i - 1)] = -story_stiffness;
            stiffness[(i - 1, i)] = -story_stiffness;
        }
        stiffness[(i, i)] += if i == num_stories - 1 {
            story_stiffness
        } else {
            2.0 * story_stiffness
        };
    }

    println!("Building Properties:");
    println!("  - Stories: {}", num_stories);
    println!("  - Story height: {:.1} m", story_height);
    println!("  - Floor mass: {:.0} kg", floor_mass);
    println!();

    // Step 1: Modal analysis (simplified using power iteration)
    println!("Step 1: Natural Frequencies (estimated)");
    println!("---------------------------------------");

    // Estimate frequencies using simplified formulas
    let mut periods = Vec::new();
    for i in 1..=num_stories {
        // Approximate period for mode i: T_i ≈ T_1 / i
        let t1 = 0.1 * num_stories as f64; // Approximate fundamental period
        let ti = t1 / i as f64;
        periods.push(ti);
        let fi = 1.0 / ti;
        println!("  Mode {}: Freq = {:.2} Hz, Period = {:.3} s", i, fi, ti);
    }

    // Step 2: Define response spectrum (simplified Eurocode 8 type)
    println!("\nStep 2: Response Spectrum");
    println!("-----------------------");

    let pga = 0.3 * 9.81; // Peak ground acceleration (0.3g)
    let tb = 0.15; // Corner period 1
    let tc = 0.5; // Corner period 2
    let eta = 1.0; // Damping correction factor

    println!("  PGA: {:.2} m/s^2 ({:.2}g)", pga, pga / 9.81);
    println!("  Tb: {:.2} s, Tc: {:.2} s", tb, tc);

    // Step 3: Compute spectral acceleration for each mode
    println!("\nStep 3: Modal Response");
    println!("--------------------");

    let mut base_shear_total: f64 = 0.0;
    let mut top_disp_total: f64 = 0.0;

    println!("  Mode | Period(s) | Sa(g) | Base Shear(kN) | Top Disp(mm)");
    println!("  -----|-----------|-------|----------------|--------------");

    for (i, &period) in periods.iter().enumerate() {
        // Compute spectral acceleration from response spectrum
        let sa = if period < tb {
            pga * (1.0 + (eta - 1.0) * period / tb)
        } else if period <= tc {
            pga * eta
        } else {
            pga * eta * tc / period
        };

        // Simplified modal contribution (using effective mass approximation)
        let effective_mass = floor_mass * num_stories as f64 / (i + 1) as f64;
        let base_shear = effective_mass * sa;

        // Approximate top displacement
        let omega = 2.0 * std::f64::consts::PI / period;
        let sd = sa / (omega * omega);
        let top_disp = sd * num_stories as f64;

        base_shear_total = base_shear_total.hypot(base_shear); // SRSS combination
        top_disp_total = top_disp_total.hypot(top_disp);

        println!("  {:4} | {:9.3} | {:5.2} | {:14.1} | {:12.1}",
            i + 1, period, sa / 9.81, base_shear / 1000.0, top_disp * 1000.0);
    }

    println!("\nStep 4: Combined Response (SRSS)");
    println!("------------------------------");
    println!("  Total Base Shear: {:.1} kN", base_shear_total / 1000.0);
    println!("  Top Displacement: {:.1} mm", top_disp_total * 1000.0);

    // Check against code limits
    let drift_limit = story_height / 150.0; // Inter-story drift limit
    let actual_drift = top_disp_total / num_stories as f64;

    println!("\nCode Compliance Check:");
    println!("  Drift limit: {:.1} mm", drift_limit * 1000.0);
    println!("  Actual drift: {:.1} mm", actual_drift * 1000.0);

    if actual_drift <= drift_limit {
        println!("  Status: PASSED - Within code limits");
    } else {
        println!("  Status: FAILED - Exceeds code limits");
    }

    Ok(())
}
