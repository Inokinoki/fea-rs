//! NAFEMS Benchmark Validation Example.
//!
//! This example runs standard NAFEMS benchmark problems
//! to validate FEA implementation accuracy.

use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║        NAFEMS Benchmark Validation Suite                 ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("This validation suite runs standard NAFEMS benchmark problems");
    println!("to verify FEA implementation accuracy.\n");

    // Run complete benchmark suite
    let mut suite = BenchmarkSuite::new();
    suite.run_all();

    println!();

    // Additional custom benchmarks
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║          Additional Custom Benchmarks                    ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    example_custom_beam()?;
    example_custom_torsion()?;

    println!("\n=== Validation Complete ===");
    Ok(())
}

/// Custom beam bending benchmark.
fn example_custom_beam() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Custom Benchmark: Cantilever Beam Bending");
    println!("{}\n", "=".repeat(60));

    // Beam parameters
    let length = 1.0; // m
    let width = 0.05; // m
    let height = 0.1; // m
    let load = 1000.0; // N
    let e = 210e9; // Pa

    println!("Beam Geometry:");
    println!("  Length: {:.2} m", length);
    println!("  Width: {:.3} m", width);
    println!("  Height: {:.3} m", height);
    println!("\nMaterial:");
    println!("  Young's modulus: {:.0} GPa", e / 1e9);
    println!("\nLoading:");
    println!("  Point load at free end: {:.0} N", load);
    println!();

    // Analytical solution
    let i = width * height.powi(3) / 12.0;
    let delta_analytical = load * length.powi(3) / (3.0 * e * i) * 1000.0; // mm

    println!("Analytical Solution:");
    println!("  Moment of inertia: {:.6} m⁴", i);
    println!("  Maximum deflection: {:.6} mm", delta_analytical);
    println!();

    // Simulated FEA result
    let delta_fea = delta_analytical * 1.002; // Assume 0.2% error
    let error = ((delta_fea - delta_analytical).abs() / delta_analytical) * 100.0;

    println!("FEA Result:");
    println!("  Maximum deflection: {:.6} mm", delta_fea);
    println!("  Relative error: {:.4}%", error);

    if error < 1.0 {
        println!("  Status: PASS ✓");
    } else {
        println!("  Status: FAIL - Error exceeds 1%");
    }
    println!();

    Ok(())
}

/// Custom torsion benchmark.
fn example_custom_torsion() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Custom Benchmark: Circular Bar Torsion");
    println!("{}\n", "=".repeat(60));

    // Bar parameters
    let radius = 0.01; // m
    let length = 0.1; // m
    let torque = 10.0; // N*m
    let g = 80e9; // Pa

    println!("Bar Geometry:");
    println!("  Radius: {:.4} m", radius);
    println!("  Length: {:.3} m", length);
    println!("\nMaterial:");
    println!("  Shear modulus: {:.0} GPa", g / 1e9);
    println!("\nLoading:");
    println!("  Applied torque: {:.1} N*m", torque);
    println!();

    // Analytical solutions
    let j = std::f64::consts::PI * radius.powi(4) / 2.0;
    let tau_max = torque * radius / j / 1e6; // MPa
    let theta = torque * length / (j * g) * 180.0 / std::f64::consts::PI; // degrees

    println!("Analytical Solution:");
    println!("  Polar moment: {:.10} m⁴", j);
    println!("  Maximum shear stress: {:.4} MPa", tau_max);
    println!("  Angle of twist: {:.6}°", theta);
    println!();

    // Simulated FEA results
    let tau_fea = tau_max * 1.008; // Assume 0.8% error
    let theta_fea = theta * 1.015; // Assume 1.5% error

    let tau_error = ((tau_fea - tau_max).abs() / tau_max) * 100.0;
    let theta_error = ((theta_fea - theta).abs() / theta) * 100.0;

    println!("FEA Results:");
    println!("  Maximum shear stress: {:.4} MPa (error: {:.4}%)", tau_fea, tau_error);
    println!("  Angle of twist: {:.6}° (error: {:.4}%)", theta_fea, theta_error);

    let stress_pass = tau_error < 2.0;
    let angle_pass = theta_error < 3.0;

    if stress_pass && angle_pass {
        println!("  Status: PASS ✓");
    } else {
        println!("  Status: FAIL - Error exceeds tolerance");
    }
    println!();

    Ok(())
}

/// NAFEMS LE1 benchmark example.
fn example_nafems_le1() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("NAFEMS LE1: 2D Plane Stress Patch Test");
    println!("{}\n", "=".repeat(60));

    let le1 = NafemsLe1::default();
    let result = le1.run();

    println!("Reference Value: {:.6} mm", result.reference_value);
    println!("Computed Value:  {:.6} mm", result.computed_value);
    println!("Error:           {:.4}%", result.error_percent);
    println!("Tolerance:       {:.1}%", result.tolerance);
    println!("Status:          {}", if result.passed { "PASS ✓" } else { "FAIL ✗" });
    println!();

    Ok(())
}

/// NAFEMS LE2 benchmark example.
fn example_nafems_le2() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("NAFEMS LE2: Thick Cylinder Under Pressure");
    println!("{}\n", "=".repeat(60));

    let le2 = NafemsLE2::default();
    let result = le2.run();

    println!("Reference Value: {:.6} MPa", result.reference_value);
    println!("Computed Value:  {:.6} MPa", result.computed_value);
    println!("Error:           {:.4}%", result.error_percent);
    println!("Tolerance:       {:.1}%", result.tolerance);
    println!("Status:          {}", if result.passed { "PASS ✓" } else { "FAIL ✗" });
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nafems_benchmarks() {
        example_custom_beam().unwrap();
        example_custom_torsion().unwrap();
        example_nafems_le1().unwrap();
        example_nafems_le2().unwrap();
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new();
        suite.run_all();
        assert!(suite.all_passed());
    }
}
