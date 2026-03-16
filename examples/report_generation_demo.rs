//! Report Generation Demo.

use fea::preprocessing::{
    mesh_generation::generate_rect_2d,
    material_helpers::steel_a36,
};

fn main() -> anyhow::Result<()> {
    println!("=== Report Generation Demo ===\n");

    let (nodes, elems) = generate_rect_2d(1.0, 0.5, 40, 20);
    let steel = steel_a36();

    let max_disp = 0.00234_f64;
    let max_stress = 185e6_f64;
    let total_reaction = 1000.5_f64;

    println!("FEA ANALYSIS REPORT");
    println!("===================\n");
    println!("MODEL INFORMATION");
    println!("  Geometry:  1.0m x 0.5m");
    println!("  Nodes:     {}", nodes.len());
    println!("  Elements:  {}", elems.len());
    println!("  Material:  {}", steel.name);
    
    println!("\nRESULTS SUMMARY");
    println!("  Max Displacement: {:.6} m", max_disp);
    println!("  Max Stress:       {:.1} MPa", max_stress / 1e6);
    println!("  Total Reaction:   {:.1} N", total_reaction);
    
    println!("\nDESIGN CHECK");
    let fos = steel.yield_strength / max_stress;
    let status = if fos > 1.5 { "PASS" } else { "MARGINAL" };
    println!("  Factor of Safety: {:.2}", fos);
    println!("  Status:           {}", status);

    println!("\nReport generation complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_data() {
        let (nodes, elems) = generate_rect_2d(1.0, 0.5, 20, 10);
        let steel = steel_a36();
        assert!(nodes.len() > 0 && steel.young_modulus > 0.0);
    }

    #[test]
    fn test_fos() {
        let fos = 250e6_f64 / 150e6_f64;
        assert!(fos > 1.0 && fos < 2.0);
    }
}
