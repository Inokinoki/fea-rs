//! 2D Stress Analysis Example.

use fea::preprocessing::mesh_generation::generate_rect_2d;
use fea::preprocessing::material_helpers::steel_a36;

fn main() -> anyhow::Result<()> {
    println!("=== 2D Stress Analysis Example ===\n");

    let (nodes, elems) = generate_rect_2d(1.0, 0.5, 40, 20);
    println!("Mesh: {} nodes, {} elements", nodes.len(), elems.len());

    let steel = steel_a36();
    println!("\nMaterial: {}", steel.name);
    println!("  E = {} GPa", (steel.young_modulus / 1e9) as i64);
    println!("  nu = {:.2}", steel.poisson_ratio);
    println!("  Yield = {} MPa", (steel.yield_strength / 1e6) as i64);

    // Stress calculation
    let e = steel.young_modulus;
    let nu = steel.poisson_ratio;
    let strain_x = 0.001_f64;
    let strain_y = 0.0005_f64;
    
    let factor = e / (1.0 - nu * nu);
    let sigma_x = factor * (strain_x + nu * strain_y);
    let sigma_y = factor * (strain_y + nu * strain_x);
    let tau_xy = 25e6_f64;
    
    let von_mises = (sigma_x * sigma_x + sigma_y * sigma_y 
        - sigma_x * sigma_y + 3.0 * tau_xy * tau_xy).sqrt();
    
    println!("\nStresses:");
    println!("  sigma_x = {:.1} MPa", sigma_x / 1e6);
    println!("  sigma_y = {:.1} MPa", sigma_y / 1e6);
    println!("  von Mises = {:.1} MPa", von_mises / 1e6);
    
    let fos = steel.yield_strength / von_mises;
    println!("\nFactor of Safety: {:.2}", fos);
    println!("  Status: {}", if fos > 1.5 { "SAFE" } else { "CHECK" });

    println!("\nComplete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2d_mesh() {
        let (n, e) = generate_rect_2d(1.0, 0.5, 20, 10);
        assert!(n.len() > 0 && e.len() > 0);
    }

    #[test]
    fn test_stress_calc() {
        let e = 210e9_f64;
        let nu = 0.3_f64;
        let factor = e / (1.0 - nu * nu);
        let sigma = factor * 0.001_f64;
        assert!(sigma > 0.0 && sigma < 500e6);
    }

    #[test]
    fn test_von_mises() {
        let sx = 100e6_f64;
        let sy = 50e6_f64;
        let txy = 25e6_f64;
        let vm = (sx*sx + sy*sy - sx*sy + 3.0*txy*txy).sqrt();
        assert!(vm > 0.0 && vm < 200e6);
    }

    #[test]
    fn test_fos() {
        let fos = 250e6_f64 / 150e6_f64;
        assert!(fos > 1.0 && fos < 2.0);
    }
}
