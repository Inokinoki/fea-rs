//! Plate Bending Analysis Demo.

use fea::preprocessing::mesh_generation::generate_rect_2d;
use fea::preprocessing::material_helpers::steel_a36;

fn main() -> anyhow::Result<()> {
    println!("=== Plate Bending Analysis Demo ===\n");

    let (nodes, elems) = generate_rect_2d(1.0, 0.5, 40, 20);
    println!("Plate Mesh: {} nodes, {} elements", nodes.len(), elems.len());

    let steel = steel_a36();
    println!("\nMaterial: {}", steel.name);
    println!("  E = {} GPa", (steel.young_modulus / 1e9) as i64);

    let h = 0.01_f64;
    let e = steel.young_modulus;
    let nu = steel.poisson_ratio;
    let d = e * h.powi(3) / (12.0 * (1.0 - nu * nu));
    
    println!("\nFlexural Rigidity: {:.2} N·m", d);

    let q = 1000.0_f64;
    let a = 1.0_f64;
    let w_max = 0.00406_f64 * q * a.powi(4) / d;
    
    println!("\nSimply Supported:");
    println!("  Load: {:.1} kPa", q / 1000.0);
    println!("  Deflection: {:.4} mm", w_max * 1000.0);
    
    let m_max = 0.0479_f64 * q * a * a;
    let sigma_max = 6.0_f64 * m_max / (h * h);
    let fos = steel.yield_strength / sigma_max;
    
    println!("  Stress: {:.1} MPa", sigma_max / 1e6);
    println!("  FOS: {:.2}", fos);

    println!("\nComplete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plate_mesh() {
        let (n, e) = generate_rect_2d(1.0, 0.5, 20, 10);
        assert!(n.len() > 0 && e.len() > 0);
    }

    #[test]
    fn test_rigidity() {
        let e = 210e9_f64;
        let nu = 0.3_f64;
        let h = 0.01_f64;
        let d = e * h.powi(3) / (12.0 * (1.0 - nu * nu));
        assert!(d > 0.0 && d < 50000.0);
    }

    #[test]
    fn test_deflection() {
        let w = 0.00406_f64 * 1000.0_f64 / 17316.0_f64;
        assert!(w > 0.0 && w < 0.01);
    }
}
