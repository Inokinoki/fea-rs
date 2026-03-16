//! Beam Bending Analysis Demo.

use fea::preprocessing::mesh_generation::generate_bar_1d;
use fea::preprocessing::material_helpers::steel_a36;

fn main() -> anyhow::Result<()> {
    println!("=== Beam Bending Analysis Demo ===\n");

    let (nodes, elems) = generate_bar_1d(5.0, 50, 0.01);
    println!("Beam: {:.0}m, {} elements", 5.0, elems.len());
    println!("  Nodes: {}", nodes.len());

    let steel = steel_a36();
    println!("\nMaterial: {}", steel.name);
    println!("  E = {} GPa", (steel.young_modulus / 1e9) as i64);

    let l = 5.0_f64;
    let p = 10000.0_f64;
    let i = 8.33e-6_f64;
    let e = steel.young_modulus;
    
    let w_max = p * l.powi(3) / (48.0 * e * i);
    let m_max = p * l / 4.0;
    let h = 0.1_f64;
    let sigma_max = m_max * (h / 2.0) / i;
    let fos = steel.yield_strength / sigma_max;
    
    println!("\nSimply Supported:");
    println!("  Span: {:.0} m", l);
    println!("  Load: {:.0} kN", p / 1000.0);
    println!("  Deflection: {:.2} mm", w_max * 1000.0);
    println!("  Moment: {:.0} kN·m", m_max / 1000.0);
    println!("  Stress: {:.0} MPa", sigma_max / 1e6);
    println!("  FOS: {:.2}", fos);

    println!("\nComplete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_mesh() {
        let (n, e) = generate_bar_1d(1.0, 10, 0.01);
        assert_eq!(n.len(), 11);
        assert_eq!(e.len(), 10);
    }

    #[test]
    fn test_deflection() {
        let w = 10000.0_f64 * 5.0_f64.powi(3) / (48.0 * 210e9_f64 * 8.33e-6_f64);
        assert!(w > 0.0 && w < 0.1);
    }
}
