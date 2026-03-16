//! Buckling Analysis Demo.

use fea::preprocessing::mesh_generation::generate_bar_1d;
use fea::preprocessing::material_helpers::steel_a36;

fn main() -> anyhow::Result<()> {
    println!("=== Buckling Analysis Demo ===\n");

    let (nodes, elems) = generate_bar_1d(3.0, 30, 0.01);
    println!("Column: {:.0}m, {} elements", 3.0, elems.len());

    let steel = steel_a36();
    println!("\nMaterial: {}", steel.name);
    println!("  E = {} GPa", (steel.young_modulus / 1e9) as i64);

    let l = 3.0_f64;
    let e = steel.young_modulus;
    let b = 0.1_f64;
    let h = 0.1_f64;
    let i = b * h.powi(3) / 12.0;

    println!("\nSection: {:.0}mm x {:.0}mm", b * 1000.0, h * 1000.0);

    let p_cr = std::f64::consts::PI.powi(2) * e * i / l.powi(2);
    println!("\nCritical Load (pinned): {:.1} kN", p_cr / 1000.0);

    let r = (i / (b * h)).sqrt();
    let slenderness = l / r;
    println!("Slenderness: L/r = {:.1}", slenderness);

    println!("\nComplete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_mesh() {
        let (n, _e) = generate_bar_1d(1.0, 10, 0.01);
        assert_eq!(n.len(), 11);
    }

    #[test]
    fn test_euler_buckling() {
        let e = 210e9_f64;
        let i = 8.33e-6_f64;
        let l = 3.0_f64;
        let p_cr = std::f64::consts::PI.powi(2) * e * i / l.powi(2);
        assert!(p_cr > 0.0);
    }

    #[test]
    fn test_slenderness() {
        let l = 3.0_f64;
        let r = 0.0289_f64;
        assert!(l / r > 100.0);
    }
}
