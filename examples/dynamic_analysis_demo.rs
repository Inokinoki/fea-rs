//! Dynamic Analysis Demo.

use fea::preprocessing::mesh_generation::generate_bar_1d;
use fea::preprocessing::material_helpers::steel_a36;

fn main() -> anyhow::Result<()> {
    println!("=== Dynamic Analysis Demo ===\n");

    let (nodes, elems) = generate_bar_1d(2.0, 20, 0.01);
    println!("Model: {} nodes, {} elements", nodes.len(), elems.len());

    let steel = steel_a36();
    println!("\nMaterial: {}", steel.name);
    println!("  E = {} GPa", (steel.young_modulus / 1e9) as i64);

    let l = 2.0_f64;
    let rho = steel.density;
    let e = steel.young_modulus;

    let c = (e / rho).sqrt();
    println!("\nWave speed: {:.0} m/s", c);

    let f1 = c / (4.0 * l);
    println!("Fundamental freq: {:.1} Hz", f1);

    let dx = l / elems.len() as f64;
    let dt_crit = dx / c;
    println!("\nElement length: {:.4} m", dx);
    println!("Critical dt: {:.6} s", dt_crit);

    let t1 = 1.0 / f1;
    println!("\nPeriod: {:.4} s", t1);

    println!("\nComplete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bar_mesh() {
        let (n, _e) = generate_bar_1d(1.0, 10, 0.01);
        assert_eq!(n.len(), 11);
    }

    #[test]
    fn test_wave_speed() {
        let e = 210e9_f64;
        let rho = 7850.0_f64;
        let c = (e / rho).sqrt();
        assert!(c > 5000.0 && c < 6000.0);
    }

    #[test]
    fn test_frequency() {
        let c = 5170.0_f64;
        let l = 2.0_f64;
        let f = c / (4.0 * l);
        assert!(f > 600.0 && f < 700.0);
    }
}
