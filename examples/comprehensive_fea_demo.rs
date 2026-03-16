//! Comprehensive FEA Demo.

use fea::preprocessing::mesh_generation::generate_rect_2d;
use fea::preprocessing::material_helpers::steel_a36;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Comprehensive FEA Demo ===\n");

    let total_start = Instant::now();

    // Generate mesh
    println!("1. Mesh Generation:");
    let (nodes, elems) = generate_rect_2d(2.0, 1.0, 20, 10);
    println!("   Nodes: {}, Elements: {}", nodes.len(), elems.len());

    // Material
    println!("\n2. Material Properties:");
    let steel = steel_a36();
    println!("   Name: {}", steel.name);
    println!("   E = {} GPa", (steel.young_modulus / 1e9) as i64);
    println!("   rho = {} kg/m3", steel.density as i64);

    // Timing demo
    println!("\n3. Performance:");
    let start = Instant::now();
    let _ = generate_rect_2d(1.0, 0.5, 100, 50);
    let mesh_time = start.elapsed().as_secs_f64() * 1000.0;
    println!("   Mesh gen (5000 elems): {:.2} ms", mesh_time);

    let total_time = total_start.elapsed().as_secs_f64() * 1000.0;
    println!("\n4. Total time: {:.2} ms", total_time);
    
    println!("\nDemo Complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_generation() {
        let (nodes, elems) = generate_rect_2d(1.0, 0.5, 10, 5);
        assert!(nodes.len() > 0);
        assert!(elems.len() > 0);
    }

    #[test]
    fn test_material_properties() {
        let steel = steel_a36();
        assert!(steel.young_modulus > 100e9);
        assert!(steel.density > 7000.0);
    }

    #[test]
    fn test_static_config_creation() {
        let _config = fea::algorithms::analysis::StaticConfig::default();
        // Config created successfully
    }
}
