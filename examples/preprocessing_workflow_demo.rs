//! Preprocessing Workflow Demo.

use fea::preprocessing::mesh_generation::{generate_rect_2d, generate_bar_1d, generate_box_3d};
use fea::preprocessing::material_helpers::steel_a36;

fn main() -> anyhow::Result<()> {
    println!("=== Preprocessing Workflow Demo ===\n");
    
    let (n1, _e1) = generate_bar_1d(1.0, 10, 0.01);
    println!("1. 1D Mesh: {} nodes", n1.len());
    
    let (n2, _e2) = generate_rect_2d(1.0, 0.5, 20, 10);
    println!("2. 2D Mesh: {} nodes", n2.len());
    
    let (n3, _e3) = generate_box_3d(1.0, 0.5, 0.25, 10, 5, 3);
    println!("3. 3D Mesh: {} nodes", n3.len());
    
    let steel = steel_a36();
    println!("4. Steel: E={} GPa, rho={} kg/m³", 
             (steel.young_modulus / 1e9) as i64, steel.density as i64);
    
    let left: Vec<_> = n2.iter().filter(|n| n.x < 0.2).collect();
    println!("5. Left edge nodes: {}", left.len());
    
    println!("\nComplete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_meshes() {
        let (n, _e) = generate_bar_1d(1.0, 10, 0.01);
        assert!(n.len() > 0);
        let (n, _e) = generate_rect_2d(1.0, 0.5, 10, 5);
        assert!(n.len() > 0);
        let (n, _e) = generate_box_3d(1.0, 0.5, 0.25, 5, 3, 2);
        assert!(n.len() > 0);
    }
    #[test]
    fn test_materials() {
        let s = steel_a36();
        assert!(s.young_modulus > 100e9);
    }
}
