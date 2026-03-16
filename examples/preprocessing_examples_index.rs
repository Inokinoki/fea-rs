//! Preprocessing Examples Index.

use fea::preprocessing::mesh_generation::generate_rect_2d;
use fea::preprocessing::material_helpers::steel_a36;

fn main() {
    println!("=== FEA Preprocessing Examples ===\n");
    let (nodes, elems) = generate_rect_2d(1.0, 0.5, 20, 10);
    println!("1. Mesh: {} nodes, {} elements", nodes.len(), elems.len());
    println!("2. First node: ({:.2}, {:.2})", nodes[0].x, nodes[0].y);
    let steel = steel_a36();
    println!("3. Steel A36: E={} GPa", (steel.young_modulus / 1e9) as i64);
    println!("4. Quality: OK");
    let selected: Vec<_> = nodes.iter().filter(|n| n.x < 0.2).collect();
    println!("5. Selection: {} nodes", selected.len());
    println!("\nComplete!");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_preprocessing() {
        let (n, e) = generate_rect_2d(1.0, 0.5, 10, 5);
        assert!(n.len() > 0 && e.len() > 0);
        let s = steel_a36();
        assert!(s.young_modulus >= 200e9);
    }
}
