//! Advanced Mesh Generation Examples.

use fea::preprocessing::mesh_generation::{
    generate_bar_1d, generate_rect_2d, generate_box_3d,
    generate_tri_2d_from_rect,
};

fn main() -> anyhow::Result<()> {
    println!("=== Advanced Mesh Generation ===\n");

    let (n, _e) = generate_bar_1d(1.0, 20, 0.01);
    println!("1. 1D Bar: {} nodes", n.len());

    let (n, e) = generate_rect_2d(1.0, 0.5, 40, 20);
    println!("2. 2D Rect: {} nodes, {} elements", n.len(), e.len());

    let (n, e) = generate_tri_2d_from_rect(1.0, 0.5, 40, 20);
    println!("3. 2D Tri: {} nodes, {} triangles", n.len(), e.len());

    let (n, e) = generate_box_3d(1.0, 0.5, 0.25, 20, 10, 5);
    println!("4. 3D Box: {} nodes, {} hexes", n.len(), e.len());

    println!("\nComplete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bar_1d() {
        let (n, e) = generate_bar_1d(1.0, 10, 0.01);
        assert_eq!(n.len(), 11);
        assert_eq!(e.len(), 10);
    }

    #[test]
    fn test_rect_2d() {
        let (n, e) = generate_rect_2d(1.0, 0.5, 10, 5);
        assert!(n.len() > 0);
        assert!(e.len() > 0);
    }

    #[test]
    fn test_tri_2d() {
        let (_n, e) = generate_tri_2d_from_rect(1.0, 0.5, 10, 5);
        assert!(e.len() > 0);
    }

    #[test]
    fn test_box_3d() {
        let (n, e) = generate_box_3d(1.0, 0.5, 0.25, 10, 5, 3);
        assert!(n.len() > 0);
        assert!(e.len() > 0);
    }
}
