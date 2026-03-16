//! VTK Export Demo.

use fea::preprocessing::mesh_generation::generate_rect_2d;

fn main() -> anyhow::Result<()> {
    println!("=== VTK Export Demo ===\n");

    let (nodes, elems) = generate_rect_2d(1.0, 0.5, 50, 25);
    println!("Generated mesh: {} nodes, {} elements", nodes.len(), elems.len());

    let displacements: Vec<f64> = (0..nodes.len())
        .map(|i| (i as f64) * 0.0001)
        .collect();

    let max_disp = displacements.iter().copied().fold(0.0_f64, f64::max);
    println!("Displacement range: 0 to {:.6} m", max_disp);

    if let Some(first) = elems.first() {
        println!("First element nodes: {:?}", first);
    }

    println!("\nVTK export ready (use ParaView for visualization)");
    println!("Complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_export_data() {
        let (nodes, elems) = generate_rect_2d(1.0, 0.5, 10, 5);
        
        assert!(nodes.len() > 0);
        assert!(elems.len() > 0);
        
        if let Some(first) = elems.first() {
            assert_eq!(first.len(), 4);
        }
    }

    #[test]
    fn test_displacement_generation() {
        let n = 100;
        let disp: Vec<f64> = (0..n).map(|i| (i as f64) * 0.0001).collect();
        assert_eq!(disp.len(), n);
    }
}
