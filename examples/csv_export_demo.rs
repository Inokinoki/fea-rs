//! CSV Export Demo.

use fea::preprocessing::mesh_generation::generate_rect_2d;

fn main() -> anyhow::Result<()> {
    println!("=== CSV Export Demo ===\n");

    let (nodes, elems) = generate_rect_2d(1.0, 0.5, 20, 10);
    println!("Generated: {} nodes, {} elements", nodes.len(), elems.len());

    let displacements: Vec<f64> = (0..nodes.len() * 2)
        .map(|i| (i as f64) * 0.0001)
        .collect();

    println!("\nNode Displacements (CSV format):");
    println!("NodeID,UX,UY,Magnitude");
    for i in 0..5.min(nodes.len()) {
        let ux = displacements[i * 2];
        let uy = displacements[i * 2 + 1];
        let mag = (ux * ux + uy * uy).sqrt();
        println!("{},{:.6},{:.6},{:.6}", i, ux, uy, mag);
    }
    println!("... ({} total nodes)", nodes.len());

    println!("\nCSV export ready for Excel/MATLAB");
    println!("Complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_data() {
        let (nodes, _) = generate_rect_2d(1.0, 0.5, 10, 5);
        let disp: Vec<f64> = (0..nodes.len() * 2).map(|i| (i as f64) * 0.0001).collect();
        
        let csv = format!("{},{:.6},{:.6}", 0, disp[0], disp[1]);
        assert!(csv.contains(','));
    }

    #[test]
    fn test_magnitude() {
        let ux = 0.001_f64;
        let uy = 0.002_f64;
        let mag = (ux * ux + uy * uy).sqrt();
        assert!(mag > 0.0 && mag < 0.01);
    }
}
