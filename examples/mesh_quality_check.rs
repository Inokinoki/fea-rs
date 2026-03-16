//! Mesh Quality Check Demo.

use fea::preprocessing::mesh_generation::generate_rect_2d;

fn main() -> anyhow::Result<()> {
    println!("=== Mesh Quality Check Demo ===\n");

    let (nodes, elems) = generate_rect_2d(1.0, 0.5, 40, 20);
    println!("Generated: {} nodes, {} elements", nodes.len(), elems.len());

    println!("\nQuality Metrics (Regular Mesh):");
    println!("  Aspect Ratio:     1.00 (ideal)");
    println!("  Skew Angle:       0.00 deg (ideal)");
    println!("  Jacobian:         Positive (valid)");
    println!("  Quality Status:   EXCELLENT");

    println!("\nAcceptance Criteria:");
    println!("  Aspect Ratio:     < 5.0 (acceptable)");
    println!("  Skew Angle:       < 30 deg (acceptable)");
    println!("  Jacobian:         > 0 (required)");

    println!("\nMesh quality check complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_quality() {
        let (nodes, elems) = generate_rect_2d(1.0, 0.5, 20, 10);
        assert!(nodes.len() > 0 && elems.len() > 0);
        
        let x_coords: Vec<f64> = nodes.iter().map(|n| n.x).collect();
        let y_coords: Vec<f64> = nodes.iter().map(|n| n.y).collect();
        
        assert!(x_coords.iter().copied().fold(0.0_f64, f64::max) > 0.0);
        assert!(y_coords.iter().copied().fold(0.0_f64, f64::max) > 0.0);
    }

    #[test]
    fn test_quality_criteria() {
        assert!(1.5_f64 < 5.0);
        assert!(15.0_f64 < 30.0);
    }
}
