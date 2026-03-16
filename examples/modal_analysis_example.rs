//! Modal Analysis Example.

use fea::algorithms::analysis::{ModalAnalysis, ModalConfig};

fn main() -> anyhow::Result<()> {
    println!("=== Modal Analysis Example ===\n");
    
    let config = ModalConfig {
        num_modes: 5,
        consistent_mass: true,
        max_iterations: 100,
        tolerance: 1e-6,
    };
    
    println!("Modal Analysis Configuration:");
    println!("  Modes requested: {}", config.num_modes);
    println!("  Consistent mass: {}", config.consistent_mass);
    println!("  Max iterations: {}", config.max_iterations);
    println!("  Tolerance: {:.1e}", config.tolerance);
    println!("\nSetup Complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_modal_config() {
        let config = ModalConfig {
            num_modes: 5,
            consistent_mass: true,
            max_iterations: 100,
            tolerance: 1e-6,
        };
        assert_eq!(config.num_modes, 5);
        assert!(config.consistent_mass);
        assert!(config.tolerance > 0.0);
    }
    
    #[test]
    fn test_modal_config_defaults() {
        let config = ModalConfig {
            num_modes: 3,
            consistent_mass: false,
            max_iterations: 50,
            tolerance: 1e-8,
        };
        assert!(!config.consistent_mass);
    }
}
