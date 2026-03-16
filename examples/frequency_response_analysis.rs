//! Frequency Response Analysis Example.

use fea::algorithms::analysis::{HarmonicConfig};

fn main() -> anyhow::Result<()> {
    println!("=== Frequency Response Analysis ===\n");
    
    let config = HarmonicConfig {
        freq_start: 1.0,
        freq_end: 100.0,
        num_points: 10,
        damping_ratio: 0.02,
        load_amplitude: vec![100.0],
    };
    
    println!("Configuration:");
    println!("  Frequency range: {:.1} - {:.1} rad/s", config.freq_start, config.freq_end);
    println!("  Damping ratio: {:.1}%", config.damping_ratio * 100.0);
    println!("  Frequency points: {}", config.num_points);
    println!("\nAnalysis Setup Complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_harmonic_config() {
        let config = HarmonicConfig {
            freq_start: 1.0,
            freq_end: 50.0,
            num_points: 5,
            damping_ratio: 0.02,
            load_amplitude: vec![100.0_f64],
        };
        assert_eq!(config.num_points, 5);
        assert!(config.damping_ratio > 0.0);
        assert!(config.freq_end > config.freq_start);
    }
    
    #[test]
    fn test_harmonic_config_default_values() {
        let config = HarmonicConfig {
            freq_start: 0.0,
            freq_end: 100.0,
            num_points: 10,
            damping_ratio: 0.05,
            load_amplitude: vec![],
        };
        assert_eq!(config.num_points, 10);
        assert!(config.damping_ratio > 0.0);
    }
}
