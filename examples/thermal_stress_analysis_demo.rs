//! Thermal Stress Analysis Demo.

use fea::algorithms::thermal::ThermalProperties;

fn main() -> anyhow::Result<()> {
    println!("=== Thermal Stress Analysis Demo ===\n");

    // Material thermal properties
    let props = ThermalProperties::default();
    
    println!("Thermal Properties (Steel):");
    println!("  CTE (alpha): {:.1e} /K", props.alpha);
    println!("  Reference temp: {:.1} K", props.reference_temp);

    // Custom thermal properties
    let aluminum_props = ThermalProperties {
        alpha: 23e-6,
        reference_temp: 293.0,
    };
    println!("\nThermal Properties (Aluminum):");
    println!("  CTE (alpha): {:.1e} /K", aluminum_props.alpha);

    // Thermal stress calculation demo
    let delta_t = 100.0_f64;
    let e_modulus = 210e9_f64;
    let thermal_strain = props.alpha * delta_t;
    let thermal_stress = e_modulus * thermal_strain;

    println!("\nThermal Effects (ΔT = {:.0} K):", delta_t);
    println!("  Thermal strain: {:.6}", thermal_strain);
    println!("  Thermal stress: {:.1} MPa", thermal_stress / 1e6);
    
    println!("\nDemo Complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_thermal_properties_default() {
        let props = ThermalProperties::default();
        assert!(props.alpha > 0.0);
        assert!(props.reference_temp > 0.0);
    }
    
    #[test]
    fn test_thermal_properties_custom() {
        let props = ThermalProperties {
            alpha: 23e-6,
            reference_temp: 293.0,
        };
        assert!((props.alpha - 23e-6).abs() < 1e-10);
    }
    
    #[test]
    fn test_thermal_strain_calculation() {
        let alpha = 12e-6_f64;
        let delta_t = 100.0_f64;
        let strain = alpha * delta_t;
        assert!((strain - 0.0012_f64).abs() < 1e-10);
    }
}
