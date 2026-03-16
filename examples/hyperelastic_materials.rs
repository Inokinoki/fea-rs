//! Hyperelastic Material Analysis Example.
//!
//! This example demonstrates hyperelastic material models for large deformation analysis:
//! - Neo-Hookean model
//! - Mooney-Rivlin model
//! - Ogden model
//! - Yeoh model
//! - Arruda-Boyce model

use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Hyperelastic Material Analysis Example           ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Example 1: Neo-Hookean model
    example_neo_hookean()?;

    // Example 2: Mooney-Rivlin model
    example_mooney_rivlin()?;

    // Example 3: Ogden model
    example_ogden()?;

    // Example 4: Yeoh model
    example_yeoh()?;

    // Example 5: Arruda-Boyce model
    example_arruda_boyce()?;

    // Example 6: Material comparison
    example_material_comparison()?;

    println!("\n=== Hyperelastic Analysis Complete ===");
    Ok(())
}

/// Example 1: Neo-Hookean material model.
fn example_neo_hookean() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 1: Neo-Hookean Model");
    println!("{}\n", "=".repeat(60));

    // Create Neo-Hookean material (rubber-like)
    let params = HyperelasticParams::neo_hookean(0.5e6, 1e9);
    let mat = HyperelasticMaterial::new(params);

    println!("Material Properties:");
    println!("  Model: Neo-Hookean");
    println!("  Shear modulus: {:.2} MPa", 0.5e6 / 1e6);
    println!("  Bulk modulus: {:.0} MPa", 1e9 / 1e6);
    println!();

    // Uniaxial tension test
    println!("Uniaxial Tension Test:");
    println!("  {:>15} | {:>15} | {:>15}", "Stretch λ", "W (J/m³)", "PK2 Stress");
    println!("  {}", "-".repeat(50));

    let stretches = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5];

    for lambda in stretches {
        // Deformation gradient for uniaxial tension
        // F = diag(lambda, lambda^(-1/2), lambda^(-1/2)) for incompressible
        let lambda_lat = (1.0 / lambda).sqrt();
        let f = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(
            &[lambda, lambda_lat, lambda_lat]
        ));

        let w = mat.strain_energy(&f);
        let s = mat.stress_pk2(&f);

        println!("  {:>15.3} | {:>15.2} | {:>15.2}", lambda, w, s[(0, 0)]);
    }
    println!();

    Ok(())
}

/// Example 2: Mooney-Rivlin material model.
fn example_mooney_rivlin() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 2: Mooney-Rivlin Model");
    println!("{}\n", "=".repeat(60));

    // Create Mooney-Rivlin material (typical rubber parameters)
    let params = HyperelasticParams::mooney_rivlin(0.4e6, 0.1e6, 1e9);
    let mat = HyperelasticMaterial::new(params);

    println!("Material Properties:");
    println!("  Model: Mooney-Rivlin");
    println!("  C10: {:.2} MPa", 0.4e6 / 1e6);
    println!("  C01: {:.2} MPa", 0.1e6 / 1e6);
    println!("  Bulk modulus: {:.0} MPa", 1e9 / 1e6);
    println!();

    // Biaxial tension test
    println!("Biaxial Tension Test:");
    println!("  {:>15} | {:>15} | {:>15}", "Stretch λ", "W (J/m³)", "PK2 Stress");
    println!("  {}", "-".repeat(50));

    let stretches = vec![1.0, 1.1, 1.2, 1.3, 1.4];

    for lambda in stretches {
        // Deformation gradient for equibiaxial tension
        let lambda3 = 1.0 / (lambda * lambda);
        let f = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(
            &[lambda, lambda, lambda3]
        ));

        let w = mat.strain_energy(&f);
        let s = mat.stress_pk2(&f);

        println!("  {:>15.3} | {:>15.2} | {:>15.2}", lambda, w, s[(0, 0)]);
    }
    println!();

    Ok(())
}

/// Example 3: Ogden material model.
fn example_ogden() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 3: Ogden Model");
    println!("{}\n", "=".repeat(60));

    // Create Ogden material (2-term Ogden)
    let mut params = HyperelasticParams::default();
    params.model_type = HyperelasticModelType::Ogden;
    params.ogden_params = vec![
        (0.8e6, 1.3),  // (mu_1, alpha_1)
        (0.2e6, -1.3), // (mu_2, alpha_2)
    ];

    let mat = HyperelasticMaterial::new(params);

    println!("Material Properties:");
    println!("  Model: Ogden (2-term)");
    println!("  Term 1: mu=0.8 MPa, alpha=1.3");
    println!("  Term 2: mu=0.2 MPa, alpha=-1.3");
    println!();

    // Simple tension
    println!("Simple Tension Test:");
    println!("  {:>15} | {:>15}", "Stretch λ", "W (J/m³)");
    println!("  {}", "-".repeat(35));

    let stretches = vec![1.0, 1.2, 1.4, 1.6, 1.8, 2.0];

    for lambda in stretches {
        let lambda_lat = (1.0 / lambda).sqrt();
        let f = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(
            &[lambda, lambda_lat, lambda_lat]
        ));

        let w = mat.strain_energy(&f);
        println!("  {:>15.3} | {:>15.2}", lambda, w);
    }
    println!();

    Ok(())
}

/// Example 4: Yeoh material model.
fn example_yeoh() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 4: Yeoh Model");
    println!("{}\n", "=".repeat(60));

    // Create Yeoh material (3-term Yeoh)
    let mut params = HyperelasticParams::default();
    params.model_type = HyperelasticModelType::Yeoh;
    params.yeoh_params = vec![0.5e6, 0.01e6, 0.001e6]; // C10, C20, C30

    let mat = HyperelasticMaterial::new(params);

    println!("Material Properties:");
    println!("  Model: Yeoh (3-term)");
    println!("  C10: {:.3} MPa", 0.5e6 / 1e6);
    println!("  C20: {:.4} MPa", 0.01e6 / 1e6);
    println!("  C30: {:.5} MPa", 0.001e6 / 1e6);
    println!();

    // Uniaxial compression and tension
    println!("Uniaxial Test (Compression & Tension):");
    println!("  {:>15} | {:>15}", "Stretch λ", "W (J/m³)");
    println!("  {}", "-".repeat(35));

    let stretches = vec![0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8];

    for lambda in stretches {
        let lambda_lat = (1.0 / lambda).sqrt();
        let f = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(
            &[lambda, lambda_lat, lambda_lat]
        ));

        let w = mat.strain_energy(&f);
        println!("  {:>15.3} | {:>15.2}", lambda, w);
    }
    println!();

    Ok(())
}

/// Example 5: Arruda-Boyce material model.
fn example_arruda_boyce() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 5: Arruda-Boyce Model (8-Chain)");
    println!("{}\n", "=".repeat(60));

    // Create Arruda-Boyce material
    let mut params = HyperelasticParams::default();
    params.model_type = HyperelasticModelType::ArrudaBoyce;
    params.mu_AB = 1.0e6;  // Initial shear modulus
    params.lambda_m = 3.0; // Locking stretch

    let mat = HyperelasticMaterial::new(params);

    println!("Material Properties:");
    println!("  Model: Arruda-Boyce");
    println!("  Initial shear modulus: {:.2} MPa", 1.0e6 / 1e6);
    println!("  Locking stretch (λ_m): {:.1}", 3.0);
    println!();

    // Uniaxial tension approaching lock-up
    println!("Uniaxial Tension (approaching lock-up):");
    println!("  {:>15} | {:>15}", "Stretch λ", "W (J/m³)");
    println!("  {}", "-".repeat(35));

    let stretches = vec![1.0, 1.5, 2.0, 2.3, 2.5, 2.7];

    for lambda in stretches {
        let lambda_lat = (1.0 / lambda).sqrt();
        let f = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(
            &[lambda, lambda_lat, lambda_lat]
        ));

        let w = mat.strain_energy(&f);
        println!("  {:>15.3} | {:>15.2}", lambda, w);
    }
    println!();

    Ok(())
}

/// Example 6: Material model comparison.
fn example_material_comparison() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("Example 6: Material Model Comparison");
    println!("{}\n", "=".repeat(60));

    println!("Comparison at λ = 1.5 (Uniaxial Tension):\n");
    println!("  {:<20} | {:>15} | {:>15}", "Model", "W (J/m³)", "PK2 Stress");
    println!("  {}", "-".repeat(55));

    let lambda = 1.5;
    let lambda_lat = (1.0 / lambda).sqrt();
    let f = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(
        &[lambda, lambda_lat, lambda_lat]
    ));

    // Neo-Hookean
    let params_nh = HyperelasticParams::neo_hookean(0.5e6, 1e9);
    let mat_nh = HyperelasticMaterial::new(params_nh);
    let w_nh = mat_nh.strain_energy(&f);
    let s_nh = mat_nh.stress_pk2(&f);
    println!("  {:<20} | {:>15.2} | {:>15.2}", "Neo-Hookean", w_nh, s_nh[(0, 0)]);

    // Mooney-Rivlin
    let params_mr = HyperelasticParams::mooney_rivlin(0.4e6, 0.1e6, 1e9);
    let mat_mr = HyperelasticMaterial::new(params_mr);
    let w_mr = mat_mr.strain_energy(&f);
    let s_mr = mat_mr.stress_pk2(&f);
    println!("  {:<20} | {:>15.2} | {:>15.2}", "Mooney-Rivlin", w_mr, s_mr[(0, 0)]);

    // Ogden (1-term for comparison)
    let mut params_og = HyperelasticParams::default();
    params_og.model_type = HyperelasticModelType::Ogden;
    params_og.ogden_params = vec![(0.8e6, 1.3)];
    let mat_og = HyperelasticMaterial::new(params_og);
    let w_og = mat_og.strain_energy(&f);
    println!("  {:<20} | {:>15.2} | {:>15}", "Ogden (1-term)", w_og, "-");

    // Yeoh
    let mut params_ye = HyperelasticParams::default();
    params_ye.model_type = HyperelasticModelType::Yeoh;
    params_ye.yeoh_params = vec![0.5e6];
    let mat_ye = HyperelasticMaterial::new(params_ye);
    let w_ye = mat_ye.strain_energy(&f);
    println!("  {:<20} | {:>15.2} | {:>15}", "Yeoh (1-term)", w_ye, "-");

    // Arruda-Boyce
    let mut params_ab = HyperelasticParams::default();
    params_ab.model_type = HyperelasticModelType::ArrudaBoyce;
    params_ab.mu_AB = 0.5e6;
    params_ab.lambda_m = 3.0;
    let mat_ab = HyperelasticMaterial::new(params_ab);
    let w_ab = mat_ab.strain_energy(&f);
    println!("  {:<20} | {:>15.2} | {:>15}", "Arruda-Boyce", w_ab, "-");

    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperelastic_examples() {
        example_neo_hookean().unwrap();
        example_mooney_rivlin().unwrap();
        example_ogden().unwrap();
        example_yeoh().unwrap();
        example_arruda_boyce().unwrap();
        example_material_comparison().unwrap();
    }
}
