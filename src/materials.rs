//! Material property presets for common engineering materials.
//!
//! This module provides:
//! - Predefined material properties (Young's modulus, Poisson's ratio, density)
//! - Material selection utilities
//! - Nonlinear material models (hyperelastic, plasticity, damage)

pub mod nonlinear;
pub mod plasticity;

/// Material properties for structural analysis.
#[derive(Debug, Clone, Copy)]
pub struct Material {
    /// Material name.
    pub name: &'static str,
    /// Young's modulus (Pa).
    pub young_modulus: f64,
    /// Poisson's ratio (dimensionless).
    pub poisson_ratio: f64,
    /// Density (kg/m^3).
    pub density: f64,
    /// Yield strength (Pa).
    pub yield_strength: f64,
}

impl Material {
    /// Creates a new material.
    pub const fn new(
        name: &'static str,
        young_modulus: f64,
        poisson_ratio: f64,
        density: f64,
        yield_strength: f64,
    ) -> Self {
        Self {
            name,
            young_modulus,
            poisson_ratio,
            density,
            yield_strength,
        }
    }

    /// Shear modulus G = E / (2 * (1 + nu)).
    pub fn shear_modulus(&self) -> f64 {
        self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Bulk modulus K = E / (3 * (1 - 2*nu)).
    pub fn bulk_modulus(&self) -> f64 {
        self.young_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }
}

/// Preset material: Structural Steel (ASTM A36).
pub const STEEL_A36: Material = Material::new(
    "Steel A36",
    200e9,   // E = 200 GPa
    0.26,    // nu = 0.26
    7850.0,  // rho = 7850 kg/m^3
    250e6,   // sigma_y = 250 MPa
);

/// Preset material: Stainless Steel (304).
pub const STAINLESS_STEEL_304: Material = Material::new(
    "Stainless Steel 304",
    193e9,
    0.29,
    8000.0,
    215e6,
);

/// Preset material: Aluminum 6061-T6.
pub const ALUMINUM_6061_T6: Material = Material::new(
    "Aluminum 6061-T6",
    68.9e9,
    0.33,
    2700.0,
    276e6,
);

/// Preset material: Aluminum 7075-T6.
pub const ALUMINUM_7075_T6: Material = Material::new(
    "Aluminum 7075-T6",
    71.7e9,
    0.33,
    2810.0,
    503e6,
);

/// Preset material: Titanium Ti-6Al-4V.
pub const TITANIUM_TI6AL4V: Material = Material::new(
    "Titanium Ti-6Al-4V",
    113.8e9,
    0.342,
    4430.0,
    880e6,
);

/// Preset material: Copper (pure).
pub const COPPER_PURE: Material = Material::new(
    "Copper (pure)",
    110e9,
    0.34,
    8960.0,
    33e6,
);

/// Preset material: Brass (yellow).
pub const BRASS_YELLOW: Material = Material::new(
    "Brass (yellow)",
    105e9,
    0.35,
    8530.0,
    100e6,
);

/// Preset material: Cast Iron (gray).
pub const CAST_IRON_GRAY: Material = Material::new(
    "Cast Iron (gray)",
    100e9,
    0.26,
    7150.0,
    150e6,
);

/// Preset material: Concrete (normal strength).
pub const CONCRETE_NORMAL: Material = Material::new(
    "Concrete (normal)",
    25e9,
    0.2,
    2400.0,
    3e6, // compressive
);

/// Preset material: Wood (Douglas Fir, along grain).
pub const WOOD_DOUGLAS_FIR: Material = Material::new(
    "Wood (Douglas Fir)",
    13e9,
    0.3,
    530.0,
    50e6,
);

/// Preset material: Carbon Fiber (unidirectional, along fibers).
pub const CARBON_FIBER_UD: Material = Material::new(
    "Carbon Fiber (UD)",
    135e9,
    0.3,
    1600.0,
    1500e6,
);

/// Preset material: Nylon 6/6.
pub const NYLON_66: Material = Material::new(
    "Nylon 6/6",
    3e9,
    0.39,
    1140.0,
    80e6,
);

/// Preset material: PVC (rigid).
pub const PVC_RIGID: Material = Material::new(
    "PVC (rigid)",
    3e9,
    0.38,
    1400.0,
    50e6,
);

/// Get a material by name (case-insensitive).
pub fn get_material_by_name(name: &str) -> Option<&'static Material> {
    let name_lower = name.to_lowercase();
    match name_lower.as_str() {
        "steel" | "steel a36" | "a36" => Some(&STEEL_A36),
        "stainless" | "stainless steel" | "304" => Some(&STAINLESS_STEEL_304),
        "aluminum" | "aluminium" | "6061" | "aluminum 6061" => Some(&ALUMINUM_6061_T6),
        "aluminum 7075" | "7075" => Some(&ALUMINUM_7075_T6),
        "titanium" | "ti-6al-4v" | "ti64" => Some(&TITANIUM_TI6AL4V),
        "copper" => Some(&COPPER_PURE),
        "brass" => Some(&BRASS_YELLOW),
        "cast iron" | "iron" => Some(&CAST_IRON_GRAY),
        "concrete" => Some(&CONCRETE_NORMAL),
        "wood" | "douglas fir" => Some(&WOOD_DOUGLAS_FIR),
        "carbon fiber" | "carbon" => Some(&CARBON_FIBER_UD),
        "nylon" => Some(&NYLON_66),
        "pvc" => Some(&PVC_RIGID),
        _ => None,
    }
}

/// Returns all preset materials.
pub fn all_preset_materials() -> &'static [&'static Material] {
    &[
        &STEEL_A36,
        &STAINLESS_STEEL_304,
        &ALUMINUM_6061_T6,
        &ALUMINUM_7075_T6,
        &TITANIUM_TI6AL4V,
        &COPPER_PURE,
        &BRASS_YELLOW,
        &CAST_IRON_GRAY,
        &CONCRETE_NORMAL,
        &WOOD_DOUGLAS_FIR,
        &CARBON_FIBER_UD,
        &NYLON_66,
        &PVC_RIGID,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_shear_modulus() {
        // For steel: G = E / (2 * (1 + nu)) = 200e9 / (2 * 1.26) = 79.4e9
        let g = STEEL_A36.shear_modulus();
        assert!((g - 79.4e9).abs() < 1e9);
    }

    #[test]
    fn test_material_bulk_modulus() {
        // For steel: K = E / (3 * (1 - 2*nu)) = 200e9 / (3 * 0.48) = 138.9e9
        let k = STEEL_A36.bulk_modulus();
        assert!((k - 138.9e9).abs() < 5e9);
    }

    #[test]
    fn test_get_material_by_name() {
        assert!(get_material_by_name("steel").is_some());
        assert!(get_material_by_name("STEEL").is_some()); // case insensitive
        assert!(get_material_by_name("aluminum").is_some());
        assert!(get_material_by_name("unknown").is_none());
    }

    #[test]
    fn test_all_preset_materials() {
        let materials = all_preset_materials();
        assert!(materials.len() >= 10); // At least 10 materials

        // All should have valid properties
        for mat in materials {
            assert!(mat.young_modulus > 0.0);
            assert!(mat.poisson_ratio > 0.0 && mat.poisson_ratio < 0.5);
            assert!(mat.density > 0.0);
        }
    }

    #[test]
    fn test_material_properties_steel() {
        assert_eq!(STEEL_A36.name, "Steel A36");
        assert_eq!(STEEL_A36.young_modulus, 200e9);
        assert_eq!(STEEL_A36.poisson_ratio, 0.26);
        assert_eq!(STEEL_A36.density, 7850.0);
    }

    #[test]
    fn test_material_properties_aluminum() {
        assert!(ALUMINUM_6061_T6.young_modulus < STEEL_A36.young_modulus);
        assert!(ALUMINUM_6061_T6.density < STEEL_A36.density);
    }
}
