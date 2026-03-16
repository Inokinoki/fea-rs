//! Material and section properties for finite element analysis.

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

/// Section properties for structural elements.
#[derive(Debug, Clone, Copy)]
pub struct Section {
    /// Section name.
    pub name: &'static str,
    /// Cross-sectional area (m^2).
    pub area: f64,
    /// Area moment of inertia about z-axis (m^4).
    pub i_z: f64,
    /// Area moment of inertia about y-axis (m^4).
    pub i_y: f64,
    /// Torsional constant (m^4).
    pub j: f64,
}

impl Section {
    /// Creates a new section.
    pub const fn new(
        name: &'static str,
        area: f64,
        i_z: f64,
        i_y: f64,
        j: f64,
    ) -> Self {
        Self {
            name,
            area,
            i_z,
            i_y,
            j,
        }
    }

    /// Creates a circular section.
    pub fn circular(name: &'static str, diameter: f64) -> Self {
        let r = diameter / 2.0;
        let area = std::f64::consts::PI * r * r;
        let i = std::f64::consts::PI * r.powi(4) / 4.0;
        Self {
            name,
            area,
            i_z: i,
            i_y: i,
            j: std::f64::consts::PI * r.powi(4) / 2.0,
        }
    }

    /// Creates a rectangular section.
    pub fn rectangular(name: &'static str, width: f64, height: f64) -> Self {
        let area = width * height;
        let i_z = width * height.powi(3) / 12.0;
        let i_y = height * width.powi(3) / 12.0;
        // Approximate torsional constant for rectangle
        let j = if width > height {
            width * height.powi(3) * (16.0 / 3.0 - 3.36 * height / width * (1.0 - height.powi(4) / (12.0 * width.powi(4))))
        } else {
            height * width.powi(3) * (16.0 / 3.0 - 3.36 * width / height * (1.0 - width.powi(4) / (12.0 * height.powi(4))))
        };
        Self {
            name,
            area,
            i_z,
            i_y,
            j,
        }
    }
}

// Preset materials
pub const STEEL_A36: Material = Material::new(
    "Steel A36",
    200e9,
    0.26,
    7850.0,
    250e6,
);

pub const STAINLESS_STEEL_304: Material = Material::new(
    "Stainless Steel 304",
    193e9,
    0.29,
    8000.0,
    215e6,
);

pub const ALUMINUM_6061_T6: Material = Material::new(
    "Aluminum 6061-T6",
    68.9e9,
    0.33,
    2700.0,
    276e6,
);

pub const ALUMINUM_7075_T6: Material = Material::new(
    "Aluminum 7075-T6",
    71.7e9,
    0.33,
    2810.0,
    503e6,
);

pub const TITANIUM_TI6AL4V: Material = Material::new(
    "Titanium Ti-6Al-4V",
    113.8e9,
    0.342,
    4430.0,
    880e6,
);
