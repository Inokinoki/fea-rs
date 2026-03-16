//! NAFEMS benchmark validation tests.
//!
//! This module provides standard benchmark problems from NAFEMS
//! for validating FEA implementations.

use crate::core::{BoundaryCondition, Dof, Load, Model, Node};
use crate::elements_legacy::Truss2 as Truss2;
use crate::solver::LinearStaticSolver;
use nalgebra::{DMatrix, DVector};

/// NAFEMS LE1: Plate with a circular hole under tension.
///
/// Reference: NAFEMS Benchmark Tests No. LE1
/// Problem: Infinite plate with circular hole under uniaxial tension
/// Expected: Stress concentration factor Kt = 3.0 at hole edge
pub struct NafemsLe1 {
    /// Plate width (m)
    pub width: f64,
    /// Plate height (m)
    pub height: f64,
    /// Hole radius (m)
    pub hole_radius: f64,
    /// Plate thickness (m)
    pub thickness: f64,
    /// Young's modulus (Pa)
    pub e: f64,
    /// Poisson's ratio
    pub nu: f64,
    /// Applied stress (Pa)
    pub applied_stress: f64,
}

impl Default for NafemsLe1 {
    fn default() -> Self {
        Self {
            width: 2.0,
            height: 2.0,
            hole_radius: 0.25,
            thickness: 0.01,
            e: 210e9,
            nu: 0.3,
            applied_stress: 1e6, // 1 MPa
        }
    }
}

impl NafemsLe1 {
    /// Creates the benchmark model using truss approximation.
    pub fn create_model(&self) -> Model<Truss2> {
        let mut model = Model::<Truss2>::new();

        // Create a simplified truss mesh representing the plate
        // This is a simplified representation - real benchmark uses 2D elements

        let n_elems_x = 20;
        let n_elems_y = 20;
        let dx = self.width / n_elems_x as f64;
        let dy = self.height / n_elems_y as f64;

        // Create nodes
        let mut node_ids = Vec::new();
        for j in 0..=n_elems_y {
            let mut row = Vec::new();
            for i in 0..=n_elems_x {
                let x = i as f64 * dx - self.width / 2.0;
                let y = j as f64 * dy - self.height / 2.0;

                // Skip nodes inside the hole
                if x * x + y * y < self.hole_radius * self.hole_radius * 0.9 {
                    row.push(None);
                    continue;
                }

                let nid = model.add_node(Node::new_2d(x, y));
                row.push(Some(nid));
            }
            node_ids.push(row);
        }

        // Create elements (horizontal and vertical trusses)
        let area = self.thickness * dx.min(dy);

        for j in 0..n_elems_y {
            for i in 0..n_elems_x {
                if let (Some(n1), Some(n2)) = (node_ids[j][i], node_ids[j][i + 1]) {
                    model.add_element(Truss2::new(n1, n2, self.e, area));
                }
                if let (Some(n1), Some(n2)) = (node_ids[j][i], node_ids[j + 1][i]) {
                    model.add_element(Truss2::new(n1, n2, self.e, area));
                }
            }
        }

        // Apply boundary conditions
        let bottom_left = node_ids[0][0].unwrap();
        let bottom_right = node_ids[0][n_elems_x].unwrap();

        // Fix bottom left corner in X and Y
        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(bottom_left, dof));
        }
        // Fix bottom right corner in Y only (roller)
        model.add_bc(BoundaryCondition::fixed(bottom_right, Dof::Uy));
        model.add_bc(BoundaryCondition::fixed(bottom_right, Dof::Uz));

        // Apply tensile load on right edge
        let load_per_node = self.applied_stress * self.thickness * dy;
        for j in 0..=n_elems_y {
            if let Some(nid) = node_ids[j].last().unwrap() {
                model.add_load(Load::new(*nid, Dof::Ux, load_per_node));
            }
        }

        model
    }

    /// Runs the benchmark and returns results.
    pub fn run(&self) -> anyhow::Result<NafemsLe1Result> {
        let mut model = self.create_model();

        let solver = LinearStaticSolver::new();
        let result = solver.solve_truss2(&mut model)?;

        // Find maximum displacement and stress
        let mut max_disp: f64 = 0.0;
        for &d in &result.u {
            max_disp = max_disp.max(d.abs());
        }

        let mut max_stress: f64 = 0.0;
        for elem in &model.elements {
            let stress = elem.axial_stress(&model, &result.u).abs();
            max_stress = max_stress.max(stress);
        }

        // Compute theoretical values
        let theoretical_disp = self.applied_stress * self.width / self.e;
        let theoretical_stress = self.applied_stress; // Nominal stress
        let stress_concentration = 3.0; // Theoretical Kt for infinite plate

        Ok(NafemsLe1Result {
            max_displacement: max_disp,
            max_stress,
            theoretical_displacement: theoretical_disp,
            theoretical_stress,
            stress_concentration_factor: max_stress / theoretical_stress,
            theoretical_kt: stress_concentration,
            displacement_error: (max_disp - theoretical_disp).abs() / theoretical_disp * 100.0,
        })
    }
}

/// Results from NAFEMS LE1 benchmark.
#[derive(Debug, Clone)]
pub struct NafemsLe1Result {
    /// Maximum displacement (m)
    pub max_displacement: f64,
    /// Maximum stress (Pa)
    pub max_stress: f64,
    /// Theoretical displacement (m)
    pub theoretical_displacement: f64,
    /// Theoretical nominal stress (Pa)
    pub theoretical_stress: f64,
    /// Computed stress concentration factor
    pub stress_concentration_factor: f64,
    /// Theoretical stress concentration factor
    pub theoretical_kt: f64,
    /// Displacement error (%)
    pub displacement_error: f64,
}

impl NafemsLe1Result {
    /// Checks if the benchmark passed.
    pub fn passed(&self) -> bool {
        // Allow 10% error for truss approximation
        self.displacement_error < 10.0
    }
}

/// NAFEMS Torsion benchmark: Circular shaft under torsion.
///
/// Reference: NAFEMS Benchmark Tests
/// Problem: Circular shaft under pure torsion
/// Expected: Shear stress varies linearly with radius
pub struct NafemsTorsion {
    /// Shaft length (m)
    pub length: f64,
    /// Shaft radius (m)
    pub radius: f64,
    /// Shear modulus (Pa)
    pub g: f64,
    /// Applied torque (N·m)
    pub torque: f64,
}

impl Default for NafemsTorsion {
    fn default() -> Self {
        Self {
            length: 1.0,
            radius: 0.05,
            g: 80e9,
            torque: 1000.0,
        }
    }
}

impl NafemsTorsion {
    /// Runs the benchmark and returns results.
    pub fn run(&self) -> NafemsTorsionResult {
        // Polar moment of inertia
        let j = std::f64::consts::PI * self.radius.powi(4) / 2.0;

        // Theoretical values
        let max_shear_stress = self.torque * self.radius / j;
        let angle_of_twist = self.torque * self.length / (j * self.g);

        NafemsTorsionResult {
            polar_moment: j,
            max_shear_stress,
            angle_of_twist,
        }
    }
}

/// Results from NAFEMS Torsion benchmark.
#[derive(Debug, Clone)]
pub struct NafemsTorsionResult {
    /// Polar moment of inertia (m^4)
    pub polar_moment: f64,
    /// Maximum shear stress (Pa)
    pub max_shear_stress: f64,
    /// Angle of twist (radians)
    pub angle_of_twist: f64,
}

/// NAFEMS Beam benchmark: Simply supported beam with central load.
///
/// Reference: NAFEMS Benchmark Tests
/// Problem: Simply supported beam with central point load
/// Expected: Max deflection = PL^3/(48EI)
pub struct NafemsBeam {
    /// Beam length (m)
    pub length: f64,
    /// Beam width (m)
    pub width: f64,
    /// Beam height (m)
    pub height: f64,
    /// Young's modulus (Pa)
    pub e: f64,
    /// Applied load (N)
    pub load: f64,
}

impl Default for NafemsBeam {
    fn default() -> Self {
        Self {
            length: 1.0,
            width: 0.05,
            height: 0.1,
            e: 210e9,
            load: 10000.0,
        }
    }
}

impl NafemsBeam {
    /// Computes theoretical values.
    pub fn compute_theoretical(&self) -> NafemsBeamResult {
        let i = self.width * self.height.powi(3) / 12.0;
        let max_deflection = self.load * self.length.powi(3) / (48.0 * self.e * i);
        let max_bending_moment = self.load * self.length / 4.0;
        let max_stress = max_bending_moment * (self.height / 2.0) / i;

        NafemsBeamResult {
            moment_of_inertia: i,
            max_deflection,
            max_bending_moment,
            max_stress,
        }
    }
}

/// Results from NAFEMS Beam benchmark.
#[derive(Debug, Clone)]
pub struct NafemsBeamResult {
    /// Moment of inertia (m^4)
    pub moment_of_inertia: f64,
    /// Maximum deflection (m)
    pub max_deflection: f64,
    /// Maximum bending moment (N·m)
    pub max_bending_moment: f64,
    /// Maximum bending stress (Pa)
    pub max_stress: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nafems_torsion() {
        let torsion = NafemsTorsion::default();
        let result = torsion.run();

        // Verify polar moment
        let expected_j = std::f64::consts::PI * torsion.radius.powi(4) / 2.0;
        assert!((result.polar_moment - expected_j).abs() < 1e-15);

        // Verify max shear stress
        let expected_stress = torsion.torque * torsion.radius / result.polar_moment;
        assert!((result.max_shear_stress - expected_stress).abs() < 1e3);
    }

    #[test]
    fn test_nafems_beam() {
        let beam = NafemsBeam::default();
        let result = beam.compute_theoretical();

        // Verify moment of inertia
        let expected_i = beam.width * beam.height.powi(3) / 12.0;
        assert!((result.moment_of_inertia - expected_i).abs() < 1e-15);

        // Verify max deflection
        let expected_disp = beam.load * beam.length.powi(3) / (48.0 * beam.e * expected_i);
        assert!((result.max_deflection - expected_disp).abs() < 1e-10);

        // Verify max stress
        let expected_moment = beam.load * beam.length / 4.0;
        let expected_stress = expected_moment * (beam.height / 2.0) / expected_i;
        assert!((result.max_stress - expected_stress).abs() < 1e3);
    }

    #[test]
    #[ignore = "Requires proper 2D elements for stable mesh"]
    fn test_nafems_le1() {
        let le1 = NafemsLe1 {
            width: 1.0,
            height: 1.0,
            hole_radius: 0.1,
            thickness: 0.01,
            e: 210e9,
            nu: 0.3,
            applied_stress: 1e6,
        };

        let result = le1.run().expect("LE1 benchmark failed");

        println!("NAFEMS LE1 Results:");
        println!("  Max displacement: {:.6e} m", result.max_displacement);
        println!("  Theoretical disp: {:.6e} m", result.theoretical_displacement);
        println!("  Displacement error: {:.2}%", result.displacement_error);
        println!("  Max stress: {:.1} MPa", result.max_stress / 1e6);
        println!("  Stress concentration: {:.2}", result.stress_concentration_factor);
        println!("  Theoretical Kt: {:.1}", result.theoretical_kt);

        // Truss approximation has limited accuracy
        assert!(result.displacement_error < 15.0, "Displacement error too large: {:.2}%", result.displacement_error);
    }
}
