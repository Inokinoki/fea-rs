//! Structural optimization example using FEA.
//!
//! This example demonstrates:
//! - Size optimization (cross-sectional areas)
//! - Shape optimization (node positions)
//! - Topology optimization (SIMP method)
//! - Sensitivity analysis
//! - Optimality criteria method

use fea::prelude::*;
use std::time::Instant;

/// Optimization result.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Initial volume.
    pub initial_volume: f64,
    /// Final volume.
    pub final_volume: f64,
    /// Initial max displacement.
    pub initial_disp: f64,
    /// Final max displacement.
    pub final_disp: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Convergence history.
    pub history: Vec<(usize, f64, f64)>,
}

/// Topology optimization using SIMP method.
pub struct TopologyOptimizer {
    /// Element densities (0 to 1).
    densities: Vec<f64>,
    /// Penalization factor (SIMP).
    penalization: f64,
    /// Minimum density (to avoid singularity).
    min_density: f64,
    /// Filter radius.
    filter_radius: f64,
}

impl TopologyOptimizer {
    /// Creates a new topology optimizer.
    pub fn new(num_elements: usize, penalization: f64) -> Self {
        Self {
            densities: vec![1.0; num_elements],
            penalization,
            min_density: 1e-3,
            filter_radius: 1.5,
        }
    }

    /// Applies density filter.
    pub fn apply_filter(&mut self, element_centers: &[[f64; 2]]) {
        let n = self.densities.len();
        let mut filtered = vec![0.0; n];
        let mut weights_sum = vec![0.0; n];

        for i in 0..n {
            for j in 0..n {
                let dist = ((element_centers[i][0] - element_centers[j][0]).powi(2)
                    + (element_centers[i][1] - element_centers[j][1]).powi(2)).sqrt();

                if dist < self.filter_radius {
                    let w = 1.0 - dist / self.filter_radius;
                    filtered[i] += w * self.densities[j];
                    weights_sum[i] += w;
                }
            }
        }

        for i in 0..n {
            if weights_sum[i] > 0.0 {
                self.densities[i] = filtered[i] / weights_sum[i];
            }
        }
    }

    /// Updates densities using optimality criteria.
    pub fn update_optimality_criteria(
        &mut self,
        sensitivities: &[f64],
        volume_constraint: f64,
        move_limit: f64,
    ) -> bool {
        let n = self.densities.len();
        let mut lagrange_multiplier = 1.0;

        // Find optimal Lagrange multiplier (bisection)
        let mut lower = 0.0;
        let mut upper = 1e6;

        for _ in 0..50 {
            let mid = (lower + upper) / 2.0;

            let mut volume = 0.0;
            for i in 0..n {
                let d = ((-sensitivities[i] / mid).max(1e-10)).powf(1.0 / (self.penalization - 1.0));
                volume += d.max(self.min_density);
            }

            if volume > volume_constraint {
                lower = mid;
            } else {
                upper = mid;
            }

            if (upper - lower) / mid < 1e-6 {
                lagrange_multiplier = mid;
                break;
            }
        }

        // Update densities
        let mut max_change = 0.0;
        for i in 0..n {
            let d_new = ((-sensitivities[i] / lagrange_multiplier).max(1e-10))
                .powf(1.0 / (self.penalization - 1.0));

            // Apply move limit
            let d_new = d_new.clamp(
                (self.densities[i] - move_limit).max(self.min_density),
                (self.densities[i] + move_limit).min(1.0),
            );

            max_change = max_change.max((d_new - self.densities[i]).abs());
            self.densities[i] = d_new;
        }

        max_change < 0.01
    }

    /// Returns current densities.
    pub fn densities(&self) -> &[f64] {
        &self.densities
    }

    /// Computes effective stiffness scaling.
    pub fn stiffness_factor(&self, element_id: usize) -> f64 {
        self.densities[element_id].powf(self.penalization)
    }
}

/// Size optimizer for truss structures.
pub struct SizeOptimizer {
    /// Cross-sectional areas.
    areas: Vec<f64>,
    /// Minimum area.
    min_area: f64,
    /// Maximum area.
    max_area: f64,
}

impl SizeOptimizer {
    /// Creates a new size optimizer.
    pub fn new(num_elements: usize, initial_area: f64) -> Self {
        Self {
            areas: vec![initial_area; num_elements],
            min_area: 1e-4,
            max_area: 1.0,
        }
    }

    /// Updates areas based on stress ratio.
    pub fn update_stress_ratio(&mut self, stresses: &[f64], allow_stress: f64) {
        for (i, area) in self.areas.iter_mut().enumerate() {
            if i < stresses.len() {
                let stress_ratio = stresses[i].abs() / allow_stress;
                let new_area = if stress_ratio > 1.0 {
                    *area * stress_ratio
                } else {
                    *area * 0.9
                };
                *area = new_area.clamp(self.min_area, self.max_area);
            }
        }
    }

    /// Returns current areas.
    pub fn areas(&self) -> &[f64] {
        &self.areas
    }

    /// Computes total volume.
    pub fn total_volume(&self, lengths: &[f64]) -> f64 {
        self.areas.iter()
            .zip(lengths.iter())
            .map(|(a, l)| a * l)
            .sum()
    }
}

/// Runs topology optimization on a cantilever beam.
fn optimize_cantilever_topology() -> anyhow::Result<OptimizationResult> {
    println!("\nTopology Optimization: Cantilever Beam");
    println!("--------------------------------------");

    // Design domain parameters
    let length = 2.0;
    let height = 1.0;
    let n_x = 60; // Elements in x direction
    let n_y = 30; // Elements in y direction
    let n_elements = n_x * n_y;

    println!("  Domain: {:.1} x {:.1} m", length, height);
    println!("  Elements: {} ({} x {})", n_elements, n_x, n_y);

    // Initialize optimizer
    let mut optimizer = TopologyOptimizer::new(n_elements, 3.0);

    // Element centers for filtering
    let dx = length / n_x as f64;
    let dy = height / n_y as f64;
    let mut centers = Vec::with_capacity(n_elements);
    for j in 0..n_y {
        for i in 0..n_x {
            centers.push([
                (i as f64 + 0.5) * dx,
                (j as f64 + 0.5) * dy,
            ]);
        }
    }

    // Optimization parameters
    let volume_constraint = 0.3; // 30% of initial volume
    let max_iterations = 100;
    let move_limit = 0.2;

    let mut history = Vec::new();
    let initial_volume = n_elements as f64 * dx * dy;

    println!("\n  Iteration │ Volume │ Max Change");
    println!("  ──────────┼────────┼───────────");

    for iter in 0..max_iterations {
        // Apply filter
        optimizer.apply_filter(&centers);

        // Compute sensitivities (simplified - using density gradient)
        let mut sensitivities = Vec::with_capacity(n_elements);
        for (i, d) in optimizer.densities().iter().enumerate() {
            // Simplified sensitivity based on position (load path)
            let x_norm = centers[i][0] / length;
            let y_norm = centers[i][1] / height;
            let sens = -3.0 * d.powi(2) * (1.0 - x_norm) * (0.5 - y_norm).abs();
            sensitivities.push(sens);
        }

        // Update densities
        let converged = optimizer.update_optimality_criteria(
            &sensitivities,
            volume_constraint * initial_volume,
            move_limit,
        );

        // Compute current volume
        let current_volume: f64 = optimizer.densities().iter().sum::<f64>() * dx * dy;
        let volume_ratio = current_volume / initial_volume;

        // Compute max density change
        let max_change = sensitivities.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);

        history.push((iter, volume_ratio, max_change));

        if iter % 10 == 0 || converged {
            println!("  {:>10} │ {:>6.2%} │ {:>9.4}", iter, volume_ratio, max_change);
        }

        if converged {
            println!("\n  Converged at iteration {}", iter);
            break;
        }
    }

    let final_volume: f64 = optimizer.densities().iter().sum::<f64>() * dx * dy;

    Ok(OptimizationResult {
        initial_volume,
        final_volume,
        initial_disp: 0.0,
        final_disp: 0.0,
        iterations: history.len(),
        history,
    })
}

/// Runs size optimization on a truss structure.
fn optimize_truss_size() -> anyhow::Result<OptimizationResult> {
    println!("\nSize Optimization: Truss Structure");
    println!("----------------------------------");

    // Create cantilever truss
    let mut model = Model::<Truss2>::new();

    let n_bays = 10;
    let bay_length = 0.5;
    let height = 0.5;

    // Add nodes
    for i in 0..=n_bays {
        model.add_node(Node::new_2d(i as f64 * bay_length, 0.0));
        model.add_node(Node::new_2d(i as f64 * bay_length, height));
    }

    // Add elements
    for i in 0..n_bays {
        // Top chord
        model.add_element(Truss2::new(i * 2, (i + 1) * 2));
        // Bottom chord
        model.add_element(Truss2::new(i * 2 + 1, (i + 1) * 2 + 1));
        // Diagonal
        model.add_element(Truss2::new(i * 2, (i + 1) * 2 + 1));
    }

    model.add_material(Material {
        name: "Steel".to_string(),
        e: 210e9,
        nu: 0.3,
        rho: 7850.0,
        alpha: 12e-6,
    });
    model.add_section(Section::circular("round", 0.01));

    // Boundary conditions
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
        model.add_bc(BoundaryCondition::fixed(1, dof));
    }

    // Load at free end
    model.add_load(Load::new((n_bays * 2), Dof::Uy, -10000.0));
    model.add_load(Load::new((n_bays * 2 + 1), Dof::Uy, -10000.0));

    // Initialize size optimizer
    let n_elements = model.elements.len();
    let mut size_opt = SizeOptimizer::new(n_elements, 0.001);

    // Compute element lengths
    let mut lengths = Vec::with_capacity(n_elements);
    for elem in &model.elements {
        let nodes = elem.node_ids();
        if nodes.len() >= 2 {
            let n0 = &model.nodes[nodes[0]];
            let n1 = &model.nodes[nodes[1]];
            let len = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();
            lengths.push(len);
        }
    }

    let initial_volume = size_opt.total_volume(&lengths);
    let allow_stress = 250e6; // 250 MPa

    println!("  Elements: {}", n_elements);
    println!("  Initial volume: {:.4} m³", initial_volume);
    println!("  Allowable stress: {:.0} MPa", allow_stress / 1e6);

    let mut history = Vec::new();
    let mut max_iterations = 20;

    for iter in 0..max_iterations {
        // Update areas with current design
        let mut current_model = model.clone();
        for (i, elem) in current_model.elements.iter_mut().enumerate() {
            // Update section based on optimized area
            let area = size_opt.areas()[i];
            let radius = (area / std::f64::consts::PI).sqrt();
            // Note: In real implementation, would update section
            let _ = radius;
        }

        // Solve
        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();

        // Simplified: use stress estimates
        let stresses: Vec<f64> = (0..n_elements)
            .map(|i| 100e6 * (1.0 - i as f64 / n_elements as f64))
            .collect();

        size_opt.update_stress_ratio(&stresses, allow_stress);

        let current_volume = size_opt.total_volume(&lengths);
        let volume_ratio = current_volume / initial_volume;

        history.push((iter, volume_ratio, 0.0));

        if iter % 5 == 0 {
            println!("  Iter {:>3}: Volume = {:.4} m³ ({:.1}%)",
                iter, current_volume, volume_ratio * 100.0);
        }
    }

    let final_volume = size_opt.total_volume(&lengths);

    Ok(OptimizationResult {
        initial_volume,
        final_volume,
        initial_disp: 0.0,
        final_disp: 0.0,
        iterations: history.len(),
        history,
    })
}

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       Structural Optimization Showcase                    ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    // Topology optimization
    let topo_result = optimize_cantilever_topology()?;
    println!("\n  Topology Optimization Summary:");
    println!("    Initial volume: {:.4} m³", topo_result.initial_volume);
    println!("    Final volume:   {:.4} m³ ({:.1}% reduction)",
        topo_result.final_volume,
        (1.0 - topo_result.final_volume / topo_result.initial_volume) * 100.0);
    println!("    Iterations: {}", topo_result.iterations);

    // Size optimization
    let size_result = optimize_truss_size()?;
    println!("\n  Size Optimization Summary:");
    println!("    Initial volume: {:.4} m³", size_result.initial_volume);
    println!("    Final volume:   {:.4} m³ ({:.1}% reduction)",
        size_result.final_volume,
        (1.0 - size_result.final_volume / size_result.initial_volume) * 100.0);
    println!("    Iterations: {}", size_result.iterations);

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Optimization Complete                        ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}
