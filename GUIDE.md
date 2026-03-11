# fea-rs: Comprehensive Guide

## Overview

`fea-rs` is a Finite Element Analysis (FEA) library written in Rust, focused on structural analysis of truss and beam systems. It provides linear static analysis, modal analysis, iterative solvers, and visualization utilities.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Building Models](#building-models)
4. [Solvers](#solvers)
5. [Modal Analysis](#modal-analysis)
6. [Beam Elements](#beam-elements)
7. [Materials](#materials)
8. [Post-Processing](#post-processing)
9. [Visualization](#visualization)
10. [Parametric Studies](#parametric-studies)
11. [API Reference](#api-reference)

---

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
fea = { git = "https://github.com/Inokinoki/fea-rs" }
anyhow = "1"
```

### Hello World: 2-Node Truss

```rust
use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    // Create model
    let mut model = Model::<Truss2>::new();

    // Add nodes
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(2.0, 0.0));

    // Add element (E=210GPa, A=1e-4 m²)
    model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));

    // Boundary conditions: fix n0 completely
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition { node: n0, dof, value: 0.0 });
    }
    // Constrain n1 in Y and Z
    for dof in [Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition { node: n1, dof, value: 0.0 });
    }

    // Apply load: 10kN in X direction at n1
    model.add_load(Load { node: n1, dof: Dof::Ux, value: 10_000.0 });

    // Solve
    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    // Get displacement at n1
    let u1 = result.u[model.dof_index(n1, Dof::Ux).unwrap()];
    println!("Displacement at n1: {:.6} m", u1);

    Ok(())
}
```

---

## Core Concepts

### Degrees of Freedom (DOF)

Each node has 3 translational DOFs in 3D space:
- `Dof::Ux` - Displacement in X
- `Dof::Uy` - Displacement in Y
- `Dof::Uz` - Displacement in Z

### Node

```rust
// 2D node (z=0 automatically)
let node_2d = Node::new_2d(1.0, 2.0);

// 3D node
let node_3d = Node::new_3d(1.0, 2.0, 3.0);

// Access coordinates
let coords = node_3d.as_array(); // [f64; 3]
```

### Model

The `Model<E>` struct holds:
- Nodes and their coordinates
- Elements (truss, beam, etc.)
- Boundary conditions
- Applied loads

```rust
let mut model = Model::<Truss2>::new();
let node_id = model.add_node(node);
model.add_element(element);
model.add_bc(bc);
model.add_load(load);
model.build_dofs_3d(); // Required before solving
```

---

## Building Models

### Truss Elements (Truss2)

2-node truss element carrying only axial load:

```rust
use fea::elements::Truss2;

let element = Truss2::new(
    node1_id,
    node2_id,
    210e9,  // Young's modulus (Pa)
    1e-4,   // Cross-sectional area (m²)
);
```

**Methods:**
- `length_and_dir(&model)` - Returns (length, direction cosines)
- `axial_stress(&model, &displacements)` - Returns stress (Pa)

### Beam Elements (Beam2D)

2D Euler-Bernoulli beam with rotational DOFs:

```rust
use fea::beam::{Beam2D, BeamModel, assemble_beam_stiffness};

let mut model = BeamModel::new();
let n0 = model.add_node([0.0, 0.0]);
let n1 = model.add_node([1.0, 0.0]);

model.add_element(Beam2D::new(
    n0, n1,
    210e9,   // E (Pa)
    1e-4,    // A (m²)
    1e-8,    // I (m⁴)
));

let k = assemble_beam_stiffness(&model, 6); // 6 DOFs total
```

---

## Solvers

### Linear Static Solver (Direct)

Dense matrix solver using LU decomposition:

```rust
use fea::solver::{LinearStaticSolver, LinearStaticResult};

let solver = LinearStaticSolver::new();
let result: LinearStaticResult = solver.solve_truss2(&mut model)?;

// Access results
let displacements: &[f64] = &result.u;
let reactions: &BTreeMap<usize, f64> = &result.reactions;
```

### Conjugate Gradient (CG)

Iterative solver for larger systems:

```rust
use fea::solvers::{ConjugateGradient, ConjugateGradientConfig, Preconditioner};

let config = ConjugateGradientConfig {
    max_iterations: 1000,
    tolerance: 1e-10,
    preconditioner: Preconditioner::Jacobi,
    anderson_acceleration: Some(3), // Optional acceleration
};

let solver = ConjugateGradient::with_config(config);
let result = solver.solve_truss2(&mut model)?;

println!("Iterations: {}", result.iterations);
println!("Converged: {}", result.converged);
```

### Preconditioned CG (PCG)

Flexible CG with multiple preconditioner options:

```rust
use fea::solvers::{PCG, Preconditioner};

let pcg = PCG::with_preconditioner(Preconditioner::Jacobi);
let (x, iterations, residual, converged) = pcg.solve(&k_matrix, &f_vector);
```

### GMRES

For non-symmetric systems:

```rust
use fea::solvers::GMRES;

let gmres = GMRES::with_restart(30); // Restart every 30 iterations
let (x, iterations, residual, converged) = gmres.solve(&k_matrix, &f_vector);
```

### Gauss-Seidel / SOR

Simple iterative solver:

```rust
use fea::solvers::GaussSeidel;

// Standard Gauss-Seidel
let gs = GaussSeidel::new();

// Successive Over-Relaxation (omega > 1 accelerates convergence)
let sor = GaussSeidel::with_sor(1.5);

let (x, iterations, residual, converged) = gs.solve(&k_matrix, &f_vector);
```

---

## Modal Analysis

Compute natural frequencies and mode shapes:

```rust
use fea::modal::{ModalSolver, ModalConfig, MassFormulation};

let config = ModalConfig {
    num_modes: 5,
    mass_formulation: MassFormulation::Lumped, // or Consistent
    max_iterations: 1000,
    tolerance: 1e-10,
};

let solver = ModalSolver::with_config(config);
let result = solver.analyze_truss2(&mut model)?;

// Access results
for (i, (freq, shape)) in result.frequencies.iter()
    .zip(result.mode_shapes.iter()).enumerate() {
    let freq_hz = freq / (2.0 * std::f64::consts::PI);
    println!("Mode {}: {:.2} Hz", i + 1, freq_hz);
}

// Export for web visualization
result.write_json("modes.json")?;
```

---

## Materials

Predefined material properties:

```rust
use fea::materials::{
    STEEL_A36, ALUMINUM_6061_T6, TITANIUM_TI6AL4V,
    Material, get_material_by_name, all_preset_materials
};

// Use preset
let e = STEEL_A36.young_modulus;
let rho = STEEL_A36.density;
let sigma_y = STEEL_A36.yield_strength;

// Derived properties
let g = STEEL_A36.shear_modulus(); // G = E/(2(1+ν))
let k = STEEL_A36.bulk_modulus();  // K = E/(3(1-2ν))

// Lookup by name
let mat = get_material_by_name("aluminum").unwrap();

// List all materials
for mat in all_preset_materials() {
    println!("{}: E={:.1} GPa", mat.name, mat.young_modulus / 1e9);
}
```

**Available Materials:**

| Material | E (GPa) | ρ (kg/m³) | σy (MPa) |
|----------|---------|-----------|----------|
| Steel A36 | 200 | 7850 | 250 |
| Stainless 304 | 193 | 8000 | 215 |
| Aluminum 6061-T6 | 68.9 | 2700 | 276 |
| Aluminum 7075-T6 | 71.7 | 2810 | 503 |
| Titanium Ti-6Al-4V | 113.8 | 4430 | 880 |
| Copper | 110 | 8960 | 33 |
| Cast Iron | 100 | 7150 | 150 |
| Concrete | 25 | 2400 | 3 |
| Carbon Fiber UD | 135 | 1600 | 1500 |

---

## Post-Processing

### Extract Results

```rust
use fea::postprocessing::{
    extract_nodal_displacements, extract_element_results,
    compute_reactions, ResultStatistics
};

let displacements = extract_nodal_displacements(&model, &result.u);
let elem_results = extract_element_results(&model, &result.u);
let reactions = compute_reactions(&model, &k_matrix, &result.u, &result.f);

// Statistics
let stats = ResultStatistics::from_results(&displacements, &elem_results);
println!("Max displacement: {:.6} m at node {}", stats.max_displacement, stats.max_displacement_node);
println!("Max stress: {:.2} MPa", stats.max_stress / 1e6);
```

### CSV Export

```rust
use fea::postprocessing::CsvWriter;

let writer = CsvWriter::new();
writer.write_displacements("displacements.csv", &displacements)?;
writer.write_element_results("elements.csv", &elem_results)?;
writer.write_reactions("reactions.csv", &reactions)?;
```

### Convergence History

```rust
use fea::postprocessing::ConvergenceHistory;

let mut history = ConvergenceHistory::new();
history.record(1, 1.0);
history.record(2, 0.1);
history.record(3, 0.01);

// Export
history.write_csv("convergence.csv")?;
history.write_svg("convergence.svg")?; // Visual plot
```

---

## Visualization

### VTK Export (ParaView)

```rust
use fea::viz::{VtkLegacyWriter, VizConfig};

let writer = VtkLegacyWriter::new();

// Basic export
writer.write_truss2("output.vtk", &model, &result.u)?;

// With auto-scaled deformation
let config = VizConfig::with_auto_scale(&model, &result.u, 0.2);
writer.write_truss2_with_states("output.vtk", &model, &result.u, &config)?;
```

### JSON Export (Web Viewer)

```rust
use fea::viz::JsonWriter;

let writer = JsonWriter::new();

// Basic export
writer.write_truss2("web/model.json", &model, &result.u)?;

// Enhanced export (includes deformed shape + metadata)
let config = VizConfig::default();
writer.write_truss2_enhanced("web/model.json", &model, &result.u, &config)?;
```

### Web Viewer

Serve the `web/` directory:

```bash
cd web
python3 -m http.server 8000
```

Open `http://localhost:8000` to view:
- Undeformed/deformed geometry toggle
- Stress-colored elements
- Displacement magnitude coloring
- Modal animation (if modes exported)

---

## Parametric Studies

### Single Variable Sweep

```rust
use fea::parametric::{ParametricStudy, ParametricVariable};

let study = ParametricStudy::new()
    .add_variable(ParametricVariable::range("area", 1e-4, 5e-4, 10))
    .with_objective("max_displacement", true); // minimize

let result = study.run(|params| {
    let area = params.get("area").unwrap();
    // Build model with this area...
    model
});

println!("Best design: area={:.2e}", result.best_design.unwrap().parameters["area"]);
```

### Multiple Variables (Full Factorial)

```rust
let study = ParametricStudy::new()
    .add_variable(ParametricVariable::range("area", 1e-4, 3e-4, 3))
    .add_variable(ParametricVariable::range("load", 1000.0, 5000.0, 3))
    .with_objective("mass", true);

let result = study.run(|params| {
    // params contains both "area" and "load"
    build_model(params)
});

println!("Total runs: {}", result.num_runs);
```

### Batch Load Cases

```rust
use fea::parametric::{LoadCase, run_batch_analysis, export_batch_results_csv};

let load_cases = vec![
    LoadCase::new("dead_load")
        .add_load(Load { node: n1, dof: Dof::Uz, value: -5000.0 }),
    LoadCase::new("wind_load")
        .add_load(Load { node: n1, dof: Dof::Ux, value: 2000.0 }),
    LoadCase::new("combined")
        .add_load(Load { node: n1, dof: Dof::Uz, value: -5000.0 })
        .add_load(Load { node: n1, dof: Dof::Ux, value: 2000.0 }),
];

let results = run_batch_analysis(&mut model, &load_cases);
export_batch_results_csv("batch_results.csv", &results)?;
```

---

## API Reference

### Module Structure

```
fea/
├── core          - Model, Node, Dof, BoundaryCondition, Load
├── elements      - Truss2 element
├── beam          - Beam2D element, BeamModel
├── solver        - LinearStaticSolver (direct)
├── solvers       - CG, PCG, GMRES, GaussSeidel
├── sparse        - CSR matrix, SparseCG
├── modal         - ModalSolver, ModalConfig
├── materials     - Material presets
├── postprocessing - CSV/SVG export, statistics
├── parametric    - ParametricStudy, LoadCase, batch analysis
└── viz           - VTK/JSON export, VizConfig
```

### Common Imports

```rust
use fea::prelude::*;  // Most common types

// Or specific:
use fea::core::{Model, Node, Dof, BoundaryCondition, Load};
use fea::elements::Truss2;
use fea::solver::LinearStaticSolver;
use fea::materials::STEEL_A36;
```

### Error Handling

All solver methods return `anyhow::Result`:

```rust
match solver.solve_truss2(&mut model) {
    Ok(result) => { /* use result */ }
    Err(e) => eprintln!("Analysis failed: {}", e),
}

// Common errors:
// - "matrix is singular" - insufficient boundary conditions
// - "dimension mismatch" - model/DOF inconsistency
// - "LU solve failed" - numerical issues
```

---

## Examples

Run included examples:

```bash
cargo run --example truss_bar      # Basic 2-node truss
cargo run --example beam_cantilever # 2D beam analysis
cargo run --example tower_3d        # 3D truss tower
```

---

## Testing

Run all tests:

```bash
cargo test
```

Test coverage spans:
- Core types (Node, Model, DOF)
- Element formulations (stiffness, stress)
- Solver accuracy and convergence
- Modal analysis
- Export formats
- Edge cases (zero-length, singular matrices, etc.)

---

## Performance Tips

1. **For small models (<1000 DOFs):** Use `LinearStaticSolver` (direct)
2. **For large models:** Use `ConjugateGradient` with `Preconditioner::Jacobi`
3. **For non-symmetric systems:** Use `GMRES`
4. **Enable Anderson acceleration** for faster CG convergence
5. **Use `MassFormulation::Lumped`** for faster modal analysis

---

## License

Apache 2.0
