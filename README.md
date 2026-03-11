# fea-rs
Finite Element Analysis library in Rust

## What’s included (so far)

- **Linear static truss/bar solver**: 2-node truss element (`Truss2`) with 3 displacement DOFs per node (Ux, Uy, Uz).
- **Visualization exporters**:
  - VTK legacy (`.vtk`) for ParaView
  - JSON (`web/model.json`) for the built-in HTML/WebGL viewer

## Quick start

Run the example that solves a 2-node bar and writes outputs:

```bash
cargo run --example truss_bar
```

### ParaView (VTK)

Open `out/truss_bar.vtk` in ParaView.

### Browser (HTML + WebGL)

The example writes `web/model.json`. Serve the `web/` folder and open `index.html`:

```bash
cd web
python3 -m http.server 8000
```

Then open `http://localhost:8000/` and you should see undeformed/deformed geometry with a deformation scale slider.
