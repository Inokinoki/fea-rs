//! FEA Framework Quick Reference Guide.
//!
//! This file provides a comprehensive overview of all available features
//! and how to use them in your projects.

// ============================================================================
// 1. BASIC FEA ANALYSIS
// ============================================================================

use fea::prelude::*;

fn basic_static_analysis() -> anyhow::Result<()> {
    // Create model
    let mut model = Model::<Truss2>::new();

    // Add nodes
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));

    // Add material and section
    model.add_material(STEEL_A36);
    model.add_section(Section::circular("round", 0.01));

    // Add element
    model.add_element(Truss2::new(n0, n1));

    // Add boundary conditions
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uz));

    // Add load
    model.add_load(Load::new(n1, Dof::Ux, 1000.0));

    // Run analysis
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    println!("Max displacement: {:?}", result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max));

    Ok(())
}

// ============================================================================
// 2. GPU-ACCELERATED SOLVERS
// ============================================================================

use fea::gpu::{
    GPUCGSolver, GPUGMRESSolver, GPUBiCGSTABSolver, GPUPCGSolver,
    GPUILUPreconditioner, GPUCSRMatrix,
};

fn gpu_accelerated_solve() -> anyhow::Result<()> {
    // Create sparse matrix in CSR format
    let row_ptr = vec![0, 2, 4, 6];
    let col_ind = vec![0, 1, 0, 1, 1, 2];
    let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0];

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, 3, 3, 0);
    let b = vec![1.0, 1.0, 1.0];

    // CG Solver
    let cg = GPUCGSolver::new(0, 1e-8, 100);
    let mut x = vec![0.0; 3];
    let result = cg.solve(&matrix, &b, &mut x)?;

    // PCG with ILU preconditioner
    let ilu = GPUILUPreconditioner::new(&matrix, 0);
    let pcg = GPUPCGSolver::new(0, 1e-8, 100, "ILU");
    let mut x = vec![0.0; 3];
    let result = pcg.solve(&matrix, Some(&ilu), &b)?;

    // GMRES with restart
    let gmres = GPUGMRESSolver::new(0, 1e-8, 100, 30);
    let mut x = vec![0.0; 3];
    let result = gmres.solve(&matrix, &b, &mut x)?;

    Ok(())
}

// ============================================================================
// 3. MULTI-GPU PARALLEL EXECUTION
// ============================================================================

use fea::gpu::MultiGPUManager;

fn multi_gpu_analysis() -> anyhow::Result<()> {
    let mgr = MultiGPUManager::new();
    let num_gpus = mgr.num_devices();

    if num_gpus == 0 {
        println!("No GPUs available");
        return Ok(());
    }

    // Domain decomposition
    let partitions = mgr.decompose_domain(10000, 5000);

    // Parallel execution
    let results = mgr.parallel_execute(|device_id, _ctx, stream| {
        // Perform computation on each GPU
        Ok(())
    });

    // Synchronize all GPUs
    mgr.synchronize_all();

    Ok(())
}

// ============================================================================
// 4. EXPLICIT DYNAMIC ANALYSIS
// ============================================================================

use fea::algorithms::explicit_dynamics::{
    ExplicitDynamicAnalyzer, ExplicitConfig, ExplicitMethod,
};

fn explicit_dynamics() -> anyhow::Result<()> {
    let n_dofs = 100;
    let mass = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_element(n_dofs, 1.0));

    let mut stiffness = nalgebra::DMatrix::zeros(n_dofs, n_dofs);
    for i in 0..n_dofs {
        stiffness[(i, i)] = 100.0;
        if i > 0 {
            stiffness[(i, i - 1)] = -50.0;
            stiffness[(i - 1, i)] = -50.0;
        }
    }

    let config = ExplicitConfig {
        method: ExplicitMethod::CentralDifference,
        time_step: 0.001,
        total_time: 0.1,
        damping_alpha: 0.05,
        damping_beta: 0.0,
        auto_time_step: false,
        output_frequency: 10,
    };

    let analyzer = ExplicitDynamicAnalyzer::with_config(mass, stiffness, config);

    let u0 = vec![0.0; n_dofs];
    let v0 = vec![0.0; n_dofs];
    let force_fn = |_t: f64, _u: &[f64]| vec![0.0; n_dofs];

    let result = analyzer.analyze(&u0, &v0, &force_fn)?;

    println!("Completed {} time steps", result.num_steps);

    Ok(())
}

// ============================================================================
// 5. NONLINEAR MATERIAL MODELS
// ============================================================================

use fea::materials::nonlinear::{
    LinearElastic, VonMisesPlasticity, NeoHookean, JohnsonCook, DamageModel,
};

fn material_models() -> anyhow::Result<()> {
    // Linear Elastic
    let elastic = LinearElastic::new(210e9, 0.3, 7850.0);
    let strain = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0];
    let stress = elastic.stress(&strain, 0.0);

    // Von Mises Plasticity
    let plastic = VonMisesPlasticity::new(210e9, 0.3, 250e6, 1e9);
    let state = fea::materials::nonlinear::StressState::default();
    let (new_state, tangent) = plastic.update(&strain, &state);

    // Neo-Hookean Hyperelastic
    let neo = NeoHookean::from_en(10e6, 0.49);

    // Johnson-Cook Viscoplasticity
    let jc = JohnsonCook::new(
        210e9, 0.3, 500e6, 500e6, 0.02, 0.3, 1.0,
        1.0, 1800.0, 293.0,
    );
    let flow_stress = jc.flow_stress(0.1, 100.0, 500.0);

    // Damage Model
    let damage = DamageModel::new(210e9, 0.3, 0.001, 0.02);
    let d = damage.compute_damage(0.005);

    Ok(())
}

// ============================================================================
// 6. ADVANCED GPU FEATURES
// ============================================================================

use fea::gpu::gpu_advanced::{
    MixedPrecisionSolver, MultiStreamExecutor, AsyncGpuTask,
    KernelFusionOptimizer, KernelInfo, MemoryAccessPattern,
    GPUMemoryPool, PersistentKernel,
};

fn advanced_gpu_features() -> anyhow::Result<()> {
    // Mixed Precision Solver (Tensor Core acceleration)
    let n = 1000;
    let a = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_element(n, 2.0));
    let b = vec![1.0f64; n];

    let mp_solver = MixedPrecisionSolver::new(0, 1e-8, 100);
    let result = mp_solver.solve_mixed(&a, &b)?;

    // Multi-Stream Execution
    let mut executor = MultiStreamExecutor::new(0, 4);
    for i in 0..20 {
        executor.submit(AsyncGpuTask::new(i, i % 4).with_priority(i as u32 % 5));
    }
    let stats = executor.execute_all()?;

    // Kernel Fusion Optimization
    let mut optimizer = KernelFusionOptimizer::new();
    optimizer.register_kernel(KernelInfo::new("axpy").with_access_pattern(MemoryAccessPattern::Coalesced));
    optimizer.register_kernel(KernelInfo::new("scale").with_access_pattern(MemoryAccessPattern::Coalesced));
    let reduction = optimizer.estimate_reduction();

    // Memory Pool
    let mut pool = GPUMemoryPool::new(0, 1_000_000, 4096);
    let block = pool.allocate();
    let utilization = pool.utilization();

    // Persistent Kernel
    let kernel = PersistentKernel::new(0, 10, 32);
    kernel.launch(|block_id, thread_id| {
        // Computation here
    })?;

    Ok(())
}

// ============================================================================
// 7. CONTACT MECHANICS
// ============================================================================

use fea::gpu::gpu_nonlinear_dynamics::{
    GPUContactHandler, ContactPair, GPUExplicitDynamics,
    GPULargeDeformation, CriticalTimeStepEstimator,
};

fn contact_analysis() -> anyhow::Result<()> {
    // Contact Handler
    let handler = GPUContactHandler::new(0, 1e6).with_friction(0.3);

    let surface1: Vec<[f64; 3]> = (0..100).map(|i| [i as f64 * 0.01, 0.0, 0.0]).collect();
    let surface2: Vec<[f64; 3]> = (0..100).map(|i| [i as f64 * 0.01, 0.001, 0.0]).collect();

    let contacts = handler.detect_contact(&surface1, &surface2, 0.01);

    // Compute contact forces
    let gap = vec![-0.001, 0.001];
    let normal = vec![1.0, 1.0];
    let tv = vec![0.1, 0.0];
    let forces = handler.compute_contact_forces(&gap, &normal, &tv);

    // Large Deformation
    let large_def = GPULargeDeformation::new(0, true);
    let grad = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]];
    let strain = large_def.green_lagrange_strain(&grad);

    // Critical Time Step
    let mass = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_element(100, 1.0));
    let stiffness = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_element(100, 100.0));
    let dt_critical = CriticalTimeStepEstimator::estimate(&mass, &stiffness);

    Ok(())
}

// ============================================================================
// 8. SPECTRAL ANALYSIS (FFT)
// ============================================================================

use fea::gpu::gpu_fft::{GPUFFT, SpectralAnalysis, window_functions};

fn spectral_analysis() -> anyhow::Result<()> {
    // FFT
    let fft = GPUFFT::new(256, 0).unwrap();
    let signal: Vec<f64> = (0..256).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 256.0).sin()).collect();
    let (real, imag) = fft.fft_real(&signal);

    // Spectral Analysis
    let analyzer = SpectralAnalysis::new(256, 100.0, 0).unwrap();
    let psd = analyzer.power_spectrum(&signal);
    let peaks = analyzer.find_peaks(&psd, 0.1);

    // Window Functions
    let hanning = window_functions::hanning(256);
    let hamming = window_functions::hamming(256);
    let blackman = window_functions::blackman(256);

    Ok(())
}

// ============================================================================
// 9. BENCHMARKING
// ============================================================================

use fea::gpu::benchmark_advanced_features;

fn run_benchmarks() -> anyhow::Result<()> {
    // Advanced GPU benchmark
    let result = benchmark_advanced_features();
    println!("Mixed precision: {:.2} ms", result.mixed_precision_time_ms);
    println!("Multi-stream: {:.2} ms", result.multi_stream_time_ms);
    println!("Kernel fusion reduction: {:.1}%", result.kernel_fusion_reduction);

    // Nonlinear dynamics benchmark
    let nl_result = fea::gpu::gpu_nonlinear_dynamics::benchmark_nonlinear_dynamics();
    println!("Explicit dynamics: {:.2} ms", nl_result.explicit_dynamics_time_ms);
    println!("Contact detection: {:.2} ms", nl_result.contact_detection_time_ms);

    Ok(())
}

// ============================================================================
// 10. COMPLETE WORKFLOW EXAMPLE
// ============================================================================

fn complete_workflow() -> anyhow::Result<()> {
    // Check GPU availability
    let gpu_available = fea::gpu::gpu_available();
    let devices = fea::gpu::list_gpu_devices();

    println!("GPU Available: {}", gpu_available);
    println!("GPU Count: {}", devices.len());

    // Create model
    let mut model = Model::<Truss2>::new();

    // ... add nodes, elements, materials, BCs, loads ...

    // Solve with GPU acceleration
    if gpu_available {
        // GPU-accelerated solve
        // ...
    } else {
        // CPU solve
        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("FEA Framework Quick Reference");
    println!("=============================\n");

    // Uncomment examples to run:
    // basic_static_analysis()?;
    // gpu_accelerated_solve()?;
    // multi_gpu_analysis()?;
    // explicit_dynamics()?;
    // material_models()?;
    // advanced_gpu_features()?;
    // contact_analysis()?;
    // spectral_analysis()?;
    // run_benchmarks()?;
    // complete_workflow()?;

    println!("See individual function implementations for usage examples.");

    Ok(())
}
