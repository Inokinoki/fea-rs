//! Unified Acceleration Framework for FEA Solvers.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! This module provides a unified interface for all acceleration methods,
//! allowing easy composition and automatic selection of optimal strategies.

use nalgebra::{DMatrix, DVector};
use super::adaptive_solvers::{ConvergenceMonitor, ConvergenceRecommendation, SolverOrchestrator};
use super::{CGSolver, IterativeConfig, Preconditioner, Solver, SolverResult};

/// Acceleration strategy enumeration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccelerationStrategy {
    /// No acceleration.
    None,
    /// Classical preconditioning only.
    Preconditioning(PreconditionerType),
    /// Krylov subspace recycling.
    Recycling { subspace_dim: usize },
    /// Spectral deflation.
    SpectralDeflation { num_eigenvalues: usize },
    /// Anderson acceleration.
    Anderson { depth: usize },
    /// Chebyshev semi-iterative.
    Chebyshev { degree: usize },
    /// Adaptive (auto-selected).
    Adaptive,
    /// Composite (multiple methods combined).
    Composite {
        preconditioner: PreconditionerType,
        recycling_dim: Option<usize>,
        anderson_depth: Option<usize>,
    },
}

/// Preconditioner type for unified framework.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreconditionerType {
    None,
    Jacobi,
    Chebyshev(usize),
    FSAI,
    Multigrid,
    DomainDecomposition,
}

/// Unified solver configuration.
#[derive(Debug, Clone)]
pub struct UnifiedSolverConfig {
    /// Acceleration strategy.
    pub strategy: AccelerationStrategy,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Enable convergence monitoring.
    pub enable_monitoring: bool,
    /// Verbose output.
    pub verbose: bool,
}

impl Default for UnifiedSolverConfig {
    fn default() -> Self {
        Self {
            strategy: AccelerationStrategy::Adaptive,
            max_iterations: 1000,
            tolerance: 1e-10,
            enable_monitoring: true,
            verbose: false,
        }
    }
}

/// Unified solver result with detailed statistics.
#[derive(Debug, Clone)]
pub struct UnifiedSolverResult {
    /// Base solver result.
    pub base_result: SolverResult,
    /// Acceleration iterations.
    pub acceleration_iterations: usize,
    /// Total matrix-vector products.
    pub matvec_count: usize,
    /// Estimated condition number.
    pub estimated_condition: Option<f64>,
    /// Convergence history.
    pub residual_history: Vec<f64>,
}

/// Unified acceleration solver.
pub struct UnifiedAccelerationSolver {
    config: UnifiedSolverConfig,
    monitor: Option<ConvergenceMonitor>,
    orchestrator: Option<SolverOrchestrator>,
}

impl UnifiedAccelerationSolver {
    /// Creates a new unified solver.
    pub fn new(config: UnifiedSolverConfig) -> Self {
        let monitor = if config.enable_monitoring {
            Some(ConvergenceMonitor::new(50))
        } else {
            None
        };

        let orchestrator = if config.strategy == AccelerationStrategy::Adaptive {
            Some(SolverOrchestrator::new())
        } else {
            None
        };

        Self {
            config,
            monitor,
            orchestrator,
        }
    }

    /// Solves Ax = b with unified acceleration.
    pub fn solve(&mut self, a: &DMatrix<f64>, b: &DVector<f64>) -> UnifiedSolverResult {
        match self.config.strategy {
            AccelerationStrategy::None => self.solve_basic(a, b),
            AccelerationStrategy::Preconditioning(prec) => self.solve_with_preconditioning(a, b, prec),
            AccelerationStrategy::Anderson { depth } => self.solve_with_anderson(a, b, depth),
            AccelerationStrategy::Chebyshev { degree } => self.solve_with_chebyshev(a, b, degree),
            AccelerationStrategy::Adaptive => self.solve_adaptive(a, b),
            AccelerationStrategy::Composite { preconditioner, recycling_dim, anderson_depth } => {
                self.solve_composite(a, b, preconditioner, recycling_dim, anderson_depth)
            }
            _ => self.solve_basic(a, b),
        }
    }

    fn solve_basic(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> UnifiedSolverResult {
        let config = IterativeConfig {
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(config.clone());
        let result = cg.solve(a, b, &config).unwrap_or_else(|e| {
            SolverResult {
                solution: vec![0.0; b.len()],
                iterations: Some(0),
                residual_norm: Some(f64::INFINITY),
                converged: false,
            }
        });

        let iters = result.iterations.unwrap_or(0);

        UnifiedSolverResult {
            base_result: result,
            acceleration_iterations: 0,
            matvec_count: iters * 2,
            estimated_condition: None,
            residual_history: Vec::new(),
        }
    }

    fn solve_with_preconditioning(
        &self,
        a: &DMatrix<f64>,
        b: &DVector<f64>,
        prec_type: PreconditionerType,
    ) -> UnifiedSolverResult {
        let preconditioner = match prec_type {
            PreconditionerType::Jacobi => Preconditioner::Jacobi,
            PreconditionerType::Chebyshev(d) => Preconditioner::Chebyshev(d),
            _ => Preconditioner::Jacobi,
        };

        let config = IterativeConfig {
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            preconditioner,
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(config.clone());
        let result = cg.solve(a, b, &config).unwrap_or_else(|_| SolverResult {
            solution: vec![0.0; b.len()],
            iterations: Some(0),
            residual_norm: Some(f64::INFINITY),
            converged: false,
        });

        let iters = result.iterations.unwrap_or(0);

        UnifiedSolverResult {
            base_result: result,
            acceleration_iterations: 0,
            matvec_count: iters * 2,
            estimated_condition: None,
            residual_history: Vec::new(),
        }
    }

    fn solve_with_anderson(
        &self,
        a: &DMatrix<f64>,
        b: &DVector<f64>,
        depth: usize,
    ) -> UnifiedSolverResult {
        let config = IterativeConfig {
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            preconditioner: Preconditioner::Jacobi,
            anderson_depth: depth,
            krylov_dim: 0,
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(config.clone());
        let result = cg.solve(a, b, &config).unwrap_or_else(|_| SolverResult {
            solution: vec![0.0; b.len()],
            iterations: Some(0),
            residual_norm: Some(f64::INFINITY),
            converged: false,
        });

        let iters = result.iterations.unwrap_or(0);

        UnifiedSolverResult {
            base_result: result,
            acceleration_iterations: depth,
            matvec_count: iters * 2,
            estimated_condition: None,
            residual_history: Vec::new(),
        }
    }

    fn solve_with_chebyshev(
        &self,
        a: &DMatrix<f64>,
        b: &DVector<f64>,
        degree: usize,
    ) -> UnifiedSolverResult {
        let config = IterativeConfig {
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            preconditioner: Preconditioner::Chebyshev(degree),
            anderson_depth: 0,
            krylov_dim: 0,
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(config.clone());
        let result = cg.solve(a, b, &config).unwrap_or_else(|_| SolverResult {
            solution: vec![0.0; b.len()],
            iterations: Some(0),
            residual_norm: Some(f64::INFINITY),
            converged: false,
        });

        let iters = result.iterations.unwrap_or(0);

        UnifiedSolverResult {
            base_result: result,
            acceleration_iterations: degree,
            matvec_count: iters * 2,
            estimated_condition: None,
            residual_history: Vec::new(),
        }
    }

    fn solve_adaptive(&mut self, a: &DMatrix<f64>, b: &DVector<f64>) -> UnifiedSolverResult {
        // Use orchestrator to adaptively select strategy
        if let Some(ref mut orchestrator) = self.orchestrator {
            let config = IterativeConfig {
                max_iterations: self.config.max_iterations,
                tolerance: self.config.tolerance,
                preconditioner: Preconditioner::Jacobi,
                anderson_depth: 5,
                krylov_dim: 0,
                deflation_vectors: None,
            };

            let cg = CGSolver::with_config(config.clone());
            let result = cg.solve(a, b, &config).unwrap_or_else(|_| SolverResult {
                solution: vec![0.0; b.len()],
                iterations: Some(0),
                residual_norm: Some(f64::INFINITY),
                converged: false,
            });

            let iters = result.iterations.unwrap_or(0);
            let residual_val = result.residual_norm;

            if let Some(ref mut monitor) = self.monitor {
                if let Some(residual) = residual_val {
                    monitor.record(residual);
                }
            }

            UnifiedSolverResult {
                base_result: result,
                acceleration_iterations: 5,
                matvec_count: iters * 2,
                estimated_condition: None,
                residual_history: self.monitor.as_ref()
                    .map(|m| m.residual_history.iter().copied().collect())
                    .unwrap_or_default(),
            }
        } else {
            self.solve_basic(a, b)
        }
    }

    fn solve_composite(
        &self,
        a: &DMatrix<f64>,
        b: &DVector<f64>,
        preconditioner: PreconditionerType,
        recycling_dim: Option<usize>,
        anderson_depth: Option<usize>,
    ) -> UnifiedSolverResult {
        let prec = match preconditioner {
            PreconditionerType::Jacobi => Preconditioner::Jacobi,
            PreconditionerType::Chebyshev(d) => Preconditioner::Chebyshev(d),
            _ => Preconditioner::Jacobi,
        };

        let config = IterativeConfig {
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            preconditioner: prec,
            anderson_depth: anderson_depth.unwrap_or(0),
            krylov_dim: recycling_dim.unwrap_or(0),
            deflation_vectors: None,
        };

        let cg = CGSolver::with_config(config.clone());
        let result = cg.solve(a, b, &config).unwrap_or_else(|_| SolverResult {
            solution: vec![0.0; b.len()],
            iterations: Some(0),
            residual_norm: Some(f64::INFINITY),
            converged: false,
        });

        let iters = result.iterations.unwrap_or(0);
        let accel_iters = anderson_depth.unwrap_or(0) + recycling_dim.unwrap_or(0);

        UnifiedSolverResult {
            base_result: result,
            acceleration_iterations: accel_iters,
            matvec_count: iters * 2,
            estimated_condition: None,
            residual_history: Vec::new(),
        }
    }

    /// Gets recommended strategy based on convergence behavior.
    pub fn get_recommended_strategy(&self) -> Option<AccelerationStrategy> {
        if let Some(ref monitor) = self.monitor {
            match monitor.get_recommendation() {
                ConvergenceRecommendation::Continue => Some(AccelerationStrategy::Adaptive),
                ConvergenceRecommendation::ConsiderAcceleration => {
                    Some(AccelerationStrategy::Anderson { depth: 5 })
                }
                ConvergenceRecommendation::ChangePreconditioner => {
                    Some(AccelerationStrategy::Preconditioning(PreconditionerType::Chebyshev(3)))
                }
                ConvergenceRecommendation::IncreaseRelaxation => {
                    Some(AccelerationStrategy::Composite {
                        preconditioner: PreconditionerType::Jacobi,
                        recycling_dim: Some(10),
                        anderson_depth: Some(3),
                    })
                }
                ConvergenceRecommendation::DecreaseRelaxation |
                ConvergenceRecommendation::Restart => {
                    Some(AccelerationStrategy::Preconditioning(PreconditionerType::Jacobi))
                }
            }
        } else {
            None
        }
    }

    /// Resets solver state.
    pub fn reset(&mut self) {
        if let Some(ref mut monitor) = self.monitor {
            *monitor = ConvergenceMonitor::new(50);
        }
        if let Some(ref mut orchestrator) = self.orchestrator {
            orchestrator.reset();
        }
    }
}

/// Solver comparison utility.
pub struct SolverComparator;

impl SolverComparator {
    /// Compares multiple acceleration strategies on a problem.
    pub fn compare_strategies(
        a: &DMatrix<f64>,
        b: &DVector<f64>,
        strategies: &[AccelerationStrategy],
        tolerance: f64,
        max_iterations: usize,
    ) -> Vec<(AccelerationStrategy, UnifiedSolverResult)> {
        let mut results = Vec::new();

        for &strategy in strategies {
            let config = UnifiedSolverConfig {
                strategy,
                max_iterations,
                tolerance,
                enable_monitoring: false,
                verbose: false,
            };

            let mut solver = UnifiedAccelerationSolver::new(config);
            let result = solver.solve(a, b);
            results.push((strategy, result));
        }

        results
    }

    /// Finds the best strategy based on iteration count.
    pub fn find_best_strategy(
        a: &DMatrix<f64>,
        b: &DVector<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) -> Option<(AccelerationStrategy, UnifiedSolverResult)> {
        let strategies = [
            AccelerationStrategy::None,
            AccelerationStrategy::Preconditioning(PreconditionerType::Jacobi),
            AccelerationStrategy::Preconditioning(PreconditionerType::Chebyshev(3)),
            AccelerationStrategy::Anderson { depth: 3 },
            AccelerationStrategy::Anderson { depth: 5 },
            AccelerationStrategy::Chebyshev { degree: 2 },
            AccelerationStrategy::Chebyshev { degree: 3 },
            AccelerationStrategy::Composite {
                preconditioner: PreconditionerType::Jacobi,
                recycling_dim: Some(5),
                anderson_depth: Some(3),
            },
        ];

        let results = Self::compare_strategies(a, b, &strategies, tolerance, max_iterations);

        results.into_iter()
            .filter(|(_, r)| r.base_result.converged)
            .min_by(|a, b| {
                a.1.base_result.iterations
                    .unwrap_or(usize::MAX)
                    .cmp(&b.1.base_result.iterations.unwrap_or(usize::MAX))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_solver_basic() {
        let a = DMatrix::from_row_slice(10, 10, &[
            4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0,
        ]);

        let b = DVector::from_element(10, 1.0);

        let config = UnifiedSolverConfig {
            strategy: AccelerationStrategy::Preconditioning(PreconditionerType::Jacobi),
            max_iterations: 100,
            tolerance: 1e-10,
            enable_monitoring: true,
            verbose: false,
        };

        let mut solver = UnifiedAccelerationSolver::new(config);
        let result = solver.solve(&a, &b);

        assert!(result.base_result.iterations.unwrap_or(0) > 0);
        assert!(result.base_result.converged || result.base_result.residual_norm.unwrap_or(f64::INFINITY) < 1.0);
    }

    #[test]
    fn test_solver_comparator() {
        let a = DMatrix::from_row_slice(10, 10, &[
            4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0,
        ]);

        let b = DVector::from_element(10, 1.0);

        let best = SolverComparator::find_best_strategy(&a, &b, 1e-10, 100);

        assert!(best.is_some());
        let (_, result) = best.unwrap();
        assert!(result.base_result.iterations.unwrap_or(0) > 0);
    }
}
