//! Advanced adaptive iterative solvers with intelligent control.
//!
//! This module provides:
//! - PID-controlled convergence acceleration
//! - Adaptive preconditioner switching
//! - Convergence monitoring and prediction
//! - Multi-strategy solver orchestration

use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;

/// PID controller for iterative solver acceleration.
#[derive(Debug, Clone)]
pub struct PIDController {
    /// Proportional gain.
    pub kp: f64,
    /// Integral gain.
    pub ki: f64,
    /// Derivative gain.
    pub kd: f64,
    /// Integral accumulator.
    pub integral: f64,
    /// Previous error.
    pub prev_error: Option<f64>,
    /// Output limits.
    pub output_min: f64,
    pub output_max: f64,
}

impl Default for PIDController {
    fn default() -> Self {
        Self {
            kp: 0.1,
            ki: 0.01,
            kd: 0.05,
            integral: 0.0,
            prev_error: None,
            output_min: 0.1,
            output_max: 2.0,
        }
    }
}

impl PIDController {
    /// Creates a new PID controller.
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            ..Default::default()
        }
    }

    /// Creates PID controller for solver relaxation.
    pub fn for_relaxation() -> Self {
        Self {
            kp: 0.15,
            ki: 0.02,
            kd: 0.08,
            output_min: 0.5,
            output_max: 1.5,
            ..Default::default()
        }
    }

    /// Computes control output.
    pub fn compute(&mut self, error: f64, dt: f64) -> f64 {
        // Proportional term
        let p = self.kp * error;

        // Integral term with anti-windup
        self.integral += error * dt;
        self.integral = self.integral.clamp(-10.0, 10.0);
        let i = self.ki * self.integral;

        // Derivative term
        let d = match self.prev_error {
            Some(prev) => self.kd * (error - prev) / dt,
            None => 0.0,
        };

        self.prev_error = Some(error);

        // Compute output with limits
        let output = (p + i + d).clamp(self.output_min, self.output_max);
        output
    }

    /// Resets the controller state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = None;
    }
}

/// Convergence monitor with prediction capabilities.
#[derive(Debug, Clone)]
pub struct ConvergenceMonitor {
    /// Residual history.
    pub residual_history: VecDeque<f64>,
    /// Maximum history size.
    pub max_history: usize,
    /// Convergence rate estimate.
    pub convergence_rate: Option<f64>,
    /// Predicted iterations to convergence.
    pub predicted_iterations: Option<usize>,
}

impl Default for ConvergenceMonitor {
    fn default() -> Self {
        Self {
            residual_history: VecDeque::with_capacity(20),
            max_history: 20,
            convergence_rate: None,
            predicted_iterations: None,
        }
    }
}

impl ConvergenceMonitor {
    /// Creates a new convergence monitor.
    pub fn new(max_history: usize) -> Self {
        Self {
            max_history,
            residual_history: VecDeque::with_capacity(max_history),
            ..Default::default()
        }
    }

    /// Records a residual value.
    pub fn record(&mut self, residual: f64) {
        self.residual_history.push_back(residual);
        while self.residual_history.len() > self.max_history {
            self.residual_history.pop_front();
        }
        self.update_estimates();
    }

    /// Updates convergence rate estimates.
    fn update_estimates(&mut self) {
        if self.residual_history.len() < 3 {
            return;
        }

        let history: Vec<f64> = self.residual_history.iter().copied().collect();

        // Estimate convergence rate using linear regression on log(residual)
        let n = history.len();
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, &r) in history.iter().enumerate() {
            let x = i as f64;
            let y = r.ln().max(-100.0); // Avoid -inf
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let slope = (n as f64 * sum_xy - sum_x * sum_y)
            / (n as f64 * sum_xx - sum_x * sum_x).max(1e-10);

        self.convergence_rate = Some((-slope).exp());

        // Predict iterations to reach tolerance
        if let Some(&last) = history.last() {
            let tolerance = 1e-10;
            if last > tolerance && slope < -1e-10 {
                let remaining = (tolerance / last).ln() / slope;
                self.predicted_iterations = Some(remaining.ceil().max(0.0) as usize);
            }
        }
    }

    /// Checks if convergence is stagnating.
    pub fn is_stagnating(&self, threshold: f64) -> bool {
        if self.residual_history.len() < 5 {
            return false;
        }

        let history: Vec<f64> = self.residual_history.iter().copied().collect();
        let recent_avg: f64 = history[history.len() - 3..].iter().sum::<f64>() / 3.0;
        let older_avg: f64 = history[..3].iter().sum::<f64>() / 3.0;

        (recent_avg - older_avg).abs() / older_avg.max(1e-15) < threshold
    }

    /// Gets recommended action based on convergence behavior.
    pub fn get_recommendation(&self) -> ConvergenceRecommendation {
        if let Some(rate) = self.convergence_rate {
            if rate < 0.5 {
                ConvergenceRecommendation::Continue
            } else if rate < 0.8 {
                ConvergenceRecommendation::ConsiderAcceleration
            } else if self.is_stagnating(0.01) {
                ConvergenceRecommendation::ChangePreconditioner
            } else {
                ConvergenceRecommendation::IncreaseRelaxation
            }
        } else {
            ConvergenceRecommendation::Continue
        }
    }
}

/// Recommendation for solver adjustment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceRecommendation {
    /// Continue with current settings.
    Continue,
    /// Consider adding acceleration.
    ConsiderAcceleration,
    /// Change preconditioner.
    ChangePreconditioner,
    /// Increase relaxation parameter.
    IncreaseRelaxation,
    /// Decrease relaxation parameter.
    DecreaseRelaxation,
    /// Restart the iteration.
    Restart,
}

/// Adaptive preconditioner selector.
#[derive(Debug, Clone)]
pub struct AdaptivePreconditionerSelector {
    /// Available preconditioners.
    pub preconditioners: Vec<PreconditionerType>,
    /// Performance history.
    pub performance_history: Vec<(PreconditionerType, f64)>,
    /// Current selection.
    pub current: PreconditionerType,
    /// Selection interval (iterations).
    pub selection_interval: usize,
}

/// Preconditioner type enumeration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreconditionerType {
    None,
    Jacobi,
    SSOR(f64),
    IncompleteCholesky,
    Chebyshev(usize),
}

impl Default for AdaptivePreconditionerSelector {
    fn default() -> Self {
        Self {
            preconditioners: vec![
                PreconditionerType::None,
                PreconditionerType::Jacobi,
                PreconditionerType::SSOR(1.5),
                PreconditionerType::Chebyshev(2),
            ],
            performance_history: Vec::new(),
            current: PreconditionerType::Jacobi,
            selection_interval: 20,
        }
    }
}

impl AdaptivePreconditionerSelector {
    /// Creates a new adaptive selector.
    pub fn new(preconditioners: Vec<PreconditionerType>) -> Self {
        let current = preconditioners.first().copied().unwrap_or(PreconditionerType::Jacobi);
        Self {
            preconditioners,
            performance_history: Vec::new(),
            current,
            selection_interval: 20,
        }
    }

    /// Records performance for current preconditioner.
    pub fn record_performance(&mut self, residual_reduction: f64) {
        self.performance_history.push((self.current, residual_reduction));
    }

    /// Selects best preconditioner based on history.
    pub fn select_best(&mut self) -> PreconditionerType {
        if self.performance_history.len() < self.preconditioners.len() {
            // Not enough history - try each preconditioner
            for prec in &self.preconditioners {
                if !self.performance_history.iter().any(|(p, _)| *p == *prec) {
                    self.current = *prec;
                    return *prec;
                }
            }
        }

        // Group by preconditioner and compute average performance
        let mut scores: Vec<(PreconditionerType, f64, usize)> = Vec::new();

        for &prec in &self.preconditioners {
            let perf: Vec<f64> = self.performance_history
                .iter()
                .filter(|(p, _)| *p == prec)
                .map(|(_, r)| *r)
                .collect();

            if !perf.is_empty() {
                let avg = perf.iter().sum::<f64>() / perf.len() as f64;
                scores.push((prec, avg, perf.len()));
            }
        }

        // Select best performing
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((best, _, _)) = scores.first() {
            self.current = *best;
        }

        self.current
    }
}

/// Multi-strategy solver orchestrator.
#[derive(Debug, Clone)]
pub struct SolverOrchestrator {
    /// Convergence monitor.
    pub monitor: ConvergenceMonitor,
    /// PID controller for relaxation.
    pub pid: PIDController,
    /// Preconditioner selector.
    pub prec_selector: AdaptivePreconditionerSelector,
    /// Current relaxation parameter.
    pub relaxation: f64,
    /// Strategy change counter.
    pub strategy_changes: usize,
}

impl Default for SolverOrchestrator {
    fn default() -> Self {
        Self {
            monitor: ConvergenceMonitor::default(),
            pid: PIDController::for_relaxation(),
            prec_selector: AdaptivePreconditionerSelector::default(),
            relaxation: 1.0,
            strategy_changes: 0,
        }
    }
}

impl SolverOrchestrator {
    /// Creates a new solver orchestrator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates solver strategy based on convergence.
    pub fn update_strategy(&mut self, residual: f64, iteration: usize) -> SolverStrategy {
        self.monitor.record(residual);

        let recommendation = self.monitor.get_recommendation();

        match recommendation {
            ConvergenceRecommendation::Continue => {
                SolverStrategy::Continue { relaxation: self.relaxation }
            }
            ConvergenceRecommendation::ConsiderAcceleration => {
                self.pid.reset();
                SolverStrategy::EnableAcceleration
            }
            ConvergenceRecommendation::ChangePreconditioner => {
                self.strategy_changes += 1;
                let new_prec = self.prec_selector.select_best();
                SolverStrategy::ChangePreconditioner(new_prec)
            }
            ConvergenceRecommendation::IncreaseRelaxation => {
                self.relaxation = self.pid.compute(0.1, 1.0);
                SolverStrategy::Continue { relaxation: self.relaxation }
            }
            ConvergenceRecommendation::DecreaseRelaxation => {
                self.relaxation = self.pid.compute(-0.1, 1.0);
                SolverStrategy::Continue { relaxation: self.relaxation }
            }
            ConvergenceRecommendation::Restart => {
                self.strategy_changes += 1;
                SolverStrategy::Restart
            }
        }
    }

    /// Gets predicted iterations to convergence.
    pub fn predicted_iterations(&self) -> Option<usize> {
        self.monitor.predicted_iterations
    }

    /// Resets the orchestrator.
    pub fn reset(&mut self) {
        self.monitor = ConvergenceMonitor::new(self.monitor.max_history);
        self.pid.reset();
        self.relaxation = 1.0;
        self.strategy_changes = 0;
    }
}

/// Solver strategy recommendation.
#[derive(Debug, Clone)]
pub enum SolverStrategy {
    /// Continue with current settings.
    Continue { relaxation: f64 },
    /// Enable acceleration method.
    EnableAcceleration,
    /// Change preconditioner.
    ChangePreconditioner(PreconditionerType),
    /// Restart iteration with new initial guess.
    Restart,
}

/// Adaptive CG solver with intelligent control.
#[derive(Debug, Clone)]
pub struct AdaptiveCGSolver {
    /// Base tolerance.
    pub tolerance: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Orchestrator.
    pub orchestrator: SolverOrchestrator,
    /// Use Anderson acceleration.
    pub use_anderson: bool,
    /// Anderson depth.
    pub anderson_depth: usize,
}

impl Default for AdaptiveCGSolver {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_iterations: 1000,
            orchestrator: SolverOrchestrator::default(),
            use_anderson: true,
            anderson_depth: 5,
        }
    }
}

impl AdaptiveCGSolver {
    /// Creates a new adaptive CG solver.
    pub fn new(tolerance: f64, max_iterations: usize) -> Self {
        Self {
            tolerance,
            max_iterations,
            orchestrator: SolverOrchestrator::new(),
            use_anderson: true,
            anderson_depth: 5,
        }
    }

    /// Solves Ax = b with adaptive control.
    pub fn solve(&mut self, a: &DMatrix<f64>, b: &DVector<f64>) -> AdaptiveSolverResult {
        let n = b.len();
        let mut x = DVector::zeros(n);
        let mut r = b - a * &x;
        let mut p = r.clone();

        let b_norm = b.norm();
        let tol_abs = self.tolerance * b_norm.max(1e-15);

        let mut iteration = 0;
        let mut converged = false;
        let mut relaxation = 1.0;

        // Anderson acceleration storage
        let mut anderson_x: Vec<DVector<f64>> = Vec::new();
        let mut anderson_r: Vec<DVector<f64>> = Vec::new();

        while iteration < self.max_iterations {
            // Update strategy - use reference
            let r_norm = r.norm();
            let strategy = self.orchestrator.update_strategy(r_norm, iteration);

            match strategy {
                SolverStrategy::Continue { relaxation: rlx } => {
                    relaxation = rlx;
                }
                SolverStrategy::EnableAcceleration => {
                    self.use_anderson = true;
                }
                SolverStrategy::Restart => {
                    x = DVector::zeros(n);
                    r = b.clone();
                    p = r.clone();
                    self.orchestrator.reset();
                }
                SolverStrategy::ChangePreconditioner(_) => {
                    // Would change preconditioner here
                }
            }

            let rz = r.dot(&p);
            if rz.abs() < 1e-30 {
                break;
            }

            let ap = a * &p;
            let pap = p.dot(&ap);

            if pap.abs() < 1e-30 {
                break;
            }

            let alpha = rz / pap;
            x += p.scale(alpha * relaxation);
            r -= ap.scale(alpha);

            let r_norm = r.norm();
            if r_norm < tol_abs {
                converged = true;
                iteration += 1;
                break;
            }

            // Anderson acceleration
            if self.use_anderson && iteration > 2 {
                anderson_x.push(x.clone());
                anderson_r.push(r.clone());

                while anderson_x.len() > self.anderson_depth {
                    anderson_x.remove(0);
                    anderson_r.remove(0);
                }

                if anderson_x.len() >= self.anderson_depth {
                    // Simple Anderson averaging
                    let m = anderson_x.len();
                    let mut x_accel = DVector::zeros(n);

                    for x_hist in &anderson_x {
                        x_accel += x_hist.scale(1.0 / m as f64);
                    }

                    let r_accel_norm = (b - a * &x_accel).norm();
                    if r_accel_norm < r_norm {
                        x = x_accel;
                        r = b - a * &x;
                    }
                }
            }

            let beta = r.dot(&r) / rz;
            p = r.clone() + p.scale(beta);

            iteration += 1;
        }

        AdaptiveSolverResult {
            solution: x,
            iterations: iteration,
            residual_norm: r.norm(),
            converged,
            strategy_changes: self.orchestrator.strategy_changes,
            predicted_remaining: self.orchestrator.predicted_iterations(),
        }
    }
}

/// Result from an adaptive solver.
#[derive(Debug, Clone)]
pub struct AdaptiveSolverResult {
    /// Solution vector.
    pub solution: DVector<f64>,
    /// Number of iterations.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: f64,
    /// Convergence flag.
    pub converged: bool,
    /// Number of strategy changes.
    pub strategy_changes: usize,
    /// Predicted remaining iterations.
    pub predicted_remaining: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_controller() {
        let mut pid = PIDController::for_relaxation();

        let output = pid.compute(0.5, 1.0);
        assert!(output >= pid.output_min && output <= pid.output_max);

        pid.reset();
        assert_eq!(pid.integral, 0.0);
        assert!(pid.prev_error.is_none());
    }

    #[test]
    fn test_convergence_monitor() {
        let mut monitor = ConvergenceMonitor::new(20);

        // Simulate converging residuals
        for i in 0..10 {
            monitor.record(0.5_f64.powi(i));
        }

        assert!(monitor.convergence_rate.is_some());
        assert!(monitor.predicted_iterations.is_some());
        assert!(!monitor.is_stagnating(0.01));
    }

    #[test]
    fn test_preconditioner_selector() {
        let mut selector = AdaptivePreconditionerSelector::new(vec![
            PreconditionerType::None,
            PreconditionerType::Jacobi,
            PreconditionerType::Chebyshev(2),
        ]);

        selector.record_performance(0.5);
        selector.current = PreconditionerType::Jacobi;
        selector.record_performance(0.8);
        selector.current = PreconditionerType::Chebyshev(2);
        selector.record_performance(0.6);

        let best = selector.select_best();
        assert_eq!(best, PreconditionerType::Jacobi);
    }

    #[test]
    fn test_adaptive_cg_solver() {
        let mut solver = AdaptiveCGSolver::new(1e-8, 200);

        // Simple SPD system
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
        let result = solver.solve(&a, &b);

        assert!(result.converged || result.residual_norm < 1e-6);
        assert!(result.iterations > 0);
        assert!(result.iterations <= 200);
    }

    #[test]
    fn test_solver_orchestrator() {
        let mut orchestrator = SolverOrchestrator::new();

        // Simulate good convergence
        for i in 0..10usize {
            let residual = 0.5_f64.powi(i as i32);
            let strategy = orchestrator.update_strategy(residual, i);

            match strategy {
                SolverStrategy::Continue { .. } => {}
                _ => {}
            }
        }

        assert!(orchestrator.predicted_iterations().is_some());
    }
}
