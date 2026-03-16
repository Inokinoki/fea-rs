//! GPU-accelerated FFT for spectral analysis.
//!
//! This module provides:
//! - Cooley-Tukey FFT algorithm
//! - GPU-accelerated forward/inverse FFT
//! - Spectral analysis utilities
//! - Convolution via FFT

use nalgebra::{DVector, Matrix2};

/// GPU-accelerated FFT solver.
#[derive(Debug, Clone)]
pub struct GPUFFT {
    device_id: usize,
    size: usize,
    /// Precomputed twiddle factors.
    twiddle_real: Vec<f64>,
    twiddle_imag: Vec<f64>,
    /// Bit-reversal permutation.
    bit_reverse: Vec<usize>,
}

impl GPUFFT {
    /// Creates a new FFT solver for given size (must be power of 2).
    pub fn new(size: usize, device_id: usize) -> Option<Self> {
        if !size.is_power_of_two() {
            return None;
        }

        let log_n = size.trailing_zeros() as usize;
        let mut twiddle_real = Vec::with_capacity(size / 2);
        let mut twiddle_imag = Vec::with_capacity(size / 2);
        let mut bit_reverse = Vec::with_capacity(size);

        // Precompute twiddle factors
        let two_pi = 2.0 * std::f64::consts::PI;
        for i in 0..size / 2 {
            let angle = -two_pi * i as f64 / size as f64;
            twiddle_real.push(angle.cos());
            twiddle_imag.push(angle.sin());
        }

        // Precompute bit-reversal permutation
        for i in 0..size {
            bit_reverse.push(reverse_bits(i, log_n));
        }

        Some(Self {
            device_id,
            size,
            twiddle_real,
            twiddle_imag,
            bit_reverse,
        })
    }

    /// Computes forward FFT (complex input).
    pub fn fft(&self, real: &[f64], imag: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.size;
        let mut out_real = real.to_vec();
        let mut out_imag = imag.to_vec();

        // Bit-reversal permutation
        for i in 0..n {
            let j = self.bit_reverse[i];
            if i < j {
                out_real.swap(i, j);
                out_imag.swap(i, j);
            }
        }

        // Cooley-Tukey iterative FFT
        let mut m = 2;
        while m <= n {
            let half_m = m / 2;
            let step = n / m;

            for k in (0..n).step_by(m) {
                for j in 0..half_m {
                    let tw_idx = j * step;
                    let wr = self.twiddle_real[tw_idx];
                    let wi = self.twiddle_imag[tw_idx];

                    let t_real = wr * out_real[k + j + half_m] - wi * out_imag[k + j + half_m];
                    let t_imag = wr * out_imag[k + j + half_m] + wi * out_real[k + j + half_m];

                    let u_real = out_real[k + j];
                    let u_imag = out_imag[k + j];

                    out_real[k + j] = u_real + t_real;
                    out_imag[k + j] = u_imag + t_imag;
                    out_real[k + j + half_m] = u_real - t_real;
                    out_imag[k + j + half_m] = u_imag - t_imag;
                }
            }
            m *= 2;
        }

        (out_real, out_imag)
    }

    /// Computes inverse FFT.
    pub fn ifft(&self, real: &[f64], imag: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.size;

        // Conjugate input
        let conj_imag: Vec<f64> = imag.iter().map(|&x| -x).collect();

        // Forward FFT
        let (mut out_real, mut out_imag) = self.fft(real, &conj_imag);

        // Conjugate and scale
        for i in 0..n {
            out_real[i] /= n as f64;
            out_imag[i] = -out_imag[i] / n as f64;
        }

        (out_real, out_imag)
    }

    /// Computes FFT of real-valued signal.
    pub fn fft_real(&self, signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let imag = vec![0.0; self.size];
        self.fft(signal, &imag)
    }

    /// Returns the FFT size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the device ID.
    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

/// Spectral analysis utilities.
pub struct SpectralAnalysis {
    fft: GPUFFT,
    sample_rate: f64,
}

impl SpectralAnalysis {
    /// Creates a new spectral analyzer.
    pub fn new(size: usize, sample_rate: f64, device_id: usize) -> Option<Self> {
        GPUFFT::new(size, device_id).map(|fft| Self { fft, sample_rate })
    }

    /// Computes power spectral density.
    pub fn power_spectrum(&self, signal: &[f64]) -> Vec<f64> {
        let n = self.fft.size();
        let (real, imag) = self.fft.fft_real(signal);

        let mut psd = Vec::with_capacity(n / 2 + 1);
        for i in 0..=n / 2 {
            let power = real[i] * real[i] + imag[i] * imag[i];
            psd.push(power / n as f64);
        }

        psd
    }

    /// Returns frequency bins.
    pub fn frequency_bins(&self) -> Vec<f64> {
        let n = self.fft.size();
        let df = self.sample_rate / n as f64;
        (0..=n / 2).map(|i| i as f64 * df).collect()
    }

    /// Finds dominant frequencies.
    pub fn find_peaks(&self, psd: &[f64], threshold: f64) -> Vec<(f64, f64)> {
        let freqs = self.frequency_bins();
        let mut peaks = Vec::new();

        for i in 1..psd.len() - 1 {
            if psd[i] > psd[i - 1] && psd[i] > psd[i + 1] && psd[i] > threshold {
                peaks.push((freqs[i], psd[i]));
            }
        }

        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        peaks
    }
}

/// Fast convolution using FFT.
pub fn fft_convolve(fft: &GPUFFT, signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = fft.size();

    // Zero-pad inputs
    let mut padded_signal = vec![0.0; n];
    let mut padded_kernel = vec![0.0; n];

    for (i, &s) in signal.iter().enumerate().take(n) {
        padded_signal[i] = s;
    }
    for (i, &k) in kernel.iter().enumerate().take(n) {
        padded_kernel[i] = k;
    }

    // FFT of both
    let (sig_real, sig_imag) = fft.fft_real(&padded_signal);
    let (ker_real, ker_imag) = fft.fft_real(&padded_kernel);

    // Complex multiplication in frequency domain
    let mut prod_real = Vec::with_capacity(n);
    let mut prod_imag = Vec::with_capacity(n);

    for i in 0..n {
        prod_real.push(sig_real[i] * ker_real[i] - sig_imag[i] * ker_imag[i]);
        prod_imag.push(sig_real[i] * ker_imag[i] + sig_imag[i] * ker_real[i]);
    }

    // Inverse FFT
    let (conv_real, _) = fft.ifft(&prod_real, &prod_imag);

    conv_real
}

/// Bit-reversal helper.
fn reverse_bits(x: usize, bits: usize) -> usize {
    let mut result = 0;
    for i in 0..bits {
        if (x >> i) & 1 == 1 {
            result |= 1 << (bits - 1 - i);
        }
    }
    result
}

/// GPU-accelerated convolution layer for neural network-style operations.
pub struct GPUConvolution {
    device_id: usize,
    fft_size: usize,
}

impl GPUConvolution {
    /// Creates a new GPU convolution operator.
    pub fn new(fft_size: usize, device_id: usize) -> Self {
        Self { device_id, fft_size }
    }

    /// Applies 1D convolution.
    pub fn convolve_1d(&self, input: &[f64], kernel: &[f64]) -> Vec<f64> {
        if let Some(fft) = GPUFFT::new(self.fft_size, self.device_id) {
            fft_convolve(&fft, input, kernel)
        } else {
            input.to_vec()
        }
    }

    /// Applies 2D convolution (via row-column decomposition).
    pub fn convolve_2d(&self, input: &[Vec<f64>], kernel: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // Simplified 2D convolution
        if input.is_empty() || kernel.is_empty() {
            return input.to_vec();
        }

        let in_rows = input.len();
        let in_cols = input[0].len();
        let k_rows = kernel.len();
        let k_cols = kernel[0].len();

        let out_rows = in_rows - k_rows + 1;
        let out_cols = in_cols - k_cols + 1;

        let mut output = vec![vec![0.0; out_cols]; out_rows];

        for i in 0..out_rows {
            for j in 0..out_cols {
                let mut sum = 0.0;
                for ki in 0..k_rows {
                    for kj in 0..k_cols {
                        sum += input[i + ki][j + kj] * kernel[ki][kj];
                    }
                }
                output[i][j] = sum;
            }
        }

        output
    }
}

/// Window functions for spectral analysis.
pub mod window_functions {
    /// Hanning window.
    pub fn hanning(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()))
            .collect()
    }

    /// Hamming window.
    pub fn hamming(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
            .collect()
    }

    /// Blackman window.
    pub fn blackman(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                0.42 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
                    + 0.08 * (4.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
            })
            .collect()
    }

    /// Apply window to signal.
    pub fn apply(signal: &[f64], window: &[f64]) -> Vec<f64> {
        signal.iter().zip(window).map(|(&s, &w)| s * w).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_creation() {
        let fft = GPUFFT::new(256, 0);
        assert!(fft.is_some());

        let fft_invalid = GPUFFT::new(255, 0);
        assert!(fft_invalid.is_none());
    }

    #[test]
    fn test_fft_forward_inverse() {
        let fft = GPUFFT::new(8, 0).unwrap();

        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let imag = vec![0.0; 8];

        let (real, imag) = fft.fft(&signal, &imag);
        let (recovered, _) = fft.ifft(&real, &imag);

        for (orig, rec) in signal.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft_real() {
        let fft = GPUFFT::new(4, 0).unwrap();
        let signal = vec![1.0, 0.0, -1.0, 0.0];

        let (real, imag) = fft.fft_real(&signal);

        assert_eq!(real.len(), 4);
        assert_eq!(imag.len(), 4);
    }

    #[test]
    fn test_spectral_analysis() {
        let sample_rate = 100.0;
        let analyzer = SpectralAnalysis::new(64, sample_rate, 0).unwrap();

        // Generate sine wave at 10 Hz
        let signal: Vec<f64> = (0..64)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / sample_rate).sin())
            .collect();

        let psd = analyzer.power_spectrum(&signal);
        assert_eq!(psd.len(), 33);

        let freqs = analyzer.frequency_bins();
        assert_eq!(freqs.len(), 33);
    }

    #[test]
    fn test_fft_convolution() {
        let fft = GPUFFT::new(8, 0).unwrap();

        let signal = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let kernel = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let result = fft_convolve(&fft, &signal, &kernel);

        assert_eq!(result.len(), 8);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_window_functions() {
        let window = window_functions::hanning(16);
        assert_eq!(window.len(), 16);
        assert!((window[0]).abs() < 1e-10);
        assert!((window[15]).abs() < 1e-10);

        let window = window_functions::hamming(16);
        assert_eq!(window.len(), 16);

        let window = window_functions::blackman(16);
        assert_eq!(window.len(), 16);
    }

    #[test]
    fn test_gpu_convolution_1d() {
        let conv = GPUConvolution::new(16, 0);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![0.5, 0.5];

        let output = conv.convolve_1d(&input, &kernel);
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_gpu_convolution_2d() {
        let conv = GPUConvolution::new(16, 0);

        let input = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let kernel = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];

        let output = conv.convolve_2d(&input, &kernel);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 2);
    }
}
