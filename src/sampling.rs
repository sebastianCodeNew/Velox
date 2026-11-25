//! Advanced token sampling engine (`veloxc::sampling`).
//!
//! This module is the Velox-native replacement for Chimera-Inference's
//! `sampling.py`. It keeps the useful algorithmic surface — temperature,
//! top-k, nucleus (top-p), min-p, repetition/frequency/presence penalties,
//! and deterministic seeded sampling — while removing the Python/PyTorch
//! runtime and its per-token allocations.
//!
//! The sampler owns reusable scratch buffers. After construction, `sample`
//! performs no heap allocation as long as the vocabulary and history do not
//! exceed the capacities reserved by the caller.

use std::cmp::Ordering;
use std::collections::HashMap;

/// Configuration for token sampling.
///
/// Defaults are greedy decoding (`temperature = 0.0`), which matches the
/// historical Velox runtime behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// Softmax temperature. Values `<= 1e-5` select greedy argmax.
    pub temperature: f32,
    /// Keep only the `top_k` highest-logit tokens. `0` disables the filter.
    pub top_k: usize,
    /// Nucleus probability mass in `(0, 1]`. `1.0` disables the filter.
    pub top_p: f32,
    /// Minimum token probability relative to the most likely token.
    /// `0.0` disables the filter.
    pub min_p: f32,
    /// Multiplicative penalty for tokens already present in the history.
    /// `1.0` disables the penalty.
    pub repetition_penalty: f32,
    /// Additive penalty proportional to a token's history frequency.
    /// `0.0` disables the penalty.
    pub frequency_penalty: f32,
    /// Additive one-time penalty for tokens present in the history.
    /// `0.0` disables the penalty.
    pub presence_penalty: f32,
    /// Seed for the deterministic SplitMix64 stream.
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0x5EED_0F_5EED_0F5E,
        }
    }
}

impl SamplingParams {
    /// Greedy decoding with no penalties.
    #[must_use]
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0x5EED_0F_5EED_0F5E,
        }
    }

    /// Returns `true` when this configuration is equivalent to greedy argmax.
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 1e-5 || self.top_p <= 0.0 || self.top_k == 1
    }

    /// Validates the parameter ranges. Returns a human-readable error on the
    /// first invalid field.
    pub fn validate(&self) -> Result<(), String> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(format!(
                "temperature must be finite and >= 0, got {}",
                self.temperature
            ));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(format!("top_p must be in [0, 1], got {}", self.top_p));
        }
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            return Err(format!("min_p must be in [0, 1], got {}", self.min_p));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(format!(
                "repetition_penalty must be finite and > 0, got {}",
                self.repetition_penalty
            ));
        }
        if !self.frequency_penalty.is_finite() {
            return Err("frequency_penalty must be finite".to_string());
        }
        if !self.presence_penalty.is_finite() {
            return Err("presence_penalty must be finite".to_string());
        }
        Ok(())
    }
}

/// SplitMix64: small, fast, deterministic PRNG with no external dependencies.
#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0, 1)` using the top 24 mantissa bits.
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) * (1.0 / 16_777_216.0)
    }
}

/// Stateful sampler with reusable scratch buffers.
#[derive(Debug)]
pub struct Sampler {
    params: SamplingParams,
    rng: SplitMix64,
    /// Adjusted logits, then exponentiated probabilities during a sample call.
    scratch: Vec<f32>,
    /// Vocabulary indices sorted by descending adjusted logit.
    indices: Vec<u32>,
    /// Token -> occurrence count for the current history.
    counts: HashMap<u32, u32>,
}

impl Sampler {
    /// Creates a sampler for a vocabulary of `vocab_size` tokens.
    ///
    /// The allocation happens once at stream setup, never inside `sample`.
    pub fn new(params: SamplingParams, vocab_size: usize) -> Result<Self, String> {
        params.validate()?;
        if vocab_size == 0 {
            return Err("vocab_size must be > 0".to_string());
        }
        Ok(Self {
            params,
            rng: SplitMix64::new(params.seed),
            scratch: vec![f32::NEG_INFINITY; vocab_size],
            indices: (0..vocab_size as u32).collect(),
            counts: HashMap::with_capacity(1024),
        })
    }

    /// Reserves capacity for the maximum history length so that penalty
    /// accounting does not reallocate during generation.
    pub fn reserve_history(&mut self, max_tokens: usize) {
        self.counts.reserve(max_tokens.saturating_sub(self.counts.capacity()));
    }

    /// Replaces the sampling parameters and resets the PRNG stream.
    pub fn reset(&mut self, params: SamplingParams) -> Result<(), String> {
        params.validate()?;
        self.params = params;
        self.rng = SplitMix64::new(params.seed);
        Ok(())
    }

    /// Returns the active parameters.
    #[must_use]
    pub fn params(&self) -> SamplingParams {
        self.params
    }

    /// Samples one token from `logits` given the tokens already emitted.
    ///
    /// `history` should contain the prompt and all previously generated
    /// tokens, excluding the token being sampled. It is used only for the
    /// repetition/frequency/presence penalties.
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let n = logits.len();
        if n == 0 {
            return 0;
        }
        self.ensure_capacity(n);

        // Penalty accounting. `clear` retains the hash-map allocation.
        self.counts.clear();
        for &token in history {
            *self.counts.entry(token).or_insert(0) += 1;
        }

        let inv_temp = if self.params.temperature > 1e-5 {
            1.0 / self.params.temperature
        } else {
            1.0
        };
        let rep = self.params.repetition_penalty;
        let freq = self.params.frequency_penalty;
        let pres = self.params.presence_penalty;

        for (i, &raw) in logits.iter().enumerate() {
            let mut value = if raw.is_finite() {
                raw
            } else {
                f32::NEG_INFINITY
            };

            if let Some(&count) = self.counts.get(&(i as u32)) {
                if rep != 1.0 {
                    value = if value > 0.0 { value / rep } else { value * rep };
                }
                if freq != 0.0 {
                    value -= freq * count as f32;
                }
                if pres != 0.0 {
                    value -= pres;
                }
            }

            if self.params.temperature > 1e-5 {
                value *= inv_temp;
            }
            self.scratch[i] = value;
            self.indices[i] = i as u32;
        }

        if self.params.is_greedy() {
            return argmax(&self.scratch[..n]);
        }

        // Sort the vocabulary by adjusted logit, descending.
        {
            let scratch = &self.scratch[..n];
            self.indices[..n].sort_unstable_by(|&a, &b| {
                scratch[b as usize]
                    .partial_cmp(&scratch[a as usize])
                    .unwrap_or(Ordering::Equal)
            });
        }

        let limit = if self.params.top_k > 0 {
            self.params.top_k.min(n)
        } else {
            n
        };
        let max_logit = self.scratch[self.indices[0] as usize];
        if !max_logit.is_finite() {
            // Every candidate is -inf/NaN; keep deterministic behavior.
            return self.indices[0];
        }

        // Exponentiate the candidate window relative to the maximum. The
        // maximum therefore has probability weight exactly 1.0, which makes
        // the min-p comparison a direct threshold.
        let mut sum_all = 0.0f32;
        for k in 0..limit {
            let idx = self.indices[k] as usize;
            let weight = (self.scratch[idx] - max_logit).exp();
            self.scratch[idx] = weight;
            sum_all += weight;
        }
        if !sum_all.is_finite() || sum_all <= 0.0 {
            return self.indices[0];
        }

        // Determine the kept prefix. The list is sorted, so once min-p fails
        // every later token fails as well.
        let mut cumulative = 0.0f32;
        let mut cutoff = 1usize;
        for k in 0..limit {
            let idx = self.indices[k] as usize;
            let weight = self.scratch[idx];
            if self.params.min_p > 0.0 && weight < self.params.min_p {
                break;
            }
            cumulative += weight;
            cutoff = k + 1;
            if self.params.top_p < 1.0 && cumulative / sum_all >= self.params.top_p {
                break;
            }
        }

        let mut kept_sum = 0.0f32;
        for k in 0..cutoff {
            kept_sum += self.scratch[self.indices[k] as usize];
        }
        if kept_sum <= 0.0 || !kept_sum.is_finite() {
            return self.indices[0];
        }

        let target = self.rng.next_f32() * kept_sum;
        let mut running = 0.0f32;
        for k in 0..cutoff {
            let idx = self.indices[k] as usize;
            running += self.scratch[idx];
            if running >= target {
                return idx as u32;
            }
        }
        self.indices[cutoff - 1]
    }

    fn ensure_capacity(&mut self, n: usize) {
        if self.scratch.len() < n {
            self.scratch.resize(n, f32::NEG_INFINITY);
            self.indices.resize(n, 0);
        }
    }
}

/// A reusable generation stream: sampler plus token history.
///
/// Construct this before entering the timed generation loop so that neither
/// the sampler scratch buffers nor the history vector allocate per token.
#[derive(Debug)]
pub struct SamplerSession {
    sampler: Sampler,
    history: Vec<u32>,
}

impl SamplerSession {
    /// Creates a session sized for `max_history` tokens.
    pub fn new(
        params: SamplingParams,
        vocab_size: usize,
        max_history: usize,
    ) -> Result<Self, String> {
        let mut sampler = Sampler::new(params, vocab_size)?;
        sampler.reserve_history(max_history);
        Ok(Self {
            sampler,
            history: Vec::with_capacity(max_history),
        })
    }

    /// Samples from `logits` using the accumulated history.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        self.sampler.sample(logits, &self.history)
    }

    /// Records a token in the history used by the penalty processors.
    pub fn push(&mut self, token: u32) {
        self.history.push(token);
    }

    /// Clears the history while retaining its allocation.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Returns the active sampling parameters.
    #[must_use]
    pub fn params(&self) -> SamplingParams {
        self.sampler.params()
    }

    /// Returns the current history.
    #[must_use]
    pub fn history(&self) -> &[u32] {
        &self.history
    }
}

fn argmax(values: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        if value > best_val {
            best_val = value;
            best_idx = idx as u32;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(temperature: f32, top_k: usize, top_p: f32, min_p: f32, seed: u64) -> SamplingParams {
        SamplingParams {
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed,
        }
    }

    #[test]
    fn test_greedy_returns_argmax() {
        let mut sampler = Sampler::new(SamplingParams::greedy(), 4).unwrap();
        let logits = [0.1, 2.0, -1.0, 0.5];
        assert_eq!(sampler.sample(&logits, &[]), 1);
    }

    #[test]
    fn test_temperature_zero_returns_argmax() {
        let mut sampler = Sampler::new(params(0.0, 0, 1.0, 0.0, 1), 4).unwrap();
        let logits = [0.1, 2.0, -1.0, 0.5];
        assert_eq!(sampler.sample(&logits, &[]), 1);
    }

    #[test]
    fn test_top_k_one_returns_argmax() {
        let mut sampler = Sampler::new(params(1.0, 1, 1.0, 0.0, 1), 4).unwrap();
        let logits = [0.1, 2.0, -1.0, 0.5];
        assert_eq!(sampler.sample(&logits, &[]), 1);
    }

    #[test]
    fn test_top_k_excludes_tail() {
        let mut sampler = Sampler::new(params(1.0, 2, 1.0, 0.0, 7), 5).unwrap();
        let logits = [5.0, 4.0, 3.0, 2.0, 1.0];
        for _ in 0..2_000 {
            let token = sampler.sample(&logits, &[]);
            assert!(token == 0 || token == 1, "top_k leaked token {token}");
        }
    }

    #[test]
    fn test_min_p_excludes_tail() {
        let mut sampler = Sampler::new(params(1.0, 0, 1.0, 0.5, 11), 3).unwrap();
        let logits = [10.0, 0.0, 0.0];
        for _ in 0..2_000 {
            assert_eq!(sampler.sample(&logits, &[]), 0);
        }
    }

    #[test]
    fn test_repetition_penalty_changes_greedy_choice() {
        let mut sampler = Sampler::new(
            SamplingParams {
                repetition_penalty: 10.0,
                ..SamplingParams::greedy()
            },
            2,
        )
        .unwrap();
        let logits = [0.5, 1.0];
        // Without history token 1 wins; with history its logit is divided by 10.
        assert_eq!(sampler.sample(&logits, &[]), 1);
        assert_eq!(sampler.sample(&logits, &[1]), 0);
    }

    #[test]
    fn test_frequency_penalty_counts_occurrences() {
        let mut sampler = Sampler::new(
            SamplingParams {
                frequency_penalty: 2.0,
                ..SamplingParams::greedy()
            },
            2,
        )
        .unwrap();
        let logits = [0.0, 5.0];
        assert_eq!(sampler.sample(&logits, &[1]), 1);
        assert_eq!(sampler.sample(&logits, &[1, 1, 1]), 0);
    }

    #[test]
    fn test_presence_penalty_is_one_time() {
        let mut sampler = Sampler::new(
            SamplingParams {
                presence_penalty: 1.0,
                ..SamplingParams::greedy()
            },
            2,
        )
        .unwrap();
        let logits = [0.0, 0.5];
        assert_eq!(sampler.sample(&logits, &[]), 1);
        assert_eq!(sampler.sample(&logits, &[1, 1, 1]), 0);
    }

    #[test]
    fn test_seeded_sampling_is_deterministic() {
        let logits = [1.0, 0.5, 0.25, 0.125];
        let config = params(1.0, 0, 1.0, 0.0, 42);

        let mut a = Sampler::new(config, logits.len()).unwrap();
        let mut b = Sampler::new(config, logits.len()).unwrap();
        for _ in 0..256 {
            assert_eq!(a.sample(&logits, &[]), b.sample(&logits, &[]));
        }
    }

    #[test]
    fn test_distribution_matches_softmax() {
        // log(0.75) and log(0.25) produce a 3:1 distribution at temperature 1.
        let logits = [0.75f32.ln(), 0.25f32.ln()];
        let mut sampler = Sampler::new(params(1.0, 0, 1.0, 0.0, 1234), 2).unwrap();

        let mut counts = [0usize; 2];
        let draws = 20_000;
        for _ in 0..draws {
            counts[sampler.sample(&logits, &[]) as usize] += 1;
        }
        let first_ratio = counts[0] as f64 / draws as f64;
        assert!(
            (first_ratio - 0.75).abs() < 0.02,
            "expected ~0.75, got {first_ratio} ({counts:?})"
        );
    }

    #[test]
    fn test_invalid_params_are_rejected() {
        assert!(Sampler::new(params(-1.0, 0, 1.0, 0.0, 1), 4).is_err());
        assert!(Sampler::new(params(1.0, 0, 1.5, 0.0, 1), 4).is_err());
        assert!(Sampler::new(params(1.0, 0, 1.0, 1.5, 1), 4).is_err());
        assert!(Sampler::new(
            SamplingParams {
                repetition_penalty: 0.0,
                ..SamplingParams::default()
            },
            4
        )
        .is_err());
    }

    #[test]
    fn test_nan_logits_are_ignored() {
        let mut sampler = Sampler::new(SamplingParams::greedy(), 3).unwrap();
        let logits = [f32::NAN, 1.0, f32::NEG_INFINITY];
        assert_eq!(sampler.sample(&logits, &[]), 1);
    }

    #[test]
    fn test_empty_logits_returns_zero() {
        let mut sampler = Sampler::new(SamplingParams::greedy(), 1).unwrap();
        assert_eq!(sampler.sample(&[], &[]), 0);
    }
}
