# Requirements Document

## Introduction

This specification defines the requirements for refactoring the wave_gpt project from a heuristic prototype into a rigorous first-principles physics engine. The model must function as a Causal Autoregressive Language Model (GPT) while adhering strictly to Wave Mechanics equations. Previous iterations used wave vocabulary (sine waves, frequencies) but violated physics principles (random initialization, dot-product attention). This refactoring pivots to theoretical validation, proving that neural computation can be modeled as interference patterns in high-dimensional fields.

## Glossary

- **Wave Packet Embedding**: Token representation as quantized resonant modes where tokens are "particles" with mass inversely proportional to their frequency
- **Zipfian Distribution**: Statistical distribution where frequency of occurrence is inversely proportional to rank
- **Harmonic Quantization**: Initialization of frequencies as strict integer multiples of a base frequency (ω_n = n · ω_0)
- **Interference Attention**: Attention mechanism based on constructive/destructive wave interference rather than dot-product correlation
- **Phase Coherence**: Alignment of wave phases across coupled oscillators in the system
- **SVD Projection**: Singular Value Decomposition used to project gradients onto coherent subspaces
- **Damped Harmonic Momentum**: Momentum update rule based on physical damping equations
- **Field Decoherence**: Loss of phase alignment in the wave field, measured as regularization term
- **QFE (Quantum Field Entanglement)**: Regularization terms enforcing phase lock, energy conservation, and harmonic fidelity
- **Annealing Schedule**: Gradual transition from standard embeddings to pure wave physics during training
- **Causal Mask**: Upper triangular mask ensuring future tokens cannot influence past predictions
- **FFT (Fast Fourier Transform)**: Algorithm for analyzing frequency content of signals
- **Autocorrelation**: Measure of signal similarity with time-shifted versions of itself

## Requirements

### Requirement 1

**User Story:** As a research physicist, I want physics-aware token embeddings based on mass-frequency relationships, so that tokens behave as quantized resonant modes following natural physical laws.

#### Acceptance Criteria

1. WHEN the WavePacketEmbedding initializes token masses THEN the system SHALL compute mass using Zipfian distribution by rank: Mass(i) ∝ 1/(i+1)
2. WHEN the WavePacketEmbedding computes base frequencies THEN the system SHALL use the formula ω_0 = 1.0/√Mass where heavy/common tokens have low frequency and light/rare tokens have high frequency
3. WHEN the WavePacketEmbedding initializes harmonic frequencies THEN the system SHALL use strict integer multiples: ω_n = n · ω_0 with zero random noise
4. WHEN the WavePacketEmbedding initializes harmonic amplitudes THEN the system SHALL use 1/n power law decay
5. WHEN the WavePacketEmbedding forward pass executes THEN the system SHALL support annealing via standard_embed_ratio parameter mixing wave and standard embeddings as: out = (1-r)*wave + r*standard

### Requirement 2

**User Story:** As a research physicist, I want attention computed via wave interference rather than dot products, so that the model follows physical principles of constructive and destructive interference.

#### Acceptance Criteria

1. WHEN the InterferenceAttention projects input THEN the system SHALL produce Frequency, Phase, and Amplitude components rather than Query/Key projections
2. WHEN the InterferenceAttention computes phase evolution THEN the system SHALL use the formula φ(t) = ω · t + φ_0 where token position maps to time
3. WHEN the InterferenceAttention computes attention weights THEN the system SHALL use the interference formula: Intensity(t_q, t_k) = A_q² + A_k² + 2·A_q·A_k·cos(Δω·(t_q - t_k) + Δφ)
4. WHEN the InterferenceAttention applies causality THEN the system SHALL use torch.triu mask to prevent future waves from interfering with present
5. WHEN the InterferenceAttention normalizes weights THEN the system SHALL normalize by total energy potential (A_q + A_k)² rather than softmax to allow superposition values exceeding 1.0

### Requirement 3

**User Story:** As a research physicist, I want a wave-native optimizer that treats parameters as coupled oscillators, so that gradient updates preserve phase coherence rather than treating parameters as independent.

#### Acceptance Criteria

1. WHEN the WaveNativeOptimizer processes 2D weight matrices THEN the system SHALL compute SVD decomposition: U, S, Vh = SVD(W)
2. WHEN the WaveNativeOptimizer projects gradients THEN the system SHALL compute grad_coherent = U @ (U.T @ grad @ Vh.T) @ Vh
3. WHEN the WaveNativeOptimizer combines gradients THEN the system SHALL use weighted combination: grad_final = 0.7 * grad_coherent + 0.3 * raw_grad
4. WHEN the WaveNativeOptimizer updates momentum THEN the system SHALL use damped harmonic formula: v_{t+1} = v_t · (1 - γ) - ∇L · η
5. WHEN the WaveNativeOptimizer updates parameters THEN the system SHALL apply: θ_{t+1} = θ_t + v_{t+1}

### Requirement 4

**User Story:** As a research physicist, I want a loss function that minimizes field decoherence alongside prediction error, so that the model maintains physical consistency during training.

#### Acceptance Criteria

1. WHEN the WaveCoherenceLoss computes primary loss THEN the system SHALL use CrossEntropyLoss for next token prediction
2. WHEN the WaveCoherenceLoss computes phase lock regularization THEN the system SHALL penalize high phase variance within local token windows
3. WHEN the WaveCoherenceLoss computes energy conservation regularization THEN the system SHALL penalize L2 norm drift between layers: ||layer_N|| - ||layer_N-1||
4. WHEN the WaveCoherenceLoss computes harmonic fidelity regularization THEN the system SHALL penalize deviation from 1/n amplitude decay
5. WHEN the WaveCoherenceLoss returns results THEN the system SHALL provide a dictionary containing {total, ce, coherence} loss components

### Requirement 5

**User Story:** As a research physicist, I want diagnostic tools that verify wave signatures in the trained model, so that I can validate the physics-based approach produces genuine wave behavior.

#### Acceptance Criteria

1. WHEN the WaveDiagnostics analyzes spectrum THEN the system SHALL perform FFT of embeddings and return True if harmonic peaks (f, 2f, 3f) exist
2. WHEN the WaveDiagnostics visualizes interference THEN the system SHALL compute autocorrelation of attention weights and return True if periodic fringes exist
3. WHEN the WaveDiagnostics analyzes trajectories THEN the system SHALL track hidden states and return True if orbits are bounded/quasi-periodic (stable)
4. WHEN diagnostic methods execute THEN the system SHALL provide quantitative metrics alongside boolean pass/fail indicators
5. WHEN diagnostics are run THEN the system SHALL generate visualizations suitable for academic publication

### Requirement 6

**User Story:** As a researcher, I want the training loop to support physics-aware annealing and conditional component loading, so that I can experiment with different physics configurations.

#### Acceptance Criteria

1. WHEN training starts THEN the system SHALL initialize standard_embed_ratio at 1.0 and decay linearly to 0.0 by step 3000
2. WHEN the model forward pass executes THEN the system SHALL pass the current standard_embed_ratio to the embedding layer
3. WHEN use_rgd configuration is True THEN the system SHALL initialize WaveNativeOptimizer instead of standard optimizer
4. WHEN use_qfe configuration is True THEN the system SHALL initialize WaveCoherenceLoss instead of standard CrossEntropyLoss
5. WHEN training initializes THEN the system SHALL assert parameter count is between 50M and 55M for fair comparison runs

### Requirement 7

**User Story:** As a researcher, I want the refactored architecture to maintain compatibility with existing experiment infrastructure, so that I can compare physics-based results with baseline models.

#### Acceptance Criteria

1. WHEN the refactored model is used THEN the system SHALL maintain the same input/output interface as the original wave_gpt model
2. WHEN experiments run THEN the system SHALL integrate with existing checkpointing and logging infrastructure
3. WHEN results are generated THEN the system SHALL produce metrics compatible with existing visualization and analysis tools
4. WHEN the model generates text THEN the system SHALL support the same generation API (temperature, top_k, max_length)
5. WHEN ablation studies run THEN the system SHALL allow toggling individual physics components (wave embeddings, interference attention, RGD, QFE) independently

### Requirement 8

**User Story:** As a researcher, I want wave_benchmark.py updated to use the physics-first components, so that I can run benchmarks with the new wave mechanics approach.

#### Acceptance Criteria

1. WHEN wave_benchmark.py imports optimizer THEN the system SHALL use WaveNativeOptimizer from wave_physics_core.py instead of ResonantGradientDescent
2. WHEN wave_benchmark.py imports loss function THEN the system SHALL use WaveCoherenceLoss from wave_physics_core.py instead of QuantumFieldEntanglementLoss
3. WHEN wave_benchmark.py runs training THEN the system SHALL implement annealing schedule decaying standard_embed_ratio from 1.0 to 0.0 over 3000 steps
4. WHEN wave_benchmark.py completes training THEN the system SHALL run WaveDiagnostics to verify harmonic peaks, interference fringes, and trajectory stability
5. WHEN wave_benchmark.py generates visualizations THEN the system SHALL include diagnostic plots showing FFT spectrum, autocorrelation, and trajectory analysis

### Requirement 9

**User Story:** As a researcher, I want backward compatibility with physics_optim.py maintained, so that existing experiments and scripts continue to work during the transition.

#### Acceptance Criteria

1. WHEN physics_optim.py is imported THEN the system SHALL display a deprecation warning recommending migration to wave_physics_core.py
2. WHEN wave_physics_core.py provides compatibility functions THEN the system SHALL include create_physics_optimizer() matching physics_optim.py signature
3. WHEN wave_physics_core.py provides compatibility functions THEN the system SHALL include create_physics_loss() matching physics_optim.py signature
4. WHEN legacy code uses ResonantGradientDescent THEN the system SHALL continue to function until explicit removal in a future version
5. WHEN migration documentation is provided THEN the system SHALL include clear instructions for updating from physics_optim.py to wave_physics_core.py

