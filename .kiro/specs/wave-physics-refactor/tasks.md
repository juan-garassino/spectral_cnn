# Implementation Plan

- [x] 1. Create wave_physics_core.py with core physics components
  - [x] 1.1 Implement WaveNativeOptimizer with SVD projection and damped harmonic momentum
    - Create optimizer class inheriting from torch.optim.Optimizer
    - Implement SVD decomposition for 2D weight matrices
    - Implement coherent gradient projection: grad_coherent = U @ (U.T @ grad @ Vh.T) @ Vh
    - Implement gradient combination: grad_final = 0.7 * grad_coherent + 0.3 * raw_grad
    - Implement damped harmonic momentum: v_{t+1} = v_t * (1 - γ) - ∇L * η
    - Implement parameter update: θ_{t+1} = θ_t + v_{t+1}
    - Add fallback to raw gradient if SVD fails
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 1.2 Write property test for SVD gradient projection
    - **Property 10: SVD Gradient Projection**
    - **Validates: Requirements 3.1, 3.2**

  - [ ]* 1.3 Write property test for gradient combination weights
    - **Property 11: Gradient Combination Weights**
    - **Validates: Requirements 3.3**

  - [ ]* 1.4 Write property test for damped harmonic parameter update
    - **Property 12: Damped Harmonic Parameter Update**
    - **Validates: Requirements 3.4, 3.5**

  - [x] 1.5 Implement WaveCoherenceLoss with QFE regularization
    - Create loss class inheriting from nn.Module
    - Implement CrossEntropyLoss as primary component
    - Implement phase lock regularization (variance within local windows)
    - Implement energy conservation regularization (L2 norm drift between layers)
    - Implement harmonic fidelity regularization (deviation from 1/n decay)
    - Return dictionary with {total, ce, coherence} keys
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 1.6 Write property test for CE component correctness
    - **Property 13: Cross-Entropy Component Correctness**
    - **Validates: Requirements 4.1**

  - [ ]* 1.7 Write property test for loss dictionary structure
    - **Property 14: Loss Dictionary Structure**
    - **Validates: Requirements 4.5**

  - [x] 1.8 Implement WaveDiagnostics class
    - Implement analyze_spectrum() with FFT and harmonic peak detection
    - Implement visualize_interference() with autocorrelation computation
    - Implement analyze_trajectories() with Lyapunov exponent estimation
    - Return both boolean indicators and quantitative metrics
    - Generate matplotlib visualizations
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 1.9 Write property test for harmonic peak detection
    - **Property 15: Harmonic Peak Detection**
    - **Validates: Requirements 5.1**

  - [x] 1.10 Implement compatibility shim functions
    - Add create_physics_optimizer() that returns WaveNativeOptimizer
    - Add create_physics_loss() that returns WaveCoherenceLoss
    - Match function signatures with physics_optim.py for drop-in replacement
    - _Requirements: 7.1, 7.5_

- [x] 2. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Refactor WavePacketEmbedding in wave_gpt.py
  - [x] 3.1 Implement physics-aware token mass computation
    - Compute mass using Zipfian distribution: Mass(i) = 1/(i+1)
    - Store masses as buffer (not learnable)
    - _Requirements: 1.1_

  - [ ]* 3.2 Write property test for Zipfian mass distribution
    - **Property 1: Zipfian Mass Distribution**
    - **Validates: Requirements 1.1**

  - [x] 3.3 Implement mass-frequency relationship
    - Compute base frequency: ω_0 = 1.0 / sqrt(Mass)
    - Verify heavy tokens have low frequency, light tokens have high frequency
    - _Requirements: 1.2_

  - [ ]* 3.4 Write property test for mass-frequency inverse relationship
    - **Property 2: Mass-Frequency Inverse Relationship**
    - **Validates: Requirements 1.2**

  - [x] 3.5 Implement harmonic quantization
    - Initialize harmonic frequencies as strict integer multiples: ω_n = n * ω_0
    - Remove all random noise from frequency initialization
    - _Requirements: 1.3_

  - [ ]* 3.6 Write property test for harmonic quantization exactness
    - **Property 3: Harmonic Quantization Exactness**
    - **Validates: Requirements 1.3**

  - [x] 3.7 Implement power law amplitude decay
    - Initialize harmonic amplitudes with 1/n decay
    - _Requirements: 1.4_

  - [ ]* 3.8 Write property test for power law amplitude decay
    - **Property 4: Power Law Amplitude Decay**
    - **Validates: Requirements 1.4**

  - [x] 3.9 Implement embedding annealing
    - Add standard_embed_ratio parameter to forward()
    - Implement linear mixing: out = (1-r)*wave + r*standard
    - _Requirements: 1.5_

  - [ ]* 3.10 Write property test for embedding annealing linearity
    - **Property 5: Embedding Annealing Linearity**
    - **Validates: Requirements 1.5**

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement InterferenceAttention in wave_gpt.py
  - [x] 5.1 Implement frequency/phase/amplitude projections
    - Replace Q/K projections with freq_proj, phase_proj, amp_proj
    - Project input to frequency, phase, and amplitude components
    - _Requirements: 2.1_

  - [x] 5.2 Implement phase evolution computation
    - Map token position P to time t
    - Compute evolved phase: φ(t) = ω * t + φ_0
    - _Requirements: 2.2_

  - [ ]* 5.3 Write property test for phase evolution formula
    - **Property 6: Phase Evolution Formula**
    - **Validates: Requirements 2.2**

  - [x] 5.4 Implement interference intensity computation
    - Compute intensity using formula: I = A_q² + A_k² + 2*A_q*A_k*cos(Δω*(t_q - t_k) + Δφ)
    - _Requirements: 2.3_

  - [ ]* 5.5 Write property test for interference intensity formula
    - **Property 7: Interference Intensity Formula**
    - **Validates: Requirements 2.3**

  - [x] 5.6 Implement causal masking
    - Apply torch.triu mask to prevent future interference
    - _Requirements: 2.4_

  - [ ]* 5.7 Write property test for causal masking enforcement
    - **Property 8: Causal Masking Enforcement**
    - **Validates: Requirements 2.4**

  - [x] 5.8 Implement energy-based normalization
    - Normalize by (A_q + A_k)² instead of softmax
    - Verify output values can exceed 1.0
    - _Requirements: 2.5_

  - [ ]* 5.9 Write property test for energy-based normalization
    - **Property 9: Energy-Based Normalization (Non-Softmax)**
    - **Validates: Requirements 2.5**

- [ ] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Update wave_experiments.py training loop
  - [x] 7.1 Implement annealing schedule
    - Initialize standard_embed_ratio at 1.0
    - Decay linearly to 0.0 by step 3000
    - Pass ratio to model forward pass
    - _Requirements: 6.1, 6.2_

  - [ ]* 7.2 Write property test for annealing schedule linearity
    - **Property 16: Annealing Schedule Linearity**
    - **Validates: Requirements 6.1**

  - [x] 7.3 Implement conditional component loading
    - If use_rgd=True, initialize WaveNativeOptimizer
    - If use_qfe=True, initialize WaveCoherenceLoss
    - _Requirements: 6.3, 6.4_

  - [x] 7.4 Add parameter count assertion
    - Assert 50M < num_params < 55M for fair comparison (warning implemented)
    - _Requirements: 6.5_

- [x] 8. Ensure API compatibility with existing infrastructure
  - [x] 8.1 Verify input/output interface compatibility
    - Ensure model accepts same inputs as original wave_gpt
    - Ensure outputs have same shapes
    - Verify generate() supports temperature, top_k, max_length
    - _Requirements: 7.1, 7.4_

  - [ ]* 8.2 Write property test for API compatibility
    - **Property 17: API Compatibility**
    - **Validates: Requirements 7.1, 7.3, 7.4**

  - [x] 8.3 Verify integration with checkpointing and logging
    - Test checkpoint save/restore with new optimizer
    - Verify metrics logging works with new loss components
    - _Requirements: 7.2_

  - [x] 8.4 Implement component independence
    - Ensure each physics component can be toggled independently
    - Test all combinations: wave embeddings, interference attention, RGD, QFE
    - _Requirements: 7.5_

  - [ ]* 8.5 Write property test for component independence
    - **Property 18: Component Independence**
    - **Validates: Requirements 7.5**

- [x] 9. Update wave_benchmark.py for physics-first approach
  - [x] 9.1 Update imports to use wave_physics_core.py
    - Import WaveNativeOptimizer instead of ResonantGradientDescent
    - Import WaveCoherenceLoss instead of QuantumFieldEntanglementLoss
    - Import WaveDiagnostics for post-training validation
    - Keep backward compatibility with physics_optim.py via try/except
    - _Requirements: 6.3, 6.4_

  - [x] 9.2 Add annealing schedule to benchmark training loop
    - Initialize standard_embed_ratio at 1.0
    - Decay linearly to 0.0 by step 3000
    - Pass ratio to model forward pass via model(x, standard_embed_ratio=r)
    - _Requirements: 6.1, 6.2_

  - [x] 9.3 Integrate WaveDiagnostics into benchmark
    - Call analyze_spectrum() after training to verify harmonic peaks
    - Call visualize_interference() to check for periodic fringes
    - Call analyze_trajectories() to verify stable orbits
    - Log diagnostic results to benchmark output
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 9.4 Update benchmark visualizations
    - Add diagnostic plots to save_wave_visualizations()
    - Include FFT spectrum plot showing harmonic peaks
    - Include autocorrelation plot showing interference fringes
    - Include trajectory stability plot
    - _Requirements: 5.5_

  - [x] 9.5 Update benchmark configuration constants
    - Add ANNEALING_STEPS = 3000 constant
    - Update USE_RGD to use WaveNativeOptimizer
    - Update USE_QFE to use WaveCoherenceLoss
    - _Requirements: 6.3, 6.4_

- [x] 10. Maintain backward compatibility with physics_optim.py
  - [x] 10.1 Add deprecation warnings to physics_optim.py
    - Add deprecation warning to ResonantGradientDescent
    - Add deprecation warning to QuantumFieldEntanglementLoss
    - Document migration path to wave_physics_core.py
    - _Requirements: 9.1, 9.4, 9.5_

- [x] 11. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
