# configs/toy_periodic.yaml
# Geometry-Native World Model — Toy Periodic World (S^1 / Torus)
#
# Designed to be robust + runnable with minimal assumptions:
# - Generates data on the fly (optional caching)
# - Supports Euclidean baseline vs S^1-native latent
# - Includes OOD test split with different frequency/amplitude/noise

experiment:
  name: toy_periodic
  seed: 42
  device: auto          # auto | cpu | cuda
  dtype: float32
  work_dir: ./runs/toy_periodic
  log_every: 50
  save_every: 1
  num_workers: 0

data:
  dataset: toy_periodic
  cache:
    enabled: true
    dir: ./data_cache/toy_periodic
    overwrite: false

  # ===== Sequence generation =====
  sequence:
    length: 32                 # T steps per sample
    num_sines: 3               # number of independent periodic factors (torus dim)
    dt: 1.0
    # frequencies are sampled per sequence
    freq:
      min: 0.03
      max: 0.15
    # amplitudes are sampled per sequence
    amp:
      min: 0.5
      max: 1.5
    # phase is uniform in [0, 2pi)
    phase_init: uniform

  # ===== Observations =====
  observation:
    # Observation is a noisy mix of sin/cos plus optional linear mixing
    type: mixed_sincos         # mixed_sincos | raw_angle
    # Each sine factor contributes (sin, cos), then we linearly mix to obs_dim
    obs_dim: 16
    mix_matrix: random_orthogonal   # random_orthogonal | random_gaussian | identity
    noise_std: 0.05
    dropout_prob: 0.0

  # ===== Missing observations =====
  missingness:
    enabled: true
    p_missing: 0.15           # probability a time step is masked/missing
    mode: bernoulli           # bernoulli | block
    block:
      min_len: 2
      max_len: 6

  split:
    train_size: 30000
    val_size: 3000
    test_size: 3000

  # ===== OOD test settings =====
  ood:
    enabled: true
    test_freq:
      min: 0.18
      max: 0.35
    test_amp:
      min: 1.0
      max: 2.0
    test_noise_std: 0.10
    test_p_missing: 0.25

model:
  encoder:
    type: mlp
    input_dim: ${data.observation.obs_dim}
    hidden_dims: [256, 256]
    output_dim: ${model.latent.dim_total}
    activation: gelu
    layernorm: true
    dropout: 0.0

  decoder:
    type: mlp
    input_dim: ${model.latent.dim_total}
    hidden_dims: [256, 256]
    output_dim: ${data.observation.obs_dim}
    activation: gelu
    layernorm: true
    dropout: 0.0

  latent:
    # Choose:
    # - euclidean_only (baseline)
    # - circle_only (S^1-native; best for periodic)
    # - torus (S^1)^k, if implemented; default to circle_only with dim=k
    geometry: circle_only

    # total latent dims produced by encoder (before mapping to geometry)
    dim_total: 32

    circle:
      # If you implement as (S^1)^k, set dim = num_sines*2 or num_sines
      # A safe convention: represent each circle component with 2D unit vector (cos,sin)
      representation: unit_vector_2d   # unit_vector_2d | angle
      num_circles: ${data.sequence.num_sines}
      # if using unit vectors, manifold dim_total should be 2*num_circles
      # but we keep dim_total independent and let your code project as needed
      project_eps: 1.0e-6

  dynamics:
    type: tangent_mlp
    hidden_dims: [256, 256]
    activation: gelu
    layernorm: true
    dropout: 0.0
    action_dim: 0

training:
  epochs: 25
  batch_size: 256
  lr: 0.0003
  weight_decay: 0.01
  grad_clip_norm: 1.0

  prediction:
    # 1-step supervised prediction loss + multi-step rollout stability regularizer
    horizon: 1

  losses:
    recon:
      enabled: true
      type: mse
      weight: 1.0

    geodesic_rollout:
      enabled: true
      horizon: 16
      gamma: 0.98
      weight: 0.5

    phase_wrap_penalty:
      # Optional: only meaningful if you implement angle-based latent
      enabled: false
      weight: 0.0

evaluation:
  metrics:
    rollout:
      enabled: true
      horizons: [1, 2, 4, 8, 16, 32, 64]

    phase_error:
      # If you expose ground-truth phase, compute phase error mod 2pi
      enabled: true

    ood:
      enabled: true

  save_figures: true
  figures_dir: ${experiment.work_dir}/figures

runtime:
  deterministic: true
  cudnn_benchmark: false
