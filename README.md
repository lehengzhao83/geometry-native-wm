
---

```markdown
# Geometry-Native World Models

This repository contains the official implementation for:

> **Geometry-Native World Models: Learning Dynamics on Curved Manifolds**

We propose a world modeling framework where latent states live on **curved manifolds** rather than Euclidean space, enabling stable long-horizon rollouts, improved OOD robustness, and structure-preserving dynamics.

---

## 🚀 Key Idea

Conventional world models assume:
```

z_t ∈ ℝ^d ,   z_{t+1} = z_t + f(z_t)

```

However, real-world structure is inherently **non-Euclidean**:
- Hierarchies → Hyperbolic space
- Periodicity → Circle / Torus
- Pose & rotation → Lie groups
- Compositionality → Product manifolds

We instead model:
```

z_t ∈ 𝓜 = 𝓗 × S¹ × ℝ^d
v_t ∈ T_{z_t}𝓜
z_{t+1} = Exp_{z_t}(v_t)

```

---

## 📦 Repository Structure

```

.
├── configs/                # YAML experiment configs
│   ├── toy_hierarchy.yaml
│   ├── toy_periodic.yaml
│   ├── toy_pose.yaml
│   ├── real_video.yaml
│   └── vlm_binding.yaml
│
├── manifolds/              # Geometry implementations
│   ├── euclidean.py
│   ├── hyperbolic.py
│   ├── circle.py
│   ├── product.py
│   └── utils.py
│
├── models/
│   ├── encoder.py
│   ├── dynamics.py
│   ├── decoder.py
│   └── world_model.py
│
├── datasets/
│   ├── toy_hierarchy.py
│   ├── toy_periodic.py
│   ├── toy_pose.py
│   └── real_wrapper.py
│
├── train.py
├── rollout_eval.py
├── ood_eval.py
│
├── run_toy.sh
├── run_real.sh
├── reproduce_main_results.sh
│
└── requirements.txt

````

---

## 🔧 Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

All experiments run **without any external datasets by default**.
If real video data is not available, the code automatically falls back to synthetic pseudo-video.

---

## 🧪 Running Experiments

### Toy Experiments

```bash
bash run_toy.sh
```

### Real / VLM-style Experiments

```bash
bash run_real.sh
```

### Reproduce Main Results (Euclid vs Manifold)

```bash
bash reproduce_main_results.sh
```

---

## 📊 Evaluation

### Long-Horizon Rollout

```bash
python rollout_eval.py --config configs/toy_periodic.yaml --horizon 50
```

### OOD Robustness

```bash
python ood_eval.py --config configs/toy_periodic.yaml
```

---

## 📈 Expected Results

* **Lower rollout error growth** over long horizons
* **Reduced OOD degradation**
* **Stable periodic / hierarchical representations**
* **Elimination of illegal latent transitions**

---

## 📄 Citation

```bibtex
@inproceedings{geometryworldmodel2026,
  title={Geometry-Native World Models: Learning Dynamics on Curved Manifolds},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

---

## ⚠️ Notes

* This codebase prioritizes **correctness and robustness** over maximum speed.
* All geometry operations are explicit and interpretable.
* Designed for ICML/NeurIPS artifact evaluation.

---

## 🤝 Acknowledgements

This work builds upon ideas from:

* World Models
* Riemannian Optimization
* Hyperbolic Representation Learning
* Structured Latent Variable Models

```



