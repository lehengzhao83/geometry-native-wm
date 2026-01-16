# geometry-native-wm
geometry-native-wm-ICML2026

下面是**完整、可直接复制到 GitHub 的 `README.md`**。
我已经注意了 **Markdown 数学公式（LaTeX）表达的兼容性**（GitHub 渲染 OK），结构和措辞也对齐 **ICML / NeurIPS Artifact Evaluation** 的标准。

---

````markdown
# Geometry-Native World Models

> **Geometry-Native World Models: Learning Dynamics on Curved Manifolds**

This repository contains the official implementation of **Geometry-Native World Models**, a framework that models latent world states on **curved manifolds** instead of Euclidean space, enabling **stable long-horizon rollouts**, **improved OOD robustness**, and **structure-preserving dynamics**.

The codebase is **fully runnable end-to-end**, includes **synthetic and real-style benchmarks**, and is designed to satisfy **top-tier conference artifact evaluation** requirements.

---

## 🚀 Core Idea

Most existing world models implicitly assume that latent states live in Euclidean space:

```math
z_t \in \mathbb{R}^d, \quad z_{t+1} = z_t + f_\theta(z_t, a_t)
````

However, real-world structure is inherently **non-Euclidean**:

* Hierarchies and trees → **Hyperbolic space**
* Periodic phenomena → **Circle / Torus**
* Pose and rotation → **Lie groups**
* Compositional structure → **Product manifolds**

We propose to model the world state as a point on a **product manifold**:

```math
z_t \in \mathcal{M}
= \mathcal{H}^{d_h} \times (S^1)^{d_p} \times \mathbb{R}^{d_e}
```

Dynamics are defined **natively on the manifold** via tangent-space updates:

```math
v_t = f_\theta(z_t, a_t) \in T_{z_t}\mathcal{M}, \quad
z_{t+1} = \operatorname{Exp}_{z_t}(v_t)
```

This ensures **closed-form, geometry-consistent state transitions**, preventing illegal interpolations and latent drift.

---

## ✨ Key Contributions

* **Geometry as State Space**
  Latent states *live on manifolds*, not in Euclidean space with post-hoc regularization.

* **Product Manifold Factorization**
  Different world factors (hierarchy, periodicity, pose, noise) are embedded into appropriate geometric components.

* **Stable Long-Horizon Rollout**
  Exponential-map updates eliminate drift and error explosion over long horizons.

* **Robustness to OOD Shifts**
  Manifold constraints preserve structure under distribution shift.

---

## 📦 Repository Structure

```text
.
├── configs/                    # Experiment configurations (YAML)
│   ├── toy_hierarchy.yaml
│   ├── toy_periodic.yaml
│   ├── toy_pose.yaml
│   ├── real_video.yaml
│   └── vlm_binding.yaml
│
├── manifolds/                  # Geometry implementations
│   ├── euclidean.py
│   ├── hyperbolic.py
│   ├── circle.py
│   ├── product.py
│   └── utils.py
│
├── models/                     # World model components
│   ├── encoder.py
│   ├── dynamics.py
│   ├── decoder.py
│   └── world_model.py
│
├── datasets/                   # Synthetic + real-style datasets
│   ├── toy_hierarchy.py
│   ├── toy_periodic.py
│   ├── toy_pose.py
│   └── real_wrapper.py
│
├── train.py                    # Training entry point
├── rollout_eval.py             # Long-horizon rollout evaluation
├── ood_eval.py                 # OOD robustness evaluation
│
├── run_toy.sh                  # Run all toy experiments
├── run_real.sh                 # Run real / VLM-style experiments
├── reproduce_main_results.sh   # Reproduce Euclid vs Manifold results
│
└── requirements.txt
```

---

## 🔧 Installation

We recommend using a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All experiments **run without any external datasets by default**.
If real video data is unavailable, the code **automatically falls back** to synthetic pseudo-video to guarantee executability.

---

## 🧪 Running Experiments

### Toy Benchmarks (Hierarchy, Periodic, Pose)

```bash
bash run_toy.sh
```

This script will:

1. Train the model
2. Run long-horizon rollout evaluation
3. Run OOD evaluation

---

### Real / VLM-Style Benchmarks

```bash
bash run_real.sh
```

If no real video frames are provided, a **FakeData-based pseudo-video** is used automatically.

---

### Reproduce Main Results (Euclidean vs Geometry)

```bash
bash reproduce_main_results.sh
```

This script runs a grid of configurations (if present), e.g.:

* Euclidean latent world model
* Hyperbolic world model
* Circular / periodic world model
* Product manifold world model

Missing configurations are **safely skipped**.

---

## 📊 Evaluation Protocols

### Long-Horizon Rollout

We evaluate error accumulation over rollout horizon ( \tau ):

```math
\mathbb{E}\left[d_{\mathcal{M}}\bigl(\hat z_{t+\tau}, z_{t+\tau}\bigr)^2\right]
```

Run manually with:

```bash
python rollout_eval.py \
  --config configs/toy_periodic.yaml \
  --horizon 50
```

---

### OOD Robustness

We measure **in-domain vs OOD degradation**, reporting both absolute error and ratios:

```math
\text{OOD Ratio} = \frac{\text{Error}_{\text{OOD}}}{\text{Error}_{\text{IND}}}
```

Run with:

```bash
python ood_eval.py \
  --config configs/toy_periodic.yaml
```

---

## 📈 Expected Results

You should observe:

* Slower error growth over long horizons
* Improved OOD robustness
* Stable periodic and hierarchical representations
* Elimination of illegal latent transitions
* Clear advantage of geometry-aligned latent spaces over Euclidean baselines

---

## 🧠 Design Principles

* **Correctness > Tricks**
  Geometry is explicit, not implicit.

* **Mechanism-Oriented Evaluation**
  Synthetic worlds are designed to validate *why* geometry helps.

* **Artifact-Ready**
  Every script is runnable on a clean machine.

---

## 📄 Citation

If you use this code, please cite:

```bibtex
@inproceedings{geometryworldmodel2026,
  title     = {Geometry-Native World Models: Learning Dynamics on Curved Manifolds},
  author    = {Anonymous},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---

## ⚠️ Notes

* The implementation prioritizes **clarity and robustness** over maximal speed.
* Geometry operations are explicit and interpretable.
* Designed for **ICML / NeurIPS artifact evaluation** and reproducibility.

---

## 🤝 Acknowledgements

This work is inspired by research on:

* World Models
* Riemannian Optimization
* Hyperbolic Representation Learning
* Structured Latent Variable Models

```
