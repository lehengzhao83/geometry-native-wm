
---

# 一、Figures 设计（Main Paper + Appendix）

下面的 figures 设计**严格对齐 ICML 审稿人阅读路径**：
👉 先理解方法
👉 再看到机制
👉 再看到 quantitative gain
👉 最后看到 failure mode 被修复

---

## **Figure 1 — Geometry-Native World Model Overview（核心总览图）**

**目的（Reviewer 1 / Area Chair）：**

> “这不是在 embedding 上加正则，而是 **state space 本身变了**。”

### 图内容结构（左 → 右）：

**(a) Euclidean World Model（对照）**

```
x_t → Encoder → z_t ∈ R^d
           ↓
     z_{t+1} = z_t + f(z_t)
           ↓
        Decoder
```

标注问题：

* Linear interpolation
* Drift in long rollout
* Illegal states

---

**(b) Geometry-Native World Model（你）**

```
x_t → Encoder → z_t ∈ M = H × S¹ × R^d
           ↓
   v_t ∈ T_{z_t}M
           ↓
 z_{t+1} = Exp_{z_t}(v_t)
           ↓
        Decoder
```

强调：

* Product manifold
* Tangent update + Exp map
* Closed-form geometry-aware rollout

📌 **必须画 Product Manifold**：
双曲（树）+ 圆（周期）+ 欧式（噪声）

---

### Caption（可直接用）

> **Figure 1:** Overview of Geometry-Native World Models.
> Unlike conventional world models that assume Euclidean latent states, our approach defines the world state on a product manifold and performs dynamics via tangent-space updates and exponential maps, enabling stable long-horizon rollouts and structure-preserving transitions.

---

## **Figure 2 — Toy Mechanism Validation（Hierarchy / Periodic）**

**目的（Reviewer 2）：**

> “你说 geometry 对症，那我要看到‘对症’的证据。”

### (a) Hierarchy World（双曲）

* x-axis：true tree distance
* y-axis：latent geodesic distance
* 对比：

  * Euclidean latent（散点、非单调）
  * Hyperbolic latent（近似线性）

📌 **这张图是杀伤力最大的机制图之一**

---

### (b) Periodic World（S¹）

* x-axis：time
* y-axis：phase error
* horizon = 50 / 100
* Euclidean：phase wrap 崩溃
* Circle manifold：稳定

---

### Caption

> **Figure 2:** Geometry-task alignment on synthetic worlds.
> Hyperbolic geometry faithfully preserves hierarchical distances, while circular manifolds stabilize periodic dynamics, demonstrating that selecting geometry aligned with world structure is crucial for robust modeling.

---

## **Figure 3 — Long-Horizon Rollout Error Curve（核心 quantitative）**

**目的（Area Chair）：**

> “你比 baseline 好在哪？是不是只在短期？”

### 图形式：

* x-axis：rollout horizon τ
* y-axis：mean squared geodesic distance
* 多条曲线：

  * Euclidean
  * Euclidean + regularization
  * Geometry-native (Product)

📌 **你必须画 log-scale 或 error growth rate**

---

### Caption

> **Figure 3:** Long-horizon rollout stability.
> Geometry-native world models significantly reduce error accumulation over long horizons, whereas Euclidean models exhibit exponential drift.

---

## **Figure 4 — OOD Robustness（ICML 必要）**

* Bar chart or line chart
* In-domain vs OOD
* 指标：

  * latent error
  * reconstruction error
* 报告 **OOD / IND ratio**

📌 强调：不是 absolute 数值，是 **robustness gap**

---

## **Figure 5 — Failure Case Visualization（解释性）**

**目的（Reviewer 3）：**

> “你到底修复了什么 failure？”

示例：

* 同一输入
* Euclidean rollout vs Manifold rollout
* 展示：

  * 计数错误
  * 位姿漂移
  * 非法插值

---

## Appendix Figures（强烈建议）

* **Curvature sweep**
* **Exp map vs retraction**
* **Ablation of product components**
* **Latent factor interpretability（correlation heatmap）**

---

# 二、README.md（可直接用）

下面是 **完整 README.md**，你可以一字不改直接放 GitHub。

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



