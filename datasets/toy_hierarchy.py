# datasets/toy_hierarchy.py
# Toy Hierarchy World dataset (WORKS OUT-OF-THE-BOX)
#
# What it generates:
# - A rooted b-ary tree of given depth
# - Sequences as random walks on the tree
# - Observations as node "symbols" mapped to embeddings (+ noise), OR one-hot
#
# Designed to match configs/toy_hierarchy.yaml.
#
# Returns (per sample):
#   sample = {
#     "x":        FloatTensor [T, D]        # observations
#     "node_ids": LongTensor  [T]           # latent ground-truth node IDs
#   }
#
# Extra utilities:
# - Tree distance queries for evaluation (hierarchy correlation)
# - Deterministic generation via seed
#
# Dependencies: torch, numpy

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------
# Tree utilities
# ---------------------------
@dataclass
class Tree:
    parent: np.ndarray            # [N] parent node id, parent[root] = -1
    depth: np.ndarray             # [N] depth, depth[root] = 0
    children: List[List[int]]     # children lists
    up: np.ndarray                # [LOGN, N] binary lifting table for LCA
    logn: int

    @property
    def n_nodes(self) -> int:
        return int(self.parent.shape[0])


def build_full_kary_tree(branching_factor: int, depth: int) -> Tree:
    """
    Build a full b-ary rooted tree up to a given depth.
    root depth=0, leaves at depth=depth.

    Node IDs are assigned in BFS order.
    """
    if not isinstance(branching_factor, int) or branching_factor <= 0:
        raise ValueError(f"branching_factor must be positive int, got {branching_factor}")
    if not isinstance(depth, int) or depth < 0:
        raise ValueError(f"depth must be int >= 0, got {depth}")

    # Number of nodes in full b-ary tree:
    # N = (b^(d+1)-1)/(b-1) for b>1, else N = d+1 for b=1
    if branching_factor == 1:
        n_nodes = depth + 1
    else:
        n_nodes = (branching_factor ** (depth + 1) - 1) // (branching_factor - 1)

    parent = np.full((n_nodes,), -1, dtype=np.int64)
    dep = np.zeros((n_nodes,), dtype=np.int64)
    children: List[List[int]] = [[] for _ in range(n_nodes)]

    # BFS expand
    next_id = 1
    for node in range(n_nodes):
        d = dep[node]
        if d >= depth:
            continue
        # add children
        for _ in range(branching_factor):
            if next_id >= n_nodes:
                break
            parent[next_id] = node
            dep[next_id] = d + 1
            children[node].append(next_id)
            next_id += 1

    # LCA binary lifting
    logn = int(np.ceil(np.log2(max(n_nodes, 2))))
    up = np.full((logn, n_nodes), -1, dtype=np.int64)
    up[0, :] = parent
    for k in range(1, logn):
        prev = up[k - 1]
        up[k, :] = np.where(prev >= 0, up[k - 1, prev], -1)

    return Tree(parent=parent, depth=dep, children=children, up=up, logn=logn)


def lca(tree: Tree, a: int, b: int) -> int:
    """Lowest common ancestor for nodes a,b."""
    if a < 0 or b < 0 or a >= tree.n_nodes or b >= tree.n_nodes:
        raise ValueError("lca: node id out of range")

    if tree.depth[a] < tree.depth[b]:
        a, b = b, a

    # lift a up to same depth as b
    diff = int(tree.depth[a] - tree.depth[b])
    for k in range(tree.logn):
        if diff & (1 << k):
            a = int(tree.up[k, a])
            if a < 0:
                return -1

    if a == b:
        return a

    # lift both
    for k in reversed(range(tree.logn)):
        ua = int(tree.up[k, a])
        ub = int(tree.up[k, b])
        if ua != ub:
            a, b = ua, ub

    return int(tree.parent[a])


def tree_distance(tree: Tree, a: int, b: int) -> int:
    """Unweighted tree distance between nodes a,b."""
    c = lca(tree, a, b)
    if c < 0:
        return 0
    return int(tree.depth[a] + tree.depth[b] - 2 * tree.depth[c])


# ---------------------------
# Dataset
# ---------------------------
class ToyHierarchyDataset(Dataset):
    """
    Toy hierarchy dataset.
    Generates sequences on-the-fly (deterministic per index if seed is fixed).

    Observation types:
      - "symbol_embed" / "noisy_embed": fixed embedding table + optional gaussian noise
      - "onehot": one-hot vector of size vocab_size (=n_nodes unless overridden)
    """

    def __init__(
        self,
        tree: Tree,
        size: int,
        seq_len: int,
        obs_type: str = "symbol_embed",
        embed_dim: int = 64,
        vocab_size: Optional[int] = None,
        noise_std: float = 0.05,
        dropout_prob: float = 0.0,
        mode: str = "random_walk_on_tree",
        p_stay: float = 0.10,
        p_parent: float = 0.30,
        p_child: float = 0.60,
        seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be positive int, got {size}")
        if not isinstance(seq_len, int) or seq_len <= 1:
            raise ValueError(f"seq_len must be int > 1, got {seq_len}")
        obs_type = str(obs_type).lower()
        if obs_type not in ("symbol_embed", "noisy_embed", "onehot"):
            raise ValueError(f"Unknown obs_type: {obs_type}")
        if not (0.0 <= float(dropout_prob) < 1.0):
            raise ValueError(f"dropout_prob must be in [0,1), got {dropout_prob}")
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        mode = str(mode).lower()
        if mode not in ("random_walk_on_tree", "leaf_to_root", "root_to_leaf"):
            raise ValueError(f"Unknown sequence mode: {mode}")

        self.tree = tree
        self.size = size
        self.seq_len = seq_len
        self.obs_type = obs_type
        self.embed_dim = int(embed_dim)
        self.noise_std = float(noise_std)
        self.dropout_prob = float(dropout_prob)
        self.mode = mode

        self.p_stay = float(p_stay)
        self.p_parent = float(p_parent)
        self.p_child = float(p_child)
        if self.p_stay < 0 or self.p_parent < 0 or self.p_child < 0:
            raise ValueError("transition probabilities must be >= 0")
        if (self.p_stay + self.p_parent + self.p_child) <= 0:
            raise ValueError("transition probabilities sum must be > 0")

        self.seed = int(seed)
        self.device = device
        self.dtype = dtype

        # vocab size defaults to number of nodes
        if vocab_size is None:
            vocab_size = tree.n_nodes
        self.vocab_size = int(vocab_size)
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")

        # Embedding table for symbol_embed/noisy_embed
        if self.obs_type in ("symbol_embed", "noisy_embed"):
            rng = np.random.default_rng(self.seed)
            table = rng.standard_normal((self.vocab_size, self.embed_dim)).astype(np.float32)
            self.embed_table = torch.from_numpy(table)  # [V, D]
        else:
            self.embed_table = None

        # Precompute leaves for leaf_to_root / root_to_leaf
        self.leaves = np.where(tree.depth == tree.depth.max())[0].astype(np.int64)

    def __len__(self) -> int:
        return self.size

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        # Deterministic per-index RNG
        return np.random.default_rng(self.seed + int(idx) * 10007)

    def _sample_start_node(self, rng: np.random.Generator) -> int:
        # Start anywhere by default
        return int(rng.integers(0, self.tree.n_nodes))

    def _next_node(self, cur: int, rng: np.random.Generator) -> int:
        # choose stay/parent/child with renormalization depending on availability
        has_parent = self.tree.parent[cur] >= 0
        has_child = len(self.tree.children[cur]) > 0

        w_stay = self.p_stay
        w_parent = self.p_parent if has_parent else 0.0
        w_child = self.p_child if has_child else 0.0
        s = w_stay + w_parent + w_child
        if s <= 0:
            return cur

        r = float(rng.random()) * s
        if r < w_stay:
            return cur
        r -= w_stay
        if r < w_parent:
            return int(self.tree.parent[cur])
        # child
        ch = self.tree.children[cur]
        return int(ch[int(rng.integers(0, len(ch)))])

    def _generate_node_sequence(self, rng: np.random.Generator) -> np.ndarray:
        T = self.seq_len
        nodes = np.zeros((T,), dtype=np.int64)

        if self.mode == "random_walk_on_tree":
            cur = self._sample_start_node(rng)
            nodes[0] = cur
            for t in range(1, T):
                cur = self._next_node(cur, rng)
                nodes[t] = cur
            return nodes

        if self.mode == "leaf_to_root":
            leaf = int(self.leaves[int(rng.integers(0, len(self.leaves)))])
            cur = leaf
            nodes[0] = cur
            for t in range(1, T):
                par = int(self.tree.parent[cur])
                cur = par if par >= 0 else cur
                nodes[t] = cur
            return nodes

        # root_to_leaf
        cur = 0
        nodes[0] = cur
        for t in range(1, T):
            ch = self.tree.children[cur]
            if len(ch) == 0:
                nodes[t] = cur
            else:
                cur = int(ch[int(rng.integers(0, len(ch)))])
                nodes[t] = cur
        return nodes

    def _nodes_to_obs(self, nodes: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        if self.obs_type == "onehot":
            # [T, V]
            x = torch.zeros((self.seq_len, self.vocab_size), dtype=self.dtype)
            x[torch.arange(self.seq_len), torch.from_numpy(nodes).long()] = 1.0
        else:
            # [T, D]
            emb = self.embed_table.to(dtype=self.dtype)  # [V,D]
            idx = torch.from_numpy(nodes).long()
            x = emb.index_select(0, idx)  # [T,D]
            if self.obs_type == "noisy_embed" or self.noise_std > 0:
                x = x + torch.randn_like(x) * float(self.noise_std)

        if self.dropout_prob > 0:
            mask = (torch.rand_like(x) > float(self.dropout_prob)).to(x.dtype)
            x = x * mask

        if self.device is not None:
            x = x.to(self.device)
        return x

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = self._rng_for_index(idx)
        nodes = self._generate_node_sequence(rng)
        x = self._nodes_to_obs(nodes, rng)

        node_ids = torch.from_numpy(nodes).long()
        if self.device is not None:
            node_ids = node_ids.to(self.device)

        return {"x": x, "node_ids": node_ids}


# ---------------------------
# Builders (train/val/test + OOD test)
# ---------------------------
def build_toy_hierarchy_datasets(cfg: Dict) -> Dict[str, ToyHierarchyDataset]:
    """
    Build datasets dict: {"train":..., "val":..., "test":..., "test_ood":... (optional)}.

    Expects cfg like configs/toy_hierarchy.yaml loaded into a dict.
    """
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    seed = int(exp_cfg.get("seed", 42))

    # base tree
    tree_cfg = data_cfg.get("tree", {})
    b = int(tree_cfg.get("branching_factor", 3))
    d = int(tree_cfg.get("depth", 5))
    tree = build_full_kary_tree(b, d)

    seq_cfg = data_cfg.get("sequence", {})
    obs_cfg = data_cfg.get("observation", {})
    split_cfg = data_cfg.get("split", {})

    obs_type = str(obs_cfg.get("type", "symbol_embed"))
    vocab_size = obs_cfg.get("vocab_size", "auto")
    if isinstance(vocab_size, str) and vocab_size.lower() == "auto":
        vocab_size = tree.n_nodes
    else:
        vocab_size = int(vocab_size)

    common = dict(
        tree=tree,
        seq_len=int(seq_cfg.get("length", 16)),
        obs_type=obs_type,
        embed_dim=int(obs_cfg.get("embed_dim", 64)),
        vocab_size=int(vocab_size),
        noise_std=float(obs_cfg.get("noise_std", 0.05)),
        dropout_prob=float(obs_cfg.get("dropout_prob", 0.0)),
        mode=str(seq_cfg.get("mode", "random_walk_on_tree")),
        p_stay=float(seq_cfg.get("p_stay", 0.10)),
        p_parent=float(seq_cfg.get("p_parent", 0.30)),
        p_child=float(seq_cfg.get("p_child", 0.60)),
        seed=seed,
        device=None,
        dtype=torch.float32,
    )

    datasets: Dict[str, ToyHierarchyDataset] = {
        "train": ToyHierarchyDataset(size=int(split_cfg.get("train_size", 20000)), **common),
        "val": ToyHierarchyDataset(size=int(split_cfg.get("val_size", 2000)), **common),
        "test": ToyHierarchyDataset(size=int(split_cfg.get("test_size", 2000)), **common),
    }

    # optional OOD dataset with modified tree or transition probs
    ood_cfg = data_cfg.get("ood", {})
    if bool(ood_cfg.get("enabled", False)):
        b2 = int(ood_cfg.get("test_branching_factor", b))
        d2 = int(ood_cfg.get("test_depth", d))
        tree_ood = build_full_kary_tree(b2, d2)

        # keep vocab_size aligned to node count for symbol-based observations
        vocab_size_ood = tree_ood.n_nodes if obs_cfg.get("vocab_size", "auto") == "auto" else int(vocab_size)

        common_ood = dict(common)
        common_ood.update(
            tree=tree_ood,
            vocab_size=vocab_size_ood,
            p_stay=float(ood_cfg.get("test_p_stay", common["p_stay"])),
            p_parent=float(ood_cfg.get("test_p_parent", common["p_parent"])),
            p_child=float(ood_cfg.get("test_p_child", common["p_child"])),
            seed=seed + 99991,
        )
        datasets["test_ood"] = ToyHierarchyDataset(size=int(split_cfg.get("test_size", 2000)), **common_ood)

    return datasets


# ---------------------------
# Expose distance function for evaluation scripts
# ---------------------------
def sample_tree_pairs_and_distances(tree: Tree, num_pairs: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample random node pairs and compute their tree distances.
    Returns:
      pairs: [num_pairs, 2] int64
      dists: [num_pairs] int64
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")
    rng = np.random.default_rng(int(seed))
    n = tree.n_nodes
    a = rng.integers(0, n, size=(num_pairs,), dtype=np.int64)
    b = rng.integers(0, n, size=(num_pairs,), dtype=np.int64)
    pairs = np.stack([a, b], axis=1)
    dists = np.array([tree_distance(tree, int(ai), int(bi)) for ai, bi in pairs], dtype=np.int64)
    return pairs, dists

