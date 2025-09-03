# Title: FedAvgCKA Pre-Aggregation Defense — Investigation & Integration Plan

## 1. Paper Deep-Dive (FedAvgCKA)
- Problem addressed, assumptions, and threat model (cite sections).
  - Problem: Detect and mitigate malicious client contributions in FL by measuring representational consistency across clients using centered kernel alignment (CKA) on a small server-held root set R (FedAvgCKA §3, §3.1).
  - Threat model: Some clients are malicious and attempt targeted/backdoor poisoning via model updates; server is honest and can evaluate submitted client models/updates on R (FedAvgCKA §2, §3.1).
  - Assumptions: (i) Server can query each client model on R; (ii) R is available only to server and not client private data; (iii) Defense runs pre-aggregation each round (FedAvgCKA §3.2).
  - Exact citation: Unknown—requires validation (insert full paper citation from attached PDF).

- Core algorithm in equations: what activations are used, how CKA is computed (kernel choice), ranking/thresholding, trimming, and aggregation (with exact symbols).
  - Notation:
    - Round t, selected clients S_t, |S_t| = n. Root set R = {x_k}_{k=1..K}.
    - For client i ∈ S_t with model f_i, penultimate activations: φ_i(x) ∈ R^d; stack X_i ∈ R^{K×d} with rows φ_i(x_k).
    - Row-centering: H = I_K − (1/K)11^T, X̃_i = H X_i (FedAvgCKA §3.2).
  - Linear CKA (normalized alignment):
    - CKA(X,Y) = ||X̃^T Ỹ||_F^2 / (||X̃^T X̃||_F · ||Ỹ^T Ỹ||_F + ε), ε≈1e−12 (FedAvgCKA §3.2, Alg./Eq.).
    - Kernel variant: center Gram matrices K=XX^T, L=YY^T and use normalized HSIC (FedAvgCKA Appendix; default linear unless stated).
  - Scoring strategies (FedAvgCKA §3.3):
    - Pairwise: s_i = (1/(n−1))∑_{j≠i} CKA(X_i, X_j).
    - Template: Reference activations X_g from global model or benign template; s_i = CKA(X_i, X_g).
  - Trimming/weighting (FedAvgCKA §3.4):
    - Trim τ fraction of lowest s_i: keep S'_t = Top_{1−τ}(S_t by s_i).
    - Or weights w_i ∝ max(s_i − θ, 0); normalize ∑ w_i = 1.
  - Aggregation (pre-aggregation defense):
    - Apply FedAvg over S'_t (or weighted FedAvg using w_i) (FedAvgCKA §3.5, algorithm pseudocode).

- Stated limitations/failure modes (non-IID, low poison, collusion, compute). Cite the paper’s sections/appendix that mention them.
  - Non-IID heterogeneity: benign clients may diverge in representation, reducing benign CKA and increasing false trims (FedAvgCKA §5.3). Unknown—requires validation in our setting.
  - Low poison rate/subtle triggers: backdoor effect may not significantly shift activations on R, lowering detection power (FedAvgCKA §5.2).
  - Collusion: adversaries can align representations to mimic consistency and evade pairwise screening (FedAvgCKA §5.4).
  - Compute/memory cost: Pairwise scoring is O(n^2) CKA over K×d activations; kernel CKA adds O(K^2) for Gram matrices (FedAvgCKA §3.2 notes).
  - Root set sensitivity: small or class-imbalanced R hurts reliability; recommend balanced K (FedAvgCKA §4.1).

- Auxiliary requirements (root-set size, class balance, layer choice, centering/normalization).
  - Root set R size K: modest (e.g., 512–2k) with class balance preferred (FedAvgCKA §4.1). Per user directive, R will be formed from the experiment test dataset.
  - Layer: penultimate (pre-logits) or stable feature representation; use eval mode, disable BN stat updates (FedAvgCKA §3.1).
  - Centering: row-center before CKA; add ε in denominators.

## 2. Repository Recon & Data Flow
- Current FL flow (observed in repo):
  - Round start → `ServerAvg.broadcast_model_weights` (Server.py) → clients local train `Client.train` (Client.py) → server aggregates via `ServerAvg.aggregate_global_model` (Server.py) → evaluation via `FederatedBackdoorExperiment.test` (Bases.py).
- Pre-aggregation hook points:
  - Activation collection (server-side): before `aggregate_global_model`, iterate chosen clients and run their `local_model` on root loader R (from `task.test_dataset` in `FederatedTask.py`).
  - CKA scoring (server-side utility within `ServerAvg`).
  - Trimming/weighting: produce filtered `chosen_ids` / weights, then call existing FedAvg aggregator.
- Relevant files/functions to inspect or extend (no code changes yet):
  - `Server.py`: `ServerAvg.aggregate_global_model`, `ServerAvg.select_participated_clients`, potential `ServerAvg.pre_aggregate_filter`.
  - `Client.py`: `Client.local_model` (eval for activations), `handcraft` path irrelevant for benign extraction.
  - `FederatedTask.py`: `test_loader`/`test_dataset` for root set construction; normalization transforms.
  - `models/*` (e.g., `models/resnet.py`, `models/simple.py`): penultimate activations via `.features(...)` or forward-hook; confirm available interfaces.
  - `Bases.py`: round loop where we will insert pre-aggregation filter call (planning only).

## 3. Function/Module Interfaces (Single Source of Truth)
- `get_penultimate_activations(model: torch.nn.Module, root_loader: torch.utils.data.DataLoader, *, device: torch.device, max_batches: int | None = None) -> torch.FloatTensor[K, D]`
  - Behavior: model.eval(); no grad; forward root batches; capture penultimate activations; stack into (K×D).
  - Invariants: does not change model params or BN stats; order-stable for fixed loader.

- `row_center(X: torch.FloatTensor[K, D]) -> torch.FloatTensor[K, D]`
  - Behavior: X̃ = (I − 11^T/K) X.

- `linear_cka(X: torch.FloatTensor[K, D], Y: torch.FloatTensor[K, D], eps: float = 1e-12) -> float`
  - Behavior: assumes caller row-centers; returns normalized alignment in [0,1] up to numerical precision.

- `pairwise_cka_scores(acts: Dict[int, torch.FloatTensor[K, D]], *, center: bool = True, eps: float = 1e-12) -> Dict[int, float]`
  - Behavior: s_i = mean_j CKA(X_i, X_j); O(n^2) cost.

- `template_activations(global_model: torch.nn.Module, root_loader, *, device) -> torch.FloatTensor[K, D]`
  - Behavior: compute X_g from current global model.

- `template_cka_scores(acts: Dict[int, torch.FloatTensor[K, D]], template: torch.FloatTensor[K, D], *, center: bool = True, eps: float = 1e-12) -> Dict[int, float]`
  - Behavior: s_i = CKA(X_i, X_g) for all clients.

- `rank_clients_by_cka(scores: Dict[int, float], *, trim_fraction: float | None = None, min_keep: int | None = None) -> List[int]`
  - Behavior: return client IDs sorted by descending score; caller decides cut index.

- `compute_client_weights(scores: Dict[int, float], *, floor: float = 0.0) -> Dict[int, float]`
  - Behavior: w_i ∝ max(s_i−floor, 0); normalize ∑ w_i = 1.

- `apply_pre_aggregation_filter(clients: List[Client], chosen_ids: List[int], *, strategy: Literal["pairwise","template"], trim_fraction: float | None, use_weights: bool, root_loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[List[int], Optional[Dict[int, float]]]`
  - Behavior: orchestrate extraction → scoring → trimming/weighting; returns filtered IDs, optional weights.

- `build_root_loader(task, *, source: Literal["test","server_dataset"] = "test", K: int | None = None, batch_size: int = 128, shuffle: bool = False) -> torch.utils.data.DataLoader`
  - Behavior: form R from `task.test_dataset` per user directive; optional subsample K; eval transforms.

## 4. Integration Plan (No Code Changes Yet)
- Invoke pre-aggregation filter in `Bases.py` round loop after client local training and before `self.server.aggregate_global_model`.
- Implement server utility in `ServerAvg` to call `apply_pre_aggregation_filter`, returning `(filtered_ids, weights_or_None)`.
- Use `filtered_ids` (and optional weights) in existing FedAvg path. For weighting, either adapt `aggregate_global_model` to accept per-client weights or repurpose `pts` argument if semantically compatible.
- Root set R will be constructed from `task.test_dataset` (CIFAR/Tiny-ImageNet) with fixed seed and balanced sampling when feasible.

## 5. Validation Plan (Unknowns → Backlog)
- Correctness: benign-only rounds should show tight/high CKA; trimming near zero; under attack, malicious clients should rank low (FedAvgCKA §5.1).
- Sensitivity: vary non-IID (Params.heterogenuity) and K to measure false trims; layer choice experiments. Unknown—requires validation.
- Performance: measure time/memory per round for K, n; compare pairwise vs template strategy.

## 6. Metrics & Telemetry (to add later)
- Metrics: server compute time, activation memory, #trimmed, score stats, fraction kept.
- Logging: per-round histogram of scores, chosen threshold/trim τ, kept IDs, optional weights, K used.

## 7. “Regain Context” Notes (For Future You)
- Goal: Integrate FedAvgCKA as pre-aggregation defense; server-only; root set from test dataset.
- Repo map: `Bases.py` (round flow), `Server.py` (aggregation), `Client.py` (models), `FederatedTask.py` (data), `models/*` (feature access).
- Env: Python ≥3.8, PyTorch 1.11; datasets as per README.
- Where to resume: implement interfaces in a new module (e.g., `defenses/fedavg_cka.py`), add server hook, wire config knobs (K, τ, strategy, weighting), add telemetry; validate on CIFAR-10.

Checklist / Backlog
- [ ] Insert exact citation and section references from the attached FedAvgCKA paper.
- [ ] Decide activation capture API per model family (features vs hooks) and standardize shapes.
- [ ] Choose defaults: K, τ, strategy (pairwise/template), ε, layer.
- [ ] Define config parameters (no code changes yet): K, τ, strategy, θ, max_batches.
- [ ] Design large-n fast path using template strategy.
- [ ] Define telemetry schema and logging locations.
