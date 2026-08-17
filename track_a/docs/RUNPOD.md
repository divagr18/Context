# RunPod runbook — Track A (W9)

Production runs (G0 LR-pick, G1 grid, gated G2/G3) execute on RunPod RTX 5090
pods. Local 4060 stays dev/test/smoke-only (PLAN 1/HW). This runbook makes a
pod reproducible from a clean image in ~10 minutes of CPU time.

## 0. Pre-flight (before spending)

- [ ] Full test suite green locally: `uv run pytest` (S8 gate).
- [ ] Smoke trained + eval report sane on the 4060 (S6 gate).
- [ ] review-work gate passed (human checkpoint, PLAN 8).
- [ ] Credit budget decided; cost monitor configured for it (§4).

## 1. Pod image requirements

| Item | Requirement |
|---|---|
| GPU | RTX 5090 (sm_120). Verify: `nvidia-smi` |
| Driver | >= 570.65 (current pods ship >> that) |
| CUDA runtime | provided by the torch cu128 wheel; no system CUDA needed |
| Python | 3.11.x |
| Disk | >= 40 GB (repo + shards + checkpoints) |

## 2. Setup on a fresh pod

```bash
git clone <repo-url> coompaction && cd coompaction
pip install uv
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
# EXACT pin as local dev (PLAN Q9): 2.8.0 cu128, cp311
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; assert torch.__version__ == '2.8.0+cu128'; print(torch.cuda.get_device_name(0))"
uv run pytest   # full suite must be green on the pod before ANY training spend
```

Same torch pin on Windows dev and Linux pods = zero win-dev/linux-run drift
(PLAN 11). Tokenizer assets are committed (zero HF-hub network dependency).

## 3. Running a training run + eval

```bash
bash scripts/runpod_entry.sh configs/train_tiny.yaml
```

G0 LR-pick (PLAN 7): run both, pick the winner by best `val/ce` (printed as
`[val]` lines and in the run's TensorBoard), then reuse that LR for the G1
grid:

```bash
bash scripts/runpod_entry.sh configs/train_g0_lr3e-4.yaml
bash scripts/runpod_entry.sh configs/train_g0_lr6e-4.yaml
```

G1 grid (PLAN 8): one config per variant, seeds {11,22,33} from the CLI so
each run dir is unique (the config's own seed is just the default):

```bash
for cfg in configs/train_tiny.yaml configs/train_tiny_V0.yaml \
           configs/train_tiny_V2.yaml configs/train_tiny_V3.yaml \
           configs/train_tiny_V4.yaml; do
  for seed in 11 22 33; do
    bash scripts/runpod_entry.sh "$cfg" --seed "$seed"
  done
done
```

`--seed N` is forwarded to `python -m track_a.train`. All five configs
share the identical non-model recipe with the G0-picked LR 6e-4 (PLAN 7);
the only axis that varies is `model.variant` (attn:FFN ratio). First 2
grid runs are the G1-contingency gate (PLAN 8): check throughput and
val-loss sanity before launching the rest.

The entrypoint: pins-checks torch, regenerates all four shard sets from the
committed split configs (deterministic; ~$0.25 CPU per pod, PLAN 5.2),
trains, then runs the eval battery on best/final checkpoint and writes
`runs/<run_tag>-<seed>/eval_report.json`.

Backend expectation (PLAN 6.1): pods print the SDPA backend probe to stderr.
On Linux sm_120 expect `[SDPA] backend=flash (probed on cuda)`; cuDNN is the
fallback (head_dim <= 128 keeps it viable), math is last resort and should
never win on a 5090. Record the line in run notes; if it says `math`, stop
and investigate before grid spend.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by the entrypoint —
it is Linux-only (Windows ignores it, harmlessly).

## 4. Cost monitor (PLAN 11)

Run alongside training in a second tmux pane:

```bash
python -m track_a.runbook.cost_monitor \
    --budget-usd 100 --rate 0.79 --interval 300 \
    [--runpod-api-key $RUNPOD_API_KEY]
```

- Pings stdout at 25/50/75/100% of the budget (wall-clock x rate; with an
  API key it blends RunPod's live reported spend, taking the max).
- G0+G1 ≈ $35-45 at $0.79/hr (PLAN 8). Set `--budget-usd` to the phase cap.
- At the 75% alert, check gates: if G1-contingency metrics are degenerate,
  stop the pod; nothing auto-recovers spend.

## 5. Artifacts to pull back

Per run dir `runs/<run_tag>-<seed>/`: `config.yaml` (resolved config),
`tb/` (TensorBoard), `best.pt` / `last.pt` / `final.pt`, `eval_report.json`.
Checkpoints are git-ignored by design — copy them off-pod before teardown.

## 6. Known platform notes

- Flash attention is NOT compiled in Windows wheels (MSVC gate): local smokes
  run cuDNN/math; this is expected and flagged in the run log. It does not
  affect pod runs (Linux wheels ship flash).
- Allocator: Windows has no `expandable_segments`, and the default caching
  allocator fragments badly on variable-length batches near full VRAM
  (observed: 14k tok/s clean bench vs <1k tok/s in-loop). On the 4060 box
  set `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` (restores ~5.3k
  tok/s at mb=8). Linux pods keep `expandable_segments:True` (entrypoint
  default). The loop's OOM recovery (micro-batch shrink + skip) remains the
  last-resort safety net on both platforms.
