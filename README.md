# Coompaction — Track A

Controlled ablation: does shifting capacity from FFN toward attention improve
needle survival under context compression at fixed parameter budget?

- **PLAN.md** — frozen, approved implementation plan (decisions register, canonical
  order spec, streaming semantics, grammar, compute gates).
- Spec origin: `track_a_architecture_ablation.md` (handoff doc, external).
- Dev: Windows RTX 4060 (build/test/smoke) · Production runs: RunPod RTX 5090.

## Quickstart
```powershell
uv venv .venv --python 3.11.8
uv pip install --python .venv\Scripts\python.exe -e .
uv pip install --python .venv\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv run pytest
```
