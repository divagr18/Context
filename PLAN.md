# Track A — Final Implementation Plan

**Question:** Holding parameter count and training data fixed, does shifting capacity from FFN
toward attention improve needle retrieval / entity binding / state-supersession accuracy under
context compression?

**Status:** Plan frozen pending approval. No code written. All Q2–Q7 amendments from the
2026-08-13 review are incorporated (marked **[A#]**). Additional decisions from the deep-pass
are marked **[LOCK]** — veto any of them before Phase 0 starts.

---

## 1. Locked decisions (full register)

| ID | Decision |
|---|---|
| HW | RunPod RTX 5090 (~$0.75–1.0/hr → ~100h). Local RTX 4060 (8GB) = dev/tests/smoke only. No FlashAttention on Windows → SDPA everywhere; flagged in run logs per spec §6.1. |
| Q2+ | Canonical C\* order = queried-first, type-rank, **position-ascending tie-break**, **chains atomic**, **decoys last, cut entirely at the margin, never used as padding**. One inspectable function in `needle_gen/generate.py`. See §2. |
| Q3+ | Per-bucket `max_seq = doc_len + max_budget + 10% slack`. **short+medium share ONE single-shot max_seq config** (θ keyed to max\_seq value, not doc length). θ = 1e4 if max_seq ≤ 4096 else 5e5. |
| Q4+ | Streaming = delta edit-ops. Parser state = **append-only op log**; current values derived by fold; **SUPERSEDE retains old value with superseded-flag**. Grading checks current-value AND prior-value recoverability. See §3. |
| Q5+ | QA-probe is diagnostic-only: tracked under `diag/` namespace (e.g. `diag/qa_probe_acc`), never in loss, never in the primary RQ1 comparison or headline plots. |
| Q6+ | Recall-weighted CE (§7.2 mechanism): every 100 steps, 8-sample greedy Ĉ, parse, `extra_weight = λ_drop·(1−r) + λ_waste·h` applied to CE(C\*). λ_drop=5.0, λ_waste=1.0. **Log r itself** (`train/recall_subsample`) and h (`train/halluc_frac_subsample`). |
| Q7+ | Hand-authored banks. **Explicit work item with human review gate.** Revisit trigger: Test-OOD tracks Test-ID suspiciously closely → one-time LLM pass. Plus an automated lexical-distinctness test (§5.6). |
| Q8 | TensorBoard; all loss components, needle metrics by type×distance×ratio, attention entropy, per-run resolved config. |
| Q9 | git init at `D:\Coompaction`; uv venv (py 3.11.8); torch pinned 2.8.x cu128 (same pin on RunPod image — sm_120 needs ≥2.7). |
| Q10 | Decoys = **true** document facts, unqueried. Emission tolerated (no waste penalty). Fabricated facts (not in doc at all) penalized + reported. |
| Q11 | Aux heads: disjoint subsets, last ⅓ of layers, default 2 heads/loss, config-driven. |
| Q12 | Smoke scale (~5M params, ~50M tokens) on local 4060 before any cloud spend. |
| TOK | **[LOCK]** GPT-2 tokenizer (vocab 50257) + 21 added special tokens = 50278, identical across variants. Tokenizer files (vocab.json/merges.txt, ~2MB) **committed to the repo** → zero HF-hub network dependency on pods. All weights random-init (from scratch); no pretrained weights anywhere. |
| BATCH | **[LOCK]** Effective batch = 262,144 tokens; micro-batch auto-tuned at step 0 (max sequences fitting max_seq with 20% VRAM headroom), grad-accum to exact effective batch; micro-batch/accum logged per run, abort if infeasible. |
| BUDGET TOKENS | **[LOCK]** `<budget=N>` ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384} (9 tokens). Eval maps target doc_len/ratio to nearest token. Compression budgets are upper bounds, not fill targets (§2). |
| SEEDS | **[LOCK]** seeds {11, 22, 33}; ≥2 seeds per grid point; report mean±std. |

---

## 2. Canonical C\* order — exact algorithm (single function in `generate.py`)

`canonical_order_and_render(fact_db, budget_tokens, tokenizer) -> C*`

1. **Units.** Each unit is one fact; a state-transition chain (k≥1 transitions) is ONE unit,
   never split. Decoys are units with class=1; queried facts class=0.
2. **Sort key:** `(class, type_rank, unit_pos, fact_id)` where
   `type_rank = {state_transition:0, binding:1, exact_value:2, relational:3, negative:4, uncertainty:5}`
   and `unit_pos` = position of the chain's **first** transition (earlier-stated wins ties [A2]).
3. **Inclusion = longest prefix, no skipping.** Render units in sorted order (token-exact, via
   tokenizer); include while cumulative tokens ≤ budget; stop at first unit that doesn't fit.
   No size-based skipping (keeps the order truly canonical); C\* may end under budget — a
   budget is a ceiling, never padded **[A2]**.
4. **Decoys** are only reachable after ALL queried units; when the prefix stops inside or
   before the decoy block, all remaining decoys are cut entirely **[A2]**.
5. **Chain rendering (inside its atomic unit):** final statement first —
   `FACT e.a = v_final supersedes=v_orig pos=p_last`, then intermediate transitions
   newest-first. Since the unit is atomic, truncation never splits it.
6. Ground truth at eval = `C*(budget)` per budget. The rate-distortion curve measures
   "of what must survive at this compression, what did." C\* fact count per budget is logged as
   `info_pressure` context on every curve.
7. **[A2 rationale metric]** `primary/salience_inversions[ratio]`: count of docs where the model
   dropped ≥1 queried fact AND emitted ≥1 decoy. This surfaces "picked decoy over real fact at
   the margin" explicitly rather than smoothing it into aggregate survival.

Identical function, identical constants for every variant, split, and budget bucket.

---

## 3. Streaming semantics — append-only state log **[A4]**

- **Ground-truth construction (generator side).** Per window: state policy = the SAME §2
  canonical order, budget-truncated. Window facts merge in; overflow drops lowest-priority
  units → deterministic diff(prev_state, next_state) emits edit-ops:
  `UPSERT ENTITY / UPSERT FACT / SUPERSEDE FACT (old => new) / UPSERT REL / DROP … /
  MARK UNRESOLVED / RESOLVE`. Entity ids are stable across windows (carried forward).
- **Parser side.** State = append-only list of applied ops. Folded view maps
  `(entity_name, attribute)` → history list `[(value, pos, superseded:bool), …]`.
  `SUPERSEDE` appends the new current and **flags the prior value superseded, retaining it**.
  `DROP` removes from the current view (history retained in the log; grading treats
  DROPped content as gone).
- **Grading contract (generation and grading agree):** for every queried chain,
  (a) current-value question — answer = final value; (b) prior-value question
  ("what was it before…") — answer = superseded value, recoverable from history.
  Both count under state-transition survival, reported as
  `primary/survival_state_current` and `primary/survival_state_prior`.
  Chains ≥2 hops always get a prior-value question; 1-hop chains get one ~50% of the time.
- Windows are **scene-aligned** (split snaps to nearest scene boundary within ±10% of the
  2048-token window target), so facts are never severed mid-mention. Overlap 10%.
  A fact belongs to the window containing its anchor sentence.
- First-window targets bootstrap from empty state — uniformly edit-ops, no special case.

---

## 4. Grammar (final)

```
# full-state records (single-shot targets; also valid streaming bootstrap content)
ENTITY <eid> type=<t> name=<name>
FACT <eid>.<attr> = <value> [supersedes=<value>] [hedged] pos=<scene_idx>
REL <eid_a> <rel> <eid_b> pos=<scene_idx>
NEG <eid>.<attr> denied pos=<scene_idx>
UNRESOLVED <eid>.<attr> candidates=[<v1>@<pos1>, <v2>@<pos2>]

# streaming edit-ops
UPSERT ENTITY|FACT|REL|NEG … | SUPERSEDE FACT <eid>.<attr> : <old> => <new> pos=<i>
DROP FACT|REL|NEG … | MARK UNRESOLVED … | RESOLVE <eid>.<attr> = <value> pos=<i>
```

- `[hedged]` flag on FACT covers uncertainty sub-type (a) single-source hedge; conflicts use
  UNRESOLVED. Picking a side on UNRESOLVED grades **wrong**; answer "unresolved" grades right.
- **Value canonicalization (locked):** pool values match `[A-Za-z0-9_$./:%-]+` (no spaces);
  matching normalizes whitespace→`_`, strips digit-group commas, trims; case-sensitive.
- **Entity binding grading:** eids are model-chosen labels; grader resolves eid→name via
  ENTITY records, then matches facts by (resolved name, attribute/relation, value).
  A FACT citing an eid with no ENTITY record = unbound = counted dropped (this is the binding
  signal itself).
- Malformed lines → `ParseFailure(line, reason)`; `parse_fail_rate` reported separately from
  factual errors. **[LOCK]** §9.3 fallback trigger: parse failures > 15% of eval errors →
  switch that metric to embedding-similarity matching, flagged in run notes.
- `RESOLVE` exists in grammar for completeness; the v1 generator never resolves conflicts
  (eval type stays unresolved per spec §5.2.5b).

---

## 5. Data system (`needle_gen/`)

### 5.1 Assets (work item W2, human-reviewed **[A7]**)
- 2 domains × **16 template families** × exactly 8 paraphrases = 256 templates, static JSON.
  - Domain A "fictional software project updates": FILL-01..04 (fact-free filler), EXACT-01/02,
    REL-01/02, STATE-01, NEG-01, UNRES-01 (hedge), UNRES-02 (conflict side, emitted twice with
    different sources/values), BIND-01 (introduction), BIND-02 (deferred reference).
  - Domain B "logistics/shipping ops": structurally analogous 16, lexically disjoint.
- Pool files per domain: ≥40 invented person names, ≥40 projects/systems/depots, attribute
  value pools per attribute type (ports, versions, ISO dates, IDs, paths, amounts) — all
  values space-free by regex rule (see §4).
- Question templates: ≥2 phrasings per fact type + prior-value phrasings for chains +
  multi-hop phrasings; answers deterministic from fact DB.
- **Lexical-distinctness test [A7]:** automated check that template+pool n-gram overlap
  between domains is below a fixed threshold (character 3-gram Jaccard < 0.15); failing this
  test is the objective gate for the "revisit with LLM pass" trigger.

### 5.2 Generation (W3)
- Docs = scene sequences from families, token-accurate length packing (±5% of target:
  short 2k / medium 8k / long 32k; xlong 128k stretch only).
- Density ≈ 1 queried fact / 100 tokens; 1–3 decoys per queried fact; ~15% of docs carry 1–2
  multi-hop question structures; chains: 2-hop majority, ≥3-hop subset placed at long distances.
- Binding needles enforce distance buckets near(<500)/mid(500–4k)/far(4k–20k)/extreme(>20k,
  long docs only) by filler padding; actual distances verified post-generation and recorded.
- Annotations emitted for aux losses: token spans of entity mentions (+alias→eid coref map),
  value spans, negation/hedge cue spans, queried/decoy labels.
- Fresh entity set per doc (no cross-doc reuse).
- Splits: train = paraphrase idx 0–5, val = 6, test-id = 7, test-ood = Domain B (all idx,
  never in any training/dev loop). Split configs (seed + families + idx ranges + doc mix)
  committed; raw corpora regenerated per pod from config (regen cost ≈ $0.25 CPU-time/pod).

### 5.3 Dataset layer (W5)
- JSONL shards (token IDs + annotations + per-split C\* renders at all budget tokens),
  streaming reader + shuffle buffer. Training sample = (doc, sampled budget token, C\*).
  Budget sampled per-sample from the 9-token set, uniform within feasible ratios (2x–32x).
- Streaming samples precomputed per window: (state_k render, window, ops_{k+1}).

---

## 6. Model & solver (W6)

- Pre-norm decoder-only: RMSNorm, RoPE (θ per §1), GQA/MHA, SwiGLU, causal SDPA
  (flash/cudnn backend → mem-efficient fallback → eager for entropy diagnostics),
  bf16 autocast + fp32 master weights/optimizer states.
- Softcap flag (tanh on attention logits) default OFF; enabled only via §8.4 deviation
  protocol for V4, documented per-run.
- Softmax temperature **[LOCK]**: 1/sqrt(head_dim) standard; logged per variant since head_dim
  varies with the solver (not a free hyperparameter — it's entailed by the ratio).
- `configs/solve_ratio.py`: inputs (d_model, n_layers, target_ratio, target_params) →
  (n_heads, head_dim∈{32,64,96,128}, n_kv_heads, ffn_hidden mult-of-128). Tied embeddings.
  Embedding counted ONCE in the budget (50278×d_model). Acceptance: params ±2%, ratio ±2pts;
  if infeasible, relax ratio first and log. Unit tests assert acceptance on all 5 variants ×
  all 3 scales before any training.
- Variant grid (per scale): V0 15% attn · V1 25% · V2 40% · V3 55% · V4 70%.
- Scales: tiny d512/L8 (~50M, max_seq 14k single-shot, 20k streaming) · small d768/L12
  (~150M) · base d1024/L16 (~300M, stretch).
- Aux losses (W6, all toggleable, weight 0.1 default): coref InfoNCE over attention of
  designated heads; temporal-order probe classifier; negation binary classifier;
  numeric/entity-binding attention supervision; needle-salience (queried vs decoy span)
  prediction. Head assignment disjoint, last ⅓ layers, config-driven.
- Entropy diagnostics: every 500 steps, 2 sequences × 512-token prefixes through the eager
  path; per-layer mean attention entropy logged (per-head optional histogram).

---

## 7. Training recipe (W7) — identical across variants at a scale point

- AdamW β=(0.9,0.95), wd=0.1, clip 1.0, cosine→10%, warmup 2% of steps, bf16.
- **LR pick first:** V1_standard × {3e-4, 6e-4} × 50M tokens → pick by val CE → one shared
  schedule for ALL variants at that scale (no per-variant LR).
- Token budgets: tiny **500M** (see §8), small 5B (gated, likely out of budget), smoke 50M.
- Total loss = CE + recall-shaping (§1/Q6+, scheduled every 100 steps on 8-doc subsample)
  + Σ enabled aux losses. Every component logged separately.
- Checkpoints: best-val + last; eval battery (§9) on best-val.
- Run dirs `runs/<run_id>/config.yaml` = fully resolved config + needle_gen config + seed.

---

## 8. Compute plan & gates

Estimates @ 5090 ≈ 40k tok/s (verified live at gate G1).

| Phase | Runs | Tok/run | Hours |
|---|---|---|---|
| G0 LR pick | 2 | 50M | ~1 |
| G1 tiny grid: 5 variants × 2 seeds | 10 | 500M | ~35–40 |
| G2 streaming (V1 + winner, long docs, 2 seeds) | 4 | 500M | ~10 |
| G3 aux screen: 6 configs × 1 seed @300M → promote top 2 × 2 seeds @500M | 6+4 | 300/500M | ~27 |
| Reserve / V4 instability re-runs | — | — | ~25 |
| **Total** | | | **~98h ≤ 100h** |

**Gates (human checkpoints, pre-committed):**
- **G1-contingency:** after the FIRST 2 grid runs, if throughput < 25k tok/s or val loss is
  degenerate → pause, re-plan budgets (fallback: 1-seed screen of all 5, full seeds on
  top-2 only).
- **G1→G2:** proceed to streaming/aux only if grid shows a non-noise trend (winner−baseline
  gap > 2× pooled seed std on Test-ID primary). Otherwise: declare null result, still run
  Test-OOD confirmation, stop cloud spend.
- **G2/G3→small-scale promotion:** only if effect survives OOD; `small`/`base`/xlong are
  stretch — flag before spending.

---

## 9. Eval battery (W8)

For each ckpt × {test-id, test-ood} × doc length × ratio {2,4,8,16,32}:
greedy C → parse → score. Primary namespace (`primary/…`):
`needle_survival_exact[type][distance][ratio]`, `needle_survival_partial[…]`,
`survival_state_current`, `survival_state_prior`, `hallucination_rate`,
`decoy_emission_rate`, `salience_inversions`, `parse_fail_rate`, `info_pressure`.
Diagnostic namespace (`diag/…` only): `qa_probe_acc[type]` (~200 Qs/split, same-model,
greedy, format `<doc> C </doc> <Q> q </Q> <A> a </A>`; catches parser-gaming per [A5]).
Streaming eval replays windows; metrics on folded final state (+ optional mid-points).
Deliverable: rate-distortion curves per variant (matplotlib), mean±std over seeds,
Test-ID headline with Test-OOD as validity gate/tie-breaker. Honest null-result write-up
is a first-class outcome.

---

## 10. Build sequence (work items, dependencies)

| W | Deliverable | Deps |
|---|---|---|
| W1 | Repo scaffold: git, uv venv, torch-pin, pytest/tensorboard, README skeleton | — |
| W2 | Authored assets: banks, pools, question templates, lexical-distinctness test | W1 |
| W3 | `needle_gen` core: types/fact-DB/questions/annotations, doc assembler, §2 canonical function, split configs | W1 (fixtures ok before W2 lands) |
| W4 | `parse_compact.py`: grammar, op-log fold w/ superseded history, malformed handling + tests | W1 |
| W5 | Dataset layer: shards, collators, budget sampler, streaming window samples | W3 |
| W6 | `solve_ratio.py` + architecture + aux losses + configs (param/ratio unit tests) | W1 |
| W7 | `train.py` + budget-conditioned generation + diagnostics | W5, W6 |
| W8 | `eval.py` + metrics + curves + report JSON | W4, W7 |
| W9 | Runbooks: RunPod setup (same torch pin), bash entrypoints, cost monitor script | W7 |
| W10 | Smoke (local) → LR pick → G1 grid → analysis → gated G2/G3 | W1–W9 |

Parallelizable: {W2, W3, W4, W6} concurrently after W1. Integration + full round-trip test
(doc → C\* → parse → score) is the W5/W7 gate before any training run.

## 11. Risks / pre-committed mitigations

- **V4 instability** → per-variant LR/2 + softcap flag, documented deviation (§8.4).
- **Parser brittleness** → pre-specified 15% trigger, embedding-match fallback (§4).
- **OOD leakage** → lexical test [A7] + Test-OOD validity gate.
- **MFU uncertainty** → G1-contingency gate above.
- **Win-dev/Linux-run drift** → pathlib-only, no shell logic in packages, identical torch pin,
  full test suite runs locally pre-upload.
- **Budget overrun** → RunPod cost monitor script (W9) pings at 25/50/75% of credit;
  phases gated (§8).

## 12. Veto points (answer ≠ silence)

1. §2 reading of your Q2 amendment: decoys = lowest-priority class, cut wholesale once the
   queried prefix stops, never padding. If you meant "decoys are never in C\* at all," say so.
2. §6 softmax temperature varies with solver-chosen head_dim (entailed, not tuned) — accepted
   as part of the ratio variable, noted for the write-up.
3. §8 tiny budget = 500M tokens (not 2B) to fit the 100h envelope; ablation validity is
   unaffected (equal budgets across variants) but absolute quality will be modest.
