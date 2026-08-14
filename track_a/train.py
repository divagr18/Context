"""Track A training entry point (contract C8).

Usage:
    python -m track_a.train --config configs/train_smoke.yaml
                            [--total-tokens N] [--micro-batch K]

Single-shot framing only in Track A. Token-budget accounting counts non-pad
tokens; gradient accumulation is token-based (step when >= effective batch).
Aux losses are wired for config but require annotation collation (tracked in
PLAN; raising until then). Recall-shaping runs every ``recall_every`` steps.
"""

from __future__ import annotations

import argparse
import random
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from track_a.data import TrainingDataset
from track_a.data.pack import IGNORE_INDEX
from track_a.model.aux_losses import AuxModules
from track_a.model.core import Transformer
from track_a.tokenize import get_tokenizer
from track_a.training.batching import collate
from track_a.training.checkpoints import save_checkpoint
from track_a.training.config_resolve import load_run_config
from track_a.training.logger import RunLogger
from track_a.training.optimizer import build_optimizer, lr_at
from track_a.training.recall_shaping import recall_shaping_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _autocast(use_bf16: bool, device: torch.device):
    if use_bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def ce_on_batch(model, batch, use_bf16: bool):
    device = batch.input_ids.device
    with _autocast(use_bf16, device):
        logits, _aux = model(batch.input_ids)
    # Per-row CE: a single fp32 (B*T, vocab) tensor costs ~4 GiB at bs=8 on
    # the 8GB 4060 and capped the micro-batch; rows keep peaks ~8x lower.
    parts = [F.cross_entropy(logits[i], batch.labels[i],
                             ignore_index=IGNORE_INDEX, reduction="sum")
             for i in range(logits.size(0))]
    n_tokens = (batch.labels != IGNORE_INDEX).sum().clamp(min=1).float()
    return torch.stack(parts).sum() / n_tokens


def _auto_tune_micro_batch(model, samples, pad_id, device, use_bf16):
    """Largest micro-batch that forward+backwards without OOM (CUDA only)."""
    if device.type != "cuda":
        return 2
    best = 1
    for size in (2, 4, 8, 16, 32):
        if size > len(samples):
            break
        try:
            batch = collate(samples[:size], pad_id, device)
            loss = ce_on_batch(model, batch, use_bf16)
            loss.backward()
            model.zero_grad(set_to_none=True)
            best = size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    return best


def evaluate(model, val_shards, tok, kind, device, use_bf16, max_samples,
             pad_id):
    model.eval()
    ds = TrainingDataset(val_shards, tok, kind=kind)
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            batch = collate([sample], pad_id, device)
            try:
                total += ce_on_batch(model, batch, use_bf16).item()
                n += 1
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[eval] skipped OOM sample {i}", flush=True)
    model.train()
    return total / max(1, n)


def train(cfg_path: str, total_tokens=None, micro_batch=None) -> Path:
    cfg = load_run_config(cfg_path)
    if total_tokens is not None:
        cfg = replace(cfg, total_tokens=int(total_tokens))
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer()
    pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    c_close_id = tok.convert_tokens_to_ids("</C>")

    model = Transformer(cfg.model).to(device)
    if cfg.aux.enabled():
        raise NotImplementedError(
            "aux-loss training requires annotation collation (PLAN follow-up)"
        )
    _aux_modules = AuxModules(cfg.model.d_model)  # reserved for aux wiring
    optimizer = build_optimizer(model.param_groups(), cfg)

    run_root = Path(cfg.run_dir) / f"{cfg.run_tag}-{cfg.seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.yaml").write_text(
        yaml.safe_dump({"resolved": str(cfg)}, sort_keys=False),
        encoding="utf-8")
    logger = RunLogger(str(run_root / "tb"))

    train_ds = TrainingDataset(cfg.train_shards, tok, kind=cfg.framing,
                               seed=cfg.seed, shuffle_buffer=64,
                               with_records=True)

    def refill(buf, it_ref, need):
        while len(buf) < need:
            try:
                buf.append(next(it_ref[0]))
            except StopIteration:
                it_ref[0] = iter(train_ds)
                buf.append(next(it_ref[0]))

    it_ref = [iter(train_ds)]
    buf: list = []
    probe_buf: list = []
    refill(probe_buf, it_ref, 32)
    mb = micro_batch or _auto_tune_micro_batch(
        model, [s for s, _ in probe_buf], pad_id, device, cfg.use_bf16)
    refill(buf, it_ref, mb)

    total_steps = max(1, cfg.total_tokens // cfg.effective_batch_tokens)
    tokens_seen = 0
    step = 0
    accum_tokens = 0
    accum_count = 0
    oom_streak = 0
    best_val = float("inf")
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    while tokens_seen < cfg.total_tokens:
        lr = lr_at(step, cfg, total_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        items = buf[:mb]
        buf[:] = buf[mb:]
        refill(buf, it_ref, mb)
        samples = []
        for s, _rec in items:
            if len(s.input_ids) > cfg.max_seq:
                continue
            samples.append(s)
        if not samples:
            continue
        batch = collate(samples, pad_id, device)
        try:
            ce = ce_on_batch(model, batch, cfg.use_bf16)
            ce.backward()
        except torch.cuda.OutOfMemoryError:
            # Longer-than-probe sequences can OOM the auto-tuned micro-batch.
            # Shrink and re-queue; at mb=1 skip the unfittable sample; abort
            # loudly if the device cannot fit anything at all.
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if mb > 1:
                mb //= 2
                buf[0:0] = items
                print(f"[oom] step={step} micro_batch shrunk to {mb}",
                      flush=True)
                continue
            oom_streak += 1
            if oom_streak > 50:
                raise RuntimeError(
                    "cannot fit any sample on device: 50 consecutive "
                    "single-sample OOMs; reduce max_seq or add VRAM")
            print(f"[oom] step={step} skipped unfittable sample "
                  f"(seqs={len(samples)})", flush=True)
            continue
        oom_streak = 0
        accum_tokens += batch.n_tokens
        accum_count += 1

        if accum_tokens < cfg.effective_batch_tokens:
            continue

        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(accum_count)
        extra_r = extra_h = 0.0
        if cfg.recall_every > 0 and step % cfg.recall_every == 0:
            subsample = []
            refill(probe_buf, it_ref, cfg.recall_subsample)
            for s, rec in probe_buf[:cfg.recall_subsample]:
                if len(s.input_ids) <= cfg.max_seq:
                    subsample.append((s, rec))
            if subsample:
                extra, extra_r, extra_h = recall_shaping_loss(
                    model, subsample, tok, cfg, pad_id, c_close_id, device)
                extra.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += accum_tokens
        accum_tokens = 0
        accum_count = 0
        step += 1

        if step % cfg.log_every == 0:
            tps = tokens_seen / max(1.0, time.time() - t0)
            logger.scalar("train/ce", ce.item(), step)
            logger.scalar("train/lr", lr, step)
            logger.scalar("train/tokens_seen", tokens_seen, step)
            logger.scalar("train/tokens_per_sec", tps, step)
            logger.scalar("train/recall_subsample", extra_r, step)
            logger.scalar("train/halluc_frac_subsample", extra_h, step)
            print(f"[train] step={step} ce={ce.item():.4f} lr={lr:.3e} "
                  f"tokens_seen={tokens_seen} tok_per_s={tps:.0f} "
                  f"recall={extra_r:.3f} halluc={extra_h:.3f}", flush=True)
        if cfg.eval_every_steps > 0 and step % cfg.eval_every_steps == 0 \
                and cfg.val_shards:
            val_ce = evaluate(model, cfg.val_shards, tok, cfg.framing, device,
                              cfg.use_bf16, cfg.val_max_samples, pad_id)
            logger.scalar("val/ce", val_ce, step)
            if val_ce < best_val:
                best_val = val_ce
                save_checkpoint(run_root / "best.pt", model, optimizer,
                                step=step, tokens_seen=tokens_seen,
                                val_ce=best_val)
                print(f"[val] step={step} val_ce={val_ce:.4f} new best",
                      flush=True)
            else:
                print(f"[val] step={step} val_ce={val_ce:.4f} "
                      f"best={best_val:.4f}", flush=True)
        if cfg.save_every_steps > 0 and step % cfg.save_every_steps == 0:
            save_checkpoint(run_root / "last.pt", model, optimizer,
                            step=step, tokens_seen=tokens_seen)

    save_checkpoint(run_root / "final.pt", model, optimizer, step=step,
                    tokens_seen=tokens_seen)
    print(f"[done] steps={step} tokens_seen={tokens_seen} "
          f"best_val={best_val:.4f} run={run_root}", flush=True)
    logger.close()
    return run_root


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Track A trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--total-tokens", type=int, default=None)
    parser.add_argument("--micro-batch", type=int, default=None)
    args = parser.parse_args(argv)
    run_root = train(args.config, total_tokens=args.total_tokens,
                     micro_batch=args.micro_batch)
    print(f"training complete -> {run_root}")


if __name__ == "__main__":
    main()
