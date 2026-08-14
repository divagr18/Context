"""Recall-shaping auxiliary signal (PLAN 7.2, mechanism Q6).

Every ``recall_every`` steps, on a small subsample: greedy-generate Ĉ, parse it,
compute fact recall ``r`` and hallucination rate ``h``, then add an extra
teacher-forced CE on the subsample weighted by ``λ_drop·(1−r) + λ_waste·h``.
Generation runs under no-grad (probe only); only the teacher-forced CE
contributes gradients, so the fragile ratio comparison is insulated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from track_a.data.pack import IGNORE_INDEX
from track_a.evalkit.grading import (
    decoy_triples, fact_triples_from_records, ground_truth_triples,
    parse_c_text, recall_and_hallucination,
)
from track_a.model.generate import greedy_generate


def prompt_target_split(sample):
    """Split a PackSample into (prompt_ids, target_ids) using the label mask."""
    ids = sample.input_ids
    labels = sample.labels
    n_target = sum(1 for l in labels if l != IGNORE_INDEX)
    prompt_len = len(ids) - n_target
    return ids[:prompt_len], ids[prompt_len:]


def recall_shaping_stats(model, subsample, tokenizer, c_close_id, device):
    """Mean (recall, hallucination_rate) over the subsample (no grad)."""
    rs: list[float] = []
    hs: list[float] = []
    for sample, fact_db in subsample:
        prompt_ids, target_ids = prompt_target_split(sample)
        max_new = len(target_ids) + 16
        gen = greedy_generate(model, prompt_ids, max_new, [c_close_id])
        text = tokenizer.decode(gen)
        records, _failures = parse_c_text(text)
        parsed = fact_triples_from_records(records)
        gt = ground_truth_triples(fact_db)
        dc = decoy_triples(fact_db)
        r, h = recall_and_hallucination(parsed, gt, dc)
        rs.append(r)
        hs.append(h)
    return sum(rs) / len(rs), sum(hs) / len(hs)


def teacher_forced_ce(model, subsample, pad_id, device, use_bf16):
    """Mean teacher-forced CE over the subsample target spans (grad path)."""
    from track_a.training.batching import collate

    samples = [s for s, _ in subsample]
    batch = collate(samples, pad_id, device)
    autocast = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if (use_bf16 and device.type == "cuda") else nullcontext())
    with autocast:
        logits, _aux = model(batch.input_ids)
    logits = logits.float()
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), batch.labels.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="sum",
    )
    n = (batch.labels != IGNORE_INDEX).sum().clamp(min=1).float()
    return loss / n


def recall_shaping_loss(model, subsample, tokenizer, cfg, pad_id, c_close_id,
                        device):
    """Return (extra_loss, r_bar, h_bar). Restores the model's training mode."""
    was_training = model.training
    r_bar, h_bar = recall_shaping_stats(model, subsample, tokenizer,
                                        c_close_id, device)
    model.train(was_training)
    ce = teacher_forced_ce(model, subsample, pad_id, device, cfg.use_bf16)
    weight = cfg.lambda_drop * (1.0 - r_bar) + cfg.lambda_waste * h_bar
    return weight * ce, r_bar, h_bar
