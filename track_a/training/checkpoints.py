"""Checkpoint save/load: model + optimizer + metadata (step/token count)."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "meta": dict(meta)}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device=None) -> dict:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("meta", {})
