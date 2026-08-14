"""Tokenizer owner (contract C5): repo-local GPT-2 + Track A special tokens.

Loads the committed ``track_a/assets/gpt2/tokenizer.json`` (native fast
format, encodes the added specials, zero network, no deprecated code paths).
If that file is ever missing, it self-heals offline: rebuilds the byte-level
BPE fast tokenizer from the also-committed ``vocab.json``/``merges.txt`` via
the ``tokenizers`` library and re-saves ``tokenizer.json``. No HF hub needed.

Token ids are stable: 0..50256 are GPT-2's; 50257..50277 are SPECIAL_TOKENS
in types.SPECIAL_TOKENS order. Import-side-effect free: built lazily.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from track_a.needle_gen.types import GPT2_BASE_VOCAB, SPECIAL_TOKENS, TOTAL_VOCAB

_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "gpt2"
_TOKENIZER_FILE = _ASSET_DIR / "tokenizer.json"
_ENDOFTEXT = "".join(["<", "|", "endoftext", "|", ">"])
_SPECIAL_KW = {"unk_token": _ENDOFTEXT, "bos_token": _ENDOFTEXT, "eos_token": _ENDOFTEXT}


def _bootstrap_offline() -> PreTrainedTokenizerFast:
    """Rebuild the fast tokenizer from committed vocab/merges and persist it."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    bpe = models.BPE.from_file(str(_ASSET_DIR / "vocab.json"), str(_ASSET_DIR / "merges.txt"))
    raw = Tokenizer(bpe)
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=raw, add_prefix_space=False, **_SPECIAL_KW)
    assert len(fast) == GPT2_BASE_VOCAB
    n_added = fast.add_tokens(list(SPECIAL_TOKENS), special_tokens=True)
    assert n_added == len(SPECIAL_TOKENS)
    backend = getattr(fast, "backend_tokenizer", None) or getattr(fast, "_tokenizer", None)
    assert backend is not None
    backend.save(str(_TOKENIZER_FILE))
    return fast


@lru_cache(maxsize=1)
def get_tokenizer() -> PreTrainedTokenizerFast:
    """Fast GPT-2 tokenizer (vocab 50278) from repo-local committed files."""
    if _TOKENIZER_FILE.exists():
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(_TOKENIZER_FILE), add_prefix_space=False, **_SPECIAL_KW
        )
    else:
        tokenizer = _bootstrap_offline()
    assert len(tokenizer) == TOTAL_VOCAB
    return tokenizer


def encode(text: str) -> list[int]:
    """Token ids for `text` (special tokens encode as single ids, no BOS/EOS)."""
    return get_tokenizer().encode(text, add_special_tokens=False)


def decode(ids: list[int]) -> str:
    """Text for `ids`; decode(encode(x)) == x for in-vocab text."""
    return get_tokenizer().decode(ids)
