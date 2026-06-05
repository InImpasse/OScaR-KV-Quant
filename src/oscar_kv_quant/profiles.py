"""Default model paths and architecture hints (override via CLI)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json


@dataclass
class ModelProfile:
    name: str
    default_model_path: Path
    # Optional overrides if config.json missing
    num_layers: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None


@dataclass
class KVGeometry:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    num_attention_layers: int | None = None
    model_type: str | None = None

    @property
    def layers_for_kv_estimate(self) -> int:
        # KV cache only exists for attention layers. Most dense-only models have
        # no layer_types field, so fall back to num_layers.
        return self.num_attention_layers or self.num_layers


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_checkpoint_dir() -> Path:
    return repo_root() / "checkpoints"


PROFILES: dict[str, ModelProfile] = {
    "granite": ModelProfile(
        name="granite-4.0-1b-base",
        default_model_path=default_checkpoint_dir() / "granite-4.0-1b-base",
        num_layers=40,
        num_kv_heads=4,
        head_dim=128,
    ),
    "gemma4": ModelProfile(
        name="gemma-4-E2B",
        default_model_path=default_checkpoint_dir() / "gemma-4-E2B",
        num_layers=None,
        num_kv_heads=None,
        head_dim=None,
    ),
}


def load_model_config(model_path: Path) -> dict[str, Any]:
    cfg = model_path / "config.json"
    if not cfg.is_file():
        return {}
    with cfg.open() as f:
        return json.load(f)


def _text_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the text model config for multimodal configs, else the raw config."""
    nested = raw.get("text_config")
    if isinstance(nested, dict):
        return nested
    return raw


def resolve_kv_geometry(
    model_path: Path | str,
    profile: ModelProfile | None,
) -> KVGeometry:
    """Return parsed KV cache geometry for estimates and rotation defaults."""
    p = Path(model_path)
    raw: dict[str, Any]
    if p.is_dir() and (p / "config.json").is_file():
        raw = load_model_config(p)
    else:
        from transformers import AutoConfig

        raw = AutoConfig.from_pretrained(
            str(model_path), trust_remote_code=True
        ).to_dict()
    text_raw = _text_config(raw)
    n_layers = int(
        text_raw.get("num_hidden_layers")
        or (profile.num_layers if profile else None)
        or 32
    )
    n_kv = int(
        text_raw.get("num_key_value_heads")
        or text_raw.get("swa_num_key_value_heads")
        or (profile.num_kv_heads if profile else None)
        or 8
    )
    hidden = text_raw.get("hidden_size")
    if hidden is None:
        hs = text_raw.get("hidden_sizes")
        if isinstance(hs, list) and hs:
            hidden = hs[0]
    n_heads = text_raw.get("num_attention_heads") or text_raw.get("num_heads")
    hd = text_raw.get("head_dim")
    if hd is None and hidden and n_heads:
        hd = int(hidden) // int(n_heads)
    if hd is None and profile and profile.head_dim:
        hd = profile.head_dim
    if hd is None:
        hd = 128
    layer_types = text_raw.get("layer_types")
    n_attention_layers: int | None = None
    if isinstance(layer_types, list):
        n_attention_layers = sum(1 for t in layer_types if "attention" in str(t))
    return KVGeometry(
        num_layers=n_layers,
        num_kv_heads=n_kv,
        head_dim=int(hd),
        num_attention_layers=n_attention_layers,
        model_type=text_raw.get("model_type") or raw.get("model_type"),
    )
