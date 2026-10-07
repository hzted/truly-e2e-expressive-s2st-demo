"""DirectS2ST model.

A frozen w2v-BERT 2.0 speech encoder; an AR temporal decoder that produces
source text, target text and the first Mimi codebook (C0) with one shared
output head; and an NAR depth Transformer for C1-C15, conditioned on a source
Mimi codec prompt and a frozen CAMPPlus speaker embedding. w2v-BERT 2.0, Mimi
and CAMPPlus are external models (see docs/model_dependencies.md).
"""
from __future__ import annotations

import math
import os
import sys
import json
import time
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    MimiModel,
    Wav2Vec2BertModel,
    WhisperFeatureExtractor,
    WhisperModel,
)

class DepthTransformer(nn.Module):
    """
    Depth decoder: processes codebook levels 1 to N-1.

    For each time position, this decoder autoregressively predicts codebook tokens
    across the depth (codebook level) dimension. Each depth step takes as input:
    - The hidden state from the temporal decoder
    - The embedding of the previous codebook level's token

    At depth step 0 (predicting level 1): input = temporal_hidden + embed(level_0_token)
    At depth step k (predicting level k+1): input = temporal_hidden + embed(level_k_token)
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,  # codebook_size + 2 (includes BOS/EOS)
        num_depth_levels: int,  # N-1 levels (levels 1 to N-1)
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        auto_adjust_heads: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_depth_levels = num_depth_levels

        # Adjust num_heads if needed
        if hidden_size % num_heads != 0:
            if not auto_adjust_heads:
                raise ValueError(
                    f"depth_hidden_size={hidden_size} must be divisible by depth_num_heads={num_heads}."
                )
            valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
            if not valid_heads:
                raise ValueError(f"Could not find valid num_heads dividing hidden_size={hidden_size}.")
            num_heads = valid_heads[0]
            warnings.warn(
                f"Adjusted depth_num_heads to {num_heads} to match hidden_size={hidden_size}.",
                stacklevel=2,
            )

        if ffn_dim is None:
            ffn_dim = 4 * hidden_size

        # Decoder layers for depth processing. The decoder uses causal self-attention
        # over time and causal cross-attention into temporal memory.
        depth_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(depth_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

        # Output projection heads for each depth level
        self.level_heads = nn.ModuleList(
            [nn.Linear(hidden_size, vocab_size) for _ in range(num_depth_levels)]
        )

    def _causal_mask(
        self, target_len: int, source_len: Optional[int], device: torch.device
    ) -> torch.Tensor:
        """Mask entries where target position i would attend to source position j > i."""
        if source_len is None:
            source_len = target_len
        return torch.triu(
            torch.ones((target_len, source_len), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self,
        depth_inputs: torch.Tensor,  # [B, T, D]
        temporal_memory: torch.Tensor,  # [B, T, D]
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the depth decoder.

        Args:
            depth_inputs: previous-level conditioning sequence, [B, T, D]
            temporal_memory: temporal decoder states, [B, T, D]
            tgt_key_padding_mask: optional padding mask for depth_inputs, True means masked
            memory_key_padding_mask: optional padding mask for temporal_memory, True means masked

        Returns:
            hidden: processed hidden states [B, T, D]
        """
        target_len = depth_inputs.size(1)
        memory_len = temporal_memory.size(1)
        tgt_mask = self._causal_mask(target_len, None, depth_inputs.device)
        memory_mask = self._causal_mask(target_len, memory_len, depth_inputs.device)
        hidden = self.transformer(
            tgt=depth_inputs,
            memory=temporal_memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        hidden = self.norm(hidden)
        return hidden


class CampPlusFiLMTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Transformer encoder layer with a runtime, source-speaker FiLM input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Runtime tensors are deliberately not parameters or buffers, keeping
        # state-dict keys identical to nn.TransformerEncoderLayer.
        self._runtime_campplus_scale: Optional[torch.Tensor] = None
        self._runtime_campplus_shift: Optional[torch.Tensor] = None

    def set_runtime_campplus_film(
        self,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> None:
        self._runtime_campplus_scale = scale
        self._runtime_campplus_shift = shift

    def clear_runtime_campplus_film(self) -> None:
        self._runtime_campplus_scale = None
        self._runtime_campplus_shift = None

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        scale = self._runtime_campplus_scale
        shift = self._runtime_campplus_shift
        if (scale is None) != (shift is None):
            raise RuntimeError("Incomplete runtime CAMPPlus FiLM condition.")
        if scale is not None:
            if src.ndim != 3 or scale.ndim != 2 or shift.ndim != 2:
                raise RuntimeError(
                    "CAMPPlus FiLM expects src=[B,T,D] and scale/shift=[B,D]."
                )
            if scale.shape != shift.shape or scale.shape != (src.size(0), src.size(2)):
                raise RuntimeError(
                    "CAMPPlus FiLM shape mismatch: "
                    f"src={tuple(src.shape)}, scale={tuple(scale.shape)}, "
                    f"shift={tuple(shift.shape)}."
                )
            src = src * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return super().forward(
            src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )


class TemporalDepthDecoder(nn.Module):
    """
    Combined decoder with temporal-decoder output and depth decoder.

    Architecture:
    1. Temporal decoder outputs shifted next-frame hidden states [B, T, D]
    2. First codebook head predicts level 0 from temporal hidden
    3. Depth decoder predicts levels 1 to N-1, where each step takes:
       - temporal_hidden + embed(previous_level_token)
    """

    def __init__(
        self,
        temporal_hidden_size: int,
        depth_hidden_size: int,
        vocab_size: int,  # codebook_size + 2 (includes BOS/EOS)
        num_quantizers: int,
        depth_num_layers: int = 4,
        depth_num_heads: int = 8,
        depth_ffn_dim: Optional[int] = None,
        depth_dropout: float = 0.1,
        depth_auto_adjust_heads: bool = True,
        num_temporal_codebook_levels: int = 1,
        source_hidden_size: Optional[int] = None,
        enable_source_conditioning: bool = False,
        source_conditioning_num_heads: int = 8,
        source_conditioning_dropout: float = 0.1,
        source_conditioning_residual_scale: float = 0.2,
        source_conditioning_detach: bool = True,
        enable_source_acoustic_conditioning: bool = False,
        source_acoustic_hidden_size: Optional[int] = None,
        source_acoustic_residual_scale: float = 0.3,
        source_acoustic_depth_start_level: int = 1,
        source_acoustic_detach: bool = True,
        enable_source_style_token_conditioning: bool = False,
        source_style_token_count: int = 8,
        source_style_token_num_heads: int = 8,
        source_style_token_dropout: float = 0.1,
        source_style_token_residual_scale: float = 0.7,
        source_style_token_depth_start_level: int = 1,
        source_style_token_gate_init: float = 0.10,
        source_style_token_c1_gate_init: float = 0.03,
        enable_nar_source_prompt: bool = False,
        nar_attention_mode: str = "full",
        nar_max_positions: int = 4096,
        nar_teacher_force_level_batch_size: int = 1,
        enable_campplus_depth_conditioning: bool = False,
        campplus_depth_conditioning_mode: str = "per_layer_film",
        campplus_embedding_size: int = 192,
        campplus_film_scale: float = 1.0,
        campplus_depth_start_level: int = 1,
        campplus_depth_identity_loss_weight: float = 0.0,
        campplus_identity_level_aggregation: str = "final",
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.vocab_size = vocab_size
        self.temporal_hidden_size = temporal_hidden_size
        self.depth_hidden_size = depth_hidden_size
        self.num_temporal_codebook_levels = max(
            1,
            min(int(num_temporal_codebook_levels), int(num_quantizers)),
        )
        self.enable_source_conditioning = bool(enable_source_conditioning)
        self.source_conditioning_residual_scale = float(source_conditioning_residual_scale)
        self.source_conditioning_detach = bool(source_conditioning_detach)
        self.enable_source_acoustic_conditioning = bool(enable_source_acoustic_conditioning)
        self.source_acoustic_residual_scale = float(source_acoustic_residual_scale)
        self.source_acoustic_depth_start_level = max(1, int(source_acoustic_depth_start_level))
        self.source_acoustic_detach = bool(source_acoustic_detach)
        self.enable_source_style_token_conditioning = bool(
            enable_source_style_token_conditioning
        )
        self.source_style_token_count = max(1, int(source_style_token_count))
        self.source_style_token_residual_scale = float(source_style_token_residual_scale)
        self.source_style_token_depth_start_level = max(
            1, int(source_style_token_depth_start_level)
        )
        self.source_style_token_gate_init = float(source_style_token_gate_init)
        self.source_style_token_c1_gate_init = float(source_style_token_c1_gate_init)
        self.enable_nar_source_prompt = bool(enable_nar_source_prompt)
        self.nar_attention_mode = str(nar_attention_mode or "full").strip().lower()
        if self.nar_attention_mode not in {"full", "prefix_causal"}:
            raise ValueError(
                "nar_attention_mode must be one of: full, prefix_causal."
            )
        self.nar_max_positions = max(2, int(nar_max_positions))
        self.nar_teacher_force_level_batch_size = max(
            1, int(nar_teacher_force_level_batch_size)
        )
        self.enable_campplus_depth_conditioning = bool(
            enable_campplus_depth_conditioning
        )
        self.campplus_depth_conditioning_mode = str(
            campplus_depth_conditioning_mode or "per_layer_film"
        ).strip().lower()
        if self.campplus_depth_conditioning_mode not in {
            "per_layer_film",
            "input_add",
            "prefix_token",
        }:
            raise ValueError(
                "campplus_depth_conditioning_mode must be one of: "
                "per_layer_film, input_add, prefix_token."
            )
        self.campplus_embedding_size = max(1, int(campplus_embedding_size))
        self.campplus_film_scale = float(campplus_film_scale)
        self.campplus_depth_start_level = max(1, int(campplus_depth_start_level))
        if (
            self.enable_campplus_depth_conditioning
            and self.campplus_depth_conditioning_mode == "prefix_token"
            and self.campplus_depth_start_level != 1
        ):
            raise ValueError(
                "prefix_token CAMPPlus conditioning is present for every NAR "
                "depth level; campplus_depth_start_level must be 1."
            )
        self.campplus_depth_identity_loss_weight = float(
            campplus_depth_identity_loss_weight
        )
        self.campplus_identity_level_aggregation = str(
            campplus_identity_level_aggregation or "final"
        ).strip().lower()
        if self.campplus_identity_level_aggregation not in {
            "final",
            "conditioned_mean",
        }:
            raise ValueError(
                "campplus_identity_level_aggregation must be one of: "
                "final, conditioned_mean."
            )
        if self.enable_campplus_depth_conditioning and not self.enable_nar_source_prompt:
            raise ValueError(
                "CAMPPlus depth conditioning currently requires the full-sequence "
                "Mimi NAR depth decoder."
            )
        self._runtime_source_codec_tokens = None
        self._runtime_source_codec_mask = None
        self._runtime_zero_source_codec_prompt = False
        self._runtime_campplus_source_embedding = None

        self.source_acoustic_proj = None
        self.source_acoustic_norm = None
        if self.enable_source_acoustic_conditioning:
            if source_acoustic_hidden_size is None:
                raise ValueError(
                    "source_acoustic_hidden_size is required when "
                    "enable_source_acoustic_conditioning=True."
                )
            source_acoustic_hidden_size = int(source_acoustic_hidden_size)
            self.source_acoustic_proj = (
                nn.Identity()
                if source_acoustic_hidden_size == depth_hidden_size
                else nn.Linear(source_acoustic_hidden_size, depth_hidden_size)
            )
            self.source_acoustic_norm = nn.LayerNorm(depth_hidden_size)

        self.source_style_token_input_norm = None
        self.source_style_token_proj = None
        self.source_style_token_position = None
        self.source_style_token_query_norm = None
        self.source_style_token_cross_attn = None
        self.source_style_token_gates = None
        if self.enable_source_style_token_conditioning:
            if source_acoustic_hidden_size is None:
                raise ValueError(
                    "source_acoustic_hidden_size is required when "
                    "enable_source_style_token_conditioning=True."
                )
            source_acoustic_hidden_size = int(source_acoustic_hidden_size)
            style_heads = max(1, int(source_style_token_num_heads))
            if depth_hidden_size % style_heads != 0:
                if not depth_auto_adjust_heads:
                    raise ValueError(
                        "depth_hidden_size must be divisible by "
                        "mimi_source_style_token_num_heads."
                    )
                valid_heads = [
                    h for h in range(style_heads, 0, -1)
                    if depth_hidden_size % h == 0
                ]
                if not valid_heads:
                    raise ValueError(
                        f"Could not find valid style-token heads for depth_hidden_size={depth_hidden_size}."
                    )
                style_heads = valid_heads[0]
                warnings.warn(
                    "Adjusted mimi_source_style_token_num_heads to "
                    f"{style_heads} to match depth_hidden_size={depth_hidden_size}.",
                    stacklevel=2,
                )
            self.source_style_token_input_norm = nn.LayerNorm(source_acoustic_hidden_size)
            self.source_style_token_proj = (
                nn.Identity()
                if source_acoustic_hidden_size == depth_hidden_size
                else nn.Linear(source_acoustic_hidden_size, depth_hidden_size)
            )
            self.source_style_token_position = nn.Parameter(
                torch.empty(1, self.source_style_token_count, depth_hidden_size)
            )
            nn.init.normal_(self.source_style_token_position, mean=0.0, std=0.02)
            self.source_style_token_query_norm = nn.LayerNorm(depth_hidden_size)
            self.source_style_token_cross_attn = nn.MultiheadAttention(
                embed_dim=depth_hidden_size,
                num_heads=style_heads,
                dropout=float(source_style_token_dropout),
                batch_first=True,
            )
            self.source_style_token_gates = nn.Parameter(
                torch.zeros(num_quantizers, dtype=torch.float32)
            )
            with torch.no_grad():
                for level in range(
                    self.source_style_token_depth_start_level,
                    num_quantizers,
                ):
                    init_value = (
                        self.source_style_token_c1_gate_init
                        if level == 1
                        else self.source_style_token_gate_init
                    )
                    self.source_style_token_gates[level] = float(init_value)

        self.source_memory_proj = None
        self.source_cross_attn = None
        self.source_conditioning_norm = None
        if self.enable_source_conditioning:
            source_hidden_size = (
                int(temporal_hidden_size)
                if source_hidden_size is None
                else int(source_hidden_size)
            )
            source_heads = max(1, int(source_conditioning_num_heads))
            if depth_hidden_size % source_heads != 0:
                if not depth_auto_adjust_heads:
                    raise ValueError(
                        "depth_hidden_size must be divisible by "
                        "depth_source_conditioning_num_heads."
                    )
                valid_heads = [
                    h for h in range(source_heads, 0, -1)
                    if depth_hidden_size % h == 0
                ]
                if not valid_heads:
                    raise ValueError(
                        f"Could not find valid source-conditioning heads for depth_hidden_size={depth_hidden_size}."
                    )
                source_heads = valid_heads[0]
                warnings.warn(
                    "Adjusted depth_source_conditioning_num_heads to "
                    f"{source_heads} to match depth_hidden_size={depth_hidden_size}.",
                    stacklevel=2,
                )
            self.source_memory_proj = (
                nn.Identity()
                if source_hidden_size == depth_hidden_size
                else nn.Linear(source_hidden_size, depth_hidden_size)
            )
            self.source_cross_attn = nn.MultiheadAttention(
                embed_dim=depth_hidden_size,
                num_heads=source_heads,
                dropout=float(source_conditioning_dropout),
                batch_first=True,
            )
            self.source_conditioning_norm = nn.LayerNorm(depth_hidden_size)

        # Temporal heads predict the first K codebooks directly from temporal hidden.
        self.first_codebook_head = nn.Linear(temporal_hidden_size, vocab_size)
        self.temporal_codebook_heads = nn.ModuleList(
            [
                nn.Linear(temporal_hidden_size, vocab_size)
                for _ in range(self.num_temporal_codebook_levels - 1)
            ]
        )

        # Embeddings for all codebook levels (includes BOS/EOS tokens)
        self.codebook_emb = nn.ModuleList(
            [nn.Embedding(vocab_size, depth_hidden_size) for _ in range(num_quantizers)]
        )

        # Projection from temporal hidden to depth hidden (if dimensions differ)
        self.temporal_to_depth = (
            nn.Identity()
            if temporal_hidden_size == depth_hidden_size
            else nn.Linear(temporal_hidden_size, depth_hidden_size)
        )

        # Depth decoder predicts levels K to N-1, conditioned on temporal levels 0..K-1.
        self.depth_transformer = DepthTransformer(
            hidden_size=depth_hidden_size,
            vocab_size=vocab_size,
            num_depth_levels=max(0, num_quantizers - self.num_temporal_codebook_levels),
            num_layers=depth_num_layers,
            num_heads=depth_num_heads,
            ffn_dim=depth_ffn_dim,
            dropout=depth_dropout,
            auto_adjust_heads=depth_auto_adjust_heads,
        )
        self.nar_depth_transformer = None
        self.nar_depth_sep = None
        self.nar_depth_segment_emb = None
        self.nar_depth_level_emb = None
        self.nar_depth_pos_emb = None
        self.nar_depth_norm = None
        self.campplus_input_norm = None
        self.campplus_input_proj = None
        self.campplus_prefix_proj = None
        self.campplus_layer_film = None
        self.campplus_identity_head = None
        if self.enable_nar_source_prompt:
            nar_layer_cls = (
                CampPlusFiLMTransformerEncoderLayer
                if self.enable_campplus_depth_conditioning
                and self.campplus_depth_conditioning_mode == "per_layer_film"
                else nn.TransformerEncoderLayer
            )
            nar_layer = nar_layer_cls(
                d_model=depth_hidden_size,
                nhead=self.depth_transformer.transformer.layers[0].self_attn.num_heads,
                dim_feedforward=(depth_ffn_dim or 4 * depth_hidden_size),
                dropout=depth_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.nar_depth_transformer = nn.TransformerEncoder(
                nar_layer,
                num_layers=depth_num_layers,
            )
            # Retained in the state dict so checkpoints created by the initial
            # implementation remain loadable. It is intentionally not used by
            # forward; TransVIP NAR has no prompt/target separator token.
            self.nar_depth_sep = nn.Parameter(torch.empty(1, 1, depth_hidden_size))
            nn.init.normal_(self.nar_depth_sep, mean=0.0, std=0.02)
            self.nar_depth_segment_emb = nn.Embedding(3, depth_hidden_size)
            self.nar_depth_level_emb = nn.Embedding(num_quantizers, depth_hidden_size)
            self.nar_depth_pos_emb = nn.Embedding(self.nar_max_positions, depth_hidden_size)
            self.nar_depth_norm = nn.LayerNorm(depth_hidden_size)
        if self.enable_campplus_depth_conditioning:
            self.campplus_input_norm = nn.LayerNorm(
                self.campplus_embedding_size,
                elementwise_affine=False,
            )
            if self.campplus_depth_conditioning_mode == "per_layer_film":
                self.campplus_layer_film = nn.ModuleList(
                    [
                        nn.Linear(self.campplus_embedding_size, 2 * depth_hidden_size)
                        for _ in range(depth_num_layers)
                    ]
                )
                # Zero initialization keeps the unconditioned NAR unchanged at
                # initialization while every layer learns its own affine.
                for film in self.campplus_layer_film:
                    nn.init.zeros_(film.weight)
                    nn.init.zeros_(film.bias)
            elif self.campplus_depth_conditioning_mode == "input_add":
                # CAMP-lite: project the source vector once and add it to target
                # depth inputs before the ordinary NAR Transformer.
                self.campplus_input_proj = nn.Linear(
                    self.campplus_embedding_size,
                    depth_hidden_size,
                )
                nn.init.zeros_(self.campplus_input_proj.weight)
                nn.init.zeros_(self.campplus_input_proj.bias)
            else:
                # Prefix conditioning must carry source-speaker information from
                # the first update, unlike the identity-preserving input-add
                # ablation whose projection intentionally starts at zero.
                self.campplus_prefix_proj = nn.Linear(
                    self.campplus_embedding_size,
                    depth_hidden_size,
                )
            if self.campplus_depth_identity_loss_weight > 0.0:
                self.campplus_identity_head = nn.Sequential(
                    nn.LayerNorm(depth_hidden_size),
                    nn.Linear(depth_hidden_size, self.campplus_embedding_size),
                )

    def set_runtime_source_codec_prompt(
        self,
        source_codec_tokens: Optional[torch.Tensor],
        source_codec_mask: Optional[torch.Tensor],
        zero_prompt: bool = False,
    ) -> None:
        """Set the per-batch frozen Mimi prompt used by full-sequence depth decoding."""
        self._runtime_source_codec_tokens = source_codec_tokens
        self._runtime_source_codec_mask = source_codec_mask
        self._runtime_zero_source_codec_prompt = bool(zero_prompt)

    def set_runtime_campplus_source_embedding(
        self,
        source_embedding: Optional[torch.Tensor],
    ) -> None:
        """Set the source-only CAMPPlus vector used by depth generation."""
        self._runtime_campplus_source_embedding = source_embedding

    def _run_nar_depth_transformer(
        self,
        nar_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        campplus_source_embedding: Optional[torch.Tensor],
        campplus_layer_affines: Optional[
            List[Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        apply_campplus_conditioning: bool = True,
    ) -> torch.Tensor:
        if (
            not self.enable_campplus_depth_conditioning
            or self.campplus_depth_conditioning_mode != "per_layer_film"
            or not apply_campplus_conditioning
        ):
            return self.nar_depth_transformer(
                nar_input,
                mask=attention_mask,
                src_key_padding_mask=padding_mask,
            )
        if campplus_layer_affines is None:
            campplus_layer_affines = self._prepare_campplus_layer_affines(
                nar_input,
                campplus_source_embedding,
            )
        if len(campplus_layer_affines) != len(self.nar_depth_transformer.layers):
            raise RuntimeError(
                "CAMPPlus FiLM/layer count mismatch: "
                f"{len(campplus_layer_affines)} != "
                f"{len(self.nar_depth_transformer.layers)}."
            )
        runtime_layers: List[CampPlusFiLMTransformerEncoderLayer] = []
        try:
            for layer, (scale, shift) in zip(
                self.nar_depth_transformer.layers,
                campplus_layer_affines,
            ):
                film_layer = self._unwrap_campplus_film_layer(layer)
                film_layer.set_runtime_campplus_film(scale, shift)
                runtime_layers.append(film_layer)
            # Enter through the TransformerEncoder module instead of invoking
            # child layers directly. Besides canonicalizing masks once, this
            # preserves the enclosing FSDP/native execution boundary.
            return self.nar_depth_transformer(
                nar_input,
                mask=attention_mask,
                src_key_padding_mask=padding_mask,
            )
        finally:
            for film_layer in runtime_layers:
                film_layer.clear_runtime_campplus_film()

    @staticmethod
    def _unwrap_campplus_film_layer(
        layer: nn.Module,
    ) -> CampPlusFiLMTransformerEncoderLayer:
        """Resolve the underlying FiLM layer through optional FSDP wrappers."""
        current = layer
        visited = set()
        while id(current) not in visited:
            visited.add(id(current))
            if isinstance(current, CampPlusFiLMTransformerEncoderLayer):
                return current
            next_module = None
            for attr_name in ("module", "_fsdp_wrapped_module"):
                candidate = getattr(current, attr_name, None)
                if isinstance(candidate, nn.Module) and candidate is not current:
                    next_module = candidate
                    break
            if next_module is None:
                break
            current = next_module
        raise RuntimeError(
            "CAMPPlus NAR layer is not FiLM-capable; restart with the current "
            "model definition before enabling CAMPPlus depth conditioning."
        )

    def _prepare_campplus_layer_affines(
        self,
        reference: torch.Tensor,
        campplus_source_embedding: Optional[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Project one source-speaker vector once for all NAR codebook passes."""
        if campplus_source_embedding is None:
            raise RuntimeError(
                "CAMPPlus depth conditioning is enabled but no source embedding "
                "was prepared from the source waveform."
            )
        if self.campplus_input_norm is None or self.campplus_layer_film is None:
            raise RuntimeError("CAMPPlus depth conditioning modules are not initialized.")
        condition = campplus_source_embedding.to(
            device=reference.device,
            dtype=reference.dtype,
        ).detach()
        if condition.ndim != 2 or condition.size(-1) != self.campplus_embedding_size:
            raise RuntimeError(
                "CAMPPlus source embedding must be [B, D] with D="
                f"{self.campplus_embedding_size}, got {tuple(condition.shape)}."
            )
        if condition.size(0) == 1 and reference.size(0) > 1:
            condition = condition.expand(reference.size(0), -1)
        if condition.size(0) != reference.size(0):
            raise RuntimeError(
                f"CAMPPlus source batch {condition.size(0)} != reference batch "
                f"{reference.size(0)}."
            )
        condition = self.campplus_input_norm(condition.float()).to(reference.dtype)
        layer_affines: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for film in self.campplus_layer_film:
            scale, shift = film(condition).chunk(2, dim=-1)
            scale = torch.tanh(scale) * self.campplus_film_scale
            shift = shift * self.campplus_film_scale
            layer_affines.append((scale, shift))
        return layer_affines

    def _prepare_campplus_input_condition(
        self,
        reference: torch.Tensor,
        campplus_source_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Project the source speaker vector once for NAR input-only conditioning."""
        if campplus_source_embedding is None:
            raise RuntimeError(
                "CAMPPlus depth conditioning is enabled but no source embedding "
                "was prepared from the source waveform."
            )
        if self.campplus_input_norm is None or self.campplus_input_proj is None:
            raise RuntimeError("CAMPPlus input conditioning modules are not initialized.")
        condition = campplus_source_embedding.to(
            device=reference.device,
            dtype=reference.dtype,
        ).detach()
        if condition.ndim != 2 or condition.size(-1) != self.campplus_embedding_size:
            raise RuntimeError(
                "CAMPPlus source embedding must be [B, D] with D="
                f"{self.campplus_embedding_size}, got {tuple(condition.shape)}."
            )
        if condition.size(0) == 1 and reference.size(0) > 1:
            condition = condition.expand(reference.size(0), -1)
        if condition.size(0) != reference.size(0):
            raise RuntimeError(
                f"CAMPPlus source batch {condition.size(0)} != reference batch "
                f"{reference.size(0)}."
            )
        normalized = self.campplus_input_norm(condition.float()).to(reference.dtype)
        return self.campplus_input_proj(normalized) * self.campplus_film_scale

    def _prepare_campplus_prefix_token(
        self,
        reference: torch.Tensor,
        campplus_source_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Project a detached source-speaker embedding into one NAR prefix token."""
        if campplus_source_embedding is None:
            raise RuntimeError(
                "CAMPPlus prefix-token conditioning is enabled but no source "
                "embedding was prepared from the source waveform."
            )
        if self.campplus_input_norm is None or self.campplus_prefix_proj is None:
            raise RuntimeError(
                "CAMPPlus prefix-token conditioning modules are not initialized."
            )
        condition = campplus_source_embedding.to(
            device=reference.device,
            dtype=reference.dtype,
        ).detach()
        if condition.ndim != 2 or condition.size(-1) != self.campplus_embedding_size:
            raise RuntimeError(
                "CAMPPlus source embedding must be [B, D] with D="
                f"{self.campplus_embedding_size}, got {tuple(condition.shape)}."
            )
        if condition.size(0) == 1 and reference.size(0) > 1:
            condition = condition.expand(reference.size(0), -1)
        if condition.size(0) != reference.size(0):
            raise RuntimeError(
                f"CAMPPlus source batch {condition.size(0)} != reference batch "
                f"{reference.size(0)}."
            )
        normalized = self.campplus_input_norm(condition.float()).to(reference.dtype)
        speaker_token = self.campplus_prefix_proj(normalized).unsqueeze(1)
        expected_shape = (reference.size(0), 1, self.depth_hidden_size)
        if tuple(speaker_token.shape) != expected_shape:
            raise RuntimeError(
                "CAMPPlus speaker prefix must be [B, 1, D_depth]: "
                f"expected {expected_shape}, got {tuple(speaker_token.shape)}."
            )
        return speaker_token

    def campplus_identity_loss(
        self,
        depth_hidden: torch.Tensor,
        valid_mask: torch.Tensor,
        identity_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Align pooled target depth states with a frozen speaker vector."""
        if self.campplus_identity_head is None:
            raise RuntimeError("CAMPPlus identity head is not initialized.")
        valid_mask = valid_mask.to(device=depth_hidden.device, dtype=torch.bool)
        weights = valid_mask.unsqueeze(-1).to(depth_hidden.dtype)
        pooled = (depth_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        predicted = self.campplus_identity_head(pooled).float()
        target = identity_embedding.to(
            device=predicted.device,
            dtype=torch.float32,
        ).detach()
        expected_target_shape = (predicted.size(0), self.campplus_embedding_size)
        if tuple(target.shape) != expected_target_shape:
            raise RuntimeError(
                "CAMPPlus identity target must be [B, D_speaker]: "
                f"expected {expected_target_shape}, got {tuple(target.shape)}."
            )
        return (1.0 - F.cosine_similarity(predicted, target, dim=-1)).mean()

    def forward(
        self,
        temporal_hidden: torch.Tensor,  # [B, T, D_temporal]
        source_memory: Optional[torch.Tensor] = None,  # [B, S, D_source]
        source_memory_mask: Optional[torch.Tensor] = None,  # [B, S], 1=valid
        source_acoustic_embedding: Optional[torch.Tensor] = None,  # [B, D_acoustic]
        campplus_source_embedding: Optional[torch.Tensor] = None,  # [B, D_speaker]
        campplus_identity_embedding: Optional[torch.Tensor] = None,  # [B, D_speaker]
        labels: Optional[torch.Tensor] = None,  # [B, T, Q]
        teacher_force: bool = True,
        ignore_index: int = -100,
        detach_temporal_for_depth: bool = False,
        forced_first_ids: Optional[torch.Tensor] = None,  # [B, T], optional C0 path for decoding
        forced_depth_ids: Optional[torch.Tensor] = None,  # [B, T, Q], offline oracle prefix
        forced_depth_prefix_levels: int = 1,
        scheduled_sampling_prob: float = 0.0,
        scheduled_sampling_mode: str = "argmax",
        scheduled_sampling_topk: int = 8,
        scheduled_sampling_temperature: float = 1.0,
        chain_scheduled_sampling_prob: float = 0.0,
        chain_scheduled_sampling_mode: str = "argmax",
        chain_scheduled_sampling_topk: int = 8,
        chain_scheduled_sampling_temperature: float = 1.0,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass:
        1. Predict first codebook (level 0) from temporal hidden
        2. For levels 1 to N-1: use the depth decoder with previous level's embedding

        Args:
            temporal_hidden: [B, T, D] shifted next-frame hidden states from the temporal decoder
            labels: [B, T, Q] shifted target codebook indices (optional, for teacher forcing)
            teacher_force: whether to use ground truth labels for embeddings
            ignore_index: index to ignore in labels
            detach_temporal_for_depth: stop depth losses from updating the temporal planner
            forced_first_ids: optional level-0 tokens to condition the depth
                decoder on while still scoring/predicting level 0 normally.
            forced_depth_ids: optional target codes used only to force
                an offline oracle prefix such as gold C0:C1.
            forced_depth_prefix_levels: number of leading codebooks to take
                from forced_depth_ids. Level 0 is supplied by forced_first_ids.

        Returns:
            logits_per_level: list of [B, T, V] logits for each level
            final_hidden: the depth hidden state
        """
        bsz, seq_len, _ = temporal_hidden.shape
        if campplus_source_embedding is None:
            campplus_source_embedding = self._runtime_campplus_source_embedding
        logits_per_level: List[torch.Tensor] = []
        per_level_identity_losses: List[torch.Tensor] = []

        def _ids_from_logits(
            logits: torch.Tensor,
            level: int,
            *,
            use_temporal_scheduled_sampling: bool,
            use_chain_scheduled_sampling: bool,
        ) -> torch.Tensor:
            if labels is not None and teacher_force:
                ids = labels[:, :, level]
                prob = 0.0
                mode = "argmax"
                topk = 1
                temperature = 1.0
                if use_temporal_scheduled_sampling:
                    prob = float(scheduled_sampling_prob)
                    mode = scheduled_sampling_mode
                    topk = int(scheduled_sampling_topk)
                    temperature = float(scheduled_sampling_temperature)
                elif use_chain_scheduled_sampling:
                    prob = float(chain_scheduled_sampling_prob)
                    mode = chain_scheduled_sampling_mode
                    topk = int(chain_scheduled_sampling_topk)
                    temperature = float(chain_scheduled_sampling_temperature)
                if self.training and prob > 0.0:
                    with torch.no_grad():
                        if mode == "sample":
                            scaled = logits.detach() / max(1.0e-6, temperature)
                            k = min(max(1, topk), scaled.size(-1))
                            top_vals, top_ids = torch.topk(scaled, k=k, dim=-1)
                            probs = torch.softmax(top_vals, dim=-1)
                            sampled_pos = torch.multinomial(
                                probs.reshape(-1, k),
                                num_samples=1,
                            ).reshape(ids.shape)
                            pred_ids = top_ids.gather(-1, sampled_pos.unsqueeze(-1)).squeeze(-1)
                        else:
                            pred_ids = logits.detach().argmax(dim=-1)
                    valid = ids != ignore_index
                    replace = (torch.rand(ids.shape, device=ids.device) < prob) & valid
                    ids = torch.where(replace, pred_ids, ids)
                return ids
            return logits.argmax(dim=-1)

        first_logits = self.first_codebook_head(temporal_hidden)
        logits_per_level.append(first_logits)

        depth_source_hidden = temporal_hidden.detach() if detach_temporal_for_depth else temporal_hidden
        depth_hidden_base = self.temporal_to_depth(depth_source_hidden)
        depth_hidden_acoustic = depth_hidden_base
        if self.enable_source_acoustic_conditioning:
            if source_acoustic_embedding is None:
                raise RuntimeError(
                    "Mimi source acoustic conditioning is enabled but "
                    "source_acoustic_embedding was not provided."
                )
            if self.source_acoustic_proj is None or self.source_acoustic_norm is None:
                raise RuntimeError(
                    "Mimi source acoustic conditioning is enabled but not initialized."
                )
            acoustic = (
                source_acoustic_embedding.detach()
                if self.source_acoustic_detach
                else source_acoustic_embedding
            )
            # The style-token path packs the global mean/std vector in
            # slot zero followed by ordered local source tokens.
            if acoustic.ndim == 3:
                if acoustic.size(1) < 1:
                    raise RuntimeError(
                        "Packed Mimi source acoustic conditioning must include "
                        "a global token in slot zero."
                    )
                acoustic = acoustic[:, 0, :]
            if acoustic.ndim != 2:
                raise RuntimeError(
                    "Mimi source acoustic conditioning expects [B, D] or "
                    "packed [B, 1+K, D] inputs."
                )
            if abs(self.source_acoustic_residual_scale) > 1.0e-8:
                acoustic = acoustic.to(
                    device=depth_hidden_base.device,
                    dtype=depth_hidden_base.dtype,
                )
                acoustic = self.source_acoustic_proj(acoustic).unsqueeze(1)
                depth_hidden_acoustic = self.source_acoustic_norm(
                    depth_hidden_base + self.source_acoustic_residual_scale * acoustic
                )

        style_attended = None
        if self.enable_source_style_token_conditioning:
            if source_acoustic_embedding is None:
                raise RuntimeError(
                    "Mimi source style-token conditioning is enabled but "
                    "source_acoustic_embedding was not provided."
                )
            if source_acoustic_embedding.ndim != 3:
                raise RuntimeError(
                    "Mimi source style-token conditioning expects packed "
                    "[B, 1+K, D] source acoustic embeddings."
                )
            expected_tokens = 1 + self.source_style_token_count
            if source_acoustic_embedding.size(1) < expected_tokens:
                raise RuntimeError(
                    "Packed Mimi source acoustic conditioning has too few tokens: "
                    f"expected {expected_tokens}, got {source_acoustic_embedding.size(1)}."
                )
            if (
                self.source_style_token_input_norm is None
                or self.source_style_token_proj is None
                or self.source_style_token_position is None
                or self.source_style_token_query_norm is None
                or self.source_style_token_cross_attn is None
                or self.source_style_token_gates is None
            ):
                raise RuntimeError(
                    "Mimi source style-token conditioning is enabled but not initialized."
                )
            style_tokens = source_acoustic_embedding[:, 1:expected_tokens]
            if self.source_acoustic_detach:
                style_tokens = style_tokens.detach()
            style_tokens = style_tokens.to(
                device=depth_hidden_base.device,
                dtype=depth_hidden_base.dtype,
            )
            style_tokens = self.source_style_token_input_norm(style_tokens)
            style_tokens = self.source_style_token_proj(style_tokens)
            style_tokens = style_tokens + self.source_style_token_position.to(
                device=style_tokens.device,
                dtype=style_tokens.dtype,
            )
            style_query = self.source_style_token_query_norm(depth_hidden_base)
            style_attended, _ = self.source_style_token_cross_attn(
                query=style_query,
                key=style_tokens,
                value=style_tokens,
                need_weights=False,
            )
        depth_hidden = (
            depth_hidden_acoustic
            if self.source_acoustic_depth_start_level <= self.num_temporal_codebook_levels
            else depth_hidden_base
        )
        if self.enable_source_conditioning and source_memory is not None:
            if self.source_cross_attn is None or self.source_conditioning_norm is None:
                raise RuntimeError("Depth source conditioning is enabled but not initialized.")
            source_for_depth = (
                source_memory.detach() if self.source_conditioning_detach else source_memory
            )
            source_for_depth = source_for_depth.to(
                device=depth_hidden.device,
                dtype=depth_hidden.dtype,
            )
            if self.source_memory_proj is not None:
                source_for_depth = self.source_memory_proj(source_for_depth)
            source_key_padding_mask = None
            if source_memory_mask is not None:
                source_key_padding_mask = source_memory_mask.to(device=depth_hidden.device)
                if source_key_padding_mask.dtype != torch.bool:
                    source_key_padding_mask = source_key_padding_mask == 0
            source_attended, _ = self.source_cross_attn(
                query=depth_hidden,
                key=source_for_depth,
                value=source_for_depth,
                key_padding_mask=source_key_padding_mask,
                need_weights=False,
            )
            depth_hidden = self.source_conditioning_norm(
                depth_hidden + self.source_conditioning_residual_scale * source_attended
            )

        if forced_first_ids is not None:
            current_ids = forced_first_ids
        else:
            current_ids = _ids_from_logits(
                first_logits,
                0,
                use_temporal_scheduled_sampling=True,
                use_chain_scheduled_sampling=False,
            )
        safe_ids = current_ids.clamp(min=0, max=self.vocab_size - 1)
        prev_emb = self.codebook_emb[0](safe_ids)
        valid = (current_ids != ignore_index).unsqueeze(-1).type_as(prev_emb)
        prev_emb = prev_emb * valid

        if self.enable_nar_source_prompt:
            if self.num_temporal_codebook_levels != 1:
                raise RuntimeError(
                    "TransVIP-style Mimi NAR depth currently requires "
                    "num_temporal_codebook_levels=1."
                )
            source_codes = self._runtime_source_codec_tokens
            source_mask = self._runtime_source_codec_mask
            if source_codes is None or source_mask is None:
                raise RuntimeError(
                    "Mimi NAR depth is enabled but the per-sample source codec prompt "
                    "was not prepared from source_acoustic_wavs."
                )
            source_codes = source_codes.to(device=temporal_hidden.device, dtype=torch.long)
            source_mask = source_mask.to(device=temporal_hidden.device, dtype=torch.bool)
            if source_codes.ndim != 3 or source_codes.size(1) != self.num_quantizers:
                raise RuntimeError(
                    "Mimi NAR source prompt must be [B, Q, S], got "
                    f"{tuple(source_codes.shape)}."
                )
            if source_codes.size(0) == 1 and bsz > 1:
                source_codes = source_codes.expand(bsz, -1, -1)
                source_mask = source_mask.expand(bsz, -1)
            if source_codes.size(0) != bsz:
                raise RuntimeError(
                    f"Mimi NAR source batch {source_codes.size(0)} != target batch {bsz}."
                )
            expected_source_mask_shape = (bsz, source_codes.size(2))
            if tuple(source_mask.shape) != expected_source_mask_shape:
                raise RuntimeError(
                    "Mimi NAR source mask must be [B, S]: "
                    f"expected {expected_source_mask_shape}, got "
                    f"{tuple(source_mask.shape)}."
                )

            oracle_prefix_levels = max(1, int(forced_depth_prefix_levels))
            if forced_depth_ids is not None:
                forced_depth_ids = forced_depth_ids.to(
                    device=temporal_hidden.device,
                    dtype=torch.long,
                )
                if forced_depth_ids.shape != (bsz, seq_len, self.num_quantizers):
                    raise RuntimeError(
                        "forced_depth_ids must match the target [B,T,Q]: "
                        f"expected {(bsz, seq_len, self.num_quantizers)}, "
                        f"got {tuple(forced_depth_ids.shape)}."
                    )
                if oracle_prefix_levels > self.num_quantizers:
                    raise RuntimeError(
                        "forced_depth_prefix_levels cannot exceed the number of "
                        f"modeled codebooks ({self.num_quantizers})."
                    )

            frame_len = seq_len
            source_emb = 0.0
            for level in range(self.num_quantizers):
                ids = source_codes[:, level].clamp(min=0, max=self.vocab_size - 1)
                source_emb = source_emb + self.codebook_emb[level](ids)
            if self._runtime_zero_source_codec_prompt:
                source_emb = torch.zeros_like(source_emb)
            else:
                source_emb = source_emb + self.nar_depth_segment_emb.weight[0].view(1, 1, -1)
            expected_source_shape = (
                bsz,
                source_codes.size(2),
                self.depth_hidden_size,
            )
            if tuple(source_emb.shape) != expected_source_shape:
                raise RuntimeError(
                    "Mimi NAR source embedding must be [B, S, D_depth]: "
                    f"expected {expected_source_shape}, got {tuple(source_emb.shape)}."
                )

            source_len = int(source_emb.size(1))
            has_campplus_prefix = (
                self.enable_campplus_depth_conditioning
                and self.campplus_depth_conditioning_mode == "prefix_token"
            )
            prefix_len = source_len + int(has_campplus_prefix)
            total_len = prefix_len + frame_len
            if total_len > self.nar_max_positions:
                raise RuntimeError(
                    f"Mimi NAR depth sequence length {total_len} exceeds "
                    f"mimi_nar_depth_max_positions={self.nar_max_positions}."
                )
            nar_attention_mask = None
            if self.nar_attention_mode == "prefix_causal":
                # SPK plus source codecs form one fully observed prefix. Prefix
                # positions cannot read target positions; target t reads the
                # complete prefix plus target <= t. Blocking prefix->target
                # prevents indirect future leakage through prefix states.
                nar_attention_mask = torch.zeros(
                    (total_len, total_len),
                    device=temporal_hidden.device,
                    dtype=torch.bool,
                )
                nar_attention_mask[:prefix_len, prefix_len:] = True
                nar_attention_mask[prefix_len:, prefix_len:] = torch.triu(
                    torch.ones(
                        (frame_len, frame_len),
                        device=temporal_hidden.device,
                        dtype=torch.bool,
                    ),
                    diagonal=1,
                )

            known_ids: List[torch.Tensor] = [current_ids]
            final_hidden = depth_hidden_base
            campplus_layer_affines = None
            campplus_input_condition = None
            campplus_prefix_token = None
            if self.enable_campplus_depth_conditioning:
                if self.campplus_depth_conditioning_mode == "per_layer_film":
                    # The source vector and FiLM projections are identical for all
                    # C1-C15 passes. Reuse them across codebook passes.
                    campplus_layer_affines = self._prepare_campplus_layer_affines(
                        source_emb,
                        campplus_source_embedding,
                    )
                elif self.campplus_depth_conditioning_mode == "input_add":
                    campplus_input_condition = self._prepare_campplus_input_condition(
                        source_emb,
                        campplus_source_embedding,
                    )
                else:
                    campplus_prefix_token = self._prepare_campplus_prefix_token(
                        source_emb,
                        campplus_source_embedding,
                    )
                    campplus_prefix_token = (
                        campplus_prefix_token
                        + self.nar_depth_segment_emb.weight[1].view(1, 1, -1)
                    )

            def _build_nar_level_input(
                actual_level: int,
                lower_ids: List[torch.Tensor],
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                target_emb = temporal_hidden.new_zeros(
                    (bsz, frame_len, self.depth_hidden_size)
                )
                target_valid = torch.ones(
                    (bsz, frame_len),
                    device=temporal_hidden.device,
                    dtype=torch.bool,
                )
                for known_level, aligned_ids in enumerate(lower_ids):
                    level_valid = aligned_ids != ignore_index
                    target_valid &= level_valid
                    safe = aligned_ids.clamp(min=0, max=self.vocab_size - 1)
                    target_emb = target_emb + self.codebook_emb[known_level](safe) * (
                        level_valid.unsqueeze(-1).type_as(target_emb)
                    )
                target_emb = target_emb + self.nar_depth_segment_emb.weight[2].view(
                    1, 1, -1
                )
                target_emb = target_emb + self.nar_depth_level_emb.weight[
                    actual_level
                ].view(1, 1, -1)
                expected_target_shape = (bsz, frame_len, self.depth_hidden_size)
                if tuple(target_emb.shape) != expected_target_shape:
                    raise RuntimeError(
                        "Mimi NAR target embedding must be [B, T, D_depth]: "
                        f"expected {expected_target_shape}, got "
                        f"{tuple(target_emb.shape)}."
                    )
                if (
                    campplus_input_condition is not None
                    and actual_level >= self.campplus_depth_start_level
                ):
                    target_emb = target_emb + campplus_input_condition.unsqueeze(1)
                # The source speaker token and source codecs are observed prefix
                # positions; only target lower-codebook states vary by level.
                if campplus_prefix_token is not None:
                    level_input = torch.cat(
                        [campplus_prefix_token, source_emb, target_emb],
                        dim=1,
                    )
                    speaker_padding_mask = torch.zeros(
                        (bsz, 1),
                        device=target_valid.device,
                        dtype=torch.bool,
                    )
                    level_padding_mask = torch.cat(
                        [speaker_padding_mask, ~source_mask, ~target_valid],
                        dim=1,
                    )
                else:
                    level_input = torch.cat([source_emb, target_emb], dim=1)
                    level_padding_mask = torch.cat(
                        [~source_mask, ~target_valid],
                        dim=1,
                    )
                positions = torch.arange(total_len, device=level_input.device)
                level_input = level_input + self.nar_depth_pos_emb(positions).unsqueeze(0)
                expected_input_shape = (bsz, total_len, self.depth_hidden_size)
                expected_padding_shape = (bsz, total_len)
                if tuple(level_input.shape) != expected_input_shape:
                    raise RuntimeError(
                        "Mimi NAR input must be [B, prefix+T, D_depth]: "
                        f"expected {expected_input_shape}, got "
                        f"{tuple(level_input.shape)}."
                    )
                if tuple(level_padding_mask.shape) != expected_padding_shape:
                    raise RuntimeError(
                        "Mimi NAR padding mask must be [B, prefix+T]: "
                        f"expected {expected_padding_shape}, got "
                        f"{tuple(level_padding_mask.shape)}."
                    )
                return level_input, level_padding_mask, target_valid

            def _level_logits_from_hidden(
                actual_level: int,
                nar_hidden: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                target_hidden = self.nar_depth_norm(nar_hidden[:, -frame_len:])
                expected_hidden_shape = (bsz, frame_len, self.depth_hidden_size)
                if tuple(target_hidden.shape) != expected_hidden_shape:
                    raise RuntimeError(
                        "Mimi NAR target hidden must be [B, T, D_depth]: "
                        f"expected {expected_hidden_shape}, got "
                        f"{tuple(target_hidden.shape)}."
                    )
                level_logits = self.depth_transformer.level_heads[
                    actual_level - 1
                ](target_hidden).to(temporal_hidden.dtype)
                return level_logits, target_hidden

            level_batch_size = min(
                self.nar_teacher_force_level_batch_size,
                self.num_quantizers - 1,
            )
            batch_teacher_forced_levels = (
                teacher_force and labels is not None and level_batch_size > 1
            )
            if batch_teacher_forced_levels:
                # Under teacher forcing every Ck input is known before the NAR
                # forward. Stack independent codebook passes along the batch
                # dimension while retaining each level-specific embedding/head.
                for actual_level in range(1, self.num_quantizers - 1):
                    if (
                        forced_depth_ids is not None
                        and actual_level < oracle_prefix_levels
                    ):
                        next_ids = forced_depth_ids[:, :, actual_level]
                    else:
                        next_ids = labels[:, :, actual_level]
                    known_ids.append(next_ids)

                for chunk_start in range(1, self.num_quantizers, level_batch_size):
                    chunk_levels = list(
                        range(
                            chunk_start,
                            min(
                                chunk_start + level_batch_size,
                                self.num_quantizers,
                            ),
                        )
                    )
                    chunk_inputs: List[torch.Tensor] = []
                    chunk_padding_masks: List[torch.Tensor] = []
                    chunk_target_valid_masks: List[torch.Tensor] = []
                    for actual_level in chunk_levels:
                        (
                            level_input,
                            level_padding_mask,
                            level_target_valid,
                        ) = _build_nar_level_input(
                            actual_level,
                            known_ids[:actual_level],
                        )
                        chunk_inputs.append(level_input)
                        chunk_padding_masks.append(level_padding_mask)
                        chunk_target_valid_masks.append(level_target_valid)

                    stacked_input = torch.cat(chunk_inputs, dim=0)
                    stacked_padding_mask = torch.cat(chunk_padding_masks, dim=0)
                    stacked_layer_affines = None
                    if campplus_layer_affines is not None:
                        stacked_layer_affines = [
                            (
                                torch.cat(
                                    [
                                        scale
                                        if level >= self.campplus_depth_start_level
                                        else torch.zeros_like(scale)
                                        for level in chunk_levels
                                    ],
                                    dim=0,
                                ),
                                torch.cat(
                                    [
                                        shift
                                        if level >= self.campplus_depth_start_level
                                        else torch.zeros_like(shift)
                                        for level in chunk_levels
                                    ],
                                    dim=0,
                                ),
                            )
                            for scale, shift in campplus_layer_affines
                        ]
                    stacked_hidden = self._run_nar_depth_transformer(
                        stacked_input,
                        nar_attention_mask,
                        stacked_padding_mask,
                        campplus_source_embedding,
                        campplus_layer_affines=stacked_layer_affines,
                        apply_campplus_conditioning=any(
                            level >= self.campplus_depth_start_level
                            for level in chunk_levels
                        ),
                    )
                    for chunk_idx, actual_level in enumerate(chunk_levels):
                        level_hidden = stacked_hidden[
                            chunk_idx * bsz : (chunk_idx + 1) * bsz
                        ]
                        level_logits, target_hidden = _level_logits_from_hidden(
                            actual_level,
                            level_hidden,
                        )
                        logits_per_level.append(level_logits)
                        final_hidden = target_hidden
                        if (
                            self.campplus_identity_level_aggregation
                            == "conditioned_mean"
                            and self.campplus_identity_head is not None
                            and campplus_identity_embedding is not None
                            and actual_level >= self.campplus_depth_start_level
                        ):
                            per_level_identity_losses.append(
                                self.campplus_identity_loss(
                                    target_hidden,
                                    chunk_target_valid_masks[chunk_idx],
                                    campplus_identity_embedding,
                                )
                            )
                mean_identity_loss = (
                    torch.stack(per_level_identity_losses).mean()
                    if per_level_identity_losses
                    else None
                )
                return logits_per_level, final_hidden, mean_identity_loss

            for actual_level in range(1, self.num_quantizers):
                nar_input, padding_mask, target_valid = _build_nar_level_input(
                    actual_level,
                    known_ids,
                )
                nar_hidden = self._run_nar_depth_transformer(
                    nar_input,
                    nar_attention_mask,
                    padding_mask,
                    campplus_source_embedding,
                    campplus_layer_affines=campplus_layer_affines,
                    apply_campplus_conditioning=(
                        actual_level >= self.campplus_depth_start_level
                    ),
                )
                level_logits, target_hidden = _level_logits_from_hidden(
                    actual_level,
                    nar_hidden,
                )
                final_hidden = target_hidden
                logits_per_level.append(level_logits)
                if (
                    self.campplus_identity_level_aggregation == "conditioned_mean"
                    and self.campplus_identity_head is not None
                    and campplus_identity_embedding is not None
                    and actual_level >= self.campplus_depth_start_level
                ):
                    per_level_identity_losses.append(
                        self.campplus_identity_loss(
                            target_hidden,
                            target_valid,
                            campplus_identity_embedding,
                        )
                    )
                if actual_level < self.num_quantizers - 1:
                    if (
                        forced_depth_ids is not None
                        and actual_level < oracle_prefix_levels
                    ):
                        next_ids = forced_depth_ids[:, :, actual_level]
                    elif labels is not None and teacher_force:
                        next_ids = labels[:, :, actual_level]
                    else:
                        next_ids = level_logits.argmax(dim=-1)
                    known_ids.append(next_ids)
            mean_identity_loss = (
                torch.stack(per_level_identity_losses).mean()
                if per_level_identity_losses
                else None
            )
            return logits_per_level, final_hidden, mean_identity_loss

        for extra_idx, temporal_head in enumerate(self.temporal_codebook_heads):
            actual_level = extra_idx + 1
            logits = temporal_head(temporal_hidden)
            logits_per_level.append(logits)
            current_ids = _ids_from_logits(
                logits,
                actual_level,
                use_temporal_scheduled_sampling=True,
                use_chain_scheduled_sampling=False,
            )
            safe_ids = current_ids.clamp(min=0, max=self.vocab_size - 1)
            prev_emb = self.codebook_emb[actual_level](safe_ids)
            valid = (current_ids != ignore_index).unsqueeze(-1).type_as(prev_emb)
            prev_emb = prev_emb * valid

        for actual_level in range(self.num_temporal_codebook_levels, self.num_quantizers):
            depth_idx = actual_level - self.num_temporal_codebook_levels
            level_depth_hidden = (
                depth_hidden_acoustic
                if actual_level >= self.source_acoustic_depth_start_level
                else depth_hidden_base
            )
            if (
                style_attended is not None
                and actual_level >= self.source_style_token_depth_start_level
            ):
                assert self.source_style_token_gates is not None
                gate = self.source_style_token_residual_scale * torch.tanh(
                    self.source_style_token_gates[actual_level]
                )
                level_depth_hidden = level_depth_hidden + gate.to(
                    dtype=level_depth_hidden.dtype
                ) * style_attended
            depth_input = level_depth_hidden + prev_emb
            depth_output = self.depth_transformer(
                depth_inputs=depth_input,
                temporal_memory=level_depth_hidden,
            )
            logits = self.depth_transformer.level_heads[depth_idx](depth_output)
            logits_per_level.append(logits)

            if actual_level < self.num_quantizers - 1:
                current_ids = _ids_from_logits(
                    logits,
                    actual_level,
                    use_temporal_scheduled_sampling=False,
                    use_chain_scheduled_sampling=True,
                )
                safe_ids = current_ids.clamp(min=0, max=self.vocab_size - 1)
                prev_emb = self.codebook_emb[actual_level](safe_ids)
                valid = (current_ids != ignore_index).unsqueeze(-1).type_as(prev_emb)
                prev_emb = prev_emb * valid

        return logits_per_level, depth_hidden, None


class WhisperTemporalDepthTransformer(nn.Module):
    """
    Speech-to-speech model with Whisper encoder, temporal decoder, and depth decoder.

    Architecture:
    1. Whisper Encoder: Processes input audio features
    2. Temporal decoder: consumes previous code frames and cross-attends to Whisper states
    3. Depth decoder: predicts modeled codebook levels 1 to N-1, where each level takes
       temporal_hidden + embed(previous_level_token) as input
    """

    def __init__(
        self,
        speech_encoder_type_: str,
        speech_encoder_name_: str,
        qwen_name: str,
        codebook_size_: int,
        vocab_size_: int,  # codebook_size + 2 (includes BOS/EOS)
        input_num_quantizers_: int,
        num_codebook_levels_: int,
        # Temporal decoder config
        temporal_num_layers: int = 8,
        temporal_num_heads: int = 16,
        temporal_hidden_size_: Optional[int] = None,
        temporal_auto_adjust_heads_: bool = True,
        temporal_ffn_dim_: Optional[int] = None,
        temporal_dropout_: float = 0.1,
        temporal_activation_: str = "gelu",
        temporal_max_positions_: int = 4096,
        temporal_context_levels_: Optional[int] = None,
        num_temporal_codebook_levels_: int = 1,
        # Depth decoder config
        depth_num_layers_: int = 4,
        depth_num_heads_: int = 8,
        depth_hidden_size_: Optional[int] = None,
        depth_ffn_dim_: Optional[int] = None,
        depth_dropout_: float = 0.1,
        depth_auto_adjust_heads_: bool = True,
        # Textless semantic prefix / source-unit auxiliary config
        enable_semantic_prefix_ar_: bool = False,
        enable_source_unit_auxiliary_: bool = False,
        source_unit_vocab_size_: int = 1000,
        source_unit_loss_weight_: float = 0.2,
        source_unit_blank_id_: Optional[int] = None,
        enable_source_unit_transformer_auxiliary_: bool = False,
        source_unit_transformer_loss_weight_: float = 0.5,
        source_unit_transformer_num_layers_: int = 2,
        source_unit_transformer_num_heads_: int = 8,
        source_unit_transformer_ffn_dim_: Optional[int] = None,
        source_unit_transformer_dropout_: float = 0.1,
        source_unit_transformer_max_positions_: int = 513,
        source_adapter_num_layers_: int = 0,
        source_adapter_num_heads_: int = 8,
        source_adapter_ffn_dim_: Optional[int] = None,
        source_adapter_dropout_: float = 0.1,
        source_adapter_residual_: bool = True,
        source_adapter_residual_scale_: float = 0.1,
        enable_depth_source_conditioning_: bool = False,
        depth_source_conditioning_num_heads_: int = 8,
        depth_source_conditioning_dropout_: float = 0.1,
        depth_source_conditioning_residual_scale_: float = 0.2,
        depth_source_conditioning_detach_: bool = True,
        enable_mimi_source_acoustic_conditioning_: bool = False,
        mimi_source_acoustic_model_name_: str = "kyutai/mimi",
        mimi_source_acoustic_input_sample_rate_: int = 16000,
        mimi_source_acoustic_pooling_: str = "mean",
        use_precomputed_source_acoustic_embeddings_: bool = False,
        skip_mimi_source_acoustic_encoder_when_precomputed_: bool = False,
        mimi_source_acoustic_residual_scale_: float = 0.3,
        mimi_source_acoustic_depth_start_level_: int = 1,
        mimi_source_acoustic_detach_: bool = True,
        enable_mimi_source_style_token_bank_: bool = False,
        mimi_source_style_token_count_: int = 8,
        mimi_source_style_token_depth_start_level_: int = 1,
        mimi_source_style_token_num_heads_: int = 8,
        mimi_source_style_token_dropout_: float = 0.1,
        mimi_source_style_token_residual_scale_: float = 0.7,
        mimi_source_style_token_gate_init_: float = 0.10,
        mimi_source_style_token_c1_gate_init_: float = 0.03,
        enable_mimi_source_acoustic_temporal_conditioning_: bool = False,
        mimi_source_acoustic_temporal_residual_scale_: float = 0.3,
        mimi_source_acoustic_temporal_detach_: bool = True,
        enable_unity_source_memory_fusion_: bool = False,
        unity_source_memory_fusion_mode_: str = "concat",
        unity_source_memory_source_: str = "inner",
        unity_source_memory_layer_: int = 8,
        unity_source_memory_scale_: float = 0.5,
        unity_source_memory_detach_: bool = True,
        enable_mimi_source_speaker_prompt_: bool = False,
        mimi_source_speaker_prompt_detach_: bool = True,
        source_speaker_prompt_backend_: str = "mimi_mean_std",
        source_speaker_prompt_speech_layer_: int = 12,
        source_speaker_prompt_num_layers_: int = 2,
        source_speaker_prompt_num_heads_: int = 16,
        source_speaker_prompt_ffn_dim_: Optional[int] = 4096,
        source_speaker_prompt_dropout_: float = 0.1,
        enable_mimi_nar_depth_prompt_: bool = False,
        mimi_nar_depth_attention_mode_: str = "full",
        mimi_nar_depth_max_positions_: int = 4096,
        mimi_nar_depth_teacher_force_level_batch_size_: int = 1,
        enable_campplus_depth_conditioning_: bool = False,
        campplus_depth_conditioning_mode_: str = "per_layer_film",
        campplus_model_root_: str = "",
        campplus_checkpoint_path_: str = "",
        campplus_embedding_size_: int = 192,
        campplus_film_scale_: float = 1.0,
        campplus_depth_start_level_: int = 1,
        campplus_depth_identity_loss_weight_: float = 0.0,
        campplus_identity_supervision_: str = "source",
        campplus_identity_level_aggregation_: str = "final",
        transvip_repo_dir_: str = "",
        transvip_model_cfg_path_: Optional[str] = None,
        transvip_model_path_: str = "",
        transvip_model_name_: str = "",
        transvip_num_new_tokens_: Optional[int] = None,
        transvip_spk_encoder_path_: str = "",
        transvip_use_length_control_: bool = True,
        transvip_use_source_speaker_prompt_: bool = False,
        transvip_prompt_codec_path_: str = "",
        transvip_prompt_max_frames_: int = 500,
        transvip_source_checkpoint_path_: str = "",
        transvip_load_source_encoder_from_checkpoint_: bool = False,
        transvip_text_decoder_checkpoint_path_: str = "",
        transvip_load_text_decoder_from_checkpoint_: bool = False,
        transvip_text_decoder_text_vocab_size_: Optional[int] = None,
        temporal_decoder_backend_: str = "direct",
        # Target-text vocabulary and losses
        text_vocab_size_: Optional[int] = None,
        text_pad_token_id_: int = 0,
        text_loss_weight_: float = 0.2,
        text_label_smoothing_: float = 0.0,
        enable_codec_mass_loss_: bool = False,
        codec_mass_loss_weight_: float = 0.0,
        enable_c0_auxiliary_loss_: bool = False,
        c0_auxiliary_loss_weight_: float = 0.0,
        enable_c0_teacher_distill_: bool = False,
        c0_teacher_distill_weight_: float = 0.0,
        c0_teacher_distill_temperature_: float = 1.0,
        enable_c0_gold_anchored_distill_: bool = False,
        c0_gold_anchored_distill_weight_: float = 0.0,
        c0_gold_anchor_mass_: float = 0.75,
        c0_gold_anchored_distill_temperature_: float = 1.0,
        enable_c0_teacher_hidden_distill_: bool = False,
        c0_teacher_hidden_distill_weight_: float = 0.0,
        c0_teacher_hidden_loss_type_: str = "cosine",
        c0_teacher_hidden_dim_: Optional[int] = None,
        c0_teacher_hidden_use_projection_: bool = True,
        enable_text_codec_kd_loss_: bool = False,
        text_codec_kd_loss_weight_: float = 0.0,
        enable_transvip_style_loss_aggregation_: bool = False,
        transvip_style_speech_loss_weight_: float = 1.0,
        transvip_style_text_loss_weight_: float = 1.0,
        transvip_style_kd_loss_weight_: float = 1.0,
        transvip_style_depth_loss_weight_: float = 1.0,
        use_unified_code0_loss_in_objective_: bool = True,
        enable_text_codec_text_path_loss_: bool = False,
        text_codec_text_path_input_: str = "target",
        text_codec_t2t_loss_weight_: float = 1.0,
        text_codec_t2c_loss_weight_: float = 1.0,
        text_path_source_token_dropout_: float = 0.0,
        enable_direct_text_to_c0_auxiliary_: bool = False,
        direct_text_to_c0_input_: str = "target",
        direct_text_to_c0_loss_weight_: float = 0.0,
        direct_text_to_c0_zero_speech_memory_: bool = False,
        direct_text_to_c0_kd_weight_: float = 0.0,
        direct_text_to_c0_kd_temperature_: float = 1.0,
        tie_text_embeddings_: bool = True,
        text_max_positions_: int = 128,
        enable_text_prefix_ar_: bool = False,
        enable_text_codec_ar_: bool = False,
        enable_quality_cot_: bool = False,
        quality_cot_prompt_all_seps_: bool = False,
        enable_uniss_content_controls_: bool = False,
        uniss_start_content_token_id_: Optional[int] = None,
        uniss_end_content_token_id_: Optional[int] = None,
        quality_cot_source_text_prefix_extra_token_ids_after_bos_: Optional[List[int]] = None,
        quality_cot_source_text_max_tokens_: Optional[int] = None,
        quality_cot_target_text_max_tokens_: Optional[int] = None,
        quality_cot_source_text_loss_weight_: float = 1.0,
        quality_cot_target_text_loss_weight_: float = 1.0,
        quality_cot_code0_loss_weight_: float = 1.0,
        quality_cot_code1_loss_weight_: float = 0.0,
        quality_cot_depth_loss_weight_: float = 1.0,
        enable_target_only_speech_auxiliary_: bool = False,
        target_only_speech_text_loss_weight_: float = 0.0,
        target_only_speech_code0_loss_weight_: float = 0.0,
        enable_mixed_source_target_auxiliary_: bool = False,
        mixed_source_target_loss_weight_: float = 0.0,
        mixed_source_prediction_prob_: float = 0.0,
        mixed_source_start_step_: int = 0,
        mixed_source_warmup_steps_: int = 0,
        mixed_source_auxiliary_seed_: int = 59,
        enable_predicted_source_target_auxiliary_: bool = False,
        predicted_source_target_loss_weight_: float = 0.0,
        predicted_source_target_start_step_: int = 0,
        predicted_source_target_warmup_steps_: int = 0,
        predicted_source_target_max_source_tokens_: int = 64,
        predicted_source_target_batch_size_: int = 2,
        text_bos_token_id_: Optional[int] = None,
        text_sep_token_id_: Optional[int] = None,
        text_prefix_extra_token_ids_after_bos_: Optional[List[int]] = None,
        text_prefix_beam_size_: int = 1,
        text_prefix_min_tokens_: int = 0,
        text_prefix_sep_penalty_: float = 0.0,
        text_prefix_length_penalty_: float = 1.0,
        text_prefix_no_repeat_ngram_size_: int = 0,
        # Training config
        teacher_force: bool = True,
        level0_depth_objective_: bool = True,
        detach_temporal_for_depth_loss_: bool = True,
        temporal_scheduled_sampling_prob_: float = 0.0,
        temporal_scheduled_sampling_start_step_: int = 0,
        temporal_scheduled_sampling_warmup_steps_: int = 0,
        temporal_scheduled_sampling_mode_: str = "argmax",
        temporal_scheduled_sampling_topk_: int = 8,
        temporal_scheduled_sampling_temperature_: float = 1.0,
        temporal_scheduled_sampling_final_prob_: Optional[float] = None,
        temporal_scheduled_sampling_decay_start_step_: int = -1,
        temporal_scheduled_sampling_decay_steps_: int = 0,
        temporal_scheduled_sampling_preserve_last_n_: int = 0,
        depth_scheduled_sampling_prob_: float = 0.0,
        depth_scheduled_sampling_start_step_: int = 0,
        depth_scheduled_sampling_warmup_steps_: int = 0,
        depth_scheduled_sampling_mode_: str = "argmax",
        depth_scheduled_sampling_topk_: int = 8,
        depth_scheduled_sampling_temperature_: float = 1.0,
        depth_chain_scheduled_sampling_prob_: float = 0.0,
        depth_chain_scheduled_sampling_start_step_: int = 0,
        depth_chain_scheduled_sampling_warmup_steps_: int = 0,
        depth_chain_scheduled_sampling_mode_: str = "argmax",
        depth_chain_scheduled_sampling_topk_: int = 8,
        depth_chain_scheduled_sampling_temperature_: float = 1.0,
        level0_loss_weight_: float = 1.0,
        depth_loss_weight_: float = 1.0,
        depth_objective_start_level_: int = 1,
        level0_eos_loss_weight_: float = 1.0,
        level0_tail_loss_weight_: float = 1.0,
        level0_tail_loss_last_n_: int = 0,
        level0_label_smoothing_: float = 0.0,
        level0_label_smoothing_train_only_: bool = False,
        codebook_loss_weights_: Optional[List[float]] = None,
        freeze_whisper_encoder: bool = False,
        finetune_transvip_speech_encoder_: bool = False,
        freeze_temporal_transformer: bool = False,
        speech_encoder_output_layer_: Optional[int] = None,
        # Inference must not load the training-only text-memory teacher.
        load_text_path_encoder_: bool = True,
        train_only_depth_decoder_: bool = False,
    ):
        super().__init__()
        self.train_only_depth_decoder = bool(train_only_depth_decoder_)
        self.temporal_activation = str(temporal_activation_ or "gelu").strip().lower()
        if self.temporal_activation not in {"gelu", "relu"}:
            raise ValueError("temporal_activation must be one of {'gelu', 'relu'}.")
        self.ignore_index = -100
        self.teacher_force = teacher_force
        self.level0_depth_objective = level0_depth_objective_
        self.detach_temporal_for_depth_loss = detach_temporal_for_depth_loss_
        self.temporal_scheduled_sampling_prob = float(temporal_scheduled_sampling_prob_)
        self.temporal_scheduled_sampling_start_step = int(temporal_scheduled_sampling_start_step_)
        self.temporal_scheduled_sampling_warmup_steps = max(0, int(temporal_scheduled_sampling_warmup_steps_))
        self.temporal_scheduled_sampling_mode = str(temporal_scheduled_sampling_mode_).lower()
        self.temporal_scheduled_sampling_topk = max(1, int(temporal_scheduled_sampling_topk_))
        self.temporal_scheduled_sampling_temperature = max(1.0e-6, float(temporal_scheduled_sampling_temperature_))
        self.temporal_scheduled_sampling_final_prob = (
            float(temporal_scheduled_sampling_prob_)
            if temporal_scheduled_sampling_final_prob_ is None
            else float(temporal_scheduled_sampling_final_prob_)
        )
        self.temporal_scheduled_sampling_decay_start_step = int(
            temporal_scheduled_sampling_decay_start_step_
        )
        self.temporal_scheduled_sampling_decay_steps = max(
            0,
            int(temporal_scheduled_sampling_decay_steps_),
        )
        self.temporal_scheduled_sampling_preserve_last_n = max(
            0,
            int(temporal_scheduled_sampling_preserve_last_n_),
        )
        self.depth_scheduled_sampling_prob = float(depth_scheduled_sampling_prob_)
        self.depth_scheduled_sampling_start_step = int(depth_scheduled_sampling_start_step_)
        self.depth_scheduled_sampling_warmup_steps = max(0, int(depth_scheduled_sampling_warmup_steps_))
        self.depth_scheduled_sampling_mode = str(depth_scheduled_sampling_mode_).lower()
        self.depth_scheduled_sampling_topk = max(1, int(depth_scheduled_sampling_topk_))
        self.depth_scheduled_sampling_temperature = max(1.0e-6, float(depth_scheduled_sampling_temperature_))
        self.depth_chain_scheduled_sampling_prob = float(depth_chain_scheduled_sampling_prob_)
        self.depth_chain_scheduled_sampling_start_step = int(depth_chain_scheduled_sampling_start_step_)
        self.depth_chain_scheduled_sampling_warmup_steps = max(
            0,
            int(depth_chain_scheduled_sampling_warmup_steps_),
        )
        self.depth_chain_scheduled_sampling_mode = str(depth_chain_scheduled_sampling_mode_).lower()
        self.depth_chain_scheduled_sampling_topk = max(1, int(depth_chain_scheduled_sampling_topk_))
        self.depth_chain_scheduled_sampling_temperature = max(
            1.0e-6,
            float(depth_chain_scheduled_sampling_temperature_),
        )
        self._current_global_step = 0
        self._gradient_diag_objective: Optional[str] = None
        self.level0_loss_weight = level0_loss_weight_
        self.depth_loss_weight = depth_loss_weight_
        self.depth_objective_start_level = max(
            int(num_temporal_codebook_levels_),
            int(depth_objective_start_level_),
        )
        self.level0_eos_loss_weight = float(level0_eos_loss_weight_)
        self.level0_tail_loss_weight = float(level0_tail_loss_weight_)
        self.level0_tail_loss_last_n = max(0, int(level0_tail_loss_last_n_))
        self.level0_label_smoothing = max(0.0, min(0.999, float(level0_label_smoothing_)))
        self.level0_label_smoothing_train_only = bool(
            level0_label_smoothing_train_only_
        )
        self.codebook_loss_weights = (
            tuple(float(w) for w in codebook_loss_weights_)
            if codebook_loss_weights_ is not None
            else None
        )
        self.enable_semantic_prefix_ar = bool(enable_semantic_prefix_ar_)
        self.enable_source_unit_auxiliary = bool(enable_source_unit_auxiliary_)
        self.enable_source_unit_transformer_auxiliary = bool(
            enable_source_unit_transformer_auxiliary_
        )
        self.source_unit_vocab_size = int(source_unit_vocab_size_)
        self.source_unit_loss_weight = float(source_unit_loss_weight_)
        self.source_unit_transformer_loss_weight = float(
            source_unit_transformer_loss_weight_
        )
        self.source_unit_transformer_num_layers = max(
            1,
            int(source_unit_transformer_num_layers_),
        )
        self.source_unit_transformer_num_heads = max(
            1,
            int(source_unit_transformer_num_heads_),
        )
        self.source_unit_transformer_ffn_dim = source_unit_transformer_ffn_dim_
        self.source_unit_transformer_dropout = float(source_unit_transformer_dropout_)
        self.source_unit_transformer_max_positions = max(
            2,
            int(source_unit_transformer_max_positions_),
        )
        self.source_adapter_num_layers = max(0, int(source_adapter_num_layers_))
        self.source_adapter_num_heads = max(1, int(source_adapter_num_heads_))
        self.source_adapter_ffn_dim = source_adapter_ffn_dim_
        self.source_adapter_dropout = float(source_adapter_dropout_)
        self.source_adapter_residual = bool(source_adapter_residual_)
        self.source_adapter_residual_scale = float(source_adapter_residual_scale_)
        self.enable_depth_source_conditioning = bool(enable_depth_source_conditioning_)
        self.depth_source_conditioning_num_heads = max(1, int(depth_source_conditioning_num_heads_))
        self.depth_source_conditioning_dropout = float(depth_source_conditioning_dropout_)
        self.depth_source_conditioning_residual_scale = float(
            depth_source_conditioning_residual_scale_
        )
        self.depth_source_conditioning_detach = bool(depth_source_conditioning_detach_)
        self.enable_mimi_source_acoustic_conditioning = bool(
            enable_mimi_source_acoustic_conditioning_
        )
        self.mimi_source_acoustic_model_name = str(
            mimi_source_acoustic_model_name_ or "kyutai/mimi"
        )
        self.mimi_source_acoustic_input_sample_rate = max(
            1, int(mimi_source_acoustic_input_sample_rate_)
        )
        self.mimi_source_acoustic_pooling = str(
            mimi_source_acoustic_pooling_ or "mean"
        ).strip().lower()
        if self.mimi_source_acoustic_pooling not in {"mean", "std", "mean_std"}:
            raise ValueError(
                "mimi_source_acoustic_pooling must be one of: mean, std, mean_std."
            )
        self.use_precomputed_source_acoustic_embeddings = bool(
            use_precomputed_source_acoustic_embeddings_
        )
        self.skip_mimi_source_acoustic_encoder_when_precomputed = bool(
            skip_mimi_source_acoustic_encoder_when_precomputed_
        )
        self.mimi_source_acoustic_residual_scale = float(
            mimi_source_acoustic_residual_scale_
        )
        self.mimi_source_acoustic_depth_start_level = max(
            1, int(mimi_source_acoustic_depth_start_level_)
        )
        self.mimi_source_acoustic_detach = bool(mimi_source_acoustic_detach_)
        self.enable_mimi_source_style_token_bank = bool(
            enable_mimi_source_style_token_bank_
        )
        self.mimi_source_style_token_count = max(
            1, int(mimi_source_style_token_count_)
        )
        self.mimi_source_style_token_depth_start_level = max(
            1, int(mimi_source_style_token_depth_start_level_)
        )
        self.mimi_source_style_token_num_heads = max(
            1, int(mimi_source_style_token_num_heads_)
        )
        self.mimi_source_style_token_dropout = float(
            mimi_source_style_token_dropout_
        )
        self.mimi_source_style_token_residual_scale = float(
            mimi_source_style_token_residual_scale_
        )
        self.mimi_source_style_token_gate_init = float(
            mimi_source_style_token_gate_init_
        )
        self.mimi_source_style_token_c1_gate_init = float(
            mimi_source_style_token_c1_gate_init_
        )
        if not 0.0 <= self.mimi_source_style_token_dropout < 1.0:
            raise ValueError("mimi_source_style_token_dropout must be in [0, 1).")
        if (
            self.enable_mimi_source_style_token_bank
            and not self.enable_mimi_source_acoustic_conditioning
        ):
            raise ValueError(
                "enable_mimi_source_style_token_bank=True requires "
                "enable_mimi_source_acoustic_conditioning=True."
            )
        if (
            self.enable_mimi_source_style_token_bank
            and self.mimi_source_acoustic_pooling != "mean_std"
        ):
            raise ValueError(
                "Mimi source style-token bank requires "
                "mimi_source_acoustic_pooling='mean_std'."
            )
        if (
            self.enable_mimi_source_style_token_bank
            and self.skip_mimi_source_acoustic_encoder_when_precomputed
        ):
            raise ValueError(
                "Mimi source style-token bank needs online Mimi frame features; set "
                "skip_mimi_source_acoustic_encoder_when_precomputed=False."
            )
        self.enable_mimi_source_acoustic_temporal_conditioning = bool(
            enable_mimi_source_acoustic_temporal_conditioning_
        )
        self.mimi_source_acoustic_temporal_residual_scale = float(
            mimi_source_acoustic_temporal_residual_scale_
        )
        self.mimi_source_acoustic_temporal_detach = bool(
            mimi_source_acoustic_temporal_detach_
        )
        self.enable_unity_source_memory_fusion = bool(enable_unity_source_memory_fusion_)
        self.unity_source_memory_fusion_mode = str(
            unity_source_memory_fusion_mode_ or "concat"
        ).strip().lower()
        if self.unity_source_memory_fusion_mode not in {"concat"}:
            raise ValueError("unity_source_memory_fusion_mode currently supports only 'concat'.")
        self.unity_source_memory_source = str(
            unity_source_memory_source_ or "inner"
        ).strip().lower()
        if self.unity_source_memory_source not in {"inner", "adaptor"}:
            raise ValueError("unity_source_memory_source must be one of: inner, adaptor.")
        self.unity_source_memory_layer = max(1, int(unity_source_memory_layer_))
        self.unity_source_memory_scale = float(unity_source_memory_scale_)
        self.unity_source_memory_detach = bool(unity_source_memory_detach_)
        self.enable_mimi_source_speaker_prompt = bool(
            enable_mimi_source_speaker_prompt_
        )
        self.source_speaker_prompt_backend = str(
            source_speaker_prompt_backend_ or "mimi_mean_std"
        ).strip().lower()
        if self.source_speaker_prompt_backend not in {
            "mimi_mean_std",
            "w2vbert_hidden",
            "mimi_latent",
        }:
            raise ValueError(
                "source_speaker_prompt_backend must be one of: "
                "mimi_mean_std, w2vbert_hidden, mimi_latent."
            )
        self.source_speaker_prompt_speech_layer = int(
            source_speaker_prompt_speech_layer_
        )
        self.source_speaker_prompt_num_layers = max(
            1, int(source_speaker_prompt_num_layers_)
        )
        self.source_speaker_prompt_num_heads = max(
            1, int(source_speaker_prompt_num_heads_)
        )
        self.source_speaker_prompt_ffn_dim = (
            None
            if source_speaker_prompt_ffn_dim_ is None
            else int(source_speaker_prompt_ffn_dim_)
        )
        self.source_speaker_prompt_dropout = float(
            source_speaker_prompt_dropout_
        )
        self.enable_mimi_nar_depth_prompt = bool(enable_mimi_nar_depth_prompt_)
        self.mimi_nar_depth_attention_mode = str(
            mimi_nar_depth_attention_mode_ or "full"
        ).strip().lower()
        if self.mimi_nar_depth_attention_mode not in {"full", "prefix_causal"}:
            raise ValueError(
                "mimi_nar_depth_attention_mode must be one of: full, prefix_causal."
            )
        self.mimi_nar_depth_max_positions = max(2, int(mimi_nar_depth_max_positions_))
        self.mimi_nar_depth_teacher_force_level_batch_size = max(
            1, int(mimi_nar_depth_teacher_force_level_batch_size_)
        )
        self.enable_campplus_depth_conditioning = bool(
            enable_campplus_depth_conditioning_
        )
        self.campplus_depth_conditioning_mode = str(
            campplus_depth_conditioning_mode_ or "per_layer_film"
        ).strip().lower()
        if self.campplus_depth_conditioning_mode not in {
            "per_layer_film",
            "input_add",
            "prefix_token",
        }:
            raise ValueError(
                "campplus_depth_conditioning_mode must be one of: "
                "per_layer_film, input_add, prefix_token."
            )
        self.campplus_model_root = str(campplus_model_root_)
        self.campplus_checkpoint_path = str(campplus_checkpoint_path_)
        self.campplus_embedding_size = max(1, int(campplus_embedding_size_))
        self.campplus_film_scale = float(campplus_film_scale_)
        self.campplus_depth_start_level = max(
            1, int(campplus_depth_start_level_)
        )
        if (
            self.enable_campplus_depth_conditioning
            and self.campplus_depth_conditioning_mode == "prefix_token"
            and self.campplus_depth_start_level != 1
        ):
            raise ValueError(
                "prefix_token CAMPPlus conditioning is present for every NAR "
                "depth level; campplus_depth_start_level must be 1."
            )
        self.campplus_depth_identity_loss_weight = float(
            campplus_depth_identity_loss_weight_
        )
        self.campplus_identity_supervision = str(
            campplus_identity_supervision_ or "source"
        ).strip().lower()
        self.campplus_identity_level_aggregation = str(
            campplus_identity_level_aggregation_ or "final"
        ).strip().lower()
        if self.campplus_identity_supervision not in {"source", "target"}:
            raise ValueError(
                "campplus_identity_supervision must be one of: source, target."
            )
        if self.campplus_identity_level_aggregation not in {
            "final",
            "conditioned_mean",
        }:
            raise ValueError(
                "campplus_identity_level_aggregation must be one of: "
                "final, conditioned_mean."
            )
        if self.enable_campplus_depth_conditioning and not self.enable_mimi_nar_depth_prompt:
            raise ValueError(
                "CAMPPlus depth conditioning currently requires "
                "enable_mimi_nar_depth_prompt=True."
            )
        self._campplus_source_model_holder: List[nn.Module] = []
        self.mimi_source_speaker_prompt_detach = bool(
            mimi_source_speaker_prompt_detach_
        )
        self._runtime_w2vbert_prompt_frames = None
        self._runtime_w2vbert_prompt_mask = None
        self._runtime_mimi_prompt_frames = None
        self._runtime_mimi_prompt_mask = None
        self.mimi_source_acoustic_model = None
        self.mimi_source_acoustic_model_sample_rate = self.mimi_source_acoustic_input_sample_rate
        self.mimi_source_acoustic_hidden_size = None
        self.mimi_source_acoustic_base_hidden_size = None
        self.transvip_repo_dir = str(transvip_repo_dir_)
        self.transvip_model_cfg_path = str(transvip_model_cfg_path_ or "")
        self.transvip_model_path = str(transvip_model_path_ or "")
        self.transvip_model_name = str(transvip_model_name_ or "")
        self.transvip_num_new_tokens = transvip_num_new_tokens_
        self.transvip_spk_encoder_path = str(transvip_spk_encoder_path_ or "")
        self.transvip_use_length_control = bool(transvip_use_length_control_)
        self.transvip_use_source_speaker_prompt = bool(transvip_use_source_speaker_prompt_)
        self.transvip_prompt_codec_path = str(transvip_prompt_codec_path_ or "")
        self.transvip_prompt_max_frames = max(1, int(transvip_prompt_max_frames_))
        self.transvip_source_checkpoint_path = str(transvip_source_checkpoint_path_ or "")
        self.transvip_load_source_encoder_from_checkpoint = bool(
            transvip_load_source_encoder_from_checkpoint_
        )
        self.transvip_text_decoder_checkpoint_path = str(transvip_text_decoder_checkpoint_path_ or "")
        self.transvip_load_text_decoder_from_checkpoint = bool(
            transvip_load_text_decoder_from_checkpoint_
        )
        self.transvip_text_decoder_text_vocab_size = (
            None
            if transvip_text_decoder_text_vocab_size_ is None
            else int(transvip_text_decoder_text_vocab_size_)
        )
        self.temporal_decoder_backend = str(temporal_decoder_backend_ or "direct").lower()
        self.use_unity_text_decoder_temporal = self.temporal_decoder_backend in {
            "unity_text_decoder",
            "transvip_text_decoder",
        }
        self.use_unity_direct_c0_temporal = self.temporal_decoder_backend in {
            "unity_direct_c0",
            "unity_shared_direct_c0",
        }
        self.use_unity_native_decoder_temporal = (
            self.use_unity_text_decoder_temporal
            or self.use_unity_direct_c0_temporal
        )
        if (
            bool(enable_direct_text_to_c0_auxiliary_)
            and self.use_unity_native_decoder_temporal
        ):
            raise ValueError(
                "enable_direct_text_to_c0_auxiliary=True is only for the Direct "
                "temporal decoder. For UnitY/TransVIP decoder runs, use "
                "enable_text_codec_text_path_loss instead."
            )
        self.source_unit_bos_id = self.source_unit_vocab_size
        self.source_unit_eos_id = self.source_unit_vocab_size + 1
        self.source_unit_pad_id = self.source_unit_vocab_size + 2
        self.source_unit_transformer_vocab_size = self.source_unit_vocab_size + 3
        self.source_unit_blank_id = (
            self.source_unit_vocab_size
            if source_unit_blank_id_ is None
            else int(source_unit_blank_id_)
        )
        if not 0 <= self.source_unit_blank_id <= self.source_unit_vocab_size:
            raise ValueError("source_unit_blank_id must be within the CTC output vocabulary.")
        self.text_label_smoothing = max(
            0.0,
            min(0.999, float(text_label_smoothing_)),
        )
        self.enable_text_prefix_ar = bool(enable_text_prefix_ar_)
        self.enable_text_codec_ar = bool(enable_text_codec_ar_)
        self.enable_quality_cot = bool(enable_quality_cot_)
        self.quality_cot_prompt_all_seps = bool(quality_cot_prompt_all_seps_)
        self.enable_uniss_content_controls = bool(enable_uniss_content_controls_)
        self.uniss_start_content_token_id = (
            None
            if uniss_start_content_token_id_ is None
            else int(uniss_start_content_token_id_)
        )
        self.uniss_end_content_token_id = (
            None
            if uniss_end_content_token_id_ is None
            else int(uniss_end_content_token_id_)
        )
        self.quality_cot_source_text_prefix_extra_token_ids_after_bos = [
            int(token_id)
            for token_id in (quality_cot_source_text_prefix_extra_token_ids_after_bos_ or [])
        ]
        self.quality_cot_source_text_max_tokens = max(
            2,
            int(
                quality_cot_source_text_max_tokens_
                if quality_cot_source_text_max_tokens_ is not None
                else text_max_positions_
            ),
        )
        self.quality_cot_target_text_max_tokens = max(
            2,
            int(
                quality_cot_target_text_max_tokens_
                if quality_cot_target_text_max_tokens_ is not None
                else text_max_positions_
            ),
        )
        self.quality_cot_source_text_loss_weight = float(
            quality_cot_source_text_loss_weight_
        )
        self.quality_cot_target_text_loss_weight = float(
            quality_cot_target_text_loss_weight_
        )
        self.quality_cot_code0_loss_weight = float(quality_cot_code0_loss_weight_)
        self.quality_cot_code1_loss_weight = float(quality_cot_code1_loss_weight_)
        self.quality_cot_depth_loss_weight = float(quality_cot_depth_loss_weight_)
        self.enable_target_only_speech_auxiliary = bool(
            enable_target_only_speech_auxiliary_
        )
        self.target_only_speech_text_loss_weight = float(
            target_only_speech_text_loss_weight_
        )
        self.target_only_speech_code0_loss_weight = float(
            target_only_speech_code0_loss_weight_
        )
        self.enable_mixed_source_target_auxiliary = bool(
            enable_mixed_source_target_auxiliary_
        )
        self.mixed_source_target_loss_weight = float(
            mixed_source_target_loss_weight_
        )
        self.mixed_source_prediction_prob = float(mixed_source_prediction_prob_)
        self.mixed_source_start_step = max(0, int(mixed_source_start_step_))
        self.mixed_source_warmup_steps = max(0, int(mixed_source_warmup_steps_))
        self.mixed_source_auxiliary_seed = int(mixed_source_auxiliary_seed_)
        self.enable_predicted_source_target_auxiliary = bool(
            enable_predicted_source_target_auxiliary_
        )
        self.predicted_source_target_loss_weight = float(
            predicted_source_target_loss_weight_
        )
        self.predicted_source_target_start_step = max(
            0,
            int(predicted_source_target_start_step_),
        )
        self.predicted_source_target_warmup_steps = max(
            0,
            int(predicted_source_target_warmup_steps_),
        )
        self.predicted_source_target_max_source_tokens = max(
            2,
            int(predicted_source_target_max_source_tokens_),
        )
        self.predicted_source_target_batch_size = max(
            1,
            int(predicted_source_target_batch_size_),
        )
        self.text_loss_weight = text_loss_weight_
        self.enable_codec_mass_loss = bool(enable_codec_mass_loss_)
        self.codec_mass_loss_weight = float(codec_mass_loss_weight_)
        self.enable_c0_auxiliary_loss = bool(enable_c0_auxiliary_loss_)
        self.c0_auxiliary_loss_weight = float(c0_auxiliary_loss_weight_)
        self.enable_c0_teacher_distill = bool(enable_c0_teacher_distill_)
        self.c0_teacher_distill_weight = float(c0_teacher_distill_weight_)
        self.c0_teacher_distill_temperature = max(1.0e-6, float(c0_teacher_distill_temperature_))
        self.enable_c0_gold_anchored_distill = bool(enable_c0_gold_anchored_distill_)
        self.c0_gold_anchored_distill_weight = float(c0_gold_anchored_distill_weight_)
        self.c0_gold_anchor_mass = max(0.0, min(1.0, float(c0_gold_anchor_mass_)))
        self.c0_gold_anchored_distill_temperature = max(
            1.0e-6,
            float(c0_gold_anchored_distill_temperature_),
        )
        self.enable_c0_teacher_hidden_distill = bool(enable_c0_teacher_hidden_distill_)
        self.c0_teacher_hidden_distill_weight = float(c0_teacher_hidden_distill_weight_)
        self.c0_teacher_hidden_loss_type = str(c0_teacher_hidden_loss_type_ or "cosine").strip().lower()
        if self.c0_teacher_hidden_loss_type not in {"cosine", "mse"}:
            raise ValueError("c0_teacher_hidden_loss_type must be one of: cosine, mse.")
        self.c0_teacher_hidden_dim = (
            None if c0_teacher_hidden_dim_ is None else int(c0_teacher_hidden_dim_)
        )
        self.c0_teacher_hidden_use_projection = bool(c0_teacher_hidden_use_projection_)
        self.enable_text_codec_kd_loss = bool(enable_text_codec_kd_loss_)
        self.text_codec_kd_loss_weight = float(text_codec_kd_loss_weight_)
        self.enable_transvip_style_loss_aggregation = bool(
            enable_transvip_style_loss_aggregation_
        )
        self.transvip_style_speech_loss_weight = float(
            transvip_style_speech_loss_weight_
        )
        self.transvip_style_text_loss_weight = float(
            transvip_style_text_loss_weight_
        )
        self.transvip_style_kd_loss_weight = float(
            transvip_style_kd_loss_weight_
        )
        self.transvip_style_depth_loss_weight = float(
            transvip_style_depth_loss_weight_
        )
        self.use_unified_code0_loss_in_objective = bool(use_unified_code0_loss_in_objective_)
        self.enable_text_codec_text_path_loss = bool(enable_text_codec_text_path_loss_)
        self.text_codec_text_path_input = str(text_codec_text_path_input_ or "target").strip().lower()
        if self.text_codec_text_path_input not in {"target", "source"}:
            raise ValueError(
                "text_codec_text_path_input must be one of: 'target', 'source'."
            )
        self.text_codec_t2t_loss_weight = float(text_codec_t2t_loss_weight_)
        self.text_codec_t2c_loss_weight = float(text_codec_t2c_loss_weight_)
        self.text_path_source_token_dropout = float(text_path_source_token_dropout_)
        if not 0.0 <= self.text_path_source_token_dropout < 1.0:
            raise ValueError("text_path_source_token_dropout must be in [0, 1).")
        self.enable_direct_text_to_c0_auxiliary = bool(enable_direct_text_to_c0_auxiliary_)
        self.direct_text_to_c0_input = str(direct_text_to_c0_input_ or "target").strip().lower()
        if self.direct_text_to_c0_input not in {"target", "source"}:
            raise ValueError(
                "direct_text_to_c0_input must be one of: 'target', 'source'."
            )
        self.direct_text_to_c0_loss_weight = float(direct_text_to_c0_loss_weight_)
        self.direct_text_to_c0_zero_speech_memory = bool(
            direct_text_to_c0_zero_speech_memory_
        )
        self.direct_text_to_c0_kd_weight = float(direct_text_to_c0_kd_weight_)
        self.direct_text_to_c0_kd_temperature = max(
            1.0e-6,
            float(direct_text_to_c0_kd_temperature_),
        )
        self.text_vocab_size = text_vocab_size_
        # Generation should only emit ids the external tokenizer can decode.
        # UnitY's native NLLB vocab can be a couple rows larger than HF's tokenizer.
        self.text_generation_vocab_size = (
            int(text_vocab_size_) if text_vocab_size_ is not None else None
        )
        self.text_pad_token_id = int(text_pad_token_id_)
        self.text_bos_token_id = (
            int(text_bos_token_id_) if text_bos_token_id_ is not None else None
        )
        self.text_sep_token_id = (
            int(text_sep_token_id_) if text_sep_token_id_ is not None else None
        )
        self.text_prefix_extra_token_ids_after_bos = [
            int(token_id) for token_id in (text_prefix_extra_token_ids_after_bos_ or [])
        ]
        self.text_prefix_beam_size = max(1, int(text_prefix_beam_size_))
        self.text_prefix_min_tokens = max(0, int(text_prefix_min_tokens_))
        self.text_prefix_sep_penalty = float(text_prefix_sep_penalty_)
        self.text_prefix_length_penalty = max(1e-6, float(text_prefix_length_penalty_))
        self.text_prefix_no_repeat_ngram_size = max(0, int(text_prefix_no_repeat_ngram_size_))
        if self.enable_quality_cot:
            if not (self.enable_text_prefix_ar and self.enable_text_codec_ar):
                raise ValueError(
                    "enable_quality_cot requires enable_text_prefix_ar=True and "
                    "enable_text_codec_ar=True."
                )
            if self.use_unity_native_decoder_temporal:
                raise ValueError(
                    "enable_quality_cot currently supports the Direct temporal decoder only; "
                    "use temporal_decoder_backend='direct'."
                )
            if not self.quality_cot_source_text_prefix_extra_token_ids_after_bos:
                raise ValueError(
                    "enable_quality_cot requires source language control ids via "
                    "quality_cot_source_text_prefix_extra_token_ids_after_bos."
                )
            if not self.text_prefix_extra_token_ids_after_bos:
                raise ValueError(
                    "enable_quality_cot requires target language control ids via "
                    "text_prefix_extra_token_ids_after_bos."
                )
            if self.text_vocab_size is None or int(self.text_vocab_size) <= 0:
                raise ValueError(
                    "enable_quality_cot requires a positive text_vocab_size."
                )
            if self.enable_uniss_content_controls:
                if self.uniss_start_content_token_id is None or self.uniss_end_content_token_id is None:
                    raise ValueError(
                        "UniSS content controls require resolved START/END token ids."
                    )

            text_vocab_size = int(self.text_vocab_size)
            structural_token_ids = {
                "text_pad_token_id": self.text_pad_token_id,
                "text_bos_token_id": self.text_bos_token_id,
                "text_sep_token_id": self.text_sep_token_id,
            }
            for name, token_id in structural_token_ids.items():
                if token_id is None or not 0 <= int(token_id) < text_vocab_size:
                    raise ValueError(
                        f"Quality-CoT {name}={token_id} is outside text vocab "
                        f"[0, {text_vocab_size - 1}]."
                    )
            if len(set(structural_token_ids.values())) != len(structural_token_ids):
                raise ValueError(
                    "Quality-CoT requires distinct PAD, target-start, and SEP token ids."
                )

            reserved_ids = {int(token_id) for token_id in structural_token_ids.values()}
            language_control_groups = {
                "source": self.quality_cot_source_text_prefix_extra_token_ids_after_bos,
                "target": self.text_prefix_extra_token_ids_after_bos,
            }
            for group_name, token_ids in language_control_groups.items():
                for token_id in token_ids:
                    token_id = int(token_id)
                    if not 0 <= token_id < text_vocab_size:
                        raise ValueError(
                            f"Quality-CoT {group_name} language control id={token_id} "
                            f"is outside text vocab [0, {text_vocab_size - 1}]."
                        )
                    if token_id in reserved_ids:
                        raise ValueError(
                            f"Quality-CoT {group_name} language control id={token_id} "
                            "collides with a structural token id."
                        )
        self.tie_text_embeddings = bool(tie_text_embeddings_)
        self.text_max_positions = int(text_max_positions_)
        self.dataset_num_quantizers = input_num_quantizers_
        self.input_num_quantizers = num_codebook_levels_
        self.num_codebook_levels = num_codebook_levels_
        self.num_temporal_codebook_levels = max(
            1,
            min(int(num_temporal_codebook_levels_), int(num_codebook_levels_)),
        )
        self.temporal_context_levels = (
            num_codebook_levels_
            if temporal_context_levels_ is None
            else int(temporal_context_levels_)
        )
        self.codebook_size = codebook_size_
        self.vocab_size = vocab_size_
        self.speech_encoder_type = speech_encoder_type_
        self.speech_encoder_name = speech_encoder_name_
        self.whisper_name = speech_encoder_name_
        self.qwen_name = qwen_name
        self.temporal_max_positions = temporal_max_positions_
        self.speech_encoder_output_layer = speech_encoder_output_layer_
        self.load_text_path_encoder = bool(load_text_path_encoder_)
        if (
            self.enable_mimi_source_speaker_prompt
            and self.source_speaker_prompt_backend == "w2vbert_hidden"
            and self.speech_encoder_type != "w2vbert2"
        ):
            raise ValueError(
                "source_speaker_prompt_backend='w2vbert_hidden' requires "
                "speech_encoder_type='w2vbert2'."
            )
        if self.use_unity_native_decoder_temporal and self.speech_encoder_type != "transvip_unity":
            raise ValueError(
                "Native UnitY temporal decoder backends require "
                "speech_encoder_type='transvip_unity'."
            )

        if self.num_codebook_levels <= 0:
            raise ValueError("num_codebook_levels must be >= 1.")
        if self.num_codebook_levels > self.dataset_num_quantizers:
            raise ValueError(
                f"num_codebook_levels={self.num_codebook_levels} cannot exceed "
                f"input_num_quantizers={self.dataset_num_quantizers}."
            )
        if self.temporal_context_levels <= 0:
            raise ValueError("temporal_context_levels must be >= 1.")
        if self.temporal_context_levels > self.input_num_quantizers:
            raise ValueError(
                f"temporal_context_levels={self.temporal_context_levels} cannot exceed "
                f"input_num_quantizers={self.input_num_quantizers}."
            )
        if self.codebook_loss_weights is not None:
            if len(self.codebook_loss_weights) != self.num_codebook_levels:
                raise ValueError(
                    f"codebook_loss_weights length={len(self.codebook_loss_weights)} must match "
                    f"num_codebook_levels={self.num_codebook_levels}."
                )
            if sum(self.codebook_loss_weights) <= 0.0:
                raise ValueError("codebook_loss_weights must contain at least one positive weight.")
        if self.depth_objective_start_level >= self.num_codebook_levels:
            raise ValueError(
                "depth_objective_start_level must be smaller than "
                f"num_codebook_levels={self.num_codebook_levels}."
            )

        # ========== Speech Encoder ==========
        self.transvip_unity = None
        self.__dict__["_text_path_unity"] = None
        if speech_encoder_type_ != "w2vbert2":
            raise ValueError(
                "Only the w2v-BERT 2.0 speech encoder is supported; got speech_encoder_type="
                f"{speech_encoder_type_!r}."
            )
        self.whisper = Wav2Vec2BertModel.from_pretrained(speech_encoder_name_)
        speech_dim = self.whisper.config.hidden_size

        if temporal_hidden_size_ is None:
            qwen_cfg = AutoConfig.from_pretrained(qwen_name)
            qwen_dim = getattr(qwen_cfg, "hidden_size", None)
            if qwen_dim is None:
                qwen_dim = getattr(qwen_cfg, "d_model", None)
            if qwen_dim is None:
                raise ValueError(
                    f"Could not infer hidden size from model config {qwen_name}."
                )
            temporal_dim = int(qwen_dim)
        else:
            # A fully specified Direct decoder does not need to instantiate or
            # inspect any language-model body.
            temporal_dim = int(temporal_hidden_size_)
        self.temporal_hidden_size = temporal_dim

        # ========== Temporal Decoder ==========
        # Adjust heads if needed
        if temporal_dim % temporal_num_heads != 0:
            if not temporal_auto_adjust_heads_:
                raise ValueError(
                    f"temporal_hidden_size={temporal_dim} must be divisible by "
                    f"temporal_num_heads={temporal_num_heads}."
                )
            valid_heads = [h for h in range(temporal_num_heads, 0, -1) if temporal_dim % h == 0]
            if not valid_heads:
                raise ValueError(
                    f"Could not find a valid num_heads dividing temporal_hidden_size={temporal_dim}."
                )
            temporal_num_heads = valid_heads[0]
            warnings.warn(
                f"Adjusted temporal_num_heads to {temporal_num_heads} to match "
                f"temporal_hidden_size={temporal_dim}.",
                stacklevel=2,
            )

        temporal_ffn = temporal_ffn_dim_ if temporal_ffn_dim_ is not None else 4 * temporal_dim

        # Encoder to temporal projection. The native UnitY decoder consumes raw
        # UnitY encoder states, so the DirectS2ST projection is not part of that
        # backend's temporal path.
        if self.use_unity_native_decoder_temporal:
            if int(speech_dim) != int(temporal_dim):
                raise ValueError(
                    "unity_text_decoder requires temporal_hidden_size to match "
                    f"UnitY encoder dim ({speech_dim}); got {temporal_dim}."
                )
            self.encoder_to_temporal = nn.Identity()
        else:
            self.encoder_to_temporal = nn.Linear(speech_dim, temporal_dim)
        if self.source_adapter_num_layers > 0:
            if temporal_dim % self.source_adapter_num_heads != 0:
                raise ValueError(
                    f"temporal_hidden_size={temporal_dim} must be divisible by "
                    f"source_adapter_num_heads={self.source_adapter_num_heads}."
                )
            source_adapter_ffn = (
                int(self.source_adapter_ffn_dim)
                if self.source_adapter_ffn_dim is not None
                else 4 * temporal_dim
            )
            source_adapter_layer = nn.TransformerEncoderLayer(
                d_model=temporal_dim,
                nhead=self.source_adapter_num_heads,
                dim_feedforward=source_adapter_ffn,
                dropout=self.source_adapter_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.source_adapter = nn.TransformerEncoder(
                source_adapter_layer,
                num_layers=self.source_adapter_num_layers,
            )
            self.source_adapter_norm = nn.LayerNorm(temporal_dim)
        else:
            self.source_adapter = None
            self.source_adapter_norm = None
        self.source_unit_head = (
            nn.Linear(temporal_dim, self.source_unit_vocab_size + 1)
            if self.enable_source_unit_auxiliary
            else None
        )
        if self.enable_source_unit_transformer_auxiliary:
            if temporal_dim % self.source_unit_transformer_num_heads != 0:
                raise ValueError(
                    f"temporal_hidden_size={temporal_dim} must be divisible by "
                    "source_unit_transformer_num_heads="
                    f"{self.source_unit_transformer_num_heads}."
                )
            source_ffn = (
                int(self.source_unit_transformer_ffn_dim)
                if self.source_unit_transformer_ffn_dim is not None
                else 4 * temporal_dim
            )
            self.source_unit_transformer_emb = nn.Embedding(
                self.source_unit_transformer_vocab_size,
                temporal_dim,
                padding_idx=self.source_unit_pad_id,
            )
            self.source_unit_transformer_pos_emb = nn.Embedding(
                self.source_unit_transformer_max_positions,
                temporal_dim,
            )
            source_layer = nn.TransformerDecoderLayer(
                d_model=temporal_dim,
                nhead=self.source_unit_transformer_num_heads,
                dim_feedforward=source_ffn,
                dropout=self.source_unit_transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.source_unit_transformer_decoder = nn.TransformerDecoder(
                source_layer,
                num_layers=self.source_unit_transformer_num_layers,
            )
            self.source_unit_transformer_norm = nn.LayerNorm(temporal_dim)
            self.source_unit_transformer_head = nn.Linear(
                temporal_dim,
                self.source_unit_transformer_vocab_size,
            )
        else:
            self.source_unit_transformer_emb = None
            self.source_unit_transformer_pos_emb = None
            self.source_unit_transformer_decoder = None
            self.source_unit_transformer_norm = None
            self.source_unit_transformer_head = None

        self.unity_source_memory_proj = None
        self.unity_source_memory_norm = None
        if self.enable_unity_source_memory_fusion:
            if not self.use_unity_native_decoder_temporal:
                raise ValueError(
                    "enable_unity_source_memory_fusion=True requires "
                    "a native UnitY temporal decoder backend."
                )
            self.unity_source_memory_proj = (
                nn.Identity()
                if int(speech_dim) == int(temporal_dim)
                else nn.Linear(int(speech_dim), int(temporal_dim))
            )
            self.unity_source_memory_norm = nn.LayerNorm(int(temporal_dim))
            print(
                "[Init:UnitySourceMemoryFusion] enabled",
                {
                    "mode": self.unity_source_memory_fusion_mode,
                    "source": self.unity_source_memory_source,
                    "layer": self.unity_source_memory_layer,
                    "scale": self.unity_source_memory_scale,
                    "detach": self.unity_source_memory_detach,
                },
            )

        if self.use_unity_native_decoder_temporal:
            # The native UnitY text decoder is the temporal AR decoder. Keep only a
            # tiny frozen position table for helpers that inspect embedding_dim/dtype.
            self.temporal_codebook_emb = nn.ModuleList()
            self.temporal_pos_emb = nn.Embedding(1, temporal_dim)
            self.temporal_pos_emb.requires_grad_(False)
            self.temporal_transformer = None
            self.temporal_norm = None
        else:
            # Temporal decoder input embeddings for modeled codebook levels only.
            # The vocab includes BOS/EOS tokens.
            self.temporal_codebook_emb = nn.ModuleList(
                [nn.Embedding(vocab_size_, temporal_dim) for _ in range(self.input_num_quantizers)]
            )
            self.temporal_pos_emb = nn.Embedding(temporal_max_positions_, temporal_dim)

            # Temporal decoder (cross-attends to Whisper encoder output)
            temporal_layer = nn.TransformerDecoderLayer(
                d_model=temporal_dim,
                nhead=temporal_num_heads,
                dim_feedforward=temporal_ffn,
                dropout=temporal_dropout_,
                activation=self.temporal_activation,
                batch_first=True,
                norm_first=True,
            )
            self.temporal_transformer = nn.TransformerDecoder(temporal_layer, num_layers=temporal_num_layers)
            self.temporal_norm = nn.LayerNorm(temporal_dim)

        # ========== Target-Text Embeddings ==========
        # Text-prefix AR uses these embeddings inside the *same* temporal decoder
        # as codec C0, matching TransVIP's text+separator+codec conditioning.
        if self.use_unity_native_decoder_temporal:
            if text_vocab_size_ is None or int(text_vocab_size_) <= 0:
                raise ValueError("text_vocab_size must be set for unity_text_decoder text-prefix AR.")
            try:
                self.text_codec_vocab_size = int(self.transvip_unity.final_proj.weight.size(0))
            except Exception:
                self.text_codec_vocab_size = int(text_vocab_size_) + int(vocab_size_)
            if self.text_codec_vocab_size < int(text_vocab_size_) + int(vocab_size_):
                raise ValueError(
                    "UnitY final projection is too small for text+codec AR: "
                    f"final_proj={self.text_codec_vocab_size}, "
                    f"text_vocab={int(text_vocab_size_)}, codec_vocab={int(vocab_size_)}."
                )
            self.text_codec_lm_head = None
            self.text_token_emb = None
            self.text_lm_head = None
        elif (
            self.enable_text_prefix_ar
            or self.enable_direct_text_to_c0_auxiliary
        ):
            if text_vocab_size_ is None or int(text_vocab_size_) <= 0:
                raise ValueError(
                    "text_vocab_size must be set when text-prefix AR or the text-to-C0 objective is enabled."
                )
            self.text_token_emb = nn.Embedding(
                int(text_vocab_size_),
                temporal_dim,
                padding_idx=self.text_pad_token_id,
            )
            self.text_lm_head = nn.Linear(
                temporal_dim,
                int(text_vocab_size_),
                bias=not self.tie_text_embeddings,
            )
            if self.tie_text_embeddings:
                self.text_lm_head.weight = self.text_token_emb.weight
            if self.enable_text_codec_ar:
                self.text_codec_vocab_size = int(text_vocab_size_) + int(vocab_size_)
                self.text_codec_lm_head = nn.Linear(temporal_dim, self.text_codec_vocab_size)
            else:
                self.text_codec_vocab_size = None
                self.text_codec_lm_head = None
        else:
            self.text_codec_vocab_size = None
            self.text_codec_lm_head = None
            self.text_token_emb = None
            self.text_lm_head = None

        if self.enable_quality_cot:
            expected_text_codec_vocab_size = int(self.text_vocab_size) + int(vocab_size_)
            if self.text_codec_vocab_size != expected_text_codec_vocab_size:
                raise ValueError(
                    "Quality-CoT requires a contiguous text+codec head: "
                    f"got text_codec_vocab_size={self.text_codec_vocab_size}, "
                    f"expected {expected_text_codec_vocab_size} "
                    f"(text={self.text_vocab_size}, codec={vocab_size_})."
                )
            print(
                "[Init:QualityCoTVocab] "
                f"text=[0,{int(self.text_vocab_size) - 1}] "
                f"codec=[{int(self.text_vocab_size)},{expected_text_codec_vocab_size - 1}] "
                f"controls={{pad:{self.text_pad_token_id},start:{self.text_bos_token_id},"
                f"sep:{self.text_sep_token_id},"
                f"src_lang:{self.quality_cot_source_text_prefix_extra_token_ids_after_bos},"
                f"tgt_lang:{self.text_prefix_extra_token_ids_after_bos}}}"
            )

        # ========== Depth Decoder ==========
        depth_dim = depth_hidden_size_ if depth_hidden_size_ is not None else temporal_dim
        self.depth_hidden_size = depth_dim

        # Mimi is needed by the global prompt, the learned Mimi-sequence
        # prompt, and the source-codec NAR path. A W2v-BERT prompt alone does
        # not introduce an otherwise unnecessary Mimi model.
        needs_mimi_source_features = (
            self.enable_mimi_source_acoustic_conditioning
            or (
                self.enable_mimi_source_speaker_prompt
                and self.source_speaker_prompt_backend in {
                    "mimi_mean_std",
                    "mimi_latent",
                }
            )
            or self.enable_mimi_source_acoustic_temporal_conditioning
            or self.enable_mimi_nar_depth_prompt
        )
        if needs_mimi_source_features:
            try:
                feature_extractor = AutoFeatureExtractor.from_pretrained(
                    self.mimi_source_acoustic_model_name
                )
                self.mimi_source_acoustic_model_sample_rate = int(
                    getattr(
                        feature_extractor,
                        "sampling_rate",
                        self.mimi_source_acoustic_input_sample_rate,
                    )
                )
            except Exception as exc:
                warnings.warn(
                    "Could not load Mimi source acoustic feature extractor; "
                    "falling back to input sample rate. Details: "
                    f"{exc}",
                    stacklevel=2,
                )
            source_acoustic_base_hidden_size: Optional[int] = None
            try:
                source_acoustic_config = AutoConfig.from_pretrained(
                    self.mimi_source_acoustic_model_name
                )
                source_acoustic_base_hidden_size = int(
                    getattr(source_acoustic_config, "hidden_size")
                )
            except Exception as exc:
                if (
                    self.use_precomputed_source_acoustic_embeddings
                    and self.skip_mimi_source_acoustic_encoder_when_precomputed
                ):
                    raise RuntimeError(
                        "skip_mimi_source_acoustic_encoder_when_precomputed=True requires "
                        "reading Mimi config hidden_size without loading the encoder."
                    ) from exc
            skip_source_acoustic_model_load = (
                self.use_precomputed_source_acoustic_embeddings
                and self.skip_mimi_source_acoustic_encoder_when_precomputed
            )
            if not skip_source_acoustic_model_load:
                self.mimi_source_acoustic_model = MimiModel.from_pretrained(
                    self.mimi_source_acoustic_model_name
                )
                self.mimi_source_acoustic_model.eval()
                for param in self.mimi_source_acoustic_model.parameters():
                    param.requires_grad = False
                if source_acoustic_base_hidden_size is None:
                    source_acoustic_base_hidden_size = int(
                        getattr(self.mimi_source_acoustic_model.config, "hidden_size")
                    )
            if source_acoustic_base_hidden_size is None:
                raise RuntimeError("Could not resolve Mimi source acoustic hidden size.")
            pooling_multiplier = 2 if self.mimi_source_acoustic_pooling == "mean_std" else 1
            self.mimi_source_acoustic_base_hidden_size = int(
                source_acoustic_base_hidden_size
            )
            self.mimi_source_acoustic_hidden_size = int(
                source_acoustic_base_hidden_size * pooling_multiplier
            )
            print(
                "[Init:MimiSourceAcoustic] enabled",
                {
                    "model": self.mimi_source_acoustic_model_name,
                    "input_sample_rate": self.mimi_source_acoustic_input_sample_rate,
                    "model_sample_rate": self.mimi_source_acoustic_model_sample_rate,
                    "base_hidden_size": int(source_acoustic_base_hidden_size),
                    "pooling": self.mimi_source_acoustic_pooling,
                    "hidden_size": self.mimi_source_acoustic_hidden_size,
                    "precomputed": self.use_precomputed_source_acoustic_embeddings,
                    "skip_encoder_when_precomputed": self.skip_mimi_source_acoustic_encoder_when_precomputed,
                    "residual_scale": self.mimi_source_acoustic_residual_scale,
                    "depth_start_level": self.mimi_source_acoustic_depth_start_level,
                    "detach": self.mimi_source_acoustic_detach,
                },
            )

        if self.enable_campplus_depth_conditioning:
            self._initialize_campplus_source_model()

        self.mimi_source_speaker_prompt_proj = None
        self.mimi_source_speaker_prompt_norm = None
        self.source_speaker_prompt_frame_proj = None
        self.source_speaker_prompt_input_norm = None
        self.source_speaker_prompt_encoder = None
        self.source_speaker_prompt_output_norm = None
        if self.enable_mimi_source_speaker_prompt:
            if not self.enable_text_prefix_ar:
                raise ValueError(
                    "enable_mimi_source_speaker_prompt=True requires "
                    "enable_text_prefix_ar=True."
                )
            if self.source_speaker_prompt_backend == "mimi_mean_std":
                if self.mimi_source_acoustic_hidden_size is None:
                    raise RuntimeError("Mimi source acoustic hidden size was not resolved.")
                self.mimi_source_speaker_prompt_proj = (
                    nn.Identity()
                    if int(self.mimi_source_acoustic_hidden_size) == int(temporal_dim)
                    else nn.Linear(int(self.mimi_source_acoustic_hidden_size), int(temporal_dim))
                )
                self.mimi_source_speaker_prompt_norm = nn.LayerNorm(int(temporal_dim))
                print(
                    "[Init:SourceSpeakerPrompt] enabled",
                    {
                        "backend": self.source_speaker_prompt_backend,
                        "hidden_size": int(self.mimi_source_acoustic_hidden_size),
                        "speaker_dim": int(temporal_dim),
                        "detach": self.mimi_source_speaker_prompt_detach,
                        "sep_token_id": self.text_sep_token_id,
                    },
                )
            else:
                if self.source_speaker_prompt_backend == "w2vbert_hidden":
                    prompt_input_dim = int(speech_dim)
                else:
                    if self.mimi_source_acoustic_base_hidden_size is None:
                        raise RuntimeError("Mimi source acoustic frame size was not resolved.")
                    prompt_input_dim = int(self.mimi_source_acoustic_base_hidden_size)
                if int(temporal_dim) % int(self.source_speaker_prompt_num_heads) != 0:
                    raise ValueError(
                        f"temporal_hidden_size={temporal_dim} must be divisible by "
                        "source_speaker_prompt_num_heads="
                        f"{self.source_speaker_prompt_num_heads}."
                    )
                prompt_ffn_dim = (
                    4 * int(temporal_dim)
                    if self.source_speaker_prompt_ffn_dim is None
                    else int(self.source_speaker_prompt_ffn_dim)
                )
                self.source_speaker_prompt_frame_proj = (
                    nn.Identity()
                    if prompt_input_dim == int(temporal_dim)
                    else nn.Linear(prompt_input_dim, int(temporal_dim))
                )
                self.source_speaker_prompt_input_norm = nn.LayerNorm(int(temporal_dim))
                prompt_layer = nn.TransformerEncoderLayer(
                    d_model=int(temporal_dim),
                    nhead=int(self.source_speaker_prompt_num_heads),
                    dim_feedforward=prompt_ffn_dim,
                    dropout=float(self.source_speaker_prompt_dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.source_speaker_prompt_encoder = nn.TransformerEncoder(
                    prompt_layer,
                    num_layers=int(self.source_speaker_prompt_num_layers),
                )
                self.source_speaker_prompt_output_norm = nn.LayerNorm(int(temporal_dim))
                print(
                    "[Init:SourceSpeakerPrompt] enabled",
                    {
                        "backend": self.source_speaker_prompt_backend,
                        "input_hidden_size": prompt_input_dim,
                        "speaker_dim": int(temporal_dim),
                        "layers": int(self.source_speaker_prompt_num_layers),
                        "heads": int(self.source_speaker_prompt_num_heads),
                        "ffn_dim": prompt_ffn_dim,
                        "dropout": float(self.source_speaker_prompt_dropout),
                        "speech_layer": (
                            int(self.source_speaker_prompt_speech_layer)
                            if self.source_speaker_prompt_backend == "w2vbert_hidden"
                            else None
                        ),
                        "pooling": "masked_mean",
                        "detach": self.mimi_source_speaker_prompt_detach,
                        "sep_token_id": self.text_sep_token_id,
                    },
                )

        self.source_acoustic_temporal_proj = None
        self.source_acoustic_temporal_norm = None
        if self.enable_mimi_source_acoustic_temporal_conditioning:
            if not self.enable_mimi_source_acoustic_conditioning:
                raise ValueError(
                    "enable_mimi_source_acoustic_temporal_conditioning=True requires "
                    "enable_mimi_source_acoustic_conditioning=True so the source acoustic "
                    "embedding can be loaded or computed."
                )
            if self.mimi_source_acoustic_hidden_size is None:
                raise RuntimeError("Mimi source acoustic hidden size was not resolved.")
            self.source_acoustic_temporal_proj = (
                nn.Identity()
                if int(self.mimi_source_acoustic_hidden_size) == int(temporal_dim)
                else nn.Linear(int(self.mimi_source_acoustic_hidden_size), int(temporal_dim))
            )
            self.source_acoustic_temporal_norm = nn.LayerNorm(int(temporal_dim))
            print(
                "[Init:MimiSourceAcousticTemporal] enabled",
                {
                    "hidden_size": int(self.mimi_source_acoustic_hidden_size),
                    "temporal_dim": int(temporal_dim),
                    "residual_scale": self.mimi_source_acoustic_temporal_residual_scale,
                    "detach": self.mimi_source_acoustic_temporal_detach,
                },
            )

        self.c0_auxiliary_head = None
        if self.enable_c0_auxiliary_loss:
            self.c0_auxiliary_head = nn.Sequential(
                nn.LayerNorm(int(temporal_dim)),
                nn.Linear(int(temporal_dim), int(vocab_size_)),
            )
            print(
                "[Init:C0Auxiliary] enabled",
                {"loss_weight": self.c0_auxiliary_loss_weight, "vocab_size": int(vocab_size_)},
            )

        if self.c0_teacher_hidden_dim is None:
            self.c0_teacher_hidden_dim = int(temporal_dim)
        self.c0_teacher_hidden_projection = None
        if self.enable_c0_teacher_hidden_distill:
            if self.c0_teacher_hidden_use_projection or int(self.c0_teacher_hidden_dim) != int(temporal_dim):
                self.c0_teacher_hidden_projection = nn.Sequential(
                    nn.LayerNorm(int(temporal_dim)),
                    nn.Linear(int(temporal_dim), int(self.c0_teacher_hidden_dim)),
                )
            else:
                self.c0_teacher_hidden_projection = nn.Identity()
            print(
                "[Init:C0TeacherHiddenDistill] enabled",
                {
                    "weight": self.c0_teacher_hidden_distill_weight,
                    "loss_type": self.c0_teacher_hidden_loss_type,
                    "teacher_hidden_dim": int(self.c0_teacher_hidden_dim),
                    "use_projection": self.c0_teacher_hidden_use_projection,
                },
            )

        # Projection from temporal to depth dimension (if different)
        self.temporal_to_depth = (
            nn.Identity() if depth_dim == temporal_dim else nn.Linear(temporal_dim, depth_dim)
        )

        # TemporalDepthDecoder handles level-0 prediction + depth decoding.
        self.depth_decoder = TemporalDepthDecoder(
            temporal_hidden_size=temporal_dim,
            depth_hidden_size=depth_dim,
            vocab_size=vocab_size_,
            num_quantizers=num_codebook_levels_,
            depth_num_layers=depth_num_layers_,
            depth_num_heads=depth_num_heads_,
            depth_ffn_dim=depth_ffn_dim_,
            depth_dropout=depth_dropout_,
            depth_auto_adjust_heads=depth_auto_adjust_heads_,
            num_temporal_codebook_levels=self.num_temporal_codebook_levels,
            source_hidden_size=temporal_dim,
            enable_source_conditioning=self.enable_depth_source_conditioning,
            source_conditioning_num_heads=self.depth_source_conditioning_num_heads,
            source_conditioning_dropout=self.depth_source_conditioning_dropout,
            source_conditioning_residual_scale=self.depth_source_conditioning_residual_scale,
            source_conditioning_detach=self.depth_source_conditioning_detach,
            enable_source_acoustic_conditioning=self.enable_mimi_source_acoustic_conditioning,
            source_acoustic_hidden_size=self.mimi_source_acoustic_hidden_size,
            source_acoustic_residual_scale=self.mimi_source_acoustic_residual_scale,
            source_acoustic_depth_start_level=self.mimi_source_acoustic_depth_start_level,
            source_acoustic_detach=self.mimi_source_acoustic_detach,
            enable_source_style_token_conditioning=self.enable_mimi_source_style_token_bank,
            source_style_token_count=self.mimi_source_style_token_count,
            source_style_token_num_heads=self.mimi_source_style_token_num_heads,
            source_style_token_dropout=self.mimi_source_style_token_dropout,
            source_style_token_residual_scale=self.mimi_source_style_token_residual_scale,
            source_style_token_depth_start_level=self.mimi_source_style_token_depth_start_level,
            source_style_token_gate_init=self.mimi_source_style_token_gate_init,
            source_style_token_c1_gate_init=self.mimi_source_style_token_c1_gate_init,
            enable_nar_source_prompt=self.enable_mimi_nar_depth_prompt,
            nar_attention_mode=self.mimi_nar_depth_attention_mode,
            nar_max_positions=self.mimi_nar_depth_max_positions,
            nar_teacher_force_level_batch_size=(
                self.mimi_nar_depth_teacher_force_level_batch_size
            ),
            enable_campplus_depth_conditioning=self.enable_campplus_depth_conditioning,
            campplus_depth_conditioning_mode=self.campplus_depth_conditioning_mode,
            campplus_embedding_size=self.campplus_embedding_size,
            campplus_film_scale=self.campplus_film_scale,
            campplus_depth_start_level=self.campplus_depth_start_level,
            campplus_depth_identity_loss_weight=(
                self.campplus_depth_identity_loss_weight
            ),
            campplus_identity_level_aggregation=(
                self.campplus_identity_level_aggregation
            ),
        )
        if self.enable_mimi_nar_depth_prompt:
            print(
                "[Init:MimiSourceCodecDepth] enabled",
                {
                    "attention_mode": self.mimi_nar_depth_attention_mode,
                    "input_layout": "source_prefix_target_lower_codebooks",
                    "max_positions": self.mimi_nar_depth_max_positions,
                    "teacher_force_level_batch_size": (
                        self.mimi_nar_depth_teacher_force_level_batch_size
                    ),
                },
            )
        if self.enable_campplus_depth_conditioning:
            print(
                "[Init:CAMPPlusDepth] enabled",
                {
                    "mode": self.campplus_depth_conditioning_mode,
                    "embedding_size": self.campplus_embedding_size,
                    "conditioning": self.campplus_depth_conditioning_mode,
                    "execution": "native_transformer_container",
                    "film_scale": self.campplus_film_scale,
                    "depth_start_level": self.campplus_depth_start_level,
                    "identity_loss_weight": self.campplus_depth_identity_loss_weight,
                    "identity_supervision": self.campplus_identity_supervision,
                    "identity_level_aggregation": (
                        self.campplus_identity_level_aggregation
                    ),
                },
            )
        if self.enable_mimi_source_style_token_bank:
            print(
                "[Init:MimiSourceStyleTokens] enabled",
                {
                    "token_count": self.mimi_source_style_token_count,
                    "depth_start_level": self.mimi_source_style_token_depth_start_level,
                    "num_heads": self.mimi_source_style_token_num_heads,
                    "dropout": self.mimi_source_style_token_dropout,
                    "residual_scale": self.mimi_source_style_token_residual_scale,
                    "gate_init": self.mimi_source_style_token_gate_init,
                    "c1_gate_init": self.mimi_source_style_token_c1_gate_init,
                    "global_residual_scale": self.mimi_source_acoustic_residual_scale,
                },
            )

        # ========== Freezing ==========
        if freeze_whisper_encoder and self.whisper is not None:
            for p in self.whisper.parameters():
                p.requires_grad = False
        if freeze_temporal_transformer:
            for _module in (
                self.temporal_transformer,
                self.temporal_codebook_emb,
                self.temporal_pos_emb,
                self.temporal_to_depth,
            ):
                if _module is not None:
                    for p in _module.parameters():
                        p.requires_grad = False

        if self.train_only_depth_decoder:
            self._freeze_for_depth_only_training()

    def _freeze_for_depth_only_training(self):
        """Keep checkpoint topology, but optimize NAR/CAMP and never AR heads."""
        if not self.depth_decoder.enable_nar_source_prompt:
            raise ValueError("train_only_depth_decoder requires the source-codec NAR backend.")
        semantic_parameters = {
            id(parameter)
            for name, module in self.named_children() if name != "depth_decoder"
            for parameter in module.parameters()
        }
        temporal_head_prefixes = ("first_codebook_head.", "temporal_codebook_heads.")
        trainable = [
            (name, parameter) for name, parameter in self.depth_decoder.named_parameters()
            if not name.startswith(temporal_head_prefixes)
        ]
        if any(id(parameter) in semantic_parameters for _, parameter in trainable):
            raise ValueError("NAR parameters are shared with AR; refusing to unfreeze the semantic path.")
        if not trainable:
            raise ValueError("No NAR parameters found for depth-only training.")
        self.requires_grad_(False)
        for _, parameter in trainable:
            parameter.requires_grad_(True)
        self.train(self.training)
        print("[DepthOnlyTraining] Frozen all non-depth modules and temporal codebook heads; "
              "training full NAR/C1-C15/CAMP:", {
                  "trainable_params": sum(p.numel() for _, p in trainable),
                  "trainable_preview": [name for name, _ in trainable[:12]],
                  "frozen_semantic_dropout": True,
              })

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "train_only_depth_decoder", False):
            # Trainer calls train() repeatedly. Frozen feature providers must
            # stay deterministic, while the trainable NAR keeps its dropout.
            for name, module in self.named_children():
                if name != "depth_decoder":
                    module.eval()
            self.depth_decoder.first_codebook_head.eval()
            self.depth_decoder.temporal_codebook_heads.eval()
        return self

    def _scheduled_sampling_ramp(
        self,
        target_prob: float,
        start_step: int,
        warmup_steps: int,
    ) -> float:
        if not self.training or target_prob <= 0.0:
            return 0.0
        step = int(getattr(self, "_current_global_step", 0) or 0)
        if step < start_step:
            return 0.0
        if warmup_steps <= 0:
            return float(target_prob)
        progress = (step - start_step) / float(warmup_steps)
        return float(target_prob) * max(0.0, min(1.0, progress))

    def _mixed_source_auxiliary_scale(self) -> float:
        """Ramp the predicted-source auxiliary without consuming global RNG state."""
        step = int(getattr(self, "_current_global_step", 0) or 0)
        if step < self.mixed_source_start_step:
            return 0.0
        if self.mixed_source_warmup_steps <= 0:
            return 1.0
        progress = (step - self.mixed_source_start_step) / float(
            self.mixed_source_warmup_steps
        )
        return max(0.0, min(1.0, progress))

    def _predicted_source_target_auxiliary_scale(self) -> float:
        """Linearly introduce the free-running-source translation objective."""
        if not self.training:
            return 0.0
        step = int(getattr(self, "_current_global_step", 0) or 0)
        if step < self.predicted_source_target_start_step:
            return 0.0
        if self.predicted_source_target_warmup_steps <= 0:
            return 1.0
        progress = (step - self.predicted_source_target_start_step) / float(
            self.predicted_source_target_warmup_steps
        )
        return max(0.0, min(1.0, progress))

    def _build_mixed_source_text_input_ids(
        self,
        source_text_input_ids: torch.Tensor,
        source_text_attention_mask: Optional[torch.Tensor],
        unified_logits: torch.Tensor,
        replacement_prob: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replace source lexical tokens with detached model predictions.

        The replacement keeps BOS, language/START controls, and END intact, so
        sequence boundaries remain aligned with the gold target-text segment.
        """
        mixed_ids = source_text_input_ids.clone()
        device = mixed_ids.device
        if source_text_attention_mask is None:
            source_lengths = (mixed_ids != self.text_pad_token_id).sum(dim=1).long()
        else:
            source_lengths = source_text_attention_mask.to(device=device).sum(dim=1).long()

        replaced = torch.zeros((), dtype=torch.float32, device=device)
        eligible = torch.zeros((), dtype=torch.float32, device=device)
        if replacement_prob <= 0.0:
            return mixed_ids, replaced, eligible

        generator = torch.Generator(device=device)
        generator.manual_seed(
            int(self.mixed_source_auxiliary_seed)
            + 1_000_003 * int(getattr(self, "_current_global_step", 0) or 0)
        )
        lexical_start = (
            1
            + len(self.quality_cot_source_text_prefix_extra_token_ids_after_bos)
            + int(self.enable_uniss_content_controls)
        )
        forbidden_ids = {
            int(token_id)
            for token_id in (
                self.text_pad_token_id,
                self.text_bos_token_id,
                self.text_sep_token_id,
                self.uniss_start_content_token_id,
                self.uniss_end_content_token_id,
                *self.quality_cot_source_text_prefix_extra_token_ids_after_bos,
                *self.text_prefix_extra_token_ids_after_bos,
            )
            if token_id is not None
        }
        text_vocab_size = int(self.text_generation_vocab_size or self.text_vocab_size)

        for sample_idx, source_len_tensor in enumerate(source_lengths):
            source_len = int(source_len_tensor.item())
            # The final source token is END_CONTENT (or EOS without UniSS controls).
            lexical_end = source_len - 1
            if lexical_end <= lexical_start:
                continue
            token_positions = torch.arange(
                lexical_start,
                lexical_end,
                dtype=torch.long,
                device=device,
            )
            logit_positions = token_positions - 1
            within_logits = logit_positions < unified_logits.size(1)
            token_positions = token_positions[within_logits]
            logit_positions = logit_positions[within_logits]
            if token_positions.numel() == 0:
                continue

            predicted_ids = unified_logits[
                sample_idx,
                logit_positions,
                :text_vocab_size,
            ].detach().argmax(dim=-1)
            valid_prediction = torch.ones_like(predicted_ids, dtype=torch.bool)
            for token_id in forbidden_ids:
                valid_prediction &= predicted_ids != token_id
            selected = torch.rand(
                token_positions.numel(),
                generator=generator,
                device=device,
            ) < float(replacement_prob)
            selected &= valid_prediction
            selected &= predicted_ids != mixed_ids[sample_idx, token_positions]
            if bool(selected.any().item()):
                mixed_ids[sample_idx, token_positions[selected]] = predicted_ids[selected]
            replaced = replaced + selected.sum().to(torch.float32)
            eligible = eligible + float(token_positions.numel())

        return mixed_ids, replaced, eligible

    def _effective_temporal_scheduled_sampling_prob(self) -> float:
        prob = self._scheduled_sampling_ramp(
            self.temporal_scheduled_sampling_prob,
            self.temporal_scheduled_sampling_start_step,
            self.temporal_scheduled_sampling_warmup_steps,
        )
        if not self.training or prob <= 0.0:
            return prob
        step = int(getattr(self, "_current_global_step", 0) or 0)
        decay_start = int(self.temporal_scheduled_sampling_decay_start_step)
        if decay_start < 0 or step < decay_start:
            return prob
        final_prob = float(self.temporal_scheduled_sampling_final_prob)
        decay_steps = int(self.temporal_scheduled_sampling_decay_steps)
        if decay_steps <= 0:
            return final_prob
        progress = max(0.0, min(1.0, (step - decay_start) / float(decay_steps)))
        return float(prob + (final_prob - prob) * progress)

    def _sample_temporal_level_ids(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temporal_scheduled_sampling_mode == "sample":
            scaled = logits.detach() / max(1.0e-6, float(self.temporal_scheduled_sampling_temperature))
            topk = min(max(1, int(self.temporal_scheduled_sampling_topk)), scaled.size(-1))
            top_values, top_ids = torch.topk(scaled, k=topk, dim=-1)
            probs = torch.softmax(top_values, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            row_sums = probs.sum(dim=-1, keepdim=True)
            fallback = row_sums.squeeze(-1) <= 0.0
            probs = torch.where(row_sums > 0.0, probs / row_sums.clamp_min(1.0e-12), probs)
            if bool(fallback.any().item()):
                probs[fallback] = 0.0
                probs[fallback, 0] = 1.0
            sampled_pos = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1).reshape(probs.shape[:-1])
            return top_ids.gather(-1, sampled_pos.unsqueeze(-1)).squeeze(-1)
        return logits.detach().argmax(dim=-1)

    def _sample_temporal_level0_ids(self, logits: torch.Tensor) -> torch.Tensor:
        return self._sample_temporal_level_ids(logits)

    def _level0_cross_entropy_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.level0_eos_loss_weight == 1.0
            and self.level0_tail_loss_weight == 1.0
            and self.level0_tail_loss_last_n <= 0
            and self.level0_label_smoothing <= 0.0
        ):
            return F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=self.ignore_index,
            )

        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_labels = labels.reshape(-1)
        valid = flat_labels != self.ignore_index
        if int(valid.sum().detach().cpu().item()) <= 0:
            return flat_logits.new_zeros(())

        losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=(
                float(self.level0_label_smoothing)
                if self.training or not self.level0_label_smoothing_train_only
                else 0.0
            ),
        ).reshape_as(labels)

        weights = torch.ones_like(labels, dtype=logits.dtype)
        valid_2d = labels != self.ignore_index
        eos_id = self.codebook_size + 1
        if self.level0_eos_loss_weight != 1.0:
            weights = torch.where(
                valid_2d & (labels == eos_id),
                weights * float(self.level0_eos_loss_weight),
                weights,
            )
        if self.level0_tail_loss_weight != 1.0 and self.level0_tail_loss_last_n > 0:
            positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
            valid_counts = valid_2d.to(torch.long).sum(dim=1, keepdim=True)
            tail_from = (valid_counts - int(self.level0_tail_loss_last_n)).clamp_min(0)
            tail = valid_2d & (positions >= tail_from)
            weights = torch.where(
                tail,
                weights * float(self.level0_tail_loss_weight),
                weights,
            )

        weighted_valid = valid_2d.to(logits.dtype)
        weighted = losses * weights * weighted_valid
        denom = (weights * weighted_valid).sum().clamp_min(1.0)
        return weighted.sum() / denom

    def _apply_temporal_scheduled_sampling(
        self,
        input_labels: torch.Tensor,
        temporal_mask: Optional[torch.Tensor],
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        scheduled_sampling_prob: float,
    ) -> torch.Tensor:
        if (
            not self.training
            or scheduled_sampling_prob <= 0.0
            or input_labels.size(1) <= 1
            or input_labels.size(2) <= 0
        ):
            return input_labels

        with torch.no_grad():
            gold_inputs = self._build_temporal_inputs(input_labels)
            if temporal_mask is None:
                gold_mask = torch.ones(
                    (gold_inputs.size(0), gold_inputs.size(1)),
                    device=gold_inputs.device,
                    dtype=torch.long,
                )
            else:
                gold_mask = temporal_mask
            gold_hidden = self.temporal_transformer(
                tgt=gold_inputs,
                memory=encoder_hidden,
                tgt_mask=self._causal_mask(gold_inputs.size(1), gold_inputs.device),
                tgt_key_padding_mask=(gold_mask == 0),
                memory_key_padding_mask=(encoder_mask == 0) if encoder_mask is not None else None,
            )
            gold_hidden = self.temporal_norm(gold_hidden)
            sample_levels = max(
                1,
                min(
                    int(getattr(self, "temporal_context_levels", 1)),
                    int(getattr(self, "num_temporal_codebook_levels", 1)),
                    int(input_labels.size(2)),
                ),
            )
            pred_levels: List[torch.Tensor] = []
            level0_logits = self.depth_decoder.first_codebook_head(gold_hidden)
            pred_levels.append(self._sample_temporal_level_ids(level0_logits))
            extra_heads = getattr(self.depth_decoder, "temporal_codebook_heads", [])
            for level in range(1, sample_levels):
                head_idx = level - 1
                if head_idx >= len(extra_heads):
                    break
                logits = extra_heads[head_idx](gold_hidden)
                pred_levels.append(self._sample_temporal_level_ids(logits))

        mixed = input_labels.clone()
        preserve_last_n = int(getattr(self, "temporal_scheduled_sampling_preserve_last_n", 0) or 0)
        for level, pred_ids in enumerate(pred_levels):
            current_ids = mixed[:, 1:, level]
            pred_prev_ids = pred_ids[:, :-1].to(device=mixed.device)
            valid = (current_ids >= 0) & (current_ids < self.vocab_size)
            if temporal_mask is not None:
                valid = valid & (temporal_mask[:, 1:].to(device=current_ids.device) > 0)
            if preserve_last_n > 0 and bool(valid.any().item()):
                positions = torch.arange(
                    current_ids.size(1),
                    device=current_ids.device,
                ).unsqueeze(0)
                valid_counts = valid.to(torch.long).sum(dim=1, keepdim=True)
                preserve_from = (valid_counts - preserve_last_n).clamp_min(0)
                protect_tail = valid & (positions >= preserve_from)
                valid = valid & (~protect_tail)
            replace = (
                torch.rand(current_ids.shape, device=current_ids.device)
                < float(scheduled_sampling_prob)
            ) & valid
            mixed[:, 1:, level] = torch.where(replace, pred_prev_ids, current_ids)
        return mixed

    def _apply_textcodec_scheduled_sampling(
        self,
        temporal_inputs: torch.Tensor,
        temporal_mask: torch.Tensor,
        code_start_positions: List[int],
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        scheduled_sampling_prob: float,
    ) -> torch.Tensor:
        """Mix predicted semantic and C0 history into unified AR inputs.

        A no-grad teacher-forced pass supplies parallel predictions. BOS, the
        semantic SEP boundary, and terminal codec specials remain gold so the
        fixed training alignment is not corrupted.
        """
        if (
            not self.training
            or scheduled_sampling_prob <= 0.0
            or temporal_inputs.size(1) <= 1
        ):
            return temporal_inputs
        if self.temporal_context_levels != 1:
            raise ValueError(
                "Unified semantic+C0 scheduled sampling currently requires "
                "temporal_context_levels=1 so replacing C0 history cannot "
                "silently discard other temporal codebook embeddings."
            )

        with torch.no_grad():
            gold_hidden = self.temporal_transformer(
                tgt=temporal_inputs,
                memory=encoder_hidden,
                tgt_mask=self._causal_mask(temporal_inputs.size(1), temporal_inputs.device),
                tgt_key_padding_mask=(temporal_mask == 0),
                memory_key_padding_mask=(encoder_mask == 0) if encoder_mask is not None else None,
            )
            gold_hidden = self.temporal_norm(gold_hidden)
            unified_logits = self.text_codec_lm_head(gold_hidden)

        mixed = temporal_inputs.clone()
        text_vocab_size = int(self.text_vocab_size)
        codec_offset = text_vocab_size
        codec_vocab_size = int(self.vocab_size)
        codec_bos_id = int(self.codebook_size)
        codec_eos_id = codec_bos_id + 1

        for sample_idx, code_target_start in enumerate(code_start_positions):
            text_len = int(code_target_start) + 1
            input_len = int(temporal_mask[sample_idx].sum().item())

            # Replace semantic tokens only; keep BOS and the final SEP boundary.
            if text_len > 2:
                semantic_logits = unified_logits[
                    sample_idx, : text_len - 2, :text_vocab_size
                ].clone()
                semantic_logits[:, int(self.text_pad_token_id)] = -torch.inf
                semantic_logits[:, int(self.text_bos_token_id)] = -torch.inf
                semantic_logits[:, int(self.text_sep_token_id)] = -torch.inf
                semantic_ids = self._sample_temporal_level_ids(semantic_logits)
                semantic_positions = torch.arange(
                    1, text_len - 1, device=temporal_inputs.device
                )
                replace = torch.rand(
                    semantic_ids.shape, device=temporal_inputs.device
                ) < float(scheduled_sampling_prob)
                semantic_emb = self.text_token_emb(semantic_ids)
                semantic_emb = semantic_emb + self.temporal_pos_emb(semantic_positions)
                mixed[sample_idx, semantic_positions] = torch.where(
                    replace.unsqueeze(-1),
                    semantic_emb,
                    mixed[sample_idx, semantic_positions],
                )

            # SEP predicts C0[0], which may replace the first codec-history input.
            code_input_start = text_len
            if input_len > code_input_start:
                code_logits = unified_logits[
                    sample_idx,
                    code_input_start - 1 : input_len - 1,
                    codec_offset : codec_offset + codec_vocab_size,
                ].clone()
                code_logits[:, codec_bos_id] = -torch.inf
                if codec_eos_id < codec_vocab_size:
                    code_logits[:, codec_eos_id] = -torch.inf
                code_ids = self._sample_temporal_level_ids(code_logits)
                code_positions = torch.arange(
                    code_input_start, input_len, device=temporal_inputs.device
                )
                replace = torch.rand(
                    code_ids.shape, device=temporal_inputs.device
                ) < float(scheduled_sampling_prob)
                code_emb = self.temporal_codebook_emb[0](code_ids)
                code_emb = code_emb + self.temporal_pos_emb(code_positions)
                mixed[sample_idx, code_positions] = torch.where(
                    replace.unsqueeze(-1),
                    code_emb,
                    mixed[sample_idx, code_positions],
                )

        return mixed

    def _effective_depth_scheduled_sampling_prob(self) -> float:
        return self._scheduled_sampling_ramp(
            self.depth_scheduled_sampling_prob,
            self.depth_scheduled_sampling_start_step,
            self.depth_scheduled_sampling_warmup_steps,
        )

    def _effective_depth_chain_scheduled_sampling_prob(self) -> float:
        return self._scheduled_sampling_ramp(
            self.depth_chain_scheduled_sampling_prob,
            self.depth_chain_scheduled_sampling_start_step,
            self.depth_chain_scheduled_sampling_warmup_steps,
        )

    def _select_encoder_hidden(
        self,
        outputs: Any,
        output_layer: Optional[int],
    ) -> torch.Tensor:
        if output_layer is None:
            return outputs.last_hidden_state

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(
                "speech_encoder_output_layer was set, but encoder hidden states were not returned."
            )
        if output_layer >= len(hidden_states):
            raise ValueError(
                f"speech_encoder_output_layer={output_layer} is out of range for "
                f"{self.speech_encoder_name}; got {len(hidden_states)} hidden-state tensors."
            )
        return hidden_states[output_layer]

    def _align_1d_mask(
        self, mask: Optional[torch.Tensor], target_len: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.ndim != 2:
            raise ValueError(f"Expected attention mask with shape [B, T], got {tuple(mask.shape)}")

        if mask.size(1) != target_len:
            m = mask.to(dtype=torch.float32).unsqueeze(1)  # [B, 1, T]
            m = F.interpolate(m, size=target_len, mode="nearest").squeeze(1)
            mask = (m > 0.5).to(dtype=torch.long)
        else:
            mask = mask.to(dtype=torch.long)

        empty_rows = mask.sum(dim=1) == 0
        if empty_rows.any():
            mask = mask.clone()
            mask[empty_rows, 0] = 1

        return mask.to(device=device)

    def _causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        # True entries are masked in nn.TransformerDecoder.
        return torch.triu(torch.ones((t, t), dtype=torch.bool, device=device), diagonal=1)

    def _build_temporal_inputs(self, labels: torch.Tensor) -> torch.Tensor:
        """Build temporal-decoder inputs from modeled codebook embeddings + position."""
        if labels.ndim != 3:
            raise ValueError(
                f"Expected labels shape [B, T, Q] for temporal inputs, got {tuple(labels.shape)}."
            )
        if labels.size(2) != self.input_num_quantizers:
            raise ValueError(
                f"Expected labels Q={self.input_num_quantizers}, got Q={labels.size(2)}."
            )

        bsz, tlen, _ = labels.shape
        if tlen > self.temporal_max_positions:
            raise ValueError(
                f"Sequence length {tlen} exceeds temporal_max_positions={self.temporal_max_positions}."
            )

        summed = torch.zeros(
            (bsz, tlen, self.temporal_pos_emb.embedding_dim),
            device=labels.device,
            dtype=self.temporal_pos_emb.weight.dtype,
        )
        for level in range(self.temporal_context_levels):
            ids = labels[:, :, level]
            valid = (ids != self.ignore_index) & (ids >= 0) & (ids < self.vocab_size)
            safe_ids = ids.clamp(min=0, max=self.vocab_size - 1)
            emb = self.temporal_codebook_emb[level](safe_ids)
            summed = summed + emb * valid.unsqueeze(-1).type_as(emb)

        pos_ids = torch.arange(tlen, device=labels.device)
        pos_emb = self.temporal_pos_emb(pos_ids).unsqueeze(0).expand(bsz, -1, -1)
        return summed + pos_emb

    def _build_temporal_text_inputs(
        self,
        text_input_ids: torch.Tensor,
        sep_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed a text-only prefix exactly as the joint temporal path does."""
        if not (self.enable_text_codec_ar or self.enable_text_prefix_ar):
            raise RuntimeError("Temporal text-prefix path is disabled.")
        if text_input_ids.ndim != 2:
            raise ValueError(
                f"Expected text_input_ids shape [B, T], got {tuple(text_input_ids.shape)}."
            )
        bsz, tlen = text_input_ids.shape
        if tlen > self.temporal_max_positions:
            raise ValueError(
                f"Temporal text length {tlen} exceeds temporal_max_positions="
                f"{self.temporal_max_positions}."
            )

        token_emb = self._embed_text_tokens_with_sep_prompt(
            text_input_ids,
            sep_prompt_embedding,
        )
        pos_ids = torch.arange(tlen, device=text_input_ids.device)
        pos_emb = self.temporal_pos_emb(pos_ids).unsqueeze(0).expand(bsz, -1, -1)
        return token_emb + pos_emb

    def _embed_text_tokens_no_pos(self, text_ids: torch.Tensor) -> torch.Tensor:
        safe_ids = text_ids.clamp(min=0, max=int(self.text_vocab_size) - 1)
        emb = self.text_token_emb(safe_ids)
        valid = (text_ids != self.text_pad_token_id).unsqueeze(-1).type_as(emb)
        return emb * valid

    def _embed_text_tokens_with_sep_prompt(
        self,
        text_ids: torch.Tensor,
        sep_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed text tokens and replace source-style SEP boundaries.

        Text-prefix runs replace only the terminal SEP. Quality-CoT can
        either preserve that behavior or replace both boundaries in
        [source text, SEP, target text, SEP, C0]. The source-derived Mimi
        prompt is never an extra inference input token.
        """
        emb = self._embed_text_tokens_no_pos(text_ids)
        if not bool(getattr(self, "enable_mimi_source_speaker_prompt", False)):
            return emb
        if self.text_sep_token_id is None:
            raise RuntimeError("Mimi source speaker prompt requires text_sep_token_id.")

        sep_mask = text_ids.eq(int(self.text_sep_token_id))
        # Source-text beam decoding begins before its first SEP exists. It must
        # remain valid until the beam emits that boundary.
        if not bool(sep_mask.any().item()):
            return emb
        if sep_prompt_embedding is None:
            raise RuntimeError(
                "Mimi source speaker prompt is enabled but SEP prompt embedding was not provided."
            )
        prompt = sep_prompt_embedding
        if prompt.ndim == 3:
            prompt = prompt[:, 0, :]
        if prompt.ndim != 2 or prompt.size(0) != text_ids.size(0):
            raise RuntimeError(
                "SEP prompt embedding must be [B, D] with the same batch size as text ids."
            )
        if prompt.size(-1) != emb.size(-1):
            raise RuntimeError(
                f"SEP prompt dim {prompt.size(-1)} does not match text embedding dim {emb.size(-1)}."
            )

        emb = emb.clone()
        replace_all = bool(
            self.enable_quality_cot and self.quality_cot_prompt_all_seps
        )
        for sample_idx in range(text_ids.size(0)):
            sep_positions = torch.nonzero(sep_mask[sample_idx], as_tuple=False).flatten()
            if sep_positions.numel() == 0:
                continue
            if (
                self.enable_quality_cot
                and not self.enable_uniss_content_controls
                and not replace_all
            ):
                # The first SEP is the source->target boundary and keeps its
                # learned token embedding. Only the second, target->C0 SEP is
                # replaced by the Mimi source prompt. During target-text beam
                # search only the first SEP exists, so replacing "the latest"
                # SEP here would introduce a train/inference mismatch.
                if sep_positions.numel() < 2:
                    continue
                sep_positions = sep_positions[-1:]
            elif not replace_all:
                sep_positions = sep_positions[-1:]
            emb[sample_idx, sep_positions] = prompt[sample_idx].to(
                device=emb.device, dtype=emb.dtype
            )
        return emb

    def _embed_code_rows_no_pos(self, labels: torch.Tensor) -> torch.Tensor:
        bsz, tlen, _ = labels.shape
        summed = torch.zeros(
            (bsz, tlen, self.temporal_pos_emb.embedding_dim),
            device=labels.device,
            dtype=self.temporal_pos_emb.weight.dtype,
        )
        for level in range(self.temporal_context_levels):
            ids = labels[:, :, level]
            valid = (ids != self.ignore_index) & (ids >= 0) & (ids < self.vocab_size)
            safe_ids = ids.clamp(min=0, max=self.vocab_size - 1)
            emb = self.temporal_codebook_emb[level](safe_ids)
            summed = summed + emb * valid.unsqueeze(-1).type_as(emb)
        return summed

    def _build_textprefix_temporal_batch(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        code_attention_mask: Optional[torch.Tensor],
        sep_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        if text_attention_mask is None:
            text_attention_mask = (text_input_ids != self.text_pad_token_id).long()
        if code_attention_mask is None:
            code_attention_mask = (labels != self.ignore_index).any(dim=-1).long()
        bsz = labels.size(0)
        device = labels.device
        text_lengths = torch.clamp(text_attention_mask.to(device=device).sum(dim=1).long(), min=2)
        code_lengths = torch.clamp(code_attention_mask.to(device=device).sum(dim=1).long(), min=2)
        total_lengths = text_lengths + code_lengths - 1
        max_total = int(total_lengths.max().item())
        if max_total > self.temporal_max_positions:
            raise ValueError(
                f"Text-prefix temporal length {max_total} exceeds temporal_max_positions="
                f"{self.temporal_max_positions}."
            )
        dim = self.temporal_pos_emb.embedding_dim
        temporal_inputs = torch.zeros((bsz, max_total, dim), device=device, dtype=self.temporal_pos_emb.weight.dtype)
        temporal_mask = torch.zeros((bsz, max_total), device=device, dtype=torch.long)
        max_text_target_len = int((text_lengths - 1).max().item())
        text_targets = torch.full((bsz, max_text_target_len), self.ignore_index, device=device, dtype=torch.long)
        text_target_mask = torch.zeros((bsz, max_text_target_len), device=device, dtype=torch.long)
        code_start_positions: List[int] = []
        for i in range(bsz):
            text_len = int(text_lengths[i].item())
            code_len = int(code_lengths[i].item())
            text_ids = text_input_ids[i : i + 1, :text_len].to(device=device)
            code_prev = labels[i : i + 1, : code_len - 1, : self.input_num_quantizers]
            text_emb = self._embed_text_tokens_with_sep_prompt(
                text_ids,
                (sep_prompt_embedding[i : i + 1] if sep_prompt_embedding is not None else None),
            )
            seq_emb = torch.cat(
                [text_emb.squeeze(0), self._embed_code_rows_no_pos(code_prev).squeeze(0)],
                dim=0,
            )
            total_len = seq_emb.size(0)
            seq_emb = seq_emb + self.temporal_pos_emb(torch.arange(total_len, device=device))
            temporal_inputs[i, :total_len] = seq_emb
            temporal_mask[i, :total_len] = 1
            target_text_len = text_len - 1
            text_targets[i, :target_text_len] = text_ids.squeeze(0)[1:]
            text_target_mask[i, :target_text_len] = 1
            code_start_positions.append(text_len)
        return temporal_inputs, temporal_mask, text_targets, text_target_mask, code_start_positions

    def _build_textcodec_temporal_batch(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        code_attention_mask: Optional[torch.Tensor],
        sep_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        if text_attention_mask is None:
            text_attention_mask = (text_input_ids != self.text_pad_token_id).long()
        if code_attention_mask is None:
            code_attention_mask = (labels != self.ignore_index).any(dim=-1).long()
        bsz = labels.size(0)
        device = labels.device
        text_lengths = torch.clamp(text_attention_mask.to(device=device).sum(dim=1).long(), min=2)
        code_lengths = torch.clamp(code_attention_mask.to(device=device).sum(dim=1).long(), min=2)
        # Inputs: full text prefix ending in SEP, then previous codec rows excluding BOS and final target row.
        input_lengths = text_lengths + torch.clamp(code_lengths - 2, min=0)
        target_lengths = (text_lengths - 1) + (code_lengths - 1)
        max_input = int(input_lengths.max().item())
        max_target = int(target_lengths.max().item())
        if max_input > self.temporal_max_positions:
            raise ValueError(
                f"Text+codec AR length {max_input} exceeds temporal_max_positions={self.temporal_max_positions}."
            )
        dim = self.temporal_pos_emb.embedding_dim
        temporal_inputs = torch.zeros((bsz, max_input, dim), device=device, dtype=self.temporal_pos_emb.weight.dtype)
        temporal_mask = torch.zeros((bsz, max_input), device=device, dtype=torch.long)
        unified_targets = torch.full((bsz, max_target), self.ignore_index, device=device, dtype=torch.long)
        unified_target_mask = torch.zeros((bsz, max_target), device=device, dtype=torch.long)
        codec_target_mask = torch.zeros((bsz, max_target), device=device, dtype=torch.long)
        code_start_positions: List[int] = []
        offset = int(self.text_vocab_size)
        for i in range(bsz):
            text_len = int(text_lengths[i].item())
            code_len = int(code_lengths[i].item())
            text_ids = text_input_ids[i : i + 1, :text_len].to(device=device)
            code_prev = labels[i : i + 1, 1 : max(1, code_len - 1), : self.input_num_quantizers]
            pieces = [
                self._embed_text_tokens_with_sep_prompt(
                    text_ids,
                    (sep_prompt_embedding[i : i + 1] if sep_prompt_embedding is not None else None),
                ).squeeze(0)
            ]
            if code_prev.size(1) > 0:
                pieces.append(self._embed_code_rows_no_pos(code_prev).squeeze(0))
            seq_emb = torch.cat(pieces, dim=0)
            in_len = seq_emb.size(0)
            seq_emb = seq_emb + self.temporal_pos_emb(torch.arange(in_len, device=device))
            temporal_inputs[i, :in_len] = seq_emb
            temporal_mask[i, :in_len] = 1

            text_target_len = text_len - 1
            unified_targets[i, :text_target_len] = text_ids.squeeze(0)[1:]
            unified_target_mask[i, :text_target_len] = 1

            code_targets = labels[i, 1:code_len, 0]
            valid_code = (code_targets != self.ignore_index) & (code_targets >= 0) & (code_targets < self.vocab_size)
            code_target_len = code_targets.numel()
            start = text_target_len
            unified_targets[i, start : start + code_target_len] = torch.where(
                valid_code,
                code_targets + offset,
                torch.full_like(code_targets, self.ignore_index),
            )
            unified_target_mask[i, start : start + code_target_len] = valid_code.long()
            codec_target_mask[i, start : start + code_target_len] = valid_code.long()
            code_start_positions.append(text_target_len)
        return temporal_inputs, temporal_mask, unified_targets, unified_target_mask, codec_target_mask, code_start_positions

    def _build_quality_cot_textcodec_temporal_batch(
        self,
        source_text_input_ids: torch.Tensor,
        source_text_attention_mask: Optional[torch.Tensor],
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        code_attention_mask: Optional[torch.Tensor],
        sep_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[int],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build [source text, SEP(prompt), target text, SEP(prompt), C0] inputs.

        The target BOS is omitted when concatenating the segments. Both language
        controls are deterministically inserted at inference, so their CE labels
        are masked while source/target lexical tokens and both SEP boundaries are
        supervised.
        """
        if source_text_attention_mask is None:
            source_text_attention_mask = (
                source_text_input_ids != self.text_pad_token_id
            ).long()
        if text_attention_mask is None:
            text_attention_mask = (text_input_ids != self.text_pad_token_id).long()
        if source_text_input_ids.size(0) != labels.size(0) or text_input_ids.size(0) != labels.size(0):
            raise ValueError("Quality-CoT source/target text batch size must match codec labels.")

        device = labels.device
        bsz = labels.size(0)
        source_lengths = torch.clamp(
            source_text_attention_mask.to(device=device).sum(dim=1).long(),
            min=2,
        )
        target_lengths = torch.clamp(
            text_attention_mask.to(device=device).sum(dim=1).long(),
            min=2,
        )
        combined_lengths = source_lengths + target_lengths - 1
        max_combined_len = int(combined_lengths.max().item())
        if max_combined_len > self.temporal_max_positions:
            raise ValueError(
                "Quality-CoT text chain length "
                f"{max_combined_len} exceeds temporal_max_positions={self.temporal_max_positions}."
            )

        combined_ids = torch.full(
            (bsz, max_combined_len),
            fill_value=int(self.text_pad_token_id),
            dtype=torch.long,
            device=device,
        )
        combined_mask = torch.zeros((bsz, max_combined_len), dtype=torch.long, device=device)
        for i in range(bsz):
            source_len = int(source_lengths[i].item())
            target_len = int(target_lengths[i].item())
            source_ids = source_text_input_ids[i, :source_len].to(
                device=device, dtype=torch.long
            )
            target_ids = text_input_ids[i, :target_len].to(device=device, dtype=torch.long)
            if int(source_ids[0].item()) != int(self.text_bos_token_id):
                raise ValueError("Quality-CoT source text must begin with text_bos_token_id.")
            if int(target_ids[0].item()) != int(self.text_bos_token_id):
                raise ValueError("Quality-CoT target text must begin with text_bos_token_id.")
            merged = torch.cat([source_ids, target_ids[1:]], dim=0)
            combined_ids[i, : merged.numel()] = merged
            combined_mask[i, : merged.numel()] = 1

        (
            temporal_inputs,
            temporal_mask,
            unified_targets,
            unified_target_mask,
            codec_target_mask,
            code_start_positions,
        ) = self._build_textcodec_temporal_batch(
            text_input_ids=combined_ids,
            text_attention_mask=combined_mask,
            labels=labels,
            code_attention_mask=code_attention_mask,
            sep_prompt_embedding=sep_prompt_embedding,
        )

        source_text_target_mask = torch.zeros_like(unified_target_mask)
        target_text_target_mask = torch.zeros_like(unified_target_mask)
        source_control_len = len(
            self.quality_cot_source_text_prefix_extra_token_ids_after_bos
        )
        target_control_len = len(self.text_prefix_extra_token_ids_after_bos)
        if self.enable_uniss_content_controls:
            # START_CONTENT is deterministic, like the language control.
            source_control_len += 1
            target_control_len += 1
        for i in range(bsz):
            source_target_len = int(source_lengths[i].item()) - 1
            target_target_len = int(target_lengths[i].item()) - 1
            target_start = source_target_len
            source_content_start = min(source_control_len, source_target_len)
            target_content_start = target_start + min(target_control_len, target_target_len)
            target_end = target_start + target_target_len

            source_text_target_mask[i, source_content_start:source_target_len] = 1
            target_text_target_mask[i, target_content_start:target_end] = 1

            # Language controls are forced at inference, not sampled by beam.
            if source_content_start > 0:
                unified_targets[i, :source_content_start] = self.ignore_index
                unified_target_mask[i, :source_content_start] = 0
            if target_content_start > target_start:
                unified_targets[i, target_start:target_content_start] = self.ignore_index
                unified_target_mask[i, target_start:target_content_start] = 0

        return (
            temporal_inputs,
            temporal_mask,
            unified_targets,
            unified_target_mask,
            codec_target_mask,
            code_start_positions,
            source_text_target_mask,
            target_text_target_mask,
        )

    def _build_textprefix_generation_batch(
        self,
        text_prefix_ids: torch.Tensor,
        code_prefix_labels: torch.Tensor,
        sep_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        # Unified text+codec training treats the structural code BOS row as
        # outside the AR stream: SEP predicts C0[0]. Keep generation aligned
        # with that training layout instead of inserting BOS after SEP.
        if self.enable_text_codec_ar and not self._uses_unity_text_decoder_temporal():
            temporal_inputs, temporal_mask, last_positions = (
                self._build_aligned_textcodec_generation_batch(
                    text_prefix_ids=text_prefix_ids,
                    code_prefix_labels=code_prefix_labels,
                    sep_prompt_embedding=sep_prompt_embedding,
                )
            )
            target_len = int(code_prefix_labels.size(1))
            code_start_positions = [
                max(0, int(last_position) - target_len + 1)
                for last_position in last_positions
            ]
            return temporal_inputs, temporal_mask, code_start_positions

        bsz = code_prefix_labels.size(0)
        device = code_prefix_labels.device
        text_lengths = torch.clamp((text_prefix_ids != self.text_pad_token_id).sum(dim=1).long(), min=1)
        code_len = code_prefix_labels.size(1)
        total_lengths = text_lengths + code_len
        max_total = int(total_lengths.max().item())
        if max_total > self.temporal_max_positions:
            raise ValueError(
                f"Text-prefix generation length {max_total} exceeds temporal_max_positions="
                f"{self.temporal_max_positions}."
            )
        dim = self.temporal_pos_emb.embedding_dim
        temporal_inputs = torch.zeros((bsz, max_total, dim), device=device, dtype=self.temporal_pos_emb.weight.dtype)
        temporal_mask = torch.zeros((bsz, max_total), device=device, dtype=torch.long)
        code_start_positions: List[int] = []
        for i in range(bsz):
            text_len = int(text_lengths[i].item())
            text_ids = text_prefix_ids[i : i + 1, :text_len].to(device=device)
            code_rows = code_prefix_labels[i : i + 1, :, : self.input_num_quantizers]
            text_emb = self._embed_text_tokens_with_sep_prompt(
                text_ids,
                (sep_prompt_embedding[i : i + 1] if sep_prompt_embedding is not None else None),
            )
            seq_emb = torch.cat(
                [text_emb.squeeze(0), self._embed_code_rows_no_pos(code_rows).squeeze(0)],
                dim=0,
            )
            total_len = seq_emb.size(0)
            seq_emb = seq_emb + self.temporal_pos_emb(torch.arange(total_len, device=device))
            temporal_inputs[i, :total_len] = seq_emb
            temporal_mask[i, :total_len] = 1
            code_start_positions.append(text_len)
        return temporal_inputs, temporal_mask, code_start_positions

    def _build_aligned_textcodec_generation_batch(
        self,
        text_prefix_ids: torch.Tensor,
        code_prefix_labels: torch.Tensor,
        sep_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Build unified AR inputs where SEP directly predicts the first C0.

        ``code_prefix_labels[:, 0]`` is the structural code BOS row. It
        is not part of the unified semantic+C0 token stream and must not be
        inserted between SEP and the first generated C0 token.
        """
        bsz = code_prefix_labels.size(0)
        device = code_prefix_labels.device
        text_lengths = torch.clamp(
            (text_prefix_ids != self.text_pad_token_id).sum(dim=1).long(),
            min=1,
        )
        code_history_len = max(0, int(code_prefix_labels.size(1)) - 1)
        total_lengths = text_lengths + code_history_len
        max_total = int(total_lengths.max().item())
        if max_total > self.temporal_max_positions:
            raise ValueError(
                f"Aligned text+codec generation length {max_total} exceeds "
                f"temporal_max_positions={self.temporal_max_positions}."
            )

        dim = self.temporal_pos_emb.embedding_dim
        temporal_inputs = torch.zeros(
            (bsz, max_total, dim),
            device=device,
            dtype=self.temporal_pos_emb.weight.dtype,
        )
        temporal_mask = torch.zeros((bsz, max_total), device=device, dtype=torch.long)
        last_positions: List[int] = []
        for i in range(bsz):
            text_len = int(text_lengths[i].item())
            text_ids = text_prefix_ids[i : i + 1, :text_len].to(device=device)
            pieces = [
                self._embed_text_tokens_with_sep_prompt(
                    text_ids,
                    (sep_prompt_embedding[i : i + 1] if sep_prompt_embedding is not None else None),
                ).squeeze(0)
            ]
            if code_history_len > 0:
                code_history = code_prefix_labels[i, 1:, 0]
                safe_history = code_history.clamp(min=0, max=self.vocab_size - 1)
                code_emb = self.temporal_codebook_emb[0](safe_history)
                code_emb = code_emb * (code_history != self.ignore_index).unsqueeze(-1).type_as(code_emb)
                pieces.append(code_emb)
            seq_emb = torch.cat(pieces, dim=0)
            total_len = int(seq_emb.size(0))
            seq_emb = seq_emb + self.temporal_pos_emb(torch.arange(total_len, device=device))
            temporal_inputs[i, :total_len] = seq_emb
            temporal_mask[i, :total_len] = 1
            last_positions.append(total_len - 1)
        return temporal_inputs, temporal_mask, last_positions

    def _uses_unity_text_decoder_temporal(self) -> bool:
        return bool(getattr(self, "use_unity_text_decoder_temporal", False))

    def _uses_unity_direct_c0_temporal(self) -> bool:
        return bool(getattr(self, "use_unity_direct_c0_temporal", False))

    def _uses_unity_native_decoder_temporal(self) -> bool:
        return bool(getattr(self, "use_unity_native_decoder_temporal", False))

    def _unity_code_offset(self) -> int:
        if self._uses_unity_native_decoder_temporal() and self.text_codec_vocab_size is not None:
            return int(self.text_codec_vocab_size) - int(self.vocab_size)
        return int(self.text_vocab_size)

    def _unity_encoder_padding_mask(self, encoder_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if encoder_mask is None:
            return None
        # TransVIP/fairseq2 attention expects a float additive padding mask
        # with 0 for valid positions and -inf for padded positions. DirectS2ST
        # stores encoder_mask as 1=valid/0=pad, so do not pass a bool mask here:
        # some fairseq2 attention paths add the mask directly to attention logits.
        if encoder_mask.dtype.is_floating_point:
            if torch.isinf(encoder_mask).any() or bool((encoder_mask < 0).any().item()):
                return encoder_mask
            valid = encoder_mask > 0
        else:
            valid = encoder_mask != 0
        if bool(valid.all().item()):
            return None
        padding = ~valid
        padding_mask = torch.zeros(
            encoder_mask.shape,
            device=encoder_mask.device,
            dtype=torch.float32,
        )
        return padding_mask.masked_fill(padding, -torch.inf)

    def _build_unity_direct_c0_token_batch(
        self,
        code_prefix_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map a Direct Mimi C0 history into the native UnitY unified vocab.

        Keep UnitY's fixed BOS/language control prefix, but never consume target
        text or SEP on this path.  The same prefix is available at inference, so
        this preserves source-only conditioning while retaining the decoder's
        pretrained target-language routing.
        """
        if not self._uses_unity_direct_c0_temporal():
            raise RuntimeError("unity_direct_c0 token mapping requested for another backend.")
        if code_prefix_labels.ndim != 3:
            raise ValueError(
                "Expected code_prefix_labels [B, T, Q], got "
                f"{tuple(code_prefix_labels.shape)}."
            )
        c0_ids = code_prefix_labels[:, :, 0]
        valid = (
            (c0_ids != int(self.ignore_index))
            & (c0_ids >= 0)
            & (c0_ids < int(self.vocab_size))
        )
        code_token_ids = torch.full_like(c0_ids, int(self.text_pad_token_id))
        code_token_ids[valid] = c0_ids[valid] + int(self._unity_code_offset())
        prefix_ids = self._unity_direct_c0_control_prefix_ids()
        if prefix_ids:
            prefix = torch.as_tensor(
                prefix_ids,
                device=c0_ids.device,
                dtype=c0_ids.dtype,
            ).unsqueeze(0).expand(c0_ids.size(0), -1)
            prefix_mask = torch.ones_like(prefix, dtype=torch.long)
            token_ids = torch.cat([prefix, code_token_ids], dim=1)
            token_mask = torch.cat([prefix_mask, valid.long()], dim=1)
        else:
            token_ids = code_token_ids
            token_mask = valid.long()
        return token_ids, token_mask

    def _unity_direct_c0_control_prefix_ids(self) -> List[int]:
        prefix: List[int] = []
        if self.text_bos_token_id is not None:
            prefix.append(int(self.text_bos_token_id))
        prefix.extend(int(token_id) for token_id in self.text_prefix_extra_token_ids_after_bos)
        return prefix

    def _unity_direct_c0_control_prefix_length(self) -> int:
        return len(self._unity_direct_c0_control_prefix_ids())

    def _build_transvip_source_speaker_prompt(
        self,
        transvip_prompt_wavs: Optional[torch.Tensor],
        transvip_prompt_wav_lens: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not bool(getattr(self, "transvip_use_source_speaker_prompt", False)):
            return None, None
        if transvip_prompt_wavs is None or transvip_prompt_wav_lens is None:
            raise ValueError(
                "transvip_use_source_speaker_prompt=True requires "
                "transvip_prompt_wavs and transvip_prompt_wav_lens from the collator/inferencer."
            )
        codec = getattr(self, "transvip_prompt_codec", None)
        if codec is None:
            raise RuntimeError("TransVIP prompt codec is not initialized.")
        unity_device = next(self.transvip_unity.parameters()).device
        try:
            codec_device = next(codec.parameters()).device
        except StopIteration:
            codec_device = unity_device
        if codec_device != unity_device:
            codec.to(unity_device)
        codec.eval()
        wavs = transvip_prompt_wavs.to(device=unity_device, dtype=torch.float32)
        wav_lens = transvip_prompt_wav_lens.to(device=unity_device, dtype=torch.long).clamp_min(1)
        with torch.no_grad():
            if hasattr(codec, "preprocess"):
                wavs_for_codec = codec.preprocess(wavs)
            else:
                wavs_for_codec = wavs
            prompts = codec.encoder(wavs_for_codec.unsqueeze(1)).transpose(1, 2)
        max_frames = min(int(self.transvip_prompt_max_frames), int(prompts.size(1)))
        prompts = prompts[:, :max_frames].detach().to(dtype=torch.float32)
        prompt_lens = torch.ceil(wav_lens.to(dtype=torch.float32) / 320.0).to(dtype=torch.long)
        prompt_lens = prompt_lens.clamp(min=1, max=max_frames)
        return prompts, prompt_lens

    def _initialize_campplus_source_model(self) -> None:
        root = os.path.abspath(os.path.expanduser(self.campplus_model_root))
        checkpoint = os.path.abspath(os.path.expanduser(self.campplus_checkpoint_path))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"CAMPPlus model root does not exist: {root}")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"CAMPPlus checkpoint does not exist: {checkpoint}")
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from modules.campplus.DTDNN import CAMPPlus
        except Exception as exc:
            raise ImportError(
                "Could not import CAMPPlus from the configured Seed-VC root: "
                f"{root}"
            ) from exc
        model = CAMPPlus(feat_dim=80, embedding_size=self.campplus_embedding_size)
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        # Keep the fixed extractor outside the registered module tree: its 27 MB
        # weights are reloaded from the declared checkpoint and are not duplicated
        # in every DirectS2ST checkpoint or sharded by FSDP.
        self._campplus_source_model_holder.append(model)
        self._campplus_input_mode_logged = False
        print(
            "[Init:CAMPPlusSource] enabled",
            {
                "checkpoint": checkpoint,
                "embedding_size": self.campplus_embedding_size,
                "trainable": False,
                "in_checkpoint_state_dict": False,
                "depth_conditioning": self.campplus_depth_conditioning_mode,
                "film_scale": self.campplus_film_scale,
                "identity_loss_weight": self.campplus_depth_identity_loss_weight,
            },
        )

    def _encode_campplus_source_embedding(
        self,
        source_wavs: Optional[torch.Tensor],
        source_wav_lens: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.enable_campplus_depth_conditioning:
            return None
        if source_wavs is None or source_wav_lens is None:
            raise ValueError(
                "CAMPPlus source conditioning requires source_acoustic_wavs and "
                "source_acoustic_wav_lens from the collator/inferencer."
            )
        if not self._campplus_source_model_holder:
            raise RuntimeError("CAMPPlus source encoder is not initialized.")
        model = self._campplus_source_model_holder[0]
        device = source_wavs.device
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = device
        if model_device != device:
            model.to(device)
        model.eval()
        wavs = source_wavs.to(device=device, dtype=torch.float32)
        wav_lens = source_wav_lens.to(device=device, dtype=torch.long).clamp_min(1)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=False)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            feature_list: List[torch.Tensor] = []
            feature_lens: List[int] = []
            for sample_idx in range(wavs.size(0)):
                wav_len = int(wav_lens[sample_idx].item())
                wav = wavs[sample_idx, :wav_len]
                # Kaldi fbank needs enough samples for at least one analysis frame.
                if wav.numel() < 400:
                    wav = F.pad(wav, (0, 400 - int(wav.numel())))
                features = torchaudio.compliance.kaldi.fbank(
                    wav.unsqueeze(0),
                    num_mel_bins=80,
                    dither=0,
                    sample_frequency=16000,
                )
                features = features - features.mean(dim=0, keepdim=True)
                feature_list.append(features)
                feature_lens.append(int(features.size(0)))

            # CAMPPlus supports x_lens. Pad fbank features and run its TDNN once
            # for the complete batch instead of launching the full network once
            # per utterance.
            max_frames = max(feature_lens)
            feature_batch = feature_list[0].new_zeros(
                (len(feature_list), max_frames, int(feature_list[0].size(1)))
            )
            for sample_idx, features in enumerate(feature_list):
                feature_batch[sample_idx, : features.size(0)] = features
            embeddings = model(feature_batch, feature_lens)
        return embeddings.detach()

    def _encode_mimi_source_acoustic_frames(
        self,
        source_acoustic_wavs: Optional[torch.Tensor],
        source_acoustic_wav_lens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return frozen Mimi frame features as ``[B, T, D]`` plus a valid-frame mask."""
        if source_acoustic_wavs is None or source_acoustic_wav_lens is None:
            raise ValueError(
                "Mimi source acoustic features require source_acoustic_wavs and "
                "source_acoustic_wav_lens from the collator/inferencer."
            )
        if self.mimi_source_acoustic_model is None:
            raise RuntimeError("Mimi source acoustic encoder is not initialized.")
        model = self.mimi_source_acoustic_model
        device = next(model.parameters()).device
        wavs = source_acoustic_wavs.to(device=device, dtype=torch.float32)
        wav_lens = source_acoustic_wav_lens.to(device=device, dtype=torch.long).clamp_min(1)
        input_sr = int(self.mimi_source_acoustic_input_sample_rate)
        model_sr = int(self.mimi_source_acoustic_model_sample_rate)

        if input_sr != model_sr:
            resampled: List[torch.Tensor] = []
            new_lens: List[int] = []
            for i in range(wavs.size(0)):
                wav_len = int(wav_lens[i].item())
                wav_i = torchaudio.functional.resample(wavs[i, :wav_len], input_sr, model_sr)
                resampled.append(wav_i)
                new_lens.append(int(wav_i.numel()))
            max_len = max(new_lens)
            padded = wavs.new_zeros((len(resampled), max_len))
            for i, wav_i in enumerate(resampled):
                padded[i, : wav_i.numel()] = wav_i
            wavs = padded
            wav_lens = torch.tensor(new_lens, device=device, dtype=torch.long).clamp_min(1)

        with torch.no_grad():
            hidden = model.encoder(wavs.unsqueeze(1))
            debug_generation = bool(
                str(os.environ.get("DIRECTS2ST_DUMP_GENERATION_DEBUG", "") or "").strip()
            )
            if debug_generation:
                self._debug_mimi_encoder_hidden = hidden.detach()
            force_math_sdpa = str(
                os.environ.get("DIRECTS2ST_MIMI_FORCE_MATH_SDPA", "") or ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            sdpa_context = (
                torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                )
                if force_math_sdpa and hidden.is_cuda
                else nullcontext()
            )
            with sdpa_context:
                encoder_outputs = model.encoder_transformer(
                    hidden.transpose(1, 2),
                    return_dict=True,
                )
            hidden = encoder_outputs[0].transpose(1, 2)
            if debug_generation:
                self._debug_mimi_transformer_hidden = hidden.detach()
            if getattr(model, "downsample", None) is not None:
                hidden = model.downsample(hidden)
            if debug_generation:
                self._debug_mimi_final_hidden = hidden.detach()

        encoded_lens = model.get_encoded_length(wav_lens).to(device=hidden.device)
        max_frames = int(hidden.size(-1))
        frame_ids = torch.arange(max_frames, device=hidden.device).unsqueeze(0)
        frame_mask = frame_ids < encoded_lens.clamp(min=1, max=max_frames).unsqueeze(1)
        return hidden.transpose(1, 2).contiguous(), frame_mask

    def _pool_mimi_source_acoustic_frames(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reproduce the existing mean/std pooling exactly on valid Mimi frames."""
        weights = frame_mask.to(dtype=frames.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        mean = (frames * weights).sum(dim=1) / denom
        if self.mimi_source_acoustic_pooling == "mean":
            return mean
        centered = (frames - mean.unsqueeze(1)) * weights
        var = centered.pow(2).sum(dim=1) / denom
        std = torch.sqrt(var.clamp_min(0.0) + 1.0e-6)
        if self.mimi_source_acoustic_pooling == "std":
            return std
        return torch.cat([mean, std], dim=-1)

    def _pack_mimi_source_style_tokens(
        self,
        global_embedding: torch.Tensor,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pack [global mean/std, ordered local style tokens] into one tensor.

        Keeping this in the existing source-acoustic payload means every training,
        callback, and offline free-running path receives identical conditioning.
        """
        if not self.enable_mimi_source_style_token_bank:
            return global_embedding
        token_count = int(self.mimi_source_style_token_count)
        global_dim = int(global_embedding.size(-1))
        frame_dim = int(frames.size(-1))
        use_local_mean_std = global_dim == 2 * frame_dim
        pooled_tokens: List[torch.Tensor] = []
        for batch_idx in range(frames.size(0)):
            valid_len = max(1, int(frame_mask[batch_idx].sum().item()))
            valid_frames = frames[batch_idx, :valid_len]
            local_tokens: List[torch.Tensor] = []
            for token_idx in range(token_count):
                start = (token_idx * valid_len) // token_count
                end = ((token_idx + 1) * valid_len) // token_count
                if end <= start:
                    segment = valid_frames[min(start, valid_len - 1) : min(start, valid_len - 1) + 1]
                else:
                    segment = valid_frames[start:end]
                mean = segment.mean(dim=0)
                if use_local_mean_std:
                    std = torch.sqrt(
                        segment.var(dim=0, unbiased=False).clamp_min(0.0) + 1.0e-6
                    )
                    local_tokens.append(torch.cat([mean, std], dim=-1))
                else:
                    local_tokens.append(mean)
            pooled_tokens.append(torch.stack(local_tokens, dim=0))
        style_tokens = torch.stack(pooled_tokens, dim=0)
        style_dim = int(style_tokens.size(-1))
        if style_dim != global_dim:
            raise RuntimeError(
                "Mimi style-token dimension must match the global source acoustic "
                f"dimension: {style_dim} != {global_dim}."
            )
        return torch.cat([global_embedding.unsqueeze(1), style_tokens], dim=1)

    def _encode_mimi_source_acoustic_embedding(
        self,
        source_acoustic_wavs: Optional[torch.Tensor],
        source_acoustic_wav_lens: Optional[torch.Tensor],
        pooled_override: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        # A terminal Mimi SEP prompt also needs this pooled representation even
        # when no acoustic residual is injected into temporal/depth decoding.
        if not (
            self.enable_mimi_source_acoustic_conditioning
            or self.enable_mimi_source_speaker_prompt
            or self.enable_mimi_source_acoustic_temporal_conditioning
            or self.enable_mimi_nar_depth_prompt
        ):
            return None
        warmup_requested = str(
            os.environ.get("DIRECTS2ST_MIMI_WARMUP_FIRST_SAMPLE", "") or ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if warmup_requested and not bool(
            getattr(self, "_mimi_source_acoustic_warmup_done", False)
        ):
            # Diagnostic parity with Trainer.evaluate(), which exercises Mimi on
            # a validation batch before EvalAudioLoggerCallback starts decoding.
            warmup_dtype = str(
                os.environ.get("DIRECTS2ST_MIMI_WARMUP_DTYPE", "fp32") or "fp32"
            ).strip().lower()
            if warmup_dtype not in {"fp32", "bf16"}:
                raise ValueError(
                    "DIRECTS2ST_MIMI_WARMUP_DTYPE must be 'fp32' or 'bf16'."
                )
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if warmup_dtype == "bf16"
                else nullcontext()
            )
            with autocast_context:
                self._encode_mimi_source_acoustic_frames(
                    source_acoustic_wavs[:1]
                    if source_acoustic_wavs is not None
                    else None,
                    source_acoustic_wav_lens[:1]
                    if source_acoustic_wav_lens is not None
                    else None,
                )
            self._mimi_source_acoustic_warmup_done = True
        frames, frame_mask = self._encode_mimi_source_acoustic_frames(
            source_acoustic_wavs,
            source_acoustic_wav_lens,
        )
        if (
            self.enable_mimi_source_speaker_prompt
            and self.source_speaker_prompt_backend == "mimi_latent"
        ):
            self._runtime_mimi_prompt_frames = (
                frames.detach() if self.mimi_source_speaker_prompt_detach else frames
            )
            self._runtime_mimi_prompt_mask = frame_mask.detach()
        if self.enable_mimi_nar_depth_prompt:
            if self.mimi_source_acoustic_model is None:
                raise RuntimeError("Mimi NAR depth requires the frozen Mimi source model.")
            with torch.no_grad():
                source_codes = self.mimi_source_acoustic_model.quantizer.encode(
                    frames.transpose(1, 2).contiguous(),
                    num_quantizers=self.num_codebook_levels,
                ).transpose(0, 1).contiguous()
            model_order = getattr(self, "codebook_model_order", None)
            if model_order is not None:
                order = torch.as_tensor(
                    model_order,
                    device=source_codes.device,
                    dtype=torch.long,
                )
                source_codes = source_codes.index_select(1, order)
            self.depth_decoder.set_runtime_source_codec_prompt(
                source_codes.detach(),
                frame_mask.detach(),
            )
        if pooled_override is None:
            pooled = self._pool_mimi_source_acoustic_frames(frames, frame_mask)
        else:
            pooled = pooled_override
            if pooled.ndim == 3:
                pooled = pooled[:, 0, :]
            if pooled.ndim != 2:
                raise RuntimeError(
                    "Precomputed Mimi source acoustic embeddings must be [B, D]."
                )
            pooled = pooled.to(device=frames.device, dtype=frames.dtype)
        packed = self._pack_mimi_source_style_tokens(pooled, frames, frame_mask)
        if self.mimi_source_acoustic_detach:
            packed = packed.detach()
        return packed

    def _source_speaker_prompt_uses_mimi_frames(self) -> bool:
        return bool(
            self.enable_mimi_source_speaker_prompt
            and self.source_speaker_prompt_backend == "mimi_latent"
        )

    def _build_learned_source_speaker_prompt_embedding(
        self,
        frames: Optional[torch.Tensor],
        frame_mask: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if frames is None or frame_mask is None:
            raise RuntimeError(
                f"Source speaker prompt backend {self.source_speaker_prompt_backend!r} "
                "did not receive its frame sequence."
            )
        modules = (
            self.source_speaker_prompt_frame_proj,
            self.source_speaker_prompt_input_norm,
            self.source_speaker_prompt_encoder,
            self.source_speaker_prompt_output_norm,
        )
        if any(module is None for module in modules):
            raise RuntimeError("Learned source speaker prompt modules are not initialized.")
        if frames.ndim != 3:
            raise RuntimeError(
                f"Source speaker prompt frames must be [B, T, D], got {tuple(frames.shape)}."
            )
        if frame_mask.ndim != 2 or frame_mask.shape[:2] != frames.shape[:2]:
            raise RuntimeError(
                "Source speaker prompt mask must be [B, T] and align with frames; "
                f"got frames={tuple(frames.shape)}, mask={tuple(frame_mask.shape)}."
            )

        prompt_frames = (
            frames.detach() if self.mimi_source_speaker_prompt_detach else frames
        ).to(device=device, dtype=dtype)
        valid_mask = frame_mask.to(device=device, dtype=torch.bool)
        empty_rows = ~valid_mask.any(dim=1)
        if empty_rows.any():
            valid_mask = valid_mask.clone()
            valid_mask[empty_rows, 0] = True

        hidden = self.source_speaker_prompt_frame_proj(prompt_frames)
        hidden = self.source_speaker_prompt_input_norm(hidden)

        # Dynamic sinusoidal positions preserve order without imposing a fixed
        # maximum source duration or adding a large position table.
        seq_len, hidden_dim = int(hidden.size(1)), int(hidden.size(2))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, hidden_dim))
        )
        pos_emb = torch.zeros(
            (seq_len, hidden_dim), device=device, dtype=torch.float32
        )
        pos_emb[:, 0::2] = torch.sin(positions * frequencies)
        if hidden_dim > 1:
            pos_emb[:, 1::2] = torch.cos(positions * frequencies)[:, : pos_emb[:, 1::2].size(1)]
        hidden = hidden + pos_emb.to(dtype=hidden.dtype).unsqueeze(0)
        hidden = self.source_speaker_prompt_encoder(
            hidden,
            src_key_padding_mask=~valid_mask,
        )
        weights = valid_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.source_speaker_prompt_output_norm(pooled)

    def _build_mimi_source_speaker_prompt_embedding(
        self,
        source_acoustic_embedding: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not bool(getattr(self, "enable_mimi_source_speaker_prompt", False)):
            return None
        if self.source_speaker_prompt_backend == "w2vbert_hidden":
            return self._build_learned_source_speaker_prompt_embedding(
                self._runtime_w2vbert_prompt_frames,
                self._runtime_w2vbert_prompt_mask,
                device=device,
                dtype=dtype,
            )
        if self.source_speaker_prompt_backend == "mimi_latent":
            return self._build_learned_source_speaker_prompt_embedding(
                self._runtime_mimi_prompt_frames,
                self._runtime_mimi_prompt_mask,
                device=device,
                dtype=dtype,
            )

        if source_acoustic_embedding is None:
            raise RuntimeError(
                "Mimi mean/std source speaker prompt is enabled but "
                "source_acoustic_embedding was not provided."
            )
        if (
            self.mimi_source_speaker_prompt_proj is None
            or self.mimi_source_speaker_prompt_norm is None
        ):
            raise RuntimeError("Mimi mean/std source speaker prompt modules are not initialized.")
        acoustic = (
            source_acoustic_embedding.detach()
            if self.mimi_source_speaker_prompt_detach
            else source_acoustic_embedding
        )
        if acoustic.ndim == 3:
            acoustic = acoustic[:, 0, :]
        if acoustic.ndim != 2:
            raise RuntimeError(
                "Mimi source speaker prompt expects [B, D] or packed [B, 1+K, D]."
            )
        acoustic = acoustic.to(device=device, dtype=dtype)
        spk_embed = self.mimi_source_speaker_prompt_proj(acoustic)
        spk_embed = self.mimi_source_speaker_prompt_norm(spk_embed)
        return spk_embed

    def _compute_text_codec_kd_loss(
        self,
        speech_logits: torch.Tensor,
        text_logits: torch.Tensor,
        unified_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if speech_logits.shape != text_logits.shape:
            raise ValueError(
                "KD requires speech/text path logits with the same shape, got "
                f"{tuple(speech_logits.shape)} vs {tuple(text_logits.shape)}."
            )
        flat_targets = unified_targets.reshape(-1)
        unity_code_offset = int(self._unity_code_offset())
        valid = (flat_targets != self.ignore_index) & (flat_targets < unity_code_offset)
        valid_tokens = valid.sum()
        if int(valid_tokens.detach().cpu().item()) <= 0:
            return speech_logits.new_zeros(()), valid_tokens

        flat_speech = speech_logits.reshape(-1, speech_logits.size(-1))
        flat_text = text_logits.reshape(-1, text_logits.size(-1)).detach()
        active = valid.nonzero(as_tuple=False).flatten()
        losses = []
        # Match TransVIP's memory-safe chunking over valid text tokens.
        chunk_size = 64
        for start in range(0, int(active.numel()), chunk_size):
            idx = active[start : start + chunk_size]
            stu = flat_speech.index_select(0, idx).float()
            tea = flat_text.index_select(0, idx).float()
            chunk_loss = F.kl_div(
                F.log_softmax(stu, dim=-1),
                F.softmax(tea, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            losses.append(chunk_loss)
        kd_loss = torch.cat(losses, dim=0).clamp(min=0.0, max=10.0).mean()
        return kd_loss.to(dtype=speech_logits.dtype), valid_tokens

    def _apply_mimi_source_acoustic_temporal_conditioning(
        self,
        hidden: torch.Tensor,
        source_acoustic_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not bool(getattr(self, "enable_mimi_source_acoustic_temporal_conditioning", False)):
            return hidden
        if source_acoustic_embedding is None:
            raise RuntimeError(
                "Mimi source acoustic temporal conditioning is enabled but "
                "source_acoustic_embedding was not provided."
            )
        if self.source_acoustic_temporal_proj is None or self.source_acoustic_temporal_norm is None:
            raise RuntimeError(
                "Mimi source acoustic temporal conditioning is enabled but not initialized."
            )
        acoustic = (
            source_acoustic_embedding.detach()
            if self.mimi_source_acoustic_temporal_detach
            else source_acoustic_embedding
        )
        if acoustic.ndim == 3:
            acoustic = acoustic[:, 0, :]
        if acoustic.ndim != 2:
            raise RuntimeError(
                "Mimi source acoustic temporal conditioning expects [B, D] or "
                "packed [B, 1+K, D]."
            )
        acoustic = acoustic.to(device=hidden.device, dtype=hidden.dtype)
        acoustic = self.source_acoustic_temporal_proj(acoustic).unsqueeze(1)
        return self.source_acoustic_temporal_norm(
            hidden + self.mimi_source_acoustic_temporal_residual_scale * acoustic
        )

    def _materialize_unity_padding_mask(
        self,
        padding_mask: Optional[Any],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if padding_mask is None:
            return torch.zeros((batch_size, seq_len), device=device)
        if hasattr(padding_mask, "materialize"):
            padding_mask = padding_mask.materialize()
        return padding_mask.to(device=device)

    def _apply_unity_source_memory_fusion(
        self,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[Any],
        source_memory: Optional[torch.Tensor],
        source_padding_mask: Optional[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not bool(getattr(self, "enable_unity_source_memory_fusion", False)):
            final_mask = self._materialize_unity_padding_mask(
                encoder_padding_mask,
                batch_size=int(encoder_out.size(0)),
                seq_len=int(encoder_out.size(1)),
                device=encoder_out.device,
            )
            return encoder_out, final_mask
        if source_memory is None:
            raise RuntimeError(
                "enable_unity_source_memory_fusion=True but no source memory was captured."
            )
        if self.unity_source_memory_proj is None or self.unity_source_memory_norm is None:
            raise RuntimeError("Unity source memory fusion modules are not initialized.")
        memory = source_memory.detach() if self.unity_source_memory_detach else source_memory
        memory = memory.to(device=encoder_out.device, dtype=encoder_out.dtype)
        final_mask = self._materialize_unity_padding_mask(
            encoder_padding_mask,
            batch_size=int(encoder_out.size(0)),
            seq_len=int(encoder_out.size(1)),
            device=encoder_out.device,
        )
        source_mask = self._materialize_unity_padding_mask(
            source_padding_mask,
            batch_size=int(memory.size(0)),
            seq_len=int(memory.size(1)),
            device=memory.device,
        )
        if int(memory.size(1)) != int(encoder_out.size(1)):
            target_len = int(encoder_out.size(1))
            memory = F.interpolate(
                memory.transpose(1, 2),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            source_valid = (source_mask == 0).to(dtype=torch.long)
            source_valid = self._align_1d_mask(source_valid, target_len=target_len, device=memory.device)
            source_mask = memory.new_full((memory.size(0), target_len), -torch.inf)
            source_mask = source_mask.masked_fill(source_valid.bool(), 0.0)
        memory = self.unity_source_memory_proj(memory)
        memory = self.unity_source_memory_norm(memory)
        memory = memory * self.unity_source_memory_scale
        return torch.cat([encoder_out, memory], dim=1), torch.cat([final_mask, source_mask], dim=1)

    def _unity_decode_token_ids(
        self,
        token_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        spk_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.transvip_unity is None:
            raise RuntimeError("UnitY temporal decoder requested but transvip_unity is not initialized.")
        if token_ids.ndim != 2:
            raise ValueError(f"Expected token_ids [B, T], got {tuple(token_ids.shape)}")
        if token_mask is None:
            token_mask = token_ids != int(self.text_pad_token_id)
        token_ids = token_ids.to(device=encoder_hidden.device, dtype=torch.long)
        token_mask = token_mask.to(device=encoder_hidden.device, dtype=torch.long)
        seq_lens = token_mask.sum(dim=1).clamp_min(1).to(dtype=torch.long)
        if spk_embed is not None:
            spk_embed = spk_embed.to(device=encoder_hidden.device, dtype=encoder_hidden.dtype)
        decoder_output, decoder_padding_mask = self.transvip_unity.decode(
            token_ids,
            seq_lens,
            encoder_hidden,
            self._unity_encoder_padding_mask(encoder_mask),
            spk_embed=spk_embed,
        )
        logits = self.transvip_unity.project(decoder_output, decoder_padding_mask).logits
        return decoder_output, logits

    def _text_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        ignore_index: Optional[int] = None,
    ) -> torch.Tensor:
        kwargs: Dict[str, Any] = {}
        if ignore_index is not None:
            kwargs["ignore_index"] = int(ignore_index)
        smoothing = float(self.text_label_smoothing) if self.training else 0.0
        if smoothing <= 0.0:
            return F.cross_entropy(logits, targets, **kwargs)

        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs, targets, reduction="none", **kwargs)
        text_vocab_size = min(int(self.text_vocab_size), int(logits.size(-1)))
        smooth = -log_probs[..., :text_vocab_size].mean(dim=-1)
        if ignore_index is None:
            valid = torch.ones_like(targets, dtype=torch.bool)
        else:
            valid = targets != int(ignore_index)
        if not bool(valid.any().item()):
            return logits.new_zeros(())
        return ((1.0 - smoothing) * nll[valid] + smoothing * smooth[valid]).mean()

    def _compute_unified_text_codec_losses(
        self,
        unified_logits: torch.Tensor,
        unified_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_targets = unified_targets.reshape(-1)
        valid = flat_targets != self.ignore_index
        unity_code_offset = int(self._unity_code_offset())
        unified_vocab_size = int(unified_logits.size(-1))
        if bool(valid.any().item()):
            max_target_id = int(flat_targets[valid].max().detach().cpu().item())
            if max_target_id >= unified_vocab_size:
                raise ValueError(
                    "Unified target id exceeds UnitY projection vocab: "
                    f"max_target={max_target_id}, vocab={unified_vocab_size}."
                )
        flat_logits = unified_logits.reshape(-1, unified_vocab_size)
        text_valid = valid & (flat_targets < unity_code_offset)
        code_valid = valid & (flat_targets >= unity_code_offset)
        if int(text_valid.sum().detach().cpu().item()) > 0:
            text_loss = self._text_cross_entropy(
                flat_logits[text_valid], flat_targets[text_valid]
            )
            text_valid_tokens = text_valid.sum()
        else:
            text_loss = unified_logits.new_zeros(())
            text_valid_tokens = torch.zeros((), dtype=torch.long, device=unified_logits.device)
        if int(code_valid.sum().detach().cpu().item()) > 0:
            code_loss = F.cross_entropy(flat_logits[code_valid], flat_targets[code_valid])
            code_valid_tokens = code_valid.sum()
        else:
            code_loss = unified_logits.new_zeros(())
            code_valid_tokens = torch.zeros((), dtype=torch.long, device=unified_logits.device)
        return text_loss, text_valid_tokens, code_loss, code_valid_tokens

    def _build_textcodec_unity_token_batch(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        code_attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        if text_attention_mask is None:
            text_attention_mask = (text_input_ids != self.text_pad_token_id).long()
        if code_attention_mask is None:
            code_attention_mask = (labels != self.ignore_index).any(dim=-1).long()
        bsz = labels.size(0)
        device = labels.device
        text_lengths = torch.clamp(text_attention_mask.to(device=device).sum(dim=1).long(), min=2)
        code_lengths = torch.clamp(code_attention_mask.to(device=device).sum(dim=1).long(), min=2)
        input_lengths = text_lengths + torch.clamp(code_lengths - 2, min=0)
        max_input = int(input_lengths.max().item())
        token_ids = torch.full(
            (bsz, max_input),
            int(self.text_pad_token_id),
            device=device,
            dtype=torch.long,
        )
        token_mask = torch.zeros((bsz, max_input), device=device, dtype=torch.long)
        code_start_positions: List[int] = []
        offset = self._unity_code_offset()
        for i in range(bsz):
            text_len = int(text_lengths[i].item())
            code_len = int(code_lengths[i].item())
            text_ids = text_input_ids[i, :text_len].to(device=device, dtype=torch.long)
            parts = [text_ids]
            if code_len > 2:
                code_history = labels[i, 1 : code_len - 1, 0].to(device=device, dtype=torch.long)
                valid = (code_history >= 0) & (code_history < int(self.vocab_size))
                safe_history = torch.where(valid, code_history + offset, torch.full_like(code_history, int(self.text_pad_token_id)))
                parts.append(safe_history)
            seq = torch.cat(parts, dim=0)
            token_ids[i, : seq.numel()] = seq
            token_mask[i, : seq.numel()] = 1
            # Same convention as _build_textcodec_temporal_batch: target C0 starts after text targets.
            code_start_positions.append(text_len - 1)
        return token_ids, token_mask, code_start_positions

    def _build_textcodec_unity_training_batch(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        code_attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Build UnitY token ids and shifted targets for [text, SEP, C0] AR.

        This is the unity_text_decoder counterpart of
        _build_textcodec_temporal_batch, but it intentionally does not touch
        DirectS2ST text/code embeddings. UnitY's native decoder consumes token
        ids directly: text ids keep their native ids, Mimi C0 ids are offset by
        the native text vocabulary size.
        """
        if text_attention_mask is None:
            text_attention_mask = (text_input_ids != self.text_pad_token_id).long()
        if code_attention_mask is None:
            code_attention_mask = (labels != self.ignore_index).any(dim=-1).long()
        bsz = labels.size(0)
        device = labels.device
        text_lengths = torch.clamp(text_attention_mask.to(device=device).sum(dim=1).long(), min=2)
        code_lengths = torch.clamp(code_attention_mask.to(device=device).sum(dim=1).long(), min=2)
        input_lengths = text_lengths + torch.clamp(code_lengths - 2, min=0)
        target_lengths = (text_lengths - 1) + (code_lengths - 1)
        max_input = int(input_lengths.max().item())
        max_target = int(target_lengths.max().item())
        if max_input > self.temporal_max_positions:
            raise ValueError(
                f"Text+codec AR length {max_input} exceeds temporal_max_positions={self.temporal_max_positions}."
            )
        token_ids = torch.full(
            (bsz, max_input),
            int(self.text_pad_token_id),
            device=device,
            dtype=torch.long,
        )
        token_mask = torch.zeros((bsz, max_input), device=device, dtype=torch.long)
        unified_targets = torch.full((bsz, max_target), self.ignore_index, device=device, dtype=torch.long)
        unified_target_mask = torch.zeros((bsz, max_target), device=device, dtype=torch.long)
        codec_target_mask = torch.zeros((bsz, max_target), device=device, dtype=torch.long)
        code_start_positions: List[int] = []
        offset = self._unity_code_offset()
        for i in range(bsz):
            text_len = int(text_lengths[i].item())
            code_len = int(code_lengths[i].item())
            text_ids = text_input_ids[i, :text_len].to(device=device, dtype=torch.long)
            parts = [text_ids]
            if code_len > 2:
                code_history = labels[i, 1 : code_len - 1, 0].to(device=device, dtype=torch.long)
                valid_history = (code_history >= 0) & (code_history < int(self.vocab_size))
                safe_history = torch.where(
                    valid_history,
                    code_history + offset,
                    torch.full_like(code_history, int(self.text_pad_token_id)),
                )
                parts.append(safe_history)
            seq = torch.cat(parts, dim=0)
            token_ids[i, : seq.numel()] = seq
            token_mask[i, : seq.numel()] = 1

            text_target_len = text_len - 1
            unified_targets[i, :text_target_len] = text_ids[1:]
            unified_target_mask[i, :text_target_len] = 1

            code_targets = labels[i, 1:code_len, 0].to(device=device, dtype=torch.long)
            valid_code = (code_targets != self.ignore_index) & (code_targets >= 0) & (code_targets < int(self.vocab_size))
            code_target_len = int(code_targets.numel())
            start = text_target_len
            unified_targets[i, start : start + code_target_len] = torch.where(
                valid_code,
                code_targets + offset,
                torch.full_like(code_targets, self.ignore_index),
            )
            unified_target_mask[i, start : start + code_target_len] = valid_code.long()
            codec_target_mask[i, start : start + code_target_len] = valid_code.long()
            code_start_positions.append(text_target_len)
        return token_ids, token_mask, unified_targets, unified_target_mask, codec_target_mask, code_start_positions

    def _build_aligned_textcodec_generation_token_batch(
        self,
        text_prefix_ids: torch.Tensor,
        code_prefix_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        bsz = code_prefix_labels.size(0)
        device = code_prefix_labels.device
        text_lengths = torch.clamp(
            (text_prefix_ids != self.text_pad_token_id).sum(dim=1).long(),
            min=1,
        )
        code_history_len = max(0, int(code_prefix_labels.size(1)) - 1)
        total_lengths = text_lengths + code_history_len
        max_total = int(total_lengths.max().item())
        token_ids = torch.full(
            (bsz, max_total),
            int(self.text_pad_token_id),
            device=device,
            dtype=torch.long,
        )
        token_mask = torch.zeros((bsz, max_total), device=device, dtype=torch.long)
        last_positions: List[int] = []
        offset = self._unity_code_offset()
        for i in range(bsz):
            text_len = int(text_lengths[i].item())
            parts = [text_prefix_ids[i, :text_len].to(device=device, dtype=torch.long)]
            if code_history_len > 0:
                code_history = code_prefix_labels[i, 1:, 0].to(device=device, dtype=torch.long)
                valid = (code_history >= 0) & (code_history < int(self.vocab_size))
                safe_history = torch.where(valid, code_history + offset, torch.full_like(code_history, int(self.text_pad_token_id)))
                parts.append(safe_history)
            seq = torch.cat(parts, dim=0)
            token_ids[i, : seq.numel()] = seq
            token_mask[i, : seq.numel()] = 1
            last_positions.append(int(seq.numel()) - 1)
        return token_ids, token_mask, last_positions

    def _text_next_logits(
        self,
        text_seq: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        spk_embed: Optional[torch.Tensor] = None,
        sep_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._uses_unity_native_decoder_temporal():
            token_mask = torch.ones_like(text_seq, dtype=torch.long, device=text_seq.device)
            _, unified_logits = self._unity_decode_token_ids(
                text_seq,
                token_mask,
                encoder_hidden,
                encoder_mask,
                spk_embed=spk_embed,
            )
            logits = unified_logits[:, -1, : int(self.text_vocab_size)].clone()
            generation_vocab_size = getattr(self, "text_generation_vocab_size", None)
            if (
                generation_vocab_size is not None
                and int(generation_vocab_size) < int(self.text_vocab_size)
            ):
                logits[:, int(generation_vocab_size) : int(self.text_vocab_size)] = -torch.inf
        else:
            text_inputs = self._build_temporal_text_inputs(
                text_seq,
                sep_prompt_embedding=sep_prompt_embedding,
            )
            text_len = text_inputs.size(1)
            hidden = self.temporal_transformer(
                tgt=text_inputs,
                memory=encoder_hidden,
                tgt_mask=self._causal_mask(text_len, text_inputs.device),
                tgt_key_padding_mask=None,
                memory_key_padding_mask=(encoder_mask == 0) if encoder_mask is not None else None,
            )
            hidden = self.temporal_norm(hidden)
            if self.enable_text_codec_ar:
                logits = self.text_codec_lm_head(hidden[:, -1, :])[:, : int(self.text_vocab_size)]
            else:
                logits = self.text_lm_head(hidden[:, -1, :])
        logits[:, self.text_pad_token_id] = -torch.inf
        logits[:, int(self.text_bos_token_id)] = -torch.inf
        forbidden_control_ids = list(
            getattr(self, "text_prefix_extra_token_ids_after_bos", [])
        )
        if self.enable_quality_cot:
            forbidden_control_ids.extend(
                getattr(
                    self,
                    "quality_cot_source_text_prefix_extra_token_ids_after_bos",
                    [],
                )
            )
        if self.enable_uniss_content_controls:
            forbidden_control_ids.append(int(self.uniss_start_content_token_id))
        for token_id in set(int(token_id) for token_id in forbidden_control_ids):
            if 0 <= token_id < logits.size(-1):
                logits[:, token_id] = -torch.inf
        return logits

    def _initial_text_prefix_ids(self, device: torch.device) -> torch.Tensor:
        prefix = [int(self.text_bos_token_id)]
        prefix.extend(int(token_id) for token_id in getattr(self, "text_prefix_extra_token_ids_after_bos", []))
        return torch.tensor(prefix, dtype=torch.long, device=device)

    def _initial_quality_cot_source_prefix_ids(self, device: torch.device) -> torch.Tensor:
        prefix = [int(self.text_bos_token_id)]
        prefix.extend(
            int(token_id)
            for token_id in self.quality_cot_source_text_prefix_extra_token_ids_after_bos
        )
        if self.enable_uniss_content_controls:
            prefix.append(int(self.uniss_start_content_token_id))
        return torch.tensor(prefix, dtype=torch.long, device=device)

    @torch.no_grad()
    def _generate_quality_cot_source_ids_greedy_batch(
        self,
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        max_tokens: int,
        sep_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate complete source-text prefixes for train-only translation loss.

        Generation is batched and greedy to keep the auxiliary tractable. The
        returned IDs are detached by construction; gradients flow only through
        the subsequent target-text forward pass.
        """
        if not self.enable_uniss_content_controls:
            raise RuntimeError(
                "Free-running source generation requires START/END content controls."
            )
        device = encoder_hidden.device
        batch_size = int(encoder_hidden.size(0))
        prefix = self._initial_quality_cot_source_prefix_ids(device)
        prefix_len = int(prefix.numel())
        max_tokens = int(max_tokens)
        if max_tokens < prefix_len + 1:
            raise ValueError(
                "predicted_source_target_max_source_tokens must reserve END_CONTENT: "
                f"max_tokens={max_tokens}, prefix_len={prefix_len}."
            )

        sequences = prefix.unsqueeze(0).expand(batch_size, -1).clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        generated_content = torch.zeros(batch_size, dtype=torch.long, device=device)
        end_id = int(self.uniss_end_content_token_id)
        sep_id = int(self.text_sep_token_id)
        pad_id = int(self.text_pad_token_id)
        min_content_tokens = max(0, int(self.text_prefix_min_tokens))
        no_repeat_ngram_size = max(
            0,
            int(self.text_prefix_no_repeat_ngram_size),
        )
        max_generation_steps = max_tokens - prefix_len

        for generation_step in range(max_generation_steps):
            logits = self._text_next_logits(
                sequences,
                encoder_hidden,
                encoder_mask,
                sep_prompt_embedding=sep_prompt_embedding,
            ).clone()
            logits[:, sep_id] = -torch.inf
            active = ~finished
            too_short = active & (generated_content < min_content_tokens)
            logits[too_short, end_id] = -torch.inf

            if no_repeat_ngram_size > 0:
                for sample_idx in torch.nonzero(active, as_tuple=False).flatten().tolist():
                    seq_ids = [
                        int(token_id)
                        for token_id in sequences[sample_idx].detach().cpu().tolist()
                    ]
                    banned_ids = set()
                    if no_repeat_ngram_size == 1:
                        banned_ids.update(seq_ids)
                    elif len(seq_ids) >= no_repeat_ngram_size - 1:
                        suffix = tuple(seq_ids[-(no_repeat_ngram_size - 1) :])
                        for start in range(
                            0,
                            len(seq_ids) - no_repeat_ngram_size + 1,
                        ):
                            if (
                                tuple(
                                    seq_ids[
                                        start : start + no_repeat_ngram_size - 1
                                    ]
                                )
                                == suffix
                            ):
                                banned_ids.add(
                                    seq_ids[start + no_repeat_ngram_size - 1]
                                )
                    if banned_ids:
                        logits[sample_idx, list(banned_ids)] = -torch.inf

            if generation_step == max_generation_steps - 1:
                next_tokens = torch.full(
                    (batch_size,),
                    end_id,
                    dtype=torch.long,
                    device=device,
                )
            else:
                next_tokens = logits.argmax(dim=-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, pad_id),
                next_tokens,
            )
            sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
            newly_finished = active & (next_tokens == end_id)
            lengths[newly_finished] = int(sequences.size(1))
            generated_content = generated_content + (
                active & ~newly_finished
            ).long()
            finished = finished | newly_finished
            if bool(finished.all().item()):
                break

        if not bool(finished.all().item()):
            raise RuntimeError("Failed to terminate every predicted source sequence.")
        max_length = int(lengths.max().item())
        predicted_ids = torch.full(
            (batch_size, max_length),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        predicted_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
            device=device,
        )
        for sample_idx, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            predicted_ids[sample_idx, :length] = sequences[sample_idx, :length]
            predicted_mask[sample_idx, :length] = 1
        mean_content_tokens = generated_content.to(torch.float32).mean()
        return predicted_ids, predicted_mask, mean_content_tokens

    def _generated_text_content_count(self, text_seq: torch.Tensor) -> int:
        return max(
            0,
            int(text_seq.numel()) - 1 - len(getattr(self, "text_prefix_extra_token_ids_after_bos", [])),
        )

    def _would_repeat_text_prefix_ngram(self, text_seq: torch.Tensor, next_token_id: int) -> bool:
        ngram_size = int(getattr(self, "text_prefix_no_repeat_ngram_size", 0))
        if ngram_size <= 0 or int(text_seq.numel()) + 1 < ngram_size:
            return False
        seq_ids = [int(x) for x in text_seq.detach().cpu().tolist()]
        candidate = tuple(seq_ids[-(ngram_size - 1) :] + [int(next_token_id)])
        for start in range(0, len(seq_ids) - ngram_size + 1):
            if tuple(seq_ids[start : start + ngram_size]) == candidate:
                return True
        return False

    @torch.no_grad()
    def _generate_one_text_prefix_ids_beam(
        self,
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        max_tokens: int,
        spk_embed: Optional[torch.Tensor] = None,
        sep_prompt_embedding: Optional[torch.Tensor] = None,
        initial_prefix_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        stop_token_id: Optional[int] = None,
        return_diagnostics: bool = False,
    ) -> Any:
        device = encoder_hidden.device
        sep_id = int(
            self.text_sep_token_id if stop_token_id is None else stop_token_id
        )
        min_text_tokens = max(0, int(self.text_prefix_min_tokens))
        sep_penalty = max(0.0, float(self.text_prefix_sep_penalty))
        length_penalty = max(1e-6, float(self.text_prefix_length_penalty))

        def rank_score(score: float, seq_len: int) -> float:
            return score / (max(1, seq_len) ** length_penalty)

        initial_prefix = (
            self._initial_text_prefix_ids(device)
            if initial_prefix_ids is None
            else initial_prefix_ids.to(device=device, dtype=torch.long).flatten()
        )
        if initial_prefix.numel() == 0:
            raise ValueError("Text beam prefix must contain at least one token.")
        beams: List[Tuple[torch.Tensor, float, bool]] = [(initial_prefix, 0.0, False)]
        finalized: List[Tuple[float, torch.Tensor, bool]] = []
        beam_size = max(1, int(self.text_prefix_beam_size))
        initial_len = max(1, int(initial_prefix.numel()))
        max_tokens = int(max_tokens)
        if max_tokens < initial_len + 1:
            raise ValueError(
                "Text beam budget must reserve one terminal SEP: "
                f"max_tokens={max_tokens}, initial_len={initial_len}."
            )
        # Reserve the final SEP. Without this, an unfinished beam can reach the
        # nominal limit and then append SEP one position beyond the text table.
        max_generation_steps = max_tokens - initial_len - 1
        generation_steps = (
            min(max_generation_steps, max(0, int(max_new_tokens)))
            if max_new_tokens is not None
            else max_generation_steps
        )
        for _ in range(generation_steps):
            active = [(seq, score, done) for seq, score, done in beams if not done]
            for seq, score, done in beams:
                if done:
                    finalized.append((rank_score(score, int(seq.numel())), seq, True))
            if not active:
                break
            seq_batch = torch.stack([seq for seq, _, _ in active], dim=0)
            enc = encoder_hidden.expand(seq_batch.size(0), -1, -1)
            enc_mask = encoder_mask.expand(seq_batch.size(0), -1) if encoder_mask is not None else None
            spk = spk_embed.expand(seq_batch.size(0), -1) if spk_embed is not None else None
            log_probs = F.log_softmax(
                self._text_next_logits(
                    seq_batch,
                    enc,
                    enc_mask,
                    spk,
                    sep_prompt_embedding=(
                        sep_prompt_embedding.expand(seq_batch.size(0), -1)
                        if sep_prompt_embedding is not None
                        else None
                    ),
                ),
                dim=-1,
            )
            if (
                stop_token_id is not None
                and int(stop_token_id) != int(self.text_sep_token_id)
            ):
                # START/END-controlled text stages must not terminate on the
                # codec-boundary SEP. That SEP is appended only after target END.
                log_probs[:, int(self.text_sep_token_id)] = -torch.inf
            candidates: List[Tuple[float, torch.Tensor, bool]] = []
            top_scores, top_ids = torch.topk(log_probs, k=min(beam_size, log_probs.size(-1)), dim=-1)
            for i, (seq, score, _) in enumerate(active):
                for token_score, token_id in zip(top_scores[i], top_ids[i]):
                    tok = int(token_id.item())
                    text_tokens_so_far = max(0, int(seq.numel()) - initial_len)
                    if tok == sep_id and text_tokens_so_far < min_text_tokens:
                        continue
                    if tok != sep_id and self._would_repeat_text_prefix_ngram(seq, tok):
                        continue
                    new_seq = torch.cat([seq, token_id.view(1)])
                    new_score = float(score) + float(token_score.item())
                    if tok == sep_id:
                        new_score -= sep_penalty
                    candidates.append((new_score, new_seq, tok == sep_id))
            if not candidates:
                # If all candidates were early SEP, continue with the best non-SEP
                # tokens from a wider local search rather than ending immediately.
                wider_k = min(max(beam_size * 4, beam_size + 1), log_probs.size(-1))
                top_scores, top_ids = torch.topk(log_probs, k=wider_k, dim=-1)
                for i, (seq, score, _) in enumerate(active):
                    for token_score, token_id in zip(top_scores[i], top_ids[i]):
                        tok = int(token_id.item())
                        if tok == sep_id:
                            continue
                        if self._would_repeat_text_prefix_ngram(seq, tok):
                            continue
                        new_seq = torch.cat([seq, token_id.view(1)])
                        new_score = float(score) + float(token_score.item())
                        candidates.append((new_score, new_seq, False))
                if not candidates:
                    for i, (seq, score, _) in enumerate(active):
                        for token_score, token_id in zip(top_scores[i], top_ids[i]):
                            tok = int(token_id.item())
                            if tok == sep_id:
                                continue
                            new_seq = torch.cat([seq, token_id.view(1)])
                            new_score = float(score) + float(token_score.item())
                            candidates.append((new_score, new_seq, False))
                            break
            candidates.sort(key=lambda item: rank_score(item[0], int(item[1].numel())), reverse=True)
            beams = [(seq, score, done) for score, seq, done in candidates[:beam_size]]
            if all(done for _, _, done in beams):
                break
        for seq, score, done in beams:
            if not done:
                seq = torch.cat([seq, torch.tensor([sep_id], dtype=torch.long, device=device)])
            finalized.append((rank_score(score, int(seq.numel())), seq, bool(done)))
        if not finalized:
            sequence = torch.cat(
                [
                    initial_prefix,
                    torch.tensor([sep_id], dtype=torch.long, device=device),
                ]
            )
            if return_diagnostics:
                return sequence, {
                    "natural_end": False,
                    "forced_end": True,
                    "generated_tokens": 0,
                    "rank_score": float("-inf"),
                }
            return sequence
        best_rank_score, best_sequence, best_natural_end = max(
            finalized,
            key=lambda item: item[0],
        )
        if return_diagnostics:
            return best_sequence, {
                "natural_end": bool(best_natural_end),
                "forced_end": not bool(best_natural_end),
                "generated_tokens": max(
                    0,
                    int(best_sequence.numel()) - int(initial_prefix.numel()) - 1,
                ),
                "rank_score": float(best_rank_score),
            }
        return best_sequence

    @torch.no_grad()
    def generate_text_prefix_ids(
        self,
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        max_tokens: Optional[int] = None,
        spk_embed: Optional[torch.Tensor] = None,
        sep_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.text_bos_token_id is None or self.text_sep_token_id is None:
            raise ValueError("text_bos_token_id/text_sep_token_id must be set for text-prefix AR.")
        seqs = []
        for i in range(encoder_hidden.size(0)):
            sample_encoder_hidden = encoder_hidden[i : i + 1]
            sample_encoder_mask = (
                encoder_mask[i : i + 1] if encoder_mask is not None else None
            )
            sample_spk = spk_embed[i : i + 1] if spk_embed is not None else None
            sample_sep_prompt = (
                sep_prompt_embedding[i : i + 1]
                if sep_prompt_embedding is not None
                else None
            )
            if self.enable_quality_cot:
                source_prefix = self._initial_quality_cot_source_prefix_ids(
                    sample_encoder_hidden.device
                )
                source_new_tokens = max(
                    0,
                    int(self.quality_cot_source_text_max_tokens) - int(source_prefix.numel()),
                )
                source_sequence = self._generate_one_text_prefix_ids_beam(
                    encoder_hidden=sample_encoder_hidden,
                    encoder_mask=sample_encoder_mask,
                    max_tokens=int(self.quality_cot_source_text_max_tokens),
                    spk_embed=sample_spk,
                    sep_prompt_embedding=sample_sep_prompt,
                    initial_prefix_ids=source_prefix,
                    max_new_tokens=source_new_tokens,
                    stop_token_id=(
                        self.uniss_end_content_token_id
                        if self.enable_uniss_content_controls
                        else None
                    ),
                )
                target_controls = torch.tensor(
                    [
                        int(token_id)
                        for token_id in self.text_prefix_extra_token_ids_after_bos
                    ],
                    dtype=torch.long,
                    device=sample_encoder_hidden.device,
                )
                if self.enable_uniss_content_controls:
                    target_controls = torch.cat(
                        [
                            target_controls,
                            torch.tensor(
                                [int(self.uniss_start_content_token_id)],
                                dtype=torch.long,
                                device=sample_encoder_hidden.device,
                            ),
                        ]
                    )
                target_prefix = torch.cat([source_sequence, target_controls], dim=0)
                target_max_tokens = max(
                    2,
                    int(max_tokens or self.quality_cot_target_text_max_tokens),
                )
                target_new_tokens = max(
                    0,
                    target_max_tokens
                    - (2 if self.enable_uniss_content_controls else 1)
                    - len(self.text_prefix_extra_token_ids_after_bos),
                )
                generated_target = self._generate_one_text_prefix_ids_beam(
                        encoder_hidden=sample_encoder_hidden,
                        encoder_mask=sample_encoder_mask,
                        max_tokens=int(target_prefix.numel()) + target_new_tokens,
                        spk_embed=sample_spk,
                        sep_prompt_embedding=sample_sep_prompt,
                        initial_prefix_ids=target_prefix,
                        max_new_tokens=target_new_tokens,
                        stop_token_id=(
                            self.uniss_end_content_token_id
                            if self.enable_uniss_content_controls
                            else None
                        ),
                    )
                if self.enable_uniss_content_controls:
                    generated_target = torch.cat(
                        [
                            generated_target,
                            torch.tensor(
                                [int(self.text_sep_token_id)],
                                dtype=torch.long,
                                device=sample_encoder_hidden.device,
                            ),
                        ]
                    )
                seqs.append(generated_target)
            else:
                segment_max_tokens = max(
                    2,
                    min(
                        int(max_tokens or self.text_max_positions),
                        self.text_max_positions,
                    ),
                )
                seqs.append(
                    self._generate_one_text_prefix_ids_beam(
                        encoder_hidden=sample_encoder_hidden,
                        encoder_mask=sample_encoder_mask,
                        max_tokens=segment_max_tokens,
                        spk_embed=sample_spk,
                    sep_prompt_embedding=sample_sep_prompt,
                    )
                )
        max_len = max(seq.numel() for seq in seqs)
        out = torch.full(
            (len(seqs), max_len),
            fill_value=self.text_pad_token_id,
            dtype=torch.long,
            device=encoder_hidden.device,
        )
        for i, seq in enumerate(seqs):
            out[i, : seq.numel()] = seq
        return out

    def _encode_source_inputs(
        self,
        input_features: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        transvip_fbank_seq_lens: Optional[torch.Tensor] = None,
        transvip_target_lengths: Optional[torch.Tensor] = None,
        transvip_vad_mask: Optional[torch.Tensor] = None,
        transvip_prompt_wavs: Optional[torch.Tensor] = None,
        transvip_prompt_wav_lens: Optional[torch.Tensor] = None,
        return_spk_embed: bool = False,
    ):
        # These tensors are batch-local. Reset them before every source encode
        # so a missing condition can never reuse the previous batch's prompt.
        self._runtime_w2vbert_prompt_frames = None
        self._runtime_w2vbert_prompt_mask = None
        self._runtime_mimi_prompt_frames = None
        self._runtime_mimi_prompt_mask = None
        if self.speech_encoder_type == "transvip_unity":
            if self.transvip_unity is None:
                raise RuntimeError("TransVIP/UnitY source encoder is not initialized.")
            if transvip_fbank_seq_lens is None:
                if audio_attention_mask is not None:
                    transvip_fbank_seq_lens = audio_attention_mask.sum(dim=1).long()
                else:
                    transvip_fbank_seq_lens = torch.full(
                        (input_features.size(0),),
                        input_features.size(1),
                        dtype=torch.long,
                        device=input_features.device,
                    )
            unity_param = next(self.transvip_unity.parameters())
            unity_device = unity_param.device
            input_features = input_features.to(device=unity_device, dtype=torch.float32)
            transvip_fbank_seq_lens = transvip_fbank_seq_lens.to(
                device=unity_device,
                dtype=torch.long,
            )
            target_lengths = None
            vad_mask = None
            if self.transvip_use_length_control and transvip_target_lengths is not None:
                target_lengths = transvip_target_lengths.to(
                    device=unity_device,
                    dtype=torch.long,
                )
                if transvip_vad_mask is not None:
                    vad_mask = transvip_vad_mask.to(
                        device=unity_device,
                        dtype=torch.long,
                    )
            if self._uses_unity_native_decoder_temporal():
                for _name in ("speech_encoder_frontend", "speech_encoder", "length_control_module"):
                    _module = getattr(self.transvip_unity, _name, None)
                    if _module is not None:
                        _module.eval()
            else:
                self.transvip_unity.eval()
            autocast_context = (
                torch.amp.autocast(device_type=input_features.device.type, enabled=False)
                if input_features.is_cuda
                else nullcontext()
            )
            prompts, prompt_lens = self._build_transvip_source_speaker_prompt(
                transvip_prompt_wavs,
                transvip_prompt_wav_lens,
            )
            source_memory = None
            source_padding_mask = None
            with torch.no_grad(), autocast_context:
                self.transvip_unity.input_modality = "speech"
                if bool(getattr(self, "enable_unity_source_memory_fusion", False)):
                    seqs, padding_mask0 = self.transvip_unity.speech_encoder_frontend(
                        input_features,
                        transvip_fbank_seq_lens,
                    )
                    captured: Dict[str, Any] = {}

                    def _capture_source_layer(layer_idx, layer_output, layer_padding_mask, num_layers):
                        if int(layer_idx) == int(self.unity_source_memory_layer) - 1:
                            captured["inner"] = layer_output
                            captured["inner_mask"] = layer_padding_mask
                            captured["num_layers"] = int(num_layers)

                    encoder_out, padding_mask = self.transvip_unity.speech_encoder(
                        seqs,
                        padding_mask0,
                        _capture_source_layer if self.unity_source_memory_source == "inner" else None,
                    )
                    if self.unity_source_memory_source == "inner":
                        if "inner" not in captured:
                            raise RuntimeError(
                                "Could not capture Unity speech encoder layer "
                                f"{self.unity_source_memory_layer}. Check the configured layer index."
                            )
                        source_memory = captured["inner"]
                        source_padding_mask = captured.get("inner_mask")
                    else:
                        source_memory = encoder_out
                        source_padding_mask = padding_mask
                    if self.transvip_use_length_control and target_lengths is not None:
                        encoder_out, padding_mask = self.transvip_unity.length_control_module(
                            encoder_out,
                            padding_mask,
                            target_lengths,
                            vad_mask,
                        )
                    if prompts is not None:
                        unity_spk_embed = self.transvip_unity.spk_encoder(prompts, prompt_lens)
                        unity_spk_embed = self.transvip_unity.transform(unity_spk_embed)
                    else:
                        unity_spk_embed = None
                else:
                    encoder_out, padding_mask, unity_spk_embed = self.transvip_unity.encode(
                        input_features,
                        transvip_fbank_seq_lens,
                        target_length=target_lengths,
                        vad_mask=vad_mask,
                        prompts=prompts,
                        prompt_lens=prompt_lens,
                    )
            encoder_out = encoder_out.detach()
            if source_memory is not None:
                source_memory = source_memory.detach()
            if unity_spk_embed is not None:
                unity_spk_embed = unity_spk_embed.detach()
            if bool(getattr(self, "enable_unity_source_memory_fusion", False)):
                encoder_out, padding_mask = self._apply_unity_source_memory_fusion(
                    encoder_out,
                    padding_mask,
                    source_memory,
                    source_padding_mask,
                )
            if padding_mask is None:
                raw_encoder_mask = torch.ones(
                    (encoder_out.size(0), encoder_out.size(1)),
                    dtype=torch.long,
                    device=encoder_out.device,
                )
            else:
                if hasattr(padding_mask, "materialize"):
                    padding_mask = padding_mask.materialize()
                padding_mask = padding_mask.to(device=encoder_out.device)
                raw_encoder_mask = (padding_mask == 0).long()
        elif self.speech_encoder_type == "whisper":
            encoder_outputs = self.whisper.encoder(
                input_features=input_features,
                attention_mask=audio_attention_mask,
                return_dict=True,
                output_hidden_states=self.speech_encoder_output_layer is not None,
            )
            encoder_out = self._select_encoder_hidden(
                encoder_outputs,
                self.speech_encoder_output_layer,
            )
            raw_encoder_mask = self._align_1d_mask(
                audio_attention_mask,
                target_len=encoder_out.size(1),
                device=encoder_out.device,
            )
        else:
            needs_w2vbert_prompt_hidden = bool(
                self.enable_mimi_source_speaker_prompt
                and self.source_speaker_prompt_backend == "w2vbert_hidden"
            )
            encoder_outputs = self.whisper(
                input_features=input_features,
                attention_mask=audio_attention_mask,
                return_dict=True,
                output_hidden_states=(
                    self.speech_encoder_output_layer is not None
                    or needs_w2vbert_prompt_hidden
                ),
            )
            encoder_out = self._select_encoder_hidden(
                encoder_outputs,
                self.speech_encoder_output_layer,
            )
            raw_encoder_mask = self._align_1d_mask(
                audio_attention_mask,
                target_len=encoder_out.size(1),
                device=encoder_out.device,
            )
            if needs_w2vbert_prompt_hidden:
                prompt_frames = self._select_encoder_hidden(
                    encoder_outputs,
                    self.source_speaker_prompt_speech_layer,
                )
                prompt_mask = self._align_1d_mask(
                    audio_attention_mask,
                    target_len=prompt_frames.size(1),
                    device=prompt_frames.device,
                )
                if prompt_mask is None:
                    prompt_mask = torch.ones(
                        prompt_frames.shape[:2],
                        dtype=torch.long,
                        device=prompt_frames.device,
                    )
                self._runtime_w2vbert_prompt_frames = (
                    prompt_frames.detach()
                    if self.mimi_source_speaker_prompt_detach
                    else prompt_frames
                )
                self._runtime_w2vbert_prompt_mask = prompt_mask.detach()

        if self._uses_unity_native_decoder_temporal() and self.speech_encoder_type == "transvip_unity":
            encoder_hidden = encoder_out
            encoder_mask = self._align_1d_mask(
                raw_encoder_mask,
                target_len=encoder_hidden.size(1),
                device=encoder_hidden.device,
            )
            if return_spk_embed:
                return encoder_hidden, encoder_mask, unity_spk_embed
            return encoder_hidden, encoder_mask

        encoder_hidden = self.encoder_to_temporal(encoder_out)
        encoder_mask = self._align_1d_mask(
            raw_encoder_mask,
            target_len=encoder_hidden.size(1),
            device=encoder_hidden.device,
        )
        if self.source_adapter is not None:
            adapted_encoder_hidden = self.source_adapter(
                encoder_hidden,
                src_key_padding_mask=(encoder_mask == 0) if encoder_mask is not None else None,
            )
            if self.source_adapter_residual:
                encoder_hidden = encoder_hidden + self.source_adapter_residual_scale * adapted_encoder_hidden
            else:
                encoder_hidden = adapted_encoder_hidden
            encoder_hidden = self.source_adapter_norm(encoder_hidden)
        if return_spk_embed:
            return encoder_hidden, encoder_mask, None
        return encoder_hidden, encoder_mask

    def forward(
        self,
        input_features: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,  # [B, T, Q]
        code_attention_mask: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        source_text_input_ids: Optional[torch.Tensor] = None,
        source_text_attention_mask: Optional[torch.Tensor] = None,
        source_unit_ids: Optional[torch.Tensor] = None,
        source_unit_attention_mask: Optional[torch.Tensor] = None,
        transvip_fbank_seq_lens: Optional[torch.Tensor] = None,
        transvip_target_lengths: Optional[torch.Tensor] = None,
        transvip_vad_mask: Optional[torch.Tensor] = None,
        transvip_prompt_wavs: Optional[torch.Tensor] = None,
        transvip_prompt_wav_lens: Optional[torch.Tensor] = None,
        source_acoustic_wavs: Optional[torch.Tensor] = None,
        source_acoustic_wav_lens: Optional[torch.Tensor] = None,
        source_acoustic_embeddings: Optional[torch.Tensor] = None,
        campplus_source_embeddings: Optional[torch.Tensor] = None,
        campplus_target_embeddings: Optional[torch.Tensor] = None,
        c0_teacher_topk_ids: Optional[torch.Tensor] = None,
        c0_teacher_topk_log_probs: Optional[torch.Tensor] = None,
        c0_teacher_hidden: Optional[torch.Tensor] = None,
        c0_teacher_hidden_mask: Optional[torch.Tensor] = None,
        return_codec_temporal_hidden: bool = False,
        depth_teacher_force: Optional[bool] = None,
        depth_force_first_level0: bool = False,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        # ========== Speech Encoder ==========
        with torch.profiler.record_function("s2st/source_speech_encoder"):
            encoder_hidden, encoder_mask, unity_spk_embed = self._encode_source_inputs(
                input_features=input_features,
                audio_attention_mask=audio_attention_mask,
                transvip_fbank_seq_lens=transvip_fbank_seq_lens,
                transvip_target_lengths=transvip_target_lengths,
                transvip_vad_mask=transvip_vad_mask,
                transvip_prompt_wavs=transvip_prompt_wavs,
                transvip_prompt_wav_lens=transvip_prompt_wav_lens,
                return_spk_embed=True,
            )
        source_acoustic_embedding = source_acoustic_embeddings
        if (
            self.enable_mimi_source_style_token_bank
            or self.enable_mimi_nar_depth_prompt
            or self._source_speaker_prompt_uses_mimi_frames()
        ):
            # Keep the precomputed global mean/std vector for the SEP prompt,
            # while deriving ordered local/style or discrete NAR prompt tokens
            # from the same source wav.
            with torch.profiler.record_function("s2st/source_mimi_prompt"):
                source_acoustic_embedding = self._encode_mimi_source_acoustic_embedding(
                    source_acoustic_wavs,
                    source_acoustic_wav_lens,
                    pooled_override=source_acoustic_embedding,
                )
        elif source_acoustic_embedding is not None:
            source_acoustic_embedding = source_acoustic_embedding.to(
                device=encoder_hidden.device,
                dtype=encoder_hidden.dtype,
            )
            if self.mimi_source_acoustic_detach:
                source_acoustic_embedding = source_acoustic_embedding.detach()
        else:
            source_acoustic_embedding = self._encode_mimi_source_acoustic_embedding(
                source_acoustic_wavs,
                source_acoustic_wav_lens,
            )
        mimi_spk_embed = self._build_mimi_source_speaker_prompt_embedding(
            source_acoustic_embedding,
            device=encoder_hidden.device,
            dtype=encoder_hidden.dtype,
        )
        if mimi_spk_embed is not None:
            unity_spk_embed = mimi_spk_embed
        with torch.profiler.record_function("s2st/source_campplus"):
            if campplus_source_embeddings is not None:
                if not self.enable_campplus_depth_conditioning:
                    raise RuntimeError(
                        "Received precomputed CAMPPlus embeddings while CAMPPlus "
                        "conditioning is disabled."
                    )
                campplus_source_embedding = campplus_source_embeddings.to(
                    device=encoder_hidden.device,
                    dtype=torch.float32,
                ).detach()
                if (
                    campplus_source_embedding.ndim != 2
                    or int(campplus_source_embedding.size(-1)) != self.campplus_embedding_size
                ):
                    raise RuntimeError(
                        "Precomputed CAMPPlus embeddings must be [B, "
                        f"{self.campplus_embedding_size}], got "
                        f"{tuple(campplus_source_embedding.shape)}."
                    )
                campplus_input_mode = "precomputed"
            else:
                campplus_source_embedding = self._encode_campplus_source_embedding(
                    source_acoustic_wavs,
                    source_acoustic_wav_lens,
                )
                campplus_input_mode = "online_batched"
        if (
            self.enable_campplus_depth_conditioning
            and not getattr(self, "_campplus_input_mode_logged", False)
        ):
            print(
                "[CAMPPlusInput]",
                {
                    "mode": campplus_input_mode,
                    "batch_size": int(encoder_hidden.size(0)),
                },
            )
            self._campplus_input_mode_logged = True
        self.depth_decoder.set_runtime_campplus_source_embedding(
            campplus_source_embedding
        )
        campplus_identity_embedding = campplus_source_embedding
        if self.campplus_identity_supervision == "target":
            if campplus_target_embeddings is not None:
                campplus_identity_embedding = campplus_target_embeddings.to(
                    device=encoder_hidden.device,
                    dtype=torch.float32,
                ).detach()
                if (
                    campplus_identity_embedding.ndim != 2
                    or int(campplus_identity_embedding.size(-1))
                    != self.campplus_embedding_size
                ):
                    raise RuntimeError(
                        "Precomputed target CAMPPlus embeddings must be [B, "
                        f"{self.campplus_embedding_size}], got "
                        f"{tuple(campplus_identity_embedding.shape)}."
                    )
            elif (
                self.training
                and labels is not None
                and self.campplus_depth_identity_loss_weight > 0.0
            ):
                raise RuntimeError(
                    "Target CAMPPlus identity supervision is enabled, but the "
                    "training batch has no campplus_target_embeddings."
                )
            else:
                campplus_identity_embedding = None

        # Source-unit auxiliaries are training-only. CTC anchors monotonic
        # frame/unit alignment; the AR decoder retrieves the full source-unit
        # sequence from shared speech memory through cross-attention.
        source_unit_loss = None
        source_unit_transformer_loss = None
        source_unit_transformer_error_rate = None
        source_unit_valid_tokens = None
        source_aux_enabled = (
            self.enable_source_unit_auxiliary
            or self.enable_source_unit_transformer_auxiliary
        )
        if source_aux_enabled and source_unit_ids is not None:
            if source_unit_attention_mask is None:
                source_unit_attention_mask = torch.ones_like(source_unit_ids)
            source_unit_attention_mask = source_unit_attention_mask.to(
                device=encoder_hidden.device,
                dtype=torch.long,
            )
            source_unit_ids = source_unit_ids.to(device=encoder_hidden.device, dtype=torch.long)
            valid_source_units = source_unit_attention_mask.bool()
            source_unit_valid_tokens = valid_source_units.sum()
            target_lengths = valid_source_units.sum(dim=1).long()

            if self.enable_source_unit_auxiliary:
                input_lengths = (
                    encoder_mask.sum(dim=1).long()
                    if encoder_mask is not None
                    else torch.full(
                        (encoder_hidden.size(0),),
                        encoder_hidden.size(1),
                        device=encoder_hidden.device,
                        dtype=torch.long,
                    )
                )
                if bool((target_lengths > input_lengths).any().item()):
                    raise ValueError(
                        "Reduced source semantic sequence is longer than the speech encoder sequence."
                    )
                source_logits = self.source_unit_head(encoder_hidden)
                source_log_probs = F.log_softmax(source_logits, dim=-1).transpose(0, 1)
                flat_source_targets = source_unit_ids.masked_select(valid_source_units)
                source_unit_loss = F.ctc_loss(
                    source_log_probs,
                    flat_source_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.source_unit_blank_id,
                    reduction="mean",
                    zero_infinity=True,
                )

            if self.enable_source_unit_transformer_auxiliary:
                max_units = int(target_lengths.max().item())
                decoder_len = max_units + 1  # BOS+units input predicts units+EOS.
                if decoder_len > self.source_unit_transformer_max_positions:
                    raise ValueError(
                        f"Source-unit AR length {decoder_len} exceeds "
                        "source_unit_transformer_max_positions="
                        f"{self.source_unit_transformer_max_positions}."
                    )
                bsz = source_unit_ids.size(0)
                source_decoder_ids = torch.full(
                    (bsz, decoder_len),
                    fill_value=self.source_unit_pad_id,
                    dtype=torch.long,
                    device=encoder_hidden.device,
                )
                source_targets = torch.full(
                    (bsz, decoder_len),
                    fill_value=self.ignore_index,
                    dtype=torch.long,
                    device=encoder_hidden.device,
                )
                source_decoder_mask = torch.zeros(
                    (bsz, decoder_len),
                    dtype=torch.long,
                    device=encoder_hidden.device,
                )
                for sample_idx, unit_len_tensor in enumerate(target_lengths):
                    unit_len = int(unit_len_tensor.item())
                    source_decoder_ids[sample_idx, 0] = self.source_unit_bos_id
                    source_decoder_mask[sample_idx, : unit_len + 1] = 1
                    if unit_len > 0:
                        units = source_unit_ids[sample_idx, :unit_len]
                        source_decoder_ids[sample_idx, 1 : unit_len + 1] = units
                        source_targets[sample_idx, :unit_len] = units
                    source_targets[sample_idx, unit_len] = self.source_unit_eos_id

                positions = torch.arange(decoder_len, device=encoder_hidden.device)
                source_decoder_inputs = self.source_unit_transformer_emb(source_decoder_ids)
                source_decoder_inputs = (
                    source_decoder_inputs
                    + self.source_unit_transformer_pos_emb(positions).unsqueeze(0)
                )
                source_hidden = self.source_unit_transformer_decoder(
                    tgt=source_decoder_inputs,
                    memory=encoder_hidden,
                    tgt_mask=torch.triu(
                        torch.ones(
                            (decoder_len, decoder_len),
                            dtype=torch.bool,
                            device=encoder_hidden.device,
                        ),
                        diagonal=1,
                    ),
                    tgt_key_padding_mask=(source_decoder_mask == 0),
                    memory_key_padding_mask=(encoder_mask == 0) if encoder_mask is not None else None,
                )
                source_hidden = self.source_unit_transformer_norm(source_hidden)
                source_ar_logits = self.source_unit_transformer_head(source_hidden)
                source_unit_transformer_loss = F.cross_entropy(
                    source_ar_logits.reshape(-1, self.source_unit_transformer_vocab_size),
                    source_targets.reshape(-1),
                    ignore_index=self.ignore_index,
                )
                source_ar_valid = source_targets != self.ignore_index
                source_ar_errors = (
                    source_ar_logits.argmax(dim=-1) != source_targets
                ) & source_ar_valid
                source_unit_transformer_error_rate = (
                    source_ar_errors.sum().to(source_ar_logits.dtype)
                    / source_ar_valid.sum().clamp_min(1).to(source_ar_logits.dtype)
                )

        # Inject source acoustic identity/prosody into the memory consumed by
        # the target text/C0 temporal decoder. Source-unit auxiliaries above keep
        # using the raw speech encoder states.
        encoder_hidden = self._apply_mimi_source_acoustic_temporal_conditioning(
            encoder_hidden,
            source_acoustic_embedding,
        )

        # ========== Target-Text/Semantic Auxiliary Decoder ==========
        effective_temporal_scheduled_sampling_prob = 0.0
        text_prefix_mode = self.enable_text_prefix_ar and text_input_ids is not None and labels is not None
        full_code_attention_mask = code_attention_mask
        text_loss = None
        text_valid_tokens = None
        target_semantic_error_rate = None
        target_semantic_content_loss = None
        target_semantic_content_error_rate = None
        target_semantic_content_valid_tokens = None
        target_semantic_sep_loss = None
        target_semantic_sep_error_rate = None
        target_semantic_sep_valid_tokens = None
        target_semantic_first_token_loss = None
        target_semantic_first_token_error_rate = None
        target_semantic_first_token_valid_tokens = None
        # ========== Temporal Decoder ==========
        target_labels = None
        target_mask = None
        code_start_positions: Optional[List[int]] = None
        text_targets = None
        text_target_mask = None
        unified_targets = None
        unified_target_mask = None
        codec_unified_target_mask = None
        quality_cot_source_text_target_mask = None
        quality_cot_target_text_target_mask = None
        unity_token_ids = None
        unity_token_mask = None
        unity_direct_token_ids = None
        # Gold AR inputs are reused by the training-only UnitY text-memory path.
        text_path_temporal_inputs = None
        if text_prefix_mode and self.enable_text_codec_ar:
            if labels.size(1) < 2:
                raise ValueError("Text+codec AR requires code labels with at least BOS and one target row.")
            target_labels = labels[:, 1:, : self.num_codebook_levels].clone()
            target_mask = code_attention_mask[:, 1:] if code_attention_mask is not None else None
            if self._uses_unity_text_decoder_temporal():
                (
                    unity_token_ids,
                    unity_token_mask,
                    unified_targets,
                    unified_target_mask,
                    codec_unified_target_mask,
                    code_start_positions,
                ) = self._build_textcodec_unity_training_batch(
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    labels=labels[:, :, : self.input_num_quantizers],
                    code_attention_mask=code_attention_mask,
                )
                temporal_inputs = None
                temporal_mask = unity_token_mask
                target_len = int(unity_token_ids.size(1))
            else:
                if self.enable_quality_cot:
                    if source_text_input_ids is None:
                        raise RuntimeError(
                            "enable_quality_cot=True requires source_text_input_ids in every batch."
                        )
                    (
                        temporal_inputs,
                        temporal_mask,
                        unified_targets,
                        unified_target_mask,
                        codec_unified_target_mask,
                        code_start_positions,
                        quality_cot_source_text_target_mask,
                        quality_cot_target_text_target_mask,
                    ) = self._build_quality_cot_textcodec_temporal_batch(
                        source_text_input_ids=source_text_input_ids,
                        source_text_attention_mask=source_text_attention_mask,
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask,
                        labels=labels[:, :, : self.input_num_quantizers],
                        code_attention_mask=code_attention_mask,
                        sep_prompt_embedding=mimi_spk_embed,
                    )
                else:
                    (
                        temporal_inputs,
                        temporal_mask,
                        unified_targets,
                        unified_target_mask,
                        codec_unified_target_mask,
                        code_start_positions,
                    ) = self._build_textcodec_temporal_batch(
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask,
                        labels=labels[:, :, : self.input_num_quantizers],
                        code_attention_mask=code_attention_mask,
                        sep_prompt_embedding=mimi_spk_embed,
                    )
                # Keep the text path teacher-forced, matching TransVIP's
                # speech/text paired objective even if speech scheduled sampling is enabled.
                text_path_temporal_inputs = temporal_inputs
                if (
                    self.training
                    and self.enable_quality_cot
                    and self.text_path_source_token_dropout > 0.0
                ):
                    if quality_cot_source_text_target_mask is None:
                        raise RuntimeError("Source-token dropout requires the Quality-CoT source mask.")
                    # Target mask position p supervises token p+1. Shift it once
                    # to identify source lexical tokens used as decoder inputs.
                    source_input_mask = torch.zeros_like(
                        quality_cot_source_text_target_mask,
                        dtype=torch.bool,
                    )
                    source_input_mask[:, 1:] = (
                        quality_cot_source_text_target_mask[:, :-1] != 0
                    )
                    sampled_drop = (
                        torch.rand_like(source_input_mask, dtype=torch.float32)
                        < self.text_path_source_token_dropout
                    ) & source_input_mask
                    if bool(sampled_drop.any()):
                        text_path_temporal_inputs = text_path_temporal_inputs.clone()
                        positions = torch.arange(
                            text_path_temporal_inputs.size(1),
                            device=text_path_temporal_inputs.device,
                        )
                        position_only = self.temporal_pos_emb(positions).unsqueeze(0)
                        text_path_temporal_inputs = torch.where(
                            sampled_drop.unsqueeze(-1),
                            position_only.to(dtype=text_path_temporal_inputs.dtype),
                            text_path_temporal_inputs,
                        )
                effective_temporal_scheduled_sampling_prob = (
                    self._effective_temporal_scheduled_sampling_prob()
                )
                temporal_inputs = self._apply_textcodec_scheduled_sampling(
                    temporal_inputs=temporal_inputs,
                    temporal_mask=temporal_mask,
                    code_start_positions=code_start_positions,
                    encoder_hidden=encoder_hidden,
                    encoder_mask=encoder_mask,
                    scheduled_sampling_prob=effective_temporal_scheduled_sampling_prob,
                )
                target_len = temporal_inputs.size(1)
        elif text_prefix_mode:
            if labels.size(1) < 2:
                raise ValueError("Text-prefix AR requires code labels with at least BOS and one target row.")
            target_labels = labels[:, 1:, : self.num_codebook_levels].clone()
            target_mask = code_attention_mask[:, 1:] if code_attention_mask is not None else None
            (
                temporal_inputs,
                temporal_mask,
                text_targets,
                text_target_mask,
                code_start_positions,
            ) = self._build_textprefix_temporal_batch(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                labels=labels[:, :, : self.input_num_quantizers],
                code_attention_mask=code_attention_mask,
                sep_prompt_embedding=mimi_spk_embed,
            )
            target_len = temporal_inputs.size(1)
        elif labels is None:
            target_len = encoder_hidden.size(1)
            if target_len > self.temporal_max_positions:
                raise ValueError(
                    f"target_len={target_len} exceeds temporal_max_positions={self.temporal_max_positions}."
                )
            pos_ids = torch.arange(target_len, device=encoder_hidden.device)
            temporal_inputs = self.temporal_pos_emb(pos_ids).unsqueeze(0).expand(encoder_hidden.size(0), -1, -1)
            temporal_mask = None
        else:
            if labels.size(1) < 2:
                raise ValueError("Next-token prediction requires labels with at least BOS and one target token.")
            input_labels = labels[:, :-1, : self.input_num_quantizers]
            target_labels = labels[:, 1:, : self.num_codebook_levels].clone()
            target_len = input_labels.size(1)
            if code_attention_mask is not None:
                target_mask = code_attention_mask[:, 1:]
                code_attention_mask = code_attention_mask[:, :-1]
            temporal_mask = self._align_1d_mask(
                code_attention_mask,
                target_len=target_len,
                device=input_labels.device,
            )
            effective_temporal_scheduled_sampling_prob = self._effective_temporal_scheduled_sampling_prob()
            if self._uses_unity_direct_c0_temporal():
                if effective_temporal_scheduled_sampling_prob > 0.0:
                    raise ValueError(
                        "unity_direct_c0 does not support temporal scheduled sampling; "
                        "set temporal_scheduled_sampling_prob=0."
                    )
                unity_direct_token_ids, temporal_mask = (
                    self._build_unity_direct_c0_token_batch(input_labels)
                )
                temporal_inputs = None
            else:
                input_labels = self._apply_temporal_scheduled_sampling(
                    input_labels=input_labels,
                    temporal_mask=temporal_mask,
                    encoder_hidden=encoder_hidden,
                    encoder_mask=encoder_mask,
                    scheduled_sampling_prob=effective_temporal_scheduled_sampling_prob,
                )
                temporal_inputs = self._build_temporal_inputs(input_labels)

        if temporal_mask is None:
            if temporal_inputs is None:
                raise RuntimeError("Missing temporal mask for UnitY temporal decoder path.")
            temporal_mask = torch.ones(
                (temporal_inputs.size(0), target_len), device=temporal_inputs.device, dtype=torch.long
            )

        # Temporal decoder: either DirectS2ST's decoder or the native UnitY text decoder.
        unified_logits = None
        if self._uses_unity_text_decoder_temporal() and text_prefix_mode and self.enable_text_codec_ar:
            if unity_token_ids is None or unity_token_mask is None:
                raise RuntimeError("UnitY temporal backend missing unified token inputs.")
            temporal_hidden, unified_logits = self._unity_decode_token_ids(
                unity_token_ids,
                unity_token_mask,
                encoder_hidden,
                encoder_mask,
                spk_embed=unity_spk_embed,
            )
            temporal_mask = unity_token_mask
            target_len = int(temporal_hidden.size(1))
        elif self._uses_unity_direct_c0_temporal():
            if unity_direct_token_ids is None:
                raise RuntimeError(
                    "unity_direct_c0 requires teacher-forced Mimi C0 token inputs."
                )
            temporal_hidden, unified_logits = self._unity_decode_token_ids(
                unity_direct_token_ids,
                temporal_mask,
                encoder_hidden,
                encoder_mask,
                spk_embed=None,
            )
            direct_prefix_len = self._unity_direct_c0_control_prefix_length()
            if direct_prefix_len > 0:
                temporal_hidden = temporal_hidden[:, direct_prefix_len:]
                unified_logits = unified_logits[:, direct_prefix_len:]
                temporal_mask = temporal_mask[:, direct_prefix_len:]
            target_len = int(temporal_hidden.size(1))
        else:
            if self.temporal_transformer is None or self.temporal_norm is None:
                raise RuntimeError("Direct temporal decoder requested but it is not initialized.")
            with torch.profiler.record_function("s2st/temporal_main"):
                temporal_hidden = self.temporal_transformer(
                    tgt=temporal_inputs,
                    memory=encoder_hidden,
                    tgt_mask=self._causal_mask(target_len, temporal_inputs.device),
                    tgt_key_padding_mask=(temporal_mask == 0),
                    memory_key_padding_mask=(encoder_mask == 0) if encoder_mask is not None else None,
                )
            temporal_hidden = self.temporal_norm(temporal_hidden)

        codec_temporal_hidden = temporal_hidden
        unified_code0_loss = None
        codec_mass_loss = None
        codec_mass_prob = None
        codec_mass_valid_tokens = None
        text_path_text_loss = None
        text_path_text_valid_tokens = None
        text_path_code0_loss = None
        text_path_code0_valid_tokens = None
        text_codec_kd_loss = None
        text_codec_kd_valid_tokens = None
        c0_auxiliary_loss = None
        c0_auxiliary_valid_tokens = None
        direct_text_to_c0_loss = None
        direct_text_to_c0_valid_tokens = None
        direct_text_to_c0_error_rate = None
        direct_speech_text_c0_kd_loss = None
        direct_speech_text_c0_kd_valid_tokens = None
        c0_teacher_distill_loss = None
        c0_teacher_distill_valid_tokens = None
        c0_teacher_distill_topk_mass = None
        c0_gold_anchored_distill_loss = None
        c0_gold_anchored_distill_valid_tokens = None
        c0_gold_anchored_distill_topk_mass = None
        c0_gold_anchored_distill_teacher_top1_acc = None
        c0_gold_anchored_distill_teacher_topk_hit = None
        c0_teacher_hidden_distill_loss = None
        c0_teacher_hidden_distill_valid_tokens = None
        c0_teacher_hidden_distill_cosine = None
        quality_cot_source_text_loss = None
        quality_cot_source_text_valid_tokens = None
        quality_cot_target_text_loss = None
        quality_cot_target_text_valid_tokens = None
        quality_cot_objective_loss = None
        target_only_speech_text_loss = None
        target_only_speech_text_valid_tokens = None
        target_only_speech_code0_loss = None
        target_only_speech_code0_valid_tokens = None
        mixed_source_target_text_loss = None
        mixed_source_target_text_valid_tokens = None
        mixed_source_replacement_ratio = None
        mixed_source_replacement_eligible_tokens = None
        mixed_source_target_effective_weight = 0.0
        predicted_source_target_text_loss = None
        predicted_source_target_text_valid_tokens = None
        predicted_source_target_mean_tokens = None
        predicted_source_target_effective_weight = 0.0
        predicted_source_target_hidden = None
        if text_prefix_mode:
            assert code_start_positions is not None
            assert target_labels is not None
            bsz = temporal_hidden.size(0)
            codec_len = target_labels.size(1)
            codec_temporal_hidden = temporal_hidden.new_zeros((bsz, codec_len, temporal_hidden.size(-1)))
            if self.enable_text_codec_ar:
                if unified_logits is None:
                    unified_logits = self.text_codec_lm_head(temporal_hidden)
                assert unified_targets is not None and codec_unified_target_mask is not None
                flat_targets = unified_targets.reshape(-1)
                valid = flat_targets != self.ignore_index
                unity_code_offset = self._unity_code_offset()
                unified_vocab_size = int(unified_logits.size(-1))
                text_valid = valid & (flat_targets < int(unity_code_offset))
                code_valid = valid & (flat_targets >= int(unity_code_offset))
                if bool(valid.any().item()):
                    max_target_id = int(flat_targets[valid].max().detach().cpu().item())
                    if max_target_id >= unified_vocab_size:
                        raise ValueError(
                            "Unified target id exceeds UnitY projection vocab: "
                            f"max_target={max_target_id}, vocab={unified_vocab_size}."
                        )
                flat_unified_logits = unified_logits.reshape(-1, unified_vocab_size)
                if int(text_valid.sum().detach().cpu().item()) > 0:
                    text_loss = F.cross_entropy(
                        flat_unified_logits[text_valid],
                        flat_targets[text_valid],
                    )
                    text_valid_tokens = text_valid.sum()
                    if not self.training:
                        diagnostic_logits = flat_unified_logits.detach()
                        text_predictions = diagnostic_logits.argmax(dim=-1)
                        target_semantic_error_rate = (
                            (text_predictions[text_valid] != flat_targets[text_valid])
                            .sum()
                            .to(unified_logits.dtype)
                            / text_valid_tokens.clamp_min(1).to(unified_logits.dtype)
                        )

                        semantic_sep_valid = text_valid & (
                            flat_targets == int(self.text_sep_token_id)
                        )
                        semantic_content_valid = text_valid & ~semantic_sep_valid
                        if bool(semantic_sep_valid.any().item()):
                            target_semantic_sep_loss = F.cross_entropy(
                                diagnostic_logits[semantic_sep_valid],
                                flat_targets[semantic_sep_valid],
                            )
                            target_semantic_sep_valid_tokens = semantic_sep_valid.sum()
                            target_semantic_sep_error_rate = (
                                (
                                    text_predictions[semantic_sep_valid]
                                    != flat_targets[semantic_sep_valid]
                                )
                                .sum()
                                .to(unified_logits.dtype)
                                / target_semantic_sep_valid_tokens.clamp_min(1).to(
                                    unified_logits.dtype
                                )
                            )
                        if bool(semantic_content_valid.any().item()):
                            target_semantic_content_valid_tokens = (
                                semantic_content_valid.sum()
                            )
                            semantic_total_nll = (
                                text_loss.detach()
                                * text_valid_tokens.to(text_loss.dtype)
                            )
                            semantic_sep_nll = semantic_total_nll.new_zeros(())
                            if (
                                target_semantic_sep_loss is not None
                                and target_semantic_sep_valid_tokens is not None
                            ):
                                semantic_sep_nll = (
                                    target_semantic_sep_loss
                                    * target_semantic_sep_valid_tokens.to(
                                        target_semantic_sep_loss.dtype
                                    )
                                )
                            target_semantic_content_loss = (
                                semantic_total_nll - semantic_sep_nll
                            ) / target_semantic_content_valid_tokens.clamp_min(1).to(
                                text_loss.dtype
                            )
                            target_semantic_content_error_rate = (
                                (
                                    text_predictions[semantic_content_valid]
                                    != flat_targets[semantic_content_valid]
                                )
                                .sum()
                                .to(unified_logits.dtype)
                                / target_semantic_content_valid_tokens.clamp_min(1).to(
                                    unified_logits.dtype
                                )
                            )

                        if unified_targets.size(1) > 0:
                            first_semantic_targets = unified_targets[:, 0]
                            first_semantic_valid = (
                                (first_semantic_targets != self.ignore_index)
                                & (first_semantic_targets < int(unity_code_offset))
                            )
                            if bool(first_semantic_valid.any().item()):
                                first_semantic_logits = unified_logits[:, 0, :].detach()
                                target_semantic_first_token_loss = F.cross_entropy(
                                    first_semantic_logits[first_semantic_valid],
                                    first_semantic_targets[first_semantic_valid],
                                )
                                target_semantic_first_token_valid_tokens = (
                                    first_semantic_valid.sum()
                                )
                                target_semantic_first_token_error_rate = (
                                    (
                                        first_semantic_logits[
                                            first_semantic_valid
                                        ].argmax(dim=-1)
                                        != first_semantic_targets[first_semantic_valid]
                                    )
                                    .sum()
                                    .to(unified_logits.dtype)
                                    / target_semantic_first_token_valid_tokens.clamp_min(
                                        1
                                    ).to(unified_logits.dtype)
                                )
                else:
                    text_loss = unified_logits.new_zeros(())
                    text_valid_tokens = text_loss.new_zeros((), dtype=torch.long)
                if int(code_valid.sum().detach().cpu().item()) > 0:
                    valid_unified_logits = flat_unified_logits[code_valid]
                    valid_code_targets = flat_targets[code_valid]
                    unified_code0_loss = F.cross_entropy(
                        valid_unified_logits,
                        valid_code_targets,
                    )
                    code_slice_start = int(unity_code_offset)
                    code_slice_end = min(
                        code_slice_start + int(self.vocab_size),
                        int(unified_vocab_size),
                    )
                    if code_slice_end <= code_slice_start:
                        raise ValueError(
                            "Invalid codec slice for codec_mass_loss: "
                            f"start={code_slice_start}, end={code_slice_end}, "
                            f"unified_vocab_size={unified_vocab_size}."
                        )
                    full_log_norm = torch.logsumexp(valid_unified_logits, dim=-1)
                    codec_log_norm = torch.logsumexp(
                        valid_unified_logits[:, code_slice_start:code_slice_end],
                        dim=-1,
                    )
                    codec_log_mass = codec_log_norm - full_log_norm
                    codec_mass_loss = -codec_log_mass.mean()
                    codec_mass_prob = codec_log_mass.exp().mean()
                    codec_mass_valid_tokens = code_valid.sum()
                for i, code_start in enumerate(code_start_positions):
                    valid_codec_len = int(target_mask[i].sum().item()) if target_mask is not None else codec_len
                    if valid_codec_len > 0:
                        codec_temporal_hidden[i, :valid_codec_len] = temporal_hidden[i, code_start : code_start + valid_codec_len]
            else:
                max_text_target_len = text_targets.size(1) if text_targets is not None else 0
                text_hidden = temporal_hidden.new_zeros((bsz, max_text_target_len, temporal_hidden.size(-1)))
                for i, code_start in enumerate(code_start_positions):
                    valid_codec_len = int(target_mask[i].sum().item()) if target_mask is not None else codec_len
                    if valid_codec_len > 0:
                        codec_temporal_hidden[i, :valid_codec_len] = temporal_hidden[i, code_start : code_start + valid_codec_len]
                    valid_text_len = int(text_target_mask[i].sum().item())
                    if valid_text_len > 0:
                        text_hidden[i, :valid_text_len] = temporal_hidden[i, :valid_text_len]
                if text_targets is not None and text_targets.numel() > 0:
                    text_logits = self.text_lm_head(text_hidden)
                    text_flat_labels = text_targets.reshape(-1)
                    text_valid_tokens = (text_flat_labels != self.ignore_index).sum()
                    if int(text_valid_tokens.detach().cpu().item()) > 0:
                        text_loss = F.cross_entropy(
                            text_logits.reshape(-1, int(self.text_vocab_size)),
                            text_flat_labels,
                            ignore_index=self.ignore_index,
                        )
                    else:
                        text_loss = text_logits.new_zeros(())

        if self.enable_quality_cot:
            if (
                unified_logits is None
                or unified_targets is None
                or quality_cot_source_text_target_mask is None
                or quality_cot_target_text_target_mask is None
            ):
                raise RuntimeError(
                    "Quality-CoT loss requires unified logits, targets, and both text-segment masks."
                )
            flat_quality_logits = unified_logits.reshape(-1, int(unified_logits.size(-1)))
            flat_quality_targets = unified_targets.reshape(-1)
            source_valid = (
                quality_cot_source_text_target_mask.reshape(-1).bool()
                & (flat_quality_targets != self.ignore_index)
            )
            target_valid = (
                quality_cot_target_text_target_mask.reshape(-1).bool()
                & (flat_quality_targets != self.ignore_index)
            )
            quality_cot_source_text_valid_tokens = source_valid.sum()
            quality_cot_target_text_valid_tokens = target_valid.sum()
            if int(quality_cot_source_text_valid_tokens.detach().cpu().item()) <= 0:
                raise RuntimeError("Quality-CoT source-text segment has no supervised tokens.")
            if int(quality_cot_target_text_valid_tokens.detach().cpu().item()) <= 0:
                raise RuntimeError("Quality-CoT target-text segment has no supervised tokens.")
            quality_cot_source_text_loss = self._text_cross_entropy(
                flat_quality_logits[source_valid],
                flat_quality_targets[source_valid],
            )
            quality_cot_target_text_loss = self._text_cross_entropy(
                flat_quality_logits[target_valid],
                flat_quality_targets[target_valid],
            )
            # Keep the historical text_ce_loss/s2t metric target-side, because
            # this is the translation segment that is generated for C0.
            text_loss = quality_cot_target_text_loss
            text_valid_tokens = quality_cot_target_text_valid_tokens

        if self.enable_mixed_source_target_auxiliary:
            if source_text_input_ids is None or text_input_ids is None or labels is None:
                raise RuntimeError(
                    "Mixed-source target auxiliary requires source text, target text, and codec labels."
                )
            mixed_source_scale = self._mixed_source_auxiliary_scale()
            mixed_source_target_effective_weight = (
                self.mixed_source_target_loss_weight * mixed_source_scale
            )
            mixed_source_effective_prob = (
                self.mixed_source_prediction_prob * mixed_source_scale
            )
            if (
                mixed_source_target_effective_weight > 0.0
                and mixed_source_effective_prob > 0.0
            ):
                mixed_source_ids, replaced_tokens, eligible_tokens = (
                    self._build_mixed_source_text_input_ids(
                        source_text_input_ids=source_text_input_ids,
                        source_text_attention_mask=source_text_attention_mask,
                        unified_logits=unified_logits,
                        replacement_prob=mixed_source_effective_prob,
                    )
                )
                (
                    mixed_inputs,
                    mixed_mask,
                    mixed_targets,
                    _mixed_unified_target_mask,
                    _mixed_codec_target_mask,
                    mixed_code_starts,
                    _mixed_source_target_mask,
                    mixed_target_text_mask,
                ) = self._build_quality_cot_textcodec_temporal_batch(
                    source_text_input_ids=mixed_source_ids,
                    source_text_attention_mask=source_text_attention_mask,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    labels=labels[:, :, : self.input_num_quantizers],
                    code_attention_mask=full_code_attention_mask,
                    sep_prompt_embedding=mimi_spk_embed,
                )
                # The auxiliary is P(target text | source speech, mixed source text).
                # Stop before the first C0 target so it cannot duplicate C0/depth loss.
                mixed_text_len = max(int(position) for position in mixed_code_starts)
                mixed_inputs = mixed_inputs[:, :mixed_text_len]
                mixed_mask = mixed_mask[:, :mixed_text_len]
                mixed_targets = mixed_targets[:, :mixed_text_len]
                mixed_target_text_mask = mixed_target_text_mask[:, :mixed_text_len]
                mixed_hidden = self.temporal_transformer(
                    tgt=mixed_inputs,
                    memory=encoder_hidden,
                    tgt_mask=self._causal_mask(mixed_text_len, mixed_inputs.device),
                    tgt_key_padding_mask=(mixed_mask == 0),
                    memory_key_padding_mask=(encoder_mask == 0)
                    if encoder_mask is not None
                    else None,
                )
                mixed_hidden = self.temporal_norm(mixed_hidden)
                flat_mixed_targets = mixed_targets.reshape(-1)
                mixed_target_valid = (
                    mixed_target_text_mask.reshape(-1).bool()
                    & (flat_mixed_targets != self.ignore_index)
                )
                mixed_source_target_text_valid_tokens = mixed_target_valid.sum()
                if int(mixed_source_target_text_valid_tokens.detach().cpu().item()) <= 0:
                    raise RuntimeError(
                        "Mixed-source target-text segment has no supervised tokens."
                    )
                # Project only supervised target-text positions. The full unified
                # head is large, and projecting padded/source positions adds no loss.
                mixed_logits = self.text_codec_lm_head(
                    mixed_hidden.reshape(-1, mixed_hidden.size(-1))[mixed_target_valid]
                )
                mixed_source_target_text_loss = F.cross_entropy(
                    mixed_logits,
                    flat_mixed_targets[mixed_target_valid],
                )
                mixed_source_replacement_eligible_tokens = eligible_tokens
                mixed_source_replacement_ratio = replaced_tokens / eligible_tokens.clamp_min(1.0)

        if self.enable_predicted_source_target_auxiliary:
            if text_input_ids is None or labels is None:
                raise RuntimeError(
                    "Predicted-source target auxiliary requires target text and codec labels."
                )
            predicted_source_scale = (
                self._predicted_source_target_auxiliary_scale()
            )
            predicted_source_target_effective_weight = (
                self.predicted_source_target_loss_weight * predicted_source_scale
            )
            if predicted_source_target_effective_weight > 0.0:
                auxiliary_batch_size = min(
                    int(self.predicted_source_target_batch_size),
                    int(encoder_hidden.size(0)),
                )
                auxiliary_start = (
                    int(getattr(self, "_current_global_step", 0) or 0)
                    * auxiliary_batch_size
                ) % int(encoder_hidden.size(0))
                auxiliary_indices = (
                    torch.arange(
                        auxiliary_batch_size,
                        device=encoder_hidden.device,
                        dtype=torch.long,
                    )
                    + auxiliary_start
                ) % int(encoder_hidden.size(0))
                auxiliary_encoder_hidden = encoder_hidden.index_select(
                    0,
                    auxiliary_indices,
                )
                auxiliary_encoder_mask = (
                    encoder_mask.index_select(0, auxiliary_indices)
                    if encoder_mask is not None
                    else None
                )
                auxiliary_prompt = (
                    mimi_spk_embed.index_select(0, auxiliary_indices)
                    if mimi_spk_embed is not None
                    else None
                )
                auxiliary_text_ids = text_input_ids.index_select(
                    0,
                    auxiliary_indices,
                )
                auxiliary_text_mask = (
                    text_attention_mask.index_select(0, auxiliary_indices)
                    if text_attention_mask is not None
                    else None
                )
                auxiliary_labels = labels.index_select(0, auxiliary_indices)
                auxiliary_code_mask = (
                    full_code_attention_mask.index_select(0, auxiliary_indices)
                    if full_code_attention_mask is not None
                    else None
                )
                (
                    predicted_source_ids,
                    predicted_source_mask,
                    predicted_source_target_mean_tokens,
                ) = self._generate_quality_cot_source_ids_greedy_batch(
                    encoder_hidden=auxiliary_encoder_hidden.detach(),
                    encoder_mask=auxiliary_encoder_mask,
                    max_tokens=self.predicted_source_target_max_source_tokens,
                    sep_prompt_embedding=(
                        auxiliary_prompt.detach()
                        if auxiliary_prompt is not None
                        else None
                    ),
                )
                (
                    predicted_inputs,
                    predicted_mask,
                    predicted_targets,
                    _predicted_unified_target_mask,
                    _predicted_codec_target_mask,
                    predicted_code_starts,
                    _predicted_source_target_mask,
                    predicted_target_text_mask,
                ) = self._build_quality_cot_textcodec_temporal_batch(
                    source_text_input_ids=predicted_source_ids,
                    source_text_attention_mask=predicted_source_mask,
                    text_input_ids=auxiliary_text_ids,
                    text_attention_mask=auxiliary_text_mask,
                    labels=auxiliary_labels[:, :, : self.input_num_quantizers],
                    code_attention_mask=auxiliary_code_mask,
                    sep_prompt_embedding=auxiliary_prompt,
                )
                # This branch supervises only P(target text | speech, generated
                # source text); C0 and depth remain owned by their main losses.
                predicted_text_len = max(
                    int(position) for position in predicted_code_starts
                )
                predicted_inputs = predicted_inputs[:, :predicted_text_len]
                predicted_mask = predicted_mask[:, :predicted_text_len]
                predicted_targets = predicted_targets[:, :predicted_text_len]
                predicted_target_text_mask = predicted_target_text_mask[
                    :, :predicted_text_len
                ]
                predicted_hidden = self.temporal_transformer(
                    tgt=predicted_inputs,
                    memory=auxiliary_encoder_hidden,
                    tgt_mask=self._causal_mask(
                        predicted_text_len,
                        predicted_inputs.device,
                    ),
                    tgt_key_padding_mask=(predicted_mask == 0),
                    memory_key_padding_mask=(auxiliary_encoder_mask == 0)
                    if auxiliary_encoder_mask is not None
                    else None,
                )
                predicted_hidden = self.temporal_norm(predicted_hidden)
                predicted_source_target_hidden = predicted_hidden
                flat_predicted_targets = predicted_targets.reshape(-1)
                predicted_target_valid = (
                    predicted_target_text_mask.reshape(-1).bool()
                    & (flat_predicted_targets != self.ignore_index)
                )
                predicted_source_target_text_valid_tokens = (
                    predicted_target_valid.sum()
                )
                if (
                    int(
                        predicted_source_target_text_valid_tokens.detach()
                        .cpu()
                        .item()
                    )
                    <= 0
                ):
                    raise RuntimeError(
                        "Predicted-source target-text segment has no supervised tokens."
                    )
                predicted_logits = self.text_codec_lm_head(
                    predicted_hidden.reshape(-1, predicted_hidden.size(-1))[
                        predicted_target_valid
                    ]
                )
                predicted_source_target_text_loss = F.cross_entropy(
                    predicted_logits,
                    flat_predicted_targets[predicted_target_valid],
                )

        if self.enable_target_only_speech_auxiliary:
            if text_input_ids is None or target_labels is None:
                raise RuntimeError(
                    "Target-only speech auxiliary requires target text and codec labels."
                )
            (
                target_only_inputs,
                target_only_mask,
                target_only_targets,
                _target_only_target_mask,
                target_only_codec_mask,
                _target_only_code_starts,
            ) = self._build_textcodec_temporal_batch(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                labels=labels[:, :, : self.input_num_quantizers],
                code_attention_mask=full_code_attention_mask,
                sep_prompt_embedding=mimi_spk_embed,
            )
            with torch.profiler.record_function("s2st/temporal_target_only_aux"):
                target_only_hidden = self.temporal_transformer(
                    tgt=target_only_inputs,
                    memory=encoder_hidden,
                    tgt_mask=self._causal_mask(
                        target_only_inputs.size(1), target_only_inputs.device
                    ),
                    tgt_key_padding_mask=(target_only_mask == 0),
                    memory_key_padding_mask=(encoder_mask == 0)
                    if encoder_mask is not None
                    else None,
                )
            target_only_hidden = self.temporal_norm(target_only_hidden)
            target_only_logits = self.text_codec_lm_head(target_only_hidden)

            target_only_text_mask = torch.zeros_like(target_only_codec_mask)
            if text_attention_mask is None:
                target_only_text_lengths = (
                    text_input_ids != self.text_pad_token_id
                ).sum(dim=1).long()
            else:
                target_only_text_lengths = text_attention_mask.sum(dim=1).long()
            target_control_len = len(self.text_prefix_extra_token_ids_after_bos)
            if self.enable_uniss_content_controls:
                target_control_len += 1
            for sample_idx, text_len_tensor in enumerate(target_only_text_lengths):
                text_target_len = max(0, int(text_len_tensor.item()) - 1)
                content_start = min(target_control_len, text_target_len)
                target_only_text_mask[
                    sample_idx, content_start:text_target_len
                ] = 1

            flat_target_only_logits = target_only_logits.reshape(
                -1, int(target_only_logits.size(-1))
            )
            flat_target_only_targets = target_only_targets.reshape(-1)
            target_only_text_valid = (
                target_only_text_mask.reshape(-1).bool()
                & (flat_target_only_targets != self.ignore_index)
            )
            target_only_code0_valid = (
                target_only_codec_mask.reshape(-1).bool()
                & (flat_target_only_targets != self.ignore_index)
            )
            target_only_speech_text_valid_tokens = target_only_text_valid.sum()
            target_only_speech_code0_valid_tokens = target_only_code0_valid.sum()
            if int(target_only_speech_text_valid_tokens.detach().cpu().item()) <= 0:
                raise RuntimeError("Target-only speech text mask has no supervised tokens.")
            if int(target_only_speech_code0_valid_tokens.detach().cpu().item()) <= 0:
                raise RuntimeError("Target-only speech C0 mask has no supervised tokens.")
            target_only_speech_text_loss = F.cross_entropy(
                flat_target_only_logits[target_only_text_valid],
                flat_target_only_targets[target_only_text_valid],
            )
            target_only_speech_code0_loss = F.cross_entropy(
                flat_target_only_logits[target_only_code0_valid],
                flat_target_only_targets[target_only_code0_valid],
            )

        if (
            self.enable_text_codec_text_path_loss
            and text_prefix_mode
            and self.enable_text_codec_ar
        ):
            if unified_targets is None:
                raise RuntimeError("Text-path text+codec loss requested but unified targets are missing.")
            text_path_input_ids = text_input_ids
            text_path_attention_mask = text_attention_mask
            if self.text_codec_text_path_input == "source":
                text_path_input_ids = source_text_input_ids
                text_path_attention_mask = source_text_attention_mask
            if text_path_input_ids is None:
                raise RuntimeError(
                    f"Text-path loss requested with text_codec_text_path_input={self.text_codec_text_path_input!r}, "
                    "but the corresponding text input ids are missing from the batch."
                )
            if not (self.enable_quality_cot and not self.load_text_path_encoder):
                raise ValueError(
                    "The text-to-text / text-to-C0 objectives require enable_quality_cot=True "
                    "and load_text_path_encoder=False."
                )
            # Teacher-forced gold source->target->C0 stream with the speech memory
            # replaced by a single zero token (no extra parameters).
            aux_memory = text_path_temporal_inputs.new_zeros(
                (text_path_temporal_inputs.size(0), 1, text_path_temporal_inputs.size(-1))
            )
            aux_memory_mask = torch.ones(
                (text_path_temporal_inputs.size(0), 1),
                dtype=torch.long,
                device=text_path_temporal_inputs.device,
            )
            if self._uses_unity_text_decoder_temporal():
                if unity_token_ids is None or unity_token_mask is None:
                    raise RuntimeError("UnitY text path is missing its unified decoder batch.")
                _, text_path_unified_logits = self._unity_decode_token_ids(
                    unity_token_ids,
                    unity_token_mask,
                    aux_memory,
                    aux_memory_mask,
                    spk_embed=unity_spk_embed,
                )
            else:
                if text_path_temporal_inputs is None or temporal_mask is None:
                    raise RuntimeError(
                        "Direct text path requires the teacher-forced text+codec temporal inputs."
                    )
                if self.temporal_transformer is None or self.temporal_norm is None:
                    raise RuntimeError("Direct text path requires the Direct temporal decoder.")
                text_memory = aux_memory.to(
                    device=text_path_temporal_inputs.device,
                    dtype=text_path_temporal_inputs.dtype,
                )
                text_memory_mask = (
                    aux_memory_mask.to(device=text_path_temporal_inputs.device)
                    if aux_memory_mask is not None
                    else None
                )
                text_path_hidden = self.temporal_transformer(
                    tgt=text_path_temporal_inputs,
                    memory=text_memory,
                    tgt_mask=self._causal_mask(
                        text_path_temporal_inputs.size(1),
                        text_path_temporal_inputs.device,
                    ),
                    tgt_key_padding_mask=(temporal_mask == 0),
                    memory_key_padding_mask=(text_memory_mask == 0)
                    if text_memory_mask is not None
                    else None,
                )
                text_path_hidden = self.temporal_norm(text_path_hidden)
                text_path_unified_logits = self.text_codec_lm_head(text_path_hidden)
            if self.enable_quality_cot:
                if quality_cot_target_text_target_mask is None or codec_unified_target_mask is None:
                    raise RuntimeError("Quality-CoT decoder-only auxiliary masks are missing.")
                flat_aux_logits = text_path_unified_logits.reshape(
                    -1, int(text_path_unified_logits.size(-1))
                )
                flat_aux_targets = unified_targets.reshape(-1)
                t2t_valid = (
                    quality_cot_target_text_target_mask.reshape(-1).bool()
                    & (flat_aux_targets != self.ignore_index)
                )
                t2c_valid = (
                    codec_unified_target_mask.reshape(-1).bool()
                    & (flat_aux_targets != self.ignore_index)
                )
                text_path_text_valid_tokens = t2t_valid.sum()
                text_path_code0_valid_tokens = t2c_valid.sum()
                if int(text_path_text_valid_tokens.detach().cpu().item()) <= 0:
                    raise RuntimeError("Quality-CoT decoder-only t2t mask has no tokens.")
                if int(text_path_code0_valid_tokens.detach().cpu().item()) <= 0:
                    raise RuntimeError("Quality-CoT decoder-only t2c mask has no tokens.")
                text_path_text_loss = self._text_cross_entropy(
                    flat_aux_logits[t2t_valid],
                    flat_aux_targets[t2t_valid],
                )
                text_path_code0_loss = F.cross_entropy(
                    flat_aux_logits[t2c_valid], flat_aux_targets[t2c_valid]
                )
            else:
                (
                    text_path_text_loss,
                    text_path_text_valid_tokens,
                    text_path_code0_loss,
                    text_path_code0_valid_tokens,
                ) = self._compute_unified_text_codec_losses(
                    text_path_unified_logits,
                    unified_targets,
                )
            if (
                self.enable_text_codec_kd_loss
                and self.text_codec_kd_loss_weight > 0.0
                and unified_logits is not None
            ):
                (
                    text_codec_kd_loss,
                    text_codec_kd_valid_tokens,
                ) = self._compute_text_codec_kd_loss(
                    unified_logits,
                    text_path_unified_logits,
                    unified_targets,
                )

        # ========== Depth Decoder ==========
        # depth_decoder predicts level 0 directly, then decodes levels 1 to N-1.
        effective_depth_teacher_force = (
            self.teacher_force and self.training
            if depth_teacher_force is None
            else bool(depth_teacher_force)
        )
        effective_depth_scheduled_sampling_prob = (
            self._effective_depth_scheduled_sampling_prob()
            if effective_depth_teacher_force and target_labels is not None
            else 0.0
        )
        effective_depth_chain_scheduled_sampling_prob = (
            self._effective_depth_chain_scheduled_sampling_prob()
            if effective_depth_teacher_force and target_labels is not None
            else 0.0
        )
        forced_first_ids_for_depth = None
        if (
            depth_force_first_level0
            and target_labels is not None
            and not effective_depth_teacher_force
        ):
            # Diagnostic path: keep depth cascade free-running for C1..C15 but
            # force its C0 condition to gold. This isolates whether C1/depth
            # failures come from predicted C0 or from the depth decoder itself.
            forced_first_ids_for_depth = target_labels[:, :, 0]
        elif (
            self.enable_text_codec_ar
            and unified_logits is not None
            and code_start_positions is not None
            and target_labels is not None
            and not effective_depth_teacher_force
        ):
            # In text+codec AR mode the final C0 logits come from the UnitY
            # unified decoder, not TemporalDepthDecoder.first_codebook_head.
            # Feed that same predicted C0 into depth cascade eval/generation so
            # C1..C15 are conditioned on the C0 path we actually score/decode.
            offset = self._unity_code_offset()
            unified_code_logits_for_depth = unified_logits[
                :, :, offset : offset + self.vocab_size
            ]
            forced_first_ids_for_depth = target_labels.new_full(
                (target_labels.size(0), target_labels.size(1)),
                self.ignore_index,
            )
            for i, code_start in enumerate(code_start_positions):
                valid_codec_len = (
                    int(target_mask[i].sum().item())
                    if target_mask is not None
                    else int(target_labels.size(1))
                )
                if valid_codec_len <= 0:
                    continue
                code_start = int(code_start)
                code_end = code_start + valid_codec_len
                forced_first_ids_for_depth[i, :valid_codec_len] = (
                    unified_code_logits_for_depth[i, code_start:code_end]
                    .detach()
                    .argmax(dim=-1)
                )
        elif (
            self._uses_unity_direct_c0_temporal()
            and unified_logits is not None
            and target_labels is not None
            and not effective_depth_teacher_force
        ):
            direct_code_logits = unified_logits[
                :, :, self._unity_code_offset() : self._unity_code_offset() + self.vocab_size
            ]
            forced_first_ids_for_depth = direct_code_logits.detach().argmax(dim=-1)
        with torch.profiler.record_function("s2st/nar_depth"):
            (
                logits_per_level,
                depth_final_hidden,
                campplus_per_level_identity_loss,
            ) = self.depth_decoder(
                temporal_hidden=codec_temporal_hidden,
                source_memory=encoder_hidden if self.enable_depth_source_conditioning else None,
                source_memory_mask=encoder_mask if self.enable_depth_source_conditioning else None,
                source_acoustic_embedding=source_acoustic_embedding,
                campplus_source_embedding=campplus_source_embedding,
                campplus_identity_embedding=campplus_identity_embedding,
                labels=target_labels,
                teacher_force=effective_depth_teacher_force,
                ignore_index=self.ignore_index,
                detach_temporal_for_depth=(
                    self.level0_depth_objective and self.detach_temporal_for_depth_loss
                ),
                forced_first_ids=forced_first_ids_for_depth,
                scheduled_sampling_prob=effective_depth_scheduled_sampling_prob,
                scheduled_sampling_mode=self.depth_scheduled_sampling_mode,
                scheduled_sampling_topk=self.depth_scheduled_sampling_topk,
                scheduled_sampling_temperature=self.depth_scheduled_sampling_temperature,
                chain_scheduled_sampling_prob=effective_depth_chain_scheduled_sampling_prob,
                chain_scheduled_sampling_mode=self.depth_chain_scheduled_sampling_mode,
                chain_scheduled_sampling_topk=self.depth_chain_scheduled_sampling_topk,
                chain_scheduled_sampling_temperature=self.depth_chain_scheduled_sampling_temperature,
            )

        logits = torch.stack(logits_per_level, dim=2)  # [B, T, num_codebook_levels, V]
        if self.enable_text_codec_ar and unified_logits is not None and code_start_positions is not None:
            offset = self._unity_code_offset()
            unified_code_logits = unified_logits[:, :, offset : offset + self.vocab_size]
            for i, code_start in enumerate(code_start_positions):
                valid_codec_len = int(target_mask[i].sum().item()) if target_mask is not None else logits.size(1)
                if valid_codec_len > 0:
                    logits[i, :valid_codec_len, 0, :] = unified_code_logits[i, code_start : code_start + valid_codec_len]
        elif self._uses_unity_direct_c0_temporal() and unified_logits is not None:
            direct_code_logits = unified_logits[
                :, :, self._unity_code_offset() : self._unity_code_offset() + self.vocab_size
            ]
            logits[:, : direct_code_logits.size(1), 0, :] = direct_code_logits

        # ========== Loss Computation ==========
        loss = None
        level0_loss = None
        temporal_loss = None
        depth_loss = None
        all_codebook_loss = None
        transvip_style_speech_loss = None
        transvip_style_text_loss = None
        transvip_style_depth_loss = None
        transvip_style_total_loss = None
        campplus_depth_identity_loss = campplus_per_level_identity_loss
        level0_valid_tokens = None
        temporal_valid_tokens = None
        depth_valid_tokens = None
        all_valid_tokens = None

        def _token_weighted_average(loss_terms):
            if not loss_terms:
                return None
            numerator = None
            denominator = None
            for term_loss, term_tokens in loss_terms:
                if term_loss is None or term_tokens is None:
                    continue
                tokens = term_tokens.to(device=term_loss.device, dtype=term_loss.dtype)
                weighted = term_loss * tokens
                numerator = weighted if numerator is None else numerator + weighted
                denominator = tokens if denominator is None else denominator + tokens
            if numerator is None or denominator is None:
                return None
            return numerator / denominator.clamp_min(1.0)

        if target_labels is not None:
            codebook_labels = target_labels
            if target_mask is not None:
                codebook_labels = codebook_labels.masked_fill(
                    target_mask.unsqueeze(-1) == 0, self.ignore_index
                )
            codebook_labels = codebook_labels.masked_fill(
                (codebook_labels < 0) | (codebook_labels >= self.vocab_size), self.ignore_index
            )
            if (
                self.enable_campplus_depth_conditioning
                and self.campplus_depth_identity_loss_weight > 0.0
                and self.campplus_identity_level_aggregation == "final"
            ):
                if campplus_identity_embedding is None:
                    raise RuntimeError(
                        "CAMPPlus identity loss is enabled but its supervision "
                        "embedding is missing."
                    )
                identity_len = min(
                    int(depth_final_hidden.size(1)),
                    int(codebook_labels.size(1)),
                )
                identity_mask = (
                    codebook_labels[:, :identity_len, 0] != self.ignore_index
                )
                campplus_depth_identity_loss = self.depth_decoder.campplus_identity_loss(
                    depth_final_hidden[:, :identity_len],
                    identity_mask,
                    campplus_identity_embedding,
                )

            if self.enable_direct_text_to_c0_auxiliary:
                helper_text_input_ids = text_input_ids
                helper_text_attention_mask = text_attention_mask
                if self.direct_text_to_c0_input == "source":
                    helper_text_input_ids = source_text_input_ids
                    helper_text_attention_mask = source_text_attention_mask
                if helper_text_input_ids is None:
                    raise RuntimeError(
                        "Direct text->C0 auxiliary requested with "
                        f"direct_text_to_c0_input={self.direct_text_to_c0_input!r}, "
                        "but the corresponding text_input_ids are missing from the batch."
                    )
                (
                    helper_temporal_inputs,
                    helper_temporal_mask,
                    _helper_text_targets,
                    _helper_text_target_mask,
                    helper_code_start_positions,
                ) = self._build_textprefix_temporal_batch(
                    text_input_ids=helper_text_input_ids,
                    text_attention_mask=helper_text_attention_mask,
                    labels=labels[:, :, : self.input_num_quantizers],
                    code_attention_mask=full_code_attention_mask,
                    sep_prompt_embedding=mimi_spk_embed,
                )
                helper_target_len = int(helper_temporal_inputs.size(1))
                helper_memory = encoder_hidden
                helper_memory_mask = encoder_mask
                if self.direct_text_to_c0_zero_speech_memory:
                    # This auxiliary path is intentionally P(C0 | text, prompt):
                    # retain the source-derived SEP prompt but remove speech memory.
                    helper_memory = helper_temporal_inputs.new_zeros(
                        (helper_temporal_inputs.size(0), 1, helper_temporal_inputs.size(-1))
                    )
                    helper_memory_mask = torch.ones(
                        (helper_temporal_inputs.size(0), 1),
                        dtype=torch.long,
                        device=helper_temporal_inputs.device,
                    )
                helper_hidden = self.temporal_transformer(
                    tgt=helper_temporal_inputs,
                    memory=helper_memory,
                    tgt_mask=self._causal_mask(
                        helper_target_len,
                        helper_temporal_inputs.device,
                    ),
                    tgt_key_padding_mask=(helper_temporal_mask == 0),
                    memory_key_padding_mask=(helper_memory_mask == 0)
                    if helper_memory_mask is not None
                    else None,
                )
                helper_hidden = self.temporal_norm(helper_hidden)
                bsz = helper_hidden.size(0)
                codec_len = int(target_labels.size(1))
                helper_codec_hidden = helper_hidden.new_zeros(
                    (bsz, codec_len, helper_hidden.size(-1))
                )
                for i, code_start in enumerate(helper_code_start_positions):
                    valid_codec_len = (
                        int(target_mask[i].sum().item())
                        if target_mask is not None
                        else codec_len
                    )
                    code_start = int(code_start)
                    span = min(
                        valid_codec_len,
                        codec_len,
                        max(0, int(helper_hidden.size(1)) - code_start),
                    )
                    if span > 0:
                        helper_codec_hidden[i, :span] = helper_hidden[
                            i,
                            code_start : code_start + span,
                        ]
                helper_c0_logits = self.depth_decoder.first_codebook_head(helper_codec_hidden)
                helper_c0_labels = codebook_labels[:, :, 0]
                direct_text_to_c0_valid_tokens = (
                    helper_c0_labels.reshape(-1) != self.ignore_index
                ).sum()
                if int(direct_text_to_c0_valid_tokens.detach().cpu().item()) > 0:
                    direct_text_to_c0_loss = self._level0_cross_entropy_loss(
                        helper_c0_logits,
                        helper_c0_labels,
                    )
                    with torch.no_grad():
                        helper_pred = helper_c0_logits.detach().argmax(dim=-1)
                        helper_valid = helper_c0_labels != self.ignore_index
                        helper_errors = (helper_pred != helper_c0_labels) & helper_valid
                        direct_text_to_c0_error_rate = (
                            helper_errors.sum().to(helper_c0_logits.dtype)
                            / helper_valid.sum().clamp_min(1).to(helper_c0_logits.dtype)
                        )
                else:
                    direct_text_to_c0_loss = helper_c0_logits.new_zeros(())
                    direct_text_to_c0_error_rate = helper_c0_logits.new_zeros(())

                if self.direct_text_to_c0_kd_weight > 0.0:
                    distill_len = min(
                        int(logits.size(1)),
                        int(helper_c0_logits.size(1)),
                        int(helper_c0_labels.size(1)),
                    )
                    if distill_len > 0:
                        kd_labels = helper_c0_labels[:, :distill_len]
                        kd_valid = kd_labels != self.ignore_index
                        direct_speech_text_c0_kd_valid_tokens = kd_valid.sum()
                        if int(direct_speech_text_c0_kd_valid_tokens.detach().cpu().item()) > 0:
                            temperature = float(self.direct_text_to_c0_kd_temperature)
                            speech_logits = logits[:, :distill_len, 0, :][kd_valid].float()
                            text_teacher_logits = helper_c0_logits[:, :distill_len, :][
                                kd_valid
                            ].detach().float()
                            direct_speech_text_c0_kd_loss = (
                                F.kl_div(
                                    F.log_softmax(speech_logits / temperature, dim=-1),
                                    F.softmax(text_teacher_logits / temperature, dim=-1),
                                    reduction="batchmean",
                                ).clamp_min(0.0)
                                * (temperature ** 2)
                            ).to(dtype=logits.dtype)
                        else:
                            direct_speech_text_c0_kd_loss = logits.new_zeros(())

            if self.enable_c0_auxiliary_loss:
                if self.c0_auxiliary_head is None:
                    raise RuntimeError("C0 auxiliary loss is enabled but the auxiliary head is not initialized.")
                c0_aux_logits = self.c0_auxiliary_head(codec_temporal_hidden)
                c0_aux_labels = codebook_labels[:, :, 0]
                c0_auxiliary_valid_tokens = (c0_aux_labels.reshape(-1) != self.ignore_index).sum()
                if int(c0_auxiliary_valid_tokens.detach().cpu().item()) > 0:
                    c0_auxiliary_loss = self._level0_cross_entropy_loss(
                        c0_aux_logits,
                        c0_aux_labels,
                    )
                else:
                    c0_auxiliary_loss = c0_aux_logits.new_zeros(())

            per_level_losses = []
            per_level_valid_tokens = []
            per_level_error_rates = []
            for level in range(self.num_codebook_levels):
                level_logits = logits[:, :, level, :]
                level_labels = codebook_labels[:, :, level]
                valid_level = level_labels.reshape(-1) != self.ignore_index
                per_level_valid_tokens.append(valid_level.sum())
                if int(valid_level.sum().detach().cpu().item()) == 0:
                    per_level_losses.append(level_logits.new_zeros(()))
                    per_level_error_rates.append(level_logits.new_zeros(()))
                    continue
                level_label_smoothing = 0.0
                if level == 0 and (
                    self.training or not self.level0_label_smoothing_train_only
                ):
                    level_label_smoothing = float(self.level0_label_smoothing)
                level_loss = F.cross_entropy(
                    level_logits.reshape(-1, level_logits.size(-1))[valid_level],
                    level_labels.reshape(-1)[valid_level],
                    label_smoothing=level_label_smoothing,
                )
                per_level_losses.append(level_loss)
                with torch.no_grad():
                    level_pred = level_logits.detach().reshape(-1, level_logits.size(-1)).argmax(dim=-1)
                    level_gold = level_labels.reshape(-1)
                    level_errors = level_pred[valid_level] != level_gold[valid_level]
                    per_level_error_rates.append(
                        level_errors.sum().to(level_logits.dtype)
                        / valid_level.sum().clamp_min(1).to(level_logits.dtype)
                    )

            if self.enable_c0_teacher_distill:
                if c0_teacher_topk_ids is None or c0_teacher_topk_log_probs is None:
                    raise RuntimeError(
                        "enable_c0_teacher_distill=True, but c0_teacher_topk_ids/log_probs "
                        "were not provided by the collator. Use a TSV with precomputed teacher top-k columns."
                    )
                teacher_ids = c0_teacher_topk_ids.to(device=logits.device, dtype=torch.long)
                teacher_log_probs = c0_teacher_topk_log_probs.to(device=logits.device, dtype=logits.dtype)
                distill_len = min(
                    int(logits.size(1)),
                    int(teacher_ids.size(1)),
                    int(teacher_log_probs.size(1)),
                )
                if distill_len > 0:
                    teacher_ids = teacher_ids[:, :distill_len, :]
                    teacher_log_probs = teacher_log_probs[:, :distill_len, :]
                    c0_labels_for_distill = codebook_labels[:, :distill_len, 0]
                    c0_logits_for_distill = logits[:, :distill_len, 0, :]
                    valid_teacher = (
                        (teacher_ids >= 0)
                        & (teacher_ids < int(self.vocab_size))
                        & torch.isfinite(teacher_log_probs)
                    )
                    valid_tokens = (
                        (c0_labels_for_distill != self.ignore_index)
                        & valid_teacher.any(dim=-1)
                    )
                    c0_teacher_distill_valid_tokens = valid_tokens.sum()
                    if int(c0_teacher_distill_valid_tokens.detach().cpu().item()) > 0:
                        flat_student_logits = c0_logits_for_distill[valid_tokens]
                        flat_teacher_ids = teacher_ids[valid_tokens]
                        flat_teacher_log_probs = teacher_log_probs[valid_tokens]
                        flat_valid_teacher = valid_teacher[valid_tokens]
                        safe_teacher_ids = flat_teacher_ids.clamp(min=0, max=int(self.vocab_size) - 1)
                        temperature = float(self.c0_teacher_distill_temperature)
                        student_log_probs = F.log_softmax(
                            flat_student_logits / temperature,
                            dim=-1,
                        )
                        gathered_student_log_probs = student_log_probs.gather(
                            dim=-1,
                            index=safe_teacher_ids,
                        ).masked_fill(~flat_valid_teacher, 0.0)
                        teacher_log_probs_masked = flat_teacher_log_probs.masked_fill(
                            ~flat_valid_teacher,
                            -float("inf"),
                        )
                        teacher_log_mass = torch.logsumexp(
                            teacher_log_probs_masked,
                            dim=-1,
                        )
                        teacher_norm_log_probs = teacher_log_probs_masked - teacher_log_mass.unsqueeze(-1)
                        teacher_probs = teacher_norm_log_probs.exp().masked_fill(
                            ~flat_valid_teacher,
                            0.0,
                        )
                        kl_per_token = (
                            teacher_probs
                            * (teacher_norm_log_probs.masked_fill(~flat_valid_teacher, 0.0) - gathered_student_log_probs)
                        ).sum(dim=-1)
                        c0_teacher_distill_loss = (
                            kl_per_token.mean().clamp_min(0.0) * (temperature ** 2)
                        )
                        c0_teacher_distill_topk_mass = teacher_log_mass.exp().mean()
                    else:
                        c0_teacher_distill_loss = logits.new_zeros(())
                        c0_teacher_distill_topk_mass = logits.new_zeros(())

            if self.enable_c0_gold_anchored_distill:
                if c0_teacher_topk_ids is None or c0_teacher_topk_log_probs is None:
                    raise RuntimeError(
                        "enable_c0_gold_anchored_distill=True, but c0_teacher_topk_ids/log_probs "
                        "were not provided by the collator. Use a TSV with precomputed teacher top-k columns."
                    )
                teacher_ids = c0_teacher_topk_ids.to(device=logits.device, dtype=torch.long)
                teacher_log_probs = c0_teacher_topk_log_probs.to(device=logits.device, dtype=logits.dtype)
                distill_len = min(
                    int(logits.size(1)),
                    int(teacher_ids.size(1)),
                    int(teacher_log_probs.size(1)),
                )
                if distill_len > 0:
                    teacher_ids = teacher_ids[:, :distill_len, :]
                    teacher_log_probs = teacher_log_probs[:, :distill_len, :]
                    c0_labels_for_distill = codebook_labels[:, :distill_len, 0]
                    c0_logits_for_distill = logits[:, :distill_len, 0, :]
                    valid_teacher = (
                        (teacher_ids >= 0)
                        & (teacher_ids < int(self.vocab_size))
                        & torch.isfinite(teacher_log_probs)
                    )
                    valid_tokens = (
                        (c0_labels_for_distill != self.ignore_index)
                        & valid_teacher.any(dim=-1)
                    )
                    c0_gold_anchored_distill_valid_tokens = valid_tokens.sum()
                    if int(c0_gold_anchored_distill_valid_tokens.detach().cpu().item()) > 0:
                        flat_student_logits = c0_logits_for_distill[valid_tokens]
                        flat_gold = c0_labels_for_distill[valid_tokens].long()
                        flat_teacher_ids = teacher_ids[valid_tokens]
                        flat_teacher_log_probs = teacher_log_probs[valid_tokens]
                        flat_valid_teacher = valid_teacher[valid_tokens]
                        safe_teacher_ids = flat_teacher_ids.clamp(min=0, max=int(self.vocab_size) - 1)
                        safe_gold = flat_gold.clamp(min=0, max=int(self.vocab_size) - 1)
                        temperature = float(self.c0_gold_anchored_distill_temperature)
                        student_log_probs = F.log_softmax(
                            flat_student_logits / temperature,
                            dim=-1,
                        )
                        gathered_teacher_student_log_probs = student_log_probs.gather(
                            dim=-1,
                            index=safe_teacher_ids,
                        ).masked_fill(~flat_valid_teacher, 0.0)
                        teacher_log_probs_masked = flat_teacher_log_probs.masked_fill(
                            ~flat_valid_teacher,
                            -float("inf"),
                        )
                        teacher_log_mass = torch.logsumexp(
                            teacher_log_probs_masked,
                            dim=-1,
                        )
                        teacher_norm_log_probs = teacher_log_probs_masked - teacher_log_mass.unsqueeze(-1)
                        teacher_probs = teacher_norm_log_probs.exp().masked_fill(
                            ~flat_valid_teacher,
                            0.0,
                        )
                        teacher_ce = -(
                            teacher_probs * gathered_teacher_student_log_probs
                        ).sum(dim=-1)
                        gold_log_probs = student_log_probs.gather(
                            dim=-1,
                            index=safe_gold.unsqueeze(-1),
                        ).squeeze(-1)
                        gold_ce = -gold_log_probs
                        gold_mass = float(self.c0_gold_anchor_mass)
                        teacher_mass = 1.0 - gold_mass
                        c0_gold_anchored_distill_loss = (
                            gold_mass * gold_ce + teacher_mass * teacher_ce
                        ).mean() * (temperature ** 2)
                        c0_gold_anchored_distill_topk_mass = teacher_log_mass.exp().mean()
                        c0_gold_anchored_distill_teacher_top1_acc = (
                            (flat_teacher_ids[:, 0] == flat_gold)
                            & flat_valid_teacher[:, 0]
                        ).to(torch.float32).mean()
                        c0_gold_anchored_distill_teacher_topk_hit = (
                            (flat_teacher_ids == flat_gold.unsqueeze(-1))
                            & flat_valid_teacher
                        ).any(dim=-1).to(torch.float32).mean()
                    else:
                        c0_gold_anchored_distill_loss = logits.new_zeros(())
                        c0_gold_anchored_distill_topk_mass = logits.new_zeros(())
                        c0_gold_anchored_distill_teacher_top1_acc = logits.new_zeros(())
                        c0_gold_anchored_distill_teacher_topk_hit = logits.new_zeros(())

            if self.enable_c0_teacher_hidden_distill:
                if c0_teacher_hidden is None:
                    raise RuntimeError(
                        "enable_c0_teacher_hidden_distill=True, but c0_teacher_hidden "
                        "was not provided by the collator. Use a TSV with precomputed teacher hidden columns."
                    )
                if self.c0_teacher_hidden_projection is None:
                    raise RuntimeError(
                        "C0 teacher hidden distillation is enabled but the projection head is missing."
                    )
                teacher_hidden = c0_teacher_hidden.to(
                    device=codec_temporal_hidden.device,
                    dtype=codec_temporal_hidden.dtype,
                )
                distill_len = min(
                    int(codec_temporal_hidden.size(1)),
                    int(teacher_hidden.size(1)),
                    int(codebook_labels.size(1)),
                )
                if distill_len > 0:
                    student_hidden = codec_temporal_hidden[:, :distill_len, :]
                    teacher_hidden = teacher_hidden[:, :distill_len, :]
                    student_hidden = self.c0_teacher_hidden_projection(student_hidden)
                    if int(student_hidden.size(-1)) != int(teacher_hidden.size(-1)):
                        raise RuntimeError(
                            "C0 teacher hidden dim mismatch after projection: "
                            f"student={tuple(student_hidden.shape)}, teacher={tuple(teacher_hidden.shape)}."
                        )
                    valid_tokens = codebook_labels[:, :distill_len, 0] != self.ignore_index
                    if c0_teacher_hidden_mask is not None:
                        teacher_mask = c0_teacher_hidden_mask.to(
                            device=valid_tokens.device,
                            dtype=torch.bool,
                        )
                        valid_tokens = valid_tokens & teacher_mask[:, :distill_len]
                    finite_teacher = torch.isfinite(teacher_hidden).all(dim=-1)
                    valid_tokens = valid_tokens & finite_teacher
                    c0_teacher_hidden_distill_valid_tokens = valid_tokens.sum()
                    if int(c0_teacher_hidden_distill_valid_tokens.detach().cpu().item()) > 0:
                        flat_student = student_hidden[valid_tokens].float()
                        flat_teacher = teacher_hidden[valid_tokens].detach().float()
                        if self.c0_teacher_hidden_loss_type == "cosine":
                            flat_student = F.normalize(flat_student, dim=-1)
                            flat_teacher = F.normalize(flat_teacher, dim=-1)
                            cosine = (flat_student * flat_teacher).sum(dim=-1).clamp(-1.0, 1.0)
                            c0_teacher_hidden_distill_loss = (1.0 - cosine).mean().to(
                                dtype=codec_temporal_hidden.dtype
                            )
                            c0_teacher_hidden_distill_cosine = cosine.mean().to(
                                dtype=codec_temporal_hidden.dtype
                            )
                        else:
                            c0_teacher_hidden_distill_loss = F.mse_loss(
                                flat_student,
                                flat_teacher,
                                reduction="mean",
                            ).to(dtype=codec_temporal_hidden.dtype)
                    else:
                        c0_teacher_hidden_distill_loss = codec_temporal_hidden.new_zeros(())
                        c0_teacher_hidden_distill_cosine = codec_temporal_hidden.new_zeros(())

            level0_loss = unified_code0_loss if unified_code0_loss is not None else per_level_losses[0]
            objective_per_level_losses = list(per_level_losses)
            if (
                self.use_unified_code0_loss_in_objective
                and unified_code0_loss is not None
                and objective_per_level_losses
            ):
                objective_per_level_losses[0] = unified_code0_loss
            level0_valid_tokens = per_level_valid_tokens[0]
            mimi_code0_loss = None
            mimi_code0_valid_tokens = None
            mimi_code1_loss = None
            mimi_code1_valid_tokens = None
            mimi_to_model_order = getattr(self, "codebook_mimi_to_model_order", None)
            if mimi_to_model_order is None:
                mimi_to_model_order = list(range(self.num_codebook_levels))
            else:
                mimi_to_model_order = list(mimi_to_model_order)
            if len(mimi_to_model_order) >= 1:
                model_level = int(mimi_to_model_order[0])
                if 0 <= model_level < len(per_level_losses):
                    mimi_code0_loss = per_level_losses[model_level]
                    mimi_code0_valid_tokens = per_level_valid_tokens[model_level]
            if len(mimi_to_model_order) >= 2:
                model_level = int(mimi_to_model_order[1])
                if 0 <= model_level < len(per_level_losses):
                    mimi_code1_loss = per_level_losses[model_level]
                    mimi_code1_valid_tokens = per_level_valid_tokens[model_level]
            temporal_levels = max(1, min(self.num_temporal_codebook_levels, self.num_codebook_levels))
            temporal_loss = sum(objective_per_level_losses[:temporal_levels]) / temporal_levels
            temporal_valid_tokens = sum(per_level_valid_tokens[:temporal_levels])
            all_codebook_loss = sum(objective_per_level_losses) / self.num_codebook_levels
            all_valid_tokens = sum(per_level_valid_tokens)
            depth_start_level = max(
                temporal_levels,
                self.depth_objective_start_level,
            )
            if self.num_codebook_levels > depth_start_level:
                depth_loss = sum(objective_per_level_losses[depth_start_level:]) / (
                    self.num_codebook_levels - depth_start_level
                )
                depth_valid_tokens = sum(per_level_valid_tokens[depth_start_level:])
            else:
                depth_loss = temporal_loss.new_zeros(())
                depth_valid_tokens = temporal_valid_tokens.new_zeros(())

            if self.enable_quality_cot:
                if (
                    quality_cot_source_text_loss is None
                    or quality_cot_target_text_loss is None
                    or unified_code0_loss is None
                    or depth_loss is None
                ):
                    raise RuntimeError(
                        "Quality-CoT objective requires source-text, target-text, unified C0, and depth losses."
                    )
                loss = (
                    self.quality_cot_source_text_loss_weight * quality_cot_source_text_loss
                    + self.quality_cot_target_text_loss_weight * quality_cot_target_text_loss
                    + self.quality_cot_code0_loss_weight * unified_code0_loss
                    + self.quality_cot_depth_loss_weight * depth_loss
                )
                if self.quality_cot_code1_loss_weight > 0.0:
                    if mimi_code1_loss is None:
                        raise RuntimeError(
                            "quality_cot_code1_loss_weight > 0 requires a Mimi C1 loss."
                        )
                    # For source-prompt NAR depth this is
                    # -log P(target C1 | source Mimi sequence, target C0).
                    loss = loss + self.quality_cot_code1_loss_weight * mimi_code1_loss
                if text_path_text_loss is not None and self.text_codec_t2t_loss_weight > 0.0:
                    loss = loss + self.text_codec_t2t_loss_weight * text_path_text_loss
                if text_path_code0_loss is not None and self.text_codec_t2c_loss_weight > 0.0:
                    loss = loss + self.text_codec_t2c_loss_weight * text_path_code0_loss
                if self.enable_target_only_speech_auxiliary:
                    if (
                        target_only_speech_text_loss is None
                        or target_only_speech_code0_loss is None
                    ):
                        raise RuntimeError(
                            "Target-only speech auxiliary losses were not computed."
                        )
                    loss = (
                        loss
                        + self.target_only_speech_text_loss_weight
                        * target_only_speech_text_loss
                        + self.target_only_speech_code0_loss_weight
                        * target_only_speech_code0_loss
                    )
                if mixed_source_target_text_loss is not None:
                    loss = (
                        loss
                        + mixed_source_target_effective_weight
                        * mixed_source_target_text_loss
                    )
                if predicted_source_target_text_loss is not None:
                    loss = (
                        loss
                        + predicted_source_target_effective_weight
                        * predicted_source_target_text_loss
                    )
                quality_cot_objective_loss = loss
            elif self.enable_transvip_style_loss_aggregation:
                transvip_style_speech_loss = _token_weighted_average(
                    [
                        (text_loss, text_valid_tokens),
                        (unified_code0_loss, codec_mass_valid_tokens),
                    ]
                )
                transvip_style_text_loss = _token_weighted_average(
                    [
                        (text_path_text_loss, text_path_text_valid_tokens),
                        (text_path_code0_loss, text_path_code0_valid_tokens),
                    ]
                )
                transvip_style_depth_loss = depth_loss
                if (
                    transvip_style_speech_loss is not None
                    and self.transvip_style_speech_loss_weight > 0.0
                ):
                    loss = self.transvip_style_speech_loss_weight * transvip_style_speech_loss
                if (
                    transvip_style_text_loss is not None
                    and self.transvip_style_text_loss_weight > 0.0
                ):
                    term = self.transvip_style_text_loss_weight * transvip_style_text_loss
                    loss = term if loss is None else loss + term
                if (
                    text_codec_kd_loss is not None
                    and self.transvip_style_kd_loss_weight > 0.0
                ):
                    term = self.transvip_style_kd_loss_weight * text_codec_kd_loss
                    loss = term if loss is None else loss + term
                if (
                    transvip_style_depth_loss is not None
                    and self.transvip_style_depth_loss_weight > 0.0
                ):
                    term = self.transvip_style_depth_loss_weight * transvip_style_depth_loss
                    loss = term if loss is None else loss + term
                transvip_style_total_loss = loss
            elif self.codebook_loss_weights is not None:
                weights = torch.as_tensor(
                    self.codebook_loss_weights,
                    dtype=per_level_losses[0].dtype,
                    device=per_level_losses[0].device,
                )
                stacked_losses = torch.stack(objective_per_level_losses)
                loss = (stacked_losses * weights).sum() / weights.sum().clamp_min(1.0e-12)
            elif self.level0_depth_objective:
                loss = self.level0_loss_weight * temporal_loss + self.depth_loss_weight * depth_loss
            else:
                loss = all_codebook_loss

        if not self.enable_transvip_style_loss_aggregation and not self.enable_quality_cot:
            if (
                codec_mass_loss is not None
                and self.enable_codec_mass_loss
                and self.codec_mass_loss_weight > 0.0
            ):
                if loss is None:
                    loss = self.codec_mass_loss_weight * codec_mass_loss
                else:
                    loss = loss + self.codec_mass_loss_weight * codec_mass_loss

            if (
                c0_auxiliary_loss is not None
                and self.enable_c0_auxiliary_loss
                and self.c0_auxiliary_loss_weight > 0.0
            ):
                if loss is None:
                    loss = self.c0_auxiliary_loss_weight * c0_auxiliary_loss
                else:
                    loss = loss + self.c0_auxiliary_loss_weight * c0_auxiliary_loss

        if (
            c0_teacher_distill_loss is not None
            and self.enable_c0_teacher_distill
            and self.c0_teacher_distill_weight > 0.0
        ):
            term = self.c0_teacher_distill_weight * c0_teacher_distill_loss
            loss = term if loss is None else loss + term

        if (
            c0_gold_anchored_distill_loss is not None
            and self.enable_c0_gold_anchored_distill
            and self.c0_gold_anchored_distill_weight > 0.0
        ):
            term = self.c0_gold_anchored_distill_weight * c0_gold_anchored_distill_loss
            loss = term if loss is None else loss + term

        if (
            c0_teacher_hidden_distill_loss is not None
            and self.enable_c0_teacher_hidden_distill
            and self.c0_teacher_hidden_distill_weight > 0.0
        ):
            term = self.c0_teacher_hidden_distill_weight * c0_teacher_hidden_distill_loss
            loss = term if loss is None else loss + term

        if (
            direct_text_to_c0_loss is not None
            and self.enable_direct_text_to_c0_auxiliary
            and self.direct_text_to_c0_loss_weight > 0.0
        ):
            term = self.direct_text_to_c0_loss_weight * direct_text_to_c0_loss
            loss = term if loss is None else loss + term

        if (
            direct_speech_text_c0_kd_loss is not None
            and self.enable_direct_text_to_c0_auxiliary
            and self.direct_text_to_c0_kd_weight > 0.0
        ):
            term = self.direct_text_to_c0_kd_weight * direct_speech_text_c0_kd_loss
            loss = term if loss is None else loss + term

        if (
            campplus_depth_identity_loss is not None
            and self.campplus_depth_identity_loss_weight > 0.0
        ):
            term = (
                self.campplus_depth_identity_loss_weight
                * campplus_depth_identity_loss
            )
            loss = term if loss is None else loss + term

        codec_objective_loss = loss
        if not self.enable_transvip_style_loss_aggregation and not self.enable_quality_cot:
            if text_loss is not None:
                if loss is None:
                    loss = self.text_loss_weight * text_loss
                else:
                    loss = loss + self.text_loss_weight * text_loss
            if (
                text_path_text_loss is not None
                and self.text_codec_t2t_loss_weight > 0.0
            ):
                if loss is None:
                    loss = self.text_codec_t2t_loss_weight * text_path_text_loss
                else:
                    loss = loss + self.text_codec_t2t_loss_weight * text_path_text_loss
            if (
                text_path_code0_loss is not None
                and self.text_codec_t2c_loss_weight > 0.0
            ):
                if loss is None:
                    loss = self.text_codec_t2c_loss_weight * text_path_code0_loss
                else:
                    loss = loss + self.text_codec_t2c_loss_weight * text_path_code0_loss
            if (
                text_codec_kd_loss is not None
                and self.text_codec_kd_loss_weight > 0.0
            ):
                if loss is None:
                    loss = self.text_codec_kd_loss_weight * text_codec_kd_loss
                else:
                    loss = loss + self.text_codec_kd_loss_weight * text_codec_kd_loss
            if source_unit_loss is not None:
                if loss is None:
                    loss = self.source_unit_loss_weight * source_unit_loss
                else:
                    loss = loss + self.source_unit_loss_weight * source_unit_loss
            if source_unit_transformer_loss is not None:
                if loss is None:
                    loss = (
                        self.source_unit_transformer_loss_weight
                        * source_unit_transformer_loss
                    )
                else:
                    loss = loss + (
                        self.source_unit_transformer_loss_weight
                        * source_unit_transformer_loss
                    )
        if loss is not None and not bool(torch.isfinite(loss).detach().all().item()):
            raise FloatingPointError(
                "Non-finite training loss detected. Check upstream logits/masks before continuing; "
                "continuing would let Trainer log a misleading loss=0.0."
            )

        diagnostic_objective = getattr(self, "_gradient_diag_objective", None)
        if diagnostic_objective is not None:
            if not self.enable_quality_cot:
                raise RuntimeError(
                    "Gradient-conflict diagnostics require enable_quality_cot=true."
                )
            diagnostic_losses: Dict[str, Optional[torch.Tensor]] = {
                "all": loss,
                "text": (
                    self.quality_cot_source_text_loss_weight
                    * quality_cot_source_text_loss
                    + self.quality_cot_target_text_loss_weight
                    * quality_cot_target_text_loss
                ),
                "c0": self.quality_cot_code0_loss_weight * unified_code0_loss,
                "depth": self.quality_cot_depth_loss_weight * depth_loss,
                "predsrc": (
                    predicted_source_target_effective_weight
                    * predicted_source_target_text_loss
                    if predicted_source_target_text_loss is not None
                    and predicted_source_target_effective_weight > 0.0
                    else None
                ),
                "identity": (
                    self.campplus_depth_identity_loss_weight
                    * campplus_depth_identity_loss
                    if campplus_depth_identity_loss is not None
                    and self.campplus_depth_identity_loss_weight > 0.0
                    else None
                ),
            }
            if text_path_text_loss is not None and self.text_codec_t2t_loss_weight > 0.0:
                diagnostic_losses["text"] = (
                    diagnostic_losses["text"]
                    + self.text_codec_t2t_loss_weight * text_path_text_loss
                )
            if text_path_code0_loss is not None and self.text_codec_t2c_loss_weight > 0.0:
                diagnostic_losses["c0"] = (
                    diagnostic_losses["c0"]
                    + self.text_codec_t2c_loss_weight * text_path_code0_loss
                )
            if self.quality_cot_code1_loss_weight > 0.0:
                diagnostic_losses["depth"] = (
                    diagnostic_losses["depth"]
                    + self.quality_cot_code1_loss_weight * mimi_code1_loss
                )
            diagnostic_loss = diagnostic_losses.get(diagnostic_objective)
            if diagnostic_loss is None:
                raise RuntimeError(
                    f"Gradient diagnostic objective {diagnostic_objective!r} is unavailable "
                    f"at global step {self._current_global_step}."
                )
            # Returning the selected loss through the normal FSDP output path is
            # essential: FSDP installs its supported pre-backward unshard hook
            # here, unlike torch.autograd.grad() on an internal activation.
            return {"loss": diagnostic_loss, "logits": logits.detach()}

        outputs = {"loss": loss, "logits": logits}
        if level0_loss is not None:
            outputs.update(
                {
                    "code0_ce_loss": level0_loss.detach(),
                    "temporal_ce_loss": temporal_loss.detach(),
                    "depth_ce_loss": depth_loss.detach(),
                    "all_codebook_ce_loss": all_codebook_loss.detach(),
                    "objective_loss": codec_objective_loss.detach(),
                    "code0_valid_tokens": level0_valid_tokens.detach().to(torch.float32),
                    "temporal_valid_tokens": temporal_valid_tokens.detach().to(torch.float32),
                    "depth_valid_tokens": depth_valid_tokens.detach().to(torch.float32),
                    "all_valid_tokens": all_valid_tokens.detach().to(torch.float32),
                    "depth_teacher_forced": torch.tensor(
                        float(effective_depth_teacher_force),
                        device=logits.device,
                    ),
                    "temporal_scheduled_sampling_prob": torch.tensor(
                        float(effective_temporal_scheduled_sampling_prob),
                        device=logits.device,
                    ),
                    "depth_scheduled_sampling_prob": torch.tensor(
                        float(effective_depth_scheduled_sampling_prob),
                        device=logits.device,
                    ),
                    "depth_chain_scheduled_sampling_prob": torch.tensor(
                        float(effective_depth_chain_scheduled_sampling_prob),
                        device=logits.device,
                    ),
                }
            )
            if mimi_code0_loss is not None and mimi_code0_valid_tokens is not None:
                outputs.update(
                    {
                        "mimi_code0_ce_loss": mimi_code0_loss.detach(),
                        "mimi_code0_valid_tokens": mimi_code0_valid_tokens.detach().to(torch.float32),
                    }
                )
            if mimi_code1_loss is not None and mimi_code1_valid_tokens is not None:
                outputs.update(
                    {
                        "mimi_code1_ce_loss": mimi_code1_loss.detach(),
                        "mimi_code1_valid_tokens": mimi_code1_valid_tokens.detach().to(torch.float32),
                    }
                )
            if campplus_depth_identity_loss is not None:
                outputs["campplus_depth_identity_loss"] = (
                    campplus_depth_identity_loss.detach()
                )
            if (
                quality_cot_source_text_loss is not None
                and quality_cot_source_text_valid_tokens is not None
            ):
                outputs.update(
                    {
                        "quality_cot_source_text_ce_loss": quality_cot_source_text_loss.detach(),
                        "quality_cot_source_text_valid_tokens": quality_cot_source_text_valid_tokens.detach().to(
                            torch.float32
                        ),
                    }
                )
            if (
                quality_cot_target_text_loss is not None
                and quality_cot_target_text_valid_tokens is not None
            ):
                outputs.update(
                    {
                        "quality_cot_target_text_ce_loss": quality_cot_target_text_loss.detach(),
                        "quality_cot_target_text_valid_tokens": quality_cot_target_text_valid_tokens.detach().to(
                            torch.float32
                        ),
                    }
                )
            if quality_cot_objective_loss is not None:
                outputs["quality_cot_objective_loss"] = quality_cot_objective_loss.detach()
            if (
                target_only_speech_text_loss is not None
                and target_only_speech_text_valid_tokens is not None
            ):
                outputs.update(
                    {
                        "target_only_speech_text_ce_loss": target_only_speech_text_loss.detach(),
                        "target_only_speech_text_valid_tokens": target_only_speech_text_valid_tokens.detach().to(
                            torch.float32
                        ),
                    }
                )
            if (
                target_only_speech_code0_loss is not None
                and target_only_speech_code0_valid_tokens is not None
            ):
                outputs.update(
                    {
                        "target_only_speech_code0_ce_loss": target_only_speech_code0_loss.detach(),
                        "target_only_speech_code0_valid_tokens": target_only_speech_code0_valid_tokens.detach().to(
                            torch.float32
                        ),
                    }
                )
            if (
                mixed_source_target_text_loss is not None
                and mixed_source_target_text_valid_tokens is not None
            ):
                outputs.update(
                    {
                        "mixed_source_target_text_ce_loss": mixed_source_target_text_loss.detach(),
                        "mixed_source_target_text_valid_tokens": mixed_source_target_text_valid_tokens.detach().to(
                            torch.float32
                        ),
                    }
                )
            if (
                mixed_source_replacement_ratio is not None
                and mixed_source_replacement_eligible_tokens is not None
            ):
                outputs.update(
                    {
                        "mixed_source_replacement_ratio": mixed_source_replacement_ratio.detach(),
                        "mixed_source_replacement_eligible_tokens": mixed_source_replacement_eligible_tokens.detach().to(
                            torch.float32
                        ),
                    }
                )
            if self.enable_predicted_source_target_auxiliary and self.training:
                outputs["predicted_source_target_effective_weight"] = torch.tensor(
                    float(predicted_source_target_effective_weight),
                    dtype=logits.dtype,
                    device=logits.device,
                )
            if (
                predicted_source_target_text_loss is not None
                and predicted_source_target_text_valid_tokens is not None
            ):
                outputs.update(
                    {
                        "predicted_source_target_text_ce_loss": (
                            predicted_source_target_text_loss.detach()
                        ),
                        "predicted_source_target_text_valid_tokens": (
                            predicted_source_target_text_valid_tokens.detach().to(
                                torch.float32
                            )
                        ),
                    }
                )
            if predicted_source_target_mean_tokens is not None:
                outputs["predicted_source_target_mean_tokens"] = (
                    predicted_source_target_mean_tokens.detach()
                )
            if text_loss is not None and text_valid_tokens is not None:
                outputs.update(
                    {
                        "s2t_ce_loss": text_loss.detach(),
                        "s2t_valid_tokens": text_valid_tokens.detach().to(torch.float32),
                    }
                )
            if unified_code0_loss is not None and codec_mass_valid_tokens is not None:
                outputs.update(
                    {
                        "s2c_ce_loss": unified_code0_loss.detach(),
                        "s2c_valid_tokens": codec_mass_valid_tokens.detach().to(torch.float32),
                    }
                )
            if (
                text_path_text_loss is not None
                and text_path_text_valid_tokens is not None
            ):
                outputs.update(
                    {
                        "t2t_ce_loss": text_path_text_loss.detach(),
                        "t2t_valid_tokens": text_path_text_valid_tokens.detach().to(torch.float32),
                    }
                )
            if (
                text_path_code0_loss is not None
                and text_path_code0_valid_tokens is not None
            ):
                outputs.update(
                    {
                        "t2c_ce_loss": text_path_code0_loss.detach(),
                        "t2c_valid_tokens": text_path_code0_valid_tokens.detach().to(torch.float32),
                    }
                )
            if codec_mass_loss is not None and codec_mass_valid_tokens is not None:
                outputs.update(
                    {
                        "codec_mass_loss": codec_mass_loss.detach(),
                        "codec_mass_prob": codec_mass_prob.detach(),
                        "codec_mass_valid_tokens": codec_mass_valid_tokens.detach().to(torch.float32),
                    }
                )
            if text_codec_kd_loss is not None and text_codec_kd_valid_tokens is not None:
                outputs.update(
                    {
                        "speech_text_kd_loss": text_codec_kd_loss.detach(),
                        "speech_text_kd_valid_tokens": text_codec_kd_valid_tokens.detach().to(torch.float32),
                    }
                )
            if transvip_style_speech_loss is not None:
                outputs.update(
                    {
                        "transvip_style_speech_ce_loss": transvip_style_speech_loss.detach(),
                    }
                )
            if transvip_style_text_loss is not None:
                outputs.update(
                    {
                        "transvip_style_text_ce_loss": transvip_style_text_loss.detach(),
                    }
                )
            if transvip_style_depth_loss is not None:
                outputs.update(
                    {
                        "transvip_style_depth_ce_loss": transvip_style_depth_loss.detach(),
                    }
                )
            if transvip_style_total_loss is not None:
                outputs.update(
                    {
                        "transvip_style_total_loss": transvip_style_total_loss.detach(),
                    }
                )
            if c0_auxiliary_loss is not None and c0_auxiliary_valid_tokens is not None:
                outputs.update(
                    {
                        "c0_auxiliary_ce_loss": c0_auxiliary_loss.detach(),
                        "c0_auxiliary_valid_tokens": c0_auxiliary_valid_tokens.detach().to(torch.float32),
                    }
                )
            if direct_text_to_c0_loss is not None and direct_text_to_c0_valid_tokens is not None:
                outputs.update(
                    {
                        "direct_text_to_c0_ce_loss": direct_text_to_c0_loss.detach(),
                        "direct_text_to_c0_valid_tokens": direct_text_to_c0_valid_tokens.detach().to(torch.float32),
                    }
                )
                if direct_text_to_c0_error_rate is not None:
                    outputs["direct_text_to_c0_error_rate"] = direct_text_to_c0_error_rate.detach()
            if (
                direct_speech_text_c0_kd_loss is not None
                and direct_speech_text_c0_kd_valid_tokens is not None
            ):
                outputs.update(
                    {
                        "direct_speech_text_c0_kd_loss": direct_speech_text_c0_kd_loss.detach(),
                        "direct_speech_text_c0_kd_valid_tokens": direct_speech_text_c0_kd_valid_tokens.detach().to(torch.float32),
                    }
                )
            if c0_teacher_distill_loss is not None and c0_teacher_distill_valid_tokens is not None:
                outputs.update(
                    {
                        "c0_teacher_distill_kl_loss": c0_teacher_distill_loss.detach(),
                        "c0_teacher_distill_valid_tokens": c0_teacher_distill_valid_tokens.detach().to(torch.float32),
                    }
                )
                if c0_teacher_distill_topk_mass is not None:
                    outputs["c0_teacher_distill_topk_mass"] = c0_teacher_distill_topk_mass.detach()
            if (
                c0_gold_anchored_distill_loss is not None
                and c0_gold_anchored_distill_valid_tokens is not None
            ):
                outputs.update(
                    {
                        "c0_gold_anchored_distill_loss": c0_gold_anchored_distill_loss.detach(),
                        "c0_gold_anchored_distill_valid_tokens": c0_gold_anchored_distill_valid_tokens.detach().to(torch.float32),
                    }
                )
                if c0_gold_anchored_distill_topk_mass is not None:
                    outputs["c0_gold_anchored_distill_topk_mass"] = (
                        c0_gold_anchored_distill_topk_mass.detach()
                    )
                if c0_gold_anchored_distill_teacher_top1_acc is not None:
                    outputs["c0_gold_anchored_distill_teacher_top1_acc"] = (
                        c0_gold_anchored_distill_teacher_top1_acc.detach()
                    )
                if c0_gold_anchored_distill_teacher_topk_hit is not None:
                    outputs["c0_gold_anchored_distill_teacher_topk_hit"] = (
                        c0_gold_anchored_distill_teacher_topk_hit.detach()
                    )
            if (
                c0_teacher_hidden_distill_loss is not None
                and c0_teacher_hidden_distill_valid_tokens is not None
            ):
                outputs.update(
                    {
                        "c0_teacher_hidden_distill_loss": c0_teacher_hidden_distill_loss.detach(),
                        "c0_teacher_hidden_distill_valid_tokens": c0_teacher_hidden_distill_valid_tokens.detach().to(torch.float32),
                    }
                )
                if c0_teacher_hidden_distill_cosine is not None:
                    outputs["c0_teacher_hidden_distill_cosine"] = (
                        c0_teacher_hidden_distill_cosine.detach()
                    )
            for mimi_level, model_level in enumerate(mimi_to_model_order):
                model_level = int(model_level)
                if not (0 <= model_level < len(per_level_losses)):
                    continue
                outputs.update(
                    {
                        f"mimi_codebook_{mimi_level:02d}_ce_loss": per_level_losses[
                            model_level
                        ].detach(),
                        f"mimi_codebook_{mimi_level:02d}_valid_tokens": per_level_valid_tokens[
                            model_level
                        ].detach().to(torch.float32),
                    }
                )
                if model_level < len(per_level_error_rates):
                    outputs[f"mimi_codebook_{mimi_level:02d}_error_rate"] = (
                        per_level_error_rates[model_level].detach()
                    )
        if text_loss is not None:
            objective_valid_tokens = text_valid_tokens
            if all_valid_tokens is not None:
                objective_valid_tokens = objective_valid_tokens + all_valid_tokens
            text_metric_name = (
                "target_semantic_ce_loss"
                if self.enable_semantic_prefix_ar
                else "text_ce_loss"
            )
            text_token_metric_name = (
                "target_semantic_valid_tokens"
                if self.enable_semantic_prefix_ar
                else "text_valid_tokens"
            )
            objective_metric_name = (
                "objective_with_prefix_loss"
                if self.enable_semantic_prefix_ar
                else "objective_with_text_loss"
            )
            objective_token_metric_name = (
                "objective_with_prefix_valid_tokens"
                if self.enable_semantic_prefix_ar
                else "objective_with_text_valid_tokens"
            )
            outputs.update(
                {
                    text_metric_name: text_loss.detach(),
                    text_token_metric_name: text_valid_tokens.detach().to(torch.float32),
                    objective_metric_name: loss.detach(),
                    objective_token_metric_name: objective_valid_tokens.detach().to(torch.float32),
                }
            )
            if self.enable_semantic_prefix_ar and target_semantic_error_rate is not None:
                outputs["target_semantic_error_rate"] = (
                    target_semantic_error_rate.detach()
                )
            semantic_metric_groups = (
                (
                    "content",
                    target_semantic_content_loss,
                    target_semantic_content_error_rate,
                    target_semantic_content_valid_tokens,
                ),
                (
                    "sep",
                    target_semantic_sep_loss,
                    target_semantic_sep_error_rate,
                    target_semantic_sep_valid_tokens,
                ),
                (
                    "first_token",
                    target_semantic_first_token_loss,
                    target_semantic_first_token_error_rate,
                    target_semantic_first_token_valid_tokens,
                ),
            )
            if self.enable_semantic_prefix_ar:
                for metric_group, group_loss, group_error_rate, group_tokens in semantic_metric_groups:
                    if group_loss is None or group_error_rate is None or group_tokens is None:
                        continue
                    outputs.update(
                        {
                            f"target_semantic_{metric_group}_ce_loss": group_loss.detach(),
                            f"target_semantic_{metric_group}_error_rate": group_error_rate.detach(),
                            f"target_semantic_{metric_group}_valid_tokens": group_tokens.detach().to(
                                torch.float32
                            ),
                        }
                    )
        if source_unit_loss is not None:
            outputs.update(
                {
                    "source_semantic_ctc_loss": source_unit_loss.detach(),
                    "source_semantic_valid_tokens": source_unit_valid_tokens.detach().to(torch.float32),
                }
            )
        if source_unit_transformer_loss is not None:
            outputs.update(
                {
                    "source_semantic_ar_loss": source_unit_transformer_loss.detach(),
                    "source_semantic_ar_error_rate": source_unit_transformer_error_rate.detach(),
                    "source_semantic_ar_valid_tokens": (
                        source_unit_valid_tokens.detach().to(torch.float32)
                        + float(source_unit_ids.size(0))
                    ),
                }
            )
        if return_codec_temporal_hidden:
            outputs["codec_temporal_hidden"] = codec_temporal_hidden.detach()
        return outputs

    def save_pretrained(self, save_directory: str, state_dict: Optional[Dict[str, torch.Tensor]] = None):
        os.makedirs(save_directory, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        with open(os.path.join(save_directory, "model_config.json"), "w") as f:
            json.dump(
                {
                    "architecture": "whisper_temporal_depth_transformer",
                    "whisper_model_name": self.whisper_name,
                    "speech_encoder_type": self.speech_encoder_type,
                    "speech_encoder_name": self.speech_encoder_name,
                    "speech_encoder_output_layer": self.speech_encoder_output_layer,
                    "qwen_model_name": self.qwen_name,
                    "codebook_size": self.codebook_size,
                    "vocab_size": self.vocab_size,  # codebook_size + 2 (BOS/EOS)
                    "bos_token_id": self.codebook_size,
                    "eos_token_id": self.codebook_size + 1,
                    "dataset_num_quantizers": self.dataset_num_quantizers,
                    "input_num_quantizers": self.input_num_quantizers,
                    "num_codebook_levels": self.num_codebook_levels,
                    "num_temporal_codebook_levels": self.num_temporal_codebook_levels,
                    "codebook_model_order": getattr(self, "codebook_model_order", None),
                    "codebook_mimi_to_model_order": getattr(self, "codebook_mimi_to_model_order", None),
                    "temporal_context_levels": self.temporal_context_levels,
                    "temporal_hidden_size": self.temporal_hidden_size,
                    "temporal_num_layers": temporal_num_layers,
                    "temporal_num_heads": temporal_num_heads,
                    "temporal_activation": self.temporal_activation,
                    "init_nllb_decoder_body_checkpoint": init_nllb_decoder_body_checkpoint,
                    "init_nllb_decoder_body_layers": init_nllb_decoder_body_layers,
                    "init_nllb_decoder_body_excludes_cross_attention": bool(
                        init_nllb_decoder_body_checkpoint
                    ),
                    "init_nllb_decoder_body_excludes_text_matrices": bool(
                        init_nllb_decoder_body_checkpoint
                    ),
                    "depth_hidden_size": self.depth_hidden_size,
                    "depth_num_layers": depth_num_layers,
                    "depth_num_heads": depth_num_heads,
                    "level0_depth_objective": self.level0_depth_objective,
                    "detach_temporal_for_depth_loss": self.detach_temporal_for_depth_loss,
                    "temporal_scheduled_sampling_prob": self.temporal_scheduled_sampling_prob,
                    "temporal_scheduled_sampling_start_step": self.temporal_scheduled_sampling_start_step,
                    "temporal_scheduled_sampling_warmup_steps": self.temporal_scheduled_sampling_warmup_steps,
                    "temporal_scheduled_sampling_mode": self.temporal_scheduled_sampling_mode,
                    "temporal_scheduled_sampling_topk": self.temporal_scheduled_sampling_topk,
                    "temporal_scheduled_sampling_temperature": self.temporal_scheduled_sampling_temperature,
                    "temporal_scheduled_sampling_final_prob": self.temporal_scheduled_sampling_final_prob,
                    "temporal_scheduled_sampling_decay_start_step": self.temporal_scheduled_sampling_decay_start_step,
                    "temporal_scheduled_sampling_decay_steps": self.temporal_scheduled_sampling_decay_steps,
                    "temporal_scheduled_sampling_preserve_last_n": self.temporal_scheduled_sampling_preserve_last_n,
                    "depth_scheduled_sampling_prob": self.depth_scheduled_sampling_prob,
                    "depth_scheduled_sampling_start_step": self.depth_scheduled_sampling_start_step,
                    "depth_scheduled_sampling_warmup_steps": self.depth_scheduled_sampling_warmup_steps,
                    "depth_scheduled_sampling_mode": self.depth_scheduled_sampling_mode,
                    "depth_scheduled_sampling_topk": self.depth_scheduled_sampling_topk,
                    "depth_scheduled_sampling_temperature": self.depth_scheduled_sampling_temperature,
                    "depth_chain_scheduled_sampling_prob": self.depth_chain_scheduled_sampling_prob,
                    "depth_chain_scheduled_sampling_start_step": self.depth_chain_scheduled_sampling_start_step,
                    "depth_chain_scheduled_sampling_warmup_steps": self.depth_chain_scheduled_sampling_warmup_steps,
                    "depth_chain_scheduled_sampling_mode": self.depth_chain_scheduled_sampling_mode,
                    "depth_chain_scheduled_sampling_topk": self.depth_chain_scheduled_sampling_topk,
                    "depth_chain_scheduled_sampling_temperature": self.depth_chain_scheduled_sampling_temperature,
                    "level0_loss_weight": self.level0_loss_weight,
                    "depth_loss_weight": self.depth_loss_weight,
                    "enable_c0_auxiliary_loss": self.enable_c0_auxiliary_loss,
                    "c0_auxiliary_loss_weight": self.c0_auxiliary_loss_weight,
                    "enable_c0_teacher_distill": self.enable_c0_teacher_distill,
                    "c0_teacher_distill_weight": self.c0_teacher_distill_weight,
                    "c0_teacher_distill_temperature": self.c0_teacher_distill_temperature,
                    "enable_c0_gold_anchored_distill": self.enable_c0_gold_anchored_distill,
                    "c0_gold_anchored_distill_weight": self.c0_gold_anchored_distill_weight,
                    "c0_gold_anchor_mass": self.c0_gold_anchor_mass,
                    "c0_gold_anchored_distill_temperature": self.c0_gold_anchored_distill_temperature,
                    "enable_c0_teacher_hidden_distill": self.enable_c0_teacher_hidden_distill,
                    "c0_teacher_hidden_distill_weight": self.c0_teacher_hidden_distill_weight,
                    "c0_teacher_hidden_loss_type": self.c0_teacher_hidden_loss_type,
                    "c0_teacher_hidden_dim": self.c0_teacher_hidden_dim,
                    "c0_teacher_hidden_use_projection": self.c0_teacher_hidden_use_projection,
                    "enable_mimi_source_acoustic_conditioning": self.enable_mimi_source_acoustic_conditioning,
                    "mimi_source_acoustic_pooling": self.mimi_source_acoustic_pooling,
                    "use_precomputed_source_acoustic_embeddings": self.use_precomputed_source_acoustic_embeddings,
                    "skip_mimi_source_acoustic_encoder_when_precomputed": self.skip_mimi_source_acoustic_encoder_when_precomputed,
                    "mimi_source_acoustic_residual_scale": self.mimi_source_acoustic_residual_scale,
                    "mimi_source_acoustic_depth_start_level": self.mimi_source_acoustic_depth_start_level,
                    "mimi_source_acoustic_detach": self.mimi_source_acoustic_detach,
                    "enable_mimi_source_style_token_bank": self.enable_mimi_source_style_token_bank,
                    "mimi_source_style_token_count": self.mimi_source_style_token_count,
                    "mimi_source_style_token_depth_start_level": self.mimi_source_style_token_depth_start_level,
                    "mimi_source_style_token_num_heads": self.mimi_source_style_token_num_heads,
                    "mimi_source_style_token_dropout": self.mimi_source_style_token_dropout,
                    "mimi_source_style_token_residual_scale": self.mimi_source_style_token_residual_scale,
                    "mimi_source_style_token_gate_init": self.mimi_source_style_token_gate_init,
                    "mimi_source_style_token_c1_gate_init": self.mimi_source_style_token_c1_gate_init,
                    "enable_mimi_source_acoustic_temporal_conditioning": self.enable_mimi_source_acoustic_temporal_conditioning,
                    "mimi_source_acoustic_temporal_residual_scale": self.mimi_source_acoustic_temporal_residual_scale,
                    "mimi_source_acoustic_temporal_detach": self.mimi_source_acoustic_temporal_detach,
                    "enable_unity_source_memory_fusion": self.enable_unity_source_memory_fusion,
                    "unity_source_memory_fusion_mode": self.unity_source_memory_fusion_mode,
                    "unity_source_memory_source": self.unity_source_memory_source,
                    "unity_source_memory_layer": self.unity_source_memory_layer,
                    "unity_source_memory_scale": self.unity_source_memory_scale,
                    "unity_source_memory_detach": self.unity_source_memory_detach,
                    "enable_mimi_source_speaker_prompt": self.enable_mimi_source_speaker_prompt,
                    "mimi_source_speaker_prompt_detach": self.mimi_source_speaker_prompt_detach,
                    "source_speaker_prompt_backend": self.source_speaker_prompt_backend,
                    "source_speaker_prompt_speech_layer": self.source_speaker_prompt_speech_layer,
                    "source_speaker_prompt_num_layers": self.source_speaker_prompt_num_layers,
                    "source_speaker_prompt_num_heads": self.source_speaker_prompt_num_heads,
                    "source_speaker_prompt_ffn_dim": self.source_speaker_prompt_ffn_dim,
                    "source_speaker_prompt_dropout": self.source_speaker_prompt_dropout,
                    "enable_mimi_nar_depth_prompt": self.enable_mimi_nar_depth_prompt,
                    "mimi_nar_depth_attention_mode": self.mimi_nar_depth_attention_mode,
                    "mimi_nar_depth_max_positions": self.mimi_nar_depth_max_positions,
                    "mimi_nar_depth_teacher_force_level_batch_size": (
                        self.mimi_nar_depth_teacher_force_level_batch_size
                    ),
                    "enable_campplus_depth_conditioning": self.enable_campplus_depth_conditioning,
                    "campplus_depth_conditioning_mode": (
                        self.campplus_depth_conditioning_mode
                    ),
                    "campplus_model_root": self.campplus_model_root,
                    "campplus_checkpoint_path": self.campplus_checkpoint_path,
                    "campplus_embedding_size": self.campplus_embedding_size,
                    "campplus_film_scale": self.campplus_film_scale,
                    "campplus_depth_start_level": self.campplus_depth_start_level,
                    "campplus_depth_identity_loss_weight": (
                        self.campplus_depth_identity_loss_weight
                    ),
                    "campplus_identity_supervision": (
                        self.campplus_identity_supervision
                    ),
                    "campplus_identity_level_aggregation": (
                        self.campplus_identity_level_aggregation
                    ),
                    "level0_eos_loss_weight": self.level0_eos_loss_weight,
                    "level0_tail_loss_weight": self.level0_tail_loss_weight,
                    "level0_tail_loss_last_n": self.level0_tail_loss_last_n,
                    "level0_label_smoothing": self.level0_label_smoothing,
                    "level0_label_smoothing_train_only": self.level0_label_smoothing_train_only,
                    "codebook_loss_weights": self.codebook_loss_weights,
                    "depth_objective_start_level": self.depth_objective_start_level,
                    "enable_semantic_prefix_ar": self.enable_semantic_prefix_ar,
                    "enable_source_unit_auxiliary": self.enable_source_unit_auxiliary,
                    "source_unit_vocab_size": self.source_unit_vocab_size,
                    "source_unit_loss_weight": self.source_unit_loss_weight,
                    "source_unit_blank_id": self.source_unit_blank_id,
                    "enable_source_unit_transformer_auxiliary": self.enable_source_unit_transformer_auxiliary,
                    "source_unit_transformer_loss_weight": self.source_unit_transformer_loss_weight,
                    "source_unit_transformer_num_layers": self.source_unit_transformer_num_layers,
                    "source_unit_transformer_num_heads": self.source_unit_transformer_num_heads,
                    "source_unit_transformer_ffn_dim": self.source_unit_transformer_ffn_dim,
                    "source_unit_transformer_dropout": self.source_unit_transformer_dropout,
                    "source_unit_transformer_max_positions": self.source_unit_transformer_max_positions,
                    "source_adapter_num_layers": self.source_adapter_num_layers,
                    "source_adapter_num_heads": self.source_adapter_num_heads,
                    "source_adapter_ffn_dim": self.source_adapter_ffn_dim,
                    "source_adapter_dropout": self.source_adapter_dropout,
                    "source_adapter_residual": self.source_adapter_residual,
                    "source_adapter_residual_scale": self.source_adapter_residual_scale,
                    "enable_depth_source_conditioning": self.enable_depth_source_conditioning,
                    "depth_source_conditioning_num_heads": self.depth_source_conditioning_num_heads,
                    "depth_source_conditioning_dropout": self.depth_source_conditioning_dropout,
                    "depth_source_conditioning_residual_scale": self.depth_source_conditioning_residual_scale,
                    "depth_source_conditioning_detach": self.depth_source_conditioning_detach,
                    "transvip_repo_dir": self.transvip_repo_dir,
                    "transvip_model_cfg_path": self.transvip_model_cfg_path,
                    "transvip_model_path": self.transvip_model_path,
                    "transvip_model_name": self.transvip_model_name,
                    "transvip_num_new_tokens": self.transvip_num_new_tokens,
                    "transvip_spk_encoder_path": self.transvip_spk_encoder_path,
                    "transvip_use_length_control": self.transvip_use_length_control,
                    "transvip_source_checkpoint_path": self.transvip_source_checkpoint_path,
                    "transvip_load_source_encoder_from_checkpoint": (
                        self.transvip_load_source_encoder_from_checkpoint
                    ),
                    "transvip_text_decoder_checkpoint_path": self.transvip_text_decoder_checkpoint_path,
                    "transvip_load_text_decoder_from_checkpoint": self.transvip_load_text_decoder_from_checkpoint,
                    "transvip_text_decoder_text_vocab_size": self.transvip_text_decoder_text_vocab_size,
                    "temporal_decoder_backend": self.temporal_decoder_backend,
                    "source_unit_bos_id": self.source_unit_bos_id,
                    "source_unit_eos_id": self.source_unit_eos_id,
                    "source_unit_pad_id": self.source_unit_pad_id,
                    "enable_text_prefix_ar": self.enable_text_prefix_ar,
                    "enable_text_codec_ar": self.enable_text_codec_ar,
                    "enable_quality_cot": self.enable_quality_cot,
                    "quality_cot_prompt_all_seps": self.quality_cot_prompt_all_seps,
                    "quality_cot_source_text_prefix_extra_token_ids_after_bos": (
                        self.quality_cot_source_text_prefix_extra_token_ids_after_bos
                    ),
                    "quality_cot_source_text_max_tokens": self.quality_cot_source_text_max_tokens,
                    "quality_cot_target_text_max_tokens": self.quality_cot_target_text_max_tokens,
                    "quality_cot_source_text_loss_weight": self.quality_cot_source_text_loss_weight,
                    "quality_cot_target_text_loss_weight": self.quality_cot_target_text_loss_weight,
                    "quality_cot_code0_loss_weight": self.quality_cot_code0_loss_weight,
                    "quality_cot_code1_loss_weight": self.quality_cot_code1_loss_weight,
                    "quality_cot_depth_loss_weight": self.quality_cot_depth_loss_weight,
                    "enable_target_only_speech_auxiliary": self.enable_target_only_speech_auxiliary,
                    "target_only_speech_text_loss_weight": self.target_only_speech_text_loss_weight,
                    "target_only_speech_code0_loss_weight": self.target_only_speech_code0_loss_weight,
                    "enable_mixed_source_target_auxiliary": self.enable_mixed_source_target_auxiliary,
                    "mixed_source_target_loss_weight": self.mixed_source_target_loss_weight,
                    "mixed_source_prediction_prob": self.mixed_source_prediction_prob,
                    "mixed_source_start_step": self.mixed_source_start_step,
                    "mixed_source_warmup_steps": self.mixed_source_warmup_steps,
                    "mixed_source_auxiliary_seed": self.mixed_source_auxiliary_seed,
                    "enable_predicted_source_target_auxiliary": (
                        self.enable_predicted_source_target_auxiliary
                    ),
                    "predicted_source_target_loss_weight": (
                        self.predicted_source_target_loss_weight
                    ),
                    "predicted_source_target_start_step": (
                        self.predicted_source_target_start_step
                    ),
                    "predicted_source_target_warmup_steps": (
                        self.predicted_source_target_warmup_steps
                    ),
                    "predicted_source_target_max_source_tokens": (
                        self.predicted_source_target_max_source_tokens
                    ),
                    "predicted_source_target_batch_size": (
                        self.predicted_source_target_batch_size
                    ),
                    "text_vocab_size": self.text_vocab_size,
                    "text_generation_vocab_size": self.text_generation_vocab_size,
                    "text_pad_token_id": self.text_pad_token_id,
                    "text_bos_token_id": self.text_bos_token_id,
                    "text_sep_token_id": self.text_sep_token_id,
                    "text_prefix_extra_token_ids_after_bos": getattr(
                        self, "text_prefix_extra_token_ids_after_bos", []
                    ),
                    # Alias used by the text collator.
                    "text_extra_token_ids_after_bos": getattr(
                        self, "text_prefix_extra_token_ids_after_bos", []
                    ),
                    "text_loss_weight": self.text_loss_weight,
                    "text_label_smoothing": self.text_label_smoothing,
                    "enable_codec_mass_loss": self.enable_codec_mass_loss,
                    "codec_mass_loss_weight": self.codec_mass_loss_weight,
                    "enable_text_codec_kd_loss": self.enable_text_codec_kd_loss,
                    "text_codec_kd_loss_weight": self.text_codec_kd_loss_weight,
                    "enable_transvip_style_loss_aggregation": self.enable_transvip_style_loss_aggregation,
                    "transvip_style_speech_loss_weight": self.transvip_style_speech_loss_weight,
                    "transvip_style_text_loss_weight": self.transvip_style_text_loss_weight,
                    "transvip_style_kd_loss_weight": self.transvip_style_kd_loss_weight,
                    "transvip_style_depth_loss_weight": self.transvip_style_depth_loss_weight,
                    "use_unified_code0_loss_in_objective": self.use_unified_code0_loss_in_objective,
                    "enable_text_codec_text_path_loss": self.enable_text_codec_text_path_loss,
                    "text_codec_text_path_input": self.text_codec_text_path_input,
                    "text_codec_t2t_loss_weight": self.text_codec_t2t_loss_weight,
                    "text_codec_t2c_loss_weight": self.text_codec_t2c_loss_weight,
                    "text_path_source_token_dropout": self.text_path_source_token_dropout,
                    "train_only_depth_decoder": self.train_only_depth_decoder,
                    "enable_direct_text_to_c0_auxiliary": self.enable_direct_text_to_c0_auxiliary,
                    "direct_text_to_c0_input": self.direct_text_to_c0_input,
                    "direct_text_to_c0_loss_weight": self.direct_text_to_c0_loss_weight,
                    "direct_text_to_c0_zero_speech_memory": (
                        self.direct_text_to_c0_zero_speech_memory
                    ),
                    "direct_text_to_c0_kd_weight": self.direct_text_to_c0_kd_weight,
                    "direct_text_to_c0_kd_temperature": self.direct_text_to_c0_kd_temperature,
                    "text_prefix_beam_size": self.text_prefix_beam_size,
                    "text_prefix_min_tokens": self.text_prefix_min_tokens,
                    "text_prefix_sep_penalty": self.text_prefix_sep_penalty,
                    "text_prefix_length_penalty": self.text_prefix_length_penalty,
                    "text_prefix_no_repeat_ngram_size": self.text_prefix_no_repeat_ngram_size,
                    "tie_text_embeddings": self.tie_text_embeddings,
                    "text_max_positions": self.text_max_positions,
                },
                f,
                indent=2,
            )


