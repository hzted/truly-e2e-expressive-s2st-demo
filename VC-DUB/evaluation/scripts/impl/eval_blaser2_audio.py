#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from scipy import stats
import torch


SONAR_TEXT_LANG_MAP = {
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "de": "deu_Latn",
    "deu": "deu_Latn",
    "es": "spa_Latn",
    "spa": "spa_Latn",
}

SONAR_SPEECH_ENCODER_MAP = {
    "en": "sonar_speech_encoder_eng",
    "eng": "sonar_speech_encoder_eng",
    "de": "sonar_speech_encoder_deu",
    "deu": "sonar_speech_encoder_deu",
    "es": "sonar_speech_encoder_spa",
    "spa": "sonar_speech_encoder_spa",
}


def ci_fields(xs: pd.Series, prefix: str) -> dict:
    s = pd.to_numeric(xs, errors="coerce").dropna()
    if len(s) == 0:
        return {}
    mean = float(s.mean())
    if len(s) <= 1:
        return {
            f"{prefix}_sem": 0.0,
            f"{prefix}_ci95_low": mean,
            f"{prefix}_ci95_high": mean,
        }
    sem = float(s.sem())
    ci_low, ci_high = stats.t.interval(0.95, df=len(s) - 1, loc=mean, scale=sem)
    return {
        f"{prefix}_sem": sem,
        f"{prefix}_ci95_low": float(ci_low),
        f"{prefix}_ci95_high": float(ci_high),
    }


def normalize_text_lang(tag: str) -> str:
    tag = str(tag).strip()
    return SONAR_TEXT_LANG_MAP.get(tag.lower(), tag)


def infer_speech_encoder(tag: str) -> str:
    tag = str(tag).strip().lower()
    if tag not in SONAR_SPEECH_ENCODER_MAP:
        raise ValueError(
            f"Cannot infer SONAR speech encoder for language tag {tag!r}. "
            "Please pass an explicit encoder override."
        )
    return SONAR_SPEECH_ENCODER_MAP[tag]


def batched(values: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


def embed_audio_paths(embedder, paths: List[str], batch_size: int) -> torch.Tensor:
    chunks = []
    for batch in batched(paths, batch_size):
        chunks.append(embedder.predict(batch))
    return torch.cat(chunks, dim=0)


def embed_texts(embedder, texts: List[str], lang: str, batch_size: int) -> torch.Tensor:
    chunks = []
    for batch in batched(texts, batch_size):
        chunks.append(embedder.predict(batch, source_lang=lang))
    return torch.cat(chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute audio-first BLASER 2.0 scores from a generic eval manifest."
    )
    parser.add_argument("--manifest", required=True, help="Generic eval manifest TSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-audio-col", default="source_audio")
    parser.add_argument("--hypo-audio-col", default="hypo_audio")
    parser.add_argument("--target-lang-col", default="target_lang")
    parser.add_argument("--source-lang", default="eng", help="Source speech language, e.g. eng / en.")
    parser.add_argument("--source-speech-encoder", default=None, help="Optional SONAR source speech encoder override.")
    parser.add_argument("--target-speech-encoder", default=None, help="Optional SONAR target speech encoder override.")
    parser.add_argument("--reference-text-col", default="reference_translation")
    parser.add_argument("--reference-audio-col", default=None)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--status-col", default="status")
    parser.add_argument("--ok-status", default="ok")
    parser.add_argument("--keep-non-ok", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    try:
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        from sonar.models.blaser.loader import load_blaser_model
    except Exception as exc:
        raise RuntimeError(
            "SONAR / BLASER 2.0 is not available in this environment. "
            "Install the official sonar-space package and matching fairseq2 first."
        ) from exc

    manifest_path = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path, sep="\t")
    required = [args.id_col, args.source_audio_col, args.hypo_audio_col, args.target_lang_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required manifest columns: {missing}")

    filtered = df.copy()
    filtered[args.source_audio_col] = filtered[args.source_audio_col].fillna("").astype(str)
    filtered[args.hypo_audio_col] = filtered[args.hypo_audio_col].fillna("").astype(str)
    filtered = filtered[
        (filtered[args.source_audio_col].str.len() > 0) &
        (filtered[args.hypo_audio_col].str.len() > 0)
    ].copy()

    if not args.keep_non_ok and args.status_col in filtered.columns:
        filtered = filtered[filtered[args.status_col].astype(str) == args.ok_status].copy()

    filtered = filtered.reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No valid rows left after filtering for non-empty audio paths / status.")

    source_lang = str(args.source_lang)
    source_encoder_name = args.source_speech_encoder or infer_speech_encoder(source_lang)
    filtered["target_encoder_name"] = filtered[args.target_lang_col].map(
        lambda x: args.target_speech_encoder or infer_speech_encoder(x)
    )
    filtered["target_text_lang"] = filtered[args.target_lang_col].map(normalize_text_lang)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    src_speech_embedder = SpeechToEmbeddingModelPipeline(
        encoder=source_encoder_name,
        device=device,
    )
    blaser_qe = load_blaser_model("blaser_2_0_qe").eval().to(device)
    blaser_ref = load_blaser_model("blaser_2_0_ref").eval().to(device)

    src_embs = embed_audio_paths(
        src_speech_embedder,
        filtered[args.source_audio_col].tolist(),
        args.batch_size,
    )

    qe_scores = torch.empty(len(filtered), dtype=torch.float32)
    ref_scores = torch.full((len(filtered),), float("nan"), dtype=torch.float32)

    ref_mode = None
    use_reference_audio = bool(args.reference_audio_col) and args.reference_audio_col in filtered.columns
    use_reference_text = args.reference_text_col in filtered.columns

    text_embedder = None
    if use_reference_text:
        text_embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
            dtype=dtype,
        )

    with torch.inference_mode():
        for encoder_name in filtered["target_encoder_name"].dropna().unique().tolist():
            mask = filtered["target_encoder_name"] == encoder_name
            idxs = filtered.index[mask].tolist()
            if not idxs:
                continue

            tgt_speech_embedder = SpeechToEmbeddingModelPipeline(
                encoder=encoder_name,
                device=device,
            )

            mt_embs = embed_audio_paths(
                tgt_speech_embedder,
                filtered.loc[idxs, args.hypo_audio_col].tolist(),
                args.batch_size,
            )
            src_subset = src_embs[idxs].to(device)
            qe = blaser_qe(src=src_subset, mt=mt_embs).detach().cpu().float().view(-1)
            qe_scores[idxs] = qe

            ref_embs = None
            if use_reference_audio:
                ref_audio = (
                    filtered.loc[idxs, args.reference_audio_col]
                    .fillna("")
                    .astype(str)
                    .tolist()
                )
                if all(x for x in ref_audio):
                    ref_embs = embed_audio_paths(tgt_speech_embedder, ref_audio, args.batch_size)
                    ref_mode = "audio"

            if ref_embs is None and use_reference_text:
                target_langs = filtered.loc[idxs, "target_text_lang"].tolist()
                ref_texts = filtered.loc[idxs, args.reference_text_col].fillna("").astype(str).tolist()
                grouped = {}
                for local_pos, (lang, text) in enumerate(zip(target_langs, ref_texts)):
                    grouped.setdefault(lang, []).append((local_pos, text))
                ref_chunks = [None] * len(idxs)
                for lang, items in grouped.items():
                    item_positions = [p for p, _ in items]
                    item_texts = [t for _, t in items]
                    lang_embs = embed_texts(text_embedder, item_texts, lang, args.batch_size)
                    for offset, local_pos in enumerate(item_positions):
                        ref_chunks[local_pos] = lang_embs[offset : offset + 1]
                ref_embs = torch.cat(ref_chunks, dim=0)
                ref_mode = "text"

            if ref_embs is not None:
                ref = blaser_ref(src=src_subset, ref=ref_embs, mt=mt_embs).detach().cpu().float().view(-1)
                ref_scores[idxs] = ref

    filtered["blaser2_qe_audio"] = qe_scores.numpy()
    filtered["blaser2_ref"] = ref_scores.numpy()
    filtered["blaser2_ref_mode"] = ref_mode if ref_mode is not None else ""

    merged_path = out_dir / "manifest_with_blaser2_audio.tsv"
    score_path = out_dir / "blaser2_audio_scores.tsv"
    summary_path = out_dir / "blaser2_audio_summary.json"

    filtered.to_csv(merged_path, sep="\t", index=False)
    filtered[
        [
            args.id_col,
            args.source_audio_col,
            args.hypo_audio_col,
            args.target_lang_col,
            "blaser2_qe_audio",
            "blaser2_ref",
            "blaser2_ref_mode",
        ]
    ].to_csv(score_path, sep="\t", index=False)

    summary = {
        "rows": int(len(filtered)),
        "source_speech_encoder": source_encoder_name,
        "target_speech_encoders": sorted(filtered["target_encoder_name"].dropna().unique().tolist()),
        "reference_mode": ref_mode,
        "blaser2_qe_audio_mean": float(filtered["blaser2_qe_audio"].mean()),
        "blaser2_qe_audio_std": float(filtered["blaser2_qe_audio"].std(ddof=0)),
        "blaser2_ref_mean": None if filtered["blaser2_ref"].isna().all() else float(filtered["blaser2_ref"].dropna().mean()),
        "blaser2_ref_std": None if filtered["blaser2_ref"].isna().all() else float(filtered["blaser2_ref"].dropna().std(ddof=0)),
        **ci_fields(filtered["blaser2_qe_audio"], "blaser2_qe_audio"),
        **ci_fields(filtered["blaser2_ref"], "blaser2_ref"),
        "merged_manifest": str(merged_path),
        "score_tsv": str(score_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote merged manifest to {merged_path}")
    print(f"Wrote score TSV to {score_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""
python VC-DUB/evaluation/scripts/impl/eval_blaser2_audio.py \
  --manifest /path/to/eval_manifest.tsv \
  --output-dir /path/to/blaser2_audio \
  --source-lang eng \
  --device cuda
"""
