#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Iterable, List

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_get(row: Dict[str, str], key: str) -> str:
    val = row.get(key, "")
    return "" if val is None else str(val).strip()


def read_tsv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter="\t")]


def write_tsv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def append_tsv_row(path: str, row: Dict[str, str], fieldnames: List[str]) -> None:
    write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()


def load_success_cache(path: str) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            audio_path = row.get("audio_path", "")
            text = row.get("text", "")
            if audio_path:
                cache[audio_path] = text
    return cache


def load_fail_set(path: str) -> set[str]:
    failed: set[str] = set()
    if not os.path.exists(path):
        return failed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                failed.add(item)
    return failed


def append_fail(path: str, audio_path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(audio_path + "\n")
        f.flush()


def append_fail_detail(path: str, audio_path: str, error: BaseException) -> None:
    fields = ["audio_path", "error_type", "error"]
    append_tsv_row(
        path,
        {
            "audio_path": audio_path,
            "error_type": type(error).__name__,
            "error": str(error).replace("\n", "\\n"),
        },
        fields,
    )


def remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def unique_in_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def load_audio_16k_mono(audio_path: str) -> np.ndarray:
    audio, sr = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
    return audio


class WhisperASR:
    def __init__(
        self,
        model_id: str,
        device: str,
        batch_size: int,
        chunk_length_s: float,
        max_new_tokens: int,
    ) -> None:
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.chunk_length_s = chunk_length_s
        self.max_new_tokens = max_new_tokens

    def transcribe_batch(self, paths: List[str], language: str) -> List[str]:
        audios = [load_audio_16k_mono(path) for path in paths]
        inputs = self.processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task="transcribe",
        )
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_features=input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=self.max_new_tokens,
            )
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [clean_text(text) for text in texts]

    def transcribe_one(self, path: str, language: str) -> str:
        return self.transcribe_batch([path], language=language)[0]


def process_paths(
    *,
    asr: WhisperASR,
    paths: List[str],
    language: str,
    cache: Dict[str, str],
    fail_set: set[str],
    cache_path: str,
    fail_path: str,
    fail_detail_path: str,
    label: str,
    batch_size: int,
) -> None:
    cache_fields = ["audio_path", "text"]
    pending = [p for p in paths if p not in cache and p not in fail_set]
    print(f"{label} pending: {len(pending)}")

    for start in tqdm(range(0, len(pending), batch_size), desc=f"Whisper {label}"):
        batch = pending[start : start + batch_size]
        try:
            texts = asr.transcribe_batch(batch, language=language)
            if len(texts) != len(batch):
                raise RuntimeError(f"batch output size mismatch: {len(texts)} vs {len(batch)}")
            for audio_path, text in zip(batch, texts):
                if not text:
                    raise RuntimeError(f"Empty transcription for {audio_path}")
                append_tsv_row(cache_path, {"audio_path": audio_path, "text": text}, cache_fields)
                cache[audio_path] = text
        except Exception as batch_error:
            print(f"[WARN] {label} batch failed at {start}: {type(batch_error).__name__}({batch_error})")
            for audio_path in batch:
                try:
                    text = asr.transcribe_one(audio_path, language=language)
                    if not text:
                        raise RuntimeError("Empty transcription")
                    append_tsv_row(cache_path, {"audio_path": audio_path, "text": text}, cache_fields)
                    cache[audio_path] = text
                except Exception as item_error:
                    append_fail(fail_path, audio_path)
                    append_fail_detail(fail_detail_path, audio_path, item_error)
                    fail_set.add(audio_path)
                    print(f"[FAIL] {label} {audio_path}: {type(item_error).__name__}({item_error})")


def build_text_meta(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)

    src_cache_path = os.path.join(args.out_dir, "src_asr_cache.tsv")
    tgt_cache_path = os.path.join(args.out_dir, "tgt_asr_cache.tsv")
    extra_cache_path = os.path.join(args.out_dir, "extra_asr_cache.tsv")

    src_fail_path = os.path.join(args.out_dir, "src_asr_fail.txt")
    tgt_fail_path = os.path.join(args.out_dir, "tgt_asr_fail.txt")
    extra_fail_path = os.path.join(args.out_dir, "extra_asr_fail.txt")
    src_fail_detail_path = os.path.join(args.out_dir, "src_asr_fail_detail.tsv")
    tgt_fail_detail_path = os.path.join(args.out_dir, "tgt_asr_fail_detail.tsv")
    extra_fail_detail_path = os.path.join(args.out_dir, "extra_asr_fail_detail.tsv")

    meta_path = os.path.join(args.out_dir, "transvip_text_meta.tsv")
    nar_list_path = os.path.join(args.out_dir, "nar_audio_list.txt")

    manifest_rows = read_tsv_rows(args.manifest)
    if not manifest_rows:
        raise RuntimeError(f"No rows found in manifest: {args.manifest}")

    required_cols = {
        "id",
        "status",
        args.src_audio_field,
        args.tgt_audio_field,
        args.asr_src_field,
        args.asr_tgt_field,
    }
    if args.extra_asr_field:
        required_cols.add(args.extra_asr_field)

    missing_cols = required_cols - set(manifest_rows[0].keys())
    if missing_cols:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_cols)}")

    usable_rows = []
    skipped_fail_status = 0
    skipped_empty_fields = 0
    skipped_missing_files = 0

    for row in manifest_rows:
        status_val = safe_get(row, "status").upper()
        if "FAIL" in status_val:
            skipped_fail_status += 1
            continue

        src_audio = safe_get(row, args.src_audio_field)
        tgt_audio = safe_get(row, args.tgt_audio_field)
        asr_src_audio = safe_get(row, args.asr_src_field)
        asr_tgt_audio = safe_get(row, args.asr_tgt_field)
        extra_asr_audio = safe_get(row, args.extra_asr_field) if args.extra_asr_field else ""

        required_paths = [src_audio, tgt_audio, asr_src_audio, asr_tgt_audio]
        if args.extra_asr_field:
            required_paths.append(extra_asr_audio)

        if not all(required_paths):
            skipped_empty_fields += 1
            continue

        if args.skip_missing_audio and not all(os.path.exists(p) for p in required_paths):
            skipped_missing_files += 1
            continue

        usable_rows.append(row)

    print(f"Loaded manifest rows: {len(manifest_rows)}")
    print(f"Skipped because status contains FAIL: {skipped_fail_status}")
    print(f"Skipped because required fields empty: {skipped_empty_fields}")
    if args.skip_missing_audio:
        print(f"Skipped because audio file missing: {skipped_missing_files}")
        print(f"Usable rows after filtering: {len(usable_rows)}")
    if not usable_rows:
        raise RuntimeError("No usable rows left after filtering.")

    print("[INFO] Whisper ASR loader: ffmpeg-free soundfile/librosa -> 16k mono numpy arrays")
    print(f"[INFO] Model: {args.model_id}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Chunk length seconds: {args.chunk_length_s}")
    print(f"[INFO] Max new tokens: {args.max_new_tokens}")

    unique_src_paths = unique_in_order(safe_get(r, args.asr_src_field) for r in usable_rows)
    unique_tgt_paths = unique_in_order(safe_get(r, args.asr_tgt_field) for r in usable_rows)
    unique_extra_paths = (
        unique_in_order(safe_get(r, args.extra_asr_field) for r in usable_rows)
        if args.extra_asr_field
        else []
    )

    src_cache = load_success_cache(src_cache_path)
    tgt_cache = load_success_cache(tgt_cache_path)
    extra_cache = load_success_cache(extra_cache_path) if args.extra_asr_field else {}

    if args.rerun_errors:
        for path in [
            src_fail_path,
            tgt_fail_path,
            extra_fail_path,
            src_fail_detail_path,
            tgt_fail_detail_path,
            extra_fail_detail_path,
        ]:
            remove_if_exists(path)

    src_fail = set() if args.rerun_errors else load_fail_set(src_fail_path)
    tgt_fail = set() if args.rerun_errors else load_fail_set(tgt_fail_path)
    extra_fail = set() if args.rerun_errors else load_fail_set(extra_fail_path)

    print(f"Unique source ASR paths ({args.asr_src_field}): {len(unique_src_paths)}")
    print(f"Unique target ASR paths ({args.asr_tgt_field}): {len(unique_tgt_paths)}")
    if args.extra_asr_field:
        print(f"Unique extra ASR paths ({args.extra_asr_field}): {len(unique_extra_paths)}")

    asr = WhisperASR(
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
        chunk_length_s=args.chunk_length_s,
        max_new_tokens=args.max_new_tokens,
    )

    process_paths(
        asr=asr,
        paths=unique_src_paths,
        language=args.src_language,
        cache=src_cache,
        fail_set=src_fail,
        cache_path=src_cache_path,
        fail_path=src_fail_path,
        fail_detail_path=src_fail_detail_path,
        label="SRC",
        batch_size=args.batch_size,
    )
    process_paths(
        asr=asr,
        paths=unique_tgt_paths,
        language=args.tgt_language,
        cache=tgt_cache,
        fail_set=tgt_fail,
        cache_path=tgt_cache_path,
        fail_path=tgt_fail_path,
        fail_detail_path=tgt_fail_detail_path,
        label="TGT",
        batch_size=args.batch_size,
    )
    if args.extra_asr_field:
        process_paths(
            asr=asr,
            paths=unique_extra_paths,
            language=args.extra_language,
            cache=extra_cache,
            fail_set=extra_fail,
            cache_path=extra_cache_path,
            fail_path=extra_fail_path,
            fail_detail_path=extra_fail_detail_path,
            label="EXTRA",
            batch_size=args.batch_size,
        )

    meta_rows = []
    nar_audio_paths = []
    for row in usable_rows:
        sample_id = safe_get(row, "id")
        src_audio = safe_get(row, args.src_audio_field)
        tgt_audio = safe_get(row, args.tgt_audio_field)
        asr_src_audio = safe_get(row, args.asr_src_field)
        asr_tgt_audio = safe_get(row, args.asr_tgt_field)
        extra_asr_audio = safe_get(row, args.extra_asr_field) if args.extra_asr_field else ""

        if asr_src_audio in src_fail or asr_tgt_audio in tgt_fail:
            continue
        if asr_src_audio not in src_cache or asr_tgt_audio not in tgt_cache:
            continue

        meta_row = {
            "path": sample_id,
            "sentence": src_cache[asr_src_audio],
            "translation": tgt_cache[asr_tgt_audio],
            "src_audio": src_audio,
            "tgt_audio": tgt_audio,
        }
        if args.extra_asr_field:
            meta_row[args.extra_asr_col] = extra_cache.get(extra_asr_audio, "")

        meta_rows.append(meta_row)
        nar_audio_paths.append(tgt_audio)

    meta_fields = ["path", "sentence", "translation", "src_audio", "tgt_audio"]
    if args.extra_asr_field:
        meta_fields.append(args.extra_asr_col)
    write_tsv(meta_path, meta_rows, meta_fields)

    with open(nar_list_path, "w", encoding="utf-8") as f:
        for path in unique_in_order(nar_audio_paths):
            f.write(path + "\n")

    print("\nDone.")
    print(f"Source ASR cache: {src_cache_path}")
    print(f"Target ASR cache: {tgt_cache_path}")
    if args.extra_asr_field:
        print(f"Extra ASR cache: {extra_cache_path}")
    print(f"TransVIP text meta: {meta_path}")
    print(f"NAR audio list: {nar_list_path}")
    print(f"Final usable meta rows: {len(meta_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="openai/whisper-large-v3")

    parser.add_argument("--src-audio-field", type=str, default="pre_src")
    parser.add_argument("--tgt-audio-field", type=str, default="out")
    parser.add_argument("--asr-src-field", type=str, default="pre_src")
    parser.add_argument("--asr-tgt-field", type=str, default="pre_tgt")
    parser.add_argument("--extra-asr-field", type=str, default="")
    parser.add_argument("--extra-asr-col", type=str, default="out_sentence")

    parser.add_argument("--src-language", type=str, default="english")
    parser.add_argument("--tgt-language", type=str, default="spanish")
    parser.add_argument("--extra-language", type=str, default="english")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-length-s", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=440)
    parser.add_argument("--skip-missing-audio", action="store_true")
    parser.add_argument("--rerun-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_text_meta(parse_args())


if __name__ == "__main__":
    main()
