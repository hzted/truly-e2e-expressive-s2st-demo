"""Turn a training checkpoint into the layout inference/infer.py loads.

    python scripts/clean_checkpoint.py --source-checkpoint-dir outputs/es/checkpoint-NNNNN \
        --output-dir checkpoints/en-es

Writes pytorch_model.bin, a model_config.json reduced to the model's
constructor arguments (no machine-specific paths), and text_tokenizer/.
Optimizer, scheduler and RNG states are not copied.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
from infer import (  # noqa: E402
    load_merged_model_config,
    matched_ctor_config_keys,
    resolve_text_tokenizer_ids,
)

# Local paths; supplied at run time instead (--campplus-model-root etc.).
LOCAL_PATH_CONFIG_KEYS = (
    "campplus_model_root",
    "campplus_checkpoint_path",
    "transvip_repo_dir",
    "transvip_model_cfg_path",
    "transvip_model_path",
    "transvip_spk_encoder_path",
    "transvip_prompt_codec_path",
    "transvip_source_checkpoint_path",
    "transvip_text_decoder_checkpoint_path",
)

# Non-constructor keys infer.py reads.
EXTRA_KEYS_INFER_NEEDS = (
    "text_tokenizer_name",
    "codebook_model_order",
    "mimi_codebook_order",
)

TOKENIZER_FILES_TO_BUNDLE = (
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
)


def scrub_config(
    model_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any], List[str]]:
    """Keep constructor arguments plus the few extra keys infer.py reads; drop local paths."""
    keep_keys = set(matched_ctor_config_keys(model_config)) | (
        set(EXTRA_KEYS_INFER_NEEDS) & set(model_config)
    )
    dropped_as_unused = sorted(set(model_config) - keep_keys)
    scrubbed = {key: model_config[key] for key in keep_keys}

    dropped_as_leaked = []
    for key in LOCAL_PATH_CONFIG_KEYS:
        if key in scrubbed:
            dropped_as_leaked.append(key)
            del scrubbed[key]

    # Flag anything else that still looks like an absolute path.
    suspicious = {
        key: value
        for key, value in scrubbed.items()
        if isinstance(value, str) and value.startswith("/")
    }
    return scrubbed, dropped_as_leaked, suspicious, dropped_as_unused


def bundle_text_tokenizer(text_tokenizer_name: str, output_dir: Path) -> None:
    source = Path(text_tokenizer_name)
    dest = output_dir / "text_tokenizer"
    dest.mkdir(parents=True, exist_ok=True)
    for filename in TOKENIZER_FILES_TO_BUNDLE:
        src_file = source / filename
        if src_file.is_file():
            shutil.copy2(src_file, dest / filename)
    skipped = sorted(
        p.name for p in source.iterdir() if p.is_file() and p.name not in TOKENIZER_FILES_TO_BUNDLE
    )
    if skipped:
        print(f"[clean] tokenizer dir: not bundling (non-standard/leaky metadata): {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(source_dir / "pytorch_model.bin", output_dir / "pytorch_model.bin")
    model_config = load_merged_model_config(source_dir)
    model_config.update(resolve_text_tokenizer_ids(model_config))

    text_tokenizer_name = model_config.get("text_tokenizer_name")
    if text_tokenizer_name and Path(text_tokenizer_name).is_dir():
        print(f"[clean] bundling text tokenizer from {text_tokenizer_name} -> text_tokenizer/")
        bundle_text_tokenizer(text_tokenizer_name, output_dir)
        model_config["text_tokenizer_name"] = "./text_tokenizer"

    scrubbed, dropped, suspicious, dropped_as_unused = scrub_config(model_config)
    print(f"[clean] kept {len(scrubbed)} config keys, dropped {len(dropped_as_unused) + len(dropped)}")
    if suspicious:
        print(f"[clean] WARNING: values that look like absolute paths: {suspicious}")

    (output_dir / "model_config.json").write_text(
        json.dumps(scrubbed, indent=2, sort_keys=True) + "\n"
    )
    print(f"[clean] done: {output_dir}")


if __name__ == "__main__":
    main()
