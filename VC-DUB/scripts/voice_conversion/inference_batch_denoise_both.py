import os
import csv
import time
import argparse
import sys
import tempfile
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import numpy as np
import torchaudio
import librosa
import soundfile as sf

import inference  # resolved from SEEDVC_ROOT via PYTHONPATH

from julius import resample_frac as julius_resample
from demucs.pretrained import get_model
from demucs.apply import apply_model


# =========================
# Global model caches
# =========================
_demucs_model = None
_demucs_model_name = None

_clearvoice_model = None
_clearvoice_model_name = None
_ClearVoiceClass = None


# =========================
# Utility
# =========================
def safe_audio_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.ndim > 1:
        # collapse to mono
        if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
            x = x.mean(axis=0)
        else:
            x = x.mean(axis=1)

    peak = np.max(np.abs(x)) if x.size > 0 else 0.0
    if peak > 1.0:
        x = x / peak
    return x.astype(np.float32)


def _as_bool_strict(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y"}


def load_prior_manifest(manifest_path: Path):
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="	")
        rows = {}
        for row in reader:
            rows[str(row.get("id", ""))] = row
        return rows


def row_is_resume_safe(row: dict, save_preprocessed_audio: bool) -> bool:
    if not row:
        return False

    out_path = Path(str(row.get("out", "")))
    if not out_path.exists() or out_path.stat().st_size == 0:
        return False

    if save_preprocessed_audio:
        pre_src = Path(str(row.get("pre_src", "")))
        pre_tgt = Path(str(row.get("pre_tgt", "")))
        if not pre_src.exists() or pre_src.stat().st_size == 0:
            return False
        if not pre_tgt.exists() or pre_tgt.stat().st_size == 0:
            return False

    return True


def row_matches_current_pair(row: dict, idx: str, src_p: Path, tgt_p: Path, out_wav: Path) -> bool:
    if not row or str(row.get("id", "")) != idx:
        return False
    try:
        row_src = Path(str(row.get("src", ""))).resolve()
        row_tgt = Path(str(row.get("tgt", ""))).resolve()
        row_out = Path(str(row.get("out", ""))).resolve()
    except Exception:
        return False
    return row_src == src_p.resolve() and row_tgt == tgt_p.resolve() and row_out == out_wav.resolve()


def format_manifest_id(idx) -> str:
    if isinstance(idx, int):
        return f"{idx:03d}"
    return str(idx)


def load_pair_audio(source_path: str, target_path: str, sr_cfg: int):
    source_audio_raw, _ = librosa.load(source_path, sr=sr_cfg, mono=True)
    target_audio_raw, _ = librosa.load(target_path, sr=sr_cfg, mono=True)
    return source_audio_raw, target_audio_raw


def save_preprocessed_pair(
    src_audio_np: np.ndarray,
    tgt_audio_np: np.ndarray,
    sr: int,
    out_wav: Path,
    output_dir: Path,
):
    """
    Save preprocessed source/target audio under:
        <output_dir.parent>/preprocessed/source/
        <output_dir.parent>/preprocessed/target/
    File name follows VC output stem.
    """
    pre_root = output_dir.parent / "preprocessed"
    src_dir = pre_root / "source"
    tgt_dir = pre_root / "target"

    src_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    stem = out_wav.stem
    src_save = src_dir / f"{stem}.wav"
    tgt_save = tgt_dir / f"{stem}.wav"

    sf.write(str(src_save), safe_audio_np(src_audio_np), sr)
    sf.write(str(tgt_save), safe_audio_np(tgt_audio_np), sr)

    return src_save, tgt_save


# =========================
# ClearVoice
# =========================
def _load_clearvoice(model_name: str, clearvoice_root: str):
    """
    clearvoice_root should be:
    /path/to/ClearerVoice-Studio/clearvoice

    because actual package path is:
    .../clearvoice/clearvoice/__init__.py
    """
    global _clearvoice_model, _clearvoice_model_name, _ClearVoiceClass

    if clearvoice_root not in sys.path:
        sys.path.insert(0, clearvoice_root)

    if _ClearVoiceClass is None:
        from clearvoice import ClearVoice as _ImportedClearVoice
        _ClearVoiceClass = _ImportedClearVoice

    if _clearvoice_model is None or _clearvoice_model_name != model_name:
        _clearvoice_model = _ClearVoiceClass(
            task="speech_enhancement",
            model_names=[model_name],
        )
        _clearvoice_model_name = model_name

    return _clearvoice_model


def denoise_with_clearvoice(
    wav_np: np.ndarray,
    sr: int,
    model_name: str,
    clearvoice_root: str,
) -> np.ndarray:
    """
    ClearVoice currently works most reliably through file I/O.
    Input: mono numpy waveform
    Output: mono numpy waveform at original sr
    """
    model = _load_clearvoice(
        model_name=model_name,
        clearvoice_root=clearvoice_root,
    )

    wav_np = safe_audio_np(wav_np)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_wav = td / "cv_in.wav"
        out_wav = td / "cv_out.wav"

        sf.write(str(in_wav), wav_np, sr)

        output_wav = model(input_path=str(in_wav), online_write=False)
        model.write(output_wav, output_path=str(out_wav))

        den, den_sr = sf.read(str(out_wav))

    den = safe_audio_np(den)

    if den_sr != sr:
        den = librosa.resample(den, orig_sr=den_sr, target_sr=sr)

    return safe_audio_np(den)


# =========================
# Demucs
# =========================
def _load_demucs(model_name: str = "htdemucs", device: str = "cuda"):
    global _demucs_model, _demucs_model_name
    if _demucs_model is None or _demucs_model_name != model_name:
        _demucs_model = get_model(model_name).to(device).eval()
        _demucs_model_name = model_name
    return _demucs_model


@torch.no_grad()
def keep_vocals_demucs(
    wav_np: np.ndarray,
    sr: int,
    model_name: str = "htdemucs",
    device: str = "cuda",
    split: bool = True,
    overlap: float = 0.25,
    target_sr: int = 44100,
) -> np.ndarray:
    """
    Keep vocals using Demucs.
    Input:
        wav_np: mono or stereo numpy waveform
        sr: input sample rate
    Output:
        mono numpy waveform at original sr
    """
    model = _load_demucs(model_name=model_name, device=device)

    x = torch.from_numpy(np.asarray(wav_np)).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)  # [1, T]

    # force 2 channels for Demucs
    if x.size(0) == 1:
        x = x.repeat(2, 1)  # [2, T]
    elif x.size(0) > 2:
        x = x[:2]

    x = x.unsqueeze(0).to(device)  # [1, 2, T]

    if sr != target_sr:
        x = julius_resample(x, sr, target_sr)

    sources = apply_model(
        model,
        x,
        device=device,
        split=split,
        overlap=overlap,
        progress=False,
    )

    stems = model.sources
    if "vocals" not in stems:
        raise RuntimeError(f"'vocals' stem not found in Demucs sources: {stems}")

    v_idx = stems.index("vocals")
    vocals = sources[0, v_idx]  # [2, T']

    vocals_mono = vocals.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, T']

    if sr != target_sr:
        vocals_mono = julius_resample(vocals_mono, target_sr, sr)

    out = vocals_mono[0, 0].detach().cpu().numpy()
    return safe_audio_np(out)


# =========================
# Unified preprocessing
# =========================
def maybe_preprocess_audio(
    wav_np: np.ndarray,
    sr: int,
    side: str,
    args,
):
    """
    side: 'source' or 'target'
    Order:
        required ClearVoice -> optional Demucs

    Returns:
        processed_audio_np, meta
    where meta records whether each stage was required/succeeded.
    """
    out = safe_audio_np(wav_np)
    meta = {
        "clearvoice_required": bool(args.clearvoice_denoise and args.clearvoice_on in (side, "both")),
        "clearvoice_ok": not bool(args.clearvoice_denoise and args.clearvoice_on in (side, "both")),
        "demucs_required": bool(args.demucs_preprocess and args.demucs_on in (side, "both")),
        "demucs_ok": not bool(args.demucs_preprocess and args.demucs_on in (side, "both")),
        "clearvoice_error": "",
        "demucs_error": "",
    }

    if meta["clearvoice_required"]:
        try:
            out = denoise_with_clearvoice(
                out,
                sr=sr,
                model_name=args.clearvoice_model,
                clearvoice_root=args.clearvoice_root,
            )
            meta["clearvoice_ok"] = True
        except Exception as e:
            meta["clearvoice_ok"] = False
            meta["clearvoice_error"] = repr(e)
            warnings.warn(f"ClearVoice failed on {side}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return safe_audio_np(out), meta

    if meta["demucs_required"]:
        try:
            out = keep_vocals_demucs(
                out,
                sr=sr,
                model_name=args.demucs_model,
                device=args.demucs_device,
                split=args.demucs_split,
                overlap=args.demucs_overlap,
                target_sr=args.demucs_target_sr,
            )
            meta["demucs_ok"] = True
        except Exception as e:
            meta["demucs_ok"] = False
            meta["demucs_error"] = repr(e)
            warnings.warn(f"Demucs failed on {side}: {e}")
            if str(args.demucs_device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return safe_audio_np(out), meta



# =========================
# VC conversion
# =========================
@torch.no_grad()
def convert_one(
    model_bundle,
    args,
    source_path: str,
    target_path: str,
    source_audio_raw=None,
    target_audio_raw=None,
):
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = model_bundle

    sr_cfg = mel_fn_args["sampling_rate"]
    f0_condition = args.f0_condition
    auto_f0_adjust = args.auto_f0_adjust
    pitch_shift = args.semi_tone_shift

    diffusion_steps = args.diffusion_steps
    length_adjust = args.length_adjust
    inference_cfg_rate = args.inference_cfg_rate

    # read at model config sr unless preloaded by CPU-side prefetch
    if source_audio_raw is None or target_audio_raw is None:
        source_audio_raw, target_audio_raw = load_pair_audio(source_path, target_path, sr_cfg)

    # required ClearVoice, then optional Demucs
    source_audio_np, source_pre_meta = maybe_preprocess_audio(source_audio_raw, sr_cfg, "source", args)
    target_audio_np, target_pre_meta = maybe_preprocess_audio(target_audio_raw, sr_cfg, "target", args)

    # Both source and target must pass ClearVoice before we allow VC to proceed.
    clearvoice_gate_ok = source_pre_meta["clearvoice_ok"] and target_pre_meta["clearvoice_ok"]
    if not clearvoice_gate_ok:
        raise RuntimeError(
            "PREPROCESS_CLEARVOICE_GATE_FAILED | "
            f"source_ok={source_pre_meta['clearvoice_ok']} source_err={source_pre_meta['clearvoice_error']} | "
            f"target_ok={target_pre_meta['clearvoice_ok']} target_err={target_pre_meta['clearvoice_error']}"
        )

    # keep consistent with your original inference path
    sr = 22050 if not f0_condition else 44100
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    source_audio = torch.tensor(source_audio_np).unsqueeze(0).float().to(inference.device)
    target_audio = torch.tensor(target_audio_np[: sr * 25]).unsqueeze(0).float().to(inference.device)

    t0 = time.time()

    # semantic tokenizer takes 16k
    source_audio_16k = torchaudio.functional.resample(source_audio, sr, 16000)

    if source_audio_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(source_audio_16k)
    else:
        overlapping_time = 5
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < source_audio_16k.size(-1):
            if buffer is None:
                chunk = source_audio_16k[:, traversed_time: traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [
                        buffer,
                        source_audio_16k[
                            :, traversed_time: traversed_time + 16000 * (30 - overlapping_time)
                        ],
                    ],
                    dim=-1,
                )

            S_alt_chunk = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_alt_chunk)
            else:
                S_alt_list.append(S_alt_chunk[:, 50 * overlapping_time:])

            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += (
                30 * 16000 if traversed_time == 0
                else chunk.size(-1) - 16000 * overlapping_time
            )

        S_alt = torch.cat(S_alt_list, dim=1)

    target_audio_16k = torchaudio.functional.resample(target_audio, sr, 16000)
    S_ori = semantic_fn(target_audio_16k)

    mel = mel_fn(source_audio.float())
    mel2 = mel_fn(target_audio.float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        target_audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    if f0_condition:
        F0_ori = f0_fn(target_audio_16k[0], thred=0.03)
        F0_alt = f0_fn(source_audio_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(inference.device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(inference.device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)

        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0_adjust:
            shifted_log_f0_alt[F0_alt > 1] = (
                log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            )

        shifted_f0_alt = torch.exp(shifted_log_f0_alt)

        if pitch_shift != 0:
            factor = 2 ** (pitch_shift / 12)
            shifted_f0_alt[F0_alt > 1] = shifted_f0_alt[F0_alt > 1] * factor
    else:
        F0_ori = None
        shifted_f0_alt = None

    cond, _, _, _, _ = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
    )
    prompt_condition, _, _, _, _ = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
    )

    max_source_window = max_context_window - mel2.size(2)

    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames: processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

        with torch.autocast(
            device_type=inference.device.type,
            dtype=torch.float16 if inference.fp16 else torch.float32,
        ):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1):]

        vc_wave = vocoder_fn(vc_target.float()).squeeze()[None, :]

        if processed_frames == 0:
            if is_last_chunk:
                generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                break
            generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

        elif is_last_chunk:
            chunk2 = vc_wave[0].cpu().numpy()
            chunk1 = previous_chunk.cpu().numpy()
            overlap = overlap_wave_len
            fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
            fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
            generated_wave_chunks.append(chunk2)
            break

        else:
            chunk2 = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            chunk1 = previous_chunk.cpu().numpy()
            overlap = overlap_wave_len
            fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
            fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
            generated_wave_chunks.append(chunk2)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    vc_wave_np = np.concatenate(generated_wave_chunks)
    vc_wave = torch.tensor(vc_wave_np)[None, :].float()

    t1 = time.time()
    rtf = (t1 - t0) / (vc_wave.size(-1) / sr)

    preprocess_meta = {
        "source_clearvoice_required": source_pre_meta["clearvoice_required"],
        "source_clearvoice_ok": source_pre_meta["clearvoice_ok"],
        "source_clearvoice_error": source_pre_meta["clearvoice_error"],
        "source_demucs_required": source_pre_meta["demucs_required"],
        "source_demucs_ok": source_pre_meta["demucs_ok"],
        "source_demucs_error": source_pre_meta["demucs_error"],
        "target_clearvoice_required": target_pre_meta["clearvoice_required"],
        "target_clearvoice_ok": target_pre_meta["clearvoice_ok"],
        "target_clearvoice_error": target_pre_meta["clearvoice_error"],
        "target_demucs_required": target_pre_meta["demucs_required"],
        "target_demucs_ok": target_pre_meta["demucs_ok"],
        "target_demucs_error": target_pre_meta["demucs_error"],
        "preprocess_gate_ok": clearvoice_gate_ok,
    }

    return vc_wave, sr, rtf, source_audio_np, target_audio_np, preprocess_meta


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_tsv", required=True, help="TSV header: id source target output, or legacy source target output")
    ap.add_argument("--output_dir", required=True, help="Directory for outputs/log bookkeeping")
    ap.add_argument("--output_sr", type=int, default=16000)
    ap.add_argument("--skip_existing", action="store_true")

    # same as inference.py
    ap.add_argument("--diffusion-steps", type=int, default=30)
    ap.add_argument("--length-adjust", type=float, default=1.0)
    ap.add_argument("--inference-cfg-rate", type=float, default=0.7)
    ap.add_argument("--f0-condition", type=inference.str2bool, default=False)
    ap.add_argument("--auto-f0-adjust", type=inference.str2bool, default=False)
    ap.add_argument("--semi-tone-shift", type=int, default=0)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--fp16", type=inference.str2bool, default=True)

    # Demucs
    ap.add_argument("--demucs-preprocess", type=inference.str2bool, default=False)
    ap.add_argument(
        "--demucs-on",
        type=str,
        default="both",
        choices=["source", "target", "both"],
        help="Apply Demucs vocal extraction on source, target, or both.",
    )
    ap.add_argument("--demucs-model", type=str, default="htdemucs")
    ap.add_argument("--demucs-device", type=str, default="cuda")
    ap.add_argument("--demucs-split", type=inference.str2bool, default=True)
    ap.add_argument("--demucs-overlap", type=float, default=0.25)
    ap.add_argument("--demucs-target-sr", type=int, default=44100)

    # ClearVoice
    ap.add_argument("--clearvoice-denoise", type=inference.str2bool, default=False)
    ap.add_argument(
        "--clearvoice-on",
        type=str,
        default="both",
        choices=["source", "target", "both"],
    )
    ap.add_argument(
        "--clearvoice-model",
        type=str,
        default="MossFormer2_SE_48K",
    )
    ap.add_argument(
        "--clearvoice-root",
        type=str,
        default=os.environ.get("CLEARVOICE_ROOT", "/path/to/ClearerVoice-Studio/clearvoice"),
        help="Parent dir of the actual clearvoice python package",
    )

    # save preprocessed audios
    ap.add_argument(
        "--save_preprocessed_audio",
        type=inference.str2bool,
        default=False,
        help="Whether to save demucs/denoise processed source and target audio."
    )
    ap.add_argument("--prefetch-workers", type=int, default=4)
    ap.add_argument("--prefetch-depth", type=int, default=16)

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = out_dir.parent / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "vc_manifest.tsv"

    prior_rows = load_prior_manifest(manifest_path) if args.skip_existing else {}
    model_bundle = inference.load_models(args)
    sr_cfg = model_bundle[6]["sampling_rate"]

    with manifest_path.open("w", encoding="utf-8", newline="") as mf:
        mf.write(
            "id\tsrc\ttgt\tout\tpre_src\tpre_tgt\tstatus\trtf\ttotal_elapsed_s\t"
            "demucs_preprocess\tdemucs_on\tdemucs_model\t"
            "clearvoice_denoise\tclearvoice_on\tclearvoice_model\t"
            "source_clearvoice_ok\ttarget_clearvoice_ok\t"
            "source_demucs_ok\ttarget_demucs_ok\tpreprocess_gate_ok\n"
        )

        def write_failed_row(i, src_p, tgt_p, out_wav, err):
            row_id = format_manifest_id(i)
            print(f"[{row_id}] FAILED: {src_p.name} -> {tgt_p.name} | {err}")
            mf.write(
                f"{row_id}\t{src_p.resolve()}\t{tgt_p.resolve()}\t{out_wav}\t\t\tFAILED:{repr(err)}\t\t\t"
                f"{args.demucs_preprocess}\t{args.demucs_on}\t{args.demucs_model}\t"
                f"{args.clearvoice_denoise}\t{args.clearvoice_on}\t{args.clearvoice_model}\t"
                f"\t\t\t\t\n"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def process_one_pair(i, src_p, tgt_p, out_wav, source_audio_raw=None, target_audio_raw=None):
            total_t0 = time.time()
            try:
                vc_wave, sr, rtf, src_audio_np, tgt_audio_np, preprocess_meta = convert_one(
                    model_bundle=model_bundle,
                    args=args,
                    source_path=str(src_p),
                    target_path=str(tgt_p),
                    source_audio_raw=source_audio_raw,
                    target_audio_raw=target_audio_raw,
                )

                pre_src_path = ""
                pre_tgt_path = ""

                if args.save_preprocessed_audio:
                    pre_src_save, pre_tgt_save = save_preprocessed_pair(
                        src_audio_np=src_audio_np,
                        tgt_audio_np=tgt_audio_np,
                        sr=sr,
                        out_wav=out_wav,
                        output_dir=out_dir,
                    )
                    pre_src_path = str(pre_src_save.resolve())
                    pre_tgt_path = str(pre_tgt_save.resolve())

                vc_wave_out = torchaudio.functional.resample(
                    vc_wave.cpu(), sr, args.output_sr
                )
                torchaudio.save(str(out_wav), vc_wave_out, args.output_sr)

                total_elapsed_s = time.time() - total_t0
                row_id = format_manifest_id(i)
                print(f"[{row_id}] saved {out_wav.name} | RTF={rtf:.3f} | total={total_elapsed_s:.3f}s")
                mf.write(
                    f"{row_id}\t{src_p.resolve()}\t{tgt_p.resolve()}\t{out_wav.resolve()}\t"
                    f"{pre_src_path}\t{pre_tgt_path}\tDONE\t{rtf:.6f}\t{total_elapsed_s:.6f}\t"
                    f"{args.demucs_preprocess}\t{args.demucs_on}\t{args.demucs_model}\t"
                    f"{args.clearvoice_denoise}\t{args.clearvoice_on}\t{args.clearvoice_model}\t"
                    f"{preprocess_meta['source_clearvoice_ok']}\t{preprocess_meta['target_clearvoice_ok']}\t"
                    f"{preprocess_meta['source_demucs_ok']}\t{preprocess_meta['target_demucs_ok']}\t{preprocess_meta['preprocess_gate_ok']}\n"
                )
            except Exception as e:
                write_failed_row(i, src_p, tgt_p, out_wav, e)

        pending = deque()
        prefetch_workers = max(0, int(args.prefetch_workers))
        prefetch_depth = max(1, int(args.prefetch_depth))
        executor = ThreadPoolExecutor(max_workers=prefetch_workers) if prefetch_workers > 0 else None

        with open(args.pair_tsv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)

            header_norm = [h.strip().lower() for h in header] if header else []
            if header_norm[:4] == ["id", "source", "target", "output"]:
                has_explicit_id = True
            elif header_norm[:3] == ["source", "target", "output"]:
                has_explicit_id = False
            else:
                raise ValueError(
                    "Expected TSV header [id, source, target, output] or "
                    f"[source, target, output], got {header}"
                )

            def drain_pending(force=False):
                while pending and (force or len(pending) >= prefetch_depth):
                    i2, src_p2, tgt_p2, out_wav2, fut = pending.popleft()
                    try:
                        source_audio_raw2, target_audio_raw2 = fut.result()
                    except Exception as e:
                        write_failed_row(i2, src_p2, tgt_p2, out_wav2, e)
                        continue
                    process_one_pair(
                        i2, src_p2, tgt_p2, out_wav2,
                        source_audio_raw=source_audio_raw2,
                        target_audio_raw=target_audio_raw2,
                    )

            try:
                for i, row in enumerate(reader):
                    if not row:
                        continue
                    if has_explicit_id:
                        if len(row) < 4:
                            continue
                        row_id = row[0].strip()
                        src = row[1].strip()
                        tgt = row[2].strip()
                        out_wav = Path(row[3].strip())
                    else:
                        if len(row) < 3:
                            continue
                        row_id = f"{i:03d}"
                        src = row[0].strip()
                        tgt = row[1].strip()
                        out_wav = Path(row[2].strip())

                    src_p = Path(src)
                    tgt_p = Path(tgt)
                    out_wav.parent.mkdir(parents=True, exist_ok=True)

                    if not src_p.exists():
                        print(f"[{row_id}] missing source: {src_p}")
                        mf.write(
                            f"{row_id}\t{src_p}\t{tgt_p}\t{out_wav}\t\t\tMISSING_SRC\t\t\t"
                            f"{args.demucs_preprocess}\t{args.demucs_on}\t{args.demucs_model}\t"
                            f"{args.clearvoice_denoise}\t{args.clearvoice_on}\t{args.clearvoice_model}\t"
                            f"\t\t\t\t\n"
                        )
                        continue

                    if not tgt_p.exists():
                        print(f"[{row_id}] missing target: {tgt_p}")
                        mf.write(
                            f"{row_id}\t{src_p}\t{tgt_p}\t{out_wav}\t\t\tMISSING_TGT\t\t\t"
                            f"{args.demucs_preprocess}\t{args.demucs_on}\t{args.demucs_model}\t"
                            f"{args.clearvoice_denoise}\t{args.clearvoice_on}\t{args.clearvoice_model}\t"
                            f"\t\t\t\t\n"
                        )
                        continue

                    prior_row = prior_rows.get(row_id)
                    if args.skip_existing and row_matches_current_pair(prior_row, row_id, src_p, tgt_p, out_wav) and row_is_resume_safe(prior_row, args.save_preprocessed_audio):
                        print(f"[{row_id}] resume verified: {out_wav.name}")
                        mf.write(
                            "\t".join([
                                str(prior_row.get("id", row_id)),
                                str(prior_row.get("src", str(src_p.resolve()))),
                                str(prior_row.get("tgt", str(tgt_p.resolve()))),
                                str(prior_row.get("out", str(out_wav.resolve()))),
                                str(prior_row.get("pre_src", "")),
                                str(prior_row.get("pre_tgt", "")),
                                str(prior_row.get("status", "DONE")),
                                str(prior_row.get("rtf", "")),
                                str(prior_row.get("total_elapsed_s", "")),
                                str(prior_row.get("demucs_preprocess", args.demucs_preprocess)),
                                str(prior_row.get("demucs_on", args.demucs_on)),
                                str(prior_row.get("demucs_model", args.demucs_model)),
                                str(prior_row.get("clearvoice_denoise", args.clearvoice_denoise)),
                                str(prior_row.get("clearvoice_on", args.clearvoice_on)),
                                str(prior_row.get("clearvoice_model", args.clearvoice_model)),
                                str(prior_row.get("source_clearvoice_ok", "True")),
                                str(prior_row.get("target_clearvoice_ok", "True")),
                                str(prior_row.get("source_demucs_ok", "")),
                                str(prior_row.get("target_demucs_ok", "")),
                                str(prior_row.get("preprocess_gate_ok", "True")),
                            ]) + "\n"
                        )
                        continue

                    if executor is None:
                        process_one_pair(row_id, src_p, tgt_p, out_wav)
                    else:
                        fut = executor.submit(load_pair_audio, str(src_p), str(tgt_p), sr_cfg)
                        pending.append((row_id, src_p, tgt_p, out_wav, fut))
                        drain_pending(force=False)

                drain_pending(force=True)
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=False)

    print("done. manifest:", manifest_path)


if __name__ == "__main__":
    main()
