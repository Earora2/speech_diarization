"""Suppress likely cross-microphone pickup in synchronized 16 kHz mono WAVs.

This is an acoustic heuristic, not speaker verification. Real overlap may be
removed when one microphone is substantially louder; review results.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from speaker_activity import run_multi_speaker_activity

SAMPLE_RATE = 16000


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if min(len(x), len(y)) < 8:
        return 0.0
    x = x.astype(np.float64) - float(np.mean(x))
    y = y.astype(np.float64) - float(np.mean(y))
    norm = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / norm) if norm > 1e-8 else 0.0


def suppress_cross_mic_bleed(
    audio_paths: dict[str, str],
    activity: pd.DataFrame,
    frame_sec: float = 0.10,
    margin_db: float = 3.0,
    correlation_threshold: float = 0.75,
    min_segment_sec: float = 0.10,
) -> pd.DataFrame:
    """Keep likely local microphone activity using level and correlation.

    Audio must be synchronized and have the same 16 kHz mono sample rate.
    Processing stops at the shortest recording duration.
    """
    columns = ["speaker", "start", "end", "duration", "source_audio"]
    if not audio_paths:
        return pd.DataFrame(columns=columns)
    if frame_sec <= 0 or min_segment_sec < 0 or margin_db < 0:
        raise ValueError("Frame size and margin must be positive; minimum duration nonnegative")
    if not 0 <= correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be between 0 and 1")

    audio = {}
    for speaker, path in audio_paths.items():
        samples, sr = sf.read(path, dtype="float32")
        if sr != SAMPLE_RATE or samples.ndim != 1:
            raise ValueError(f"{path}: expected mono {SAMPLE_RATE} Hz WAV")
        audio[speaker] = samples

    n_samples = min(map(len, audio.values()))
    frame_samples = max(1, round(frame_sec * SAMPLE_RATE))
    n_frames = (n_samples + frame_samples - 1) // frame_samples
    if n_frames == 0:
        return pd.DataFrame(columns=columns)

    starts = np.arange(n_frames) * frame_samples / SAMPLE_RATE
    ends = np.minimum((np.arange(n_frames) + 1) * frame_samples, n_samples) / SAMPLE_RATE
    masks = {}
    levels = {}
    eps = 1e-10
    for speaker, samples in audio.items():
        mask = np.zeros(n_frames, dtype=bool)
        for row in activity.loc[activity["speaker"] == speaker].itertuples():
            mask |= (starts < row.end) & (ends > row.start)
        masks[speaker] = mask
        rms = np.array([
            np.sqrt(np.mean(np.square(samples[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)].astype(np.float64))))
            for a, b in zip(starts, ends)
        ])
        reference_values = rms[mask] if np.any(mask) else rms
        reference = max(float(np.percentile(reference_values, 90)), eps)
        levels[speaker] = 20 * np.log10(np.maximum(rms, eps) / reference)

    keep = {speaker: np.zeros(n_frames, dtype=bool) for speaker in audio}
    for i in range(n_frames):
        active = [speaker for speaker in audio if masks[speaker][i]]
        if not active:
            continue
        strongest = max(levels[speaker][i] for speaker in active)
        candidates = [speaker for speaker in active if levels[speaker][i] >= strongest - margin_db]
        survivors = set(candidates)
        a, b = i * frame_samples, min((i + 1) * frame_samples, n_samples)
        for j, first in enumerate(candidates):
            for second in candidates[j + 1:]:
                if first not in survivors or second not in survivors:
                    continue
                if abs(_safe_corr(audio[first][a:b], audio[second][a:b])) >= correlation_threshold:
                    loser = second if levels[first][i] >= levels[second][i] else first
                    survivors.discard(loser)
        for speaker in survivors:
            keep[speaker][i] = True

    rows = []
    for speaker, mask in keep.items():
        edges = np.diff(np.r_[False, mask, False].astype(np.int8))
        for begin, finish in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            start, end = float(starts[begin]), float(ends[finish - 1])
            if end - start >= min_segment_sec - 1e-9:
                rows.append({"speaker": speaker, "start": round(start, 3),
                             "end": round(end, 3), "duration": round(end - start, 3),
                             "source_audio": str(audio_paths[speaker])})
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["start", "end", "speaker"], ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Suppress likely bleed across three synchronized microphones")
    parser.add_argument("wav", nargs=3, metavar="WAV", help="WAVs in speaker order")
    parser.add_argument("--output", default="outputs/cleaned_activity.csv")
    parser.add_argument("--margin-db", type=float, default=3.0)
    parser.add_argument("--correlation-threshold", type=float, default=0.75)
    args = parser.parse_args()
    paths = {f"speaker_{i}": path for i, path in enumerate(args.wav)}
    raw = run_multi_speaker_activity(paths)
    cleaned = suppress_cross_mic_bleed(paths, raw, margin_db=args.margin_db,
                                       correlation_threshold=args.correlation_threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)
    print(f"Saved {len(cleaned)} cleaned intervals to {output}")


if __name__ == "__main__":
    main()
