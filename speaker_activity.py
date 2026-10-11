"""Initial per-microphone speech activity baseline for synchronized recordings.

Each WAV corresponds to one known participant. This first-stage energy detector
is intentionally simple; later commits add cross-microphone bleed suppression
and more advanced speech detection from the original notebook.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SAMPLE_RATE = 16000
FRAME_SEC = 0.10
MIN_ACTIVITY_SEC = 0.20
RELATIVE_THRESHOLD = 0.15


def detect_activity(path: str, speaker: str) -> pd.DataFrame:
    """Return candidate active intervals from a single mono 16 kHz WAV."""
    audio, sr = sf.read(path, dtype="float32")
    if sr != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"{path}: expected mono {SAMPLE_RATE} Hz WAV")
    frame_size = int(sr * FRAME_SEC)
    n_frames = len(audio) // frame_size
    if n_frames == 0:
        return pd.DataFrame(columns=["speaker", "start", "end"])

    frames = audio[:n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    # Threshold relative to a robust high-energy reference; avoids pure zeros.
    threshold = max(1e-4, float(np.percentile(rms, 90)) * RELATIVE_THRESHOLD)
    active = rms >= threshold

    rows = []
    begin = None
    for idx, is_active in enumerate(np.r_[active, False]):
        if is_active and begin is None:
            begin = idx
        elif not is_active and begin is not None:
            start, end = begin * FRAME_SEC, idx * FRAME_SEC
            if end - start >= MIN_ACTIVITY_SEC:
                rows.append({"speaker": speaker, "start": round(start, 3), "end": round(end, 3)})
            begin = None
    return pd.DataFrame(rows, columns=["speaker", "start", "end"])


def run_multi_speaker_activity(paths: dict[str, str]) -> pd.DataFrame:
    """Detect candidate activity independently for each known microphone."""
    results = [detect_activity(path, speaker) for speaker, path in paths.items()]
    return pd.concat(results, ignore_index=True).sort_values(
        ["start", "speaker"], ignore_index=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate activity in three synchronized WAVs")
    parser.add_argument("wav", nargs=3, metavar="WAV", help="WAVs in speaker order")
    parser.add_argument("--output", default="outputs/raw_activity.csv")
    args = parser.parse_args()
    paths = {f"speaker_{i}": name for i, name in enumerate(args.wav)}
    result = run_multi_speaker_activity(paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} candidate activity intervals to {output}")
