"""Detect non-verbal vocal events on three synchronized participant microphones.

Adapted from the non-verbal vocalizations section of the supplied notebook.
The WAV files and generated CSV remain local. The pretrained model may be
retrieved from Hugging Face on first use.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
EVENT_PATTERNS = {
    "laughter": [r"laugh", r"giggle", r"chuckle"],
    "humming": [r"humming", r"\bhum\b"],
    "cough": [r"cough"],
    "sneeze": [r"sneeze"],
    "sigh": [r"sigh"],
    "breath": [r"breathing", r"breath"],
    "throat_clear": [r"throat clearing"],
    "cry": [r"crying", r"sob"],
    "whisper": [r"whisper"],
}
COLUMNS = ["speaker", "start", "end", "event", "raw_label", "score"]


def normalize_event_label(label: str) -> str | None:
    label = label.lower()
    for event, patterns in EVENT_PATTERNS.items():
        if any(re.search(pattern, label) for pattern in patterns):
            return event
    return None


def merge_events(events: pd.DataFrame, hop: float) -> pd.DataFrame:
    """Combine adjacent overlapping event windows for the same microphone."""
    if events.empty:
        return pd.DataFrame(columns=COLUMNS)
    merged = []
    for (_, _), group in events.groupby(["speaker", "event"], sort=False):
        current = None
        for row in group.sort_values("start").to_dict("records"):
            if current is None:
                current = row.copy()
            elif row["start"] <= current["end"] + hop * 0.55:
                current["end"] = max(current["end"], row["end"])
                if row["score"] > current["score"]:
                    current["score"] = row["score"]
                    current["raw_label"] = row["raw_label"]
            else:
                merged.append(current)
                current = row.copy()
        if current is not None:
            merged.append(current)
    return pd.DataFrame(merged, columns=COLUMNS).sort_values(
        ["start", "speaker", "event"], ignore_index=True
    )


def detect_for_speaker(path, speaker, activity, extractor, model, device,
                       window=1.0, hop=0.5, threshold=0.22):
    import torch

    samples, sr = sf.read(path, dtype="float32")
    if sr != 16000 or samples.ndim != 1:
        raise ValueError(f"{path}: expected mono 16000 Hz WAV")
    duration = len(samples) / sr
    intervals = activity.loc[activity["speaker"] == speaker, ["start", "end"]]
    if intervals.empty:
        return pd.DataFrame(columns=COLUMNS)

    rows = []
    t = 0.0
    while t < duration:
        end = min(t + window, duration)
        if end - t < 0.2:
            break
        if not ((intervals["start"] < end) & (intervals["end"] > t)).any():
            t += hop
            continue
        clip = samples[int(t * sr):int(end * sr)]
        inputs = extractor(clip, sampling_rate=sr, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
            # Preserve the original notebook's multi-label scoring heuristic.
            scores = torch.sigmoid(logits)
            top = torch.topk(scores, k=min(15, scores.numel()))
        for score, index in zip(top.values.tolist(), top.indices.tolist()):
            raw_label = model.config.id2label[int(index)]
            event = normalize_event_label(raw_label)
            if event and score >= threshold:
                rows.append({"speaker": speaker, "start": round(t, 3),
                             "end": round(end, 3), "event": event,
                             "raw_label": raw_label, "score": float(score)})
        t += hop
    return merge_events(pd.DataFrame(rows, columns=COLUMNS), hop)


def main():
    parser = argparse.ArgumentParser(description="Detect non-verbal vocalizations")
    parser.add_argument("--audio", nargs=3, required=True, metavar="WAV",
                        help="Recordings for speaker_0, speaker_1, speaker_2")
    parser.add_argument("--activity", default="outputs/cleaned_activity.csv")
    parser.add_argument("--output", default="outputs/vocal_events.csv")
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--hop", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    if args.window <= 0 or args.hop <= 0 or not 0 <= args.threshold <= 1:
        parser.error("window/hop must be positive and threshold must be in [0, 1]")
    for path in args.audio:
        if not Path(path).is_file():
            parser.error(f"Missing audio file: {path}")

    activity = pd.read_csv(args.activity)
    required = {"speaker", "start", "end"}
    if not required.issubset(activity.columns):
        raise ValueError(f"Activity CSV must contain {sorted(required)}")

    import torch
    from transformers import AutoFeatureExtractor, ASTForAudioClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading AST model on {device} (first run may download weights)")
    extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = ASTForAudioClassification.from_pretrained(args.model).to(device).eval()
    frames = []
    for index, path in enumerate(args.audio):
        speaker = f"speaker_{index}"
        print(f"Analyzing {speaker}")
        frames.append(detect_for_speaker(path, speaker, activity, extractor, model,
                                         device, args.window, args.hop, args.threshold))
    result = pd.concat(frames, ignore_index=True)
    if not result.empty:
        result = result.sort_values(["start", "speaker", "event"], ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} vocal events to {output}")
    print("Review predictions manually: event scores are heuristic, not calibrated probabilities.")


if __name__ == "__main__":
    main()
