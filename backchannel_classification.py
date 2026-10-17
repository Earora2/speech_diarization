"""Transcribe and label German secondary turns from synchronized microphone audio.

Uses the lexical rules in the supplied diarization notebook. Audio stays local;
faster-whisper may download model weights on its first run.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

GERMAN_BACKCHANNEL_CUES = {
    "ja", "jaja", "mhm", "hm", "hmm", "mm", "mmm", "genau", "okay",
    "ok", "klar", "stimmt", "richtig", "achso", "ach so", "aha",
    "jo", "jep", "super", "interessant", "wirklich", "cool",
}
GERMAN_NOT_BACKCHANNEL_STARTERS = {
    "ich", "du", "er", "sie", "es", "wir", "ihr", "und", "aber",
    "also", "weil", "denn", "dann", "wenn", "dass", "ob", "was",
    "warum", "wie", "wann", "wo", "wer", "welche", "welcher", "welches",
}


def classify_german_secondary(text, max_words=3, min_backchannel_proportion=0.5):
    """Classify transcribed secondary speech: backchannel, secondary, or other."""
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^\wäöüß\s-]", " ", str(text).lower())).strip()
    if not cleaned:
        return "other"
    words = cleaned.split()
    if cleaned in GERMAN_BACKCHANNEL_CUES:
        return "backchannel"
    if len(words) > max_words or words[0] in GERMAN_NOT_BACKCHANNEL_STARTERS:
        return "secondary"
    cue_hits = sum(word in GERMAN_BACKCHANNEL_CUES for word in words)
    return "backchannel" if cue_hits / len(words) >= min_backchannel_proportion else "secondary"


def transcribe_words(audio_path, model, language="de"):
    """Transcribe one recording and return timestamped words."""
    segments, _ = model.transcribe(str(audio_path), language=language,
                                   word_timestamps=True, vad_filter=False)
    rows = []
    for segment in segments:
        for word in segment.words or []:
            if word.start is not None and word.end is not None:
                rows.append({"start": float(word.start), "end": float(word.end),
                             "word": word.word.strip()})
    return pd.DataFrame(rows, columns=["start", "end", "word"])


def attach_text_to_segments(segments, words):
    """Match words by speaker and overlapping time, following notebook logic."""
    result = segments.copy()
    if result.empty:
        result["text"] = pd.Series(dtype=str)
        return result
    texts = []
    for _, turn in result.iterrows():
        matches = words[(words["speaker"] == turn["speaker"]) &
                        (words["end"] > turn["start"]) &
                        (words["start"] < turn["end"])].sort_values("start")
        texts.append(" ".join(matches["word"].astype(str)).strip())
    result["text"] = texts
    return result


def main():
    parser = argparse.ArgumentParser(description="German secondary-turn transcription and labeling")
    parser.add_argument("--secondary", default="outputs/secondary_turns.csv")
    parser.add_argument("--audio", nargs=3, required=True, metavar="WAV",
                        help="WAV paths corresponding to speaker_0, speaker_1, speaker_2")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model", default="small", help="faster-whisper model size")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu",
                        help="Use cpu on macOS (including Apple Silicon)")
    args = parser.parse_args()

    secondary = pd.read_csv(args.secondary)
    required = {"speaker", "start", "end"}
    if not required.issubset(secondary.columns):
        raise ValueError(f"Secondary turns CSV must contain: {sorted(required)}")
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device=args.device,
                         compute_type="int8" if args.device == "cpu" else "default")
    all_words = []
    for index, path in enumerate(args.audio):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        speaker = f"speaker_{index}"
        words = transcribe_words(path, model)
        words["speaker"] = speaker
        all_words.append(words)
        print(f"Transcribed {speaker}: {len(words)} words")
    words = pd.concat(all_words, ignore_index=True)
    labeled = attach_text_to_segments(secondary, words)
    labeled["secondary_type"] = labeled["text"].map(classify_german_secondary)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    words.to_csv(output / "transcript_words.csv", index=False)
    labeled.to_csv(output / "classified_secondary_turns.csv", index=False)
    print(f"Saved {output / 'classified_secondary_turns.csv'}")
    print(f"Saved {output / 'transcript_words.csv'}")
    print("Note: transcript-based duplicate suppression is not yet applied.")


if __name__ == "__main__":
    main()
