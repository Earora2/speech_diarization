"""Construct conversational turns from cleaned three-microphone activity.

Adapted from the turn-construction section of the supplied analysis notebook.
Input: outputs/cleaned_activity.csv from bleed_suppression.py.
"""
import argparse
from pathlib import Path

import pandas as pd

TURN_COLUMNS = ["speaker", "start", "end", "duration", "raw_segments"]
PAUSE_COLUMNS = ["speaker", "start", "end", "duration", "pause_type"]


def merge_within_speaker(diar: pd.DataFrame, max_pause: float = 1.5):
    """Merge consecutive same-speaker regions and retain positive internal pauses."""
    if max_pause < 0:
        raise ValueError("max_pause must be nonnegative")
    if diar.empty:
        return pd.DataFrame(columns=TURN_COLUMNS), pd.DataFrame(columns=PAUSE_COLUMNS)
    required = {"speaker", "start", "end"}
    if not required.issubset(diar.columns):
        raise ValueError(f"Activity CSV must contain {sorted(required)}")
    if (diar["end"] <= diar["start"]).any():
        raise ValueError("Every activity interval must have end > start")

    turns, pauses = [], []
    for speaker, group in diar.sort_values(["speaker", "start", "end"]).groupby("speaker", sort=False):
        segments = group[["start", "end"]].to_records(index=False)
        start, end = map(float, segments[0])
        raw = [(start, end)]
        for segment_start, segment_end in segments[1:]:
            segment_start, segment_end = float(segment_start), float(segment_end)
            gap = segment_start - end
            if gap <= max_pause:
                if gap > 0:
                    pauses.append({"speaker": speaker, "start": end, "end": segment_start,
                                   "duration": gap, "pause_type": "intra_turn"})
                end = max(end, segment_end)
                raw.append((segment_start, segment_end))
            else:
                turns.append({"speaker": speaker, "start": start, "end": end,
                              "duration": end - start, "raw_segments": raw})
                start, end = segment_start, segment_end
                raw = [(start, end)]
        turns.append({"speaker": speaker, "start": start, "end": end,
                      "duration": end - start, "raw_segments": raw})
    return (pd.DataFrame(turns, columns=TURN_COLUMNS).sort_values(["start", "end", "speaker"]).reset_index(drop=True),
            pd.DataFrame(pauses, columns=PAUSE_COLUMNS))


def classify_primary_secondary(candidate_turns: pd.DataFrame, containment_tolerance: float = 0.02):
    """A turn strictly contained in another speaker's turn is secondary."""
    t = candidate_turns.copy().reset_index(drop=True)
    if t.empty:
        t["turn_type"] = pd.Series(dtype=str)
        t["parent_candidate"] = pd.Series(dtype="object")
        primary = t.copy()
        primary["turn_id"] = pd.Series(dtype=int)
        secondary = t.copy()
        secondary["parent_turn_id"] = pd.Series(dtype="object")
        return primary, secondary

    labels, parents = [], []
    for i, row in t.iterrows():
        containers = []
        for j, other in t.iterrows():
            if i == j or row.speaker == other.speaker:
                continue
            contained = (row.start >= other.start - containment_tolerance and
                         row.end <= other.end + containment_tolerance)
            strictly_larger = (other.start < row.start - containment_tolerance or
                               other.end > row.end + containment_tolerance)
            if contained and strictly_larger:
                containers.append((j, other.end - other.start))
        parent = max(containers, key=lambda item: item[1])[0] if containers else None
        labels.append("secondary" if parent is not None else "primary")
        parents.append(parent)
    t["turn_type"] = labels
    t["parent_candidate"] = pd.Series(parents, dtype="object")

    # Match notebook's handling of nested secondary candidates.
    for i in t.index[t["turn_type"] == "secondary"]:
        parent = t.at[i, "parent_candidate"]
        if parent is not None and t.at[int(parent), "turn_type"] != "primary":
            t.at[i, "turn_type"] = "primary"
            t.at[i, "parent_candidate"] = None

    primary = t[t.turn_type == "primary"].copy().sort_values(["start", "end"]).reset_index(drop=True)
    secondary = t[t.turn_type == "secondary"].copy().sort_values(["start", "end"]).reset_index(drop=True)
    primary["turn_id"] = range(1, len(primary) + 1)
    parent_ids = []
    for _, row in secondary.iterrows():
        matches = primary[(primary.speaker != row.speaker) &
                          (primary.start <= row.start + containment_tolerance) &
                          (primary.end >= row.end - containment_tolerance)]
        parent_ids.append(int(matches.sort_values("duration", ascending=False).iloc[0].turn_id)
                          if not matches.empty else None)
    secondary["parent_turn_id"] = parent_ids
    return primary, secondary


def compute_turn_intervals(primary: pd.DataFrame) -> pd.DataFrame:
    """Positive intervals are gaps; negative intervals are overlaps; zero is a latch."""
    columns = ["from_turn", "to_turn", "from_speaker", "to_speaker", "start", "end",
               "interval", "interval_type"]
    p = primary.sort_values(["start", "end"]).reset_index(drop=True)
    rows = []
    for i in range(len(p) - 1):
        a, b = p.iloc[i], p.iloc[i + 1]
        delta = float(b.start - a.end)
        rows.append({"from_turn": int(a.turn_id), "to_turn": int(b.turn_id),
                     "from_speaker": a.speaker, "to_speaker": b.speaker,
                     "start": min(float(a.end), float(b.start)),
                     "end": max(float(a.end), float(b.start)), "interval": delta,
                     "interval_type": "gap" if delta > 0 else ("overlap" if delta < 0 else "latch")})
    return pd.DataFrame(rows, columns=columns)


def main():
    parser = argparse.ArgumentParser(description="Construct primary/secondary conversational turns")
    parser.add_argument("--input", default="outputs/cleaned_activity.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-pause", type=float, default=1.5)
    args = parser.parse_args()
    diar = pd.read_csv(args.input)
    candidates, pauses = merge_within_speaker(diar, max_pause=args.max_pause)
    primary, secondary = classify_primary_secondary(candidates)
    intervals = compute_turn_intervals(primary)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("candidate_turns", candidates), ("primary_turns", primary),
                        ("secondary_turns", secondary), ("intra_turn_pauses", pauses),
                        ("turn_intervals", intervals)]:
        frame.to_csv(output / f"{name}.csv", index=False)
        print(f"Saved {len(frame)} rows: {output / (name + '.csv')}")


if __name__ == "__main__":
    main()
