"""Render local diarization CSVs as speaker timelines (no audio uploaded)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

REQUIRED = {'speaker', 'start', 'end'}
COLORS = ['#2563eb', '#f59e0b', '#10b981', '#a855f7', '#ef4444', '#06b6d4']


def read_intervals(path: str | Path, *, optional: bool = False) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        if optional:
            return pd.DataFrame(columns=['speaker', 'start', 'end'])
        raise FileNotFoundError(f'Missing required CSV: {path}')
    data = pd.read_csv(path)
    missing = REQUIRED.difference(data.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}; found {list(data.columns)}')
    data = data.copy()
    data['start'] = pd.to_numeric(data['start'], errors='coerce')
    data['end'] = pd.to_numeric(data['end'], errors='coerce')
    data = data.dropna(subset=['speaker', 'start', 'end'])
    data = data.loc[(data['start'] >= 0) & (data['end'] > data['start'])]
    return data.sort_values(['start', 'speaker']).reset_index(drop=True)


def draw_timeline(primary: pd.DataFrame, secondary: pd.DataFrame,
                  events: pd.DataFrame, output: str | Path, dpi: int = 200) -> None:
    all_data = pd.concat([primary, secondary, events], ignore_index=True)
    if all_data.empty:
        raise ValueError('No valid intervals to visualize')
    speakers = sorted(all_data['speaker'].astype(str).unique())
    positions = {speaker: i for i, speaker in enumerate(speakers)}
    colors = {speaker: COLORS[i % len(COLORS)] for i, speaker in enumerate(speakers)}
    duration = float(all_data['end'].max())
    fig, ax = plt.subplots(figsize=(max(12, min(24, duration / 20)), max(3.5, len(speakers) * 1.1 + 1.6)))
    for data, height, offset, alpha in [(primary, 0.32, -0.19, 0.9),
                                         (secondary, 0.20, 0.17, 0.7)]:
        for row in data.itertuples(index=False):
            speaker = str(row.speaker)
            ax.broken_barh([(float(row.start), float(row.end - row.start))],
                           (positions[speaker] + offset - height / 2, height),
                           facecolors=colors[speaker], alpha=alpha)
    for row in events.itertuples(index=False):
        speaker = str(row.speaker)
        event = str(getattr(row, 'event', 'vocal event'))
        center = (float(row.start) + float(row.end)) / 2
        ax.plot(center, positions[speaker] + 0.39, marker='o', markersize=4,
                color='#be123c', linestyle='none')
        if len(events) <= 35:
            ax.annotate(event, (center, positions[speaker] + 0.39),
                        xytext=(0, 5), textcoords='offset points', fontsize=6,
                        rotation=35, ha='left')
    ax.set_yticks(range(len(speakers)), speakers)
    ax.set_ylim(-0.7, len(speakers) - 0.1)
    ax.set_xlim(0, max(duration * 1.01, 1))
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Speaker turns and non-verbal vocalizations')
    ax.grid(axis='x', alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor='#64748b', alpha=0.9, label='Primary turn (lower bar)'),
                       Patch(facecolor='#64748b', alpha=0.5, label='Secondary turn (upper bar)'),
                       plt.Line2D([], [], color='#be123c', marker='o', linestyle='none', label='Vocal event')],
              loc='upper right', fontsize=8)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', default='outputs/primary_turns.csv')
    parser.add_argument('--secondary', default='outputs/secondary_turns.csv')
    parser.add_argument('--events', default='outputs/vocal_events.csv')
    parser.add_argument('--output', default='outputs/speaker_timeline.png')
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error('--dpi must be at least 72')
    primary = read_intervals(args.primary)
    secondary = read_intervals(args.secondary, optional=True)
    events = read_intervals(args.events, optional=True)
    draw_timeline(primary, secondary, events, args.output, args.dpi)
    print(f'Saved timeline: {args.output}')
    if not Path(args.events).is_file():
        print('Note: vocal events CSV missing; plotted turns only.')


if __name__ == '__main__':
    main()
