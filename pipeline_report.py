"""Check pipeline outputs and generate a local visualization and JSON summary.

This integrates existing CSV outputs; it does not re-run audio processing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from timeline_visualization import draw_timeline, read_intervals

STAGES = {
    'activity': 'raw_activity.csv',
    'bleed_suppression': 'cleaned_activity.csv',
    'candidate_turns': 'candidate_turns.csv',
    'primary_turns': 'primary_turns.csv',
    'secondary_turns': 'secondary_turns.csv',
    'backchannels': 'classified_secondary_turns.csv',
    'transcript_words': 'transcript_words.csv',
    'vocal_events': 'vocal_events.csv',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error('--dpi must be at least 72')
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    status = {stage: (directory / filename).is_file() for stage, filename in STAGES.items()}
    primary_path = directory / STAGES['primary_turns']
    summary = {'stages': status, 'visualization': None, 'counts': {}}
    if status['primary_turns']:
        primary = read_intervals(primary_path)
        secondary = read_intervals(directory / STAGES['secondary_turns'], optional=True)
        events = read_intervals(directory / STAGES['vocal_events'], optional=True)
        summary['counts'] = {'primary_turns': len(primary), 'secondary_turns': len(secondary),
                             'vocal_events': len(events)}
        if not all(frame.empty for frame in [primary, secondary, events]):
            image_path = directory / 'speaker_timeline.png'
            draw_timeline(primary, secondary, events, image_path, args.dpi)
            summary['visualization'] = str(image_path)
    else:
        print(f'Cannot plot timeline: missing {primary_path}')
    report = directory / 'pipeline_summary.json'
    report.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Saved report: {report}')
    for stage, exists in status.items():
        print(f'{"OK     " if exists else "MISSING"}  {stage}: {STAGES[stage]}')
    if summary['visualization']:
        print(f'Saved timeline: {summary["visualization"]}')


if __name__ == '__main__':
    main()
