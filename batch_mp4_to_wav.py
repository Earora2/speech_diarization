"""Convert the three participant MP4 recordings to mono 16 kHz WAV files."""

import argparse
from pathlib import Path

from mp4_to_wav import mp4_to_wav


def convert_recordings(input_files: list[str], output_dir: str) -> list[Path]:
    """Convert exactly three recordings, retaining each input filename stem."""
    if len(input_files) != 3:
        raise ValueError("Expected exactly three participant MP4 files")

    sources = [Path(name) for name in input_files]
    if any(source.suffix.lower() != ".mp4" for source in sources):
        raise ValueError("All inputs must be MP4 files")
    if len({source.stem for source in sources}) != 3:
        raise ValueError("Input filenames must have distinct stems")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for source in sources:
        output = destination / f"{source.stem}.wav"
        outputs.append(mp4_to_wav(str(source), str(output)))
        print(f"Created: {output}")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio from three participant MP4s")
    parser.add_argument("videos", nargs=3, metavar="MP4", help="Three participant MP4 files")
    parser.add_argument("--output-dir", default="audio", help="Output folder (default: audio)")
    args = parser.parse_args()
    convert_recordings(args.videos, args.output_dir)
