"""Convert a participant's MP4 recording to 16 kHz mono WAV audio."""

import argparse
import subprocess
from pathlib import Path


def mp4_to_wav(input_mp4: str, output_wav: str | None = None) -> Path:
    """Extract PCM audio with FFmpeg and return the output path."""
    source = Path(input_mp4)
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")

    destination = Path(output_wav) if output_wav else source.with_suffix(".wav")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source), "-vn",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(destination),
        ],
        check=True,
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP4 to mono 16 kHz WAV")
    parser.add_argument("input_mp4", help="Input MP4 video")
    parser.add_argument("output_wav", nargs="?", help="Optional WAV output path")
    args = parser.parse_args()
    print(f"Created: {mp4_to_wav(args.input_mp4, args.output_wav)}")
