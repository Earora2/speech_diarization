# Speech Diarization — Three-Participant Recordings

A Python research pipeline for analyzing **three synchronized, individually recorded microphones**. It extracts audio from MP4 files, estimates speaker activity, suppresses cross-microphone bleed, constructs conversational turns, identifies possible German backchannels and non-verbal vocalizations, and visualizes the resulting timelines.

> **Privacy:** Audio/video recordings, transcripts, intermediate CSVs, and generated figures are intended to remain local. Do not commit participant data or credentials. Review the **entire Git history** before publishing: `.gitignore` does not remove files already committed.

## What it does

| Stage                       | Script                                  | Main output                                                              |
| --------------------------- | --------------------------------------- | ------------------------------------------------------------------------ |
| MP4 to WAV                  | `mp4_to_wav.py` / `batch_mp4_to_wav.py` | Mono, 16 kHz WAV files in `audio/`                                       |
| Speaker activity            | `speaker_activity.py`                   | `outputs/raw_activity.csv`                                               |
| Cross-mic bleed suppression | `bleed_suppression.py`                  | `outputs/cleaned_activity.csv`                                           |
| Turn construction           | `turn_construction.py`                  | Primary, secondary, candidate turns and pause CSVs                       |
| German backchannels         | `backchannel_classification.py`         | `outputs/transcript_words.csv`, `outputs/classified_secondary_turns.csv` |
| Non-verbal vocalizations    | `vocal_event_detection.py`              | `outputs/vocal_events.csv`                                               |
| Visualization               | `timeline_visualization.py`             | `outputs/speaker_timeline.png`                                           |
| Pipeline report             | `pipeline_report.py`                    | `outputs/pipeline_summary.json` and timeline image                       |

This is a **microphone-based diarization and turn-analysis prototype**, not a validated speaker-recognition or clinical annotation system. Classification and activity thresholds are heuristic; manually review outputs before interpreting them.

## Requirements

- macOS or another platform with Python 3.12 and `ffmpeg` available on `PATH`.
- Three time-aligned recordings, one per participant. Input order maps to `speaker_0`, `speaker_1`, and `speaker_2`.
- Python packages in `requirements.txt`, including NumPy, pandas, SoundFile, Faster-Whisper, PyAV, PyTorch, Transformers, and Matplotlib.
- Internet access on the first model run to download pretrained model weights. Audio is processed locally by these scripts; inspect third-party libraries and network behavior as appropriate for your privacy requirements.

### Set up on macOS

```bash
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

If `faster-whisper` fails with `TypeError: open() got an unexpected keyword argument 'metadata_errors'`, check your installed `av` version. The project has used `faster-whisper==1.2.1` with `av==16.0.1`; verify that this combination works in your environment.

## Run the pipeline

Run these commands from the repository root. The example input filenames are placeholders; replace them with your **local** recording names. Do not upload recordings to GitHub.

### 1. Extract three WAV files

```bash
python batch_mp4_to_wav.py \
  audio/participant_1.mp4 \
  audio/participant_2.mp4 \
  audio/participant_3.mp4 \
  --output-dir audio
```

This produces `audio/participant_1.wav`, `audio/participant_2.wav`, and `audio/participant_3.wav`. If you already have synchronized mono 16 kHz WAVs, skip conversion.

### 2. Detect activity and suppress bleed

```bash
python speaker_activity.py \
  audio/participant_1.wav audio/participant_2.wav audio/participant_3.wav

python bleed_suppression.py \
  audio/participant_1.wav audio/participant_2.wav audio/participant_3.wav
```

The bleed-suppression script calculates its own initial activity estimates from the WAVs; it does not read `raw_activity.csv` as an input.

### 3. Construct turns

```bash
python turn_construction.py \
  --input outputs/cleaned_activity.csv \
  --output-dir outputs \
  --max-pause 1.5
```

This generates `candidate_turns.csv`, `primary_turns.csv`, `secondary_turns.csv`, `intra_turn_pauses.csv`, and `turn_intervals.csv` under `outputs/`.

### 4. Classify possible German backchannels

```bash
python backchannel_classification.py \
  --secondary outputs/secondary_turns.csv \
  --audio audio/participant_1.wav audio/participant_2.wav audio/participant_3.wav \
  --output-dir outputs \
  --model small \
  --device cpu
```

This step uses Faster-Whisper to transcribe speech and classify short responses such as _ja_, _mhm_, _genau_, and _okay_. These classifications are approximate and depend on transcription quality.

### 5. Detect possible non-verbal vocalizations

```bash
python vocal_event_detection.py \
  --audio audio/participant_1.wav audio/participant_2.wav audio/participant_3.wav \
  --activity outputs/cleaned_activity.csv \
  --output outputs/vocal_events.csv
```

The AudioSet-trained AST model scores candidate events such as laughter, coughing, sighing, breathing, and throat clearing. Scores are heuristic, not calibrated probabilities; false positives are possible.

### 6. Draw timelines and create a summary

```bash
python timeline_visualization.py \
  --primary outputs/primary_turns.csv \
  --secondary outputs/secondary_turns.csv \
  --events outputs/vocal_events.csv \
  --output outputs/speaker_timeline.png \
  --dpi 200

python pipeline_report.py --output-dir outputs
```

The reporting step checks for existing outputs; it does **not** rerun the audio-analysis stages. The visualization can omit `vocal_events.csv` if that stage has not been run.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── mp4_to_wav.py
├── batch_mp4_to_wav.py
├── speaker_activity.py
├── bleed_suppression.py
├── turn_construction.py
├── backchannel_classification.py
├── vocal_event_detection.py
├── timeline_visualization.py
├── pipeline_report.py
├── audio/                 # local only; ignored
└── outputs/               # local only; ignored
```

## Data privacy and Git

The `.gitignore` should exclude at least `audio/`, `outputs/`, `*.mp4`, `*.wav`, `*.mp3`, `*.m4a`, `*.csv`, `*.png`, `.venv/`, `.env`, `__pycache__/`, and `*.pyc`. Avoid adding any real participant names or paths to example commands, issues, or commit messages.

Before the first push, check both currently tracked files and historical filenames:

```bash
git ls-files
git log --all --name-only --pretty=format: | sort -u
```

If sensitive data appears in any commit, stop and clean the Git history before publishing.

## Limitations

- Assumes three synchronized recordings, each dominated by one participant.
- Energy thresholds, correlation-based bleed suppression, and turn merging can miss quiet speech or misattribute overlap.
- Whisper backchannels and AST vocal events are model predictions, not ground-truth labels.
- CPU inference can be slow, particularly on long recordings.
- No participant recordings or example outputs are distributed with this repository.
