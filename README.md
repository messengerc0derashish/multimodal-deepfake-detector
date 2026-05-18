# Multimodal Deepfake Detector

A full-stack deepfake detection project with a FastAPI backend, React/Vite frontend, and multimodal model pipeline for video, audio, and optional transcript evidence.

The app accepts a video upload, runs the analysis pipeline, and returns a final verdict with confidence, modality scores, suspicious frames, plots, and explanation text.

## Features

- Frame-level video analysis with fake/real frame ratio scoring
- Audio-based scoring from waveform/spectrogram features
- Optional transcript/text evidence when available
- Frame-ratio verdict bands: `REAL`, `SUSPICIOUS`, or `FAKE`
- Formula-based final decision score using frame, audio, and text signals
- Suspicious frame preview and modality breakdown plots
- React frontend for upload, preview, progress, and results
- FastAPI backend with async job flow and sync analysis endpoint

## Project Structure

```text
multimodal-deepfake_-etector/
│
├── backend/
│   ├── api/
│   │   └── main.py
│   │
│   ├── utils/
│   │   └── extraction.py
│   │
│   ├── config.py
│   ├── inference_pipeline.py
│   ├── realtime.py
│   └── worker.py
│
├── data/
│   └── preprocess.py
│
├── frontend/
│   ├── src/
│   │   ├── App.css
│   │   ├── App.jsx
│   │   └── main.jsx
│   │
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── models/
│   ├── audio/
│   ├── fusion/
│   ├── text/
│   ├── video/
│   └── explainability.py
│
├── notebooks/
│
├── training/
│   ├── evaluate.py
│   ├── train_audio.py
│   └── train_video.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
└── README.md
```

## Requirements

- Python 3.11+
- Node.js 18+
- FFmpeg installed and available from the terminal
- Optional: CUDA-capable GPU for faster inference/training

## Backend Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run The Backend

```bash
uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

Useful URLs:

- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend usually runs at:

- [http://127.0.0.1:5173](http://127.0.0.1:5173)

By default, Vite proxies `/api` requests to the backend. To use a direct backend URL, set:

```bash
VITE_API_URL=http://127.0.0.1:8000
```

## API Endpoints

The FastAPI app is defined in `backend/api/main.py`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Check backend health. |
| `POST` | `/api/upload` | Upload a video and receive a job id. |
| `POST` | `/api/analyze/{job_id}` | Start analysis for an uploaded video. |
| `GET` | `/api/result/{job_id}` | Poll analysis status/result. |
| `POST` | `/api/analyze_sync` | Upload and analyze in one request. |
| `GET` | `/api/jobs` | List known jobs. |
| `DELETE` | `/api/job/{job_id}` | Delete a job. |

Typical result fields:

- `label`
- `fake_probability`
- `confidence`
- `video_score`
- `audio_score`
- `text_score`
- `frame_analysis`
- `modality_details`
- `explanation`
- `suspicious_frames`
- `audio_plot`
- `breakdown_plot`
- `processing_time_s`

## Decision Logic

The current video verdict is primarily driven by frame-level evidence.

Each analysed frame receives a risk score. A frame is counted as fake when its score is greater than or equal to:

```python
FRAME_FAKE_THRESHOLD = 0.60
```

The frame ratio verdict is:

```python
if fake_ratio >= 0.25:
    verdict = "FAKE"
elif fake_ratio >= 0.15:
    verdict = "SUSPICIOUS"
else:
    verdict = "REAL"
```

The final displayed decision score is calculated from the frame component and available modalities:

```python
frame_component = fake_ratio * 0.5 + avg_frame_risk * 0.5
```

When audio is available:

```python
final_fake_score = (
    frame_component * 0.70 +
    audio_score * 0.15 +
    text_score * 0.15
)
```

When audio is not available:

```python
final_fake_score = frame_component
```

The UI labels this value as the decision score. The label itself comes from the frame-ratio verdict bands above, so a video can be labelled `FAKE` even if the final weighted score is below 50%, provided the fake-frame ratio crosses the configured `25%` threshold.

## Models Used

The system uses separate models for video, audio, and optional text/lip-sync evidence. Each model produces a fake-probability score, and the final decision combines the available signals.

### Video Model

- **Model type:** EfficientNet-B4 feature extractor with temporal attention and a binary classification head.
- **Input:** face crops extracted from video frames.
- **Why used:** visual deepfakes often leave artifacts around the face, skin texture, lighting, boundaries, expressions, or frame-to-frame consistency.
- **How used:** frames are extracted from the uploaded video, faces are detected with MTCNN, each face crop is scored, and the fake-frame ratio is calculated. This ratio is the main source of the final `REAL`, `SUSPICIOUS`, or `FAKE` verdict.

### Audio Model

- **Model type:** CNN + BiLSTM + attention model over MFCC and log-mel spectrogram features.
- **Input:** audio waveform extracted from the uploaded video.
- **Why used:** voice cloning, text-to-speech, or audio manipulation can create unnatural frequency patterns, prosody, or temporal artifacts.
- **How used:** FFmpeg extracts audio, the backend computes MFCC and mel-spectrogram features, and the audio model returns an audio fake score. If usable audio is available, this score contributes `15%` to the final decision score.

### Text And Lip-Sync Model

- **Model type:** lightweight transcript heuristics with optional transformer-based text scoring and lip-sync consistency checks.
- **Input:** transcript segments and timing information generated from the video audio.
- **Why used:** some manipulated videos have mismatches between speech, transcript content, and visible mouth movement.
- **How used:** when audio/transcript data is available, the text branch checks transcript quality and sync cues. Its score contributes `15%` to the final decision score. If text is unavailable but audio exists, the backend uses a neutral fallback score.

### Fusion / Final Scoring

- **Primary verdict:** determined by the fake-frame ratio.
- **Decision score:** calculated from video frame evidence and, when available, audio/text scores.
- **Reason for this design:** frame-level evidence is the most direct signal for visual deepfake detection in this project, while audio and text provide supporting evidence instead of overriding the frame-ratio verdict.

## Training

Train the video model:

```bash
python -m training.train_video ^
  --data_dir data/processed/video ^
  --save_dir data/models ^
  --backbone efficientnet_b4 ^
  --epochs 30 ^
  --batch_size 32 ^
  --lr 1e-4 ^
  --device cuda
```

Train the audio model:

```bash
python -m training.train_audio ^
  --data_dir data/processed/audio ^
  --save_dir data/models ^
  --epochs 40 ^
  --batch_size 64 ^
  --device cuda
```

Evaluate saved models:

```bash
python -m training.evaluate ^
  --data_dir data/processed/video/test ^
  --video_model data/models/video_model.pth ^
  --audio_model data/models/audio_model.pth ^
  --output_dir eval_results
```

For macOS/Linux, replace `^` line continuations with `\`.

## Known Limitations And Failure Analysis

The current project can show very high accuracy on the available test data, but that does not always mean the detector will generalize to real-world deepfake videos. A result such as "fake videos are detected as 100% fake" and "real videos are detected as 100% real" should be treated as a warning sign and verified carefully.

Main reasons this can happen:

- **Frame-level data leakage:** the video model is trained on extracted face crops. If crops from the same source video are randomly split across train and validation sets, the model may see almost identical faces, lighting, compression, and backgrounds during both training and testing.
- **Audio leakage or source bias:** audio files from the same speaker, generator, codec, or source distribution can appear in both training and validation. This lets the model learn dataset fingerprints instead of deepfake-specific evidence.
- **Dataset is too small or too clean:** if the test videos come from the same dataset, same manipulation method, or same preprocessing pipeline as training, the task becomes much easier than real-world detection.
- **Synthetic/dummy data is not representative:** generated demo samples are useful for checking that the pipeline works, but they often contain obvious visual or audio artifacts that real deepfakes do not have.
- **Fixed thresholding can overstate confidence:** the evaluator uses a default fake threshold of `0.5`. This may work on one test set but fail on new datasets unless the threshold is calibrated.
- **Fusion can amplify one strong modality:** the final multimodal score may become very confident if one modality gives a high fake score, even when other modalities are weak, missing, or contradictory.
- **Real uploads differ from training data:** uploaded videos may have lower resolution, heavy compression, poor lighting, no clear face, no audio, background music, edits, subtitles, screen recording artifacts, or deepfake methods unseen during training.

Because of these issues, high internal accuracy should not be presented as proof that the system reliably detects all fake videos. It mainly shows performance on the current dataset and preprocessing setup.

## Suggested Improvements

- **Split by source video, identity, and generator:** keep all frames/crops from the same original video in only one split. For stronger validation, keep identities and manipulation methods separate between train and test.
- **Use an external holdout test set:** train on one dataset and test on another, such as training on FakeAVCeleb and testing on Celeb-DF, DFDC, FaceForensics++, or new manually collected videos.
- **Report more than accuracy:** include precision, recall, F1-score, ROC-AUC, confusion matrix, false positives, false negatives, and per-class metrics. For safety, false negatives on fake videos should be reviewed separately.
- **Calibrate the decision threshold:** choose the fake threshold from a validation ROC/PR curve, then freeze it before final testing. Add an `UNCERTAIN` band for borderline scores.
- **Evaluate each modality separately:** report video-only, audio-only, text-only, and fused performance. This shows whether one modality is dominating the final verdict.
- **Add difficult real-world augmentations:** train with compression, blur, resizing, frame drops, lighting changes, rotation, background noise, music, silence, re-encoded social media videos, and partial face occlusion.
- **Use diverse manipulation methods:** include multiple face-swap, lip-sync, reenactment, voice-cloning, and text-to-speech generation methods so the model does not memorize one generator.
- **Improve fusion logic:** reduce hard confidence boosting from a single modality and handle disagreement between video, audio, and text as lower confidence instead of a forced final label.
- **Keep a manual benchmark folder:** maintain a small set of known real and fake videos from outside the training data and rerun it after every model or threshold change.
- **Document dataset boundaries:** clearly state which datasets, manipulation methods, identities, and preprocessing steps were used so reported metrics are interpreted correctly.

## Model Artifacts

Large model checkpoints are not GitHub-friendly. Do not commit files such as:

- `*.pth`
- `*.pt`
- `*.onnx`
- local datasets under `data/raw/` or `data/processed/`

For a public repo, upload checkpoints to a release, cloud drive, or model registry and add the download link here.

## Docker

Docker support files are included:

- `Dockerfile`
- `docker-compose.yml`
- `nginx.conf`

Local FastAPI + Vite development is recommended while iterating.

## Notes

- If a video has no usable audio, the pipeline can skip audio/transcript analysis instead of failing the whole request.
- The text branch is optional and should fail gracefully when external model downloads are unavailable.

## Dataset

This project uses the **FakeAVCeleb** dataset for training and evaluation of multimodal deepfake detection models.

Dataset Source:
https://github.com/DASH-Lab/FakeAVCeleb
