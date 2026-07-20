import zipfile
from pathlib import Path

import numpy as np
from datasets import Audio, Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import WhisperFeatureExtractor

EMOTIONS = ["neutral", "calm", "happy", "sad",
            "angry", "fearful", "disgust", "surprised"]

RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
TEST_ACTORS = {2, 5, 14, 21}    # speaker-disjoint test split


def download_ravdess(data_dir: Path) -> Path: #download and extract RAVDESS
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    zip_path = data_dir / "ravdess.zip"
    extract_dir = data_dir / "audio"

    if extract_dir.exists():
        return extract_dir

    if not zip_path.exists():
        print(f"Downloading RAVDESS (~215 MB) → {zip_path}")
        import urllib.request
        urllib.request.urlretrieve(RAVDESS_URL, zip_path)

    print("Extracting …")
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def parse_filename(path: Path) -> dict: # RAVDESS filename: MM-CC-EE-II-SS-RR-AA.wav (EE=emotion, AA=actor)
    parts = path.stem.split("-")
    return {
        "path": str(path),
        "label": int(parts[2]) - 1,
        "actor": int(parts[6]),
        "intensity": int(parts[3]),
    }


def build_dataset(audio_dir: Path) -> Dataset: #build a HF Dataset from RAVDESS audio files
    wav_files = list(Path(audio_dir).rglob("*.wav"))
    assert len(wav_files) == 1440, f"Expected 1440 files, got {len(wav_files)}"

    records = [parse_filename(p) for p in wav_files]
    ds = Dataset.from_list(records)
    ds = ds.cast_column("path", Audio(sampling_rate=16_000))
    return ds.rename_column("path", "audio")


def speaker_disjoint_split(ds: Dataset, test_actors=TEST_ACTORS):
    train_ds = ds.filter(lambda x: x["actor"] not in test_actors)
    eval_ds  = ds.filter(lambda x: x["actor"]     in test_actors)
    return train_ds, eval_ds


def preprocess_audio(ds: Dataset, feature_extractor: WhisperFeatureExtractor) -> Dataset: # convert audio: log-mel spectrograms (3000 frames, Whisper requirement)
    def _extract(batch):
        inputs = feature_extractor(
            [a["array"] for a in batch["audio"]],
            sampling_rate=16_000, return_tensors="np",
            return_attention_mask=True,   # marks real speech vs 30s padding
        )
        batch["input_features"] = inputs.input_features
        batch["attention_mask"] = inputs.attention_mask
        return batch

    drop_cols = [c for c in ["audio", "actor", "intensity"] if c in ds.column_names]
    if "input_features" not in ds.column_names:
        ds = ds.map(_extract, batched=True, batch_size=16, remove_columns=drop_cols)
    return ds


def get_class_weights(train_ds: Dataset) -> np.ndarray:
    return compute_class_weight(
        "balanced",
        classes=np.arange(len(EMOTIONS)),
        y=np.array(train_ds["label"]),
    )
