# Whisper Emotion LoRA

Parameter-efficient speech emotion recognition using OpenAI Whisper and LoRA fine-tuning.

This project fine-tunes the Whisper encoder for audio emotion classification on the RAVDESS dataset. Instead of full model fine-tuning, it compares several lightweight LoRA configurations to understand how much performance can be gained by training only a small percentage of model parameters.

## Task

Classify speech audio into 8 emotions: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`.

## Approach
 
```
Audio → Log-Mel → Whisper-small Encoder + LoRA → Mean Pool → MLP → 8 emotions
```
 
- **`openai/whisper-small`** as the audio encoder (frozen)
- **LoRA adapters** injected into selected projections
- **MLP classification head** on the mean-pooled encoder output
- **Speaker-disjoint split**: 4 actors held out entirely from training
- **Class-weighted cross-entropy** with label smoothing, early stopping on weighted F1


## Setup
 
```bash
pip install -r requirements.txt
```
 
## Usage
 
```bash
python train.py                       # full ablation - downloads RAVDESS (~215 MB) automatically
python train.py --smoke               # 1-epoch sanity check on a small subset
python train.py --only lora_q_v_mlp   # run a single configuration
```


## Experiments

Four fine-tuning strategies were compared:
 
| Method | Trainable | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Frozen baseline (head only) | 0.23% | 80.0% | 0.794 | 0.790 |
| LoRA Q+V | 0.89% | 82.1% | 0.811 | 0.818 |
| LoRA Q+K+V+O (full attention) | 1.54% | 82.1% | 0.815 | 0.817 |
| **LoRA Q+V + MLP** | **2.50%** | **90.8%** | **0.909** | **0.907** |

## Results

The best model was LoRA Q+V+MLP, achieving 90.8% accuracy and 0.907 weighted F1 on unseen speakers while training only 2.5% of the model parameters - a 10.8-point accuracy improvement over the frozen-encoder baseline (80.0%).

## Tech Stack

Python, PyTorch, Hugging Face Transformers, PEFT/LoRA, Datasets, Scikit-learn

## References

**RAVDESS dataset** — Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391. Dataset available at Zenodo: https://zenodo.org/record/1188976

