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

## Experiments

Four fine-tuning strategies were compared:

| Method | Trainable | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Frozen baseline (head only) | 0.23% | 59.6% | 0.584 | 0.583 |
| LoRA Q+V | 0.89% | 77.9% | 0.778 | 0.776 |
| LoRA Q+K+V+O (full attention) | 1.54% | 80.4% | 0.797 | 0.804 |
| **LoRA Q+V + MLP** | **2.50%** | **86.7%** | **0.863** | **0.863** |

## Results

The best model was **LoRA Q+V+MLP**, achieving **86.7% accuracy** and **0.863 weighted F1** while training only **2.5%** of the model parameters.

Compared with the frozen baseline, LoRA fine-tuning improved accuracy from **59.6% to 86.7%**, showing that lightweight adaptation of Whisper can significantly improve speech emotion recognition performance.

## Key Finding

LoRA on attention layers already provides a strong improvement, but adding LoRA to the feed-forward layers gives the best performance on this task.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- Datasets
- Scikit-learn
- RAVDESS
- Whisper


