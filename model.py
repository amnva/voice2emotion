import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import WhisperModel
from transformers.modeling_outputs import SequenceClassifierOutput

MODEL_ID = "openai/whisper-small"


class WhisperEmotionClassifier(nn.Module):
    def __init__(self, encoder, hidden, num_labels,
                 class_weights=None, dropout=0.3, label_smoothing=0.1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )
        weight = (torch.tensor(class_weights, dtype=torch.float32)
                  if class_weights is not None else None)
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, input_features, attention_mask=None, labels=None):
        target_dtype = next(self.encoder.parameters()).dtype
        h = self.encoder(input_features.to(target_dtype)).last_hidden_state
        h = h.float()                          # back to fp32 for the head

        if attention_mask is not None:
            mask = attention_mask[:, ::2][:, :h.shape[1]].unsqueeze(-1).float()
            h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            h = h.mean(dim=1)

        logits = self.head(h)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


def build_model(method: str, num_labels: int, target_modules=None,
                class_weights=None, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """
    method: 'frozen' or 'lora'
    target_modules: e.g. ['q_proj', 'v_proj', 'fc1', 'fc2'] (lora only)
    """
    base = WhisperModel.from_pretrained(MODEL_ID)
    encoder = base.encoder
    hidden = encoder.config.d_model

    if method == "frozen":
        for p in encoder.parameters():
            p.requires_grad = False
    elif method == "lora":
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout, bias="none",
        )
        encoder = get_peft_model(encoder, cfg)
        encoder.print_trainable_parameters()
    else:
        raise ValueError(f"Unknown method: {method}")

    model = WhisperEmotionClassifier(encoder, hidden, num_labels,
                                     class_weights=class_weights)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train/1e6:.2f}M / {n_total/1e6:.1f}M  "
          f"({100*n_train/n_total:.2f}%)")

    return model, n_train, n_total
