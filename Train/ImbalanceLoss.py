from ToTune.Train.CustomTrainer import CustomTrainer
from transformers import Trainer,TrainingArguments


class LIWtrainer(CustomTrainer):
    def __init__(self, *args, class_balanced_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_balanced_loss = class_balanced_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # ---- Cast logits to FP32 (QLoRA safety) ----
        logits_for_loss = logits.float()

        if self.class_balanced_loss is not None:
            loss = self.class_balanced_loss(logits_for_loss, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits_for_loss, labels)

        return (loss, outputs) if return_outputs else loss

from types import MethodType
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
def enable_resample_for_trainer(trainer: "LIWtrainer"):
    """
    Locally override get_train_dataloader for ONE trainer instance only.
    """

    def get_train_dataloader_override(self):
        train_dataset = self.train_dataset
        labels = np.array(train_dataset["labels"])

        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    trainer.get_train_dataloader = MethodType(
        get_train_dataloader_override, trainer
    )

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedClassificationLoss(nn.Module):
    """
    Supports:
    1. Standard CrossEntropy
    2. Cost-Sensitive CrossEntropy (manual class_weights)
    3. Class-Balanced Loss (samples_per_class + beta)
    4. Focal Loss (gamma)
    5. CB + Focal
    6. Cost-Sensitive + Focal

    Priority:
    - If class_weights is provided → use it
    - Else if samples_per_class is provided → compute CB weights
    - Else → no weighting
    """

    def __init__(
        self,
        class_weights=None,
        samples_per_class=None,
        beta=0.9999,
        gamma=0.0,
    ):
        super().__init__()
        self.gamma = gamma

        # ---- 1. Manual cost-sensitive weights ----
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)

        # ---- 2. Class-balanced weights ----
        elif samples_per_class is not None:
            samples = torch.tensor(samples_per_class, dtype=torch.float32)
            weights = (1 - beta) / (1 - beta ** samples)
            weights = weights / weights.sum() * len(samples)

        # ---- 3. No weighting ----
        else:
            weights = None

        if weights is not None:
            self.register_buffer("weights", weights)
        else:
            self.weights = None

    def forward(self, logits, targets):
        # ---- 1. AMP-safe logits ----
        logits = logits.float()

        # ---- 2. Move weights ----
        weights = self.weights.to(logits.device) if self.weights is not None else None

        # ---- 3. CrossEntropy per-sample ----
        ce = F.cross_entropy(
            logits,
            targets,
            weight=weights,
            reduction="none"
        )

        # ---- 4. Optional focal modulation ----
        if self.gamma > 0:
            p = torch.exp(-ce)
            ce = (1 - p) ** self.gamma * ce

        return ce.mean()


def _build_li_trainer(*args, loss_fn=None, **kwargs):
    resample = kwargs.pop("resample", False)  # default = False

    trainer = LIWtrainer(
        *args,
        class_balanced_loss=loss_fn,
        **kwargs
    )

    if resample:
        enable_resample_for_trainer(trainer)

    return trainer


from sklearn.utils.class_weight import compute_class_weight
import numpy as np
def _get_labels(kwargs):
    train_dataset = kwargs["train_dataset"]
    return np.array(train_dataset["labels"])
def _get_cost_weights(labels):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(weights, dtype=torch.float32)
def _get_samples_per_class(labels):
    return np.bincount(labels)




def set_CS(*args, **kwargs):
    labels = _get_labels(kwargs)
    loss_fn = UnifiedClassificationLoss(
        class_weights=_get_cost_weights(labels)
    )
    return _build_li_trainer(*args, loss_fn=loss_fn, **kwargs)
def set_Focal(*args, gamma=2.0, **kwargs):
    loss_fn = UnifiedClassificationLoss(gamma=gamma)
    return _build_li_trainer(*args, loss_fn=loss_fn, **kwargs)
def set_CB(*args, beta=0.9999, **kwargs):
    labels = _get_labels(kwargs)
    loss_fn = UnifiedClassificationLoss(
        samples_per_class=_get_samples_per_class(labels),
        beta=beta
    )
    return _build_li_trainer(*args, loss_fn=loss_fn, **kwargs)
def set_CS_Focal(*args, gamma=2.0, **kwargs):
    labels = _get_labels(kwargs)
    loss_fn = UnifiedClassificationLoss(
        class_weights=_get_cost_weights(labels),
        gamma=gamma
    )
    return _build_li_trainer(*args, loss_fn=loss_fn, **kwargs)
def set_CB_Focal(*args, beta=0.9999, gamma=2.0, **kwargs):
    labels = _get_labels(kwargs)
    loss_fn = UnifiedClassificationLoss(
        samples_per_class=_get_samples_per_class(labels),
        beta=beta,
        gamma=gamma
    )
    return _build_li_trainer(*args, loss_fn=loss_fn, **kwargs)
def set_CB_CS(*args, beta=0.9999, **kwargs):
    labels = _get_labels(kwargs)
    loss_fn = UnifiedClassificationLoss(
        class_weights=_get_cost_weights(labels),
        samples_per_class=_get_samples_per_class(labels),
        beta=beta
    )
    return _build_li_trainer(*args, loss_fn=loss_fn, **kwargs)
def set_CB_CS_Focal(*args, beta=0.9999, gamma=2.0, **kwargs):
    labels = _get_labels(kwargs)
    loss_fn = UnifiedClassificationLoss(
        class_weights=_get_cost_weights(labels),
        samples_per_class=_get_samples_per_class(labels),
        beta=beta,
        gamma=gamma
    )
    return _build_li_trainer(*args, loss_fn=loss_fn, **kwargs)
def set_CS_Focal_NoCB(*args, gamma=2.0, **kwargs):
    labels = _get_labels(kwargs)
    loss_fn = UnifiedClassificationLoss(
        class_weights=_get_cost_weights(labels),
        gamma=gamma
    )
    return _build_li_trainer(*args, loss_fn=loss_fn, **kwargs)
