# classifier/evaluation/resnet50.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

from classifier.evaluation.load_split import load_split_csv
from classifier.evaluation.metrics import print_report


@dataclass
class ResNet50Config:
    backbone_model: str = "microsoft/resnet-50"
    split_csv: str = "splits/split_seed42.csv"
    seed: int = 42
    fold: Optional[int] = None

    # training
    fine_tune: bool = True
    head_epochs: int = 2
    ft_epochs: int = 10
    head_lr: float = 1e-3
    ft_lr: float = 5e-5
    train_bs: int = 16
    eval_bs: int = 32

    # output dirs
    out_dir_head: str = "runs/resnet50_head"
    out_dir_ft: str = "runs/resnet50_finetuned"
    out_dir_eval: str = "runs/tmp_eval"

    # misc
    fp16: bool = False
    logging_steps: int = 50


class ImagePathDataset(Dataset):
    """
    Dataset that reads images from paths and produces:
      - pixel_values: Tensor [3,224,224]
      - labels: int
    """
    def __init__(self, df, label2id, processor, augment=None):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")

        if self.augment is not None:
            img = self.augment(img)

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # remove batch dim
        inputs["labels"] = int(self.label2id[row["label"]])
        return inputs


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _freeze_backbone_resnet(model, freeze: bool) -> None:
    # For transformers ResNetForImageClassification, backbone is model.resnet
    for p in model.resnet.parameters():
        p.requires_grad = not freeze


def _compute_metrics(p) -> Dict[str, float]:
    preds = np.argmax(p.predictions, axis=1)
    acc = float((preds == p.label_ids).mean())
    return {"accuracy": acc}


def run_resnet50(cfg: ResNet50Config) -> Dict[str, Any]:
    """
    Callable entry point for benchmark runners.

    Returns a dict with key results:
      {
        "model": "resnet50",
        "accuracy": float,
        "num_train": int,
        "num_test": int,
        "class_names": [...],
        "preds": [...],
        "refs": [...],
      }
    """
    _set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[resnet50] Device:", device)
    print("[resnet50] Split:", cfg.split_csv)

    train_df, test_df, class_names = load_split_csv(cfg.split_csv, cfg.fold)
    print("[resnet50] Classes:", class_names)
    print("[resnet50] Train:", len(train_df), "Test:", len(test_df))

    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {v: k for k, v in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(cfg.backbone_model)

    model = AutoModelForImageClassification.from_pretrained(
        cfg.backbone_model,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    # augmentations (NO flips for ASL letters)
    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    ])

    train_ds = ImagePathDataset(train_df, label2id, processor, augment=train_aug)
    test_ds = ImagePathDataset(test_df, label2id, processor, augment=None)

    data_collator = DefaultDataCollator()

    if not cfg.fine_tune:
        print("[resnet50] Evaluating without fine-tune (expected poor).")
        preds, refs = predict(model, test_ds, data_collator, cfg.out_dir_eval)
        acc = float((np.array(preds) == np.array(refs)).mean())
        print_report(refs, preds, class_names)
        return {
            "model": "resnet50",
            "accuracy": acc,
            "num_train": len(train_df),
            "num_test": len(test_df),
            "class_names": class_names,
            "preds": preds,
            "refs": refs,
        }

    # ---- Stage 1: train head only ----
    _freeze_backbone_resnet(model, freeze=True)

    args_head = TrainingArguments(
        output_dir=cfg.out_dir_head,
        learning_rate=cfg.head_lr,
        per_device_train_batch_size=cfg.train_bs,
        per_device_eval_batch_size=cfg.eval_bs,
        num_train_epochs=cfg.head_epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=cfg.logging_steps,
        remove_unused_columns=False,
        report_to=[],
        fp16=cfg.fp16,
    )

    trainer_head = Trainer(
        model=model,
        args=args_head,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )
    trainer_head.train()

    # ---- Stage 2: fine-tune full network ----
    _freeze_backbone_resnet(model, freeze=False)

    args_ft = TrainingArguments(
        output_dir=cfg.out_dir_ft,
        learning_rate=cfg.ft_lr,
        per_device_train_batch_size=cfg.train_bs,
        per_device_eval_batch_size=cfg.eval_bs,
        num_train_epochs=cfg.ft_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        report_to=[],
        fp16=cfg.fp16,
    )

    trainer_ft = Trainer(
        model=model,
        args=args_ft,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )
    trainer_ft.train()

    out = trainer_ft.predict(test_ds)
    preds = np.argmax(out.predictions, axis=1).tolist()
    refs = out.label_ids.tolist()
    acc = float((np.array(preds) == np.array(refs)).mean())

    print_report(refs, preds, class_names)

    return {
        "model": "resnet50",
        "accuracy": acc,
        "num_train": len(train_df),
        "num_test": len(test_df),
        "class_names": class_names,
        "preds": preds,
        "refs": refs,
    }


def predict(model, ds, data_collator, out_dir: str) -> Tuple[List[int], List[int]]:
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=out_dir,
            per_device_eval_batch_size=32,
            remove_unused_columns=False,
            report_to=[],
        ),
        data_collator=data_collator,
    )
    out = trainer.predict(ds)
    preds = np.argmax(out.predictions, axis=1).tolist()
    refs = out.label_ids.tolist()
    return preds, refs


def main():
    # Minimal CLI (keeps old behavior)
    cfg = ResNet50Config()
    run_resnet50(cfg)


if __name__ == "__main__":
    main()
