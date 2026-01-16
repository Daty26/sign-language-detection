# classifier/evaluation/vit.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
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
class VitConfig:
    model_name: str = "google/vit-base-patch16-224"
    split_csv: str = "splits/split_seed42.csv"
    seed: int = 42
    fold: Optional[int] = None


    fine_tune: bool = True

    # training
    learning_rate: float = 2e-5
    train_bs: int = 16
    eval_bs: int = 16
    num_epochs: int = 5

    # output dirs
    out_dir: str = "runs/vit_finetuned"
    out_dir_eval: str = "runs/vit_eval"

    # misc
    fp16: bool = False
    logging_steps: int = 50
    use_fast_processor: bool = True


class ImagePathDataset(Dataset):
    """
    Dataset that reads images from paths and produces:
      - pixel_values: Tensor [3,224,224] (or whatever processor returns)
      - labels: int
    """
    def __init__(self, df, label2id, processor):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # remove batch dim
        inputs["labels"] = int(self.label2id[row["label"]])
        return inputs


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_metrics(p) -> Dict[str, float]:
    preds = np.argmax(p.predictions, axis=1)
    acc = float((preds == p.label_ids).mean())
    return {"accuracy": acc}


def run_vit(cfg: VitConfig) -> Dict[str, Any]:
    """
    Callable entry point for benchmark runners.

    Returns a dict:
      {
        "model": "vit",
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
    print("[vit] Device:", device)
    print("[vit] Split:", cfg.split_csv)

    train_df, test_df, class_names = load_split_csv(cfg.split_csv, cfg.fold)
    print("[vit] Classes:", class_names)
    print("[vit] Train:", len(train_df), "Test:", len(test_df))

    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {v: k for k, v in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(
        cfg.model_name,
        use_fast=cfg.use_fast_processor,
    )

    model = AutoModelForImageClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_ds = ImagePathDataset(train_df, label2id, processor)
    test_ds = ImagePathDataset(test_df, label2id, processor)

    data_collator = DefaultDataCollator()

    if not cfg.fine_tune:
        print("[vit] Evaluating without fine-tune (expected poor).")
        preds, refs = predict(model, test_ds, data_collator, cfg.out_dir_eval, cfg.eval_bs)
        acc = float((np.array(preds) == np.array(refs)).mean())
        print_report(refs, preds, class_names)
        return {
            "model": "vit",
            "accuracy": acc,
            "num_train": len(train_df),
            "num_test": len(test_df),
            "class_names": class_names,
            "preds": preds,
            "refs": refs,
        }

    args = TrainingArguments(
        output_dir=cfg.out_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_bs,
        per_device_eval_batch_size=cfg.eval_bs,
        num_train_epochs=cfg.num_epochs,
        eval_strategy="epoch",          # your transformers build uses eval_strategy
        save_strategy="epoch",
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,    # IMPORTANT for custom dataset dict output
        fp16=cfg.fp16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    out = trainer.predict(test_ds)
    preds = np.argmax(out.predictions, axis=1).tolist()
    refs = out.label_ids.tolist()
    acc = float((np.array(preds) == np.array(refs)).mean())

    print_report(refs, preds, class_names)

    return {
        "model": "vit",
        "accuracy": acc,
        "num_train": len(train_df),
        "num_test": len(test_df),
        "class_names": class_names,
        "preds": preds,
        "refs": refs,
    }


def predict(model, ds, data_collator, out_dir: str, eval_bs: int) -> tuple[List[int], List[int]]:
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=out_dir,
            per_device_eval_batch_size=eval_bs,
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
    cfg = VitConfig()
    run_vit(cfg)


if __name__ == "__main__":
    main()
