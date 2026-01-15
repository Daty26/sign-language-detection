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


BACKBONE_MODEL = "microsoft/resnet-50"
SPLIT_CSV = "splits/split_seed42.csv"

# Training settings
FINE_TUNE = True
SEED = 42


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
        # remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = self.label2id[row["label"]]
        return inputs


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = (preds == p.label_ids).mean().item() if hasattr((preds == p.label_ids), "mean") else float((preds == p.label_ids).mean())
    return {"accuracy": acc}


def freeze_backbone_resnet(model, freeze: bool):
    # For transformers ResNetForImageClassification, backbone is model.resnet
    for p in model.resnet.parameters():
        p.requires_grad = not freeze


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_df, test_df, class_names = load_split_csv(SPLIT_CSV)
    print("Classes:", class_names)
    print("Train:", len(train_df), "Test:", len(test_df))

    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {v: k for k, v in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(BACKBONE_MODEL)

    model = AutoModelForImageClassification.from_pretrained(
        BACKBONE_MODEL,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # expected (1000 -> 26)
    ).to(device)

    # Augmentations: OK for ASL letters, but consider disabling flips (can change meaning)
    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    ])

    train_ds = ImagePathDataset(train_df, label2id, processor, augment=train_aug)
    test_ds = ImagePathDataset(test_df, label2id, processor, augment=None)

    data_collator = DefaultDataCollator()

    if not FINE_TUNE:
        print("\n=== Evaluating without fine-tune (not meaningful) ===")
        preds, refs = predict(model, test_ds, data_collator)
        print_report(refs, preds, class_names)
        return

    # ---- Stage 1: train head only ----
    freeze_backbone_resnet(model, freeze=True)

    args_head = TrainingArguments(
        output_dir="runs/resnet50_head",
        learning_rate=1e-3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer_head = Trainer(
        model=model,
        args=args_head,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer_head.train()

    # ---- Stage 2: fine-tune full network ----
    freeze_backbone_resnet(model, freeze=False)

    args_ft = TrainingArguments(
        output_dir="runs/resnet50_finetuned",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        report_to=[],
        fp16=False,
    )

    trainer_ft = Trainer(
        model=model,
        args=args_ft,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer_ft.train()

    out = trainer_ft.predict(test_ds)
    preds = np.argmax(out.predictions, axis=1).tolist()
    refs = out.label_ids.tolist()
    print_report(refs, preds, class_names)


def predict(model, ds, data_collator):
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="runs/tmp_eval",
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


if __name__ == "__main__":
    main()
