import numpy as np
import torch
from torchvision import transforms
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)


BACKBONE_MODEL = "microsoft/resnet-50"
DATA_DIR = "dataset"
SEED = 42
TEST_SIZE = 0.2
FINE_TUNE = True

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds_all = load_dataset("imagefolder", data_dir=DATA_DIR)["train"]
    split = ds_all.train_test_split(test_size=TEST_SIZE, seed=SEED, stratify_by_column="label")
    train_ds, test_ds = split["train"], split["test"]

    class_names = train_ds.features["label"].names
    print("Classes:", class_names)
    print("Train:", len(train_ds), "Test:", len(test_ds))

    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {v: k for k, v in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(BACKBONE_MODEL)

    # Build model with YOUR label set (head will be trained on your data)
    model = AutoModelForImageClassification.from_pretrained(
        BACKBONE_MODEL,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    ])

    def train_transform(examples):
        images = [train_aug(im.convert("RGB")) for im in examples["image"]]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = torch.tensor(examples["label"])
        return inputs

    def test_transform(examples):
        images = [im.convert("RGB") for im in examples["image"]]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = torch.tensor(examples["label"])
        return inputs

    train_ds.set_transform(train_transform)
    test_ds.set_transform(test_transform)

    data_collator = DefaultDataCollator()

    if not FINE_TUNE:
        # This will evaluate a randomly-initialized head; expect poor results
        print("\n=== Evaluating without fine-tune (not meaningful) ===")
        preds, refs = predict(model, test_ds, data_collator)
        report(refs, preds, class_names)
        return
    
    # ---- Stage 1: train classifier head only ----
    for p in model.resnet.parameters():
        p.requires_grad = False

    args_head = TrainingArguments(
        output_dir="resnet50_head",
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
    for p in model.resnet.parameters():
        p.requires_grad = True

    args_ft = TrainingArguments(
        output_dir="resnet50_finetuned",
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
    report(refs, preds, class_names)


def predict(model, ds, data_collator):
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="tmp_eval",
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


def report(y_true, y_pred, class_names):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()
