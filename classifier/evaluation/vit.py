import numpy as np
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

MODEL_NAME = "google/vit-base-patch16-224"
DATA_DIR = "dataset"

def main():
    # 1) Load local folder dataset
    ds_dict = load_dataset("imagefolder", data_dir=DATA_DIR)
    ds_all = ds_dict["train"]  # Dataset with columns: image, label

    # 2) Stratified split
    split = ds_all.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    train_ds, test_ds = split["train"], split["test"]

    print("Train columns:", train_ds.column_names)
    print("Label names:", train_ds.features["label"].names)
    print("Train size:", len(train_ds), "Test size:", len(test_ds))

    id2label = {i: name for i, name in enumerate(train_ds.features["label"].names)}
    label2id = {v: k for k, v in id2label.items()}

    # 3) Model + processor
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # 4) Transform: convert PIL -> model inputs
    def transform(examples):
        images = [im.convert("RGB") for im in examples["image"]]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs

    # Use set_transform so the dataloader always sees pixel_values/labels
    train_ds.set_transform(transform)
    test_ds.set_transform(transform)

    metric_acc = evaluate.load("accuracy")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return metric_acc.compute(predictions=preds, references=p.label_ids)

    # 5) Training config
    args = TrainingArguments(
        output_dir="vit_finetuned",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        eval_strategy="epoch",        # <-- for your transformers version
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,  # IMPORTANT with set_transform
        fp16=False,
        report_to=[],                 # disables wandb/etc.
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=processor,   # replaces deprecated tokenizer=
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())

if __name__ == "__main__":
    main()
