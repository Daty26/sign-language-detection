from mediapipe_model_maker import gesture_recognizer

DATASET_DIR = "dataset"  # folder with A/, B/, C/

# Load dataset (expects subfolders per label)
data = gesture_recognizer.Dataset.from_folder(
    dirname=DATASET_DIR,
    hparams=gesture_recognizer.HandDataPreprocessingParams(
        min_detection_confidence=0.3  # try 0.2–0.5 depending on quality
    )
)

train_data, rest = data.split(0.8)
val_data, test_data = rest.split(0.5)

hparams = gesture_recognizer.HParams(export_dir="mp_exported_model", epochs=20)
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)

model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=val_data,
    options=options
)

loss, acc = model.evaluate(test_data, batch_size=1)
print("Test loss:", loss, "Test acc:", acc)

model.export_model()  # writes mp_exported_model/gesture_recognizer.task
