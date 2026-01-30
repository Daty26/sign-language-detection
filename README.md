# Sign Language Detection

A computer vision–based sign language detection system that recognizes hand gestures from a live camera feed using MediaPipe landmarks and machine learning classifiers.

## 📌 Project Overview

This project detects and classifies hand gestures (sign language) in real time. It uses:

- MediaPipe for hand landmark detection  
- Feature extraction based on hand geometry and temporal motion  
- Machine Learning models (rule-based + ML classifier)  
- Live camera streaming for real-time inference  

The system can be trained on custom datasets and evaluated using built-in scripts.

## 🗂️ Project Structure

```
sign_language_detection/
│
├── app.py                     # Main application entry point
├── requirements.txt           # Python dependencies
│
├── camera/                    # Camera stream handling
│   ├── camera_stream.py
│   └── camera_stream_test.py
│
├── landmarks/                 # MediaPipe hand landmark utilities
│   ├── hand_landmarks.py
│   └── mediapipe_wrapper.py
│
├── features/                  # Feature extraction logic
│   ├── feature_extractor.py
│   ├── geometry_utils.py
│   └── temporal_features.py
│
├── classifier/                # Gesture classification
│   ├── gesture_classifier.py
│   ├── ml_classifier.py
│   ├── rule_based.py
│   ├── model.pkl              # Trained ML model
│   ├── evaluation/            # Model evaluation scripts
│   └── training/              # Data collection & training scripts
│
├── dataset/                   # Training dataset
│   └── dataset.csv
│
├── ui/                        # UI components
│   ├── ui_main.py
│   └── ui_components.py
│
└── tests/                     # Automated tests
    ├── test_flow.py           # Full pipeline integration test
    └── test_performance.py    # Camera FPS / performance test

```

## ⚙️ Installation  

### 🐍 Python Version Requirement

This project requires **Python 3.10**.

Check your version:
```bash
python --version
```

### 1. Clone the repository
```bash
git clone https://gitlab.hof-university.de/Daty/sign_language_detection.git
cd sign_language_detection
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Running the Application

To start real-time sign language detection:
```bash
python app.py
```

Make sure your webcam is connected and accessible.

## 🧠 Training the Model

### Collect Training Data
```bash
python classifier/training/collect_data.py
```

### Prepare Dataset
```bash
python classifier/training/prepare_dataset.py
```

### Train the Model
```bash
python classifier/training/train_model.py
```

The trained model will be saved as `model.pkl`.

## 📊 Model Evaluation

```bash
python classifier/evaluation/evaluate_model.py
```

This will output accuracy and performance metrics.

## 🧪 Testing

The project includes the following tests:

- `test_flow.py` – integration test of the complete pipeline  
  (camera → landmarks → features → classifier)

- `test_performance.py` – performance test  
  measures camera FPS and real-time processing stability

To run the tests:
```bash
python tests/test_flow.py
python tests/test_performance.py


## 🚀 Future Improvements

- Support for dynamic sign sequences (full words/sentences)  
- Deep learning–based classifier (LSTM / CNN)  
- Multi-hand and multi-user detection  
- Improved UI and visualization  


## 📜 License

This project is licensed under the **MIT License**.

## 👤 Authors

- Temirlan Altayev  
- Alexandra Daradur  
- Meirlan Kudreyev  
- Kiara Minbaeva  
- Daniil Nikolayev  
- Adilkhan Yerbolat  
