# 🛡️ Deepfake Detection Engine

A robust, dual-branch deep learning pipeline and interactive Streamlit dashboard for detecting AI-manipulated media.

This system moves beyond simple frame-by-frame analysis by combining **Spatiotemporal Tracking** (to catch unnatural physical motion) with **Frequency Domain Analysis** (to detect microscopic AI-generation artifacts).

## 🚀 Features

* **Advanced Face Extraction:** Uses MTCNN to isolate and crop facial sequences, ignoring irrelevant background noise.
* **Dual-Branch Architecture:**
  * **Branch 1 (Motion & Visuals):** A ResNeXt-50 CNN extracts spatial features, which are fed into a Bidirectional LSTM to track unnatural temporal inconsistencies (like stuttering motion or failed blending).
  * **Branch 2 (GAN Artifacts):** Applies a 2D Fast Fourier Transform (FFT) to convert image data into the frequency domain, exposing the hidden grid-like mathematical signatures left behind by GAN upsampling.
* **Interactive Dashboard:** A clean, mission-control style Streamlit frontend for drag-and-drop video analysis.

## 📁 Project Structure

```text
deepfake_engine/
├── app.py                # Streamlit web dashboard
├── config.py             # Hyperparameters and system settings
├── dataset.py            # PyTorch Dataset for video/face sequences
├── inference.py          # Standalone inference engine 
├── train.py              # Training loop with validation and checkpointing
├── requirements.txt      # Python dependencies
├── models/
│   ├── detector.py       # Dual-branch model architecture
│   └── layers.py         # Custom FFT frequency layer
└── utils/
    ├── face_extractor.py # OpenCV and MTCNN processing
    └── transforms.py     # Data augmentation (Albumentations)
```

## ⚙️ Installation

1. Clone or extract the project directory.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Note: If you have a CUDA-capable GPU, ensure your PyTorch installation is configured for CUDA to significantly speed up extraction and inference).*

## 💻 Usage

### 1. Running the Web Dashboard (Inference)
To start the Streamlit interface and test videos:

```bash
streamlit run app.py
```
Open your browser to the local URL provided (usually `http://localhost:8501`). Drag and drop `.mp4`, `.mov`, or `.avi` files to run the analysis.

### 2. Training the Model
To train the neural networks on your own dataset to accurately distinguish real from fake media:

1. Create a `data/` folder in the root directory.
2. Add your training videos to the respective subfolders:
   * `data/real/` (for authentic videos)
   * `data/fake/` (for manipulated videos)
3. Run the training script:

```bash
python train.py
```
The system will automatically split your data, train for the epochs defined in `config.py`, and save the best weights as `best_deepfake_detector.pth`.

## ⚠️ Limitations & Notes
* **Untrained Weights:** Out of the box, if no `best_deepfake_detector.pth` is provided, the system initializes with random weights and will not provide accurate predictions until trained.
* **Video Compression:** Heavy social media compression (like WhatsApp or Twitter) can destroy the high-frequency AI artifacts that the FFT branch relies on, which may reduce detection accuracy in real-world scenarios.