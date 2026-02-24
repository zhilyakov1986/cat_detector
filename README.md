# Cat Detector

A simple image classification app using Fastai and Gradio that tells you whether a picture contains a Cat or a Dog.

## Installation & Setup
This project uses `uv` for dependency management. 

1. Clone the repository:
   ```bash
   git clone https://github.com/zhilyakov1986/cat_detector.git
   cd cat_detector
   ```

2. Make sure you have `uv` installed, then run the installer:
   ```bash
   uv sync
   ```

## Generating the Model
To keep the repository small and lightweight, the 80MB+ Trained Model (`model.pkl`) is **NOT** included in source control.

Instead, the model is generated *on the fly* the first time you run the training script.

1. Run the training script:
   ```bash
   uv run train.py
   ```
   **What this does:**
   - Downloads the 800MB Oxford Pets Dataset.
   - Downloads a pre-trained ResNet34 model.
   - Fine-tunes the model to distinguish between Cats and Dogs.
   - Saves the final working model into the hidden `~/.fastai/data/oxford-iiit-pet/images/model.pkl` directory on your computer.

## Running the Web UI
Once `train.py` has finished successfully, you can host the interactive web UI:

```bash
uv run app.py
```
This will start a local Gradio web application. Open the provided `127.0.0.1` link in your browser to upload pictures and test the model!
