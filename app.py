import gradio as gr
from fastai.vision.all import *

# Fastai needs the same labeling function defined when loading the model
def is_cat(x): 
    return x[0].isupper()

# Load the trained model exported from train.py
# By default, fastai's Learner.export() saves to the Learner's path (which is the dataset images folder)
import os
model_path = os.path.expanduser('~/.fastai/data/oxford-iiit-pet/images/model.pkl')

try:
    learn = load_learner(model_path)
except FileNotFoundError:
    print(f"Error: model not found at {model_path}")
    print("Please run `uv run train.py` first to train and export the model.")
    exit(1)

# Prediction labels
categories = ('Dog', 'Cat')

def classify_image(img):
    # predict returns: prediction class, tensor index, tensor of probabilities
    pred, idx, probs = learn.predict(img)
    # create a dictionary of all categories mapping to their probability
    return dict(zip(categories, map(float, probs)))

# Create the Gradio Interface
image_input = gr.Image(type="pil", label="Upload a picture of a Pet")
label_output = gr.Label(label="Prediction:")

# Launch a simple UI
demo = gr.Interface(
    fn=classify_image,
    inputs=image_input,
    outputs=label_output,
    title="Cat vs Dog Image Classifier",
    description="Upload an image to see whether our Fastai ResNet34 model thinks it's a cat or a dog!"
)

if __name__ == "__main__":
    demo.launch()
