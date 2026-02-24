from fastai.vision.all import *

# Define label function outside so it can be pickled/accessed easily by fastai DataLoaders
def is_cat(x): 
    return x[0].isupper()

def main():
    print("Downloading and extracting PETS dataset...")
    path = untar_data(URLs.PETS)/'images'

    print("Setting up DataLoaders...")
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=is_cat, item_tfms=Resize(224)
    )

    print("Initializing Vision Learner...")
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    
    print("Fine-tuning model...")
    learn.fine_tune(1)
    
    print("Exporting model to 'model.pkl'...")
    learn.export('model.pkl')
    
    print("Done!")

if __name__ == "__main__":
    main()
