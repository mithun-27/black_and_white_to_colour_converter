import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from model import build_transfer_unet
from utils import preprocess_image, postprocess_image
from tensorflow.keras.models import load_model

def colorize(image_path, model_path, output_path=None):
    """
    Takes an image, predicts its colors using the model, and displays/saves the result.
    """
    if not os.path.exists(image_path):
        print(f"Image {image_path} does not exist.")
        return

    # Load model
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, compile=False)
    else:
        print(f"Model {model_path} not found. Using an untrained model for demonstration.")
        model = build_transfer_unet()

    # Preprocess
    L, _ = preprocess_image(image_path)
    if L is None:
        return

    # Predict AB channels
    L_input = np.expand_dims(L, axis=0) # Add batch dimension (1, 256, 256, 1)
    predicted_AB = model.predict(L_input)
    
    # Postprocess
    colored_img = postprocess_image(L, predicted_AB[0])

    # Plot original and colored
    original_img = plt.imread(image_path)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original (B&W Input)")
    plt.imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("AI Predicted Colorization")
    plt.imshow(colored_img)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Colorized image saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Colorization Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to grayscale/test image")
    parser.add_argument("--model", type=str, default="colorization_model.keras", help="Path to the trained model (.keras)")
    parser.add_argument("--output", type=str, default="output_colorized.png", help="Path to save colorized result")

    args = parser.parse_args()
    colorize(args.image, args.model, args.output)
