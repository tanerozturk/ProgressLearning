import torch
import torch.nn as nn
import requests
from PIL import Image
from torchvision import models, transforms
import os

# --- Configuration ---
MODEL_DIR = "model"
WEIGHTS_FILENAME = "vgg16_weights.pth"
LOCAL_PATH = os.path.join(MODEL_DIR, WEIGHTS_FILENAME)
IMAGE_DIR = "images"  # New directory variable


IMAGE_FILENAMES = (
    "cat.jpg",
    "koala-in-tree.webp",
    "roe-deer-eating-apple.webp",
    "tiger-roaring.webp"
)

def setup_model_and_weights():
    """Initializes VGG16 and loads weights from a local path, downloading if necessary."""
    print("Setting up VGG16 model and loading weights...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(LOCAL_PATH):
        print(f"Weights not found locally at {LOCAL_PATH}. Downloading now...")
        VGG16_DOWNLOAD_URL = models.VGG16_Weights.DEFAULT.url

        # Download and save the state dictionary
        state_dict = torch.hub.load_state_dict_from_url(
            VGG16_DOWNLOAD_URL, model_dir=MODEL_DIR, file_name=WEIGHTS_FILENAME
        )
        print(f"Weights successfully downloaded and saved to {LOCAL_PATH}.")
    else:
        print(f"Weights found locally at {LOCAL_PATH}. Loading from disk...")
        state_dict = torch.load(LOCAL_PATH)

    model = models.vgg16(weights=None)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_imagenet_class_names():
    """Downloads the ImageNet class index file and returns a list of names."""
    LABELS_URL = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )

    try:
        response = requests.get(LABELS_URL)
        class_names = [
            line.strip() for line in response.text.split("\n") if line.strip()
        ]
        if len(class_names) == 1000:
            return class_names
        else:
            print("Warning: Could not load exactly 1000 class names.")
            return None
    except Exception as e:
        print(f"Error loading class names: {e}")
        return None


def prepare_image(filename):
    """Loads an image, applies VGG16 preprocessing, and returns the batch tensor."""

    file_path = os.path.join(IMAGE_DIR, filename)

    # PyTorch models require specific transformations (resize, center crop, normalize)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        # Open the image directly from the local file path 🚨
        img = Image.open(file_path).convert(
            "RGB"
        )  # Use .convert('RGB') to handle WEBP/PNG transparency issues
        print(f"Image loaded from disk: {filename}. Size: {img.size}")

        # Apply transforms
        img_t = preprocess(img)

        # Add a batch dimension: (C, H, W) -> (1, C, H, W)
        img_batch = img_t.unsqueeze(0)

        return img_batch

    except Exception as e:
        print(f"ERROR: Could not process image file '{filename}': {e}")
        return None


def run_inference(model, input_batch):
    """Makes the prediction and returns the raw output."""
    print("\nRunning classification...")
    with torch.no_grad():
        output = model(input_batch)
    return output


def display_results(output, class_labels, top_k=5):
    """Converts model output to probabilities, finds top K, and prints human-readable labels."""
    probabilities = nn.Softmax(dim=1)(output)[0]
    top_prob, top_catid = torch.topk(probabilities, top_k)

    print(f"\n--- Top {top_k} Predictions ---")
    for i in range(top_prob.size(0)):
        index = top_catid[i].item()
        confidence = top_prob[i].item()

        if index < len(class_labels):
            label = class_labels[index].capitalize()
        else:
            label = f"Index {index} (Label List Incomplete)"

        print(f"{i + 1}. {label} (Confidence: {confidence:.2f})")
    print("--------------------------")


def main():
    model = setup_model_and_weights()
    class_labels = get_imagenet_class_names()
    if class_labels is None:
        return

    print("\nStarting local image classification batch...\n")

    # Loop through all specified filenames
    for i, filename in enumerate(IMAGE_FILENAMES):
        print("\n=======================================================")
        print(f"Processing Image {i + 1} of {len(IMAGE_FILENAMES)}: {filename}")
        print("=======================================================")

        # Prepare Image (now loads from disk)
        input_batch = prepare_image(filename)

        if input_batch is not None:
            output = run_inference(model, input_batch)
            display_results(output, class_labels, top_k=5)
        else:
            # Prints the error message from within prepare_image
            continue


if __name__ == "__main__":
    main()