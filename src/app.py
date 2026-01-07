import os
from typing import Tuple

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STERILITY_REQUIRED = 95
MODEL_PATH = os.getenv("MODEL_PATH", "../models/efficientnet_b0_tune_sterility_v0.0.1.pth")


@st.cache_resource
def load_model(model_path: str) -> nn.Module:
    """
    Load the EfficientNet-B0 model with pre-trained weights for sterility detection.

    Args:
        model_path (str): Path to the saved model file (.pth)

    Returns:
        nn.Module: Loaded PyTorch model ready for inference

    Raises:
        SystemExit: If model file is not found
    """
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    # Create the same architecture as during training
    model = torch.hub.load(
        "pytorch/vision:v0.14.0", 
        "efficientnet_b0", 
        # weights='IMAGENET1K_V1'
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()  # Important! Disables dropout/batchnorm
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for model inference.

    Applies resize, normalization, and converts to tensor format expected by the model.
    Uses ImageNet normalization values for compatibility with pre-trained EfficientNet.

    Args:
        image (Image.Image): Input PIL image in any format

    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 3, 224, 224)
    """
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    return transform(image).unsqueeze(0)  # Add batch dimension


def predict(
    model: nn.Module, image_tensor: torch.Tensor, threshold: float = 0.5
) -> Tuple[int, float]:
    """
    Make sterility prediction on a preprocessed image tensor.

    Args:
        model (nn.Module): Trained PyTorch model for inference
        image_tensor (torch.Tensor): Preprocessed image tensor
        threshold (float, optional): Classification threshold. Defaults to 0.5.

    Returns:
        Tuple[int, float]: A tuple containing:
            - pred (int): Predicted class (0=sterile, 1=non-sterile)
            - prob (float): Probability of being non-sterile (class 1)
    """
    with torch.no_grad():
        output = model(image_tensor.to(DEVICE)).squeeze()
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > threshold else 0
    return pred, prob


def main() -> None:
    """
    Main Streamlit application for instrument sterility detection.

    Creates a web interface that allows users to upload medical instrument images
    and get sterility predictions using a trained EfficientNet-B0 model.
    """
    st.set_page_config(
        page_title="Instrument Sterility Detection", page_icon="🔬", layout="wide"
    )

    st.title("🔬 Instrument Sterility Detection")

    # Load model
    with st.spinner("Loading model..."):
        model = load_model(MODEL_PATH)

    # Subtitle row
    sub_col1, sub_col2 = st.columns([1, 1])
    with sub_col1:
        st.markdown("Upload an image of a medical instrument to check its sterility status")
    with sub_col2:
        st.markdown(f"✅ {DEVICE.upper()}")

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["png", "jpg", "jpeg", "tiff", "bmp"],
            help="Upload an image of a medical instrument",
        )

        # Sterility confidence slider (user-friendly: higher = stricter)
        # sterility_required = st.slider(
        #     "🎯 Required Sterility Confidence",
        #     min_value=0,
        #     max_value=100,
        #     value=90,
        #     step=1,
        #     format="%d%%",
        #     help="Set the minimum sterility confidence required to mark an instrument as Sterile. "
        #          "Higher value = stricter check. Recommended: 90% for medical use.",
        # )
        # st.caption(f"✅ Instruments with **≥{sterility_required}%** sterility confidence will pass")
        
        # # Convert to internal threshold (contamination risk threshold)
        # threshold = 1 - (sterility_required / 100)

        threshold = 1 - (STERILITY_REQUIRED / 100)

        st.markdown("---")
        st.markdown("**Model Information:**")
        st.markdown("- Architecture: EfficientNet-B0")
        st.markdown(f"- Device: {DEVICE}")
        st.markdown(f"- Input Size: {IMG_SIZE}x{IMG_SIZE}")

        st.markdown("---")
        st.markdown("**📁 Quick Start:**")
        st.markdown("""
1. 📥 [Download sample images from Yandex Disk](https://disk.360.yandex.ru/d/4dmDJqBSVoxnwQ)
2. 📤 Upload an image using the file picker above
3. 🔍 View cleanliness verification results
""")

    with col2:
        st.header("🔍 Results")

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width='stretch')

            # Make prediction
            with st.spinner("Analyzing image..."):
                # Preprocess image
                img_tensor = preprocess_image(image)

                # Get prediction
                pred, prob = predict(model, img_tensor, threshold)

            # Display results
            st.markdown("### 📊 Prediction Results")

            # Main prediction
            if pred == 1:
                st.error("🚨 **Non-Sterile**")
            else:
                st.success("✅ **Sterile**")

        else:
            st.info("👆 Please upload an image to begin analysis")

    # Footer
    st.markdown("---")
    st.markdown(
        "**About:** This model was trained to detect sterility in medical instruments using computer vision."
    )


if __name__ == "__main__":
    main()
