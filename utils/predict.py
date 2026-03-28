"""
Prediction utilities — load saved model and classify a single MRI image.
"""

import numpy as np
from PIL import Image
import tensorflow as tf

CLASSES   = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE  = (224, 224)


def load_model(model_path: str = "model/brain_tumor_cnn.h5"):
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, normalise, and add batch dimension."""
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # (1, 224, 224, 3)


def predict(model, image: Image.Image) -> tuple[str, float, list[float]]:
    """
    Returns
    -------
    predicted_class : str
    confidence      : float  (0–100)
    all_probs       : list[float]  — probability per class
    """
    processed   = preprocess_image(image)
    probs       = model.predict(processed, verbose=0)[0]
    idx         = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]) * 100, probs.tolist()
