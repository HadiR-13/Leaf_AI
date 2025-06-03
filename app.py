import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

import numpy as np
import cv2

import streamlit as st
from PIL import Image
import tempfile

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 8

GYMNOSPERMAE = [
    "abies_concolor", "abies_nordmanniana", "cedrus_atlantica", "cedrus_deodara", 
    "cedrus_libani", "chamaecyparis_pisifera", "chamaecyparis_thyoides", "cryptomeria_japonica", 
    "ginkgo_biloba", "juniperus_virginiana", "larix_decidua", "metasequoia_glyptostroboides", 
    "picea_abies", "picea_orientalis", "picea_pungens", "pinus_bungeana", "pinus_cembra", 
    "pinus_densiflora", "pinus_echinata", "pinus_flexilis", "pinus_koraiensis", "pinus_nigra", 
    "pinus_parviflora", "pinus_peucea", "pinus_pungens", "pinus_resinosa", "pinus_rigida", 
    "pinus_strobus", "pinus_sylvestris", "pinus_taeda", "pinus_thunbergii", "pinus_virginiana",
    "pinus_wallichiana", "pseudolarix_amabilis", "taxodium_distichum", "tsuga_canadensis"
]

test_dir = "./Test_AI"

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',   
    shuffle=False
)

class_indices = test_generator.class_indices
labels = {v: k for k, v in class_indices.items()}

@register_keras_serializable(package="Custom", name="f1_score")
def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

model = load_model("leaf_latin_model.keras")

def predict_image(image_np):
    resized = cv2.resize(image_np, IMAGE_SIZE)
    array = img_to_array(resized) / 255.0
    array = np.expand_dims(array, axis=0)

    pred = model.predict(array)
    predicted_class_index = np.argmax(pred)
    confidence = pred[0][predicted_class_index] * 100
    latin_leaf_name = labels[predicted_class_index]
    genome_ = ["Angiospermae", "Gymnospermae"][latin_leaf_name in GYMNOSPERMAE]

    return resized, latin_leaf_name, genome_, confidence

st.title("🌿 Leaf Classifier")

choice = st.sidebar.radio("Choose Input Method", ["Upload File", "Use Camera"])

if choice == "Upload File":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        st.image(img, caption="Uploaded Image")
        result_img, name, genome, conf = predict_image(img_np)
        st.success(f"🧬 Predicted: **{name.replace('_', ' ').title()}** ({genome}) with **{conf:.2f}%** confidence")
        st.image(result_img, caption="Processed Image")

elif choice == "Use Camera":
    camera_img = st.camera_input("Capture Image")
    if camera_img:
        img = Image.open(camera_img).convert("RGB")
        img_np = np.array(img)
        st.image(img, caption="Captured Image")
        result_img, name, genome, conf = predict_image(img_np)
        st.success(f"🧬 Predicted: **{name.replace('_', ' ').title()}** ({genome}) with **{conf:.2f}%** confidence")
        st.image(result_img, caption="Processed Image")
