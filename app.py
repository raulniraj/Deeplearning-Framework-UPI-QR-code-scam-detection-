import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode
import json
import io

# --- Parameters ---
URL_MODEL_PATH = 'url_model.h5'
URL_TOKENIZER_PATH = 'url_tokenizer.json'
QR_MODEL_PATH = 'qr_model.h5'
MAX_SEQUENCE_LENGTH = 100
IMG_HEIGHT = 128
IMG_WIDTH = 128

# --- 1. Load Models and Tokenizer ---
#st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_all_models():
    """Load models and tokenizer from disk."""
    url_model = load_model(URL_MODEL_PATH)
    qr_model = load_model(QR_MODEL_PATH)
    
    with open(URL_TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
        url_tokenizer = tokenizer_from_json(tokenizer_json)
        
    return url_model, qr_model, url_tokenizer

url_model, qr_model, url_tokenizer = load_all_models()

# --- 2. Helper Functions ---
def decode_qr_code(image_pil):
    """Decodes QR code and returns the first data found."""
    try:
        decoded_objects = decode(image_pil)
        if decoded_objects:
            return decoded_objects[0].data.decode('utf-8')
    except Exception as e:
        st.error(f"Error decoding QR code: {e}")
    return None

def predict_url_risk(url_string):
    """Generates a risk score for a given URL string."""
    try:
        sequence = url_tokenizer.texts_to_sequences([url_string])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        prediction = url_model.predict(padded_sequence)
        return prediction[0][0] # Get the scalar value
    except Exception as e:
        st.error(f"Error predicting URL risk: {e}")
        return None

def predict_qr_image_risk(image_pil):
    """Generates a risk score for the visual properties of a QR code image."""
    try:
        # Preprocess the image
        img = image_pil.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img)
        
        # Ensure 3 channels (for JPGs, etc.)
        if img_array.ndim == 2: # Grayscale
            img_array = np.stack((img_array,)*3, axis=-1)
        
        # Keep only 3 channels if 4 (like PNG)
        if img_array.shape[2] == 4:
            img_array = img_array[..., :3]

        img_array = img_array / 255.0 # Rescale
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

        prediction = qr_model.predict(img_array)
        return prediction[0][0] # Get the scalar value
    except Exception as e:
        st.error(f"Error predicting QR image risk: {e}")
        return None

# --- 3. Streamlit UI ---
st.title("UPI/QR Code Phishing Detector 🛡️")
st.write("Upload a QR code image to analyze it for potential phishing risks.")

uploaded_file = st.file_uploader("Choose a QR code image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. Load Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded QR Code', use_column_width=True)
    
    st.info("Analyzing... Please wait.")
    
    # 2. Decode URL from QR
    decoded_url = decode_qr_code(image)
    
    if not decoded_url:
        st.error("Could not decode a URL from this QR code.")
    else:
        st.subheader("Analysis Results:")
        
        # 3. Run Predictions
        url_risk = predict_url_risk(decoded_url)
        qr_visual_risk = predict_qr_image_risk(image)
        
        # 4. Combine Scores (Simple Weighted Average)
        # In a real project, you would tune these weights.
        url_weight = 0.6
        visual_weight = 0.4
        final_risk = (url_risk * url_weight) + (qr_visual_risk * visual_weight)
        
        # 5. Display Results
        st.markdown(f"**Decoded URL:** `{decoded_url}`")
        st.markdown(f"**URL Phishing Risk:** `{url_risk:.2%}`")
        st.markdown(f"**Visual Deception Risk:** `{qr_visual_risk:.2%}`")
        
        st.subheader(f"Final Combined Risk: {final_risk:.2%}")
        
        # 6. Final Verdict
        if final_risk > 0.6: # 60% threshold
            st.error("🔴 **High Risk!** This QR code shows strong signs of a phishing scam. **Do not proceed.**")
        elif final_risk > 0.3: # 30% threshold
            st.warning("🟡 **Medium Risk.** This QR code contains suspicious elements. Please be cautious.")
        else:
            st.success("🟢 **Low Risk.** This QR code appears to be safe.")
