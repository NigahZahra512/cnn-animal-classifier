import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Pro Animal Classifier", page_icon="🐱")

st.title("🐾 Multiclass Animal Classifier")
st.write("Current focus: **Cats, Dogs, and Snakes**")

# 2. Load Model
@st.cache_resource
def load_my_model():
    # Ensure this file exists in your folder
    return tf.keras.models.load_model('Animal_Classifier_Model.keras')

model = load_my_model()

# 3. Image Upload
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and show image
    image = Image.open(uploaded_file)
    st.image(image, caption='Target Image', use_container_width=True)
    
    # 4. Precise Preprocessing
    # Using Resampling.LANCZOS for high-quality resizing
    size = (256, 256)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert to float32 and normalize
    img_array = np.asarray(image_resized).astype(np.float32) / 255.0
    
    # Add batch dimension [1, 256, 256, 3]
    final_img = np.expand_dims(img_array, axis=0)

    # 5. Prediction
    prediction = model.predict(final_img)
    
    # --- CRITICAL: MATCHING THE CLASSES ---
    # Based on your previous errors, we need to ensure this matches 
    # your folder structure exactly (Alphabetical is default).
    class_names = ['Cat', 'Dog', 'Snake'] 
    
    # Get the index with the highest score
    index = np.argmax(prediction)
    result = class_names[index]
    confidence = prediction[0][index] * 100

    # 6. Displaying the "Thinking Process"
    st.write("### AI Confidence Scores:")
    
    # Create columns to show scores for all three
    cols = st.columns(3)
    for i, name in enumerate(class_names):
        cols[i].metric(name, f"{prediction[0][i]*100:.1f}%")

    st.write("---")
    
    # Final Verdict
    if confidence > 40: # Threshold
        st.success(f"## Predicted: {result}")
    else:
        st.error("## Result: Unclear (The AI is confused)")

else:
    st.info("Please upload a file to begin.")