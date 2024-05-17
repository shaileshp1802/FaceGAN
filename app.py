import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

# Function to load the generator model
@st.cache(allow_output_mutation=True)
def load_generator(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to generate an image
def generate_image(generator, latent_dim=100):
    noise = np.random.normal(size=(1, latent_dim))
    generated_image = generator.predict(noise)
    generated_image = (generated_image * 127.5 + 127.5).astype(np.uint8)  # Scale to [0, 255]
    return generated_image[0]

# Convert image to bytes for download
def convert_image_to_bytes(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()

# Streamlit UI
st.title("Anime Face Image Generator")

# Path to the saved model
model_path = "dcgan_generator.h5"  # Ensure this path matches where you saved your model
latent_dim = 100  # Latent dimension size

# Load the model
generator = load_generator(model_path)

# Button to generate or regenerate a new image
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

if st.session_state.generated_image is None:
    if st.button("Generate Image"):
        st.session_state.generated_image = generate_image(generator, latent_dim)
else:
    if st.button("Regenerate Image"):
        st.session_state.generated_image = generate_image(generator, latent_dim)

# Display the generated image
if st.session_state.generated_image is not None:
    st.image(Image.fromarray(st.session_state.generated_image), caption="Generated Image", use_column_width=False)
    image = Image.fromarray(st.session_state.generated_image)

    # Save button to download the image
    image_bytes = convert_image_to_bytes(image)
    st.download_button(label="Save Image", data=image_bytes, file_name="generated_image.png", mime="image/png")

# Run the Streamlit app
# Run this script with the command: streamlit run app.py
