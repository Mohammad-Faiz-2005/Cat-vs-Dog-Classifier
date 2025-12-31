import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🐱 Cat vs Dog Classifier")

model = tf.keras.models.load_model("cat_dog_model888.keras")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((128,128))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    if pred[0][0] >= 0.5:
        st.success("🐶 Dog")
    else:
        st.success("🐱 Cat")
