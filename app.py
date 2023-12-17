import streamlit as st
from test import make_prediction
import tempfile
import os

st.title("Computer Vision Chest Cancer Prediction")

# Upload potential image 
image = st.file_uploader("Upload a Chest Scan", accept_multiple_files=False)
if image: 
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, image.name)
    with open(path, "wb") as f:
        f.write(image.getvalue())
# Show uploaded image 
    st.write("Your inputted imaged: ")
    st.image(image)

    if st.button("Run Model on Image"): 
        with st.spinner("Processing..."):
            preds, probabilities = make_prediction(path)
            st.write("predicted class: ", preds)
            st.write("probabilities of each class", probabilities)


# # Graph confidence / probabilities of each class -- bar graph that changes with each upload
# st.bar_chart()
# st.area_chart()


# Stretch Goal: Deconvolution activation maps to show the focus-points of the image
