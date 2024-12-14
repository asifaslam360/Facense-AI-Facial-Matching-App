import streamlit as st
from PIL import Image
from deepface import DeepFace
import numpy as np
import os

#code to change backgroud

st.markdown(
    """
    <style>
html, body, [data-testid="stAppViewContainer"] {
            background-color: #6A669D;  
            color: white; /* Text color for visibility */
            font-family: 'Roboto', sans-serif;
            height: 100%;
            margin: 0;
            padding: 0;
        }
        </style>
    """,
    unsafe_allow_html=True,
)

 



# Importing custom HTML and CSS
def load_custom_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_custom_html():
    with open("header.html") as f:
        st.markdown(f.read(), unsafe_allow_html=True)

# Load custom HTML and CSS
load_custom_html()
load_custom_css()

# Create two columns for uploading
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Query Image")
    query_image_file = st.file_uploader("Choose a query image", type=["jpg", "jpeg", "png"], key="query")

with col2:
    st.subheader("Upload Target Images")
    target_images_files = st.file_uploader("Choose target images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="targets"
    )

# Display uploaded images
col3, col4 = st.columns(2)

with col3:
    st.subheader("Query Image")
    if query_image_file:
        query_image = Image.open(query_image_file)
        st.image(query_image, caption="Uploaded Query Image", use_container_width=True)

with col4:
    st.subheader("Matched Image")
    matched_image_placeholder = st.empty()

# Adding Start Matching button and progress spinner
start_matching = st.button("Start Matching")

if start_matching:
    if query_image_file and target_images_files:
        with st.spinner("Matching in progress..."):
            try:
                # Save query image temporarily
                query_path = "temp_query.jpg"
                with open(query_path, "wb") as f:
                    f.write(query_image_file.getvalue())

                match_found = False

                # Iterate through target images
                for target_file in target_images_files:
                    # Save each target image temporarily
                    target_path = f"temp_target_{os.path.basename(target_file.name)}"
                    with open(target_path, "wb") as f:
                        f.write(target_file.getvalue())

                    # Use DeepFace for verification
                    try:
                        result = DeepFace.verify(img1_path=query_path, img2_path=target_path, enforce_detection=False)
                        if result["verified"]:
                            matched_image = Image.open(target_file)
                            matched_image_placeholder.image(matched_image, caption="Matched Image", use_container_width=True)
                            match_found = True
                            break
                    except Exception as e:
                        st.error(f"Error verifying {target_file.name}: {e}")

                if not match_found:
                    st.warning("No match found.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Clean up temporary files
                if os.path.exists(query_path):
                    os.remove(query_path)
                for target_file in target_images_files:
                    temp_target_path = f"temp_target_{os.path.basename(target_file.name)}"
                    if os.path.exists(temp_target_path):
                        os.remove(temp_target_path)
    else:
        st.error("Please upload both query and target images.")
