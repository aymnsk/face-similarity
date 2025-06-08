import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Face Similarity Checker", layout="centered")
st.title("🧠 Face Similarity Checker using Siamese Logic")

def get_face_embedding(image):
    image_np = np.array(image)
    face_locations = face_recognition.face_locations(image_np)
    if len(face_locations) == 0:
        return None
    face_encodings = face_recognition.face_encodings(image_np, face_locations)
    return face_encodings[0] if face_encodings else None

def compare_embeddings(embedding1, embedding2):
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance, distance < 0.6

col1, col2 = st.columns(2)

with col1:
    img1 = st.file_uploader("Upload First Face", type=["jpg", "jpeg", "png"], key="1")
with col2:
    img2 = st.file_uploader("Upload Second Face", type=["jpg", "jpeg", "png"], key="2")

if img1 and img2:
    image1 = Image.open(img1).convert("RGB")
    image2 = Image.open(img2).convert("RGB")

    st.image([image1, image2], caption=["Image 1", "Image 2"], width=200)

    with st.spinner("Comparing faces..."):
        embedding1 = get_face_embedding(image1)
        embedding2 = get_face_embedding(image2)

        if embedding1 is not None and embedding2 is not None:
            distance, is_same = compare_embeddings(embedding1, embedding2)
            st.write(f"🧪 Euclidean Distance: `{distance:.3f}`")
            if is_same:
                st.success("✅ Faces are of the same person.")
            else:
                st.error("❌ Faces are different.")
        else:
            st.warning("Could not detect face in one or both images. Try clearer images.")
