"""
Milestone 2: Multi-image upload + CLIP embeddings.

Images are uploaded in order, converted to embeddings using a
pretrained CLIP model, and the embedding shape/preview is shown per image.
"""

import streamlit as st
import torch
from utils.image_utils import load_images_in_order
from models.clip_encoder import ClipEncoder

st.set_page_config(page_title="Sequential Image Reasoning", layout="wide")

st.title("Sequential Multi-Image Reasoning")
st.caption("Milestone 2: CLIP embeddings")


@st.cache_resource
def load_clip_encoder() -> ClipEncoder:
    """
    Load the CLIP model once and cache it across Streamlit reruns.

    Streamlit reruns the ENTIRE script top-to-bottom every time you
    interact with the UI (upload a file, click a button, etc.). Without
    caching, we'd reload the CLIP model from disk on every single
    interaction — slow, and wasteful of VRAM since we'd keep creating
    new copies of the model in memory.

    @st.cache_resource tells Streamlit: "run this function once, keep
    the result alive across reruns, and give every rerun the same
    object instead of recreating it."
    """
    return ClipEncoder()


# --- Upload ---
uploaded_files = st.file_uploader(
    "Upload images in the order you want them processed",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload two or more images to see them in sequence.")
    st.stop()

images = load_images_in_order(uploaded_files)

st.subheader(f"Sequence ({len(images)} images)")

# --- Ordered preview ---
cols_per_row = 4
for row_start in range(0, len(images), cols_per_row):
    row_images = images[row_start: row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for i, img in enumerate(row_images):
        position = row_start + i + 1
        with cols[i]:
            st.image(img, caption=f"Image {position}")

st.divider()

# --- CLIP Embeddings ---
st.subheader("CLIP Embeddings")

encoder = load_clip_encoder()
st.caption(f"Running on device: **{encoder.device}**")

with st.spinner("Generating embeddings..."):
    embeddings = encoder.encode_images(images)

for i, emb in enumerate(embeddings):
    position = i + 1
    with st.expander(f"Image {position} — embedding info"):
        st.write(f"Shape: `{tuple(emb.shape)}`")
        st.write(f"First 5 values: `{emb[:5].tolist()}`")
        st.write(f"Norm (should be ~1.0 after normalization): `{emb.norm().item():.4f}`")

st.divider()
st.write(
    "Embeddings generated. Similarity comparison comes in Milestone 3."
)