"""
Milestone 1: Multi-image upload with order-preserving preview.

No AI models are involved yet. The goal of this milestone is purely
to prove that Streamlit can accept multiple images and that we can
display them in a stable, predictable sequence — because every later
milestone depends on this ordering being trustworthy.
"""

import streamlit as st
from utils.image_utils import load_images_in_order

st.set_page_config(page_title="Sequential Image Reasoning", layout="wide")

st.title("Sequential Multi-Image Reasoning")
st.caption("Milestone 1: Ordered upload + preview (no AI yet)")

# --- Upload ---
uploaded_files = st.file_uploader(
    "Upload images in the order you want them processed",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload two or more images to see them in sequence.")
    st.stop()

# --- Convert to PIL, preserving order ---
# NOTE: st.file_uploader returns files in the order the user selected
# them in the OS file picker (on most platforms). We do not re-sort
# them by filename or anything else — whatever order Streamlit hands
# us is the order we treat as the sequence.
images = load_images_in_order(uploaded_files)

st.subheader(f"Sequence ({len(images)} images)")

# --- Ordered preview ---
# Using columns so images are visibly numbered left-to-right, top-to-bottom,
# matching how a person reads a sequence.
cols_per_row = 4
for row_start in range(0, len(images), cols_per_row):
    row_images = images[row_start: row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for i, img in enumerate(row_images):
        position = row_start + i + 1  # 1-indexed for human readability
        with cols[i]:
            st.image(img, caption=f"Image {position}")

st.divider()
st.write(
    "This is just the upload + ordering step. "
    "CLIP embeddings come in Milestone 2."
)