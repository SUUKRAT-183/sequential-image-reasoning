"""
Milestone 7: Final integrated pipeline.

Upload images -> preserve order -> CLIP embeddings -> similarity ->
BLIP captions -> ordered context -> user question -> Gemini answer.
"""

import streamlit as st
import pandas as pd

from utils.image_utils import load_images_in_order
from models.clip_encoder import ClipEncoder
from models.blip_captioner import BlipCaptioner
from models.llm_client import GeminiClient
from pipeline.similarity import compute_similarity_matrix
from pipeline.image_processor import build_processed_sequence
from pipeline.prompt_builder import build_sequence_context

st.set_page_config(page_title="Sequential Image Reasoning", layout="wide")


# --- Cached model loaders ---

@st.cache_resource
def load_clip_encoder() -> ClipEncoder:
    return ClipEncoder()


@st.cache_resource
def load_blip_captioner() -> BlipCaptioner:
    return BlipCaptioner()


@st.cache_resource
def load_llm_client() -> GeminiClient:
    return GeminiClient()


# --- Sidebar: pipeline status ---
with st.sidebar:
    st.header("Pipeline Status")
    encoder_preview = load_clip_encoder()
    st.write(f"**Device:** {encoder_preview.device}")
    st.write("**CLIP:** openai/clip-vit-base-patch32")
    st.write("**BLIP:** Salesforce/blip-image-captioning-base")
    st.write("**LLM:** Gemini (gemini-3.5-flash)")
    st.caption("Models are loaded once and cached across reruns.")


# --- Main title ---
st.title("Sequential Multi-Image Reasoning")
st.caption(
    "Upload ordered images. The system embeds, compares, captions, "
    "and reasons across the full sequence."
)

# --- Step 1: Upload ---
st.header("Step 1 — Upload Images in Order")
uploaded_files = st.file_uploader(
    "Upload images in the order you want them processed",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload two or more images to begin.")
    st.stop()

images = load_images_in_order(uploaded_files)
filenames = [f.name for f in uploaded_files]

if len(images) > 8:
    st.warning(
        f"You've uploaded {len(images)} images. Large batches may take "
        "longer to process and images are automatically resized before "
        "being sent to the LLM to avoid connection issues."
    )

with st.expander(f"Preview sequence ({len(images)} images)", expanded=True):
    cols_per_row = 4
    for row_start in range(0, len(images), cols_per_row):
        row_images = images[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, img in enumerate(row_images):
            position = row_start + i + 1
            with cols[i]:
                st.image(img, caption=f"Image {position}")

st.divider()

# --- Step 2: CLIP embeddings + similarity ---
st.header("Step 2 — Visual Embeddings & Similarity")

encoder = load_clip_encoder()
with st.spinner("Generating CLIP embeddings..."):
    embeddings = encoder.encode_images(images)

with st.expander("Embedding details (shape, sample values)", expanded=True):
    for i, emb in enumerate(embeddings):
        position = i + 1
        st.write(
            f"**Image {position}** — shape `{tuple(emb.shape)}`, "
            f"norm `{emb.norm().item():.4f}`"
        )
        st.caption(f"First 5 values: {emb[:5].tolist()}")

similarity_matrix = compute_similarity_matrix(embeddings)
labels = [f"Image {i + 1}" for i in range(len(images))]
similarity_df = pd.DataFrame(
    similarity_matrix.numpy(), index=labels, columns=labels
)

with st.expander("Similarity matrix", expanded=True):
    st.dataframe(
        similarity_df.style.format("{:.2f}").background_gradient(cmap="Blues")
    )

st.divider()

# --- Step 3: BLIP captions ---
st.header("Step 3 — Captions")

captioner = load_blip_captioner()
with st.spinner("Generating captions..."):
    captions = captioner.generate_captions(images)

with st.expander("Captions per image", expanded=True):
    for i, caption in enumerate(captions):
        position = i + 1
        st.write(f"**Image {position}:** {caption}")

st.divider()

# --- Step 4: Ordered sequence context ---
st.header("Step 4 — Ordered Sequence Context")

processed_images = build_processed_sequence(filenames, embeddings, captions)
sequence_context = build_sequence_context(processed_images)

with st.expander("Text context sent to the LLM", expanded=True):
    st.text(sequence_context)

st.divider()

# --- Step 5: Ask the LLM ---
st.header("Step 5 — Ask About the Sequence")

question = st.text_input(
    "Your question", placeholder="What happened across these images?"
)

if st.button("Ask") and question:
    llm_client = load_llm_client()
    with st.spinner("Thinking..."):
        try:
            answer = llm_client.generate_answer(sequence_context, images, question)
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Something went wrong while contacting the LLM: {e}")