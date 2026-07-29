"""
Small helpers for loading and validating uploaded images.
Kept separate from app.py so Streamlit UI code doesn't get tangled
with image I/O logic — makes both easier to test/reason about.
"""

from PIL import Image
from typing import List
from io import BytesIO


def load_image_from_upload(uploaded_file) -> Image.Image:
    """
    Convert a Streamlit UploadedFile object into a PIL Image.

    Streamlit gives us a file-like object (BytesIO-backed), not a path.
    We convert to RGB explicitly because:
      - PNGs can be RGBA (alpha channel)
      - Some JPEGs are grayscale (mode 'L')
    Downstream models (CLIP, BLIP) expect consistent 3-channel RGB input,
    so normalizing this here avoids shape/channel errors later.
    """
    image = Image.open(BytesIO(uploaded_file.getvalue()))
    return image.convert("RGB")


def load_images_in_order(uploaded_files: List) -> List[Image.Image]:
    """
    Convert a list of Streamlit UploadedFile objects into PIL Images,
    preserving the exact order they were passed in.

    This function is intentionally trivial right now — its whole job
    is to be the ONE place order-preservation happens, so later, if a
    bug causes images to shuffle, we know exactly where to look.
    """
    return [load_image_from_upload(f) for f in uploaded_files]