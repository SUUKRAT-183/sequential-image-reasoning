"""
CLIP image encoder.

Wraps a pretrained CLIP model to convert PIL images into normalized
embedding vectors. We never train or fine-tune this model — it's used
purely for inference (feature extraction).
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from typing import List


# Using the base CLIP model: small, fast, well-documented, and produces
# 512-dimensional embeddings. There are larger CLIP variants (e.g.
# clip-vit-large-patch14) that give better embeddings at the cost of
# more compute and VRAM — base is the right starting point for us.
MODEL_NAME = "openai/clip-vit-base-patch32"


def get_device() -> torch.device:
    """
    Pick the best available compute device.

    We check for CUDA (NVIDIA GPU) first since your RTX 4050 will run
    inference much faster than CPU. If no GPU is available (e.g. code
    runs on a different machine later), we fall back to CPU so the
    project still works everywhere — just slower.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ClipEncoder:
    """
    Loads a pretrained CLIP model once and reuses it to encode images.

    We wrap this in a class (rather than loading the model inside a
    function) so the model is loaded ONE time and kept in memory,
    instead of being reloaded from disk on every single image — that
    would be extremely slow and wasteful of VRAM.
    """

    def __init__(self):
        self.device = get_device()

        # CLIPModel: the actual neural network (vision + text towers).
        # We only use the vision tower here, but the model ships as one
        # object containing both.
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)

        # CLIPProcessor: handles image preprocessing — resizing, cropping,
        # pixel normalization to match what CLIP was trained on. We must
        # use CLIP's own processor, not generic PIL resizing, because the
        # model expects a very specific input format (fixed size, specific
        # mean/std normalization values).
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)

        # eval() switches off training-specific behavior (like dropout
        # layers randomly zeroing activations). We are never training
        # this model, only running inference, so eval() must always be
        # set — forgetting it is a common bug that causes inconsistent
        # outputs between runs.
        self.model.eval()

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Convert a single PIL image into a normalized CLIP embedding.

        Returns a 1D tensor of shape (embedding_dim,) — for CLIP-base,
        embedding_dim = 512.
        """
        # The processor converts the PIL image into a batch of pixel
        # tensors CLIP expects. return_tensors="pt" means "give me
        # PyTorch tensors" (as opposed to numpy or TensorFlow tensors).
        inputs = self.processor(images=image, return_tensors="pt")

        # Move the input tensor(s) to the same device as the model.
        # PyTorch requires the model and its inputs to be on the same
        # device (both CPU or both GPU) — mismatched devices throw a
        # runtime error.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # torch.no_grad() disables gradient tracking. Gradients are only
        # needed during training (to compute how to update weights).
        # Since we're doing pure inference, tracking gradients would
        # waste memory and compute for no benefit.
        with torch.no_grad():
            # get_image_features runs only the vision tower of CLIP,
            # skipping the text tower entirely since we're not encoding
            # any text right now.
            embedding = self.model.get_image_features(**inputs)

        # embedding shape at this point: (1, 512) — the leading 1 is the
        # "batch" dimension, since the processor always returns a batch
        # even for a single image. squeeze(0) removes that batch dim,
        # leaving a clean (512,) vector.
        embedding = embedding.squeeze(0)

        # L2-normalize the embedding: divide by its own magnitude so its
        # length becomes exactly 1. CLIP embeddings are designed to be
        # compared via cosine similarity, and normalizing here means a
        # simple dot product between two embeddings IS the cosine
        # similarity — no need to re-normalize at comparison time later.
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

        # Move back to CPU before returning. Downstream code (Streamlit
        # display, similarity math, storage) doesn't need GPU tensors,
        # and CPU tensors are safer to pass around / convert to numpy.
        return embedding.cpu()

    def encode_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Encode a list of images, preserving order.

        We keep this simple (loop, not batched) for now. Batching all
        images into a single forward pass would be faster, but adds
        complexity (padding, variable batch sizes) we don't need yet
        for a college project with a handful of images per sequence.
        """
        return [self.encode_image(img) for img in images]