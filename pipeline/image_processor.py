"""
Defines the ProcessedImage structure — the unified representation of
one image's position, filename, embedding, and caption.

This is the single source of truth for "everything we know about
image N in the sequence." Building the CLIP embedding, BLIP caption,
and position into ONE object (instead of three parallel lists) means
they can never accidentally get shuffled relative to each other.
"""

from dataclasses import dataclass
import torch


@dataclass
class ProcessedImage:
    """
    Holds all processed information for a single image in the sequence.

    We use a dataclass here instead of a plain dict because:
    - Fields are explicit and typed (position: int, not a guessable
      dict key like data["pos"] vs data["position"]).
    - Typos in field names are caught by your editor/type checker
      instead of failing silently at runtime.
    - It's still simple — no custom methods or complex behavior needed,
      just a clean bundle of related data.
    """

    position: int          # 1-indexed order in the sequence — metadata, not positional encoding
    filename: str          # original uploaded filename, useful for debugging/display
    embedding: torch.Tensor  # CLIP embedding, shape (512,)
    caption: str            # BLIP-generated caption


def build_processed_sequence(
    filenames: list[str],
    embeddings: list[torch.Tensor],
    captions: list[str],
) -> list[ProcessedImage]:
    """
    Combine parallel lists (filenames, embeddings, captions) into a
    single ordered list of ProcessedImage objects.

    This is the ONE place where position numbers get assigned. Every
    other part of the app should read position from a ProcessedImage
    object rather than recomputing "index + 1" independently — keeping
    a single source of truth avoids ordering bugs creeping in as the
    project grows.
    """
    if not (len(filenames) == len(embeddings) == len(captions)):
        # Defensive check: if these three lists ever get out of sync
        # (e.g. one image failed to caption), we want a loud, clear
        # error here rather than a silent mismatch further downstream.
        raise ValueError(
            f"Mismatched lengths: {len(filenames)} filenames, "
            f"{len(embeddings)} embeddings, {len(captions)} captions."
        )

    return [
        ProcessedImage(
            position=i + 1,
            filename=filenames[i],
            embedding=embeddings[i],
            caption=captions[i],
        )
        for i in range(len(filenames))
    ]