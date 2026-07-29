"""
Cosine similarity utilities for comparing CLIP embeddings.
"""

import torch
from typing import List


def compute_similarity_matrix(embeddings: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between all embeddings.

    Returns an (N, N) tensor where entry [i][j] is the similarity
    between image i and image j. Diagonal entries are always 1.0
    (an image is perfectly similar to itself).

    Because our CLIP embeddings are already L2-normalized (we did this
    in clip_encoder.py), cosine similarity between two normalized
    vectors is simply their dot product — no extra normalization
    needed here. This is why normalizing at embedding-generation time
    saves work later.
    """
    # Stack the list of 1D (512,) tensors into a single 2D tensor of
    # shape (N, 512) — N images, 512-dim embeddings each.
    stacked = torch.stack(embeddings)

    # Matrix multiply the stack by its own transpose:
    # (N, 512) @ (512, N) -> (N, N)
    # Each entry [i][j] in the result is the dot product of embedding i
    # and embedding j — which, since both are unit-length, equals their
    # cosine similarity.
    similarity_matrix = stacked @ stacked.T

    return similarity_matrix