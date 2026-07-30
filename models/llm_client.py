"""
LLM client abstraction.

Defines a common interface (LLMClient) so the rest of the app never
depends on which specific provider (Gemini, OpenAI, etc.) is being
used. GeminiClient is the first concrete implementation, built on
Google's current Gen AI SDK (google-genai package).
"""

import os
from abc import ABC, abstractmethod
from typing import List
from PIL import Image

from google import genai
from dotenv import load_dotenv

load_dotenv()


def _resize_for_llm(image: Image.Image, max_dimension: int = 1024) -> Image.Image:
    """
    Downscale an image so its longest side is at most max_dimension
    pixels, preserving aspect ratio.

    This significantly reduces upload payload size for large images
    (especially screenshots, which can be several MB each) without
    meaningfully hurting Gemini's ability to read text/UI content —
    Gemini's vision encoder internally downsamples images anyway, so
    sending ultra-high-resolution originals wastes bandwidth without
    adding real understanding, and increases the risk of connection
    errors when sending many images in one request.
    """
    width, height = image.size
    if max(width, height) <= max_dimension:
        return image  # already small enough, no resizing needed

    scale = max_dimension / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


class LLMClient(ABC):
    """
    Abstract base class every LLM provider implementation must follow.

    Using an ABC here means Python enforces the contract at the
    language level — any subclass that forgets to implement
    generate_answer will fail immediately, not silently at runtime.
    """

    @abstractmethod
    def generate_answer(
        self,
        context_text: str,
        images: List[Image.Image],
        question: str,
    ) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    """
    Gemini implementation of LLMClient, using Google's current Gen AI SDK.

    Sends the ordered caption context AND the raw (resized) images
    together, so Gemini can reason using both BLIP's captions and its
    own direct vision understanding of each image — important for
    image types like screenshots where BLIP's captions are weak but
    Gemini can still read/interpret the actual pixels.
    """

    MODEL_NAME = "gemini-3.5-flash"

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Make sure you created a .env "
                "file with GEMINI_API_KEY=your_key in the project root."
            )

        self.client = genai.Client(api_key=api_key)

    def generate_answer(
        self,
        context_text: str,
        images: List[Image.Image],
        question: str,
    ) -> str:
        contents = [
            "The following images are provided in chronological order.",
            context_text,
        ]

        for i, img in enumerate(images):
            contents.append(f"Image {i + 1}:")
            contents.append(_resize_for_llm(img))

        contents.append(f"\nQuestion: {question}")

        response = self.client.models.generate_content(
            model=self.MODEL_NAME,
            contents=contents,
        )

        return response.text