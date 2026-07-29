"""
LLM client abstraction.

Defines a common interface (LLMClient) so the rest of the app never
depends on which specific provider (Gemini, OpenAI, etc.) is being
used. GeminiClient is the first concrete implementation, built on
Google's current "Gen AI SDK" (google-genai package) — the officially
supported successor to the older google-generativeai package.
"""

import os
from abc import ABC, abstractmethod
from typing import List
from PIL import Image

from google import genai
from dotenv import load_dotenv

load_dotenv()


class LLMClient(ABC):
    """
    Abstract base class every LLM provider implementation must follow.
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

    Sends the ordered caption context AND the raw images together, so
    Gemini can reason using both BLIP's captions and its own direct
    vision understanding of each image.
    """

    # Flash is the current free-tier-eligible model family. Using the
    # versioned name here rather than an alias, since alias behavior
    # has changed across SDK versions — safer to be explicit.
    MODEL_NAME = "gemini-3.5-flash"

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Make sure you created a .env "
                "file with GEMINI_API_KEY=your_key in the project root."
            )

        # The new SDK uses a Client object rather than a global
        # configure() call — a cleaner pattern that avoids hidden
        # global state.
        self.client = genai.Client(api_key=api_key)

    def generate_answer(
        self,
        context_text: str,
        images: List[Image.Image],
        question: str,
    ) -> str:
        # The new SDK accepts a list mixing strings and PIL Image
        # objects directly in `contents`, same convenience as before.
        contents = [
            "The following images are provided in chronological order.",
            context_text,
        ]

        for i, img in enumerate(images):
            contents.append(f"Image {i + 1}:")
            contents.append(img)

        contents.append(f"\nQuestion: {question}")

        response = self.client.models.generate_content(
            model=self.MODEL_NAME,
            contents=contents,
        )

        return response.text