"""
LLM client abstraction.

Defines a common interface (LLMClient) so the rest of the app never
depends on which specific provider (Gemini, OpenAI, etc.) is being
used. GeminiClient is the first concrete implementation, built on
Google's current Gen AI SDK (google-genai package).
"""

import os
import time
from abc import ABC, abstractmethod
from typing import List
from PIL import Image

from google import genai
from google.genai import errors as genai_errors
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
    """

    # Google's Flash models get renamed/deprecated fairly often. If the
    # primary model is temporarily overloaded (503) or unavailable, we
    # fall back to trying the next name in this list before giving up.
    MODEL_CANDIDATES = [
        "gemini-3.5-flash",
        "gemini-flash-latest",
        "gemini-3.1-flash-lite",
    ]

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 5

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
            contents.append(img)

        contents.append(f"\nQuestion: {question}")

        last_error = None

        # Try each candidate model in order. This handles both:
        # (a) a specific model name being deprecated/renamed, and
        # (b) a specific model being temporarily overloaded (503) while
        #     another equivalent model is available.
        for model_name in self.MODEL_CANDIDATES:
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                    )
                    return response.text

                except genai_errors.ServerError as e:
                    # 503 = temporarily overloaded. Worth retrying the
                    # SAME model a couple of times before moving to the
                    # next candidate, since it may free up quickly.
                    last_error = e
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAY_SECONDS)
                    continue

                except genai_errors.ClientError as e:
                    # 404 = model doesn't exist/deprecated. No point
                    # retrying the same model — move straight to the
                    # next candidate in the list.
                    last_error = e
                    break

        # If we've exhausted every candidate model and every retry,
        # surface the last error so it's visible in the Streamlit UI
        # rather than failing silently.
        raise RuntimeError(
            f"All Gemini model candidates failed. Last error: {last_error}"
        )