"""
Converts an ordered sequence of ProcessedImage objects into a
human-readable text block — the format we'll later hand to the LLM
as context.
"""

from pipeline.image_processor import ProcessedImage


def build_sequence_context(processed_images: list[ProcessedImage]) -> str:
    """
    Build a text block describing the ordered sequence, e.g.:

        IMAGE 1:
        A man approaches a car.

        IMAGE 2:
        The man opens the driver's door.

    This text is built purely from captions and position numbers —
    embeddings are NOT included here. Sending thousands of raw
    floating-point numbers to an LLM as text would waste enormous
    amounts of context space and give the LLM nothing meaningful to
    reason about; captions are the human-readable signal, embeddings
    stay internal to our own similarity math.
    """
    lines = []
    for img in processed_images:
        lines.append(f"IMAGE {img.position}:")
        lines.append(img.caption)
        lines.append("")  # blank line between entries for readability

    return "\n".join(lines).strip()