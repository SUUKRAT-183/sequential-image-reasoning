"""
BLIP image captioner.

Wraps a pretrained BLIP model to generate a natural-language caption
for each image. Unlike CLIP (which gives us a numeric embedding),
BLIP's output is human-readable text — this is what will actually
get sent to the LLM as context in later milestones.
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from typing import List

from models.clip_encoder import get_device


# Using the "base" BLIP captioning model — smaller and faster than the
# "large" variant, which matters since we're already running CLIP
# alongside it and sharing your GPU's VRAM between both models.
MODEL_NAME = "Salesforce/blip-image-captioning-base"


class BlipCaptioner:
    """
    Loads a pretrained BLIP captioning model once and reuses it to
    caption images, same pattern as ClipEncoder.
    """

    def __init__(self):
        self.device = get_device()

        self.processor = BlipProcessor.from_pretrained(MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(self.device)

        # Same reasoning as CLIP: we're only doing inference, never
        # training, so eval() must be set to disable training-only
        # behavior (e.g. dropout).
        self.model.eval()

    def generate_caption(self, image: Image.Image) -> str:
        """
        Generate a single caption for one image.
        """
        # Preprocess the image into BLIP's expected input tensor format.
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # generate() runs BLIP's decoder to produce a sequence of
            # token IDs representing the caption.
            #
            # num_beams=5: instead of greedily picking the single most
            # likely next word at each step (which tends to loop into
            # repeated phrases when the model is uncertain), beam search
            # keeps track of the 5 most promising partial captions at
            # every step and expands all of them, then picks the best
            # complete caption at the end. This produces noticeably more
            # coherent, less repetitive output at the cost of a bit more
            # compute per image.
            #
            # repetition_penalty=1.5: directly discourages the model
            # from re-using words/phrases it has already generated in
            # this same caption, further reducing loops like
            # "the text's name and the text's name".
            #
            # max_new_tokens=30: caps caption length so it stays a
            # concise sentence rather than rambling on.
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=30,
                num_beams=5,
                repetition_penalty=1.5,
            )

        # Convert token IDs back into a readable string. skip_special_tokens
        # strips out BLIP's internal markers (like start/end-of-sequence
        # tokens) that aren't meant to be shown to a person.
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)

        return caption

    def generate_captions(self, images: List[Image.Image]) -> List[str]:
        """
        Caption a list of images, preserving order — same pattern as
        ClipEncoder.encode_images.
        """
        return [self.generate_caption(img) for img in images]