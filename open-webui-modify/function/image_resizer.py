"""
title: Image Resizer
author: EaV Solution
version: 0.1
"""

from pydantic import BaseModel, Field
from typing import Optional

import base64
from PIL import Image
import io


def resize_images_in_messages(messages, max_dimension=768):
    for message in messages:
        for item in message.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image_url":
                image_data_url = item["image_url"]["url"]
                # Check if it starts with 'data:image/'
                if image_data_url.startswith("data:image/"):
                    # Get the base64 data
                    header, encoded = image_data_url.split(",", 1)
                    # Get image format from header
                    image_mime_type = header.split(":")[1].split(";")[0]
                    image_format = image_mime_type.split("/")[1].upper()
                    # Decode the base64 data
                    image_data = base64.b64decode(encoded)
                    # Open image with PIL
                    image = Image.open(io.BytesIO(image_data))
                    # Get the original size
                    width, height = image.size
                    # Compute the scaling factor
                    if max(width, height) > max_dimension:
                        scaling_factor = max_dimension / max(width, height)
                        new_width = int(width * scaling_factor)
                        new_height = int(height * scaling_factor)
                        # Resize the image
                        image = image.resize((new_width, new_height), Image.LANCZOS)
                        # Save image back to base64
                        buffered = io.BytesIO()
                        image.save(buffered, format=image_format)
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        # Reconstruct data URL
                        new_data_url = f"data:{image_mime_type};base64,{img_str}"
                        # Update the item with the resized image
                        item["image_url"]["url"] = new_data_url
    return messages


class Filter:
    class Valves(BaseModel):
        max_dimension: int = Field(default=768, description="Maximum image dimension.")
        pass

    class UserValves(BaseModel):
        pass

    def __init__(self):
        # Indicates custom file handling logic. This flag helps disengage default routines in favor of custom
        # implementations, informing the WebUI to defer file-related operations to designated methods within this class.
        # Alternatively, you can remove the files directly from the body in from the inlet hook
        # self.file_handler = True

        # Initialize 'valves' with specific configurations. Using 'Valves' instance helps encapsulate settings,
        # which ensures settings are managed cohesively and not confused with operational flags like 'file_handler'.
        self.valves = self.Valves()
        pass

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        messages = body["messages"]
        messages = resize_images_in_messages(messages, self.valves.max_dimension)
        body["messages"] = messages
        return body
