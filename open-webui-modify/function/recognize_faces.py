"""
title: Compreface - Face Recognize
author: EaV Solution
version: 0.2
"""

from pydantic import BaseModel
from typing import Optional

import base64
import io
from io import BytesIO
import requests
from PIL import Image

ADDRESS = "http://192.168.1.50:8000"
API_KEY = "e464ac2e-3382-42f7-b419-db11a12d3c44"


def get_subject_if_ready(data, thresh_score=0.98):
    subjects = []
    if data.get("result"):
        for res in data["result"]:
            if res.get("subjects"):
                for subject_info in res["subjects"]:
                    if subject_info["similarity"] > thresh_score:
                        subjects.append(subject_info["subject"])
    return subjects if subjects else []


def recognize_face(image):
    url = f"{ADDRESS}/api/v1/recognition/recognize"

    # Params
    params = {
        "limit": "0",
        "prediction_count": "1",
        "det_prob_threshold": "0.2",
        "face_plugins": "",
        "status": "true",
        "detect_faces": "true",
    }

    # Headers for the POST request
    headers = {"x-api-key": API_KEY}

    # Save the image to bytes
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    files = {"file": ("image.jpg", image_bytes)}

    response = requests.post(url, params=params, headers=headers, files=files)

    # Checking the response
    if response.status_code == 200:
        data = response.json()
        print("data=", data)
        subjects = get_subject_if_ready(data)
        return subjects
    else:
        print(response.text)
        return "Error"


def process_images_in_messages(messages):
    for message in messages[::-1]:
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

                    # Convert image to RGB if it's in RGBA mode
                    if image.mode == "RGBA":
                        image = image.convert("RGB")

                    # Now, perform face recognition on the image
                    face_names = recognize_face(image)
                    print("face_names=", face_names)

                    # Add string "Face recognize {face_name}" to message content
                    # Assuming message["content"] is a list of items, we can append a new text item
                    if face_names:
                        message["content"].append(
                            {
                                "type": "text",
                                "text": f"The recognized faces are: {face_names}",
                            }
                        )
                    return messages
    return messages


class Filter:
    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        pass

    def __init__(self):
        # Initialize 'valves' with specific configurations.
        self.valves = self.Valves()

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        messages = body["messages"]
        messages = process_images_in_messages(messages)
        # for message in messages:
        #     print("message=", message)
        body["messages"] = messages
        return body
