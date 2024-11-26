"""
title: Llama Index Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description:
"""

from typing import List, Union, Generator, Iterator


class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):

        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        print(f"Received body: {body}")
        files = body.get("files", [])
        for file in files:
            content_url = file["url"] + "/content"
            print(f"file available at {content_url}")
            # read the file content as binary and do something ...
        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print("user_message:", user_message)
        print("model_id:", model_id)
        print("messages:", messages)
        print("body:", body)

        # query_engine = self.index.as_query_engine(streaming=True)
        # response = query_engine.query(user_message)
        #
        # return response.response_gen
        return user_message
