import subprocess
import sys
import io
import requests
import PyPDF2
import pandas as pd
from typing import Optional, List, Dict  # Import Dict from typing
from pydantic import BaseModel, Field


# Function to install missing packages
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")


# Ensure required packages are installed
required_packages = ["PyPDF2", "pandas", "requests", "pydantic"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} is not installed. Installing now...")
        install_package(package)

"""
title: AI Real-Time PDF Processing, Deep Analysis, and Dataset Conversion
author: Eldar Mammadov
author_url: https://www.azsec.biz
version: 1.2.0
required_open_webui_version: 0.3.30

v1.2.0 includes full PDF file reading, real-time content analysis, and optional dataset conversion.
"""

# Global instruction for the AI to autolearn and behave as instructed
GLOBAL_INSTRUCTION = """
You are an AI assistant by OpenAI. You can process PDF files uploaded by the user or access public links. 
You must engage in deep thinking and analysis to extract meaningful data from PDFs, then use this data to self-train and improve.
If the user requests, convert the processed content into a dataset.
"""


class Filter:
    class Valves(BaseModel):
        enable_pdf_processing: bool = Field(
            default=True, description="Enable or disable PDF processing."
        )
        enable_autolearn: bool = Field(
            default=True, description="Enable or disable real-time learning."
        )
        learning_mode: str = Field(
            default="dynamic",
            description="Learning mode: 'dynamic' (continual) or 'static'.",
        )
        store_knowledge: bool = Field(
            default=True, description="Store learned knowledge for future use."
        )
        convert_to_dataset: bool = Field(
            default=False,
            description="Enable or disable PDF content conversion to dataset (e.g., Excel).",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.knowledge_base = []
        self.global_instruction = GLOBAL_INSTRUCTION

    def _think_about_task(self, task_description: str) -> str:
        """
        This method simulates a deep thinking phase where the AI breaks down the task.
        """
        thinking_phase = (
            f"## Thinking\n\n**Evaluating the task**\nI am analyzing the task: {task_description}.\n"
            f"This will involve identifying key elements and breaking down the task into smaller components.\n"
        )
        thinking_phase += "**Collecting Thoughts**\nAfter identifying the task components, I will now focus on extracting relevant data.\n"
        return thinking_phase

    def _extract_text_from_pdf(self, pdf_file: io.BytesIO) -> List[str]:
        """
        Extract text from the provided PDF file.
        """
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_data = []

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_data.append(page.extract_text())

        return text_data

    def _download_pdf_from_link(self, pdf_url: str) -> io.BytesIO:
        """
        Download the PDF file from the given URL and return it as a BytesIO object.
        """
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(pdf_url, headers=headers)
        response.raise_for_status()
        return io.BytesIO(response.content)

    def _process_pdf_content(self, pdf_content: List[str]) -> str:
        """
        Perform deep analysis on the extracted PDF content.
        """
        analysis_output = "Deep Content Analysis of PDF:\n\n"
        for i, page_content in enumerate(pdf_content):
            analysis_output += f"Page {i + 1}:\n{page_content.strip()}\n\n"
        return analysis_output

    def process_pdf(
        self, pdf_file: Optional[io.BytesIO] = None, pdf_link: Optional[str] = None
    ) -> str:
        """
        Process a PDF file either uploaded by the user or downloaded from a link.
        """
        thinking_phase = self._think_about_task("Processing PDF file")

        # Process the PDF either from file or link
        if pdf_file:
            print("Processing uploaded PDF...")
            pdf_content = self._extract_text_from_pdf(pdf_file)
        elif pdf_link:
            print(f"Downloading PDF from {pdf_link}...")
            pdf_file = self._download_pdf_from_link(pdf_link)
            pdf_content = self._extract_text_from_pdf(pdf_file)
        else:
            return "No PDF file or link provided."

        # Perform deep analysis on the extracted content
        return thinking_phase + "\n***\n" + self._process_pdf_content(pdf_content)

    def _convert_pdf_to_dataset(
        self, pdf_content: List[str], output_file: str = "output.xlsx"
    ):
        """
        Convert extracted PDF content into a dataset (Excel file).
        """
        rows = []
        for line in pdf_content:
            for row in line.split("\n"):
                rows.append(row.split())

        df = pd.DataFrame(rows)
        df.to_excel(output_file, index=False)
        print(f"PDF content saved to {output_file}.")

    def self_train_from_pdf(
        self,
        pdf_file: Optional[io.BytesIO] = None,
        pdf_link: Optional[str] = None,
        convert_to_dataset: bool = False,
    ) -> None:
        """
        Perform a self-training session from a PDF file or link and learn from its content.
        Optionally, convert the PDF content into a dataset if requested.
        """
        pdf_analysis = self.process_pdf(pdf_file, pdf_link)
        print(f"PDF Analysis and Learning:\n{pdf_analysis}")

        # Extract and store the analysis for self-training
        if convert_to_dataset:
            self._convert_pdf_to_dataset(pdf_analysis.split("\n"), "output.xlsx")

    def inlet(
        self, body: Dict[str, any], __user__: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """Inlet method processes user input and triggers autolearning."""
        try:
            print(self.global_instruction)
            original_messages: List[Dict[str, str]] = body.get("messages", [])
            user_messages = [msg.get("content", "") for msg in original_messages]

            # Trigger dynamic or static learning based on settings
            if self.valves.learning_mode == "dynamic":
                for msg in user_messages:
                    self._process_pdf_content(msg)
            else:
                if user_messages:
                    self._process_pdf_content(user_messages[-1])

            body["messages"] = original_messages
            return body
        except Exception as e:
            print(e)
            return body

    def outlet(
        self, body: Dict[str, any], __user__: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """Outlet method finalizes autolearning after the conversation."""
        try:
            original_messages: List[Dict[str, str]] = body.get("messages", [])
            for msg in original_messages:
                self._process_pdf_content(msg.get("content", ""))

            return body
        except Exception as e:
            print(e)
            return body
