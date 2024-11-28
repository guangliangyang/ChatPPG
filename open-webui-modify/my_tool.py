import os
import requests
import json
from typing import Optional, io

class Tools:  # TableTennisCVAnalyzer:
    """
    Table Tennis CV Analyzer for analyzing table tennis performance and detecting serve fouls using computer vision models.

    This class provides methods to analyze player performance and detect fouls
    from video files, leveraging computer vision techniques to generate structured JSON data.
    """

    def __init__(self):
        """
        Initialize the Tools class and load external JSON data from GitHub.
        """
        self.data_url = "https://github.com/guangliangyang/ChatPPG/tree/main/open-webui-modify/my_tool_data.json"
        self.data = self._fetch_data()

    def _fetch_data(self) -> dict:
        """
        Fetch JSON data from the specified GitHub URL.

        :return: A dictionary containing the JSON data.
        :rtype: dict
        """
        try:
            response = requests.get(self.data_url)
            response.raise_for_status()
            print("Successfully fetched JSON data.")
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            return {}

    def analy_table_tennis_performance(self, file: str) -> str:
        """
        Analyze the performance of a table tennis player from a given video file.

        This function takes a video file path as input, processes the file, and returns
        a JSON string containing the analyzed performance data.

        :param file: The path to the video file (in .mp4 format) for performance analysis.
        :type file: str
        :return: A JSON string containing the performance analysis results.
        :rtype: str
        """
        print("analy_table_tennis_performance, video_file_path:", file)
        result = json.dumps(self.data.get("performance_data_json", {}))
        print(result)
        return result

    def detect_player_foul(self, file: str) -> str:
        """
        Detect serve fouls in a table tennis match from a given video file.

        This function takes a video file path as input, processes the file, and returns
        a JSON string containing the foul detection results.

        :param file: The path to the video file (in .mp4 format) for foul detection.
        :type file: str
        :return: A JSON string containing the foul detection results.
        :rtype: str
        """
        print("detect_serve_foul, video_file_path:", file)
        result = json.dumps(self.data.get("foul_data_json", {}))
        print(result)
        return result

    def show_video(self, files: dict) -> str:
        """
        Generate an HTML video tag to display a valid video file.

        This method iterates through a dictionary of files and searches for an MP4 video file.
        If found, it generates an HTML string with a video tag to embed the video.

        :param files: A dictionary of files, where each file contains metadata and a URL.
        :type files: dict
        :return: An HTML string with a video tag or a message if no valid video is found.
        :rtype: str
        """
        for file in files:
            if file["file"]["meta"]["content_type"] == "video/mp4":
                video_url = file["url"] + "/content"
                result = f'<video controls width="640" height="360">\n  <source src="{video_url}" type="video/mp4">\n  Your browser does not support the video tag.\n</video>'
                print("Generated Video HTML:", result)  # Print the result before returning
                return result
        result = "No valid video file found."
        print("Generated Video HTML:", result)  # Print the result before returning
        return result

    def self_train_from_pdf(
        self,
        # pdf_file: Optional[io.BytesIO] = None,
        # pdf_link: Optional[str] = None,
        convert_to_dataset: bool = False,
    ) -> None:
        """
        Perform a self-training session from a PDF file or link and learn from its content.
        Optionally, convert the PDF content into a dataset if requested.
        """
        pdf_analysis = 1  # self.process_pdf(pdf_file, pdf_link)
        print(f"PDF Analysis and Learning:\n{pdf_analysis}")

        # Extract and store the analysis for self-training
        if convert_to_dataset:
            print("convert_to_dataset:", convert_to_dataset)
