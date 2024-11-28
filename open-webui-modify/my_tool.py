import os
import requests
from datetime import datetime



class Tools:
    def __init__(self):
        pass

    # Add your custom tools using pure Python code here, make sure to add type hints
    # Use Sphinx-style docstrings to document your tools, they will be used for generating tools specifications
    # Please refer to function_calling_filter_pipeline.py file from pipelines project for an example

    def analy_table_tennis_perfomance(self, file_path: str) -> str:
        print("file_path:", str)
        result = " Speed Statistics : 3m per second ,Calories Burned:400 kcal, Average Calories Burned per Hour: 600 kcal,   Covered Area: 15 m square - Swing Count: 200 - Footwork: 4000"

        return result

    def detect_serve_foul(self, file_path: str):
        result = ("Tossed from Below Table Surface:1 time, occur at 1:50; "
                  "In Front of the End Line: 1 time, occur at 2:50; "
                  "Beyond the sideline extension 1 time, occur at 3:50; "
                  "Backward Angle More Than 30°: 1 time, occur at 4:50; "
                  "Tossed Upward Less Than 16 cm: 0 time")

        return result