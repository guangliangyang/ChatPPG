import os
import requests
from datetime import datetime
import json

performance_data_json = {
    "Speed Statistics": {
        "Depth": {"Max (km/h)": 0.06, "Average (km/h)": 0.02},
        "Forward": {"Max (km/h)": 0.02, "Average (km/h)": 0.01},
        "Overall": {"Max (km/h)": 0.06, "Average (km/h)": 0.03},
        "Sideways": {"Max (km/h)": 0.03, "Average (km/h)": 0.01},
    },
    "Calories Burned": {
        "Total Exercise Duration": "00:00:48",
        "Total Calories Burned (kcal)": 3.7,
        "Average Calories Burned per Hour (kcal)": 276.5,
        "Intensity": "Entertainment",
    },
    "Covered Area (m²)": 2.25,
    "Cell Ratios (%)": {
        "R22": 0.0, "R12": 0.0, "R02": 0.0, "R21": 0.0, "R11": 0.0, "R01": 0.0,
        "R20": 0.0, "R10": 0.0, "R00": 0.0, "L20": 0.0, "L10": 0.0, "L00": 0.0,
        "L21": 0.0, "L11": 17.2, "L01": 82.7, "L22": 0.1, "L12": 0.0, "L02": 0.0,
    },
    "Stroke": {
        "Total Count": 62,
        "Templates": {
            "Forehand-Backspin": {"Count": 3, "Percentage (%)": 4.8},
            "Forehand-Topspin": {"Count": 0, "Percentage (%)": 0.0},
            "Backhand-Backspin": {"Count": 4, "Percentage (%)": 6.5},
            "Backhand-Topspin": {"Count": 55, "Percentage (%)": 88.7},
        },
    },
    "Footwork": {
        "Total Count": 133,
        "Templates": {
            "Little-Step": {"Count": 66, "Percentage (%)": 49.6},
            "Striding-Step": {"Count": 67, "Percentage (%)": 50.4},
            "Cross-Step": {"Count": 0, "Percentage (%)": 0.0},
        },
    },
}

foul_data_json1 = {
    "Foul Serves/Total Serve Actions": {"Foul Serves": 4, "Total Serve Actions": 20},
    "Foul Statistics": {
        "In Front of the End Line": 2,
        "Beyond the Sideline Extension": 0,
        "Tossed from Below Table Surface": 0,
        "Backward Angle More Than 30°": 3,
        "Tossed Upward Less Than 16 cm": 2,
    },
}

foul_data_json = {
    "Summary of Serves": {
        "Total Serves": 20,
        "Foul Serves": 4
    },
    "Details of Foul Statistics": {
        "Serves in Front of the End Line": 2,
        "Serves Beyond the Sideline Extension": 0,
        "Serves Tossed from Below the Table Surface": 0,
        "Serves with Backward Angle Greater Than 30 Degrees": 3,
        "Serves Tossed Upward Less Than 16 cm": 2
    }
}



class Tools:
    def __init__(self):
        pass

    def analy_table_tennis_performance(self, file: str) -> str:
        print("analy_table_tennis_performance,video_file_path:", file)
        result = json.dumps(performance_data_json)
        print(result)
        return result

    def detect_player_foul(self, file: str):
        print("detect_serve_foul,video_file_path:", file)
        result = json.dumps(foul_data_json)
        print(result)
        return result
