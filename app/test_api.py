# test_api.py
import requests

sample_transaction = {
    "Time": 0.0,
    "V1": -1.35, "V2": -0.07, "V3": 2.53,
    "V4": 1.37, "V5": -0.33, "V6": 0.46,
    "V7": 0.23, "V8": 0.09, "V9": 0.36,
    "V10": 0.09, "V11": -0.55, "V12": -0.61,
    "V13": -0.99, "V14": -0.31, "V15": 1.46,
    "V16": -0.47, "V17": 0.20, "V18": 0.02,
    "V19": 0.40, "V20": 0.25, "V21": -0.01,
    "V22": 0.27, "V23": -0.11, "V24": 0.06,
    "V25": 0.12, "V26": -0.18, "V27": 0.13,
    "V28": -0.02, "Amount": 149.62
}

response = requests.post(
    "http://localhost:8000/predict",
    json=sample_transaction
)
print(response.json())