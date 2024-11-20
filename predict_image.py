import requests

url = "http://127.0.0.1:8000/predict"
file_path = "practice/27.jpg"

with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

print("Prediction:", response.text)

# try:
#     json_data = response.json()  # This is where the error occurs
#     print("JSON Response:", json_data)
# except requests.exceptions.JSONDecodeError:
#     print("Failed to decode JSON response. Raw response:", response.text)
