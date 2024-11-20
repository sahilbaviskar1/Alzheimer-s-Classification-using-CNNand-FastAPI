import requests

response = requests.post("http://127.0.0.1:8000/train")
print(response.json())
