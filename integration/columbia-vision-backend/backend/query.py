import requests

# response = requests.get("http://127.0.0.1:8000/predict")
response = requests.post("http://127.0.0.1:8000/predict", json={"img_string": "some string"})
print(response)