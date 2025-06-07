

import requests

url = 'http://127.0.0.1:8000/predict'

#dummy data to test API
data = {
  "feature_X_1": 5,
  "feature_X_2": 5
}

#make a POST request to the API
response = requests.post(url, json=data)

#print response
response.json()