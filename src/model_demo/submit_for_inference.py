
## For single input data prediction
import numpy as np
import requests

url = 'http://127.0.0.1:8000/predict'

#dummy data to test API
data = {
  "feature_X_1": 5,
  "feature_X_2": 5
}

# make a POST request to the API
response = requests.post(url, json=data)

#print response
response.json()


## For batch input data prediction
import numpy as np
import requests

url = 'http://127.0.0.1:8000/batch_predict'

#dummy data to test API
batch_2d_list =  [[20, 10], [10, 5], [5,2]]

data = {
  "input_array": batch_2d_list
}
# make a POST request to the API
responses = requests.post(url, json=data)

#print response
responses.json()
