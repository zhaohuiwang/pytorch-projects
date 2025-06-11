

"""
requests:
A synchronous HTTP client library for making HTTP requests (GET, POST, etc.) to APIs or websites.
Used for client-side operations, such as fetching data from external servers or testing local APIs.

"""

"""
## For single input data prediction
import requests

url = 'http://127.0.0.1:8000/predict'

# dummy data to test API
data = {
  "feature_X_1": 20,
  "feature_X_2": 10
}

# make a POST request to the API
response = requests.post(url, json=data)

#print response
print(response.json())
"""

## For batch input data prediction
import json
import numpy as np
import requests

url = 'http://127.0.0.1:8000/batch_predict'

# dummy data to test API - illustrating with three data points
batch_2d_list =  [[20, 10], [10, 5], [5,2]]

data = {
  "input_data": batch_2d_list
}
# make a POST request to the API
responses = requests.post(url, json=data)

# get the responses in json and text formats
responses_json = responses.json()
responses_text = json.dumps(responses.json(), indent=4)  # Pretty-print with indentation

#print response
# print(response_json["Model prediction"])
print(responses_text)

# Save response to a file
output_file = "data/model_demo/response_output.txt"

with open(output_file, "a+", encoding="utf-8") as file:
    file.write(str(data) + "\n" + str(responses_json) + "\n")

print(f"Response saved to {output_file}")