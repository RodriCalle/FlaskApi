import requests
import datetime
import pybase64
import time
import jwt
import json

PROJECT_ID = "fashionapp-405020"
FILE_NAME = "fashionapp-405020-cdf392b49002.json"

API_URL = "https://us-central1-aiplatform.googleapis.com/"

url = f"{API_URL}v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/imagegeneration:predict"

with open(FILE_NAME, "r") as file:
    data = json.load(file)

iat = time.time()
exp = iat + 3600
payload = {'iss': data['client_email'],
           'sub': data['client_email'],
           'aud': API_URL,
           'iat': iat,
           'exp': exp}
additional_headers = {'kid': data['private_key_id']}
signed_jwt = jwt.encode(payload, data['private_key'], headers=additional_headers, algorithm='RS256')

headers = {
    "Authorization": f"Bearer {signed_jwt}",
    "Content-Type": "application/json"
}

data = {
  "instances": [
    {
      "prompt": "Fashion image of a full length man smiling and dressing Casual Blue T-Shirt and Dark Wash Denim Jeans and Black Canvas Sneakers taken from far away"
    }
  ],
  "parameters": {
    "sampleCount": 1
  }
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print("Solicitud exitosa")
    response_data = response.json()

    print(response_data)

    for prediction in response_data['predictions']:
        imageb64 = prediction['bytesBase64Encoded']
        image_data = pybase64.b64decode(imageb64)

        mimeType = prediction['mimeType']

        output_path = 'images/' + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + '.' + mimeType.split('/')[1]

        with open(output_path, "wb") as img_file:
            img_file.write(image_data)
else:
    print(f"Error {response.status_code}: {response.text}")
