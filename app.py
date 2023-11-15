from flask import Flask, request, jsonify
from PIL import Image
from rembg import remove
from colorthief import ColorThief
import tensorflow as tf
import numpy as np
import cv2
import io

import requests
import datetime
import pybase64
import time
import jwt

import json
import os
from openai import OpenAI
# from craiyon import Craiyon

OPENAI_API_KEY="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"

app = Flask(__name__)

def remove_background(image):
    image = Image.open(image)
    image = remove(image)
    return image
    
def predict_cloth_type(image, main_path='converted_tflite_cloth_classification'):
    model_tflite = f'{main_path}/model_unquant.tflite'

    with open(f"{main_path}/labels.txt", "r") as f:
        labels = f.read().splitlines()

    interpreter = tf.lite.Interpreter(model_path=model_tflite)
    interpreter.allocate_tensors()

    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)

    input_image = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), -1)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (224, 224)) 
    input_image = input_image.astype(np.float32) 
    input_image /= 255.0 

    input_image = np.expand_dims(input_image, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_image)

    interpreter.invoke()

    results = interpreter.get_tensor(output_details[0]['index'])

    return labels[np.argmax(results[0])], str(np.max(results[0] * 100))

def get_colors(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)

    color_thief = ColorThief(buffer)
    palette = color_thief.get_palette(color_count = 3, quality = 1)
    return {'palette': palette }

def generate_promt_to_chat_gpt(request):
    return f"{request['sex']} {request['style']} {request['color']} {request['type']} for {request['season']} in {request['temperature']} Celsius degrees"

def generate_promt_for_image_generation(request, outfit):
    prompt = f"Fashion image of a full length {request['sex']} smiling and wearing {outfit} taken from far away."
    return prompt

def generate_text_with_outfit(outfit):
    outfit_text = ""
    items = list(outfit.items())
    for index, (key, value) in enumerate(items):
        if isinstance(value, str):
            outfit_text += value
            if index < len(items) - 1:
                outfit_text += " and "
    return outfit_text

def generate_outfits(request_data):
    # generar outfits text
    prompt_chat_gpt = generate_promt_to_chat_gpt(request_data)
    quantity = "two"

    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        messages=[
            {
                "role": "system", 
                # "content": f'''You are a fashion expert specializing in clothing design. You will receive descriptions of clothing items, including the type, color, style, and gender. Additionally, the description will include the temperature of the environment. Your task is to provide {quantity} outfits in a JSON array. Each outfit should be represented as a JSON ARRAY, with attributes for "top," "bottom," and "shoes." Ensure that each attribute is a string describing the corresponding clothing item.'''},
                "content": f'''You are a fashion expert specialized in clothing design. You will receive a description of an item of clothing, including type, color, style, and gender. Additionally, the description includes the ambient temperature. Your task is to provide {quantity} sets of clothing that include the item and fit the main description. Each set should be represented as a JSON object, with attributes for "top", "bottom", and "shoes". Make sure each attribute is a string that describes the corresponding item of clothing in the format of: style color item of clothing. The response must be a JSON object with an outfits element that is an array of the generated objects.'''},
            {
                "role": "user", 
                "content": prompt_chat_gpt
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    rpta = completion.choices[0].message
    # json_string = '[ {"top": "Casual Blue T-Shirt", "bottom": "Black Jeans", "footwear": "Sneakers"} ]'
    # print(rpta.content)

    outfits_array = json.loads(rpta.content)['outfits']
    # print(outfits_array.outfits)

    outfits_array = generate_images_from_prompt(request_data, outfits_array)

    return outfits_array

def generate_images_from_prompt(request_data, outfits_array):
    FILE_NAME = "fashionapp-405020-cdf392b49002.json"
    API_URL = "https://us-central1-aiplatform.googleapis.com/"

    with open(FILE_NAME, "r") as file:
        data = json.load(file)

    PROJECT_ID = data['project_id']

    url = f"{API_URL}v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/imagegeneration:predict"

    iat = time.time()
    exp = iat + 3600
    payload = {'iss': data['client_email'],
               'sub': data['client_email'],
               'aud': API_URL,
               'iat': iat,
               'exp': exp}
    additional_headers = {'kid': data['private_key_id']}
    signed_jwt = jwt.encode(payload, data['private_key'], headers=additional_headers, algorithm='RS256')

    # print("TOKEN")
    # print(signed_jwt)

    headers = {
        "Authorization": f"Bearer {signed_jwt}",
        "Content-Type": "application/json"
    }

    for outfit in outfits_array:
        outfit_text = generate_text_with_outfit(outfit)
        prompt = generate_promt_for_image_generation(request_data, outfit_text)

        outfit['id'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '-' + str(outfits_array.index(outfit))
        outfit['name'] = outfit_text
        outfit['prompt'] = prompt

        data = {
          "instances": [
            {
              "prompt": prompt
            }
          ],
          "parameters": {
            "sampleCount": 1
          }
        }

        try:
            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()

                for prediction in response_data['predictions']:
                    imageb64 = prediction['bytesBase64Encoded']
                    outfit['image'] = imageb64
            else:
                outfit['image'] = ""
                print(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            print("Ocurrio una excepción", e)

    outfits_array = list(filter(lambda outfit: 'image' in outfit, outfits_array))
    return outfits_array 

@app.route('/closet', methods=['POST'])
def get_cloths():
    request_data = request.get_json()

    if request_data is None:
        return jsonify({'error': 'Solicitud no contiene datos JSON'}), 400
    
    outfits = generate_outfits(request_data)
    return jsonify({'outfits': outfits})

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image_without_background = remove_background(image)

    #image_without_background.show()

    cloth_type, probability_type = predict_cloth_type(image_without_background, main_path='converted_tflite_cloth_type')
    cloth_season, probability_season = predict_cloth_type(image_without_background, main_path='converted_tflite_cloth_season')
    cloth_style, probability_style = predict_cloth_type(image_without_background, main_path='converted_tflite_cloth_style')
    colors = get_colors(image_without_background)
    return jsonify(
        {
        'type': { 'name': cloth_type, 'probability': probability_type }, 
        'season': { 'name': cloth_season, 'probability': probability_season }, 
        'style': { 'name': cloth_style, 'probability': probability_style }, 
        'colors': colors
        },
        )


@app.route('/', methods=['GET']) 
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
        app.run(host='0.0.0.0')