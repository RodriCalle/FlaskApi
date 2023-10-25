from flask import Flask, request, jsonify
from PIL import Image
from rembg import remove
from colorthief import ColorThief
import tensorflow as tf
import numpy as np
import cv2
import io

import json
import os
import openai
from craiyon import Craiyon

openai.organization = "org-yJpmbqbka0cb7xqz5uPDYmmq"
openai.api_key = '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'

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
    return f"{request['style']} {request['color']} {request['type']} for {request['sex']} in {request['temperature']} degrees Celsius"

def generate_promt_to_craiyon(request, outfit):
    prompt = f"Full length {request['sex']} wearing a "

    items = list(outfit.items())
    for index, (key, value) in enumerate(items):
        prompt += value
        if index < len(items) - 1:
            prompt += " and "
    prompt += "."
    return prompt

def generate_outfits(request_data):
    # generar outfits text
    prompt_chat_gpt = generate_promt_to_chat_gpt(request_data)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role": "system", 
           "content": '''You are an expert in fashion and clothing design expert, you will be provided with statements with specific data about an item of clothing, including the type of item, color, style, and what gender it is intended for, along with the temperature of the environment. Your task is to provide two sets of clothing (clothes only, no accessories) based on the main description of the clothing item. Clothing sets must be presented in JSON format, where each clothing set will be represented as a JSON object within an array.'''},
          {"role": "user", 
           "content": prompt_chat_gpt}
        ]
    )

    rpta = completion.choices[0].message
    # json_string = '[ {"top": "Casual Blue T-Shirt", "bottom": "Black Jeans", "footwear": "Sneakers"} ]'
    # print(rpta.content)

    outfits_array = json.loads(rpta.content)

    print(outfits_array)

    # generar outfits images
    generator = Craiyon()

    for outfit in outfits_array:
        prompt_to_craiyon = generate_promt_to_craiyon(request_data, outfit)
        print(prompt_to_craiyon)
        result = generator.generate(prompt= prompt_to_craiyon, negative_prompt="accessories", model_type="photo")
        print(result.images[0])
        outfit['image'] = result.images[0]
        outfit['craiyon_prompt'] = prompt_to_craiyon

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
    app.run(host='0.0.0.0', port=8080)