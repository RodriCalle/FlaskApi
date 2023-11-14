
import openai
import json

openai.organization = "org-yJpmbqbka0cb7xqz5uPDYmmq"
openai.api_key = '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'

quantity = "two"

completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        messages=[
          {"role": "system", 
           "content": f'''You are a fashion expert specializing in clothing design. You will receive descriptions of clothing items, including the type, color, style, and gender. Additionally, the description will include the temperature of the environment. Your task is to provide {quantity} outfits in JSON format. Each outfit should be represented as a JSON object within an array, with attributes for "top," "bottom," and "shoes." Ensure that each attribute is a string describing the corresponding clothing item.'''},
          {"role": "user", 
           "content": "Casual Blue T-Shirt for summer in 25 degrees Celsius"}
        ]
    )

rpta = completion.choices[0].message

outfits_array = json.loads(rpta.content)

# guardar array en un archivo json
with open('outfits.json', 'w') as json_file:
    json.dump(outfits_array, json_file)