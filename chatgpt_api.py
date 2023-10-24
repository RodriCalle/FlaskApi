import os
import openai
openai.organization = "org-yJpmbqbka0cb7xqz5uPDYmmq"
openai.api_key = "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"
openai.Model.list()

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", 
     "content": "You are an expert in fashion and clothing design. You will be provided statements with data specific to an item of clothing (type of item of clothing, color, style, for which gender) and the temperature of the environment. Your task is to provide sets of clothing (clothing only, accessories not included) based on the main description of the clothing item for later use in DALLE."},
    {"role": "user", 
     "content": "Casual Blue T-Shirt for Man in 23 degrees Celsius."}
  ]
)

print(completion.choices[0].message.content)