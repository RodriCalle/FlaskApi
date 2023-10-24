from craiyon import Craiyon

generator = Craiyon() # Instantiates the api wrapper
promt = "Full length man wearing a blue casual t-shirt, light wash jeans and white sneakers"
result = generator.generate(prompt= promt, negative_prompt="spoon", model_type="photo")
# result.save_images()

# print(result.images) # Prints a list of the Direct Image URLs hosted on https://img.craiyon.com

# Loops through the list and prints each image URL one by one
for url in result.images:
    print(url)