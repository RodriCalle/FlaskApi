
import  openai
openai.organization = ""
openai.api_key = ""

res = openai.Image.create(
    model="dall-e-3",
    prompt="a white siamese cat",
    size="1024x1024",
    quality="standard",
    n=1
)

print(res)

image_url = res.data[0].url