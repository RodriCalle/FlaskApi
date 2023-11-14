
import  openai
openai.organization = "org-yJpmbqbka0cb7xqz5uPDYmmq"
openai.api_key = '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'

res = openai.Image.create(
    model="dall-e-3",
    prompt="a white siamese cat",
    size="1024x1024",
    quality="standard",
    n=1
)


print(res)

image_url = res.data[0].url