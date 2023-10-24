from rembg import remove
from PIL import Image
input_path = 'negro_camisero.jpg'
output_path = 'res.png'
input = Image.open(input_path)
output = remove(input)
output.save(output_path)