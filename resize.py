from PIL import Image
import os
from tqdm import tqdm

path =(r"C:\Research\Dataset\new_images")
new_path = r"C:\Research\Dataset\resized_images"
dirs = os.listdir(path)

def resize(width, height):
	for item in tqdm(dirs):
		image_path = path + "\\" + item
		if os.path.isfile(image_path):
			img = Image.open(image_path)
			new_image = img.resize((width, height))
			new_image.save(new_path + "\\" + item)
			
resize(224, 224)