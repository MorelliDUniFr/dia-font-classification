from PIL import Image, ImageOps
import os

# Fonts names
font_names = [
    "Algerian", "Arial", "Baskerville", "Calibry", "Calligraphy",
    "Cambria", "Comic Sans MS", "Courier", "Elephant", "Fascinate",
    "Georgia", "Helvetica", "Lucida Bright", "Nasalization", "Times New Roman"
]

# Input
input_folder = "fonts/fonts"

# Output
output_folder = "padded-grayscale/"

# Maximal dimensions of our images
final_width = 358
final_height = 32

# Creating the folder
for font_name in font_names:
    font_output_folder = os.path.join(output_folder, font_name)
    os.makedirs(font_output_folder, exist_ok=True)


for font_name in font_names:
    font_input_folder = os.path.join(input_folder, font_name)
    font_output_folder = os.path.join(output_folder, font_name)
    
    # Images files
    image_files = os.listdir(font_input_folder)
    
    # Grayscaling + White Padding
    for image_file in image_files:
        input_image_path = os.path.join(font_input_folder, image_file)
        
        image = Image.open(input_image_path)
        
        # grayscaling
        grayscale_image = image.convert("L")
        
        # Dimensions to pad
        pad_width = final_width - image.width
        pad_height = final_height - image.height
        
        # Margin to put the original image in the middle
        left_margin = pad_width // 2
        top_margin = pad_height // 2
        
        # Padding + Centralizing
        padded_image = ImageOps.expand(grayscale_image, border=(left_margin, top_margin, pad_width - left_margin, pad_height - top_margin), fill='white')
        
        # saving
        output_image_path = os.path.join(font_output_folder, image_file)
        padded_image.save(output_image_path)

print("Grayscale Padding has been added")
