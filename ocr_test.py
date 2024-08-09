from PIL import Image
import pytesseract

# Load images
image1 = Image.open("image1.png")
image2 = Image.open("image2.png")

# Extract text
text1 = pytesseract.image_to_string(image1)
text2 = pytesseract.image_to_string(image2)
print(text1)
print("--")
print(text2)
