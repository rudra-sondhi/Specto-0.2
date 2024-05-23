from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    """
    Extracts text from an image file using Tesseract OCR.

    Args:
    image_path (str): The path to the image file from which to extract text.

    Returns:
    str: The extracted text.
    """
    # Load the image from the path
    img = Image.open(image_path)

    # Set the path to the tesseract executable if you are on Windows
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(img)

    return text

# Example usage:
if __name__ == "__main__":
    image_path = '/Users/rudrasondhi/Downloads/Screenshot 2024-05-01 at 7.38.31 PM.jpeg'  # Replace with your image path
    extracted_text = extract_text_from_image(image_path)
    print("Extracted Text:")
    print(extracted_text)
