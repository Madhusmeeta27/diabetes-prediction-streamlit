import streamlit as st
from PIL import Image, ImageEnhance, ExifTags
import pytesseract
import cv2
import numpy as np
import Levenshtein

# Path to Tesseract OCR executable (adjust path as per your environment)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Function to calculate character-level accuracy
def calculate_char_accuracy(extracted_text, reference_text):
    if not reference_text:
        return 0.0
    correct_chars = sum(1 for a, b in zip(extracted_text, reference_text) if a == b)
    return (correct_chars / len(reference_text)) * 100


# Function to calculate word-level accuracy
def calculate_word_accuracy(extracted_text, reference_text):
    extracted_words = extracted_text.split()
    reference_words = reference_text.split()

    matching_words = sum(1 for a, b in zip(extracted_words, reference_words) if a == b)

    if len(reference_words) == 0:
        return 0.0
    return (matching_words / len(reference_words)) * 100


# Function to calculate Levenshtein Distance
def calculate_levenshtein_distance(extracted_text, reference_text):
    return Levenshtein.distance(extracted_text, reference_text)


# Function to extract text from image using Tesseract
def extract_text(image):
    img_array = np.array(image)

    # Convert to grayscale if necessary
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img_array

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(gray_image, config=custom_config)

    return text


# Detect if image is rotated
def is_rotated(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            if exif[orientation] == 3:
                return "Image is upside down."
            elif exif[orientation] == 6:
                return "Image is rotated 90 degrees clockwise."
            elif exif[orientation] == 8:
                return "Image is rotated 90 degrees counterclockwise."
        return "Image orientation is correct."
    except (AttributeError, KeyError, IndexError):
        return "Image orientation could not be determined."


# Detect if image is skewed
def is_skewed(image):
    img_array = np.array(image.convert('L'))
    edges = cv2.Canny(img_array, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        angles = []
        for line in lines:
            for rho, theta in line:
                angle = np.degrees(theta) - 90
                angles.append(angle)
        mean_angle = np.mean(angles)
        if abs(mean_angle) > 1:  # Assuming an image with a mean skew angle over 1 degree is skewed
            return f"Image is skewed by {mean_angle:.2f} degrees."
    return "Image is not skewed."


# Detect contrast issues
def has_contrast_issues(image):
    img_array = np.array(image.convert('L'))
    hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
    low_contrast = np.count_nonzero(hist[:5]) + np.count_nonzero(hist[-5:])
    if low_contrast > 0.1 * img_array.size:
        return "Image has low contrast."
    return "Image contrast is adequate."


# Detect noise in the image
def has_noise(image):
    img_array = np.array(image.convert('L'))
    laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
    if laplacian_var < 100:  # Arbitrary threshold for noise detection
        return "Image has noise."
    return "Image does not have significant noise."


# Detect unwanted lines in the image
def has_lines(image):
    img_array = np.array(image.convert('L'))
    edges = cv2.Canny(img_array, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        return "Image has unwanted lines."
    return "Image does not have unwanted lines."


# Streamlit App Setup
st.title("Text Extraction and Accuracy Metrics")

# Step 1: Enter Ground Truth
reference_text = st.text_area("Enter the correct text (Ground Truth):", height=150)

# Step 2: Upload Image
uploaded_file = st.file_uploader("Upload an image to extract text:", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Analyze the image for problems and display them in the sidebar
    st.sidebar.subheader("Detected Image Problems:")
    problems = []
    problems.append(is_rotated(image))
    problems.append(is_skewed(image))
    problems.append(has_contrast_issues(image))
    problems.append(has_noise(image))
    problems.append(has_lines(image))

    for problem in problems:
        st.sidebar.write(problem)

    # Step 3: Extract text from image
    if reference_text:
        extracted_text = extract_text(image)
        st.subheader("Extracted Text from Image:")
        st.text_area("Extracted Text", extracted_text, height=150)

        # Step 4: Calculate Accuracy Metrics
        st.subheader("Accuracy Metrics")

        # Character-Level Accuracy
        char_accuracy = calculate_char_accuracy(extracted_text, reference_text)
        st.write(f"Character-Level Accuracy: {char_accuracy:.2f}%")

        # Word-Level Accuracy
        word_accuracy = calculate_word_accuracy(extracted_text, reference_text)
        st.write(f"Word-Level Accuracy: {word_accuracy:.2f}%")

        # Levenshtein Distance
        levenshtein_distance = calculate_levenshtein_distance(extracted_text, reference_text)
        st.write(f"Levenshtein Distance: {levenshtein_distance} edits")
    else:
        st.write("Please provide the ground truth text to calculate accuracy.")
