# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import os
import re


# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r"\W+", " ", text)
    return text


# Function to extract text from an image using pytesseract
def extract_text_from_image(image):
    import pytesseract

    return pytesseract.image_to_string(image)


def get_token_dict(tokens):
    from collections import Counter

    # Create a dictionary of token frequencies
    return Counter(tokens)


def jaccard_similarity(dict1, dict2):
    # Convert the dictionaries to sets of tokens
    set1 = set(dict1.keys())
    set2 = set(dict2.keys())

    # Calculate Jaccard similarity
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    # Handle the case where the union is 0
    if union == 0:
        return 0.0

    return intersection / union


def tokenize_no_whitespace(text):
    import re

    # Remove all whitespace characters (spaces, tabs, newlines, etc.)
    text_no_whitespace = re.sub(r"\s+", "", text.lower())
    # Tokenize by individual characters
    tokens = list(text_no_whitespace)
    return tokens


def slides_are_the_same(text1, text2):
    threshold = 0.9

    # Tokenizing by character
    tokens1_char = tokenize_no_whitespace(text1)
    tokens2_char = tokenize_no_whitespace(text2)

    dict1_char = get_token_dict(tokens1_char)
    dict2_char = get_token_dict(tokens2_char)

    # Calculating Jaccard similarity for character tokens
    jaccard_sim_char = jaccard_similarity(dict1_char, dict2_char)

    if jaccard_sim_char >= threshold:
        return True

    return False


def remove_duplicates_preserve_order(input_list):
    result = []

    prev_item = ""
    for index, slide_obj in enumerate(input_list):
        if slides_are_the_same(slide_obj["slide_text"].lower(), prev_item.lower()):
            print("this slide is the same as the previous one!")
            prev_item = slide_obj["slide_text"]
            continue

        result.append(slide_obj)
        prev_item = slide_obj["slide_text"]
    return result


def ocr_with_pytesseract(image_path):
    import pytesseract
    from PIL import Image

    # Load images
    image = Image.open(image_path)

    # Extract text
    text = pytesseract.image_to_string(image)
    return text


def extract_markdown(text):
    # Define the regex pattern to capture the content between <edited_transcript> and </edited_transcript>
    pattern = r"<markdown>(.*?)</markdown>"

    # Use re.findall to find all occurrences of the pattern
    results = re.findall(pattern, text, re.DOTALL)

    # Check if any results were found
    if results:
        return results[0]
    else:
        return text


def extract_blog_post_section(text):
    # Define the regex pattern to capture the content between <edited_transcript> and </edited_transcript>
    pattern = r"<blog_post_section>(.*?)</blog_post_section>"

    # Use re.findall to find all occurrences of the pattern
    results = re.findall(pattern, text, re.DOTALL)

    # Check if any results were found
    if results:
        return results[0]
    else:
        return text


def extract_batch_and_count(image_url):
    # extract the batch and count number from the image_url

    pattern = r"slide_batch_(\d+)_count_(\d+).png"

    # Use re.findall to find all occurrences of the pattern
    results = re.findall(pattern, image_url, re.DOTALL)

    # Check if any results were found
    if results:
        return results[0]
    else:
        return None


def extract_slide_text(text):
    pattern = r'<image_analysis id="([\w-]+)">\s*<slide_text>\s*([\s\S]*?)\s*</slide_text>\s*</image_analysis>'

    # Use re.findall to find all occurrences of the pattern
    results = re.findall(pattern, text, re.DOTALL)

    # Check if any results were found
    if results:
        for result in results:
            image_id, slide_text = result
            yield (image_id, slide_text)
        # return results
    else:
        return [text]


def ensure_directory_exists(file_path):
    # Extract the directory path
    directory = os.path.dirname(file_path)
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

        # create a subdirectory /slides
        os.makedirs(f"{directory}/slides")
