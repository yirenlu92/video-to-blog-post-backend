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


# # Function to compute cosine similarity between two texts
# def compute_similarity(text1, text2):
#     # Preprocess the texts
#     text1 = preprocess_text(text1)
#     text2 = preprocess_text(text2)

#     # Create a CountVectorizer to count token occurrences
#     vectorizer = CountVectorizer().fit_transform([text1, text2])
#     vectors = vectorizer.toarray()

#     # Compute cosine similarity
#     cosine_sim = cosine_similarity(vectors)

#     # Cosine similarity between the two texts
#     return cosine_sim[0, 1]


def slides_are_the_same(text1, text2):
    return text1 == text2
    # # Compute similarity between the texts
    # similarity = compute_similarity(text1, text2)

    # # Define a threshold for similarity to consider the texts as the same slide
    # threshold = 0.8

    # if similarity >= threshold:
    #     print(f"text1: {text1}")
    #     print(f"text2: {text2}")
    #     print("The frames contain the same slide.")
    #     return True
    # else:
    #     print(f"text1: {text1}")
    #     print(f"text2: {text2}")
    #     print("The frames contain different slides.")
    #     return False


def remove_duplicates_preserve_order(input_list):
    result = []

    prev_item = ""
    for index, full_item in enumerate(input_list):
        print("full_item: ", full_item)
        (i, item, start, end) = full_item

        if slides_are_the_same(item.lower(), prev_item.lower()):
            print("this slide is the same as the previous one!")
            continue

        result.append(full_item)
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
