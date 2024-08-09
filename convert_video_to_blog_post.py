import os
from modal import Image, App, Secret, Function, asgi_app
from modal.functions import FunctionCall
from pydantic import BaseModel
from pprint import pprint


import fastapi
from fastapi.middleware.cors import CORSMiddleware

import cv2
import argparse


class YouTubeRequest(BaseModel):
    youtube_url: str


image = (
    Image.debian_slim()
    .apt_install("ffmpeg", "tesseract-ocr", "libtesseract-dev")
    .pip_install(
        "pytube",
        "pydub",
        "numpy",
        "moviepy",
        "requests",
        "pyjwt",
        "boto3",
        "pyairtable",
        "opencv-python",
        "numpy",
        "openai",
        "assemblyai",
        "anthropic",
        "pybase64",
        "httpx",
        "yt_dlp",
        "pytesseract",
        "Pillow",
        "ffmpeg-python",
        "scikit-learn",
    )
)

app = App("video-to-blog-post", image=image)

web_app = fastapi.FastAPI()

origins = [
    "http://localhost:5173",
    "https://video-to-blog-post-frontend.onrender.com",
    # Add other origins you want to allow
]

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


@app.function()
@asgi_app()
def fastapi_app():
    return web_app


def extract_markdown(text):
    import re

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
    import re

    # Define the regex pattern to capture the content between <edited_transcript> and </edited_transcript>
    pattern = r"<blog_post_section>(.*?)</blog_post_section>"

    # Use re.findall to find all occurrences of the pattern
    results = re.findall(pattern, text, re.DOTALL)

    # Check if any results were found
    if results:
        return results[0]
    else:
        return text


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


def extract_batch_and_count(image_url):
    # extract the batch and count number from the image_url
    import re

    pattern = r"slide_batch_(\d+)_count_(\d+).png"

    # Use re.findall to find all occurrences of the pattern
    results = re.findall(pattern, image_url, re.DOTALL)

    # Check if any results were found
    if results:
        return results[0]
    else:
        return None


def extract_slide_text(text):
    import re

    pattern = r'<image_analysis id="([\w-]+)">\s*<slide_text>\s*([\s\S]*?)\s*</slide_text>\s*</image_analysis>'

    # # Define the regex pattern to capture the content between <edited_transcript> and </edited_transcript>
    # pattern = r"<slide_text>(.*?)</slide_text>"

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


@app.function(timeout=6000, secrets=[Secret.from_name("r2-secret")])
def upload_multiple_files_to_r2(file_paths):
    for file_path in file_paths:
        upload_file_to_r2(file_path)


def upload_file_to_r2(file_path):
    import boto3

    # # Load credentials from the JSON file
    # with open("credentials.json") as f:
    #     credentials = json.load(f)

    # Extract credentials
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    s3 = boto3.client(
        service_name="s3",
        endpoint_url="https://36cc38112bef9dac3e0dce835950cd6e.r2.cloudflarestorage.com",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="auto",  # Must be one of: wnam, enam, weur, eeur, apac, auto
    )

    # Upload/Update single file
    s3.upload_file(
        file_path,
        Bucket="video-to-blog-post-uploads",
        Key=file_path,
    )

    # get the public url from r2
    public_url = f"https://pub-f1ee73dd9450494a95fae11b75fb5a42.r2.dev/{file_path}"

    print(f"public_url: {public_url}")
    return public_url


class YoutubeVideo:
    def __init__(self, url):
        self.url = url

    def get_title(self):
        import yt_dlp

        ydl_opts = {
            "quiet": True,  # Suppress verbose output
            "skip_download": True,  # Skip the actual download
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract information without downloading
            info_dict = ydl.extract_info(self.url, download=False)

            # Get the title
            title = info_dict.get("title", None)

        self.title = title
        return title

    def download_youtube_video(self):
        import yt_dlp
        import re
        from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
        from moviepy.editor import VideoFileClip

        # lower-case the title of the youtube video, remove punctuation and replace spaces with underscores
        file_name = re.sub(r"[^\w\s]", "", self.title.lower()).replace(" ", "_")

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",  # Ensure the format is mp4
            "outtmpl": f"{file_name}.mp4",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(self.url, download=True)
            downloaded_file = ydl.prepare_filename(info_dict)

        # Print the info_dict to see all information provided by yt_dlp
        print(f"Info dict: {info_dict}")

        print(f"Expected downloaded file: {downloaded_file}")

        # List all files in the current directory to see if the file is present
        print("Files in the current directory:", os.listdir("."))

        # Check if the file exists
        if not os.path.exists(downloaded_file):
            raise FileNotFoundError(f"File not found: {downloaded_file}")

        # upload the whole file to s3
        whole_video_file_url = upload_file_to_r2(downloaded_file)

        # get the public url from r2

        # Load the video file
        video = VideoFileClip(downloaded_file)
        duration = video.duration  # duration in seconds

        # Split the video into chunks of 5 minutes (300 seconds)
        chunk_duration = 300
        start_time = 0
        chunk_number = 1

        audio_urls = []

        while start_time < duration:
            end_time = min(start_time + chunk_duration, duration)
            chunk_file_name = f"{file_name}_chunk_{chunk_number}.mp4"

            ffmpeg_extract_subclip(
                downloaded_file, start_time, end_time, targetname=chunk_file_name
            )

            # Check if the chunk file exists
            if not os.path.exists(chunk_file_name):
                raise FileNotFoundError(f"Chunk file not found: {chunk_file_name}")

            # Upload the chunk to R2
            audio_url = upload_file_to_r2(chunk_file_name)
            audio_urls.append(audio_url)

            # Remove the chunk file after upload to save space
            os.remove(chunk_file_name)

            start_time += chunk_duration
            chunk_number += 1

        # Remove the original downloaded file to save space
        os.remove(downloaded_file)

        print("after upload file to s3")
        return whole_video_file_url, audio_urls


@app.function(timeout=6000, secrets=[Secret.from_name("video-to-tutorial-keys")])
def read_text_from_slides_with_openai(image_urls):
    # get the base64 string for each image

    print("image_urls to ocr:", image_urls)

    from openai import OpenAI

    image_text_prompt = """
You are an AI assistant tasked with analyzing a series of images containing presentation slides. Your job is to extract and structure the text from these slides, as well as render any diagrams present. Follow these instructions carefully for each image:

For each image, provided with its image_id please do the following:

1. Perform Optical Character Recognition (OCR) on the text in the slide portion of the image. Ignore any text in other sections of the image, such as titles or parts showing the speaker.

2. Structure the OCR'ed text to resemble its appearance on the slide as closely as possible. Present this text within <slide_text> tags.

3. If the slide contains any diagrams, render them as best you can in ascii art. Include any labels or annotations present in the diagram.

4. Output the results for each slide along with the image_id. Format your output as follows:

<image_analysis id="[Insert image_id here]">
<slide_text>
[Insert structured OCR'ed text here]
</slide_text>

</image_analysis>

<image_analysis id="[Insert image_id here]">
<slide_text>
[Insert structured OCR'ed text here]
</slide_text>
</image_analysis>

[Continue for all images in the set]

So if the image_id is batch_0_count_2, then the output should look something like this:

<image_analysis id="batch_0_count_2">
<slide_text>
Title of Slide

• Bullet point 1
• Bullet point 2

Additional text on the slide
</slide_text>
</image_analysis>

Example of how to structure the OCR'ed text:

<slide_text>
Title of Slide

• Bullet point 1
• Bullet point 2
   - Sub-bullet point A
   - Sub-bullet point B
• Bullet point 3

Additional text on the slide
</slide_text>

Remember to maintain ordering of the images in your analysis (ascending by batch/count number, so batch_0_count_0, batch_0_count_1, batch_1_count_0, batch_1_count_1, etc.) and to include all relevant information from each slide.
    """

    client = OpenAI()

    messages = [
        {
            "role": "user",
            "content": [],
        }
    ]

    for image_url in image_urls:
        # extract the batch and count number from the image_url
        batch_and_count = extract_batch_and_count(image_url)

        messages[0]["content"].append(
            {
                "type": "text",
                "text": f"image_id: batch_{batch_and_count[0]}_count_{batch_and_count[1]}:",
            },
        )
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        )

    messages[0]["content"].append({"type": "text", "text": image_text_prompt})

    try:
        message = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=2000, messages=messages
        )
        print("results from openai")
        print(message.choices[0].message.content)
        return message.choices[0].message.content

    except Exception as e:
        print(f"Error in OpenAI API: {e}")
        return f"Error in OpenAI API: {e}"


@app.function(timeout=6000, secrets=[Secret.from_name("video-to-tutorial-keys")])
def read_text_from_slides_with_anthropic(image_urls):
    # get the base64 string for each image
    import base64
    import httpx

    import anthropic

    image_text_prompt = """
You are an AI assistant tasked with analyzing a series of images containing presentation slides. Your job is to extract and structure the text from these slides, as well as render any diagrams present. Follow these instructions carefully for each image:

For each image, numbered in ascending order, please do the following:

1. Perform Optical Character Recognition (OCR) on the text in the slide portion of the image. Ignore any text in other sections of the image, such as titles or parts showing the speaker.

2. Structure the OCR'ed text to resemble its appearance on the slide as closely as possible. Present this text within <slide_text> tags.

3. If the slide contains any diagrams, render them as best you can in ascii art. Include any labels or annotations present in the diagram.

4. Output the results for each slide in ascending order based on the image numbers. Format your output as follows:

<image_analysis id="[Insert image id here]">
<slide_text>
[Insert structured OCR'ed text here]
</slide_text>

</image_analysis>

<image_analysis id="[Insert image id here]">
<slide_text>
[Insert structured OCR'ed text here]
</slide_text>
</image_analysis>

[Continue for all images in the set]

So if the image id is batch_0_count_2, then the output should look something like this:

<image_analysis id="batch_0_count_2">
<slide_text>
Title of Slide

• Bullet point 1
• Bullet point 2

Additional text on the slide
</slide_text>
</image_analysis>

Example of how to structure the OCR'ed text:

<slide_text>
Title of Slide

• Bullet point 1
• Bullet point 2
   - Sub-bullet point A
   - Sub-bullet point B
• Bullet point 3

Additional text on the slide
</slide_text>


Remember to maintain the exact order of the image numbering in your analysis and to include all relevant information from each slide.
    """

    client = anthropic.Anthropic()

    messages = [
        {
            "role": "user",
            "content": [],
        }
    ]

    for image_url in image_urls:
        # extract the batch and count number from the image_url
        batch_and_count = extract_batch_and_count(image_url)
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        image_media_type = "image/png"

        messages[0]["content"].append(
            {
                "type": "text",
                "text": f"Image id: batch_{batch_and_count[0]}_count_{batch_and_count[1]}:",
            },
        )
        messages[0]["content"].append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type,
                    "data": image_data,
                },
            }
        )

    messages[0]["content"].append({"type": "text", "text": image_text_prompt})

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=2000, messages=messages
        )
        print("results from anthropic")
        print(message.content[0].text)
        return message.content[0].text

    except Exception as e:
        print(f"Error in Anthropic API: {e}")
        return f"Error in Anthropic API: {e}"


# Function to preprocess text
def preprocess_text(text):
    import re

    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r"\W+", " ", text)
    return text


# Function to extract text from an image using pytesseract
def extract_text_from_image(image):
    import pytesseract

    return pytesseract.image_to_string(image)


# Function to compute cosine similarity between two texts
def compute_similarity(text1, text2):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Preprocess the texts
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Create a CountVectorizer to count token occurrences
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()

    # Compute cosine similarity
    cosine_sim = cosine_similarity(vectors)

    # Cosine similarity between the two texts
    return cosine_sim[0, 1]


def slides_are_the_same(text1, text2):
    # Compute similarity between the texts
    similarity = compute_similarity(text1, text2)

    # Define a threshold for similarity to consider the texts as the same slide
    threshold = 0.8

    if similarity >= threshold:
        print(f"text1: {text1}")
        print(f"text2: {text2}")
        print("The frames contain the same slide.")
        return True
    else:
        print(f"text1: {text1}")
        print(f"text2: {text2}")
        print("The frames contain different slides.")
        return False


@app.function(timeout=6000, secrets=[Secret.from_name("video-to-tutorial-keys")])
def write_section(paragraphs, slide_text_list_elem, all_output_folders):
    from openai import OpenAI

    slide_id, slide_text, start, end = slide_text_list_elem

    # extract the batch_number from the slide_id
    batch_number = int(slide_id.split("_")[1])

    print("all_output_folders")
    print(all_output_folders)

    slide_image_url = f"https://pub-f1ee73dd9450494a95fae11b75fb5a42.r2.dev/{all_output_folders[batch_number]}/slides/slide_{slide_id}.png"

    client = OpenAI()

    prompt = f"""You are a technical writing expert tasked with converting a portion of a conference talk transcript into a section of a technical blog post. You will be provided with the following information:

1. A public URL of an image (slide from the talk)
2. OCR'ed text corresponding to the image
3. The corresponding portion of the talk transcript

Here are the inputs:

<slide_image_url>
{slide_image_url}
</slide_image_url>

<slide_text>
{slide_text}
</slide_text>

<transcript>
{paragraphs[slide_id]}
</transcript>

Your task is to lightly edit the transcript for clarity while preserving the first-person voice and converting it into a segment of a technical blog post. Follow these guidelines:

1. Create writing that is down-to-earth and not pedantic or overly expository.
2. Remember that this is only one section of the blog post, so it doesn't need an introduction or conclusion. Avoid phrases like "In this section, " and "To summarize." In general, don't say that you are going to say the thing, just say the thing.
3. Include all the details from the transcript and slide text, but in a more polished and readable form with grammatically correct sentences.
4. Use the slide title as the subheading for the section, in H2 format and sentence case.
5. Incorporate information from the slide text, including verbatim text when appropriate (lists, diagrams, code samples, or images).

Format the blog post section as follows:

1. Start with the subheading (H2, sentence case).
2. Include the slide image in markdown format.
3. Add the raw transcript text in italics, using markdown format.
4. Present the lightly edited transcript text in markdown format.

Output the entire section in markdown format, enclosed in <blog_post_section> tags.

Remember to maintain the original speaking style while improving readability and clarity. Your goal is to create a polished, engaging, and informative blog post section that accurately represents the content of the talk.
"""

    try:
        message = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=5000,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        print(message.choices[0].message.content)
        return extract_blog_post_section(message.choices[0].message.content)
    except Exception as e:
        print(f"Error in OpenAI API: {e}")
        return f"Error in OpenAI API: {e}"


def get_sentences_timestamps_from_transcript(transcript_id):
    import requests

    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}/sentences"
    headers = {"Authorization": os.environ.get("ASSEMBLYAI_API_KEY")}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def transcribe_with_assembly(
    audio_url=None,
):
    import os

    # Make call to Assembly AI to transcribe with speaker labels and
    import assemblyai as aai

    aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

    transcriber = aai.Transcriber()

    config = aai.TranscriptionConfig(speaker_labels=True)

    transcript = transcriber.transcribe(audio_url, config)

    id = transcript.id

    sentences = get_sentences_timestamps_from_transcript(id)

    return sentences["sentences"]


def ensure_directory_exists(file_path):
    import os

    # Extract the directory path
    directory = os.path.dirname(file_path)
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

        # create a subdirectory /slides
        os.makedirs(f"{directory}/slides")


def ocr_with_pytesseract(image_path):
    from PIL import Image
    import pytesseract

    # Load images
    image = Image.open(image_path)

    # Extract text
    text = pytesseract.image_to_string(image)
    return text


@app.function(
    timeout=6000,
    secrets=[
        Secret.from_name("video-to-tutorial-keys"),
        Secret.from_name("r2-secret"),
    ],
)
def save_slides_from_video(batch_number, video_path, frame_interval=300, threshold=0.1):
    # name of output folder should be human readable version of video file name
    output_folder = video_path.split("/")[-1].split(".")[0]

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the duration of the video in milliseconds
    duration_in_milliseconds = (frame_count / fps) * 1000

    slide_count = 0
    prev_frame = None
    slides = {}

    slide_ms_timestamp_start = 0
    slide_ms_timestamp_end_prev = 0
    slide_path = ""
    image_urls = []

    ensure_directory_exists(f"{output_folder}/slides")

    prev_slide_text = ""

    for i in range(0, frame_count, frame_interval):
        # Set video to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        # Read frame
        ret, current_frame = cap.read()
        if not ret:
            break

        # if this is the first frame, save it as the first slide
        if prev_frame is None:
            # save the first frame
            slide_path = f"{output_folder}/slides/slide_batch_{batch_number}_count_{slide_count}.png"
            cv2.imwrite(slide_path, current_frame)
            print(f"Slide {slide_count} saved.")

            image_url = upload_file_to_r2(slide_path)

        # if this is not the first frame, compare it to the previous frame
        if prev_frame is not None:
            # # Calculate the difference between frames
            # diff = cv2.absdiff(current_frame, prev_frame)
            # non_zero_count = np.count_nonzero(diff)

            # # If difference is significant, it's a new slide
            # if non_zero_count > threshold * diff.size:
            # Record the timestamp of the slide

            # get the slide_path
            slide_path = f"{output_folder}/slides/slide_batch_{batch_number}_count_{slide_count}.png"

            cv2.imwrite(slide_path, current_frame)
            print(f"Slide {slide_count} saved.")

            # ocr the slide with pytesseract
            slide_text = ocr_with_pytesseract(slide_path)

            # update current_slide_text
            if slides_are_the_same(slide_text.lower(), prev_slide_text.lower()):
                prev_slide_text = slide_text
                print("this slide is the same as the previous one!")
                continue

            # detected a possible new slide!
            prev_slide_text = slide_text

            timestamp = i / fps
            print(f"New slide at {timestamp} seconds.")

            ms_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            # end of the previous slide was just before the start of the current slide
            slide_ms_timestamp_end_prev = ms_timestamp - 100

            # save the previous slide
            slides[f"batch_{batch_number}_count_{slide_count}"] = {
                "slide_count": slide_count,
                "start": slide_ms_timestamp_start,
                "end": slide_ms_timestamp_end_prev,
            }

            # start the next slide
            slide_ms_timestamp_start = ms_timestamp

            # increment the slide count
            slide_count += 1

            # upload the slide to r2
            image_url = upload_file_to_r2(slide_path)
            print("Image URL:", image_url)

            image_urls.append(image_url)

        # Update previous frame
        prev_frame = current_frame

    # Save the last slide after the loop
    ms_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    slide_ms_timestamp_end_prev = ms_timestamp
    slides[f"batch_{batch_number}_count_{slide_count}"] = {
        "slide_count": slide_count,
        "start": slide_ms_timestamp_start,
        "end": slide_ms_timestamp_end_prev,
    }

    cap.release()
    print("Finished processing video.")

    return (
        duration_in_milliseconds,
        batch_number,
        output_folder,
        slide_count,
        slides,
        image_urls,
    )


def polish_written_sections(written_sections):
    from openai import OpenAI

    polish_prompt = """
You are a technical writing expert tasked with converting a series of sections of a technical blog post into a polished, engaging, and informative blog post. 

Each section is currently written as standalone, and includes both the original section transcript and the AI-written blog post section that it was converted to.
The sections are currently a bit overlapping.

1. Please output the blog post in Markdown format, in <markdown> tags.
2. Include all the detail from all the sections. Do not summarize or leave out any information.
2. Include all the screenshots in the appropriate locations, also in Markdown format.

Here are the sections: 

    """

    for section in written_sections:
        polish_prompt += section

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=16000,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": polish_prompt},
        ],
    )

    return extract_markdown(response.choices[0].message.content)


@app.function(
    timeout=6000,
    secrets=[
        Secret.from_name("video-to-tutorial-keys"),
        Secret.from_name("r2-secret"),
    ],
)
def create_video_to_post(youtube_url):
    import time

    # get the youtube video and other assorted metadata
    video = YoutubeVideo(youtube_url)

    video.get_title()

    # download the youtube video
    whole_video_url, video_urls = video.download_youtube_video()

    # get the video_url
    print(video_urls)

    save_slides_from_video_f = Function.lookup(
        "video-to-blog-post", "save_slides_from_video"
    )

    all_slide_times = {}
    all_output_folders = {}
    total_image_urls = []
    sum_video_time = 0
    for j, output in enumerate(
        save_slides_from_video_f.starmap(
            [(i, video_url, 100, 0.3) for i, video_url in enumerate(video_urls)]
        )
    ):
        (
            total_video_time,
            batch_number,
            output_folder,
            length_slides,
            slide_times,
            image_urls,
        ) = output

        print("total time: ", total_video_time)
        total_image_urls.extend(image_urls)

        # iterate through slide_times and update the start and end times to take into account the total video time
        for slide_id, slide_time in slide_times.items():
            slide_time["start"] += sum_video_time
            slide_time["end"] += sum_video_time

        sum_video_time += total_video_time

        # merge the slide_times into a big dictionary
        all_slide_times.update(slide_times)

        # merge the output_folders into a big dictionary
        all_output_folders[batch_number] = output_folder

    print("all_slide_times:")
    pprint(all_slide_times, indent=4)

    read_text_from_slides_with_openai_f = Function.lookup(
        "video-to-blog-post", "read_text_from_slides_with_openai"
    )

    slide_text_list = []
    batch_size = 5
    # time it

    # chunk up total_image_urls into batches of 5

    start_time = time.time()
    for x in read_text_from_slides_with_openai_f.map(
        [
            total_image_urls[x : min(x + batch_size, len(total_image_urls))]
            for x in range(1, len(total_image_urls), batch_size)
        ]
    ):
        slide_text_list.append(x)
    end_time = time.time()

    print(f"Time taken with .map: {end_time - start_time}")

    slide_text_list = [extract_slide_text(x) for x in slide_text_list if x is not None]

    print("slide_text_list after extracting openai response:")
    pprint(slide_text_list, indent=4)

    # flatten list

    slide_text_list = [item for sublist in slide_text_list for item in sublist]

    # add in the extra metadata

    new_slide_text_list = []
    for pair in slide_text_list:
        slide_id, slide_text = pair

        try:
            new_slide_text_list.append(
                (
                    slide_id,
                    slide_text,
                    all_slide_times[slide_id]["start"],
                    all_slide_times[slide_id]["end"],
                )
            )
        except KeyError:
            print(f"KeyError: {slide_id}")
            continue

    # check the first line of each extracted text and dedupe based on that
    slide_text_list = remove_duplicates_preserve_order(new_slide_text_list)

    print("slide_text_list")
    pprint(slide_text_list, indent=4)

    # # transcribe the video
    transcript_sentences = transcribe_with_assembly(audio_url=whole_video_url)

    print("transcript")
    print(transcript_sentences)

    write_section_f = Function.lookup("video-to-blog-post", "write_section")

    # for each slide text, grab the corresponding portion of the transcript, and rewrite it into a section of the blog post

    # match up the slides and the transcript sentences

    # iterate through the slide_text_list and add the transcript sentences that fall between the start and end timestamps of the slide

    paragraphs = {}

    for i, slide in enumerate(slide_text_list):
        slide_id, slide_text, start, end = slide

        # print slide_id, slide_text, start, end

        print(f"slide {slide_id}:")
        print(slide_text)

        print(f"start: {start}, end: {end}")

        # if it's the first slide, then get the sentences that fall between the very beginning and the end of the slide
        if i == 0:
            next_slide_number, next_slide_text, next_slide_start, next_slide_end = (
                slide_text_list[i + 1]
            )
            slide_sentences = [
                sentence
                for sentence in transcript_sentences
                if sentence["end"] <= next_slide_start + 2000
            ]

            # pprint(slide_sentences, indent=4)
        elif i + 1 < len(slide_text_list):
            next_slide_number, next_slide_text, next_slide_start, next_slide_end = (
                slide_text_list[i + 1]
            )

            # get the sentences that fall between the start and end of the slides
            slide_sentences = [
                sentence
                for sentence in transcript_sentences
                if sentence["start"] >= start - 10000
                and sentence["end"] <= next_slide_start + 2000
            ]

            # pprint(slide_sentences, indent=4)
        else:
            slide_sentences = [
                sentence
                for sentence in transcript_sentences
                if sentence["start"] >= start - 10000 and sentence["end"] <= end + 2000
            ]

            # pprint(slide_sentences, indent=4)

        # combine the sentences into a single paragraph

        paragraph = " ".join([sentence["text"] for sentence in slide_sentences])
        paragraphs[slide_id] = paragraph

        print(f"slide {slide_id}:")
        print(paragraph)

        # show the embedded slide
        # show the raw transcript sentences

        # show the edited section

    written_sections = []

    for section in write_section_f.starmap(
        [(paragraphs, x, all_output_folders) for x in slide_text_list]
    ):
        written_sections.append(section)

    return "\n\n".join(written_sections)


@web_app.post("/accept")
def accept_create_video_to_post_job(request: YouTubeRequest):
    # get the youtube url from the request body
    youtube_url = request.youtube_url

    call = create_video_to_post.spawn(youtube_url)
    return {"call_id": call.object_id}


@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    function_call = FunctionCall.from_id(call_id)
    try:
        # return as { "result": { "markdown": function_call.get(timeout=0)} }
        result = function_call.get(timeout=0)
        print("result:", result)
        return {"markdown": result}
    except TimeoutError:
        return fastapi.responses.JSONResponse({"status": "pending"}, status_code=202)


def main():
    parser = argparse.ArgumentParser(description="Convert Youtube video to blog post")

    parser.add_argument(
        "--youtube_url",
        type=str,
        required=True,
        help="Url of the Youtube video that you would like to convert to a blog post",
    )
    args = parser.parse_args()

    create_video_to_post_f = Function.lookup(
        "video-to-blog-post", "create_video_to_post"
    )

    # Submit the job to Modal
    create_video_to_post_f.remote(args.youtube_url)
    # create_video_to_post.remote(args.youtube_url)


if __name__ == "__main__":
    main()
