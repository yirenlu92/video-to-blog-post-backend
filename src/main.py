from modal import Secret, Function, Image, App
from pprint import pprint
import argparse

from .helper import (
    extract_batch_and_count,
    ocr_with_pytesseract,
    slides_are_the_same,
    ensure_directory_exists,
    extract_markdown,
    remove_duplicates_preserve_order,
)

from .r2_utils import upload_file_to_r2, download_file_from_url
from .assemblyai_utils import transcribe_with_assembly

# extensive debug output
verbose_level = 0

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
        "opencv-python",
        "numpy",
        "assemblyai",
        "anthropic",
        "pybase64",
        "httpx",
        "yt_dlp",
        "pytesseract",
        "Pillow",
        "ffmpeg-python",
        "openai",
    )
)

app = App("video-to-blog-post", image=image)


@app.function(timeout=6000, secrets=[Secret.from_name("video-to-tutorial-keys")])
def read_text_from_slides_with_openai(image_urls):
    # get the base64 string for each image

    print("image_urls to ocr:", image_urls)

    from openai import OpenAI
    from pydantic import BaseModel

    class SlideTextSection(BaseModel):
        slide_id: str
        slide_title: str
        slide_text: str
        slide_diagram: str

    image_text_prompt = """
You are an AI assistant tasked with analyzing a series of images containing presentation slides. Your job is to extract and structure the text from these slides, as well as render any diagrams present. Follow these instructions carefully for each image:

For each image, provided with its image_id please do the following:

1. Perform Optical Character Recognition (OCR) on the text in the slide portion of the image. Ignore any text in other sections of the image, such as titles or parts showing the speaker.

2. Structure the OCR'ed text to resemble its appearance on the slide as closely as possible. Present this text within slide_bullet area.

3. If the slide contains any diagrams, render them as best you can in ascii art. Include any labels or annotations present in the diagram.

4. Output the results for each slide along with the image_id. Format your output as follows:

Here is a description of the parameters, along with examples of the format you should we using inside the <slide_text_example> tags:

- slide_id: the slide id
    <example>
    put slide id here. the slide id should always be present
    </example>
- slide_title: The title of the slide 
    <example>
    This is a slide
    </example>
- slide_text_info: orc'ed slide text, ignore this section of there is no slide text. Please format it nicely in the format that it shows up on the slide.
    <example>
    • Bullet point 1
    • Bullet point 2
       - Sub-bullet point A
       - Sub-bullet point B
    • Bullet point 3
    </example>
- slide_diagram: ocr'ed diagram in ascii form, empty if there is no diagram

Remember to maintain ordering of the images in your analysis (ascending by batch/count number, so batch_0_count_0, batch_0_count_1, batch_1_count_0, batch_1_count_1, etc.) and to include all relevant information from each slide.
    """

    slide_text_section = []

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
        message = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            max_tokens=2000,
            messages=messages,
            response_format=SlideTextSection,
        )
        structured_response = message.choices[0].message.parsed

    except Exception as e:
        print(f"Error in OpenAI API: {e}")
        return f"Error in OpenAI API: {e}"

    # turn structured_response into json

    return structured_response.json()


@app.function(timeout=6000, secrets=[Secret.from_name("video-to-tutorial-keys")])
def write_section(paragraphs, slide_text_list_elem, subdirectory_name):
    from openai import OpenAI
    from textwrap import dedent

    from pydantic import BaseModel

    class OneSlideSection(BaseModel):
        subheading: str
        slide_image: str
        raw_transcript_section: str
        polished_transcript_section: str

    slide_id, slide_text, start, end = slide_text_list_elem

    # extract the batch_number from the slide_id
    batch_number = int(slide_id.split("_")[1])

    slide_image_url = f"https://pub-f1ee73dd9450494a95fae11b75fb5a42.r2.dev/{subdirectory_name}/batch_{batch_number}/slides/slide_{slide_id}.png"

    client = OpenAI()

    prompt = f"""You are a technical writing expert tasked with converting a portion of a conference talk transcript into a section of a technical blog post. 
    
You will be provided with the following information:

1. A public URL of an image (slide from the talk)
2. OCR'ed text corresponding to the image
3. The corresponding section of the talk transcript

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

Your task is to lightly edit the transcript for clarity while preserving the first-person voice and converting it into a segment of a technical blog post, and then output it along with some other artifacts following the schema provided.

Follow these guidelines:

1. Create writing that is down-to-earth and not pedantic or overly expository.
2. Remember that this is only one section of the blog post, so it doesn't need an introduction or conclusion. Avoid phrases like "In this section, " and "To summarize." In general, don't say that you are going to say the thing, just say the thing.
3. Include all the details from the transcript and slide text, but in a more polished and readable form with grammatically correct sentences.
4. Use the slide title as the subheading for the section, in H2 format and sentence case.
5. Incorporate information from the slide text, including verbatim text when appropriate (lists, diagrams, code samples, or images).


Here is a description of the parameters, along with examples of the format you should we using inside the <example> tags:

- subheading: The title of the slide, adapted from the first line of the slide_text, in H2, in markdown format and sentence case (so with ## at the beginning). 
    <example>
    ## This is a subheading
    </example>
- slide_image: The URL of the slide image, in markdown format
    <example>
    ![Slide](https://path-to-slide-image.png)
    </example>
- raw transcript section: The original transcript text, in markdown format
    <example>
    Transcript text here
    </example>
- polished transcript section: The polished/edited transcript text, in markdown format
    <example>
    Polished transcript text here
    </example>


Remember to maintain the original speaking style while improving readability and clarity. Your goal is to create a polished, engaging, and informative blog post section that accurately represents the content of the talk.
"""

    try:
        message = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            temperature=0.2,
            messages=[
                {"role": "system", "content": dedent(prompt)},
            ],
            response_format=OneSlideSection,
        )

        # combine each of the fields values into a single string
        structured_response = message.choices[0].message.parsed

    except Exception as e:
        print(f"Error in OpenAI API: {e}")
        return f"Error in OpenAI API: {e}"

    markdown_section = "\n\n".join(
        [
            structured_response.subheading,
            structured_response.slide_image,
            "Raw transcript",
            structured_response.raw_transcript_section,
            "Polished transcript",
            structured_response.polished_transcript_section,
        ]
    )

    return markdown_section


@app.function(
    timeout=6000,
    secrets=[
        Secret.from_name("video-to-tutorial-keys"),
        Secret.from_name("r2-secret"),
    ],
)
def save_slides_from_video(batch_number, video_path, frame_interval=300, threshold=0.1):
    import cv2
    import os

    # name of output folder should be human readable version of video file name, with dashes instead of spaces
    subdirectory_name = video_path.split("/")[-3]

    print("subdirectory_name:", subdirectory_name)

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

    os.makedirs(f"{subdirectory_name}/batch_{batch_number}/slides", exist_ok=True)

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
            slide_path = f"{subdirectory_name}/batch_{batch_number}/slides/slide_batch_{batch_number}_count_{slide_count}.png"
            cv2.imwrite(slide_path, current_frame)
            print(f"Slide {slide_count} saved.")

            image_url = upload_file_to_r2(slide_path)
            image_urls.append(image_url)

            prev_slide_text = ocr_with_pytesseract(slide_path)

            slide_count += 1

        # if this is not the first frame, compare it to the previous frame
        if prev_frame is not None:
            # get the slide_path
            slide_path = f"{subdirectory_name}/batch_{batch_number}/slides/slide_batch_{batch_number}_count_{slide_count}.png"

            cv2.imwrite(slide_path, current_frame)
            print(f"Slide {slide_count} saved.")

            # ocr the slide with pytesseract
            slide_text = ocr_with_pytesseract(slide_path)

            print("prev_slide_text:")
            print(prev_slide_text)
            # print the slide_text
            print("slide_text:")
            print(slide_text)

            # update current_slide_text
            if slides_are_the_same(
                slide_text.lower().strip(), prev_slide_text.lower().strip()
            ):
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
            slides[f"batch_{batch_number}_count_{slide_count-1}"] = {
                "slide_count": slide_count - 1,
                "start": slide_ms_timestamp_start,
                "end": slide_ms_timestamp_end_prev,
            }

            # start the next slide
            slide_ms_timestamp_start = ms_timestamp

            # upload the slide to r2
            image_url = upload_file_to_r2(slide_path)
            print("Image URL:", image_url)

            image_urls.append(image_url)

            # increment the slide count
            slide_count += 1

        # Update previous frame
        prev_frame = current_frame

    # Save the last slide after the loop
    ms_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    slide_ms_timestamp_end_prev = ms_timestamp
    slides[f"batch_{batch_number}_count_{slide_count-1}"] = {
        "slide_count": slide_count - 1,
        "start": slide_ms_timestamp_start,
        "end": slide_ms_timestamp_end_prev,
    }

    cap.release()
    print("Finished processing video.")

    return (
        duration_in_milliseconds,
        batch_number,
        subdirectory_name,
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


def download_file_and_split_into_batchs(video_url):
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    from moviepy.editor import VideoFileClip
    import os

    # download the file
    local_file_path = "temp_file.mp4"
    download_file_from_url(video_url, local_file_path)

    # get the file name from the public url
    file_name = video_url.split("/")[-1].split(".")[0]

    # get the subdirectory name
    subdirectory_name = video_url.split("/")[-2]

    # Load the video file
    video = VideoFileClip(local_file_path)
    duration = video.duration  # duration in seconds

    # Split the video into batchs of 5 minutes (300 seconds)
    batch_duration = 300
    start_time = 0
    batch_number = 0

    audio_urls = []

    while start_time < duration:
        end_time = min(start_time + batch_duration, duration)
        batch_file_name = f"{subdirectory_name}/batch_{batch_number}/{file_name}_batch_{batch_number}.mp4"

        ensure_directory_exists(batch_file_name)

        ffmpeg_extract_subclip(
            local_file_path, start_time, end_time, targetname=batch_file_name
        )

        # Check if the batch file exists
        if not os.path.exists(batch_file_name):
            raise FileNotFoundError(f"batch file not found: {batch_file_name}")

        # Upload the batch to R2
        audio_url = upload_file_to_r2(batch_file_name)
        audio_urls.append(audio_url)

        # Remove the batch file after upload to save space
        os.remove(batch_file_name)

        start_time += batch_duration
        batch_number += 1

    # Remove the original downloaded file to save space
    os.remove(local_file_path)

    print("after upload file to s3")
    return video_url, audio_urls


@app.function(
    timeout=6000,
    secrets=[
        Secret.from_name("video-to-tutorial-keys"),
        Secret.from_name("r2-secret"),
    ],
)
def create_video_to_post(video_r2_url):
    import time

    head_time = 5000
    tail_time = 2000

    # download the video from r2
    whole_video_url, video_urls = download_file_and_split_into_batchs(video_r2_url)

    # # get the youtube video and other assorted metadata
    # video = YoutubeVideo(youtube_url)

    # video.get_title()

    # # download the youtube video
    # whole_video_url, video_urls = video.download_youtube_video()

    # get the video_url
    print(video_urls)

    save_slides_from_video_f = Function.lookup(
        "video-to-blog-post", "save_slides_from_video"
    )

    all_slide_times = {}
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

    print("all_slide_times:")
    if verbose_level >= 1:
        pprint(all_slide_times, indent=4)

    read_text_from_slides_with_openai_f = Function.lookup(
        "video-to-blog-post", "read_text_from_slides_with_openai"
    )

    slide_text_list = []
    batch_size = 5
    # time it

    # batch up total_image_urls into batches of 5
    start_time = time.time()
    for x in read_text_from_slides_with_openai_f.map(
        [
            total_image_urls[x : min(x + batch_size, len(total_image_urls))]
            for x in range(0, len(total_image_urls), batch_size)
        ]
    ):
        # print(x)
        slide_text_list.append(x)

    end_time = time.time()

    print(f"Time taken with .map: {end_time - start_time}")

    # remove any remaining duplicate slides
    slide_text_list = remove_duplicates_preserve_order(slide_text_list)

    print("slide_text_listi after remove duplicate")
    if verbose_level >= 1:
        pprint(slide_text_list, indent=4)

    # add in the metadata
    for i, slide_obj in enumerate(slide_text_list):
        try:
            slide_obj["start"] = all_slide_times[slide_obj["slide_id"]]["start"]
            slide_obj["end"] = all_slide_times[slide_obj["slide_id"]]["end"]
        except KeyError:
            print(f"KeyError: {slide_obj['slide_id']}")
            continue

    # # transcribe the video
    transcript_sentences = transcribe_with_assembly(audio_url=whole_video_url)

    if verbose_level >= 2:
        print(transcript_sentences)

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
                if sentence.end <= next_slide_start + tail_time
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
                if sentence.start >= start - head_time
                and sentence.end <= next_slide_start + tail_time
            ]

            # pprint(slide_sentences, indent=4)
        else:
            slide_sentences = [
                sentence
                for sentence in transcript_sentences
                if sentence.start >= start - head_time
                and sentence.end <= end + tail_time
            ]

            # pprint(slide_sentences, indent=4)

        # combine the sentences into a single paragraph

        paragraph = " ".join([sentence.text for sentence in slide_sentences])
        paragraphs[slide_id] = paragraph

        print(f"slide {slide_id}:")
        if verbose_level >= 2:
            print(paragraph)

        # show the embedded slide
        # show the raw transcript sentences

        # show the edited section

    # output the raw transcript sentences for each slide

    constructed_sections = []
    # for each slide
    # get the corresponding transcript sentences
    for x in slide_text_list:
        slide_id, slide_text, start, end = x

        batch_number = int(slide_id.split("_")[1])

        slide_url = f"https://pub-f1ee73dd9450494a95fae11b75fb5a42.r2.dev/{output_folder}/batch_{batch_number}/slides/slide_{slide_id}.png"

        raw_transcript_sentences = paragraphs[slide_id]

        constructed_section = f"""![Slide]({slide_url})

{raw_transcript_sentences}"""

        constructed_sections.append(constructed_section)

    written_sections = []

    write_section_f = Function.lookup("video-to-blog-post", "write_section")

    for section in write_section_f.starmap(
        [(paragraphs, x, output_folder) for x in slide_text_list]
    ):
        section
        written_sections.append(section)

    return "\n\n".join(written_sections)


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
