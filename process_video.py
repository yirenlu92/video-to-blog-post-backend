import cv2
import numpy as np
from openai import OpenAI
import base64
from pydub import AudioSegment
import json
from pprint import pprint
from modal import app


@app.function(timeout=6000)
def transcribe_and_edit(
    audio=None,
    audio_url=None,
    notes="",
    transcript_context=None,
    level_of_editing="light",
    output_as_html=False,
):
    # generate id
    import uuid
    import io
    import soundfile as sf

    id = uuid.uuid4()

    # if there is an audio_url, then can skip over the rest of this:

    if not audio_url:
        # grab the audio file content
        (sample_rate, numpy_data) = audio

        # Convert the numpy array data to an audio file and save it to an in-memory buffer
        bits = io.BytesIO()
        sf.write(bits, numpy_data, sample_rate, format="WAV")
        bits.seek(0)

        # Upload to r2 bucket
        import boto3

        s3 = boto3.client("s3")

        s3 = boto3.client(
            service_name="s3",
            endpoint_url="https://36cc38112bef9dac3e0dce835950cd6e.r2.cloudflarestorage.com",
            aws_access_key_id="9f3b47a1d325dad6b74dc32f10896fb4",
            aws_secret_access_key="8ce8f3def1008a18d3069e58955d7e43de148deb7b7f5a7ef2d82de1f0a232b4",
            region_name="auto",  # Must be one of: wnam, enam, weur, eeur, apac, auto
        )

        # Upload/Update single file
        upload_file_name = f"{id}_audio.wav"
        s3.put_object(
            Bucket="edit-for-clarity-audio-uploads",
            Key=upload_file_name,
            Body=bits,
            ContentType="audio/wav",  # or the appropriate MIME type for your file
        )

        # get the audio file public url from r2
        audio_url = (
            f"https://pub-679f4064e7bc4d6ab1cc082ab36ea2ae.r2.dev/{upload_file_name}"
        )

        print(f"audio_url: {audio_url}")

    # Make call to Assembly AI to transcribe with speaker labels and
    import assemblyai as aai

    transcriber = aai.Transcriber()

    config = aai.TranscriptionConfig(speaker_labels=True)

    transcript = transcriber.transcribe(audio_url, config)

    print("raw transcript text--------")
    print(transcript.text)

    utterance_batches = []

    # iterate through transcript.utterances and batch into batches of 10 utterances

    # if there's only one speaker
    if len(transcript.utterances) == 1:
        print("There's only one speaker!")
        # split the utterance into chunks of 25 sentences each
        num_sentence_batch = 25
        sentences = transcript.text.split(".")
        for i in range(0, len(sentences), num_sentence_batch):
            batch = sentences[i : min(i + num_sentence_batch, len(sentences))]
            batch_text = ". ".join(batch)
            utterance_batches.append(batch_text)
    else:
        batch_size = 10
        print("length of transcript utterances")
        print(len(transcript.utterances))
        for i in range(0, len(transcript.utterances), batch_size):
            batch = transcript.utterances[
                i : min(i + batch_size, len(transcript.utterances))
            ]
            batch_text = "\n\n".join(
                [
                    f"Speaker {utterance.speaker} : {utterance.text}"
                    for utterance in batch
                ]
            )
            utterance_batches.append(batch_text)

    edited_for_clarity_output = []

    # check how long the utterance_batches are
    len_utterances_batches = len(utterance_batches)
    print("len_utterances_batches--------")
    print(len_utterances_batches)

    # check the length of utterance batches. If it's too short (there's only one speaker), then will need to chunk according to something different
    # if len_utterances_batches == 1:

    # get the deployed call_anthropic function
    call_anthropic = modal.Function.lookup(
        "edit-for-clarity-frontend", "call_anthropic"
    )

    # map the call_anthropic function
    edited_for_clarity_output = call_anthropic.starmap(
        [
            (
                utterance_batch,
                transcript_context,
                notes,
                level_of_editing,
                output_as_html,
            )
            for utterance_batch in utterance_batches
        ]
    )

    print(f"edited_for_clarity_output: {edited_for_clarity_output}")

    # return the edited transcript
    return "\n\n".join(edited_for_clarity_output)


schema_example = """
{
    "sections":
    [
        {
            "title": "Example 1",
            "total_slide_text:" "example slide OCR'ed text",
            "timestamp": "00:00:00,750 --> 00:00:08,250",
        }, 
        {
            "title": "Example 2",
            "total_slide_text:" "example slide OCR'ed text",
            "timestamp": "00:00:08,350 --> 00:00:12,250",
        },
    ]
}
"""

schema_example_extract = """
{
    "slide_text_segment": "example slide OCR'ed text",
    "transcript_segment": "example transcript segment text"
}
"""


def get_main_sections(slide_text):
    # generate the blog post section
    client = OpenAI()

    prompt = f"""We have OCRed the text off the slides of a recorded talk. Can you group the slides into main sections of the talk?
    Please include the timestamps of the beginning and end of each section.
    Please output the main sections as a JSON object that adheres to the following schema:
    
    ```
    {schema_example}
    ```
    
    For example, if you have several slides that look like this: 

    <slide_text>
    {{
    "slides":
    [
        {{
            "id": "slide_1",
            "slide_text_segment": "Lightning Talk:\nHow to Win at Technical Interviews\nThe Secret Protocol You're Expected to Follow",
            "timestamp": "00:00:04,170 --> 00:00:08,241"
        }},
        {{
            "id": "slide_2",
            "slide_text_segment": "Why Listen to Me",
            "timestamp": "00:00:08,341 --> 00:00:12,412"
        }},
        {{
            "id": "slide_3",
            "slide_text_segment": "",
            "timestamp": "00:00:12,512 --> 00:00:16,583"
        }},
        {{
            "id": "slide_4",
            "slide_text_segment": "",
            "timestamp": "00:00:16,683 --> 00:00:20,754"
        }},
        {{
            "id": "slide_5",
            "slide_text_segment": "Repeat the Question",
            "timestamp": "00:00:20,854 --> 00:00:29,095"
        }},
        {{
            "id": "slide_6",
            "slide_text_segment": "",
            "timestamp": "00:00:29,195 --> 00:00:33,266"
        }},
        {{
            "id": "slide_7",
            "slide_text_segment": "",
            "timestamp": "00:00:33,366 --> 00:00:37,437"
        }},
        {{
            "id": "slide_8",
            "slide_text_segment": "",
            "timestamp": "00:00:37,537 --> 00:00:41,608"
        }},
        {{
            "id": "slide_9",
            "slide_text_segment": "Write the Interface",
            "timestamp": "00:00:41,708 --> 00:00:49,950"
        }},
        {{
            "id": "slide_10",
            "slide_text_segment": "Use a hash map",
            "timestamp": "00:00:50,050 --> 00:00:54,120"
        }},
        {{
            "id": "slide_11",
            "slide_text_segment": "",
            "timestamp": "00:00:54,220 --> 00:01:02,462"
        }},
        {{
            "id": "slide_12",
            "slide_text_segment": "",
            "timestamp": "00:01:02,562 --> 00:01:06,633"
        }},
        {{
            "id": "slide_13",
            "slide_text_segment": "",
            "timestamp": "00:01:06,733 --> 00:01:14,975"
        }},
        {{
            "id": "slide_14",
            "slide_text_segment": "Use <algorithm>",
            "timestamp": "00:01:15,075 --> 00:01:19,145"
        }},
        {{
            "id": "slide_15",
            "slide_text_segment": "",
            "timestamp": "00:01:19,245 --> 00:01:27,487"
        }},
        {{
            "id": "slide_16",
            "slide_text_segment": "Example\n• https://leetcode.com/problems/majority-element-ii/\n• Given an integer array of size n, find all elements that appear more than n/3 times.",
            "timestamp": "00:01:27,587 --> 00:01:31,658"
        }},
        {{
            "id": "slide_17",
            "slide_text_segment": "",
            "timestamp": "00:01:31,758 --> 00:01:35,829"
        }},
        {{
            "id": "slide_18",
            "slide_text_segment": "",
            "timestamp": "00:01:35,929 --> 00:01:40,000"
        }},
        {{
            "id": "slide_19",
            "slide_text_segment": "",
            "timestamp": "00:01:40,100 --> 00:01:44,170"
        }},
        {{
            "id": "slide_20",
            "slide_text_segment": "",
            "timestamp": "00:01:44,270 --> 00:01:52,512"
        }}
    ]
}}
    </slide_text>

    Then you might output the following, since several of the slides are part of the same concept/section.
    ```
    [
        {{
            "id": "slide_1",
            "slide_text_segment": "Lightning Talk:\nHow to Win at Technical Interviews\nThe Secret Protocol You're Expected to Follow",
            "timestamp": "00:00:04,170 --> 00:00:08,241"
        }},
        {{
            "id": "slide_2",
            "slide_text_segment": "Why Listen to Me",
            "timestamp": "00:00:08,341 --> 00:00:20,754"
        }},
        {{
            "id": "slide_5",
            "slide_text_segment": "Repeat the Question",
            "timestamp": "00:00:20,854 --> 00:00:41,608"
        }},
        {{
            "id": "slide_9",
            "slide_text_segment": "Write the Interface",
            "timestamp": "00:00:41,708 --> 00:00:49,950"
        }},
        {{
            "id": "slide_10",
            "slide_text_segment": "Use a hash map",
            "timestamp": "00:00:50,050 --> 00:01:14,975"
        }},
        {{
            "id": "slide_14",
            "slide_text_segment": "Use <algorithm>",
            "timestamp": "00:01:15,075 --> 00:01:19,145"
        }},
        {{
            "id": "slide_15",
            "slide_text_segment": "",
            "timestamp": "00:01:19,245 --> 00:01:27,487"
        }},
        {{
            "id": "slide_16",
            "slide_text_segment": "Example\n• https://leetcode.com/problems/majority-element-ii/\n• Given an integer array of size n, find all elements that appear more than n/3 times.",
            "timestamp": "00:01:27,587 --> 00:01:31,658"
        }},
        {{
            "id": "slide_17",
            "slide_text_segment": "",
            "timestamp": "00:01:31,758 --> 00:01:35,829"
        }},
        {{
            "id": "slide_18",
            "slide_text_segment": "",
            "timestamp": "00:01:35,929 --> 00:01:40,000"
        }},
        {{
            "id": "slide_19",
            "slide_text_segment": "",
            "timestamp": "00:01:40,100 --> 00:01:44,170"
        }},
        {{
            "id": "slide_20",
            "slide_text_segment": "",
            "timestamp": "00:01:44,270 --> 00:01:52,512"
        }}
    ]
    ```

    You should use the context provided by the slide_text_segment to create meaningful titles that could reasonably become section subheadings in a blog post.

    Here is the entire slide text:

    <slide_text>
    {slide_text}
    </slide_text>
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs JSON",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return json.loads(response.choices[0].message.content)


def get_transcript_by_timestamp_for_section(section, transcript):
    # convert srt format to ms
    def srt_to_ms(srt_timestamp):
        print("srt_timestamp:")
        print(srt_timestamp)
        hours = int(srt_timestamp.split(":")[0])
        minutes = int(srt_timestamp.split(":")[1])
        seconds = int(srt_timestamp.split(":")[2].split(",")[0])
        ms = int(srt_timestamp.split(":")[2].split(",")[1])
        return ms + seconds * 1000 + minutes * 1000 * 60 + hours * 1000 * 60 * 60

    # grab the section timestamps
    section_timestamps = section["timestamp"].split(" --> ")
    section_timestamp_start = srt_to_ms(section_timestamps[0])
    section_timestamp_end = srt_to_ms(section_timestamps[1])

    # grab the transcript segment that corresponds to the section
    transcript_segments = transcript.split("\n\n")
    section_segments = []
    for segment in transcript_segments:
        print("segment:")
        print(segment)

        if segment.strip() == "":
            continue

        transcript_segment_start = srt_to_ms(segment.split("\n")[1].split(" --> ")[0])
        srt_to_ms(segment.split("\n")[1].split(" --> ")[1])

        # if the segment fits inside the section
        if (
            transcript_segment_start >= section_timestamp_start - 5000
            and transcript_segment_start <= (section_timestamp_end + 5000)
        ):
            # this segment corresponds to this section
            section_segments.append(segment)

    # ultimately want to join together all the transcript segments into one large transcript segment
    transcript_segment = " ".join(
        list(map(lambda x: x.split("\n")[2], section_segments))
    )
    return transcript_segment


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def transcribe_segment(path_to_audio_file):
    client = OpenAI()

    audio_file = open(path_to_audio_file, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, response_format="srt"
    )

    return transcript


slide_text_ocr_schema = """
{
    "slides":
    [
        {
            "image_path": "path/to/image1.png",
            "slide_text_segment": "example slide OCR'ed text",
        },
        {
            "image_path": "path/to/image2.png",
            "slide_text_segment": "example slide OCR'ed text",
        }
    ]
}
"""


def turn_ocred_text_into_json(slide_text, image_paths):
    # use OpenAI's json mode
    client = OpenAI()

    prompt = f"""Please turn the OCR'ed text from the slides into a JSON object that adheres to the following schema:

    ```
    {slide_text_ocr_schema}
    ```

    Here are the image paths for the set of images:
    ```
    {image_paths}
    ```

    Here is the entire slide text:

    <slide_text>
    {slide_text}
    </slide_text>

    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs JSON",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return json.loads(response.choices[0].message.content)


def read_text_from_slides_with_openai(image_urls):
    # get the base64 string for each image
    from openai import OpenAI

    image_text_prompt = """For each image, please do the following:
    
    1. OCR the text in the slide portion of the image (Do not tell me about the other sections of the images, like the title or the part of the image showing the speaker. Just the text in the slide itself)
    2. Return the OCR'ed text in a structured form, similar to how it appeared on the slide, in <slide_text> tags. 
    """

    client = OpenAI()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": image_text_prompt,
                },
            ],
        }
    ]

    for image_url in image_urls:
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        )

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2000,
    )

    return response.choices[0].message.content


def break_up_audio_into_segments(slide_markings):
    song = AudioSegment.from_file("coding_interviews.mp4", "mp4")

    # segment the audio into chunks based on the slide markings
    # save each chunk as a separate mp3 file. The slide lasts until the next slide starts
    for i, slide_transition in enumerate(slide_markings):
        if i == len(slide_markings) - 1:
            segment = song[slide_transition * 1000 :]
            segment.export(f"output_clips/slide_{i}.mp3", format="mp3")
            break
        elif i == 0:
            segment = song[: slide_markings[i + 1] * 1000]
        else:
            segment = song[slide_transition * 1000 : slide_markings[i + 1] * 1000]

        segment.export(f"output_clips/slide_{i}.mp3", format="mp3")


def ms_to_srt_timestamp(ms):
    hours = int(ms / (1000 * 60 * 60))
    ms -= hours * 1000 * 60 * 60
    minutes = int(ms / (1000 * 60))
    ms -= minutes * 1000 * 60
    seconds = int(ms / 1000)
    ms -= seconds * 1000

    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"


def save_slides_from_video(
    video_path, output_folder, frame_interval=100, threshold=0.3
):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    slide_count = 0
    prev_frame = None
    slides = []

    slide_ms_timestamp_start = 0
    slide_ms_timestamp_end_prev = 0
    slide_path = ""

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
            slide_path = f"{output_folder}/slide_{slide_count}.png"
            cv2.imwrite(slide_path, current_frame)
            print(f"Slide {slide_count} saved.")

        # if this is not the first frame, compare it to the previous frame
        if prev_frame is not None:
            # Calculate the difference between frames
            diff = cv2.absdiff(current_frame, prev_frame)
            non_zero_count = np.count_nonzero(diff)

            # If difference is significant, it's a new slide
            if non_zero_count > threshold * diff.size:
                # Record the timestamp of the slide
                timestamp = i / fps
                print(f"New slide at {timestamp} seconds.")

                ms_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                # end of the previous slide was just before the start of the current slide
                slide_ms_timestamp_end_prev = ms_timestamp - 100

                # save the previous slide
                slides.append(
                    (
                        slide_count,
                        slide_ms_timestamp_start,
                        slide_ms_timestamp_end_prev,
                        slide_path,
                    )
                )

                # start the next slide
                slide_ms_timestamp_start = ms_timestamp

                # increment the slide count
                slide_count += 1

                # get the slide_path
                slide_path = f"{output_folder}/slide_{slide_count}.png"

                # add a thin black rectangle to the bottom of the image with white text with the image path
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_black = np.zeros((100, current_frame.shape[1], 3), np.uint8)
                cv2.putText(
                    bottom_black,
                    f"image_path: {slide_path}",
                    (10, 70),
                    font,
                    2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(slide_path, current_frame)
                print(f"Slide {slide_count} saved.")

        # Update previous frame
        prev_frame = current_frame

    cap.release()
    print("Finished processing video.")
    return slides


def blog_post_output_has_been_cutoff(output):
    prompt = f"Please determine whether the text inside the <text> tags below constitutes a complete blog post or whether it has been cut off. If it has been cut off, please return 'True'. If it has not been cut off, please return 'False'.\n <text> {output} </text>"

    # generate the blog post section
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert technical writer with both an engineering background as a senior engineer and a writing background as someone who maintained documentation. You are helping turn these recorded talks into blog posts on behalf of the speakers and conference organizers.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content == "True"


def srt_timestamp_to_milliseconds(timestamp):
    """
    Convert an SRT timestamp to milliseconds.

    Args:
    timestamp (str): The SRT timestamp string in the format 'hours:minutes:seconds,milliseconds'.

    Returns:
    int: The time in milliseconds.
    """
    hours, minutes, seconds_milliseconds = timestamp.split(":")
    seconds, milliseconds = seconds_milliseconds.split(",")

    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    milliseconds = int(milliseconds)

    return (hours * 3600000) + (minutes * 60000) + (seconds * 1000) + milliseconds


def transcribe_and_ocr():
    # get full transcript
    # full_transcript = transcribe_segment("coding_interviews.mp4")

    f = open("full_transcript.srt", "r+")

    # write the full transcript to a file
    full_transcript = f.read()

    print(full_transcript)

    # split up the transcript, which is in srt format, and turn the timestamps into milliseconds

    full_transcript_segments = full_transcript.split("\n\n")
    segments = []
    for seg in full_transcript_segments:
        lines = seg.split("\n")
        print(lines)
        id = lines[0]
        start_and_end = lines[1].split(" --> ")
        start = srt_timestamp_to_milliseconds(start_and_end[0])
        # convert start timestamp to milliseconds
        end = srt_timestamp_to_milliseconds(start_and_end[1])
        transcript = lines[2]

        segments.append(
            {"id": id, "start": start, "end": end, "transcript": transcript}
        )

    # Example usage
    slides = save_slides_from_video("coding_interviews.mp4", "output_slides")

    # add a buffer of 2 seconds
    buffer = 2000

    # for each slide, find the segments of the transcript that corresponds to the duration of that slide
    slides_info = []
    for slide in slides:
        slide_info_element = {"id": slide[0], "audio_text": "", "image_path": slide[3]}

        slide_start = slide[1]
        slide_end = slide[2]

        slide_info_element["start"] = slide_start
        slide_info_element["end"] = slide_end

        for segment in segments:
            if (
                segment["start"] >= slide_start - buffer
                and segment["end"] <= slide_end + buffer
            ):
                # this segment is part of the slide

                slide_info_element["audio_text"] += f" {segment['transcript']}"

            # if the segment start is after the slide_end + buffer, then we can break out of the loop
            if segment["start"] > slide_end + buffer:
                break

        slides_info.append(slide_info_element)

    return slides_info


def main():
    slides_info = transcribe_and_ocr()

    # pretty print the slides info
    pprint(slides_info)

    # write the slides info into a file, pretty printed
    with open("slides_info.json", "w") as f:
        pprint(slides_info, f)

    # now need to integrate all the slide text transcription

    # get slides from slide_info in batches of 5
    slide_info_batches = [slides_info[i : i + 5] for i in range(0, len(slides_info), 5)]

    for batch in slide_info_batches:
        image_paths = list(map(lambda x: x["image_path"], batch))
        slide_text_for_batch = read_text_from_slides_with_openai(image_paths)

        slide_text_for_batch_json = turn_ocred_text_into_json(
            slide_text_for_batch, image_paths
        )

        pprint(slide_text_for_batch_json)

        for slide in slide_text_for_batch_json["slides"]:
            for slide_info in batch:
                if slide_info["image_path"] == slide["image_path"]:
                    slide_info["slide_text"] = slide["slide_text_segment"]

    # now we have the full slides_info object
    # we want to condense the object
    # all slides that have the same image should be condensed

    # we want to save this as a JSON file
    with open("slides_info.json", "w") as f:
        json.dump(slides_info, f)


if __name__ == "__main__":
    main()
