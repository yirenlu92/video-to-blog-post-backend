from modal import Secret, app
from utils import extract_batch_and_count

import base64
import httpx

import anthropic


@app.function(timeout=6000, secrets=[Secret.from_name("video-to-tutorial-keys")])
def read_text_from_slides_with_anthropic(image_urls):
    # get the base64 string for each image

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
