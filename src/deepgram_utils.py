# main.py (python example)

import os

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
)


def transcribe_with_deepgram(
    audio_url=None,
):
    try:
        # STEP 1 Create a Deepgram client using the API key
        deepgram = DeepgramClient(os.environ.get("DEEPGRAM_API_KEY"))

        # STEP 2: Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            paragraphs=True,
            diarize=True,
        )

        # STEP 3: Call the transcribe_url method with the audio payload and options
        response = deepgram.listen.prerecorded.v("1").transcribe_url(audio_url, options)

        # STEP 4: Print the response
        print(response.to_json(indent=4))

        # # get all the sentences
        # for paragraph in response["paragraphs"]["paragraphs"]:
        #     sentences = paragraph["sentences"]
        #     for sentence in sentences:

    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    transcribe_with_deepgram(
        "https://pub-f1ee73dd9450494a95fae11b75fb5a42.r2.dev/designing__building_metric_trees.mp4"
    )
