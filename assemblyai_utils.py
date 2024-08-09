import os


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
