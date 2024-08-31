from modal import Image

audio_format = "wav"
whisper_dir = "/whisper/"
alignment_dir = "/whisper_align/"


def download_whisper():
    import whisperx

    device = "cuda"
    compute_type = "float16"
    whisperx.load_model(
        "large-v2", device, compute_type=compute_type, download_root=whisper_dir
    )
    whisperx.load_align_model(
        language_code="en", device=device, model_dir=alignment_dir
    )


whisperx_image = (
    Image.debian_slim()
    .pip_install("whisperx")
    .run_function(download_whisper, gpu="T4")
    .apt_install("ffmpeg")
)
