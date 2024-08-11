import os
import yt_dlp
import re


class YoutubeVideo:
    def __init__(self, url):
        self.url = url

    def get_title(self):
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

        from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
        from moviepy.editor import VideoFileClip
        from r2_utils import upload_file_to_r2

        # lower-case the title of the youtube video, remove punctuation and replace spaces with underscores
        file_name = re.sub(r"[^\w\s]", "", self.title.lower()).replace(" ", "_")

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",  # Ensure the format is mp4
            "outtmpl": f"{file_name}.mp4",
            "skip_download": True,  # Skip the actual download
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
