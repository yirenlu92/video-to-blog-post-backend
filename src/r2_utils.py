def upload_file_to_r2(file_path):
    import boto3
    import os

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


def upload_file_to_r2_streaming(video):
    import boto3
    import os

    # Extract credentials
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    # Print credentials for debugging purposes (optional)
    print(f"AWS_ACCESS_KEY_ID: {aws_access_key_id}")
    print(f"AWS_SECRET_ACCESS_KEY: {aws_secret_access_key}")

    s3 = boto3.client(
        service_name="s3",
        endpoint_url="https://36cc38112bef9dac3e0dce835950cd6e.r2.cloudflarestorage.com",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="auto",  # Must be one of: wnam, enam, weur, eeur, apac, auto
    )

    print(f"Uploading file: {video.filename}")

    # Upload/Update single file
    s3.upload_fileobj(video.file, "video-to-blog-post-uploads", video.filename)

    # get the public url from r2
    public_url = f"https://pub-f1ee73dd9450494a95fae11b75fb5a42.r2.dev/{video.filename}"

    print(f"public_url: {public_url}")
    return public_url


def download_file_from_url(url, save_path):
    import requests

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully to {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
