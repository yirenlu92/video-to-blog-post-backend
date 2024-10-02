def create_subdirectory_name(file_name):
    import os

    # Remove the file extension
    base_name = os.path.splitext(file_name)[0]
    # remove non-alphanumeric characters
    base_name = "".join(e for e in base_name if e.isalnum() or e == " ")

    # Replace spaces with hyphens
    base_name = base_name.replace(" ", "_")
    # Convert to lowercase
    subdirectory_name = base_name.lower()
    return subdirectory_name


def create_readable_filename(file_name):
    import os

    # Remove the file extension
    base_name = os.path.splitext(file_name)[0]
    # remove non-alphanumeric characters
    base_name = "".join(e for e in base_name if e.isalnum() or e == " ")

    # Replace spaces with hyphens
    base_name = base_name.replace(" ", "_")
    # Convert to lowercase
    subdirectory_name = base_name.lower()

    # add back the file extension
    subdirectory_name = f"{subdirectory_name}.mp4"
    return subdirectory_name


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

    subdirectory_name = create_subdirectory_name(video.filename)
    readable_file_path = create_readable_filename(video.filename)

    # create a bucket subdirectory for this particular file/project
    filename_in_r2 = f"{subdirectory_name}/{readable_file_path}"

    # Print the filename for debugging purposes (optional)

    print(f"Uploading file: {video.filename}")

    # Upload/Update single file
    s3.upload_fileobj(video.file, "video-to-blog-post-uploads", filename_in_r2)

    # get the public url from r2
    public_url = f"https://pub-f1ee73dd9450494a95fae11b75fb5a42.r2.dev/{subdirectory_name}/{readable_file_path}"

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
        raise Exception(f"Failed to download file. Status code: {response.status_code}")
