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
