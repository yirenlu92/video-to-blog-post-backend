import modal


def main():
    list_of_image_urls = [
        f"https://pub-f1ee73dd9450494a95fae11b75fb5a42.r2.dev/bigdataisdeadmotherduck/batch_1/slides/slide_batch_1_count_{i}.png"
        for i in range(0, 30, 5)
    ]

    f = modal.Function.lookup("video-to-blog-post", "read_text_from_slides_with_openai")

    json_response = f.remote(list_of_image_urls)

    print(json_response)


if __name__ == "__main__":
    main()
