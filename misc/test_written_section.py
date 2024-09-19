import modal


def main():
    f = modal.Function.lookup("video-to-blog-post", "write_section")

    paragraphs = {
        "batch_1_slide_1": "And we're doing some augmented reality stuff that that Mike's helping out with as well. So when you gift it, it's not just a sticker that says thank you Miss you love you Happy Valentine's, you scan a QR code. And literally cookies are popping up as if they were on your desk, you know. So it's pretty cool, we're still developing that same thing, you know, every week, we have new cookies, send a text message, choose your new cookies, we're going to put a link in that text message that says, like, see what the cookie looks like. So on, you know, you just put your phone on your counter. And then you can see the actual cookie being broken open, the size of it, what's inside the color, everything. So it's a much more interactive experience. With Mike bringing in the augmented reality or virtual reality side as well."
    }
    slide_id = "batch_1_slide_1"
    slide_text = "Augmented Reality for marketing"
    start = 0
    end = 10
    slide_text_list_elem = (slide_id, slide_text, start, end)

    subdirectory_name = "mother_duck_batch_1"
    markdown = f.remote(paragraphs, slide_text_list_elem, subdirectory_name)

    print(markdown)


if __name__ == "__main__":
    main()
