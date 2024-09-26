from helper import ocr_with_pytesseract


def main():
    # OCR with pytesseract
    ocred_text = ocr_with_pytesseract(
        "big_data_is_dead__motherduck_slides_slide_50.png"
    )
    print(ocred_text)


if __name__ == "__main__":
    main()
