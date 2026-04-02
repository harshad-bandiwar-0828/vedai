import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    pdf = fitz.open(file_path)
    text = ""

    for page in pdf:
        text += page.get_text()

    return text


if __name__ == "__main__":
    file_path = "sample.pdf"
    extracted_text = extract_text_from_pdf("C:/Users/harsh/Downloads/SQL_CheatSheet.pdf")

    print("Extracted Text:\n")
    print(extracted_text[:1000])  # show first 1000 characters