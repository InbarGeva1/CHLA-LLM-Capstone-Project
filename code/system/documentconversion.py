import os
import docx
import fitz
import pypandoc

# destination
file_path = ""

class DocumentConverter:
    def __init__(self, directory):
        self.directory = directory

    def convert_docx_to_text(self, file_path):
        try:
            doc = docx.Document(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

        full_text = [para.text.strip() for para in doc.paragraphs]
        return "\n".join(full_text)

    def convert_pdf_to_text(self, file_path):
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

        full_text = [page.get_text() for page in doc]
        return "\n".join(full_text)

    def convert_doc_to_text(self, file_path):
        try:
            converted_file = file_path + ".converted.docx"
            pypandoc.convert_file(file_path, 'docx', outputfile=converted_file)
            text = self.convert_docx_to_text(converted_file)
            os.remove(converted_file)
            return text
        except Exception as e:
            print(f"Error converting {file_path}: {e}")
            return ""

    def save_text_to_file(self, file_path, text):
        text_file_path = file_path + ".txt"
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)
        return text_file_path

    def convert_all_to_text_files(self):
        for filename in os.listdir(self.directory):
            file_path = os.path.join(self.directory, filename)
            text = ""
            if filename.endswith(".docx"):
                text = self.convert_docx_to_text(file_path)
            elif filename.endswith(".pdf"):
                text = self.convert_pdf_to_text(file_path)
            elif filename.endswith(".doc"):
                text = self.convert_doc_to_text(file_path)

            if text:
                self.save_text_to_file(file_path, text)



directory = "/Users/andrewmorris/PycharmProjects/CHLA-LLM-Capstone-Project/sample"
converter = DocumentConverter(directory)
converter.convert_all_to_text_files()
