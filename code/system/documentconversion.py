import os
import docx
import fitz
import pypandoc
import shutil

class DocumentConverter:
    def __init__(self, source_directory, destination_directory, exceptions_directory):
        self.source_directory = source_directory
        self.destination_directory = destination_directory
        self.exceptions_directory = exceptions_directory

        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
        if not os.path.exists(exceptions_directory):
            os.makedirs(exceptions_directory)

    def convert_docx_to_text(self, file_path):
        try:
            doc = docx.Document(file_path)
            full_text = [para.text.strip() for para in doc.paragraphs]
            return "\n".join(full_text)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            self.move_to_exceptions(file_path)
            return ""

    def convert_pdf_to_text(self, file_path):
        try:
            doc = fitz.open(file_path)
            full_text = [page.get_text() for page in doc]
            return "\n".join(full_text)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            self.move_to_exceptions(file_path)
            return ""

    def convert_doc_to_text(self, file_path):
        try:
            converted_file = file_path + ".converted.docx"
            pypandoc.convert_file(file_path, 'docx', outputfile=converted_file)
            text = self.convert_docx_to_text(converted_file)
            os.remove(converted_file)
            return text
        except Exception as e:
            print(f"Error converting {file_path}: {e}")
            self.move_to_exceptions(file_path)
            return ""

    def save_text_to_file(self, filename, text):
        text_file_path = os.path.join(self.destination_directory, filename + ".txt")
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)
        return text_file_path

    def move_to_exceptions(self, file_path):
        try:
            shutil.move(file_path, self.exceptions_directory)
            print(f"Moved {file_path} to {self.exceptions_directory}")
        except Exception as e:
            print(f"Error moving {file_path} to {self.exceptions_directory}: {e}")

    def convert_all_to_text_files(self):
        for filename in os.listdir(self.source_directory):
            file_path = os.path.join(self.source_directory, filename)
            text = ""
            if filename.endswith(".docx"):
                text = self.convert_docx_to_text(file_path)
            elif filename.endswith(".pdf"):
                text = self.convert_pdf_to_text(file_path)
            elif filename.endswith(".doc"):
                text = self.convert_doc_to_text(file_path)

            if text:
                base_filename = os.path.splitext(filename)[0]
                self.save_text_to_file(base_filename, text)


# Usage
chla_dir = "data/CHLA"
chla_destination = "data/CHLA_text"
bad_docs = "data/CHLA_exceptions"
converter = DocumentConverter(chla_dir, chla_destination, bad_docs)
converter.convert_all_to_text_files()

cdc_dir = "data/CDC"
cdc_destination = "data/CDC_text"
converter = DocumentConverter(cdc_dir, cdc_destination)
converter.convert_all_to_text_files()