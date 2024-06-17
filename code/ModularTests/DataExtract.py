import os
import docx
import fitz


class DocumentExtractor:
    def __init__(self, directory):
        self.directory = directory
        self.extracted_texts = {}

    def extract_text_from_docx(self, file_path):
        try:
            doc = docx.Document(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

        full_text = []
        capture = False
        purpose_section = False
        procedure_section = False

        for para in doc.paragraphs:
            text = para.text.strip()
            if text.startswith("PURPOSE"):
                capture = True
                purpose_section = True
                procedure_section = False
            if text.startswith("PROCEDURE"):
                capture = True
                procedure_section = True
                purpose_section = False
            if text.startswith("REFERENCES") or text.startswith("POLICY OWNER"):
                capture = False
            if capture:
                if purpose_section:
                    full_text.append(f"PURPOSE: {text}")
                    purpose_section = False  # To avoid repeating the section name
                elif procedure_section:
                    full_text.append(f"PROCEDURE: {text}")
                    procedure_section = False  # To avoid repeating the section name
                else:
                    full_text.append(text)

        return "\n".join(full_text)

    def extract_text_from_pdf(self, file_path):
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

        full_text = []
        for page in doc:
            full_text.append(page.get_text())
        return "\n".join(full_text)

    def extract_all_texts(self):
        for filename in os.listdir(self.directory):
            file_path = os.path.join(self.directory, filename)
            if filename.endswith(".docx"):
                self.extracted_texts[filename] = self.extract_text_from_docx(file_path)
            elif filename.endswith(".pdf"):
                self.extracted_texts[filename] = self.extract_text_from_pdf(file_path)
        return self.extracted_texts



