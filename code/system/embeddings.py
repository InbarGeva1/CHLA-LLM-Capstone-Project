import os
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader

loader = DirectoryLoader('sample', glob='*.txt', loader_cls=TextLoader)

docs = loader.load()



class TextExtractor:
    def __init__(self, directory):
        self.directory = directory
        self.extracted_texts = {}

    def extract_text_from_txt(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def extract_all_texts(self):
        for filename in os.listdir(self.directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.directory, filename)
                text = self.extract_text_from_txt(file_path)
                self.extracted_texts[filename] = text
        return self.extracted_texts

# extraction
directory = "/Users/andrewmorris/PycharmProjects/CHLA-LLM-Capstone-Project/sample"
extractor = TextExtractor(directory)
extracted_texts = extractor.extract_all_texts()

# Print extracted texts
#for filename, text in extracted_texts.items():
    #print(f"Text from {filename}:\n{text}\n")

