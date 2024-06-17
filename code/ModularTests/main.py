from DataExtract import DocumentExtractor
from VectorSearch import FAISS

directory = "./sample"

# Extract texts from documents
extractor = DocumentExtractor(directory)
extracted_texts = extractor.extract_all_texts()

# Initialize faiss
searcher = FAISS(extracted_texts)

# Define the user prompt 
user_prompt = "What are the infection control procedures for therapy dogs at CHLA?"
similarity_threshold = 0.7  # Define your similarity threshold here

# Perform the search
relevant_texts, similarities = searcher.search(user_prompt, similarity_threshold)

# Display the relevant texts and their similarities
for text, similarity in zip(relevant_texts, similarities):
    print(f"Similarity: {similarity}\nDocument:\n{text}\n{'-'*80}")
