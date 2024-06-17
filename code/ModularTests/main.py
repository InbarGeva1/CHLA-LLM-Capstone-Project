import streamlit as st
from DataExtract import DocumentExtractor
from VectorSearch import FAISS
from transformers import LlamaForCausalLM, LlamaTokenizer
from PromptEng import PromptEng
import torch

# Load LLaMA3 
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def main():
    st.title("CHLA RAG Prototype")

    # User query
    user_prompt = st.text_input("Enter your query:")

    # User input for similarity threshold
    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7)

    if st.button("Search"):
        if user_prompt:
            # Extract texts from documents
            directory = "./sample"
            extractor = DocumentExtractor(directory)
            extracted_texts = extractor.extract_all_texts()

            # Initialize the document searcher with the extracted texts
            searcher = FAISS(extracted_texts)

            # Perform the search
            relevant_texts, similarities = searcher.search(user_prompt, similarity_threshold)

            # Display the relevant texts and their similarities
            st.subheader("Relevant Documents:")
            for text, similarity in zip(relevant_texts, similarities):
                st.markdown(f"**Similarity:** {similarity}")
                st.markdown(f"**Document:**\n{text}")
                st.markdown("---")

            # Generate a response using the LLaMA model with prompt engineering
            relevant_content = " ".join(relevant_texts)
            prompt_engineer = PromptEng(model, tokenizer, device)
            generated_response = prompt_engineer.process(relevant_content, user_prompt)

            st.subheader("Generated Response:")
            st.markdown(generated_response)
        else:
            st.error("Please enter a query.")


if __name__ == "__main__":
    main()
