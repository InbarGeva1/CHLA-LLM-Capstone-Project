import streamlit as st
import sys
import os

try:
    base_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(base_dir, "../../ModularTests"))
except NameError:
    # Fallback to a direct path specification if __file__ is not available
    sys.path.append("/Users/andrewmorris/PycharmProjects/CHLA-LLM-Capstone-Project/code/ModularTests")

from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from DataExtract import TextExtractor
from VectorSearch import ChromaVectorSearch
from PromptEng import PromptEng
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Paths to your logo and icon
logo_path = "childrens-hospital-la-logo.png"
icon_path = "childrens-hospital-la-icon.jpg"

# Initialize the Ollama model
ollama_llm = Ollama(model="llama3", base_url="http://localhost:11434")

# Initialize TextExtractor
extractor = TextExtractor(directory="/path/to/your/documents")
extracted_texts = extractor.extract_all_texts()

# Initialize ChromaVectorSearch
vector_search = ChromaVectorSearch(extracted_texts)

# Initialize PromptEng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
prompt_eng = PromptEng(model, tokenizer, device)

# Define the updated prompt template
prompt_template = PromptTemplate.from_template("""
Context: {context}

User Question: {input_text}

Please provide a detailed and natural-sounding answer based on the context above. Maintain all medical terminology and ensure the response is clear and concise.

Answer:
""")

# Create the LLMChain
chain = LLMChain(llm=ollama_llm, prompt=prompt_template)


def generate_response(user_prompt):
    try:
        # Retrieve relevant context
        relevant_texts, _ = vector_search.search(user_prompt, similarity_threshold=st.session_state.slider)
        combined_context = "\n".join(relevant_texts)

        # Generate the combined prompt
        combined_prompt = prompt_eng.combine_prompt(combined_context, user_prompt)

        # Run the chain with the combined prompt
        response = chain.run({"context": combined_context, "input_text": user_prompt})
        return response
    except Exception as e:
        return "Apologies for the inconvenience. System still in progress. Come back soon!"


# Boot chat interface
def boot():
    # Display logo
    st.image(logo_path, width=300)

    # Title and description
    st.title("CHLA Chatbot Prototype")
    st.write("Welcome to the Children's Hospital Los Angeles Chatbot Prototype. Ask a question regarding CHLA policy!")

    # Initialize session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Similarity threshold for testing
    st.session_state.slider = st.slider("Similarity Threshold (For Testing Purposes)", 0.0, 1.0, 0.7)

    # Display the conversation history
    for message in st.session_state.messages:
        st.chat_message("human").write(message[0])
        st.chat_message("ai").write(message[1])

    # User input for query
    if query := st.chat_input("Type your message..."):
        st.session_state.messages.append([query, ""])  # Placeholder for bot response
        st.chat_message("human").write(query)
        bot_response = generate_response(query)
        st.session_state.messages[-1][1] = bot_response
        st.chat_message("ai").write(bot_response)


if __name__ == "__main__":
    boot()

# Display the icon at the bottom
st.image(icon_path, width=100)


