import streamlit as st
from DataExtract import TextExtractor
from VectorSearch import LlamaIndex
from transformers import LlamaForCausalLM, LlamaTokenizer
from PromptEng import PromptEng
import torch
from langchain import Chain, Memory, Prompt, LangChain

# Load LLaMA3
auth_token = "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"
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = LlamaTokenizer.from_pretrained(model_name, auth_token=auth_token)
model = LlamaForCausalLM.from_pretrained(model_name, auth_token=auth_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

memory = Memory()
prompt = Prompt()

# function to create langchain
def create_chain():
    chain = Chain(
        steps=[
            {"type": "input", "name": "user_prompt"},
            {"type": "custom", "name": "retrieve_documents", "function": retrieve_documents},
            {"type": "custom", "name": "generate_response", "function": generate_response},
        ],
        memory=memory,
    )
    return chain

def retrieve_documents(user_prompt, similarity_threshold=0.7):
    # Extract text
    directory = "./sample"
    extractor = TextExtractor(directory)
    extracted_texts = extractor.extract_all_texts()

    # Initialize chroma
    searcher = LlamaIndex(extracted_texts)

    # Perform search
    relevant_texts, similarities = searcher.search(user_prompt, similarity_threshold)
    relevant_content = " ".join(relevant_texts)

    return relevant_content

def generate_response(relevant_content, user_prompt):
    # gnerate response with Llama and prompteng
    prompt = PromptEng(model, tokenizer, device)
    generated_response = prompt.process(relevant_content, user_prompt)
    return generated_response

def main():
    st.title("CHLA Chatbot Prototype")

    # User query
    user_prompt = st.text_input("Enter your query:")

    # User input for similarity threshold
    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7)

    if st.button("Search"):
        if user_prompt:
            # start the chain
            chain = create_chain()
            result = chain.run({"user_prompt": user_prompt, "similarity_threshold": similarity_threshold})

            # Display relevant documents and the generated response
            st.subheader("Relevant Documents:")
            for text in result['retrieve_documents']:
                st.markdown(f"**Document:**\n{text}")
                st.markdown("---")

            st.subheader("Generated Response:")
            st.markdown(result['generate_response'])
        else:
            st.error("Please enter a query.")

if __name__ == "__main__":
    main()