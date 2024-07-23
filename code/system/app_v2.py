import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.globals import set_verbose, get_verbose
from langchain_core.output_parsers import StrOutputParser
import time
import re

chla_dir = 'chla_vectorstore'
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
chla_vectordb = Chroma(embedding_function=embedding, persist_directory=chla_dir)
chla_retriever = chla_vectordb.as_retriever(search_kwargs={'k': 1})

cdc_dir = 'cdc_vectorstore'
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
cdc_vectordb = Chroma(embedding_function=embedding, persist_directory=cdc_dir)
cdc_retriever = cdc_vectordb.as_retriever(search_kwargs={'k': 1})

def extract_url(text):
    if not isinstance(text, str):
        # Convert to string if it's not already a string
        text = str(text)
    
    # Strip trailing newline characters
    text = text.strip()
    
    # Define the regex pattern to match URLs starting with 'http' or 'https'
    pattern = r'https?://[^\s]+'
    
    # Find all matches of the pattern in the text
    urls = re.findall(pattern, text)
    
    return urls

prompt_template = PromptTemplate.from_template("""

### CHLA Context

CHLA Documentation: {chla_context}

### End of CHLA Context

### CDC Context

CDC Documentation: {cdc_context}

### End of CDC Context

### Instructions:
You are a policy guidance chatbot for the Children's Hospital Los Angeles (CHLA). 
You can only answer questions regarding CHLA IPC policy as well as supporting CDC guidance. 
If the context is not sufficient to answer the question, apologize and state that you were unable to retrieve the answer to the question. \n

Use the provided context to summarize the information and provide answers to the question.  
Please provide a thorough and detailed response that is faithfull to the documentation above. 
Provide separate and detailed summaries for the CHLA DOCUMENTATION and CDC DOCUMENTATION.
Maintain all medical terminology and ensure the response is clear. 
Use bullet points and step-by-step instructions for clarity when applicable. 
The answers to the question should each be sourced from the CHLA and CDC context respectively. \n

Attach this static link at the end of the CHLA summary: https://chla.sharepoint.com/:f:/r/teams/LMUCHLACollaboration-T/Shared%20Documents/LLM%20Policy%20Bot%20Capstone/Infection%20Control?csf=1&web=1&e=kZAdVc \n
Attach this link at the end of the CDC summary: {cdc_url}

### Example:

**CHLA Guidance:**

Summary based on CHLA context \n
CHLA Citation Link: https://chla.sharepoint.com/:f:/r/teams/LMUCHLACollaboration-T/Shared%20Documents/LLM%20Policy%20Bot%20Capstone/Infection%20Control?csf=1&web=1&e=kZAdVc

**CDC Guidance:**

Summary based on CDC context \n
CDC Citation Link: {cdc_url}

### End of Example


### User Query
Given this information, please provide me with an answer to the following: {input_text} 

""")

ollama_llm = Ollama(model="llama3", base_url="http://localhost:11434", temperature=0.01)
chain = prompt_template | ollama_llm | StrOutputParser()

context_template_chla = PromptTemplate.from_template("""

### Context

{context}

### End Context

### Instructions
You are responsible taking the input CHLA context and outputting a structured, cleaned output. Do not remove any text or information from the original document.
Do not provide any additional conversational response other than the requested output.
### End Instructions

""")

context_template_cdc = PromptTemplate.from_template("""

### Context
{context}

### End Context

### Instructions
You are responsible taking the input CDC context and outputting a structured, cleaned output. Do not remove any text or information from the original document.
Do not provide any additional conversational response other than the requested output.
### End Instructions

""")

context_llm = Ollama(model="llama3", base_url="http://localhost:11434", temperature=0.01)
context_chain_chla = context_template_chla | context_llm | StrOutputParser()

context_llm = Ollama(model="llama3", base_url="http://localhost:11434", temperature=0.01)
context_chain_cdc = context_template_cdc | context_llm | StrOutputParser()

logo_path = "childrens-hospital-la-logo.png"
icon_path = "childrens-hospital-la-icon.jpg"


def boot():
    st.image(logo_path, width=300)

    st.title("IPC Policy Assistant")
    st.write("Welcome to the Children's Hospital Los Angeles IPC Chatbot. Ask a question regarding CHLA IPC policy!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message[0] == "human":
            st.chat_message("human").write(message[1])
        else:
            st.chat_message("ai").write(message[1])

    if query := st.chat_input("Type your message..."):
        st.session_state.messages.append(["human", query])
        st.chat_message("human").write(query)

        chla_context = chla_retriever.invoke(query)
        cdc_context = cdc_retriever.invoke(query)

        cdc_url = extract_url(cdc_context)
        st.write("CHLA Context: ", chla_context)
        st.write("CDC URL: ", cdc_url)
        st.write("CDC Context: ", cdc_context)

        chla_context = context_chain_chla.invoke({"context": chla_context, "query": query})
        cdc_context = context_chain_cdc.invoke({"context": cdc_context, "query": query})

        combined_prompt = prompt_template.format(cdc_url=cdc_url, chla_context=chla_context, cdc_context=cdc_context, input_text=query)
        st.write("Combined Prompt: ", combined_prompt)
        
        response = chain.invoke({"cdc_url": cdc_url, "chla_context": chla_context, "cdc_context": cdc_context, "input_text": query})
        st.session_state.messages.append(["ai", response])

        def stream_data():
            for word in response.split(" "):
                yield word + " "
                time.sleep(0.04)

        st.chat_message("ai").write_stream(stream_data)


if __name__ == "__main__":
    boot()

st.image(icon_path, width=100)
