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
    # This pattern matches until it encounters a non-URL character or end of line
    pattern = r'https?://[^\s)\'"]+'
    
    # Find all matches of the pattern in the text
    urls = re.findall(pattern, text)

    # Strip trailing non-URL characters including whitespace characters from each URL
    urls = [url.rstrip(' \n)]\'\"') for url in urls]
    
    return urls

def extract_title(text):
    # Define the regex pattern to match the title before the .txt extension
    pattern = r'IC\s*-\s*\d+\.\d+\s*[^.]+(?=\.txt)'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    # If a match is found, return the matched string
    if match:
        return match.group(0)
    
    # If no match is found, return None
    return None

def remove_trail(text):
    if not isinstance(text, str):
        # Convert to string if it's not already a string
        text = str(text)
    
    new_text = re.sub(r'\n$', '', text)

    return(new_text)

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
If the CHLA or CDC context does not correctly answer the user query/question, tell the user that "I am unable to locate the relevant policy documentation relevant to your question. Please consult with CHLA's IPC." \n

Provide separate and detailed and thorough summaries for the CHLA DOCUMENTATION and CDC DOCUMENTATION that is faithfull to the documentation above.
The answers to the question should each be sourced from the CHLA and CDC context respectively.
Maintain all medical terminology and ensure the response is clear. 
Use bullet points and step-by-step instructions for clarity when applicable.  \n

Attach the title of the document, {chla_title} and the static link at the end of the CHLA summary: https://chla.sharepoint.com/:f:/r/teams/LMUCHLACollaboration-T/Shared%20Documents/LLM%20Policy%20Bot%20Capstone/Infection%20Control?csf=1&web=1&e=kZAdVc \n
Remove brackets [] or backslash n from the link: {cdc_url} and attach this link to the end of the CDC summary.

### Example:

**CHLA Guidance:**

Summary based on CHLA context \n
CHLA Citation Title and Link: 

**CDC Guidance:**

Summary based on CDC context \n
CDC Citation Link: 

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
You are responsible taking the input CHLA context and outputting a thorough and detailed summary, preserving all technical medical terminology and policies relevant to this question: {query} \n
The provided summary should not answer the question, but instead provide all information relevant to the question. \n
Do not provide any additional conversational response other than the requested output.
### End Instructions

""")

context_template_cdc = PromptTemplate.from_template("""

### Context
{context}

### End Context

### Instructions
You are responsible taking the input CDC context and outputting a thorough and detailed summary, preserving all technical medical terminology and policies relevant to this question: {query} \n
The provided summary should not answer the question, but instead provide all information relevant to the question. \n
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
        
        cdc_urls = extract_url(cdc_context)
        chla_titles = extract_title(chla_context)
        cdc_url = remove_trail(cdc_urls)
        chla_title = remove_trail(chla_titles_

        chla_context = context_chain_chla.invoke({"context": chla_context, "query": query})
        cdc_context = context_chain_cdc.invoke({"context": cdc_context, "query": query})

        combined_prompt = prompt_template.format(cdc_url=cdc_url, chla_title=chla_title, chla_context=chla_context, cdc_context=cdc_context, input_text=query)
        
        response = chain.invoke({"cdc_url": cdc_url, "chla_title": chla_title, "chla_context": chla_context, "cdc_context": cdc_context, "input_text": query})
        st.session_state.messages.append(["ai", response])

        def stream_data():
            for word in response.split(" "):
                yield word + " "
                time.sleep(0.04)

        st.chat_message("ai").write_stream(stream_data)


if __name__ == "__main__":
    boot()

st.image(icon_path, width=100)
