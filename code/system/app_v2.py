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

chla_dir = 'chla_vectorstore'
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
chla_vectordb = Chroma(embedding_function=embedding, persist_directory=chla_dir)
chla_retriever = chla_vectordb.as_retriever(search_kwargs={'k': 1})

cdc_dir = 'cdc_vectorstore'
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
cdc_vectordb = Chroma(embedding_function=embedding, persist_directory=cdc_dir)
cdc_retriever = cdc_vectordb.as_retriever(search_kwargs={'k': 1})

prompt_template = PromptTemplate.from_template("""

### Context
We have provided CHLA context information below: \n

CHLA Documentation: {chla_context} \n

We have provided CDC context information below, including citation link at the bottom: \n

CDC Documentation: {cdc_context} \n

### End of Context

### Instructions:
You are a policy guidance chatbot for the Children's Hospital Los Angeles (CHLA). \n

Use the provided context to summarize the information and provide answers to the question.                                    
Please provide a thorough and detailed response that is faithfull to the documentation above. 
Provide separate and detailed summaries for the CHLA DOCUMENTATION and CDC DOCUMENTATION.
Maintain all medical terminology and ensure the response is clear. 
Use bullet points and step-by-step instructions for clarity when applicable. 
The answers to the question should each be sourced from the CHLA and CDC context respectively. \n

Attach this static link at the end of the CHLA summary: https://chla.sharepoint.com/:f:/r/teams/LMUCHLACollaboration-T/Shared%20Documents/LLM%20Policy%20Bot%20Capstone/Infection%20Control?csf=1&web=1&e=kZAdVc
Attach the CDC citation link found in the the CDC Documentation context at the end of the CDC summary. 
If the link is not at the end of the CDC context, check the rest of the context for "Source URL: ". If no link is found, state that you were unable to locate the citation link for the context used in this response. \n

If the context is not sufficient to answer the question, apologize and state that you were unable to retrieve the answer to the question.

### Example:

**CHLA Guidance:**

Summary based on CHLA context \n
CHLA Citation Link:

**CDC Guidance:**

Summary based on CDC context \n
CDC Citation Link: 

### End of Example


### User Query
Given this information, please provide me with an answer to the following: {input_text} 

""")

ollama_llm = Ollama(model="llama3", base_url="http://localhost:11434", temperature=0.1)
chain = prompt_template | ollama_llm | StrOutputParser()

context_template = PromptTemplate.from_template("""
You are responsible taking the input context and outputting a structured, cleaned output. Do not remove any text or information from the original document.

In the output, preserve and return each CDC citation link at the end of each CDC documentation that begins with "Source URL: " at the end of the output.
If "Source URL: " appears multiple times (at the end of each document), preserve all of those reference links in the final output. 

{context}
""")

context_llm = Ollama(model="llama3", base_url="http://localhost:11434", temperature=0.1)
context_chain = context_template | context_llm | StrOutputParser()


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
        st.write("CHLA Context: ", chla_context)
        st.write("CDC Context: ", cdc_context)

        chla_context = context_chain.invoke({"context": chla_context, "query": query})
        cdc_context = context_chain.invoke({"context": cdc_context, "query": query})

        combined_prompt = prompt_template.format(chla_context=chla_context, cdc_context=cdc_context, input_text=query)
        st.write("Combined Prompt: ", combined_prompt)
        
        response = chain.invoke({"chla_context": chla_context, "cdc_context": cdc_context, "input_text": query})
        st.session_state.messages.append(["ai", response])

        def stream_data():
            for word in response.split(" "):
                yield word + " "
                time.sleep(0.04)

        st.chat_message("ai").write_stream(stream_data)


if __name__ == "__main__":
    boot()

st.image(icon_path, width=100)
