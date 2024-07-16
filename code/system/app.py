import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import time

chla_dir = 'chla_vectorstore'
embedding = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
chla_vectordb = Chroma(embedding_function=embedding, persist_directory=chla_dir)
chla_retriever = chla_vectordb.as_retriever(search_kwargs={'k': 2})

cdc_dir = 'cdc_vectorstore'
embedding = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
cdc_vectordb = Chroma(embedding_function=embedding, persist_directory=cdc_dir)
cdc_retriever = cdc_vectordb.as_retriever(search_kwargs={'k': 2})

prompt_template = PromptTemplate.from_template("""

You are a policy guidance chatbot for the Children's Hospital Los Angeles (CHLA).

We have provided context information below. 

CHLA Documentation: {chla_context}
CHLA CITATION LINK: https://chla.sharepoint.com/:f:/r/teams/LMUCHLACollaboration-T/Shared%20Documents/LLM%20Policy%20Bot%20Capstone/Infection%20Control?csf=1&web=1&e=kZAdVc

CDC Documentation: {cdc_context}
Within the CDC Documentation context above is the CDC CITATION LINK for this context identified by "Source URL:". Find this link in the context and use this as the citation for CDC guidance relevant to this context.

Do not give me an answer if it is not mentioned in the context as a fact. 
                                               
If the user asks a question regarding CHLA or CDC guidance on protocol, regulations, standard procedures or any other related information, provide a detailed response that is faithful to the documentation above. 
Provide separate paragraphs of summarization for the CHLA DOCUMENTATION and CDC DOCUMENTATION.
Each summary should be followed by the corresponding CHLA CITATION LINK for CHLA DOCUMENTATION, and CDC CITATION LINK for CDC DOCUMENTATION.
Maintain all medical terminology and ensure the response is clear and detailed. 
Use bullet points and step-by-step instructions for clarity when applicable.

Here is an example output structure:
**CHLA Guidance:**
detailed summary based on CHLA Documentation context
Source: **CHLA Citation Link**

**CDC Guidance:**
detailed summary based on CDC Documentation context
Source: **CDC Citation Link**

Given this information, please provide me with an answer to the following: {input_text}

""")

ollama_llm = Ollama(model="llama3", base_url="http://localhost:11434", temperature=0.3)

chain = LLMChain(llm=ollama_llm, prompt=prompt_template)

logo_path = "childrens-hospital-la-logo.png"
icon_path = "childrens-hospital-la-icon.jpg"



def boot():
    st.image(logo_path, width=300)

    st.title("IPC Policy Assitant")
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

        chla_context = chla_retriever.get_relevant_documents(query)
        cdc_context = cdc_retriever.get_relevant_documents(query)
        combined_prompt = prompt_template.format(chla_context=chla_context, cdc_context=cdc_context,input_text=query)

        try:
            response = chain.run({"chla_context": chla_context, "cdc_context": cdc_context, "input_text": query})
            st.session_state.messages.append(["ai", response])
        except Exception as e:
            response = f"An error occurred: {str(e)}"
            st.session_state.messages.append(["ai", response])

        def stream_data():
            for word in response.split(" "):
                yield word + " "
                time.sleep(0.04)

        st.chat_message("ai").write_stream(stream_data)


if __name__ == "__main__":
    boot()

st.image(icon_path, width=100)



