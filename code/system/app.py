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

CHLA Documentation: {chla_context}
CDC Documentation: {cdc_context}

User Question: {input_text}

You are a policy guidance chatbot for the Children's Hospital Los Angeles (CHLA).
                                               
If the user asks a question regarding CHLA or CDC guidance on protocals, regulations, standard procedures or any other related information, use the prompt template below.                                              

Here are a few examples of possible questions that meet this criteria:
1. When can I deisolate a patient who had a respiratory virus? 
2. What is the policy on sending a patient home who is being treated for active TB?
3. How should a room be cleaned after a patient who had C. difficile diarrhea?
4. Does a patient with MRSA require isolation?
5. I was just exposed to a patient with varicella. What do I do?

Please provide a detailed response that is faithfull to the documentation above. Provide separate paragraphs of summarization for the CHLA DOCUMENTATION and CDC DOCUMENTATION.
Maintain all medical terminology and ensure the response is clear and concise. Use bullet points and step-by-step instructions for clarity when applicable.
Only provide the summarizations using the following markdown format and begin by your response by saying:

**CHLA Guidance:**
(newline)
summary based on chla context

**CDC Guidance:**
(newline)
summary based on cdc context

Attach this link at the end of the chla paragraph: https://chla.sharepoint.com/:f:/r/teams/LMUCHLACollaboration-T/Shared%20Documents/LLM%20Policy%20Bot%20Capstone/Infection%20Control?csf=1&web=1&e=kZAdVc
For the cdc paragraph, attach the link found at the very end of the cdc_context.

for both links, use the following markdown format:
"Reference:" (Link)

If the the user asks a follow-up question or something where this response template is not applicable
act as a general chatbot and provide a direct response based on the context of the question.

Answer:
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



