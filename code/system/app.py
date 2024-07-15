import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

persist_dir = 'chla_vectorstore'
embedding = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

vectordb = Chroma(embedding_function=embedding, persist_directory=persist_dir)

retriever = vectordb.as_retriever()

retriever = vectordb.as_retriever(search_kwargs={'k': 2})



prompt_template = PromptTemplate.from_template("""
Documentation: {context}

User Question: {input_text}

Please provide a detailed and natural-sounding answer based on the documentation above. Provide separate paragraphs of summarization for the CHLA DOCUMENTATION and CDC DOCUMENTATION.
Maintain all medical terminology and ensure the response is clear and concise. Use bullet points and step-by-step instructions for clarity when applicable.
Only provide the summarizations using the following markdown format and begin by your response by saying:

**CHLA Recommendation:**
(newline)
summary based on chla context

**CDC Recommendation:**
(newline)
summary based on cdc context

Attach this link at the end of the chla paragraph: https://teams.microsoft.com/v2/
Attach the source URL from the cdc dosumentaion at the end of the CDC paragraph.
Answer:
""")

# Initialize the Ollama model
ollama_llm = Ollama(model="llama3", base_url="http://localhost:11434", temperature=0.3)

# Create the LLMChain
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
            st.chat_message("human").write_stream(message[1])
        else:
            st.chat_message("ai").write_stream(message[1])

    if query := st.chat_input("Type your message..."):
        st.session_state.messages.append(["human", query])
        st.chat_message("human").write(query)

        context = retriever.get_relevant_documents(query)
        combined_prompt = prompt_template.format(context=context, input_text=query)

        try:
            response = chain.run({"context": context, "input_text": query})
            st.session_state.messages.append(["ai", response])
        except Exception as e:
            response = f"An error occurred: {str(e)}"
            st.session_state.messages.append(["ai", response])

        st.chat_message("ai").write_stream(response)


if __name__ == "__main__":
    boot()

# Display the icon at the bottom
st.image(icon_path, width=100)



