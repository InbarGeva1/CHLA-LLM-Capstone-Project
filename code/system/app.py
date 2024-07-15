import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

chla_dir = 'chla_vectorstore'
embedding = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

chla_vectordb = Chroma(embedding_function=embedding, persist_directory=chla_dir)
chla_retriever = chla_vectordb.as_retriever(search_kwargs={'k': 2})

cdc_dir = 'cdc_vectorstore'
embedding = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
cdc_vectordb = Chroma(embedding_function=embedding, persist_directory=cdc_dir)
cdc_retriever = cdc_vectordb.as_retriever(search_kwargs={'k': 2})

prompt_template = PromptTemplate.from_template("""

In the below prompt template, if something starts with "Instruction:", that means it is instruction for you, the policy chatbot, and that text should be masked (hidden) from the user asking the question.
In the below prompt template, if something starts with "Template:", that means it is the structure of the output for you, the policy chatbot, and that text should be masked (hidden) from the user asking the question.

Instruction: You are a policy chatbot for Children's Hospital Los Angeles designed to assist CHLA staff in quickly retreiving necessary information relating to IPC documentation, procedures, and policy. 
Instruction: If you are asked about policy information that cannot be found for CHLA or CDC, please state that you do not have access to either CHLA, CDC, or all of that documentation. 

Instruction: If the user asks a question about policy procedure, protocols, or regulations, use the prompt and template below.
Instruction: Otherwise, if the user asks questions unrelated to policy procedure, protocols, or regulations, DO NOT use the below prompt and template and instead act as a CHLA chatbot in a healthcare setting that can answer accurate, detailed, and interpretable responses.

User Question: {input_text}

CHLA Documentation: {chla_context}
CDC Documentation: {cdc_context}

Instruction: Only use the below if the question is about policy procedure, protocols, or regulations
Instruction: Please provide a detailed and natural-sounding answer based on the documentation above. Provide separate paragraphs of summarization for the CHLA DOCUMENTATION and CDC DOCUMENTATION.
Instruction: Maintain all medical terminology and ensure the response is clear and concise. Use bullet points and step-by-step instructions for clarity when applicable. Do not return information if you are unable to also attach the relevant citation link.
Instruction: Only provide the summarizations using the following markdown format and begin by your response by saying:

Template:
**CHLA Recommendation:**
(newline)
summary based on chla context
Instruction: Attach this link at the end of the chla paragraph: https://chla.sharepoint.com/:f:/r/teams/LMUCHLACollaboration-T/Shared%20Documents/LLM%20Policy%20Bot%20Capstone/Infection%20Control?csf=1&web=1&e=kZAdVc

**CDC Recommendation:**
(newline)
summary based on cdc context
Instruction: Attach the source URL from the cdc documentaion at the end of the CDC paragraph.

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

        st.chat_message("ai").write(response)


if __name__ == "__main__":
    boot()

# Display the icon at the bottom
st.image(icon_path, width=100)



