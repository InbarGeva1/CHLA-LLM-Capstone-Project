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

CDC Documentation: {cdc_context}

Do not give me an answer if it is not mentioned in the context as a fact. 
                                               
If the user asks a question regarding CHLA or CDC guidance on protocol, regulations, standard procedures or any other related information, provide a detailed response that is faithful to the documentation above. 
Provide separate paragraphs of summarization for the CHLA DOCUMENTATION and CDC DOCUMENTATION.
Maintain all medical terminology and ensure the response is clear and concise. 
Use bullet points and step-by-step instructions for clarity when applicable.

Attach this link at the end of every CHLA summary: https://chla.sharepoint.com/:f:/r/teams/LMUCHLACollaboration-T/Shared%20Documents/LLM%20Policy%20Bot%20Capstone/Infection%20Control?csf=1&web=1&e=kZAdVc
If CDC content is summarized, attach the link found at the very end of the CDC Documentation context.

Here is an example output structure:
**CHLA Guidance:**
detailed summary based on CHLA Documentation context
Source: **CHLA Citation Link**

**CDC Guidance:**
detailed summary based on CDC Documentation context
Source: **CDC Citation Link**

Here is an complete example output with example prompt:

Prompt: When can I discharge a patient who has active TB with smear-positive sputum?

Output:
  CHLA guidance: 
  Have three (3) consecutive negative AFB sputum smear or gastric aspirate results collected at least 8 hours apart with at least one collected early morning (note that specimens collected in early AM produce the best results);
  OR all of the following:
  Have completed at least two (2) weeks of multi-drug anti-tuberculosis therapy that is consistent with CDPH/CTCA "Guidelines for the Treatment of Tuberculosis and Tuberculosis Infection for California," (4/97); AND
  Exhibit clinical improvement (e.g. reduction in fever and cough); AND Have continued close medical supervision, including directly observed therapy
  (DOT), if needed; AND Continues multi-drug therapy, even if another pulmonary process is diagnosed, pending negative culture results from at least three (3) sputum or gastric aspirate specimens
  Reference: https://chla.sharepoint.com/:f:/r/teams/LMUCHLACollaboration-T/Shared%20Documents/LLM%20Policy%20Bot%20Capstone/Infection%20Control?csf=1&web=1&e=kZAdVc

CDC guidance:
  If a hospitalized patient who has suspected or confirmed TB disease is deemed medically stable (including patients with positive AFB sputum smear results indicating pulmonary TB disease), the patient can be discharged from the hospital before converting the positive AFB sputum smear results to negative AFB sputum smear results, if the following parameters have been met:
  a specific plan exists for follow-up care with the local TB-control program;
  the patient has been started on a standard multidrug antituberculosis treatment regimen, and DOT has been arranged;
  no infants and children aged <4 years or persons with immunocompromising conditions are present in the household;
  all immunocompetent household members have been previously exposed to the patient; and
  the patient is willing to not travel outside of the home except for health-care–associated visits until the patient has negative sputum smear results.
  Patients with suspected or confirmed infectious TB disease should not be released to health-care settings or homes in which the patient can expose others who are at high risk for progressing to TB disease if infected (e.g., persons infected with HIV or infants and children aged <4 years). Coordination with the local health department TB program is indicated in such circumstances.
  Reference: https://www.cdc.gov/mmwr/preview/mmwrhtml/rr5417a1.htm

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



