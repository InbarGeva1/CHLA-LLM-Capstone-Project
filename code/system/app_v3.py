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

# Directories for CHLA and CDC vector stores
chla_dir = 'chla_vectorstore'
cdc_dir = 'cdc_vectorstore'

# Initialize embedding model
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Initialize CHLA vector store and retriever
chla_vectordb = Chroma(embedding_function=embedding, persist_directory=chla_dir)
chla_retriever = chla_vectordb.as_retriever(search_kwargs={'k': 1})

# Initialize embedding model
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Initialize CDC vector store and retriever
cdc_vectordb = Chroma(embedding_function=embedding, persist_directory=cdc_dir)
cdc_retriever = cdc_vectordb.as_retriever(search_kwargs={'k': 1})

# Function to extract URLs from text
def extract_url(text):
    if not isinstance(text, str):
        text = str(text)
    
    text = text.strip()
    pattern = r'https?://[^\s)\'"]+'
    urls = re.findall(pattern, text)
    urls = [url.rstrip(' \n)]\'\"') for url in urls]
    
    return urls

# Function to extract titles from text
def extract_title(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    pattern = r'IC\s*-\s*\d+\.\d+\s*[^.]+(?=\.txt)'
    match = re.search(pattern, text)
    
    if match:
        return match.group(0)
    
    return None

# Function to remove trailing newlines from text
def remove_trail(text):
    if not isinstance(text, str):
        text = str(text)
    
    new_text = re.sub(r'\n$', '', text)

    return(new_text)

# Define the prompt template for generating responses
prompt_template = PromptTemplate.from_template("""

### Previous Context

{previous_context}

### Current Context

CHLA Documentation: {chla_context}

### End of CHLA Context

CDC Documentation: {cdc_context}

### End of CDC Context

### Instructions:
You are a policy guidance chatbot for the Children's Hospital Los Angeles (CHLA) responsible for providing both CHLA and CDC policy guidance. 
You should only answer questions regarding CHLA IPC policy with supporting CDC guidance. 
If the CHLA or CDC context does not correctly answer the user query/question, tell the user that "I am unable to locate the relevant policy documentation relevant to your question. Please consult with CHLA's IPC." \n

If the current question is related to the previous context, answer the question directly based on the previous context. \n

Provide TWO separate, detailed, and thorough summaries for BOTH the CHLA DOCUMENTATION and CDC DOCUMENTATION that is faithful to the documentation above.
The answers to the question should each be sourced from the CHLA and CDC context respectively.
Maintain all medical terminology and ensure the response is clear. 
Use bullet points and step-by-step instructions for clarity when applicable. 

Attach the title of the document, {chla_title} and the static link at the end of the CHLA summary: https://chla.sharepoint.com/:f:/r/teams/LMUCHLACollaboration-T/Shared%20Documents/LLM%20Policy%20Bot%20Capstone/Infection%20Control?csf=1&web=1&e=kZAdVc \n
Remove brackets [] or backslash n from the link: {cdc_url} and attach this link to the end of the CDC summary.

### CHLA & CDC SUMMARY Example:

**CHLA Guidance:**

Summary based on CHLA context 
CHLA Citation Title and Link: 

**CDC Guidance:**

Summary based on CDC context 
CDC Citation Link: 

### End of Example

### User Query
Given this information, please provide me with an answer to the following: {input_text} 

""")

# Initialize the LLM
ollama_llm = Ollama(model="llama3.1", base_url="http://localhost:11434", temperature=0.01)
chain = prompt_template | ollama_llm | StrOutputParser()

# Define templates for generating context summaries
context_template_chla = PromptTemplate.from_template("""

### Context

{context}

### End Context

### Instructions
You are responsible for taking the input CHLA context and outputting a thorough and detailed summary, preserving all technical medical terminology and policies relevant to this question: {query} 
The provided summary should not answer the question, but instead provide all information relevant to the question. 
Do not provide any additional conversational response other than the requested output.
### End Instructions

""")

context_template_cdc = PromptTemplate.from_template("""

### Context
{context}

### End Context

### Instructions
You are responsible for taking the input CDC context and outputting a thorough and detailed summary, preserving all technical medical terminology and policies relevant to this question: {query} 
The provided summary should not answer the question, but instead provide all information relevant to the question. 
Do not provide any additional conversational response other than the requested output.

### End Instructions

""")

# Initialize LLM for context chains
context_llm = Ollama(model="llama3.1", base_url="http://localhost:11434", temperature=0.01)
context_chain_chla = context_template_chla | context_llm | StrOutputParser()

context_llm = Ollama(model="llama3.1", base_url="http://localhost:11434", temperature=0.01)
context_chain_cdc = context_template_cdc | context_llm | StrOutputParser()

# Paths to logo and icon images
logo_path = "childrens-hospital-la-logo.png"
icon_path = "childrens-hospital-la-icon.jpg"

# Function to initialize and run the chat interface
def boot():
    # Display the logo
    st.image(logo_path, width=300)

    # Display the title and description 
    st.title("IPC Policy Assistant")
    st.write("Welcome to the Children's Hospital Los Angeles IPC Chatbot. Ask a question regarding CHLA IPC policy!")

    # Initialize session state for storing messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

     # Display the conversation history
    for message in st.session_state.messages:
        if message[0] == "human":
            st.chat_message("human").write(message[1])
        else:
            st.chat_message("ai").write(message[1])

    # Handle user input for a query
    if query := st.chat_input("Type your message..."):
        st.session_state.messages.append(["human", query])
        st.chat_message("human").write(query)

        # Retrieve contexts from CHLA and CDC vector stores
        chla_context = chla_retriever.invoke(query)
        cdc_context = cdc_retriever.invoke(query)
        
        # Extract URLs and titles from the contexts
        cdc_urls = extract_url(cdc_context)
        chla_titles = extract_title(chla_context)
        cdc_url = remove_trail(cdc_urls)
        chla_title = remove_trail(chla_titles)

        # Generate context summaries
        chla_context = context_chain_chla.invoke({"context": chla_context, "query": query})
        cdc_context = context_chain_cdc.invoke({"context": cdc_context, "query": query})

        # Combine previous AI responses to form the previous context
        previous_context = "\n".join([msg[1] for msg in st.session_state.messages if msg[0] == "ai"])
        
        # Format the prompt with the relevant information
        combined_prompt = prompt_template.format(
            previous_context=previous_context,
            cdc_url=cdc_url,
            chla_title=chla_title,
            chla_context=chla_context,
            cdc_context=cdc_context,
            input_text=query
        )
        
        # Generate the response using the combined prompt
        response = chain.invoke({
            "previous_context": previous_context,
            "cdc_url": cdc_url,
            "chla_title": chla_title,
            "chla_context": chla_context,
            "cdc_context": cdc_context,
            "input_text": query
        })
        st.session_state.messages.append(["ai", response])

        # Stream the response for a better user experience
        def stream_data():
            for word in response.split(" "):
                yield word + " "
                time.sleep(0.04)

        st.chat_message("ai").write_stream(stream_data)

# Run the main function
if __name__ == "__main__":
    boot()

# Display the icon image at the bottom
st.image(icon_path, width=100)
