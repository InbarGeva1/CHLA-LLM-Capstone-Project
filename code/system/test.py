from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load CHLA vector store
chla_dir = 'chla_vectorstore'
chla_embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
chla_vectordb = Chroma(embedding_function=chla_embedding, persist_directory=chla_dir)

# Test retrieval
query = "What is the hand washing protocol for COVID-19?"
chla_retriever = chla_vectordb.as_retriever(search_kwargs={'k': 2})

# Debugging: Check if retriever is initialized correctly
print("CHLA Retriever Initialized:", chla_retriever)

# Debugging: Retrieve documents for the query
chla_context = chla_retriever.get_relevant_documents(query)
print("CHLA Context:", chla_context)

# Load CDC vector store
cdc_dir = 'cdc_vectorstore'
cdc_embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
cdc_vectordb = Chroma(embedding_function=cdc_embedding, persist_directory=cdc_dir)

# Test retrieval
cdc_retriever = cdc_vectordb.as_retriever(search_kwargs={'k': 2})

# Debugging: Check if retriever is initialized correctly
print("CDC Retriever Initialized:", cdc_retriever)

# Debugging: Retrieve documents for the query
cdc_context = cdc_retriever.get_relevant_documents(query)
print("CDC Context:", cdc_context)
