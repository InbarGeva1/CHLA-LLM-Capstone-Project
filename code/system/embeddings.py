from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings 

# Load CHLA documents
chla_loader = DirectoryLoader('data/CHLA_TEXT_FULL', glob='*.txt', loader_cls=TextLoader)
chla_docs = chla_loader.load()

# Ensure documents are loaded correctly
if not chla_docs:
    print("No documents loaded from CHLA directory.")
else:
    print(f"{len(chla_docs)} documents loaded from CHLA directory.")
    # Print some documents to inspect content
    for doc in chla_docs[:5]:
        print("Document content:", doc)

# Persist CHLA vector store
chla_persist_dir = 'chla_vectorstore'
chla_embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
chla_vectordb = Chroma.from_documents(documents=chla_docs, embedding=chla_embedding, persist_directory=chla_persist_dir)

# Load CDC documents
cdc_loader = DirectoryLoader('data/CDC_TEXT_FULL', glob='*.txt', loader_cls=TextLoader)
cdc_docs = cdc_loader.load()

# Ensure documents are loaded correctly
if not cdc_docs:
    print("No documents loaded from CDC directory.")
else:
    print(f"{len(cdc_docs)} documents loaded from CDC directory.")
    # Print some documents to inspect content
    for doc in cdc_docs[:5]:
        print("Document content:", doc)

# Persist CDC vector store
cdc_persist_dir = 'cdc_vectorstore'
cdc_embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
cdc_vectordb = Chroma.from_documents(documents=cdc_docs, embedding=cdc_embedding, persist_directory=cdc_persist_dir)

print("Vector stores created and persisted successfully.")

