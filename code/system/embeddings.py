from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings

chla_loader = DirectoryLoader('data/CHLA_MARKDOWN', glob='./*.md', loader_cls=UnstructuredMarkdownLoader)

docs = chla_loader.load()

persist_dir = 'chla_vectorstore'

embedding = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_dir)

vectordb.persist()


cdc_loader = DirectoryLoader('data/CDC_text', glob='*.txt', loader_cls=TextLoader)

docs = cdc_loader.load()

persist_dir = 'cdc_vectorstore'

embedding = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_dir)

vectordb.persist()

