from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings



loader = DirectoryLoader('sample', glob='*.txt', loader_cls=TextLoader)

docs = loader.load()

persist_dir = 'vectorstore'

embedding = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_dir)

vectordb.persist()

vectordb = Chroma(embedding_function=embedding, persist_directory=persist_dir)


retriever = vectordb.as_retriever()

query = "what are the surgical site infection policies?"

docs = retriever.get_relevant_documents(query)

retriever = vectordb.as_retriever(search_kwargs={'k': 2})

