import os
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader

loader = DirectoryLoader('sample', glob='*.txt', loader_cls=TextLoader)

docs = loader.load()






