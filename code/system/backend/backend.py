import sys
import os
from langchain.llms import Ollama
# Add the path to the ModularTests folder
try:
    base_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(base_dir, "../../ModularTests"))
    sys.path.append(os.path.join(base_dir, "../../sample"))
except NameError:
    # Fallback to a direct path specification if __file__ is not available
    sys.path.append("/Users/andrewmorris/PycharmProjects/CHLA-LLM-Capstone-Project/code/ModularTests")

from DataExtract import TextExtractor
from VectorSearch import FAISS

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate


# Load LLaMA3
model = Ollama(model="llama3", base_url="http://10.3.8.195", temperature=0.3)


class QueryRequest(BaseModel):
    user_prompt: str
    similarity_threshold: float = 0.7

app = FastAPI()


def retrieve_documents(user_prompt, similarity_threshold=0.7):
    directory = "../../sample"
    extractor = TextExtractor(directory)
    extracted_texts = extractor.extract_all_texts()
    searcher = FAISS(extracted_texts)
    relevant_texts, similarities = searcher.search(user_prompt, similarity_threshold)
    relevant_content = " ".join(relevant_texts)
    return relevant_content

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

Attach this link at the end of the chla paragraph: https://lmu.app.box.com/file/1562757601538
Attach this link at the end of the CDC paragraph: https://www.cdc.gov/infection-control/hcp/surgical-site-infection/index.html

Answer:
""")



@app.post("/query/")
def query_documents(request: QueryRequest):
    user_prompt = request.user_prompt
    similarity_threshold = request.similarity_threshold
    relevant_content = retrieve_documents(user_prompt, similarity_threshold)
    generated_response = model.generate(prompt_template, user_prompt, context=relevant_content)

    return {"relevant_texts": relevant_content, "generated_response": generated_response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
