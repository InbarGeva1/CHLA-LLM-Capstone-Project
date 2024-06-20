import sys
import os

# Add the path to the ModularTests folder
try:
    base_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(base_dir, "../../ModularTests"))
except NameError:
    # Fallback to a direct path specification if __file__ is not available
    sys.path.append("/Users/andrewmorris/PycharmProjects/CHLA-LLM-Capstone-Project/code/ModularTests")

from DataExtract import TextExtractor
from VectorSearch import ChromaVectorSearch
from PromptEng import PromptEng

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Load LLaMA3
auth_token = "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"
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=auth_token)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))


class QueryRequest(BaseModel):
    user_prompt: str
    similarity_threshold: float = 0.7

app = FastAPI()


def retrieve_documents(user_prompt, similarity_threshold=0.7):
    directory = "../../sample"
    extractor = TextExtractor(directory)
    extracted_texts = extractor.extract_all_texts()
    searcher = ChromaVectorSearch(extracted_texts)
    relevant_texts, similarities = searcher.search(user_prompt, similarity_threshold)
    relevant_content = " ".join(relevant_texts)
    return relevant_content

def generate_response(relevant_content, user_prompt):
    prompt = PromptEng(model, tokenizer, device)
    generated_response = prompt.process(relevant_content, user_prompt)
    return generated_response

def chain_logic(user_prompt, similarity_threshold):
    # Step 1: Retrieve documents
    relevant_content = retrieve_documents(user_prompt, similarity_threshold)
    # Step 2: Generate response
    generated_response = generate_response(relevant_content, user_prompt)
    return relevant_content, generated_response


@app.post("/query/")
def query_documents(request: QueryRequest):
    user_prompt = request.user_prompt
    similarity_threshold = request.similarity_threshold

    relevant_content, generated_response = chain_logic(user_prompt, similarity_threshold)

    return {"relevant_texts": relevant_content, "generated_response": generated_response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
