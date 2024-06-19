from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import os
import sys

# Add the path to the ModularTests folder
sys.path.append(os.path.join(os.path.dirname(__file__), "../../ModularTests"))

from DataExtract import TextExtractor
from VectorSearch import LlamaIndex
from PromptEng import PromptEng

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


@app.post("/query/")
def query_documents(request: QueryRequest):
    # Extract texts
    directory = os.path.join(os.path.dirname(__file__), "../../sample")
    extractor = TextExtractor(directory)
    extracted_texts = extractor.extract_all_texts()

    # Vector search
    searcher = LlamaIndex(extracted_texts)
    relevant_texts, similarities = searcher.search(request.user_prompt, request.similarity_threshold)
    relevant_content = " ".join(relevant_texts)

    # Generate response
    prompt_eng = PromptEng(model, tokenizer, device)
    generated_response = prompt_eng.process(relevant_content, request.user_prompt)

    return {"relevant_texts": relevant_texts, "similarities": similarities, "generated_response": generated_response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
