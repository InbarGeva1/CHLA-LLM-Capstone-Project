from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

class FAISS:
    def __init__(self, extracted_texts, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.texts = list(extracted_texts.values())
        self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)
        self.embeddings_np = self.embeddings.cuda().numpy()
        
        dimension = self.embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings_np)
        
    def search(self, user_prompt, similarity_threshold=0.7):
        query_embedding = self.model.encode([user_prompt], convert_to_tensor=True).cuda().numpy()
        D, I = self.index.search(query_embedding, len(self.texts))
        
        # Convert distances to similarities
        similarities = 1 - D[0] / 2  
        
        # Filter based on similarity threshold
        relevant_indices = [index for index, similarity in enumerate(similarities) if similarity >= similarity_threshold]
        relevant_texts = [self.texts[index] for index in relevant_indices]
        
        return relevant_texts, similarities[relevant_indices]
