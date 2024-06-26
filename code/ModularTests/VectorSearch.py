from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import chromadb
from llama_index.core import VectorStoreIndex, Document


class FAISS:
    def __init__(self, extracted_texts, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.texts = list(extracted_texts.values())
        self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)
        self.embeddings_np = self.embeddings.cpu().numpy()

        dimension = self.embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings_np)

    def search(self, user_prompt, similarity_threshold=0.7):
        query_embedding = self.model.encode([user_prompt], convert_to_tensor=True).cpu().numpy()
        D, I = self.index.search(query_embedding, len(self.texts))

        # Convert distances to similarities
        similarities = 1 - D[0] / 2

        # Filter based on similarity threshold
        relevant_indices = [index for index, similarity in enumerate(similarities) if
                            similarity >= similarity_threshold]
        relevant_texts = [self.texts[index] for index in relevant_indices]

        return relevant_texts, similarities[relevant_indices]


class ChromaVectorSearch:
    def __init__(self, extracted_texts, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.texts = list(extracted_texts.values())
        self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)
        self.embeddings_np = self.embeddings.cpu().numpy()

        self.client = chromadb.Client()
        self.collection = self.client.create_collection("documents")

        for i, (text, embedding) in enumerate(zip(self.texts, self.embeddings_np)):
            self.collection.add(
                documents=[text],
                metadatas=[{"id": i}],
                embeddings=[embedding.tolist()]
            )

    def search(self, user_prompt, similarity_threshold=0.7):
        query_embedding = self.model.encode([user_prompt], convert_to_tensor=True).cpu().numpy().tolist()[0]

        # Search documents
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=len(self.texts),
            include_embeddings=False
        )

        relevant_texts = []
        relevant_similarities = []
        for result in results['documents'][0]:
            similarity = result['score']
            if similarity >= similarity_threshold:
                relevant_texts.append(result['document'])
                relevant_similarities.append(similarity)

        return relevant_texts, relevant_similarities

    class LlamaIndex:
        def __init__(self, extracted_texts, model_name='paraphrase-MiniLM-L6-v2'):
            self.model = SentenceTransformer(model_name)
            self.texts = list(extracted_texts.values())
            self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)
            self.embeddings_np = self.embeddings.cpu().numpy()

            # Convert texts to Document objects
            self.documents = [Document(text=text) for text in self.texts]

            # Create the LlamaIndex
            self.index = VectorStoreIndex(documents=self.documents, embedding_model=self.model)

        def search(self, user_prompt, similarity_threshold=0.7):
            query_embedding = self.model.encode([user_prompt], convert_to_tensor=True).cpu().numpy().tolist()[0]

            # Perform search
            query = {
                "text": user_prompt,
                "embedding": query_embedding
            }
            results = self.index.query(query, top_k=len(self.texts))

            relevant_texts = []
            relevant_similarities = []
            for result in results:
                similarity = result.similarity
                if similarity >= similarity_threshold:
                    relevant_texts.append(result.document.text)
                    relevant_similarities.append(similarity)

            return relevant_texts, relevant_similarities
