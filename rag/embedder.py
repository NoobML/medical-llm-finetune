import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from rag.chunker import chunk_all_textbooks

def load_embedder():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def embed_chunks(model, chunks):
    embedded_chunk = []
    for chunk in chunks:
        embedding = model.encode(chunk['text'])
        embedded_chunk.append({
            'source': chunk['source'],
            'text': chunk['text'],
            'embedding': embedding
        })
    return embedded_chunk

def embed_query(model, query):
    query_embedding = model.encode(query)
    return query_embedding


if __name__ == "__main__":
    model = load_embedder()
    chunks = chunk_all_textbooks()
    embeddings = embed_chunks(model, chunks)


