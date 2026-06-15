import sys
import os
import numpy as np
import faiss
import pickle

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_PATH, 'results', 'embeddings', 'index.faiss')
CHUNKS_PATH = os.path.join(BASE_PATH, 'results', 'embeddings', 'chunks.pkl')
EMBEDDINGS_DIR = os.path.join(BASE_PATH, 'results', 'embeddings')
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
sys.path.append(BASE_PATH)

from rag.embedder import load_embedder, embed_chunks
from rag.chunker import chunk_all_textbooks


def build_vector_store(embedded_chunk):
    # extract just the embedding vector
    vectors = np.array([chunk['embedding'] for chunk in embedded_chunk]).astype('float32')
    dimension = vectors.shape[1] # 384 for miniLM
    index = faiss.IndexFlatIP(dimension) # IP = COSINSE SIMILARITY
    index.add(vectors)
    return index

def save_vector_store(index):
    faiss.write_index(index, VECTOR_STORE_PATH)

def load_vector_store():
    return faiss.read_index(VECTOR_STORE_PATH)

def save_chunks(chunks):
    with open(CHUNKS_PATH, 'wb') as f:
       pickle.dump(chunks, f)

def load_chunks():
    with open(CHUNKS_PATH, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    chunks = chunk_all_textbooks()
    embedder = load_embedder()
    embedded_chunks = embed_chunks(embedder, chunks)
    index = build_vector_store(embedded_chunks)
    save_vector_store(index)
    chunks_only = [{'source': c['source'], 'text': c['text']} for c in embedded_chunks]
    save_chunks(chunks_only)

    print(f"Vector store built with {index.ntotal} vectors")



