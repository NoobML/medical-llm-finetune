import sys
import os
import numpy as np

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from rag.embedder import load_embedder, embed_query
from rag.vector_store import load_chunks , load_vector_store


# load once at module level
embedder = load_embedder()
index = load_vector_store()
chunks = load_chunks()

def retrieve(query, k=5):
    query_vector = embed_query(embedder, query)
    query_vector = np.array([query_vector]).astype('float32')
    distances, indices = index.search(query_vector, k)

    results = []
    for i in indices[0]:
        results.append({
            'source': chunks[i]['source'],
            'text': chunks[i]['text']
        })
    return results


if __name__ == '__main__':
    query = "What is the first line treatment for UTI in pregnant women?"
    results = retrieve(query)
    for i, result in enumerate(results):
        print(f"\n ------ Result {i + 1} from {result['source']}")
        print(f"\n {result['text']}")
