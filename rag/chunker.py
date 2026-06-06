import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_DIR = os.path.join(BASE_DIR, 'data', 'textbooks/en')

import yaml

def load_config():
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

def load_textbooks():
    textbooks = []
    for filename in os.listdir(FILE_DIR):
        if filename.endswith('.txt'):
            filepath = os.path.join(FILE_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            textbooks.append({
                'filename': filename,
                'text': text
            })
    return textbooks

def chunk_text(text):
    words = text.split()
    chunks = []
    for start in range(0, len(words), CONFIG['rag']['chunk_size'] - CONFIG['rag']['overlap']):
        chunk = words[start : start + CONFIG['rag']['chunk_size']]
        chunks.append(" ".join(chunk))
    return chunks

def chunk_all_textbooks():
    textbooks = load_textbooks()
    all_chunks = []
    for book in textbooks:
        chunks = chunk_text(book['text'])
        for chunk in chunks:
            all_chunks.append({
                'source': book['filename'],
                'text' : chunk
            })
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all_textbooks()
    print(f"Total Chunks: {len(chunks)}")
    print(f"Sample Chunk: {chunks[0]}")
    print(f"Sample Chunk: {chunks[1]}")