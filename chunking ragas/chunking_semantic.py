import os, re, numpy as np
from ingestion import load_documents
from sentence_transformers import SentenceTransformer

base_path = os.path.abspath("..")
md_texts = load_documents(base_path)

all_text = "\n".join(doc["text"] for doc in md_texts)
paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", all_text) if len(p.strip()) > 75]

model = SentenceTransformer("all-MiniLM-L6-v2")  # or all-mpnet-base-v2
embeds = model.encode(paragraphs, convert_to_numpy=True, normalize_embeddings=True)

threshold = 0.3  # raise to be stricter, lower to merge more
chunk_texts = []
current_text = paragraphs[0]
current_vec = embeds[0]

for text, vec in zip(paragraphs[1:], embeds[1:]):
    sim = float(np.dot(current_vec, vec))  
    if sim >= threshold:
        current_text = current_text + "\n\n" + text
        current_vec = (current_vec + vec) / np.linalg.norm(current_vec + vec)
    else:
        chunk_texts.append(current_text)
        current_text, current_vec = text, vec

chunk_texts.append(current_text) 
