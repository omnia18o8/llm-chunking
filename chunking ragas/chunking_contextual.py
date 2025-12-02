import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion import load_documents

TOP_K = 3
GROUP_SIZE = 10


def batch_chunks(chunks, size):
    for i in range(0, len(chunks), size):
        yield chunks[i:i + size]


def build_contextual_chunks(client_vo, embedding_model, base_path=".."):
    md_texts = load_documents(os.path.abspath(base_path))

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=256,
        chunk_overlap=50
    )

    all_chunks = []
    for doc in md_texts:
        chunks = splitter.create_documents([doc["text"]])
        all_chunks.extend(chunks)

    chunk_groups = list(batch_chunks(all_chunks, GROUP_SIZE))
    inputs = [[c.page_content for c in group] for group in chunk_groups]

    response = client_vo.contextualized_embed(
        inputs=inputs,
        model=embedding_model,
        input_type="document"
    )


    expanded_texts = []
    embeddings_output = []

    for group, doc_result in zip(chunk_groups, response.results):
        raw_embeddings = [np.array(e, dtype=np.float32) for e in doc_result.embeddings]
        normed_embeddings = [v / np.linalg.norm(v) for v in raw_embeddings]

        texts = [c.page_content for c in group]

        for i, (chunk_text, vec) in enumerate(zip(texts, normed_embeddings)):
            sims = cosine_similarity([vec], normed_embeddings)[0]
            sims[i] = -1  
            top_k_idx = sims.argsort()[-TOP_K:]
            similar_texts = [texts[j] for j in top_k_idx]

            expanded = chunk_text + " " + " ".join(similar_texts)

            expanded_texts.append(expanded)
            embeddings_output.append(vec)

    chunk_embeddings = np.array(embeddings_output, dtype=np.float32)

    return expanded_texts, chunk_embeddings
