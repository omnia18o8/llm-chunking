import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
import voyageai as vo  
import os
from dotenv import load_dotenv
load_dotenv()

embedding_model = "voyage-context-3"

client_vo = vo.Client(api_key=os.getenv("VOYAGE_API_KEY")) 

# Instantiate reranker globally
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def dedupe_chunks(chunks, embeddings, threshold=0.92):
    embeddings = np.array(embeddings, dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    keep_idx = []
    keep_vecs = []

    for i, v in enumerate(embeddings):
        if not keep_vecs:
            keep_vecs.append(v)
            keep_idx.append(i)
            continue

        keep_mat = np.stack(keep_vecs)
        sims = np.dot(keep_mat, v)

        if np.max(sims) < threshold:
            keep_vecs.append(v)
            keep_idx.append(i)

    keep_vecs = np.stack(keep_vecs)
    return [chunks[i] for i in keep_idx], keep_vecs


def mmr(query_embedding, doc_embeddings, lambda_param=0.5, top_k=3):
    query_sim = cosine_similarity([query_embedding], doc_embeddings)[0]
    doc_sim = cosine_similarity(doc_embeddings)

    selected = []
    remaining = list(range(len(doc_embeddings)))

    for _ in range(top_k):
        if not remaining:
            break
        rel = query_sim[remaining]
        div = np.max(doc_sim[remaining][:, selected], axis=1) if selected else np.zeros(len(remaining))
        mmr_score = lambda_param * rel - (1 - lambda_param) * div
        chosen_idx = remaining[np.argmax(mmr_score)]
        selected.append(chosen_idx)
        remaining.remove(chosen_idx)

    return selected

def embed_query(text: str):
    return client_vo.contextualized_embed(
        inputs=[[text]],
        model=embedding_model,
        input_type="query"
    ).results[0].embeddings[0]


def build_bm25(texts):
    return BM25Retriever.from_texts(texts)


def build_context(idx, chunks):
    valid_idx = [i for i in idx if i < len(chunks)]
    if not valid_idx:
        return "No valid context available."
    return "\n\n".join(f"- {chunks[i]}" for i in valid_idx)


def build_prompt(context, query):
    return f"""
You are assisting HR professionals and managers in understanding general workplace guidance.
Use ONLY the provided information from the context below to answer the questions.

Documents:
{context}

Query:
{query}

Answer:
"""


def rag(
    query, bm25, chunk_embeddings, chunk_texts, embed_query,
    alpha=0.5, lambda_param=0.5, top_k=3
):
    q = embed_query(query)
    q = q / np.linalg.norm(q)
    chunk_norms = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    vec_scores = np.dot(chunk_norms, q)

    bm_docs = bm25.invoke(query)
    bm_scores = np.zeros(len(chunk_texts))
    for doc in bm_docs[:20]:
        text = doc.page_content
        try:
            idx = chunk_texts.index(text)
            bm_scores[idx] = getattr(doc, "score", 1.0)
        except ValueError:
            continue

    hybrid_scores = alpha * vec_scores + (1 - alpha) * bm_scores
    hybrid_top_idx = np.argsort(hybrid_scores)[-10:][::-1]
    candidate_embeds = chunk_embeddings[hybrid_top_idx]

    selected_local = mmr(q, candidate_embeds, lambda_param=lambda_param, top_k=top_k)
    mmr_idx = [hybrid_top_idx[i] for i in selected_local]
    mmr_texts = [chunk_texts[i] for i in mmr_idx]

    pairs = [(query, doc) for doc in mmr_texts]
    scores = reranker.predict(pairs)
    reranked_idx = np.argsort(scores)[::-1][:top_k]
    final_idx = [mmr_idx[i] for i in reranked_idx]

    return final_idx


def response_llm(
    questions, client, chat_model,
    chunk_embeddings, chunk_texts,
    embed_query, k=3, alpha=0.5, lambda_param=0.5
):
    bm25 = build_bm25(chunk_texts)
    llm_answers = []

    for i, query in enumerate(questions):
        print(f"\nQ{i+1}: {query}")

        try:
            top_idx = rag(
                query=query,
                bm25=bm25,
                chunk_embeddings=chunk_embeddings,
                chunk_texts=chunk_texts,
                embed_query=embed_query,
                alpha=alpha,
                lambda_param=lambda_param,
                top_k=k
            )
        except Exception as e:
            print(f"Retrieval failed for Q{i+1}: {e}")
            top_idx = []

        context = build_context(top_idx, chunk_texts) if top_idx else "No relevant context found."
        prompt = build_prompt(context, query)

        try:
            response = client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Model error: {e}"

        llm_answers.append({
            "query": query,
            "answer": answer,
            "context_used": context
        })

        print("Answer:", answer)

    return llm_answers


__all__ = ["response_llm", "embed_query"]
