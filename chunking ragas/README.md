# Reproductive Health RAG Evaluation

This project evaluates different chunking strategies for answering HR-related questions about reproductive and fertility health policies using Retrieval-Augmented Generation (RAG). 

questions.py produces the questions list, ground truth answer (used later for evaluation) and context used. It produces 2 jsons: one with more FAQ questions, and the other with answers that are around 500 words long.
ingestion.py is where the document ingestion code occurs.
rag.py contains the code for the hybrid retrieval method, including deduplication, searches, MMR and reranking. 
Begin with contextual as that is where most of the notes are.

## Setup Summary

- **LLM**: OpenAI GPT models (gpt 3.5-turbo and 4o-mini)
- **Embedding**: `voyage-context-3` (VoyageAI)
- **Retrieval**: Hybrid (BM25 + dense + MMR + reranking)
- **Evaluation**: [RAGAS](https://github.com/explodinggradients/ragas) metrics

## Parameters

- **Top-k**: 3
- **Alpha (α)**: 0.6 — balance between semantic vs keyword scores
- **Lambda (λ)**: 0.7 — balance between relevance and diversity (MMR)
- **Chunk Size**:
  - Short answers: ~1500 tokens (felt redundant to do 2000 tokens as we know it'll produce worse results)
  - Long answers: ~1500 tokens and ~2000 tokens

---

## Short Answer Evaluation

| Chunk Type   | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|--------------|--------------|------------------|----------------|-------------------|
| CTS          | 0.876        | 0.977            | 0.967          | 1.0000            |
| RCTS         | 0.885        | 0.969            | 0.975          | 1.0000            |
| Contextual   | 0.939        | 0.969            | 0.838          | 1.0000            |
| Semantic     | 0.936        | 0.973            | 0.817          | 1.0000            |

---

## Long Answer (500-word reports) Evaluation with 2000 Size Chunks

| Chunk Type   | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|--------------|--------------|------------------|----------------|-------------------|
| CTS          | 0.918        | 0.918            | 0.972          | 1.0000            |
| RCTS         | 0.879        | 0.918            | 0.975          | 1.0000            |
| Contextual   | 0.863        | 0.911            | 1.000          | 1.0000            |
| Semantic     | 0.879        | 0.918            | 0.975          | 1.0000            |

---

## Long Answer (500-word reports) Evaluation with 1500 Size Chunks

| Chunk Type   | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|--------------|--------------|------------------|----------------|-------------------|
| CTS          | 0.973        | 0.916            | 1.000          | 1.000             |
| RCTS         | 0.934        | 0.921            | 1.000          | 1.000             |
| Contextual   | 0.890        | 0.923            | 1.000          | 1.000             |
| Semantic     | 0.892        | 0.916            | 1.000          | 1.000             |




In short answer tasks, CTS and Semantic chunking offer the best performance for generation, with high faithfulness and answer relevancy. Semantic matches Contextual in faithfulness, while CTS edges out the rest in answer relevancy. For retrieval, CTS and RCTS lead in context recall, ensuring that relevant supporting information is retrieved consistently.

For long-form answers, CTS and RCTS dominate generation quality, particularly in faithfulness and relevancy across both 2000 and 1500 chunk sizes. Semantic also ranks highly in the 2000-size setting but slightly lags behind in the smaller chunks. Retrieval performance is strongest in the 2000 setting for Contextual, RCTS, and Semantic due to perfect recall, while in the 1500-token setup, all methods perform equally well with full recall scores.

Overall, RCTS has the highest average ranking. 

Note: The results may be biased because the model often relies on a single large chunk to generate each answer. This means even if multiple relevant chunks are retrieved, the model tends to use just one of them. As a result, scores like context precision and faithfulness can seem higher than they should be, since the correct answer happens to be in that one chunk.
This makes it harder to see the true benefit of chunking methods that are better at retrieving multiple relevant pieces. In real-world use, where information is often spread out, these methods might perform better than the evaluation shows.
