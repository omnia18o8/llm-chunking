# Chunking RAG Evaluation

Chunking is a common step when preparing documents for retrieval in AI systems. It means breaking large documents into smaller, meaningful pieces that are easier to embed, search, and pass to an LLM. This report looks at how different chunking strategies affect retrieval performance in a way that reflects real RAG usage.

Even though modern LLMs can handle large context windows, sometimes entire documents, putting everything into the prompt isn’t efficient and can distract the model. Most queries only need a small part of the text, but the model still processes every token you include. Because of that, a good retrieval system should focus on pulling in only the *relevant* tokens for each query [Chroma Research — Evaluating Chunking](https://research.trychroma.com/evaluating-chunking?utm_source=chatgpt.com). The report focuses on the following chunking methods:

chunking.ipynb: Notebook to run/evaluate multiple chunking strategies side by side (imports the chunking scripts, embeds, runs RAG).

chunking_cts.py: Code to produce chunk_texts using CharacterTextSplitter (fixed-size token chunks).

chunking_rcts.py: Code to produce chunk_texts using RecursiveCharacterTextSplitter (hierarchical splitter).

chunking_semantic.py: Code to produce chunk_texts via paragraph splits; can be extended for semantic merges.

chunking_contextual.py: Code to produce all_chunks with smaller splits (for contextual embeddings); used by contextual pipelines.

ingestion.py: Loads source documents (PDF/DOCX/PPTX) into memory for the evaluation runs.

rag.py: Retrieval-and-generation pipeline: BM25 + embedding hybrid, MMR, cross-encoder rerank, and LLM response helper.

questions.py: Holds question sets used for evaluation/runs.

qa_dataset.json / qa_long_dataset.json: Ground-truth Q/A datasets for scoring. There are 20 FAQ like questions, and 4 report like answers. Of course this represents a limitation in terms of accuracy, but evaluating these types of questions require an expensive amount of tokens. 

## RAGAS Evaluation

This project evaluates different chunking strategies for answering HR-related questions about reproductive and fertility health policies using Retrieval-Augmented Generation (RAG). RAG performance is measured through a set of metrics that highlight different parts of the system, from how well retrieval works to how reliable the final answer is. For this, we use **RAGAS**, an evaluation framework built around LLM-based metrics for both generation quality and retrieval quality.

Evaluation uses the official **[RAGAS](https://github.com/explodinggradients/ragas)** metrics. We measure four key elements:
- **Faithfulness** — measured by asking an LLM to verify whether each part of the generated answer can be supported by the retrieved context. If statements cannot be grounded, the score decreases.
- **Answer Relevancy** — measured by comparing the generated answer to the original question using an LLM-based similarity check to see how directly and accurately the answer responds to the query.
- **Context Precision** — measured by having an LLLM examine each retrieved chunk and judge whether it is relevant to the question. Precision increases when fewer irrelevant chunks are retrieved.
- **Context Recall** — measured by checking whether the retrieved chunks contain all the information needed to answer the question. If important information is missing, recall drops.




## Parameters

### LLM Models

We use two different models in the evaluation pipeline—one for generating answers and one for judging them. This separation keeps costs manageable while ensuring that the evaluation remains reliable.

**Generator (chat_evaluator_model = gpt-5-nano)** - A lightweight model is used for answer generation to keep both token usage and latency low. The goal here is not to produce the best possible answer, but to provide a consistent, economical way to test how retrieval performs under realistic constraints. Using a smaller model also exposes weaknesses in retrieval more clearly—if the generator can only rely on retrieved context, it becomes easier to detect when retrieval fails.

**Judge (chat_judge_model = gpt-4.1-mini)** - The judge model is intentionally stronger, since evaluation depends on its ability to correctly assess faithfulness, relevance, and context usage. More capable judge models produce higher-quality critiques, reducing false positives (incorrectly saying an answer is faithful) and false negatives (penalizing correct answers). A stronger judge also handles nuanced policy language better, which is essential when scoring HR-related questions where wording, scope, and exceptions matter.```



### RAG Parameters

These three parameters control how hybrid search behaves and how clean the retrieved context is before being passed to the model.

- **alpha (0.6)** — Tilts the hybrid score toward embeddings while still allowing BM25 to contribute. Pure dense (1.0) misses exact-term matches, and pure BM25 (0.0) misses semantic links. A range of 0.5–0.7 is standard; 0.6 favors semantic similarity without losing lexical precision.

- **lambda_param (0.7, MMR)** — Controls relevance vs. diversity. Higher values keep the most relevant chunks; lower values increase variety. With reranking applied later, 0.7 keeps the core relevant chunks while reducing redundancy. A value like 0.5 would diversify more but risk pulling in less relevant items.

- **deduplication threshold (0.92 cosine)** — Removes near-duplicate chunks after normalization. Values above 0.9 catch repeated or heavily overlapping passages while preserving meaningful variations. Lower thresholds may delete useful context; higher ones allow duplicates to slip through.




### Chunk Size and Top_k 

To choose the right parameters, the first thing to look at is **chunk size**, since it directly affects how well the system handles both long, detailed questions and short, FAQ-style prompts. Longer questions typically benefit from larger chunks because they carry more context, while shorter questions perform better with smaller, more focused chunks. If chunks are too long, the model may get distracted by irrelevant details and hallucinate; if they’re too short, the context becomes fragmented and the model may struggle to form a complete answer.

There are two common strategies. Using **large chunks with a small top-k** works well for long-form questions because it preserves richer context, but it tends to reduce precision for short, fact-based queries. On the other hand, using **small chunks with a large top-k** is ideal for FAQs and direct questions, but it can break up the context too much for longer answers, making it harder for the model to reconstruct meaning. A practical middle ground is to use a medium chunk size and adjust the number of retrieved chunks depending on the query type.

In general RAG practice, chunk sizes between **128–512 tokens** are widely used. Smaller chunks (128–256 tokens) improve precise retrieval, while larger chunks (256–512 tokens) provide stronger context for conceptual questions [MILVUS](https://milvus.io/ai-quick-reference/what-is-the-optimal-chunk-size-for-rag-applications)

Recent work reinforces this trade-off. According to *ChunkRAG* (https://arxiv.org/pdf/2407.01219), a chunk size of **512 tokens** achieves the highest **faithfulness**, while **256 tokens** gives the best **answer relevancy**. Since faithfulness is more important for this project, we choose **512 tokens** as the base chunk size. Because our contextualised chunking method groups similar chunks and retrieves the top three, a size of 128 (512/4) ensures each chunk maintains enough context while still benefiting from grouping.

To keep things simple for now, instead of building an adaptive top-k agent, we will fix the chunk size and vary only the number of retrieved chunks. We use **top-k = 5** for short, FAQ-style questions and **top-k = 10** for longer, context-heavy ones. In practice, FAQs will now rely on documents around **1000 tokens**, while long-form answers may require up to **5000 characters** of contextual material. We will later evaluate how well this setup performs across both query types.



### Embedding Model

To embed both the chunks and the queries, we use two models from the Voyage API: **voyage-context-3** and **voyage-large-3**. Since their behaviour can differ depending on the retrieval setup, I tested both directly to see which one performs better for this project.

`voyage-context-3` is specifically designed for retrieval tasks because it incorporates document-level information into each chunk embedding. This gives it a clear advantage in most retrieval benchmarks, where it outperforms `voyage-large-3` by **7.96%** at the chunk level and **2.70%** at the document level in recall. Another benefit is that `voyage-context-3` is much less sensitive to chunk size: on document-level retrieval it shows only **2.06% variance**, compared to **4.34%** for `voyage-large-3`. With very small chunks—such as **64 tokens**—`voyage-context-3` performs especially well, beating `voyage-large-3` by **6.63%**. As chunk sizes get larger the gap narrows, but context-3 still maintains more stable performance overall.

One important limitation is that `voyage-large-3` cannot be used with contextualised chunking, so it is excluded from that part of the evaluation. Overall, because of its stability across chunk sizes and its stronger retrieval signal, `voyage-context-3` is the preferred model for this setup.


## Context 3 vs. Large 3 Metrics

| Chunk Type | Dataset    | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|------------|------------|--------------|------------------|-------------------|----------------|
| CTS        | Context 3  | 0.980        | 0.918            | 1.000             | 1.000          |
| CTS        | Large 3    | 0.990        | 0.922            | 1.000             | 1.000          |
| RCTS       | Context 3  | 0.996        | 0.927            | 1.000             | 1.000          |
| RCTS       | Large 3    | 0.993        | 0.920            | 1.000             | 1.000          |
| Semantic   | Context 3  | 0.986        | 0.926            | 1.000             | 0.986          |
| Semantic   | Large 3    | 0.995        | 0.925            | 1.000             | 0.965         |


Although Large 3 shows small gains in faithfulness for CTS and Semantic, Context 3 performs more consistently across all metrics, especially for RCTS and Semantic, where recall and relevancy hold stronger.


## Evaluation 

### Short Answer – Generation Metrics
For short answer generation, **RCTS** performs best in **faithfulness** (0.982), which is expected because the questions were generated from RCTS chunks — giving it a built-in advantage. However, **Contextual** comes in a close second (0.974), showing it can still produce accurate and relevant answers despite the evaluation bias. Since all methods score similarly on **answer relevancy**, faithfulness becomes the main factor, and Contextual's strong showing highlights its robustness even when it's not the “native” chunker.

### Short Answer – Retrieval Metrics
For short answer retrieval, **Contextual** is the most balanced overall. It ranks second in **context precision** (0.850, tied with Semantic), second in **context recall** (0.890), and second in **faithfulness**. While **CTS** leads in precision (0.950), it’s more rigid and can miss broader context. On the other hand, **RCTS** includes too much irrelevant content (lowest precision: 0.800). **Contextual** finds the middle ground—retrieving high-quality, faithful chunks with minimal noise, making it ideal for retrieval use cases.

### Long Answer – Generation Metrics
In long answer generation, **Semantic** performs the best overall. It ranks second in **faithfulness** (0.986) and **answer relevancy** (0.926), while maintaining **perfect precision** and near-perfect recall. This suggests that its semantic chunking boundaries support coherent and complete answer generation. While **RCTS** scores highest in faithfulness (0.996), this is again inflated by question alignment. **Contextual** also performs very well (faithfulness: 0.983), making it a strong alternative, especially for more flexible or non-biased tasks.

### Long Answer – Retrieval Metrics
For long answer retrieval, **all methods perform near-perfectly**, with **1.000 context precision** and **context recall between 0.986 and 1.000**. Because of this, there's no clear leader. However, **Semantic** and **Contextual** stand out by pairing high **faithfulness** with this retrieval accuracy, showing they consistently return relevant and complete information. While RCTS performs well, its lead is again partially due to the biased question set. Overall, **Semantic** is the best for long-form generation, and **Contextual** is the most balanced for both short and long-form retrieval.

### Short Answer Metrics
| Chunk Type   | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|--------------|--------------|------------------|-------------------|----------------|
| CTS          | 0.970        | 0.965            | 0.950             | 0.900          |
| RCTS         | 0.982        | 0.967            | 0.800             | 0.900          |
| Semantic     | 0.957        | 0.966            | 0.850             | 0.767          |
| Contextual   | 0.974        | 0.966            | 0.850             | 0.890          |
### Long Answer Metrics
| Chunk Type   | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|--------------|--------------|------------------|-------------------|----------------|
| CTS          | 0.980        | 0.918            | 1.000             | 1.000          |
| RCTS         | 0.996        | 0.927            | 1.000             | 1.000          |
| Semantic     | 0.986        | 0.926            | 1.000             | 0.986          |
| Contextual   | 0.983        | 0.923            | 1.000             | 0.995          |

