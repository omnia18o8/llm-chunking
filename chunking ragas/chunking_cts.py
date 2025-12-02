from langchain_text_splitters import CharacterTextSplitter

character_text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=512,
    chunk_overlap=50,
)

all_chunks = []

import os
from ingestion import load_documents

base_path = os.path.abspath("..")
md_texts = load_documents(base_path)

for doc in md_texts:
    chunks = character_text_splitter.create_documents([doc["text"]])
    for c in chunks:
        c.metadata = {"source": doc["file"]} 
    all_chunks.extend(chunks)


chunk_texts = [c.page_content for c in all_chunks]