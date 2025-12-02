from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI() 

chat_model = "gpt-5-nano"

import os
from ingestion import load_documents

base_path = os.path.abspath("..")

md_texts = load_documents(base_path)

print(f"Loaded {len(md_texts)} documents.")

"""
It's more practical to chunk here as we will use this as the reference for the ground truth. To avoid longer evaluation time, it's better to have shorter chunks.
Moreover, reference should not have any irrelevant pieces of information in it. 

"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1500,
    chunk_overlap=100
)

all_chunks = []
for i, doc in enumerate(md_texts):
    chunks = splitter.create_documents([doc["text"]])
    for c in chunks:
        c.metadata = {"source": doc["file"], "doc_id": i+1}
    all_chunks.extend(chunks)

print(f"{len(all_chunks)} total chunks created")

""" The aim here is to get 'short answer questions' across the documents. Since theres 80 chunks, we use every 4th chunk to generate a question.
This way we know that the context is indeed relevant, and the question definitely has an answer. """
import json, re

qa_pairs = []

#function cleans the json schema
def safe_parse_json(text): #frankly i used stackoverflow + copilot for this one. The LLM kept adding words outside the JSON format, probably shouldve used response.format
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            j = match.group(0)
            try:
                return json.loads(j)
            except json.JSONDecodeError:
                j = re.sub(r',\s*}', '}', j)
                j = re.sub(r',\s*$', '', j)
                return json.loads(j)
    except:
        return {}

for i, chunk in enumerate(all_chunks, start=1):
    if i % 8 != 0:
        continue

    context = chunk.page_content.strip()
    
    prompt = (
    "You are assisting HR professionals and managers in understanding general workplace guidance. "
    "The text below provides universal principles and best practices—not information about any specific company. "
    "Using only the context, create ONE realistic and natural-sounding question that an employee might ask "
    "where the answer is long (300-500 words) about reproductive or fertility health policies, inclusivity, workplace support, or how such policies should be developed or applied. "
    "Then provide a factual report answer grounded entirely in the text. "
    "Respond with ONLY a valid JSON object containing these keys: "
    "'doc_id', 'question', 'ground_truth_answer', and 'context_used'. "
    "The 'context_used' must include the full text provided. "
    "No explanations, no markdown, no extra text—return strictly JSON.\n\n"
    f"Text:\n{context}"
)



    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        text = response.choices[0].message.content.strip()
        parsed = safe_parse_json(text)

        if parsed and all(k in parsed for k in ["question", "ground_truth_answer"]):
            parsed["doc_id"] = i
            parsed["context_used"] = context
            qa_pairs.append(parsed)

    except:
        continue

with open("qa_long_dataset.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)