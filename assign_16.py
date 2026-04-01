'''
In this assignment, your knowledge base will come from one or more web pages instead. 

Build a small Python project that:
1. Accepts a list of URLs (at least 2 distinct pages on the same site or different sites; your choice, 
but must be publicly accessible and allowed to fetch for educational use). 
Example: course syllabus page + documentation page; or two Wikipedia articles.
2. Fetches each URL and extracts main text from HTML (strip scripts/styles/navigation as much as you reasonably can).
3. Chunks the text (same idea as class: sliding windows with overlap).
4. Embeds chunks and builds a vector index (e.g. FAISS) for retrieval.
5. Provides a simple interface.

For each user question, the program must:
Retrieve top-k chunks.
Print or display evidence (chunk text + which URL it came from).
Generate an answer conditioned on that context.
Share the GitHub Repository Link.

'''

import os
import numpy as np
import faiss 
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")


# 
URLS = [
"https://en.wikipedia.org/wiki/Machine_learning",
"https://en.wikipedia.org/wiki/Artificial_intelligence"
]

HEADERS = {"User-Agent": "Mozilla/5.0"}

# This model turns text into a 384 dimension list of numbers
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# A small, efficient 'brain' for answering questions
GEN_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'

DEFAULT_MAX_NEW_TOKENS = 200

# Setup local folders so we don't clof up our system drive
MODEL_CACHE = os.path.abspath('.cache/models')
os.makedirs(MODEL_CACHE, exist_ok = True)
os.environ.setdefault('HF_HOME', os.path.abspath('.cache/huggingface'))


def _fetch_main_text(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    return ' '.join([p.get_text() for p in soup.find_all('p')])


def load_documents(source_urls : list[str] | None = None) -> tuple[list[str], list[str]]:
    urls = URLS or source_urls
    docs = []
    url_out = []

    for url in urls:
        try:
            text = _fetch_main_text(url)
            if text.strip():
                docs.append(text)
                url_out.append(url)
        except Exception as exc:
            print(f"Warning: could not fetch {url}: {exc}")

    return docs, url_out

# Data Preprocessing 
def load_all_text(source_urls: list[str] | None = None, chunk_size=500, overlap_size=30):

    chunks_out = []
    sources_out = []

    docs, urls = load_documents(source_urls)

    if len(docs) == 0:
        return [], []
    
    for doc_idx, doc in enumerate(docs):
        words = doc.split()
        url = URLS[doc_idx] if doc_idx < len(urls) else f"Source {doc_idx+1}"

        i = 0
        while i < len(words):
            snippet = ' '.join(words[i : i+chunk_size])
            if snippet.strip():
                chunks_out.append(snippet)
                sources_out.append(url)
            
            i += chunk_size - overlap_size
    
    return chunks_out, sources_out

# --- 3. THE SEARCH ENGINE (INDEXING) ---
def build_index(chunks: list[str], embedder: SentenceTransformer):

    """Converts text chunks into a searchable mathematical index."""
    # Step 1: Turn text into numbers (Embeddings)
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    
    # Step 2: Initialize FAISS (L2 = Euclidean distance / 'straight line' distance)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Step 3: Add the numbers to the index
    index.add(embeddings.astype("float32"))
    return index

def retrieval(query, embedder, index, chunks, sources, k = 3):
    if not chunks:
        return []
    
    query_vec = embedder.encode([query], convert_to_numpy=True).astype('float32')
    k_eff = min(k, len(chunks))
    distances, indices = index.search(query_vec, k_eff)

    results = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx <0 or idx >= len(chunks):
            continue
        results.append({
            'text' : chunks[idx],
            'source' : sources[idx],
            'score' : float(np.exp(-dist)) # Convert distance to a 0-1 similarity score
        })
    return results

# --5. The AI Brain (Generation) ---

def answer_question(question:  str, context_hits, generator, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
    ''' Sends the context and question to the LLM for a final answer.'''
    if not context_hits:
        context_str = 'No relevant information found in the documents.'
    else:
        # Trim chunk text so prompt processing stays fast.
        context_parts = [f"Source: {h['source']} \nContent: {h['text'][:700]}" for h in context_hits]
        context_str = "\n---\n".join(context_parts)

    # The "System Prompt" tells the AI how to behave
    prompt = (
        "You are a helpful assistant. Use the provided context to answer the question. "
        "If the answer isn't in the context, say you don't know. Be concise.\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    response = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        truncation=True,
        return_full_text=False
    )

    return response[0]["generated_text"].strip()


# --- 6. MAIN PROGRAM ---
def main():
    print("Step 1: Loading and Chunking PDFs...")
    chunks, sources = load_all_text()
    
    if not chunks:
        print(f"Error: No text found. Please put url in the URL section.")
        return

    print(f"Step 2: Creating embeddings for {len(chunks)} chunks from {len(set(sources))} web page(s)...")
    embedder = SentenceTransformer(EMBED_MODEL)
    index = build_index(chunks, embedder)

    print("Step 3: Loading AI Generation Model (this may take a minute)...")
    tokenizer = AutoTokenizer.from_pretrained(
        GEN_MODEL,
        cache_dir=MODEL_CACHE,
        token=HF_TOKEN
    )
    
    # Fix for certain models that don't have a clear "stop" or "padding" token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        cache_dir=MODEL_CACHE,
        device_map="auto", # Automatically uses GPU if available
        torch_dtype="auto",
        token=HF_TOKEN
    )
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print("\n--- System Ready! ---")
    while True:
        user_input = input("\nAsk a question (or type 'exit'): ").strip()
        if user_input.lower() in ["exit", "quit", ""]:
            break
            
        # 1. Retrieve
        hits = retrieval(user_input, embedder, index, chunks, sources)
        
        # 2. Generate
        answer = answer_question(user_input, hits, generator)
        
        print(f"\n[Sources used:]")
        for url in set(h['source'] for h in hits):
            print(f"  - {url}")
        print(f"AI: {answer}")


if __name__ == "__main__":
    main()