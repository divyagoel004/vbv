# import os
# import faiss
# import pickle
# import numpy as np
# from bs4 import BeautifulSoup
# from sentence_transformers import SentenceTransformer
# from langchain_community.utilities import GoogleSerperAPIWrapper
 
# # Constants
# SERPER_API_KEY = os.getenv("SERPER_API_KEY","4dd51fc28ee7339f4993df71c7e3247cc8faaf6b")
# VECTOR_DB_PATH = "vector_store.faiss"
# INDEX_METADATA_PATH = "doc_metadata.pkl"
# CONTENT_TYPES = ["blogs", "articles", "case studies", "strategies"]
# HEADERS = {
#     "User-Agent": (
#         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#         "AppleWebKit/537.36 (KHTML, like Gecko) "
#         "Chrome/115.0.0.0 Safari/537.36"
#     )
# }
# from langfuse import Langfuse
 
# langfuse = Langfuse(
#     public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
#     secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
#     host="https://cloud.langfuse.com",
# )
 
# # Embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")
 
# def serper_search(topic, max_results_per_type=5):
#     trace = langfuse.trace(
#         name="serper_search",
#         input={"topic": topic, "max_results_per_type": max_results_per_type}
#     )
#     os.environ["SERPER_API_KEY"] = SERPER_API_KEY
#     search = GoogleSerperAPIWrapper()
   
#     all_results = []
 
#     for ctype in CONTENT_TYPES:
#         query = f"{topic} {ctype}"
#         print(f"[LangChain Serper] Searching: {query}")
 
#         search_span = trace.span(
#             name="search_content_type",
#             input={"query": query, "type": ctype}
#         )
#         try:
#             search_result = search.results(query)
#             organic = search_result.get("organic", [])
#             urls = []
#             for r in organic[:max_results_per_type]:
#                 title = r.get("title", "")
#                 snippet = r.get("snippet", "")
#                 link = r.get("link", "")
#                 link_span = search_span.span(
#                     name="search_result",
#                     input={"title": title, "link": link, "snippet": snippet}
#                 )
#                 link_span.end()
#                 urls.append(link)
#                 all_results.append((title, snippet, link, ctype))
           
#         except Exception as e:
#             search_span.output = {"error": str(e)}
#             print(f"[ERROR] Failed search for '{query}': {e}")
#         search_span.end()
 
   
 
#     text_blocks = []
#     metadata_blocks = []
 
#     for title, snippet, url, ctype in all_results:
#         print(f"[Processing] {ctype.upper()} → {url}")
 
#         # scrape_span = trace.span(name="scrape_url", input=url)
#         text = extract_text_from_url(url)
#         # scrape_span.update(output={"text_length": len(text) if text else 0})
#         # scrape_span.end()
 
#         if not text or len(text) < 300:
#             text = f"{title}\n{snippet}"
 
#         if len(text) >= 300:
#             from textwrap import wrap
#             chunks = wrap(text, 800)
#             for chunk in chunks:
#                 text_blocks.append(chunk)
#                 metadata_blocks.append({
#                     "type": ctype,
#                     "source": url,
#                     "summary": chunk[:300]
#                 })
 
#     if text_blocks:
#         print("[Vector Store] Storing extracted documents...")
#         # store_span = trace.span(name="store_vector_db", input={"num_blocks": len(text_blocks)})
#         store_in_vector_db(text_blocks, metadata_blocks)
#         # store_span.update(output={"status": "stored"})
#         # store_span.end()
#     else:
#         print("[Warning] No valid content found.")
#         # trace.update(output={"warning": "no_valid_content"})
 
   
 
# def extract_text_from_url(url, timeout=10):
#     try:
#         import requests
#         res = requests.get(url, headers=HEADERS, timeout=timeout)
#         if res.status_code != 200:
#             return None
#         soup = BeautifulSoup(res.text, "html.parser")
#         paragraphs = soup.find_all("p")
#         text = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
#         return text
#     except Exception as e:
#         print(f"[ERROR] Scraping failed at {url}: {e}")
#         return None
 
# def store_in_vector_db(text_blocks, metadata_blocks):
#     if os.path.exists(VECTOR_DB_PATH):
#         index = faiss.read_index(VECTOR_DB_PATH)
#         with open(INDEX_METADATA_PATH, "rb") as f:
#             metadata = pickle.load(f)
#     else:
#         index = faiss.IndexFlatL2(384)
#         metadata = []
 
#     embeddings = model.encode(text_blocks)
#     index.add(np.array(embeddings).astype("float32"))
#     metadata.extend(metadata_blocks)
 
#     faiss.write_index(index, VECTOR_DB_PATH)
#     with open(INDEX_METADATA_PATH, "wb") as f:
#         pickle.dump(metadata, f)
 
# from collections import defaultdict
# from textwrap import shorten
 
# def query_vector_db(user_query, top_k=10, chunk_limit=500, rerank_k=30):
#     """
#     Query FAISS DB and return concise, merged results with LLM reranking.

#     Args:
#         user_query (str): Search query.
#         top_k (int): Number of unique documents to return.
#         chunk_limit (int): Max characters per document summary.
#         rerank_k (int): Number of candidates to fetch before LLM reranking.
#     """
#     if not os.path.exists(VECTOR_DB_PATH):
#         return ["[ERROR] Vector DB is empty. Please run a search first."]

#     # Load FAISS index and metadata
#     index = faiss.read_index(VECTOR_DB_PATH)
#     with open(INDEX_METADATA_PATH, "rb") as f:
#         metadata = pickle.load(f)

#     # Encode query & search (over-fetch for LLM reranking)
#     query_embedding = model.encode([user_query]).astype("float32")
#     D, I = index.search(query_embedding, rerank_k)

#     # Merge chunks per source
#     docs = defaultdict(list)
#     for idx in I[0]:
#         if idx < len(metadata):
#             entry = metadata[idx]
#             docs[entry['source']].append(entry['summary'])

#     # Combine & trim
#     candidates = []
#     for source, chunks in docs.items():
#         combined_text = " ".join(chunks)
#         concise_text = shorten(combined_text, width=chunk_limit, placeholder="...")
#         candidates.append(f"{source}\n\n{concise_text}")

#     # --- LLM-based reranking ---
#     if len(candidates) > top_k:
#         from openai import OpenAI
#         client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#         docs_text = "\n\n".join([f"[{i}] {doc}" for i, doc in enumerate(candidates)])
#         prompt = f"""
#         You are a retrieval reranker. Rank the following documents by how well they answer the query.

#         Query: {user_query}

#         Documents:
#         {docs_text}

#         Return the indices of the top {top_k} most relevant documents, sorted in order of relevance.
#         """

#         resp = client.chat.completions.create(
#             model="gpt-4o-mini",  # or gpt-4o
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0
#         )
#         ranked_text = resp.choices[0].message.content.strip()

#         # Extract indices from LLM output (basic parse)
#         import re
#         indices = [int(x) for x in re.findall(r"\d+", ranked_text)]
#         indices = [i for i in indices if i < len(candidates)]
#         if len(indices) == 0:
#             return candidates[:top_k]  # fallback
#         return [candidates[i] for i in indices[:top_k]]

#     # If fewer than top_k, return directly
#     return candidates[:top_k]
import os
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain_community.utilities import GoogleSerperAPIWrapper
from collections import defaultdict
from textwrap import shorten
import requests
import streamlit as st
from datetime import datetime

# Constants
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "4dd51fc28ee7339f4993df71c7e3247cc8faaf6b")
CONTENT_TYPES = ["blogs", "articles", "case studies", "strategies"]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}

from langfuse import Langfuse

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com",
)

# Embedding model - initialize once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

class InMemoryVectorStore:
    """In-memory vector store using cosine similarity"""
    
    def __init__(self):
        self.embeddings = []
        self.metadata = []
        self.texts = []
    
    def add_documents(self, texts, metadata_list):
        """Add documents to the vector store"""
        if not texts:
            return
            
        # Generate embeddings
        new_embeddings = model.encode(texts)
        
        # Store everything in memory
        self.embeddings.extend(new_embeddings)
        self.metadata.extend(metadata_list)
        self.texts.extend(texts)
    
    def similarity_search(self, query, k=10):
        """Search for similar documents"""
        if not self.embeddings:
            return []
        
        # Encode query
        query_embedding = model.encode([query])
        
        # Calculate cosine similarities
        embeddings_array = np.array(self.embeddings)
        similarities = np.dot(embeddings_array, query_embedding.T).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return results with metadata
        results = []
        for idx in top_indices:
            results.append({
                'text': self.texts[idx],
                'metadata': self.metadata[idx],
                'score': similarities[idx]
            })
        
        return results
    
    def clear(self):
        """Clear all stored documents"""
        self.embeddings = []
        self.metadata = []
        self.texts = []
    
    def get_stats(self):
        """Get store statistics"""
        return {
            'total_documents': len(self.texts),
            'total_embeddings': len(self.embeddings),
            'metadata_count': len(self.metadata)
        }

# Initialize global vector store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore()

def categorize_source_type(url, title="", content=""):
    """Categorize source type based on URL and content"""
    url_lower = url.lower()
    title_lower = title.lower()
    
    if any(domain in url_lower for domain in ['arxiv.org', 'ieee.org', 'acm.org', 'springer.com']):
        return 'Academic Paper'
    elif any(domain in url_lower for domain in ['wikipedia.org', 'wiki']):
        return 'Wikipedia'
    elif any(domain in url_lower for domain in ['github.com', 'gitlab.com']):
        return 'Code Repository'
    elif any(domain in url_lower for domain in ['medium.com', 'blog', 'dev.to']):
        return 'Blog'
    elif any(domain in url_lower for domain in ['stackoverflow.com', 'stackexchange.com']):
        return 'Q&A Forum'
    elif any(word in title_lower for word in ['tutorial', 'guide', 'how to']):
        return 'Tutorial'
    elif any(domain in url_lower for domain in ['news', 'reuters.com', 'bbc.com']):
        return 'News Article'
    elif any(domain in url_lower for domain in ['docs.', 'documentation']):
        return 'Documentation'
    else:
        return 'Web Article'

def serper_search(topic, max_results_per_type=5):
    """Enhanced search function that stores source metadata in session state"""
    trace = langfuse.trace(
        name="serper_search",
        input={"topic": topic, "max_results_per_type": max_results_per_type}
    )
    
    # Initialize research sources list if not exists
    if 'research_sources' not in st.session_state:
        st.session_state.research_sources = []
    
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY
    search = GoogleSerperAPIWrapper()
    
    all_results = []

    for ctype in CONTENT_TYPES:
        query = f"{topic} {ctype}"
        print(f"[LangChain Serper] Searching: {query}")

        search_span = trace.span(
            name="search_content_type",
            input={"query": query, "type": ctype}
        )
        try:
            search_result = search.results(query)
            organic = search_result.get("organic", [])
            
            for r in organic[:max_results_per_type]:
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                link = r.get("link", "")
                
                link_span = search_span.span(
                    name="search_result",
                    input={"title": title, "link": link, "snippet": snippet}
                )
                link_span.end()
                
                all_results.append((title, snippet, link, ctype))
                
        except Exception as e:
            search_span.output = {"error": str(e)}
            print(f"[ERROR] Failed search for '{query}': {e}")
        search_span.end()

    text_blocks = []
    metadata_blocks = []

    for title, snippet, url, ctype in all_results:
        print(f"[Processing] {ctype.upper()} → {url}")

        # Extract full content
        text = extract_text_from_url(url)
        
        if not text or len(text) < 300:
            text = f"{title}\n{snippet}"

        # Store source information for the research sources display
        source_info = {
            'title': title,
            'url': url,
            'content': text,
            'snippet': snippet,
            'type': categorize_source_type(url, title, text),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'word_count': len(text.split()) if text else 0,
            'topic': topic,
            'content_type': ctype
        }
        
        # Add to research sources for display
        st.session_state.research_sources.append(source_info)

        # Process for vector store
        if len(text) >= 300:
            from textwrap import wrap
            chunks = wrap(text, 800)
            for i, chunk in enumerate(chunks):
                text_blocks.append(chunk)
                metadata_blocks.append({
                    "type": ctype,
                    "source": url,
                    "summary": chunk[:300],
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source_type": categorize_source_type(url, title, text)
                })

    if text_blocks:
        print(f"[Vector Store] Storing {len(text_blocks)} document chunks...")
        store_in_vector_db(text_blocks, metadata_blocks)
    else:
        print("[Warning] No valid content found.")
        trace.update(output={"warning": "no_valid_content"})

def extract_text_from_url(url, timeout=10):
    """Extract text content from URL"""
    try:
        res = requests.get(url, headers=HEADERS, timeout=timeout)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from paragraphs
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        
        # Fallback to all text if no paragraphs found
        if not text:
            text = soup.get_text()
            # Clean up whitespace
            text = ' '.join(text.split())
        
        return text
    except Exception as e:
        print(f"[ERROR] Scraping failed at {url}: {e}")
        return None

def store_in_vector_db(text_blocks, metadata_blocks):
    """Store documents in the in-memory vector store"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = InMemoryVectorStore()
    
    st.session_state.vector_store.add_documents(text_blocks, metadata_blocks)
    
    stats = st.session_state.vector_store.get_stats()
    print(f"[Vector Store] Total documents: {stats['total_documents']}")

def query_vector_db(user_query, top_k=10, chunk_limit=500, rerank_k=30):
    """
    Query in-memory vector DB and return concise, merged results with optional LLM reranking.
    """
    if 'vector_store' not in st.session_state or not st.session_state.vector_store.embeddings:
        return ["[ERROR] Vector DB is empty. Please run a search first."]

    # Search vector store
    search_results = st.session_state.vector_store.similarity_search(user_query, k=rerank_k)
    
    if not search_results:
        return ["[WARNING] No relevant results found."]

    # Merge chunks per source
    docs = defaultdict(list)
    for result in search_results:
        metadata = result['metadata']
        source = metadata['source']
        docs[source].append({
            'text': result['text'],
            'score': result['score'],
            'metadata': metadata
        })

    # Combine and format results
    candidates = []
    for source, chunks in docs.items():
        # Sort chunks by score (highest first)
        chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
        
        # Combine text from chunks
        combined_text = " ".join([chunk['text'] for chunk in chunks])
        concise_text = shorten(combined_text, width=chunk_limit, placeholder="...")
        
        # Get metadata from first chunk (highest score)
        first_chunk = chunks[0]
        source_type = first_chunk['metadata'].get('source_type', 'Unknown')
        title = first_chunk['metadata'].get('title', 'Untitled')
        
        formatted_result = f"**{title}** ({source_type})\nSource: {source}\n\n{concise_text}"
        candidates.append(formatted_result)

    # Optional LLM reranking (if you have OpenAI key and want to use it)
    if len(candidates) > top_k and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            docs_text = "\n\n".join([f"[{i}] {doc}" for i, doc in enumerate(candidates)])
            prompt = f"""
            You are a retrieval reranker. Rank the following documents by how well they answer the query.

            Query: {user_query}

            Documents:
            {docs_text}

            Return only the indices of the top {top_k} most relevant documents, separated by commas.
            Example: 0,3,1,5
            """

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            ranked_text = resp.choices[0].message.content.strip()

            # Extract indices
            import re
            indices = [int(x.strip()) for x in ranked_text.split(',') if x.strip().isdigit()]
            indices = [i for i in indices if i < len(candidates)]
            
            if indices:
                return [candidates[i] for i in indices[:top_k]]
                
        except Exception as e:
            print(f"[WARNING] LLM reranking failed: {e}")

    # Return top results without LLM reranking
    return candidates[:top_k]

def clear_vector_db():
    """Clear the in-memory vector database"""
    if 'vector_store' in st.session_state:
        st.session_state.vector_store.clear()
    if 'research_sources' in st.session_state:
        st.session_state.research_sources = []
    print("[Vector Store] Cleared all data")

def get_vector_db_stats():
    """Get statistics about the current vector database"""
    if 'vector_store' not in st.session_state:
        return {"total_documents": 0, "total_embeddings": 0, "metadata_count": 0}
    
    return st.session_state.vector_store.get_stats()

# Additional utility functions for the research sources feature

def get_filtered_sources(excluded_indices):
    """Get research sources excluding specified indices"""
    if 'research_sources' not in st.session_state:
        return []
    
    return [
        source for i, source in enumerate(st.session_state.research_sources)
        if i not in excluded_indices
    ]

def rebuild_vector_db_with_filtered_sources(excluded_indices):
    """Rebuild vector database with filtered sources"""
    if 'research_sources' not in st.session_state:
        return
    
    # Clear existing vector store
    clear_vector_db()
    
    # Get filtered sources
    filtered_sources = get_filtered_sources(excluded_indices)
    
    # Rebuild vector store with filtered sources
    text_blocks = []
    metadata_blocks = []
    
    for source in filtered_sources:
        content = source.get('content', '')
        if len(content) >= 300:
            from textwrap import wrap
            chunks = wrap(content, 800)
            for i, chunk in enumerate(chunks):
                text_blocks.append(chunk)
                metadata_blocks.append({
                    "type": source.get('content_type', 'article'),
                    "source": source.get('url', ''),
                    "summary": chunk[:300],
                    "title": source.get('title', ''),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source_type": source.get('type', 'Unknown')
                })
    
    if text_blocks:
        store_in_vector_db(text_blocks, metadata_blocks)
        print(f"[Vector Store] Rebuilt with {len(text_blocks)} chunks from {len(filtered_sources)} sources")