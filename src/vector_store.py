"""
Vector store management using FAISS for similarity search
Handles embedding creation, storage, and retrieval using FastEmbed
"""
import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from fastembed import TextEmbedding
from langchain.docstore.document import Document
import config

class VectorStore:
    """FAISS-based vector store for document similarity search"""
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.dimension = None
        # load or initialize embedding model
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load FastEmbed model for embeddings"""
        try:
            print(f"Loading FastEmbed model: {self.model_name}")
            
            # initialize FastEmbed TextEmbedding
            self.embedding_model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=config.CACHE_DIR
            )
            
            # get embedding dimension by testing with sample text
            sample_embeddings = list(self.embedding_model.embed(["test"]))
            self.dimension = len(sample_embeddings[0])
            print(f"FastEmbed model loaded successfully. Dimension: {self.dimension}")
            
        except Exception as e:
            print(f"Error loading FastEmbed model: {str(e)}")
            print("Falling back to default model...")
            try:
                # fallback to a more reliable model
                self.model_name = "BAAI/bge-small-en-v1.5"
                self.embedding_model = TextEmbedding(
                    model_name=self.model_name,
                    cache_dir=config.CACHE_DIR
                )
                sample_embeddings = list(self.embedding_model.embed(["test"]))
                self.dimension = len(sample_embeddings[0])
                print(f"Fallback model loaded. Dimension: {self.dimension}")
            except Exception as e2:
                print(f"Fallback also failed: {str(e2)}")
                raise e2
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for list of texts using FastEmbed"""
        try:
            if self.embedding_model is None:
                raise ValueError("Embedding model is not initialized.")
            print(f"Creating embeddings for {len(texts)} texts...")
            
            # fastEmbed returns a generator, convert to list
            embeddings_list = list(self.embedding_model.embed(texts))
            embeddings = np.array(embeddings_list, dtype="float32")
            
            # l2-normalize so inner product == cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
            print(f"Embeddings created: shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"Error creating embeddings: {str(e)}")
            raise e
    
    def create_index(self, documents: List[Document]) -> bool:
        """Create FAISS index from documents"""
        if not documents:
            print("No documents provided for indexing")
            return False
        if self.dimension is None:
            print("Embedding model not initialized")
            return False
        
        try:
            print(f"Creating FAISS index for {len(documents)} documents...")
            # store documents
            self.documents = documents
            # extract texts for embedding
            texts = [doc.page_content for doc in documents]
            # create embeddings
            embeddings = self.create_embeddings(texts)
            # create FAISS index use inner product with IDMap, embeddings l2-normalized -> cosine
            base = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(base)
            ids = np.arange(len(embeddings), dtype=np.int64)
            self.index.add_with_ids(embeddings.astype("float32"), ids)
            print(f"FAISS index created successfully with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            print(f"Error creating FAISS index: {str(e)}")
            return False
    
    def save_index(self, path: str = config.VECTOR_STORE_PATH):
        """Save FAISS index and documents to disk"""
        try:
            # create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # save FAISS index
            faiss.write_index(self.index, f"{path}.index")        
            # save documents and metadata
            with open(f"{path}.docs", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'model_name': self.model_name,
                    'dimension': self.dimension
                }, f)
            print(f"Vector store saved to {path}")
            return True
            
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
            return False
    
    def load_index(self, path: str = config.VECTOR_STORE_PATH) -> bool:
        """Load FAISS index and documents from disk"""
        try:
            # check if files exist
            if not os.path.exists(f"{path}.index") or not os.path.exists(f"{path}.docs"):
                print(f"Vector store files not found at {path}")
                return False
            # load FAISS index
            self.index = faiss.read_index(f"{path}.index")
            # load documents and metadata
            with open(f"{path}.docs", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                saved_model = data.get('model_name', self.model_name)
                self.dimension = data.get('dimension', self.dimension)
            # check model compatibility
            if saved_model != self.model_name:
                print(f"Warning: Loaded model ({saved_model}) differs from current ({self.model_name})")
            print(f"Vector store loaded: {len(self.documents)} documents, {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = config.MAX_CHUNKS_FOR_CONTEXT) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if not self.index or not self.documents:
            print("Vector store not initialized")
            return []
        try:
            # create query embedding
            query_embedding = self.create_embeddings([query])
            # search FAISS index
            scores, indices = self.index.search(query_embedding.astype("float32"), k)
            # return documents with similarity scores
            results: List[Tuple[Document, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                if idx < len(self.documents):
                    # score is cosine similarity in [-1, 1]
                    results.append((self.documents[int(idx)], float(score)))
            return results
            
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []
    
    # Convenience for RAG prompt building & citations
    def topk_with_citations(self, query: str, k: int = config.MAX_CHUNKS_FOR_CONTEXT) -> List[Dict]:
        """Return top-k hits with compact citation strings for Markdown sections"""
        hits = self.similarity_search(query, k=k)
        out = []
        for rank, (doc, score) in enumerate(hits, start=1):
            meta = doc.metadata or {}
            src_name = Path(meta.get("source", "Unknown")).name  # filename
            sec_path = meta.get("section_path") or meta.get("section") or ""
            chunk_idx = meta.get("chunk_index",
                        meta.get("chunk_in_section", rank - 1))  # consistent key

            citation = f"{src_name}" + (f" §{sec_path}" if sec_path else "")
            uid = (src_name, sec_path, chunk_idx)                # ligns with BM25

            out.append({
                "doc": doc,
                "score": float(score),
                "rank": rank,
                "source": src_name,
                "section_path": sec_path,
                "chunk_index": chunk_idx,
                "citation": citation,
                "uid": uid,
            })
        return out

    def build_context(self, hits: List[Dict], max_chars: int = config.MAX_CONTEXT_LENGTH) -> str:
        seen = set()
        parts, total = [], 0
        for h in hits:
            doc: Document = h["doc"]
            meta = doc.metadata or {}
            src = Path(meta.get("source", "Unknown")).name
            sec = meta.get("section_path") or meta.get("section") or ""
            key = (src, sec)
            if key in seen:  # skip duplicate sections
                continue
            seen.add(key)

            cite: str = f"{src}" + (f" §{sec}" if sec else "")
            frag = f"[source: {cite}]\n{doc.page_content}\n"
            if total + len(frag) > max_chars:
                break
            parts.append(frag)
            total += len(frag)
            if len(seen) >= config.MAX_DISTINCT_CITATIONS:
                break
        return "\n".join(parts).strip()
    
    def get_relevant_context(self, query: str, max_chars: int = config.MAX_CONTEXT_LENGTH) -> str:
        """Get relevant context for RAG system (with citations)"""
        hits = self.topk_with_citations(query, k=config.MAX_CHUNKS_FOR_CONTEXT)
        if not hits:
            return "No relevant information found."
        return self.build_context(hits, max_chars=max_chars)
    
    def get_stats(self) -> dict:
        """Get vector store statistics"""
        return {
            "total_documents": len(self.documents),
            "total_vectors": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.dimension,
            "model_name": self.model_name
        }

def build_vector_store(documents: List[Document], force_rebuild: bool = False) -> VectorStore:
    """Build or load vector store"""
    vector_store = VectorStore()
    
    # try to load existing index
    if not force_rebuild and vector_store.load_index():
        print("Loaded existing vector store")
        return vector_store
    
    # build new index
    print("Building new vector store...")
    if vector_store.create_index(documents):
        vector_store.save_index()
        print("Vector store built and saved successfully")
    else:
        print("Failed to build vector store")
        return vector_store
    
    return vector_store

if __name__ == "__main__":
    # Test vector store functionality
    from .document_processor import DocumentProcessor
    
    # Process documents
    processor = DocumentProcessor()
    documents = processor.process_directory(str(config.DATA_DIR))
    
    if documents:
        print(f"Found {len(documents)} document chunks to process")
        
        # Build vector store
        vector_store = build_vector_store(documents, force_rebuild=True)
        
        if vector_store:
            print("\n" + "="*50)
            print("VECTOR STORE TEST RESULTS")
            print("="*50)
            
            # Test similarity search
            test_queries = [
                "How many days of annual leave do I get?",
                "What are the remote work requirements?", 
                "What happens if I violate the code of conduct?",
                "When are performance reviews conducted?"
            ]
            
            for query in test_queries:
                print(f"\n => Query: {query}")
                print("-" * 40)
                results = vector_store.similarity_search(query, k=3)
                
                if results:
                    for i, (doc, score) in enumerate(results):
                        source = doc.metadata.get('source', 'Unknown')
                        preview = doc.page_content[:100].replace('\n', ' ') + "..."
                        print(f"  {i+1}. Score: {score:.3f} | Source: {source}")
                        print(f"     Preview: {preview}")
                else:
                    print("     No results found")
            
            # Print statistics
            stats = vector_store.get_stats()
            print(f"\nSTATISTICS:")
            print(f"   Total documents: {stats['total_documents']}")
            print(f"   Total vectors: {stats['total_vectors']}")
            print(f"   Embedding dimension: {stats['embedding_dimension']}")
            print(f"   Model name: {stats['model_name']}")
            
        else:
            print("Failed to build vector store")
    else:
        print("No documents found to process")