"""
Vector store management using FAISS for similarity search
Handles embedding creation, storage, and retrieval
"""
import os
import pickle
from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
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

        # load or initialize embbeding model
        self._load_embedding_model()

    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(
                self.model_name,
                cache_folder=config.CACHE_DIR
            )
            # get embedding dimension
            sample_embedding = self.embedding_model.encode(["test"])
            self.dimension = sample_embedding.shape[1]
            print(f"Embedding model loaded. Dimension: {self.dimension}")
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            raise e
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for list of texts"""
        try:
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            return embeddings
        except Exception as e:
            print(f"Error creating embeddings: {str(e)}")
            raise e
        
    def create_index(self, documents: List[Document]) -> bool:
        """Create FAISS index from documents"""
        if not documents:
            print("No documents provided for indexing")
            return False
        try:
            print(f"Creating FAISS index for {len(documents)} documents...")
            # store documents
            self.documents =  documents
            # extract texts for embedding
            texts = [doc.page_content for doc in documents]
            # create embeddings
            embeddings = self.create_embeddings(texts)
            # create FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype('float32'))

            print(f"FAISS index created with {self.index.ntotal} vectors")
            return True
        
        except Exception as e:
            print(f"Error creating FAISS index: {str(e)}")
            return e
        
    def save_index(self, path: str = config.VECTOR_STORE_PATH):
        """Save FAISS index and documents to disk"""
        try:
            # create directory if it doesnt exist
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
                print(f"Warning: Loaded model ({saved_model}) does not match current model ({self.model_name})")
            
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
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # return documents with similarity scores
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    # convert L2 distance to similarity score (higher is better)
                    similarity = 1.0 / (1.0 + distance)
                    results.append((self.documents[idx], similarity))
            
            return results
            
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []
        
    def get_relevant_context(self, query: str, max_chars: int = config.MAX_CONTEXT_LENGTH) -> str:
        """Get relevant context for RAG system"""
        # get similar documents
        results = self.similarity_search(query)
        
        if not results:
            return "No relevant information found."
        
        # combine relevant chunks into context
        context_parts = []
        current_chars = 0
        
        for doc, similarity in results:
            doc_text = doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            
            # format context with source
            formatted_text = f"[From {source}]\n{doc_text}\n"
            
            # check if adding this would exceed limit
            if current_chars + len(formatted_text) > max_chars:
                break
            
            context_parts.append(formatted_text)
            current_chars += len(formatted_text)
        
        context = "\n".join(context_parts)
        return context.strip()
    
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
    
    # Try to load existing index
    if not force_rebuild and vector_store.load_index():
        print("Loaded existing vector store")
        return vector_store
    
    # Build new index
    print("Building new vector store...")
    if vector_store.create_index(documents):
        vector_store.save_index()
        print("Vector store built and saved successfully")
    else:
        print("Failed to build vector store")
        return None
    
    return vector_store

if __name__ == "__main__":
    # test vector store functionality
    from .document_processor import DocumentProcessor
    
    # process documents
    processor = DocumentProcessor()
    documents = processor.process_directory(config.DATA_DIR)
    
    if documents:
        # build vector store
        vector_store = build_vector_store(documents, force_rebuild=True)
        
        if vector_store:
            # test similarity search
            test_queries = [
                "How many days of annual leave do I get?",
                "What are the remote work requirements?",
                "What happens if I violate the code of conduct?",
                "When are performance reviews conducted?"
            ]
            
            for query in test_queries:
                print(f"\nQuery: {query}")
                results = vector_store.similarity_search(query, k=3)
                for i, (doc, score) in enumerate(results):
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"  {i+1}. Score: {score:.3f} | Source: {source}")
                    print(f"     Text: {doc.page_content[:100]}...")
    else:
        print("No documents found to process")