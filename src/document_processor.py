"""
Document processing utilities for HR Policy documents
Handles text extraction, chunking, and preprocessing
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import config

class DocumentProcessor:
    """Process various document formats for RAG system"""
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error reading PDF ")

    def extract_text_from_md(self, md_path: str) -> str:
        """Extract text from Markdown file"""
        try:
            with open(md_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # clean md formatting while keeping structure
                content = self.clean_markdown(content)
                return content.strip()
        except Exception as e:
            print(f"Error reading markdown {md_path}: {str(e)}")
            return ""
        
    def clean_markdown(self, text: str) -> str:
        """Clean markdown formatting but preserve structure"""
        # remove markdown headers but keep as sections
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # remove bold/italic markers
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
        # remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def chunk_document(self, text: str, source: str = "") -> List[Document]:
        """Split document into chunks for vector storage"""
        # create chunks using LangChain text splitter
        chunks = self.text_splitter.split_text(text)

        # convert to document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            # clean chunk text
            chunk = chunk.strip()
            if len(chunk) < 50: # skip very short chunks
                continue

            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "chunk_length": len(chunk)
                }
            )
            documents.append(doc)
            
        return documents
    
    def process_documents(self, file_path: str) -> List[Document]:
        """Process single document and return chunks"""
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"File not found: {file_path}")
            return []
        
        # extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            text = self.extract_text_from_pdf(str(file_path))
        elif file_path.suffix.lower() in ['.md', '.txt']:
            text = self.extract_text_from_md(str(file_path))
        else:
            print(f"Unsupported file type: {file_path.suffix}")
            return []
        
        if not text:
            print(f"No text extracted from {file_path}")
            return []
        
        # create chunks
        documents = self.chunk_document(text, source=str(file_path))
        print(f"Processed {file_path.name}: {len(documents)} chunks")

        return documents
    
    def process_directory(self, dir_path: str) -> List[Document]:
        """Process all supported documents in a directory"""
        directory = Path(dir_path)
        all_documents = []

        supported_extensions = ['.pdf', '.md', '.txt']

        for file_path in directory.iterdir():
            if file_path.suffix.lower() in supported_extensions:
                documents = self.process_documents(file_path)
                all_documents.extend(documents)
        
        print(f"Total documents processed: {len(all_documents)} chunks from {directory}")
        return all_documents
    
    def get_doc_stats(self, documents: List[Document]) -> Dict:
        """Get statistics about processed documents"""
        if not documents:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "sources": []
            }
        total_chars = sum(len(doc.page_content) for doc in documents)
        # Use list() to create a list instance, not typing.List
        sources = list(set(doc.metadata.get("source", "unknown") for doc in documents))

        return {
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // len(documents) if documents else 0,
            "sources": sources
        }
    
if __name__ == "__main__":
    # test document processing
    processor = DocumentProcessor()
    documents = processor.process_directory(config.DATA_DIR)
    stats = processor.get_doc_stats(documents)

    print("\nDocument Processing Results:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total characters: {stats['total_characters']}")
    print(f"Average chunk size: {stats['avg_chunk_size']}")
    print(f"Sources: {stats['sources']}")