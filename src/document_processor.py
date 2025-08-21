"""
Document processing utilities for HR Policy documents
Handles text extraction, chunking, and preprocessing
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import config

# Regex for md headings
HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$')

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
            print(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""

    def extract_pages_from_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Extract pages from PDF"""
        try:
            import PyPDF2
            pages = []
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages, start=1):
                    txt = page.extract_text() or ""
                    pages.append((i, txt))
            return pages
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return []

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
        # remove markdown headers (NOT md citation)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # remove bold/italic markers
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
        # remove inline code markers
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _parse_md_sections(self, raw: str) -> List[Tuple[int, str, str]]:
        """Parse Markdown into sections by headings. Returns list of (level, title, body).
        Content before the first heading becomes level 0 'Introduction'"""
        lines = raw.splitlines()
        sections: List[Tuple[int, str, str]] = []
        lvl, title, body = 0, "Introduction", []

        def flush():
            if body:
                sections.append((lvl, title.strip() or "Introduction", "\n".join(body).strip()))

        for line in lines:
            m = HEADING_RE.match(line)
            if m:
                flush()
                lvl = len(m.group(1))
                title = m.group(2).strip()
                body = []
            else:
                body.append(line)
        flush()
        return sections

    def _clean_inline_md(self, text: str) -> str:
        """Clean inline markdown WITHOUT removing headings 
        (used after section split)"""
        t = text
        # keep headings out of section bodies already, mild cleanup here:
        t = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', t)           # bold/italic
        t = re.sub(r'`([^`]+)`', r'\1', t)                        # inline code
        t = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', t)            # links
        t = re.sub(r'\n{3,}', '\n\n', t)
        return t.strip()
    
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
    
    def process_documents(self, file_path: Union[str, Path]) -> List[Document]:
        """Process single document and return chunks"""
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"File not found: {file_path}")
            return []

        suffix = file_path.suffix.lower()

        # PDF: flat extraction 
        if suffix == '.pdf':
            text = self.extract_text_from_pdf(str(file_path))
            if not text:
                print(f"No text extracted from {file_path}")
                return []
            documents = self.chunk_document(text, source=str(file_path))

        # MD: section-aware parsing for good citations
        elif suffix in ['.md', '.markdown']:
            try:
                raw = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"Error reading markdown {file_path}: {e}")
                return []

            sections = self._parse_md_sections(raw)
            documents: List[Document] = []

            # breadcrumb stack for nested headings
            breadcrumb: List[Tuple[int, str]] = []
            for sec_idx, (lvl, sec_title, sec_body) in enumerate(sections):
                # update breadcrumb (only for lvl > 0)
                while breadcrumb and breadcrumb[-1][0] >= lvl and lvl > 0:
                    breadcrumb.pop()
                if lvl > 0:
                    breadcrumb.append((lvl, sec_title))

                path_titles = [t for _, t in breadcrumb] if breadcrumb else [sec_title]
                section_path = " » ".join(path_titles) if path_titles else (sec_title or "Introduction")

                # clean inline markdown inside the section body
                body_clean = self._clean_inline_md(sec_body or "")

                # chunk within this section and attach metadata for citations
                chunks = self.text_splitter.split_text(body_clean)
                for chunk_i, chunk in enumerate(chunks):
                    c = chunk.strip()
                    if len(c) < 50:
                        continue
                    documents.append(Document(
                        page_content=c,
                        metadata={
                            "source": str(file_path),
                            "section": sec_title or "Introduction",
                            "section_path": section_path,
                            "heading_level": lvl,
                            "chunk_in_section": chunk_i,
                            "section_index": sec_idx
                        }
                    ))

        # TXT: simple read + cleaner -> flat chunking
        elif suffix == '.txt':
            text = self.extract_text_from_md(str(file_path))  # reuse light cleaner
            if not text:
                print(f"No text extracted from {file_path}")
                return []
            documents = self.chunk_document(text, source=str(file_path))

        else:
            print(f"Unsupported file type: {file_path.suffix}")
            return []

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
        # use list() to create a list instance, not typing.List
        sources = list(set(doc.metadata.get("source", "unknown") for doc in documents))

        return {
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // len(documents),
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