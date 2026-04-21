import logging
import re
from typing import List, Dict, Any, Optional
import pymupdf

logger = logging.getLogger(__name__)


class PDFParser:
    

    def __init__(
        self,
        chunk_size_tokens: int = 500,
        chunk_overlap_tokens: int = 50,
        encoding: str = "cl100k_base",
    ):
        
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.encoding = encoding
        
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(encoding)
        except ImportError:
            logger.warning("tiktoken not available, using character-based chunking")
            self.tokenizer = None

    def extract_text_with_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
       
        try:
            doc = pymupdf.open(pdf_path)
            pages_text = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                pages_text.append({"page": page_num + 1, "text": text})

            doc.close()
            logger.info(f"Extracted text from {len(pages_text)} pages in {pdf_path}")
            return pages_text

        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise RuntimeError(f"Failed to extract PDF text: {e}") from e

    def extract_and_chunk(self, pdf_path: str) -> List[Dict[str, Any]]:
       
        # Extract pages
        pages_text = self.extract_text_with_pages(pdf_path)

        # Combine all text with page markers
        full_text = ""
        page_mapping = []  # Track which page each character belongs to
        current_page = 1

        for page_data in pages_text:
            text = page_data["text"]
            page = page_data["page"]

            # Add page marker
            full_text += f"\n--- Page {page} ---\n"
            page_mapping.extend([page - 1] * (len(full_text) - len(page_mapping)))
            current_page = page

            # Add text
            for char in text:
                page_mapping.append(page)
            full_text += text

        # Create chunks
        chunks = self._create_token_chunks(full_text, page_mapping, pdf_path)
        logger.info(f"Created {len(chunks)} chunks from {pdf_path}")

        return chunks

    def _create_token_chunks(
        self, text: str, page_mapping: List[int], source: str
    ) -> List[Dict[str, Any]]:
        
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        chunks = []
        chunk_id = 0

        if self.tokenizer is None:
            # Fallback to character-based chunking
            return self._create_char_chunks(text, page_mapping, source)

        # Tokenize the full text
        try:
            tokens = self.tokenizer.encode(text)
        except Exception as e:
            logger.warning(f"Tokenization failed, falling back to char-based: {e}")
            return self._create_char_chunks(text, page_mapping, source)

        start_token_idx = 0

        while start_token_idx < len(tokens):
            end_token_idx = min(
                start_token_idx + self.chunk_size_tokens, len(tokens)
            )

            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(tokens[start_token_idx:end_token_idx])

            # Find the page number for this chunk
            if page_mapping:
                start_char_idx = len(self.tokenizer.decode(tokens[:start_token_idx]))
                page = page_mapping[min(start_char_idx, len(page_mapping) - 1)]
                # Ensure page is at least 1 (0-based mapping from list needs +1)
                page = max(page, 1)
            else:
                page = 1

            chunks.append(
                {
                    "chunk_id": f"{source}_{chunk_id}",
                    "text": chunk_text,
                    "source": source,
                    "page": page,
                    "token_count": end_token_idx - start_token_idx,
                }
            )

            chunk_id += 1
            # Move start position with overlap
            start_token_idx = max(
                start_token_idx + self.chunk_size_tokens - self.chunk_overlap_tokens,
                start_token_idx + 1,  # Ensure progress
            )

        return chunks

    def _create_char_chunks(
        self, text: str, page_mapping: List[int], source: str
    ) -> List[Dict[str, Any]]:
       
        chunks = []
        chunk_id = 0

        # Estimate tokens: ~4 chars per token
        char_size = int(self.chunk_size_tokens * 4)
        char_overlap = int(self.chunk_overlap_tokens * 4)

        start = 0
        while start < len(text):
            end = min(start + char_size, len(text))
            chunk_text = text[start:end]

            # Get page with safeguard against 0-based indexing
            page = page_mapping[start] if page_mapping and start < len(page_mapping) else 1
            page = max(page, 1)  # Ensure page is at least 1

            chunks.append(
                {
                    "chunk_id": f"{source}_{chunk_id}",
                    "text": chunk_text,
                    "source": source,
                    "page": page,
                    "token_count": len(chunk_text) // 4,  # Estimate
                }
            )

            chunk_id += 1
            start = max(end - char_overlap, start + 1)
        return chunks