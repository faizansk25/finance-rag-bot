"""
Domain-Aware Chunking for Finance Documents

Instead of splitting documents by character count (generic RAG),
this module splits by document STRUCTURE — headers, line items,
terms, etc. — and enriches each chunk with metadata.

This is what makes RAG work for finance:
- An invoice header chunk has vendor, amount, date metadata
- A line items chunk has per-item details
- A terms chunk has payment terms and due dates

Metadata-enriched embeddings let you filter BEFORE or AFTER
vector search for much more precise retrieval.
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class ChunkMetadata:
    """Metadata attached to every chunk for filtering and ranking."""
    doc_type: str           # invoice, purchase_order, receipt, contract
    chunk_type: str         # header, line_items, terms, body, summary
    vendor: Optional[str] = None
    invoice_number: Optional[str] = None
    amount: Optional[float] = None
    date: Optional[str] = None
    due_date: Optional[str] = None
    status: Optional[str] = None     # paid, overdue, pending, approved
    currency: str = "USD"
    line_item_count: int = 0
    gl_account: Optional[str] = None
    source_file: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_search_filter(self) -> dict:
        """Generate filters for metadata-enhanced search."""
        filters = {}
        if self.vendor:
            filters["vendor"] = self.vendor
        if self.doc_type:
            filters["doc_type"] = self.doc_type
        if self.status:
            filters["status"] = self.status
        if self.amount is not None:
            filters["amount"] = self.amount
        return filters


@dataclass
class DocumentChunk:
    """A single chunk of a document with metadata."""
    content: str
    metadata: ChunkMetadata
    chunk_id: str = ""
    token_count: int = 0

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"{self.metadata.doc_type}_{hash(self.content) % 100000}"
        self.token_count = len(self.content.split())


class InvoiceChunker:
    """
    Splits invoices into domain-aware chunks.

    An invoice is NOT just a block of text. It has structure:
    - Header (vendor, invoice number, date, total)
    - Line items (each item with description, quantity, price)
    - Terms (payment terms, due date, bank details)

    Each chunk type gets different metadata and is useful for
    different kinds of questions.
    """

    def __init__(self, max_chunk_tokens: int = 512, overlap_tokens: int = 50):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_invoice(self, invoice_text: str, source_file: str = "") -> List[DocumentChunk]:
        """
        Parse and chunk a single invoice into domain-aware pieces.

        Returns a list of DocumentChunk objects, each with metadata.
        """
        chunks = []

        # Extract header information
        header_metadata = self._extract_header_metadata(invoice_text)
        header_text = self._extract_header_text(invoice_text)

        if header_text:
            chunks.append(DocumentChunk(
                content=header_text,
                metadata=ChunkMetadata(
                    doc_type="invoice",
                    chunk_type="header",
                    vendor=header_metadata.get("vendor"),
                    invoice_number=header_metadata.get("invoice_number"),
                    amount=header_metadata.get("amount"),
                    date=header_metadata.get("date"),
                    due_date=header_metadata.get("due_date"),
                    status=header_metadata.get("status", "pending"),
                    currency=header_metadata.get("currency", "USD"),
                    source_file=source_file,
                ),
            ))

        # Extract line items
        line_items = self._extract_line_items(invoice_text)
        if line_items:
            items_text = self._format_line_items(line_items)
            chunks.append(DocumentChunk(
                content=items_text,
                metadata=ChunkMetadata(
                    doc_type="invoice",
                    chunk_type="line_items",
                    vendor=header_metadata.get("vendor"),
                    invoice_number=header_metadata.get("invoice_number"),
                    amount=header_metadata.get("amount"),
                    line_item_count=len(line_items),
                    source_file=source_file,
                ),
            ))

        # Extract payment terms
        terms_text = self._extract_terms(invoice_text)
        if terms_text:
            chunks.append(DocumentChunk(
                content=terms_text,
                metadata=ChunkMetadata(
                    doc_type="invoice",
                    chunk_type="terms",
                    vendor=header_metadata.get("vendor"),
                    invoice_number=header_metadata.get("invoice_number"),
                    due_date=header_metadata.get("due_date"),
                    source_file=source_file,
                ),
            ))

        # If no structured chunks were found, fall back to
        # character-based chunking with overlap
        if not chunks:
            chunks = self._fallback_chunk(invoice_text, source_file, header_metadata)

        return chunks

    def _extract_header_metadata(self, text: str) -> dict:
        """Extract structured metadata from invoice header."""
        metadata = {}

        # Invoice number patterns
        inv_patterns = [
            r'Invoice\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'Inv\s*(?:No|Number|#)\s*:?\s*([A-Z0-9\-]+)',
            r'Bill\s*#?\s*:?\s*([A-Z0-9\-]+)',
        ]
        for pat in inv_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                metadata["invoice_number"] = m.group(1).strip()
                break

        # Vendor/supplier
        vendor_patterns = [
            r'(?:From|Vendor|Supplier|Bill From)\s*:?\s*(.+?)(?:\n|$)',
            r'(?:Company)\s*:?\s*(.+?)(?:\n|$)',
        ]
        for pat in vendor_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                metadata["vendor"] = m.group(1).strip()
                break

        # Amount (total)
        amount_patterns = [
            r'(?:Total|Amount Due|Balance Due)\s*:?\s*\$?([\d,]+\.?\d*)',
            r'(?:Grand Total)\s*:?\s*\$?([\d,]+\.?\d*)',
        ]
        for pat in amount_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                metadata["amount"] = float(m.group(1).replace(",", ""))
                break

        # Date
        date_patterns = [
            r'(?:Invoice\s*Date|Date)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:Invoice\s*Date|Date)\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4})',
        ]
        for pat in date_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                metadata["date"] = m.group(1).strip()
                break

        # Due date
        due_patterns = [
            r'(?:Due\s*Date|Payment\s*Due)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:Due\s*Date|Payment\s*Due)\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4})',
        ]
        for pat in due_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                metadata["due_date"] = m.group(1).strip()
                break

        # Status
        if re.search(r'(?:Status|Payment Status)\s*:?\s*(\w+)', text, re.IGNORECASE):
            status_match = re.search(r'(?:Status|Payment Status)\s*:?\s*(\w+)', text, re.IGNORECASE)
            metadata["status"] = status_match.group(1).lower()

        return metadata

    def _extract_header_text(self, text: str) -> str:
        """Extract the header section of an invoice."""
        lines = text.strip().split('\n')
        header_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Header ends when we hit line items or long text blocks
            if len(stripped) > 200:
                break
            header_lines.append(stripped)
            # Stop after ~10 header lines
            if len(header_lines) >= 10:
                break
        return '\n'.join(header_lines)

    def _extract_line_items(self, text: str) -> List[dict]:
        """Extract line items from invoice."""
        items = []
        lines = text.split('\n')
        in_items = False

        for line in lines:
            stripped = line.strip()
            # Detect start of line items section
            if re.search(r'(?:Item|Description|Product|Service)\s+(?:No|Name|#|Code)', stripped, re.IGNORECASE):
                in_items = True
                continue
            if re.search(r'(?:Qty|Quantity)\s+(?:Price|Rate|Amount)', stripped, re.IGNORECASE):
                in_items = True
                continue

            if in_items and stripped:
                # Try to parse a line item: description, quantity, unit price, amount
                parts = re.split(r'\s{2,}|\t', stripped)
                if len(parts) >= 2:
                    item = {"description": parts[0]}
                    # Try to find numeric values
                    nums = re.findall(r'[\d,]+\.?\d*', stripped)
                    if len(nums) >= 2:
                        item["quantity"] = nums[0]
                        item["unit_price"] = nums[1]
                        if len(nums) >= 3:
                            item["amount"] = nums[2]
                    items.append(item)

                # Stop if we hit a total line
                if re.search(r'(?:Total|Subtotal|Tax)', stripped, re.IGNORECASE):
                    in_items = False

        return items

    def _format_line_items(self, items: List[dict]) -> str:
        """Format line items into readable text."""
        lines = ["Line Items:"]
        for i, item in enumerate(items, 1):
            desc = item.get("description", "Unknown")
            qty = item.get("quantity", "")
            price = item.get("unit_price", "")
            amount = item.get("amount", "")
            line = f"  {i}. {desc}"
            if qty:
                line += f" | Qty: {qty}"
            if price:
                line += f" | Price: ${price}"
            if amount:
                line += f" | Amount: ${amount}"
            lines.append(line)
        return '\n'.join(lines)

    def _extract_terms(self, text: str) -> str:
        """Extract payment terms and conditions."""
        terms_lines = []
        lines = text.split('\n')
        in_terms = False

        for line in lines:
            stripped = line.strip()
            if re.search(r'(?:Terms|Payment\s*Terms|Conditions|Notes)', stripped, re.IGNORECASE):
                in_terms = True
                continue
            if in_terms and stripped:
                terms_lines.append(stripped)
                if len(terms_lines) >= 10:
                    break

        return '\n'.join(terms_lines) if terms_lines else ""

    def _fallback_chunk(self, text: str, source_file: str, metadata: dict) -> List[DocumentChunk]:
        """Fallback to character-based chunking when structure extraction fails."""
        chunks = []
        words = text.split()
        chunk_size = self.max_chunk_tokens

        for i in range(0, len(words), chunk_size - self.overlap_tokens):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(DocumentChunk(
                content=chunk_text,
                metadata=ChunkMetadata(
                    doc_type="invoice",
                    chunk_type="body",
                    vendor=metadata.get("vendor"),
                    invoice_number=metadata.get("invoice_number"),
                    amount=metadata.get("amount"),
                    source_file=source_file,
                ),
            ))

        return chunks


class MultiDocChunker:
    """
    Chunks multiple document types: invoices, POs, receipts, contracts.
    Each document type has its own chunking strategy.
    """

    def __init__(self):
        self.invoice_chunker = InvoiceChunker()

    def chunk_document(self, text: str, doc_type: str = "invoice",
                       source_file: str = "") -> List[DocumentChunk]:
        """Route to the appropriate chunker based on document type."""
        if doc_type == "invoice":
            return self.invoice_chunker.chunk_invoice(text, source_file)
        else:
            # Generic chunking for other document types
            return self._generic_chunk(text, doc_type, source_file)

    def _generic_chunk(self, text: str, doc_type: str,
                       source_file: str) -> List[DocumentChunk]:
        """Generic chunking for non-invoice documents."""
        chunks = []
        paragraphs = text.split('\n\n')

        for i, para in enumerate(paragraphs):
            if para.strip():
                chunks.append(DocumentChunk(
                    content=para.strip(),
                    metadata=ChunkMetadata(
                        doc_type=doc_type,
                        chunk_type="body",
                        source_file=source_file,
                    ),
                ))

        return chunks


def demo():
    """Self-check: verify chunker works on a sample invoice."""
    sample = """
Invoice
Invoice Number: INV-0042
Date: 03/15/2025
Due Date: 04/14/2025
Status: pending

From:
Vendor: Acme Corp
GL Account: 5000

Item No    Description                          Qty    Unit Price    Amount
--------   -----------------------------------  -----  ----------   ----------
1          Cloud Hosting (Monthly)               1      $299.99      $299.99
2          API Calls (100K batch)                5      $49.99       $249.95

                          Subtotal:    $549.94
                          Tax (8%):    $44.00
                          Total:       $593.94

Payment Terms: Net 30

Notes: Please reference invoice number INV-0042 on your payment.
"""

    chunker = InvoiceChunker()
    chunks = chunker.chunk_invoice(sample, source_file="test.txt")

    assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"

    header = [c for c in chunks if c.metadata.chunk_type == "header"]
    assert len(header) == 1, f"Expected 1 header chunk, got {len(header)}"
    assert header[0].metadata.invoice_number == "INV-0042"
    assert header[0].metadata.amount == 593.94
    assert header[0].metadata.vendor == "Acme Corp"

    items = [c for c in chunks if c.metadata.chunk_type == "line_items"]
    assert len(items) == 1, f"Expected 1 line_items chunk, got {len(items)}"
    assert items[0].metadata.line_item_count == 2

    print(f"demo OK: {len(chunks)} chunks, header=INV-0042, amount=593.94, items=2")
    return chunks


if __name__ == "__main__":
    demo()
