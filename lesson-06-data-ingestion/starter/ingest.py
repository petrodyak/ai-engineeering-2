"""Lesson 6: ingest a folder of heterogeneous documents into clean, deduplicated chunks.

    python starter/ingest.py             # ingest data/raw -> output/
    python starter/ingest.py --full      # ignore the manifest and re-ingest everything
    python starter/check.py              # check the output against data/expected.json

Pipeline: extract text -> normalize -> deduplicate -> chunk -> attach metadata -> manifest.
TODO 1-5 at home (extract, normalize), TODO 6-10 in the lesson.
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ACCESS = ROOT / "data" / "access.json"
OUT = ROOT / "output"

PIPELINE_VERSION = "ingest-v1"
CHUNK_TOKENS = 120          # max tokens per chunk (small, because the demo documents are short)
FIXED_OVERLAP = 20          # overlap for the fixed-size baseline
NEAR_DUP_JACCARD = 0.8      # shingle overlap above which two documents are near-duplicates
MIN_CHARS_PER_PAGE = 50     # a PDF page with less text than this has no usable text layer

ENC = tiktoken.get_encoding("cl100k_base")


# Counts tokens the way the LLM sees them (tiktoken, cl100k_base).
# Returns: int, the number of tokens in text.
def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


# One input file on its way through the pipeline: its text, status, hashes and chunks.
@dataclass
class Doc:
    source: str                       # path relative to data/raw, e.g. "it/vpn_setup.docx"
    raw_hash: str                     # sha256 of the file bytes: did the file change?
    text: str = ""
    title: str = ""
    effective: str | None = None      # "Effective YYYY-MM-DD" found in the document
    status: str = "ok"                # ok | ocr_required | exact_duplicate | near_duplicate | error
    note: str = ""
    doc_hash: str = ""                # sha256 of the normalized text: is the content the same?
    unchanged: bool = False           # same file as in the last run: reuse its chunks
    chunks: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- 1. extract

# Reads a .txt / .md file: strict UTF-8 first, then charset_normalizer guesses a legacy
# encoding (cp1251, ...); the BOM is removed.
# Returns: str, the decoded text.
def read_text_file(path: Path) -> str:
    """TODO 1: .txt / .md. Detect the encoding instead of assuming UTF-8."""
    # Hint: charset_normalizer.from_bytes(data).best(); strip the BOM ("\ufeff").
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")  # strict: most files are UTF-8, and a wrong guess raises here
    except UnicodeDecodeError:
        best = from_bytes(data).best()  # legacy encodings (cp1251, latin-1, ...): let the detector guess
        text = str(best) if best else data.decode("utf-8", errors="replace")
    return text.lstrip("\ufeff")


# Keeps only the article of an HTML page (drops scripts, styles, nav, footer, cookie banner)
# and converts it to markdown.
# Returns: str, "#"/"##"/"###" headings, "- " list items and paragraphs, blocks separated
# by a blank line.
def extract_html(path: Path) -> str:
    """TODO 2: keep the article, drop navigation, cookie banners, scripts and the footer.
    Headings become markdown headings so the chunker can see the structure."""
    # Hint: BeautifulSoup; decompose script/style/nav/footer and the cookie banner;
    # take <main>; turn <h1>-<h3> into "# ", "## ", "### " lines and <li> into "- " lines.
    # 1. Parse: read with encoding detection (TODO 1), build a tree of tags.
    soup = BeautifulSoup(read_text_file(path), "html.parser")

    # 2. Drop boilerplate: decompose() removes the tag together with everything inside it.
    #    script/style = code, nav/header/aside = menus, footer = "All rights reserved".
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    #    The cookie banner is a plain <div>, so find it by id or class.
    for tag in soup.select("#cookie-banner, .cookie"):
        tag.decompose()

    # 3. Keep only the article: <main> if the page has one, otherwise <body>, otherwise everything.
    root = soup.find("main") or soup.body or soup

    # 4. Convert to markdown: headings get "#" so the chunker (TODO 8) can see sections,
    #    list items get "- ", paragraphs stay as they are.
    prefix = {"h1": "# ", "h2": "## ", "h3": "### ", "li": "- ", "p": ""}
    blocks: list[str] = []
    prev = ""
    for el in root.find_all(list(prefix)):  # document order
        # Collapse newlines and runs of spaces; no separator, so inline <a>, <b> keep the source spacing.
        text = " ".join(el.get_text().split())
        if not text:
            continue
        line = prefix[el.name] + text
        if el.name == "li" and prev == "li":
            blocks[-1] += "\n" + line  # one list = one block, so the chunker keeps it together
        else:
            blocks.append(line)
        prev = el.name

    # 5. Blocks separated by a blank line: the chunker splits paragraphs on it.
    return "\n\n".join(blocks)


# Extracts the text layer page by page and detects scans (too few characters per page).
# Returns: tuple (text, needs_ocr): text with pages separated by a form feed;
# needs_ocr=True means: send to OCR, not to the index.
def extract_pdf(path: Path) -> tuple[str, bool]:
    """TODO 3: text per page, pages separated by \\f. Returns (text, needs_ocr).
    A scan has pages but (almost) no text layer: we flag it instead of indexing nothing."""
    # Hint: compare the number of characters with MIN_CHARS_PER_PAGE * number of pages.
    # 1. Extract the text layer of each page; a page that is only an image gives None -> "".
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]

    # 2. Count real text: strip() so spaces and newlines do not count as content.
    n_chars = sum(len(p.strip()) for p in pages)

    # 3. A scan: too little text for this many pages (a stamp or a page number is not a document).
    #    A PDF with no pages has nothing to index either, so it goes to review as well.
    needs_ocr = not pages or n_chars < MIN_CHARS_PER_PAGE * len(pages)

    # 4. Pages joined by \f (form feed): normalize() (TODO 5) splits on it to find repeating headers.
    return "\f".join(pages), needs_ocr


# Walks a Word document in document order and keeps its structure as markdown.
# Returns: str, "#"/"##" headings, "1. " numbered items, "| a | b |" table rows,
# blocks separated by a blank line.
def extract_docx(path: Path) -> str:
    """TODO 4: keep headings (as markdown), numbered lists and tables (as markdown rows).
    python-docx lists paragraphs and tables separately, so walk the body in document order."""
    # Hint: Title -> "# ", "Heading 1" -> "## ", "List Number" -> "1. ", a table -> "| a | b |" rows.
    # 1. Paragraph styles -> markdown prefixes. A Word file has real styles, unlike a PDF.
    doc = Document(path)
    prefix = {"Title": "# ", "Heading 1": "## ", "Heading 2": "### ", "Heading 3": "#### ", "List Bullet": "- "}
    blocks: list[str] = []
    prev = ""      # kind of the previous block: "list" or something else
    number = 0     # counter for "List Number": Word stores the numbers as formatting, not as text

    # 2. Walk the body in document order: doc.paragraphs and doc.tables would lose
    #    where the table stands between the paragraphs.
    for el in doc.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]  # "{namespace}p" -> "p"

        # 3. A table -> markdown rows; the header row gets a "| --- |" separator.
        if tag == "tbl":
            rows = [[" ".join(cell.text.split()) for cell in row.cells] for row in Table(el, doc).rows]
            if rows:
                lines = ["| " + " | ".join(r) + " |" for r in rows]
                lines.insert(1, "| " + " | ".join("---" for _ in rows[0]) + " |")
                blocks.append("\n".join(lines))  # one table = one block: the chunker never cuts it
            prev = "table"
            continue
        if tag != "p":
            continue  # section properties and other non-content elements

        # 4. A paragraph: skip empty ones, pick the prefix by style.
        para = Paragraph(el, doc)
        text = " ".join(para.text.split())
        if not text:
            continue
        style = para.style.name if para.style is not None else ""
        if style == "List Number":
            number = number + 1 if prev == "list" else 1  # a new list starts again from 1
            line = f"{number}. {text}"
        else:
            line = prefix.get(style, "") + text

        # 5. Consecutive list items form one block, so the chunker keeps the list together.
        is_list = style in ("List Number", "List Bullet")
        if is_list and prev == "list":
            blocks[-1] += "\n" + line
        else:
            blocks.append(line)
        prev = "list" if is_list else "para"

    # 6. Blocks separated by a blank line, as in extract_html().
    return "\n\n".join(blocks)


# Picks the extractor by file extension (.txt/.md, .html/.htm, .pdf, .docx).
# Returns: tuple (text, needs_ocr); needs_ocr can be True only for a PDF.
# Raises: ValueError for an unsupported file type.
def extract(path: Path) -> tuple[str, bool]:
    """Dispatch by file type. Returns (text, needs_ocr)."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return read_text_file(path), False
    if suffix in (".html", ".htm"):
        return extract_html(path), False
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".docx":
        return extract_docx(path), False
    raise ValueError(f"unsupported file type: {suffix}")


# ---------------------------------------------------------------- 2. normalize

PAGE_NUMBER = re.compile(r"^\s*(page\s+)?\d+\s*(of|/)\s*\d+\s*$", re.IGNORECASE)
HEADING = re.compile(r"^(#{1,6} |\d+\.\s+[A-ZА-ЯІЇЄҐ])")
ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")  # zero-width space/joiners, word joiner, BOM


# Decides whether a line starts a section: "# ..." or a short numbered title "2. Eligibility".
# Returns: bool.
def is_heading(line: str) -> bool:
    """A markdown heading, or a short numbered title like "2. Eligibility" (a PDF has no styles).
    "1. Open the Software Center." is a list item, not a heading: it is a sentence."""
    line = line.strip()
    if line.startswith("#"):
        return bool(HEADING.match(line))
    return bool(HEADING.match(line)) and len(line) < 60 and not line.endswith((".", ":", ";"))




# Joins lines that a PDF broke in the middle of a sentence.
# Returns: str, the text with each broken sentence back on one line.
def unwrap(text: str) -> str:
    """Join lines that a PDF broke inside a sentence; keep headings, list items and table rows."""
    out: list[str] = []
    for line in text.split("\n"):
        prev = out[-1] if out else ""
        # A wrapped line is long and does not end a sentence; a short line is a title or a label.
        joinable = (prev and line and len(prev) >= 60 and not HEADING.match(line) and not HEADING.match(prev)
                    and not line.lstrip().startswith(("- ", "|")) and not prev.lstrip().startswith("|")
                    and not re.match(r"^\d+\.\s", line) and not prev.rstrip().endswith((".", ":", "!", "?")))
        if joinable:
            out[-1] = f"{prev.rstrip()} {line.lstrip()}"
        else:
            out.append(line)
    return "\n".join(out)


# Brings any extracted text to one canonical form: NFKC, \n line endings, no invisible chars,
# no headers/footers/page numbers, no hyphenation, single spaces.
# Returns: str, the cleaned text, at most one blank line between paragraphs.
def normalize(text: str) -> str:
    """TODO 5: one canonical text form, so that hashing, dedup and chunking see the same thing."""
    # Steps, in this order:
    # 1. unicodedata.normalize("NFKC", ...), CRLF -> LF, remove zero-width characters and the BOM
    # 2. split pages on "\f", remove lines that repeat on most pages (headers, footers)
    # 3. remove page numbers ("Page 2 of 3") with PAGE_NUMBER
    # 4. join words hyphenated across lines: "secur-\nity" -> "security"
    # 5. strip trailing spaces, collapse runs of spaces, unwrap(text), max one blank line

    # 1. Characters: NFKC turns look-alikes into one form (non-breaking space -> space, "ﬁ" -> "fi");
    #    Windows (\r\n) and old Mac (\r) line endings -> \n; drop invisible zero-width chars and the BOM.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = ZERO_WIDTH.sub("", text)

    # 2. Headers and footers: a line found on most pages of a multi-page PDF is page furniture
    #    ("Northwind Logistics — Internal"), not content. One page gives no evidence, so skip it.
    pages = text.split("\f")
    if len(pages) > 1:
        on_pages: dict[str, int] = {}
        for page in pages:
            for line in {ln.strip() for ln in page.split("\n") if ln.strip()}:  # a set: count each page once
                on_pages[line] = on_pages.get(line, 0) + 1
        repeated = {line for line, n in on_pages.items() if n > len(pages) / 2}
        pages = ["\n".join(ln for ln in page.split("\n") if ln.strip() not in repeated) for page in pages]
    text = "\n".join(pages)  # the page boundaries are no longer needed

    # 3. Page numbers: a whole line like "Page 2 of 3" or "2/3".
    text = "\n".join(ln for ln in text.split("\n") if not PAGE_NUMBER.match(ln))

    # 4. Hyphenation: a word broken at the end of a line ("secur-\nity") -> "security".
    #    Only a letter-hyphen-newline-lowercase letter, so "e-mail" or a "- " list item stay as they are.
    text = re.sub(r"(\w)-\n([a-zа-яіїєґ])", r"\1\2", text)

    # 5. Whitespace: no trailing spaces, one space between words, join lines the PDF broke
    #    inside a sentence (unwrap), at most one blank line between paragraphs.
    text = "\n".join(re.sub(r"[ \t]+", " ", ln).rstrip() for ln in text.split("\n"))
    text = unwrap(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------- 3. deduplicate

# Hashes the content, not the bytes: whitespace and line endings do not change the result.
# Returns: str, 64-character hex sha256.
def content_hash(text: str) -> str:
    """TODO 6: the same content gives the same hash, whatever the line endings or spacing."""
    # Hint: hash a canonical form, not the raw string.
    # 1. Same characters: NFKC and no invisible chars, in case the text did not go through normalize().
    canonical = ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))

    # 2. Same spacing: every run of spaces, tabs and line breaks (\n, \r\n, blank lines) -> one space.
    #    Two files that differ only in where the lines wrap now give the same string.
    canonical = " ".join(canonical.split())

    # 3. Hash the canonical form: sha256 of the UTF-8 bytes, as a hex string.
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Splits a text into all runs of k consecutive lower-cased words (the near-duplicate fingerprint).
# "a b c d e f", k=5 -> ("a", "b", "c", "d", "e"), ("b", "c", "d", "e", "f").
# Returns: set of k-word tuples.
def shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


# Similarity of two sets: shared items / all items.
# Returns: float from 0.0 (nothing in common) to 1.0 (the same set).
def jaccard(a: set, b: set) -> float:
    """How similar two sets are: shared items / all items (|a ∩ b| / |a ∪ b|).
    1.0 = the same set, 0.0 = nothing in common. Used on shingle sets to find near-duplicate documents,
    e.g. two versions of a policy that differ in a few words score ~0.9.
    Two empty sets return 0.0 instead of dividing by zero."""
    return len(a & b) / len(a | b) if a | b else 0.0


# Compares every pair of documents by their shingles.
# Returns: list of (doc_a, doc_b, score) for pairs with score >= NEAR_DUP_JACCARD;
# an empty list if there are none.
def near_duplicates(docs: list[Doc]) -> list[tuple[Doc, Doc, float]]:
    """TODO 7: pairs of documents whose word 5-shingles overlap above NEAR_DUP_JACCARD.
    O(n^2) is fine for a folder; for millions of documents use MinHash + LSH."""
    # Hint: shingles() and jaccard() are ready; compare every pair once.
    # 1. Shingles once per document, not once per pair: n documents -> n calls instead of n^2.
    sets = [shingles(d.text) for d in docs]

    # 2. Every pair exactly once: j starts after i, so (a, b) is never also checked as (b, a)
    #    and a document is never compared with itself.
    pairs: list[tuple[Doc, Doc, float]] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            # 3. Jaccard = shared shingles / all shingles: 1.0 = same text, 0.0 = nothing in common.
            score = jaccard(sets[i], sets[j])
            # 4. Above the threshold: the same document in two versions. run() keeps the newer one.
            if score >= NEAR_DUP_JACCARD:
                pairs.append((docs[i], docs[j], score))
    return pairs


# ---------------------------------------------------------------- 4. chunk

# Baseline chunker: a sliding window of size tokens with overlap tokens of overlap.
# Returns: list of chunk dicts {"text", "section": ""}.
def chunk_fixed(text: str, size: int = CHUNK_TOKENS, overlap: int = FIXED_OVERLAP) -> list[dict]:
    """Baseline: a sliding window of tokens. Simple, but cuts sentences, tables and sections."""
    tokens = ENC.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        piece = ENC.decode(tokens[start:start + size])
        chunks.append({"text": piece, "section": ""})
        if start + size >= len(tokens):
            break
        start += size - overlap
    return chunks


# Splits the text into sections by headings and packs whole paragraphs into chunks
# of about max_tokens; never cuts a sentence, list item or table row.
# Returns: list of chunk dicts {"text", "section"}; section is the heading path,
# e.g. "Expense Policy > Limits", or "" if the document has no headings.
def chunk_by_headings(text: str, max_tokens: int = CHUNK_TOKENS) -> list[dict]:
    """TODO 8: split on headings, then pack whole paragraphs up to max_tokens.
    A paragraph or a table is never cut in the middle; each chunk knows its section path."""
    # Hint: split the text into blocks on blank lines; is_heading(first line) starts a new section;
    # keep the heading path (["Expense Policy", "Limits"]) and pack paragraphs while n_tokens() fits.
    # 1. Sections: walk line by line. A heading starts a new section, a blank line ends a paragraph.
    #    Line by line, not only on blank lines: in a PDF the headings ("2. Eligibility") sit on their own
    #    line with no blank line around them.
    sections: list[tuple[str, list[str]]] = []   # (section path, blocks)
    path: list[tuple[int, str]] = []              # heading stack: (level, title)
    blocks: list[str] = []
    para: list[str] = []
    has_body = False                              # the current section has more than its headings
    for line in text.split("\n") + [""]:          # the extra "" closes the last paragraph
        if is_heading(line):
            if para:
                blocks.append("\n".join(para))
                para = []
            # 2. Close the previous section, unless it is only headings ("# Title" right before "## Part"):
            #    then they stay and open the next section, so no chunk is a lone heading.
            if has_body:
                sections.append((" > ".join(t for _, t in path), blocks))
                blocks, has_body = [], False
            # 3. Heading path: "## Limits" under "# Expense Policy" -> "Expense Policy > Limits".
            #    A markdown heading has its level in the number of "#"; "2. Eligibility" (PDF) is level 2.
            level = len(line) - len(line.lstrip("#")) if line.startswith("#") else 2
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, line.lstrip("#").strip()))
            blocks.append(line.strip())
        elif not line.strip():
            if para:
                blocks.append("\n".join(para))
                para = []
        else:
            para.append(line)
            has_body = True
    if blocks:
        sections.append((" > ".join(t for _, t in path), blocks))

    # 4. Pack: add pieces to the chunk while it fits in max_tokens, otherwise start a new chunk.
    #    Pieces of one block are joined with "\n" (list items, table rows), blocks with "\n\n".
    chunks: list[dict] = []
    for section, section_blocks in sections:
        cur, cur_has_body = "", False
        for block in section_blocks:
            for k, piece in enumerate(split_block(block, max_tokens)):
                sep = "\n" if k else "\n\n"
                candidate = cur + sep + piece if cur else piece
                # A chunk with only headings is never flushed: it would be useless in retrieval.
                if cur_has_body and n_tokens(candidate) > max_tokens:
                    chunks.append({"text": cur, "section": section})
                    cur, cur_has_body = piece, False
                else:
                    cur = candidate
                cur_has_body = cur_has_body or not is_heading(piece)
        if cur:
            chunks.append({"text": cur, "section": section})
    return chunks


# Cuts a block that is too big for one chunk at the safest place:
# lines -> sentences -> tokens (last resort).
# Returns: list of str pieces, each within max_tokens; [block] if it already fits.
def split_block(block: str, max_tokens: int) -> list[str]:
    """A block that does not fit in one chunk, cut at the safest place that works:
    whole block -> lines (list items, table rows) -> sentences -> tokens (last resort)."""
    # 1. Fits: keep the paragraph, list or table whole.
    if n_tokens(block) <= max_tokens:
        return [block]
    pieces: list[str] = []
    for line in block.split("\n"):
        # 2. Too big as a whole: a line is a list item or a table row, never cut inside it if it fits.
        if n_tokens(line) <= max_tokens:
            pieces.append(line)
            continue
        # 3. A line too long for a chunk: cut between sentences.
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if n_tokens(sentence) <= max_tokens:
                pieces.append(sentence)
            else:
                # 4. One sentence longer than a chunk: nothing safe is left, cut by tokens.
                pieces += [c["text"] for c in chunk_fixed(sentence, max_tokens, 0)]
    return pieces


# Measures chunk quality: how many chunks start mid-sentence or cut a table, and their size.
# Returns: dict with chunks, starts_mid_sentence, cut_tables, avg_tokens, max_tokens.
def chunk_stats(chunks: list[dict]) -> dict:
    """How often a chunk starts mid-sentence or cuts a table: the cost of naive chunking."""
    mid = sum(1 for c in chunks if c["text"][:1].islower())
    cut_tables = sum(1 for c in chunks if any("|" in line and not (line.startswith("|") and line.endswith("|"))
                                              for line in c["text"].strip().split("\n")))
    sizes = [n_tokens(c["text"]) for c in chunks] or [0]
    return {"chunks": len(chunks), "starts_mid_sentence": mid, "cut_tables": cut_tables,
            "avg_tokens": round(sum(sizes) / len(sizes)), "max_tokens": max(sizes)}


# ---------------------------------------------------------------- 5. metadata

# Reads the access rules from data/access.json, skipping comment keys that start with "_".
# Returns: dict folder prefix -> groups, e.g. {"finance/": ["finance"], "": ["all_employees"]}.
def load_access() -> dict[str, list[str]]:
    rules = json.loads(ACCESS.read_text(encoding="utf-8"))
    return {k: v for k, v in rules.items() if not k.startswith("_")}


# Finds who may read a file: the rule with the longest matching folder prefix wins.
# Returns: list of group names, e.g. ["finance"].
# Raises: ValueError if no prefix matches (keep the catch-all "" rule).
def access_for(source: str, rules: dict[str, list[str]]) -> list[str]:
    prefix = max((p for p in rules if source.startswith(p)), key=len)
    return rules[prefix]


# Takes the first non-empty line of the text, without the markdown "#", as the title.
# Returns: str, at most 120 characters; the file name if the text is empty.
def title_of(doc: Doc) -> str:
    for line in doc.text.splitlines():
        if line.strip():
            return line.lstrip("#").strip()[:120]
    return Path(doc.source).stem


# Reads the title from the PDF metadata (the first line of a PDF may be a stamp).
# Returns: str; "" for a non-PDF or a PDF without a title.
def pdf_title(path: Path) -> str:
    """A PDF often carries its real title in the metadata; the first line may be a stamp."""
    if path.suffix.lower() != ".pdf":
        return ""
    meta = PdfReader(path).metadata
    return (meta.title or "").strip() if meta else ""


# Attaches the metadata retrieval needs to every chunk: id, source, title, section, access,
# versions (effective, doc_hash, pipeline_version, ingested_at) and n_tokens.
# Returns: list of dicts, one per chunk; chunk_id is "<source>#<chunk_index>".
def chunk_records(doc: Doc, chunks: list[dict], rules: dict[str, list[str]], now: str) -> list[dict]:
    """TODO 9: everything retrieval will need later: where it came from, who may see it,
    which version of the document and of the pipeline produced it."""
    # Fields: chunk_id, source, title, section, access (access_for), effective, doc_hash,
    # chunk_index, n_tokens, text, pipeline_version, ingested_at
    # 1. Who may see it: one lookup per document, all its chunks share the folder's access groups.
    #    Retrieval filters on this field before ranking, so a user never gets a chunk they may not read.
    access = access_for(doc.source, rules)

    records = []
    for i, c in enumerate(chunks):
        records.append({
            # 2. Identity: a stable id ("it/vpn_setup.docx#2") and where the chunk sits in its document.
            "chunk_id": f"{doc.source}#{i}",
            "source": doc.source,
            "chunk_index": i,
            # 3. Context for the answer and the citation: document title and the heading path.
            #    A document without headings (q3_budget_summary.pdf) falls back to its title.
            "title": doc.title,
            "section": c["section"] or doc.title,
            "access": access,
            # 4. Versions: which edition of the document (effective date, hash of its content)
            #    and which pipeline produced the chunk, so stale chunks can be found and re-ingested.
            "effective": doc.effective,
            "doc_hash": doc.doc_hash,
            "pipeline_version": PIPELINE_VERSION,
            "ingested_at": now,
            # 5. The content and its size in tokens: used to fit chunks into the LLM context budget.
            "n_tokens": n_tokens(c["text"]),
            "text": c["text"],
        })
    return records


# ---------------------------------------------------------------- 6. manifest, incremental runs, audit

# Reads output/manifest.json from the previous run.
# Returns: dict with "files": source -> {raw_hash, doc_hash, status, ...};
# {"files": {}} on the first run.
def load_manifest() -> dict:
    path = OUT / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"files": {}}


# Builds one id for the exact set of input files plus the pipeline version.
# Returns: str, 16 hex characters; the same inputs always give the same id.
def dataset_version(files: dict) -> str:
    """TODO 10: one id for the exact set of inputs + pipeline version: same inputs, same id."""
    # Hint: sha256 of json.dumps(..., sort_keys=True) over PIPELINE_VERSION and each file's raw_hash.
    # 1. What defines the dataset: the code version and the exact bytes of every input file.
    #    Not the time or the run: two runs on the same files must give the same id.
    #    Statuses, titles and chunk counts are left out: they follow from these two.
    key = {"pipeline_version": PIPELINE_VERSION,
           "files": {source: info["raw_hash"] for source, info in files.items()}}

    # 2. A stable string: sort_keys=True, so the order of the files in the dict does not change the id.
    canonical = json.dumps(key, sort_keys=True)

    # 3. Hash it; 16 hex chars are short enough to quote in a chat or a commit and still unique here.
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# Appends one event (ts, event, source, extra fields) to output/audit.jsonl.
# Returns: None.
def audit(event: str, source: str, **extra) -> None:
    """TODO 10: record what happened to each file, so any chunk can be traced back."""
    # One JSON line per event in OUT / "audit.jsonl": ts, event, source and the extra fields.
    # 1. The event: when (UTC), what happened, to which file, plus the details from the caller
    #    (e.g. of= for an exact duplicate, kept= and jaccard= for a near duplicate).
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "event": event, "source": source, **extra}

    # 2. Append, never overwrite: the log keeps the history of every run.
    #    One JSON object per line (JSONL), so it can be grepped or read line by line.
    OUT.mkdir(exist_ok=True)
    with (OUT / "audit.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- run

# Runs the whole pipeline over data/raw: extract -> normalize -> deduplicate -> chunk -> metadata.
# Unchanged files reuse their old chunks unless full=True. Writes output/manifest.json,
# output/chunks.jsonl and output/report.json.
# Returns: dict report: dataset_version, number of files, skipped_unchanged, status/note/chunks
# per file, chunking stats for the fixed-size and the heading-based chunker.
def run(full: bool) -> dict:
    OUT.mkdir(exist_ok=True)
    rules = load_access()
    manifest = {"files": {}} if full else load_manifest()
    previous = manifest["files"]
    old_chunks = {}
    if not full and (OUT / "chunks.jsonl").exists():
        for line in (OUT / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            old_chunks.setdefault(c["source"], []).append(c)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    docs: list[Doc] = []
    skipped = 0
    for path in sorted(p for p in RAW.rglob("*") if p.is_file() and not p.name.startswith(".")):
        source = path.relative_to(RAW).as_posix()
        doc = Doc(source=source, raw_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        prev = previous.get(source)
        if prev and prev["raw_hash"] == doc.raw_hash and prev.get("pipeline_version") == PIPELINE_VERSION:
            doc.status, doc.note, doc.doc_hash = prev["status"], prev.get("note", ""), prev.get("doc_hash", "")
            doc.title, doc.effective = prev.get("title", ""), prev.get("effective")
            doc.chunks = old_chunks.get(source, [])
            doc.text = "\n".join(c["text"] for c in doc.chunks)
            doc.unchanged = True
            skipped += 1
            docs.append(doc)
            continue
        try:
            raw_text, needs_ocr = extract(path)
        except Exception as e:  # one broken file must not stop the batch
            doc.status, doc.note = "error", f"{type(e).__name__}: {e}"
            audit("error", source, error=doc.note)
            docs.append(doc)
            continue
        if needs_ocr:
            doc.status, doc.note = "ocr_required", "no text layer: send to OCR, not to the index"
            audit("ocr_required", source)
            docs.append(doc)
            continue
        m = re.search(r"Effective (\d{4}-\d{2}-\d{2})", raw_text)
        doc.effective = m.group(1) if m else None
        doc.text = normalize(raw_text)
        doc.doc_hash = content_hash(doc.text)
        doc.title = pdf_title(path) or title_of(doc)
        docs.append(doc)

    # exact duplicates: same normalized content -> keep the first path, drop the rest
    seen: dict[str, Doc] = {}
    for doc in docs:
        if doc.status != "ok" or doc.unchanged:
            if doc.status == "ok":
                seen.setdefault(doc.doc_hash, doc)
            continue
        if doc.doc_hash in seen:
            doc.status, doc.note = "exact_duplicate", f"same content as {seen[doc.doc_hash].source}"
            audit("exact_duplicate", doc.source, of=seen[doc.doc_hash].source)
        else:
            seen[doc.doc_hash] = doc

    # near duplicates: keep the newer version (by effective date), drop the older one
    active = [d for d in docs if d.status == "ok"]
    for a, b, score in near_duplicates(active):
        older, newer = sorted([a, b], key=lambda d: d.effective or "")
        older.status, older.note = "near_duplicate", f"older version of {newer.source} (jaccard {score:.2f})"
        older.chunks = []
        audit("near_duplicate", older.source, kept=newer.source, jaccard=round(score, 2))

    fixed_all, heading_all = [], []
    for doc in docs:
        if doc.status != "ok" or doc.unchanged:
            continue
        fixed_all += chunk_fixed(doc.text)
        chunks = chunk_by_headings(doc.text)
        heading_all += chunks
        doc.chunks = chunk_records(doc, chunks, rules, now)
        audit("ingested", doc.source, chunks=len(doc.chunks), doc_hash=doc.doc_hash[:12])

    files = {d.source: {"raw_hash": d.raw_hash, "doc_hash": d.doc_hash, "status": d.status, "note": d.note,
                        "title": d.title, "effective": d.effective, "chunks": len(d.chunks),
                        "pipeline_version": PIPELINE_VERSION} for d in docs}
    version = dataset_version(files)
    (OUT / "manifest.json").write_text(json.dumps({"dataset_version": version, "pipeline_version": PIPELINE_VERSION,
                                                   "created_at": now, "files": files}, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    all_chunks = [c for d in docs if d.status == "ok" for c in d.chunks]
    with (OUT / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    report = {"dataset_version": version, "files": len(docs), "skipped_unchanged": skipped,
              "docs": {d.source: {"status": d.status, "note": d.note, "chunks": len(d.chunks)} for d in docs},
              "chunking": {"fixed": chunk_stats(fixed_all), "by_headings": chunk_stats(heading_all)}}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# Prints the report: a table per file, the chunking comparison and the FOR CHAT line.
# Returns: None; the output goes to stdout.
def print_report(report: dict) -> None:
    print(f"{'file':<40} {'status':<16} chunks  note")
    for source, d in report["docs"].items():
        print(f"{source:<40} {d['status']:<16} {d['chunks']:>6}  {d['note']}")
    fixed, heading = report["chunking"]["fixed"], report["chunking"]["by_headings"]
    if fixed["chunks"]:
        print("\nchunking            chunks  mid-sentence  cut tables  avg tok  max tok")
        for name, s in (("fixed 120/20", fixed), ("by headings", heading)):
            print(f"{name:<18} {s['chunks']:>7} {s['starts_mid_sentence']:>13} {s['cut_tables']:>11} "
                  f"{s['avg_tokens']:>8} {s['max_tokens']:>8}")
    ok = sum(1 for d in report["docs"].values() if d["status"] == "ok")
    n_chunks = sum(d["chunks"] for d in report["docs"].values())
    print(f"\nFOR CHAT: files {report['files']} | indexed {ok} | chunks {n_chunks} | "
          f"skipped unchanged {report['skipped_unchanged']} | dataset {report['dataset_version']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="ignore the manifest and re-ingest every file")
    args = ap.parse_args()
    if not RAW.exists():
        sys.exit(f"no input folder {RAW}")
    print_report(run(args.full))
