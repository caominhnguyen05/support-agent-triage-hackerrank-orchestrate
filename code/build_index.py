"""
build_index.py

Builds a local vector database from scraped text files.

- Reads .txt files from scraper output
- Splits text into overlapping chunks (tiktoken-based)
- Generates embeddings using OpenAI text-embedding-3-small
- Stores vectors in ChromaDB

Output:
  data/chroma_db/ (persistent vector store used by agent.py)

Usage:
  python build_index.py
  python build_index.py --reset         # wipe and rebuild from scratch
  python build_index.py --domain claude # rebuild only one domain's chunks

Requires: OPENAI_API_KEY
"""

import argparse
import os
from pathlib import Path
import chromadb
import tiktoken
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Config
BASE_DIR   = Path(__file__).parent.parent / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Chunk sizing for text-embedding-3-small
CHUNK_SIZE    = 400   # no. of tokens in each chunk
CHUNK_OVERLAP = 60    # token overlap between adjacent chunks

EMBEDDING_MODEL   = "text-embedding-3-small"
TIKTOKEN_ENCODING = "cl100k_base"

DOMAINS = ["hackerrank", "claude", "visa"]

EMBED_BATCH_SIZE = 100



def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using text-embedding-3-small."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


_tokenizer = tiktoken.get_encoding(TIKTOKEN_ENCODING)

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by token count, not word count.
    """
    tokens = _tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + size
        chunk_tokens = tokens[start:end]
        chunk = _tokenizer.decode(chunk_tokens)
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(tokens):
            break
        start += size - overlap
    return chunks


# Article parsing
def parse_article(path: Path) -> dict | None:
    """
    Parse a scraped .txt file back into its metadata + body.
    Returns {source, title, domain, text} or None if the file is malformed.
    """
    raw = path.read_text(encoding="utf-8")
    header, separator, body = raw.partition("---\n")

    if not separator:
        print(f"  [warn] {path.name}: missing '---' separator — skipping")
        return None

    meta = {}
    for line in header.strip().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            meta[k.strip().lower()] = v.strip()

    text = body.strip()
    if not text:
        print(f"  [warn] {path.name}: empty body after separator — skipping")
        return None

    return {
        "source": meta.get("source", str(path)),
        "title":  meta.get("title",  path.stem),
        "domain": meta.get("domain", "unknown"),
        "text":   text,
    }


def _expected_ids_for_domain(domain: str, domain_dir: Path) -> set[str]:
    """Return the full set of chunk IDs that should exist for this domain."""
    expected = set()
    for file_path in domain_dir.glob("*.txt"):
        raw = file_path.read_text(encoding="utf-8")
        _, separator, body = raw.partition("---\n")
        if not separator:
            continue
        text = body.strip()
        if not text:
            continue
        n_chunks = len(chunk_text(text))
        for i in range(n_chunks):
            expected.add(f"{domain}::{file_path.stem}::chunk{i}")
    return expected


def remove_stale_chunks(collection: chromadb.Collection, domain: str, domain_dir: Path) -> None:
    """
    Delete chunks from the collection whose source file no longer exists or
    whose chunk count has changed (e.g. file was renamed or content shrank).
    """
    # Fetch all IDs for this domain
    existing = collection.get(where={"domain": {"$eq": domain}}, include=[])
    if not existing or not existing.get("ids"):
        return

    existing_ids = set(existing["ids"])
    expected_ids = _expected_ids_for_domain(domain, domain_dir)
    stale_ids    = existing_ids - expected_ids

    if stale_ids:
        collection.delete(ids=list(stale_ids))
        print(f"  [cleanup] Removed {len(stale_ids)} stale chunks for domain '{domain}'")


def build_index(domains: list[str], reset: bool = False) -> None:
    """Build (or incrementally update) the vector database."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    collection_name = "support_corpus"

    if reset:
        try:
            chroma_client.delete_collection(collection_name)
            print(f"[reset] Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    # Use cosine distance 
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    openai_client = get_openai_client()
    total_chunks  = 0

    for domain in domains:
        txt_dir    = BASE_DIR / f"{domain}_txt"
        legacy_dir = BASE_DIR / domain

        if txt_dir.exists() and any(txt_dir.glob("*.txt")):
            domain_dir = txt_dir
        elif legacy_dir.exists():
            domain_dir = legacy_dir
        else:
            print(f"[skip] No corpus found for '{domain}' — run scraper.py first")
            continue

        files = list(domain_dir.glob("*.txt"))
        print(f"\n[{domain}] Found {len(files)} article files in {domain_dir}")

        # Remove chunks for files that no longer exist or have changed size
        if not reset:
            remove_stale_chunks(collection, domain, domain_dir)

        for file_path in files:
            article = parse_article(file_path)
            if article is None:
                continue

            chunks = chunk_text(article["text"])
            if not chunks:
                continue

            ids = [
                f"{domain}::{file_path.stem}::chunk{i}"
                for i in range(len(chunks))
            ]
            metadatas = [
                {
                    "domain": article["domain"],
                    "source": article["source"],
                    "title": article["title"][:200],
                    "chunk_i": i,
                }
                for i in range(len(chunks))
            ]

            # Embed and upsert in batches
            for b_start in range(0, len(chunks), EMBED_BATCH_SIZE):
                b_slice = slice(b_start, b_start + EMBED_BATCH_SIZE)
                b_chunks = chunks   [b_slice]
                b_ids = ids      [b_slice]
                b_metadatas = metadatas[b_slice]

                embeddings = embed_texts(openai_client, b_chunks)

                collection.upsert(
                    ids        = b_ids,
                    documents  = b_chunks,
                    embeddings = embeddings,
                    metadatas  = b_metadatas,
                )

            total_chunks += len(chunks)
            print(f"  indexed {file_path.name} → {len(chunks)} chunks")

    print(f"\nDone. Total chunks indexed: {total_chunks}")
    print(f"ChromaDB path: {CHROMA_DIR.resolve()}")
    print(f"Collection total count: {collection.count()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index (OpenAI embeddings)")
    parser.add_argument("--reset",  action="store_true", help="Wipe and rebuild from scratch")
    parser.add_argument(
        "--domain", choices=DOMAINS,
        help="Rebuild only this domain (default: all)",
    )
    args = parser.parse_args()

    domains = [args.domain] if args.domain else DOMAINS
    build_index(domains, reset=args.reset)


if __name__ == "__main__":
    main()